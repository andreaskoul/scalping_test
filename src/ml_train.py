"""
Train the directional model (Rank 5) from a historical Binance trade tape.

Replaces the hand-picked logistic weights with ones fit to data: build features
(OFI + fast/slow returns, the transform the live DirectionalModel uses) from
aggTrades, label each sample by the realised next-move over `horizon`, fit an
L2-regularised logistic, and report a **walk-forward** AUC so we don't trust an
in-sample number. Persisted to model_weights.json, which the bot hot-loads.

  python -m src.ml_train --symbol btcusdt --hours 6     # fetch tape + train
  python -m src.ml_train --symbol btcusdt --hours 6 --min-auc 0.55

Honesty: this is a transparent 3-feature logistic, not the paper's 17-feature
RF (no historical L2 here). arXiv:2511.15960 warns raw binary-direction ML is
hard — so we gate writing on walk-forward AUC clearing --min-auc, and the live
overlay keeps ML_WEIGHT small and converts physical→risk-neutral downstream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time

import numpy as np
from scipy.stats import rankdata  # type: ignore

from .microstructure import MicroReplay, MicroFeatures, transform_features, FEATURE_KEYS

log = logging.getLogger(__name__)


def build_dataset(
    replay: MicroReplay, sigma: float = 0.5, horizon: float = 30.0,
    step: float = 10.0, warmup: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y): logistic design rows + next-move labels from a trade tape."""
    if not replay.ts:
        return np.empty((0, len(FEATURE_KEYS))), np.empty((0,))
    t0 = replay.ts[0] + warmup
    t1 = replay.ts[-1] - horizon
    X, y = [], []
    t = t0
    while t <= t1:
        p_now = replay.price_at(t)
        p_fut = replay.price_at(t + horizon)
        if p_now > 0 and p_fut > 0 and p_fut != p_now:
            f = replay.features(t, spot=p_now, sigma=sigma, tte=horizon)
            X.append(transform_features(f))
            y.append(1.0 if p_fut > p_now else 0.0)
        t += step
    return np.array(X), np.array(y)


def train_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                   lr: float = 0.5, epochs: int = 3000) -> np.ndarray:
    """Batch gradient-descent logistic regression with L2 (bias unregularised)."""
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) / n
        reg = l2 * w
        reg[0] = 0.0
        w -= lr * (grad + reg)
    return w


def auc(y: np.ndarray, scores: np.ndarray) -> float:
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = rankdata(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def walk_forward(X: np.ndarray, y: np.ndarray, folds: int = 4) -> float:
    """Mean out-of-sample AUC, training only on the past of each fold."""
    n = len(y)
    if n < folds * 10:
        return 0.5
    fold = n // folds
    aucs = []
    for k in range(1, folds):
        tr = slice(0, k * fold)
        te = slice(k * fold, (k + 1) * fold)
        ytr = y[tr]
        if ytr.sum() in (0, len(ytr)) or len(y[te]) < 10:
            continue
        w = train_logistic(X[tr], ytr)
        aucs.append(auc(y[te], X[te] @ w))
    return float(np.mean(aucs)) if aucs else 0.5


def weights_dict(w: np.ndarray) -> dict:
    return {k: float(v) for k, v in zip(FEATURE_KEYS, w)}


def train_from_tape(
    trades: list[tuple[float, float, float, float]],
    sigma: float = 0.5, horizon: float = 30.0, step: float = 10.0, folds: int = 4,
) -> dict:
    """End-to-end: tape → dataset → walk-forward AUC → final fit on all data."""
    replay = MicroReplay(trades)
    X, y = build_dataset(replay, sigma=sigma, horizon=horizon, step=step)
    if len(y) < 50:
        return {"n": int(len(y)), "auc": 0.5, "weights": None}
    wf_auc = walk_forward(X, y, folds=folds)
    w = train_logistic(X, y)
    return {
        "n": int(len(y)), "auc": wf_auc, "weights": weights_dict(w),
        "horizon_secs": horizon, "version": time.time(),
    }


async def _fetch_and_train(symbol: str, hours: float, min_auc: float) -> None:
    import aiohttp
    from .backtest import _fetch_agg_trades

    now = time.time()
    start = now - hours * 3600.0
    async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as s:
        log.info("Fetching %s aggTrades for last %.1fh...", symbol, hours)
        trades = await _fetch_agg_trades(s, symbol, start, now)
    log.info("Fetched %d trades", len(trades))
    if not trades:
        print("No trades fetched (network/geo-block?) — cannot train.")
        return

    res = train_from_tape(trades)
    print(f"samples={res['n']}  walk-forward AUC={res['auc']:.4f}")
    if res["weights"] is None:
        print("Too few samples to train.")
        return
    print(f"weights={res['weights']}")
    if res["auc"] < min_auc:
        print(f"AUC {res['auc']:.4f} < min-auc {min_auc} — NOT writing model_weights.json "
              "(directional edge not established; keeping the cold-start defaults).")
        return
    tmp = "model_weights.json.tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    import os
    os.replace(tmp, "model_weights.json")
    print("Wrote model_weights.json — the bot will hot-load it on next start.")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Train the directional model from a Binance trade tape")
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--min-auc", type=float, default=0.53,
                    help="only write weights if walk-forward AUC clears this")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    asyncio.run(_fetch_and_train(args.symbol, args.hours, args.min_auc))


if __name__ == "__main__":
    cli()
