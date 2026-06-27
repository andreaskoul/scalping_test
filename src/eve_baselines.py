"""Eve baselines + post-cost walk-forward evaluator.

This is the bar a transformer must clear **before** it is worth building. Per
``docs/EVE_TRANSFORMER_RESEARCH.md`` (Training Gates), the required baseline set
is no-trade, last-value/no-change, logistic regression, and the current rule
stack, and *"the transformer must beat simple baselines after costs, not only on
raw prediction metrics."*

What this module does, and why:

  - Labels and scores everything through :mod:`src.eve_labels` so the only
    headline metric is **post-cost expectancy** (the FI-2010 critique: raw
    accuracy is not profit; arXiv:1705.03233, LOBCAST benchmark study).
  - Evaluates out-of-sample under an **anchored walk-forward** (train on the
    past, test on the next block, roll forward) — the standard guard against
    overfitting for non-stationary financial series, and what the bitcoin
    walk-forward study (arXiv:2606.00060) and de Prado both insist on.
  - Reuses :mod:`src.stats` (block-bootstrap CI, HAC t-stat, Deflated Sharpe)
    so significance is penalized by the number of baselines tried, never a raw
    mean. A baseline only "wins" if its post-cost expectancy beats no-trade
    *and* clears the significance gate.

Pure-numpy: no sklearn/torch dependency, so it runs in CI without a GPU or
extra installs (matching the rest of the Eve substrate).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import stats as _stats
from .eve_data import EveLake, EveSequenceSample, build_bar_sequences, read_symbol_bars
from .eve_labels import (
    DOWN,
    FLAT,
    UP,
    CostModel,
    cost_aware_label,
    label_distribution,
    post_cost_pnls,
)

_ACTIONS = (DOWN, FLAT, UP)
_T_STAT_MIN = 2.0


# --------------------------------------------------------------------------- #
# Feature helpers
# --------------------------------------------------------------------------- #
def _flatten(sample: EveSequenceSample) -> np.ndarray:
    return np.asarray(sample.features, dtype=float).ravel()


def _last_step_return(sample: EveSequenceSample) -> float:
    """Most recent close-to-close return inside the feature window.

    Feature index 3 is the bar close (see ``eve_data._bar_features``). Returns
    0.0 when the window is too short or a close is non-positive.
    """
    feats = sample.features
    if len(feats) < 2:
        return 0.0
    prev_close = float(feats[-2][3])
    last_close = float(feats[-1][3])
    if prev_close <= 0 or last_close <= 0:
        return 0.0
    return last_close / prev_close - 1.0


def _window_momentum(sample: EveSequenceSample) -> float:
    """Return over the whole feature window (first close -> last close)."""
    feats = sample.features
    if len(feats) < 2:
        return 0.0
    first_close = float(feats[0][3])
    last_close = float(feats[-1][3])
    if first_close <= 0 or last_close <= 0:
        return 0.0
    return last_close / first_close - 1.0


# --------------------------------------------------------------------------- #
# Baselines — each .fit(train, cost) then .predict(test) -> list[int]
# --------------------------------------------------------------------------- #
class NoTradeBaseline:
    """Always no-trade. The hurdle every model must beat: expectancy 0."""

    name = "no_trade"

    def fit(self, train: Sequence[EveSequenceSample], cost: float) -> "NoTradeBaseline":
        return self

    def predict(self, test: Sequence[EveSequenceSample]) -> list[int]:
        return [FLAT] * len(test)


class PersistenceBaseline:
    """Last-value / no-change: bet the last step's move continues, cost-gated."""

    name = "persistence"

    def __init__(self) -> None:
        self._cost = 0.0

    def fit(self, train: Sequence[EveSequenceSample], cost: float) -> "PersistenceBaseline":
        self._cost = float(cost)
        return self

    def predict(self, test: Sequence[EveSequenceSample]) -> list[int]:
        return [cost_aware_label(_last_step_return(s), self._cost) for s in test]


class MomentumRuleBaseline:
    """Transparent momentum rule, the stand-in for Eve's "current rule stack".

    Adam's actual rule stack is Polymarket-L2/Binance specific and does not
    apply to Eve's broad equity/crypto bars, so the contract's "current rule
    stack" baseline is realized here as a whole-window momentum rule gated by
    cost. It is intentionally simple and auditable.
    """

    name = "momentum_rule"

    def __init__(self) -> None:
        self._cost = 0.0

    def fit(self, train: Sequence[EveSequenceSample], cost: float) -> "MomentumRuleBaseline":
        self._cost = float(cost)
        return self

    def predict(self, test: Sequence[EveSequenceSample]) -> list[int]:
        return [cost_aware_label(_window_momentum(s), self._cost) for s in test]


class LogisticBaseline:
    """Multinomial (softmax) logistic regression, pure numpy.

    Fits on cost-aware labels of the training block with standardized flattened
    window features. Deterministic (zero init, fixed iterations). Exposes
    ``predict_proba`` so the evaluator can score calibration (Brier).
    """

    name = "logistic"

    def __init__(self, *, iters: int = 300, lr: float = 0.5, l2: float = 1e-3) -> None:
        self.iters = int(iters)
        self.lr = float(lr)
        self.l2 = float(l2)
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sd

    def fit(self, train: Sequence[EveSequenceSample], cost: float) -> "LogisticBaseline":
        if not train:
            return self
        X = np.vstack([_flatten(s) for s in train])
        y = np.asarray(
            [_ACTIONS.index(cost_aware_label(s.future_return, cost)) for s in train],
            dtype=int,
        )
        self._mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        Xs = self._standardize(X)
        n, d = Xs.shape
        k = len(_ACTIONS)
        self._W = np.zeros((d, k), dtype=float)
        self._b = np.zeros(k, dtype=float)
        Y = np.eye(k)[y]
        for _ in range(self.iters):
            P = _softmax(Xs @ self._W + self._b)
            grad_logits = (P - Y) / n
            self._W -= self.lr * (Xs.T @ grad_logits + self.l2 * self._W)
            self._b -= self.lr * grad_logits.sum(axis=0)
        return self

    def predict_proba(self, test: Sequence[EveSequenceSample]) -> np.ndarray:
        if self._W is None or not test:
            return np.full((len(test), len(_ACTIONS)), 1.0 / len(_ACTIONS))
        X = np.vstack([_flatten(s) for s in test])
        return _softmax(self._standardize(X) @ self._W + self._b)

    def predict(self, test: Sequence[EveSequenceSample]) -> list[int]:
        proba = self.predict_proba(test)
        return [_ACTIONS[i] for i in proba.argmax(axis=1)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / e.sum(axis=1, keepdims=True)


def default_baselines() -> list[Any]:
    """The required baseline set (no-trade, last-value, logistic, rule)."""
    return [NoTradeBaseline(), PersistenceBaseline(), MomentumRuleBaseline(), LogisticBaseline()]


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def _chrono(samples: Sequence[EveSequenceSample]) -> list[EveSequenceSample]:
    return sorted(samples, key=lambda s: (s.symbol, s.end_ts, s.label_ts))


@dataclass(frozen=True)
class WalkForwardResult:
    """Concatenated out-of-sample predictions for one baseline across folds."""

    name: str
    actions: list[int]
    future_returns: list[float]
    probas: list[list[float]] | None  # for calibration; None if non-probabilistic
    n_folds: int


def walk_forward(
    samples: Sequence[EveSequenceSample],
    baselines: Sequence[Any],
    cost: float,
    *,
    n_folds: int = 5,
    min_train: int = 50,
) -> list[WalkForwardResult]:
    """Anchored walk-forward: train on the past, test on the next block, roll.

    Splits the chronologically ordered samples into ``n_folds + 1`` contiguous
    blocks; fold k trains on everything before block k and tests on block k, so
    no test sample is ever before its training data. Per-baseline out-of-sample
    actions are concatenated across folds for one honest OOS series.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    ordered = _chrono(samples)
    n = len(ordered)
    results: dict[str, WalkForwardResult] = {}
    acts: dict[str, list[int]] = {b.name: [] for b in baselines}
    frs: dict[str, list[float]] = {b.name: [] for b in baselines}
    probs: dict[str, list[list[float]] | None] = {b.name: [] for b in baselines}
    folds_used = 0

    if n >= (min_train + n_folds):
        # Block boundaries over [first_test_start, n]; train is everything prior.
        edges = np.linspace(min_train, n, n_folds + 1, dtype=int)
        edges = sorted(set(int(e) for e in edges))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if hi <= lo:
                continue
            train = ordered[:lo]
            test = ordered[lo:hi]
            if len(train) < min_train or not test:
                continue
            folds_used += 1
            for b in baselines:
                b.fit(train, cost)
                preds = b.predict(test)
                acts[b.name].extend(preds)
                frs[b.name].extend(float(s.future_return) for s in test)
                proba_fn = getattr(b, "predict_proba", None)
                if proba_fn is not None and probs[b.name] is not None:
                    proba = proba_fn(test)
                    if hasattr(proba, "tolist"):  # numpy array or torch tensor
                        proba = proba.tolist()
                    probs[b.name].extend(proba)
                else:
                    probs[b.name] = None

    for b in baselines:
        results[b.name] = WalkForwardResult(
            name=b.name,
            actions=acts[b.name],
            future_returns=frs[b.name],
            probas=probs[b.name] if probs[b.name] else None,
            n_folds=folds_used,
        )
    return [results[b.name] for b in baselines]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineMetrics:
    name: str
    n: int
    coverage: float          # fraction of non-flat actions (turnover proxy)
    post_cost_expectancy: float
    post_cost_total: float
    t_stat: float
    ci_lo: float
    ci_hi: float
    sharpe: float
    deflated_sharpe: float   # P(true edge > 0) penalized for #baselines
    hit_rate: float          # share of taken trades with positive post-cost PnL
    accuracy: float          # vs cost-aware oracle label (diagnostic only)
    brier: float             # multiclass Brier vs oracle one-hot (nan if n/a)

    def to_row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "coverage": round(self.coverage, 4),
            "post_cost_expectancy": self.post_cost_expectancy,
            "post_cost_total": self.post_cost_total,
            "t_stat": round(self.t_stat, 3),
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "sharpe": round(self.sharpe, 4),
            "deflated_sharpe": round(self.deflated_sharpe, 4),
            "hit_rate": round(self.hit_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "brier": (None if np.isnan(self.brier) else round(self.brier, 4)),
        }


def evaluate_result(
    result: WalkForwardResult,
    cost: float,
    *,
    n_trials: int,
    seed: int = 0,
) -> BaselineMetrics:
    """Post-cost metrics + significance for one baseline's OOS series."""
    actions = result.actions
    frs = result.future_returns
    n = len(actions)
    if n == 0:
        return BaselineMetrics(
            result.name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float("nan")
        )
    pnl = np.asarray(post_cost_pnls(actions, frs, cost), dtype=float)
    taken = np.asarray([a != FLAT for a in actions], dtype=bool)
    coverage = float(taken.mean())
    expectancy = float(pnl.mean())
    total = float(pnl.sum())

    ci = _stats.block_bootstrap_ci(pnl, seed=seed)
    tstat = _stats.t_stat(pnl)
    sharpe = _stats.per_obs_sharpe(pnl)
    dsr = _stats.deflated_sharpe_ratio(
        sharpe, n_trials=max(1, n_trials), n_obs=n, sr_std=_sr_std_floor(sharpe)
    )

    taken_pnl = pnl[taken]
    hit_rate = float((taken_pnl > 0).mean()) if taken_pnl.size else 0.0

    oracle = np.asarray([cost_aware_label(r, cost) for r in frs], dtype=int)
    accuracy = float((np.asarray(actions, dtype=int) == oracle).mean())

    brier = float("nan")
    if result.probas is not None:
        P = np.asarray(result.probas, dtype=float)
        if P.shape == (n, len(_ACTIONS)):
            onehot = np.eye(len(_ACTIONS))[[_ACTIONS.index(int(o)) for o in oracle]]
            brier = float(((P - onehot) ** 2).sum(axis=1).mean())

    return BaselineMetrics(
        name=result.name,
        n=n,
        coverage=coverage,
        post_cost_expectancy=expectancy,
        post_cost_total=total,
        t_stat=tstat,
        ci_lo=ci.lo,
        ci_hi=ci.hi,
        sharpe=sharpe,
        deflated_sharpe=dsr,
        hit_rate=hit_rate,
        accuracy=accuracy,
        brier=brier,
    )


def _sr_std_floor(sharpe: float) -> float:
    """Cross-trial Sharpe dispersion for the deflated-Sharpe penalty.

    With only a handful of baselines we cannot estimate the true dispersion, so
    we use a conservative floor of max(|SR|, 0.1): never let the selection
    hurdle collapse to zero, which would make Deflated Sharpe trivially pass.
    """
    return max(abs(sharpe), 0.1)


@dataclass(frozen=True)
class EveBaselineReport:
    cost: float
    n_folds: int
    metrics: list[BaselineMetrics]
    best_name: str
    best_beats_notrade: bool
    label_dist: dict[str, int]
    gate_reasons: list[str] = field(default_factory=list)

    def machine_line(self) -> str:
        beats = "true" if self.best_beats_notrade else "false"
        best = next((m for m in self.metrics if m.name == self.best_name), None)
        edge = best.post_cost_expectancy if best else 0.0
        t = best.t_stat if best else 0.0
        return (
            f"eve_baseline: best={self.best_name} "
            f"post_cost_edge={edge:+.6f} t={t:.2f} "
            f"beats_notrade={beats}"
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "cost": self.cost,
                "n_folds": self.n_folds,
                "label_dist": self.label_dist,
                "best_name": self.best_name,
                "best_beats_notrade": self.best_beats_notrade,
                "gate_reasons": self.gate_reasons,
                "metrics": [m.to_row() for m in self.metrics],
            },
            indent=2,
            sort_keys=True,
        )


def build_report(
    samples: Sequence[EveSequenceSample],
    *,
    cost_model: CostModel | None = None,
    n_folds: int = 5,
    min_train: int = 50,
    seed: int = 0,
    extra_models: Sequence[Any] | None = None,
) -> EveBaselineReport:
    """Run all baselines walk-forward and score them on post-cost expectancy.

    The "winner" is the highest post-cost expectancy. It only counts as
    beating no-trade if its expectancy is positive, its 95% CI excludes 0, and
    its HAC t-stat clears ``_T_STAT_MIN`` — the same discipline as Adam's
    verdict. ``extra_models`` (e.g. the Eve transformer) are evaluated on the
    *same* out-of-sample series as the baselines, so the comparison is fair.
    """
    cm = cost_model or CostModel()
    cost = cm.round_trip()
    baselines = list(default_baselines())
    if extra_models:
        baselines.extend(extra_models)
    results = walk_forward(samples, baselines, cost, n_folds=n_folds, min_train=min_train)
    n_trials = len(baselines)
    metrics = [evaluate_result(r, cost, n_trials=n_trials, seed=seed) for r in results]

    # Best directional baseline = highest post-cost expectancy among those that
    # actually trade (exclude the always-flat hurdle from "winning").
    directional = [m for m in metrics if m.coverage > 0 and m.n > 0]
    if directional:
        best = max(directional, key=lambda m: m.post_cost_expectancy)
    else:
        best = max(metrics, key=lambda m: m.post_cost_expectancy)

    reasons: list[str] = []
    if best.post_cost_expectancy <= 0:
        reasons.append("post_cost_expectancy<=0")
    if not (best.ci_lo > 0):
        reasons.append("ci95_includes_0")
    if not (best.t_stat >= _T_STAT_MIN):
        reasons.append(f"t_stat<{_T_STAT_MIN:g}")
    beats = not reasons

    oracle_actions = [cost_aware_label(s.future_return, cost) for s in samples]
    return EveBaselineReport(
        cost=cost,
        n_folds=results[0].n_folds if results else 0,
        metrics=metrics,
        best_name=best.name,
        best_beats_notrade=beats,
        label_dist=label_distribution(oracle_actions),
        gate_reasons=reasons,
    )


# --------------------------------------------------------------------------- #
# Dataset loading + CLI
# --------------------------------------------------------------------------- #
def load_samples_from_dataset(path: str | Path) -> list[EveSequenceSample]:
    """Load all train/val/test sequence samples from a lake dataset dir."""
    base = Path(path)
    out: list[EveSequenceSample] = []
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        f = base / name
        if not f.exists():
            continue
        with f.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out.append(EveSequenceSample(**row))
    return out


def build_report_from_lake(
    lake: EveLake,
    *,
    symbol: str,
    asset_class: str,
    timeframe: str = "1Min",
    provider: str = "alpaca",
    window: int = 16,
    horizon: int = 1,
    cost_model: CostModel | None = None,
    n_folds: int = 5,
    min_train: int = 200,
    seed: int = 0,
    extra_models: Sequence[Any] | None = None,
) -> "tuple[EveBaselineReport, int]":
    """Load one symbol's real bars from the lake and run the baseline verdict.

    Returns the report and the number of bars read. This is the bridge from
    `eve_ingest` (real Alpaca bars on disk) to the post-cost baseline bar, so
    the baselines (and any ``extra_models`` like the transformer) run on real
    data instead of synthetic AR(1) series.
    """
    from .eve_ingest import bars_dataset

    bars = read_symbol_bars(
        lake, provider=provider, asset_class=asset_class, symbol=symbol,
        dataset=bars_dataset(timeframe),
    )
    samples = build_bar_sequences(bars, window=window, horizon=horizon, threshold=0.0)
    report = build_report(
        samples, cost_model=cost_model, n_folds=n_folds, min_train=min_train, seed=seed,
        extra_models=extra_models,
    )
    return report, len(bars)


def _print_human(report: EveBaselineReport) -> None:
    print("\n=== Eve Baseline Verdict (post-cost walk-forward) ===")
    print(f"Round-trip cost:  {report.cost * 1e4:.2f} bps  | folds: {report.n_folds}")
    print(f"Label dist:       {report.label_dist}")
    print(
        f"{'baseline':<14}{'n':>7}{'cover':>8}{'edge/ev':>12}"
        f"{'t':>7}{'CI95':>20}{'DSR':>7}{'acc':>7}"
    )
    for m in report.metrics:
        ci = f"[{m.ci_lo:+.5f},{m.ci_hi:+.5f}]"
        print(
            f"{m.name:<14}{m.n:>7}{m.coverage:>8.2f}{m.post_cost_expectancy:>+12.6f}"
            f"{m.t_stat:>7.2f}{ci:>20}{m.deflated_sharpe:>7.2f}{m.accuracy:>7.2f}"
        )
    print()
    if report.gate_reasons:
        print(f"best '{report.best_name}' blocked by: {', '.join(report.gate_reasons)}")
    print(report.machine_line())


def cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Eve baseline post-cost walk-forward verdict")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="lake dataset dir with train/val/test.jsonl")
    src.add_argument("--lake-symbol", help="run on real lake bars for this symbol, e.g. 'BTC/USD'")
    ap.add_argument("--asset-class", choices=["equity", "crypto"], help="required with --lake-symbol")
    ap.add_argument("--lake", default=None, help="lake root (default $EVE_LAKE_DIR or data/lake)")
    ap.add_argument("--timeframe", default="1Min", help="bar timeframe for --lake-symbol")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--half-spread-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    cm = CostModel(
        taker_fee_bps=args.fee_bps,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
    )
    if args.lake_symbol:
        if not args.asset_class:
            raise SystemExit("--asset-class is required with --lake-symbol")
        lake = EveLake(Path(args.lake)) if args.lake else EveLake.from_env()
        report, n_bars = build_report_from_lake(
            lake, symbol=args.lake_symbol, asset_class=args.asset_class,
            timeframe=args.timeframe, window=args.window, horizon=args.horizon, cost_model=cm,
            n_folds=args.folds, min_train=args.min_train, seed=args.seed,
        )
        print(f"# {args.lake_symbol} [{args.asset_class}] {args.timeframe} {n_bars} bars, window={args.window} horizon={args.horizon}")
    else:
        samples = load_samples_from_dataset(args.dataset)
        report = build_report(
            samples, cost_model=cm, n_folds=args.folds, min_train=args.min_train, seed=args.seed
        )
    if args.json:
        print(report.to_json())
    else:
        _print_human(report)


if __name__ == "__main__":
    cli()
