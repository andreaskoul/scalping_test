"""Eve ensemble: transformer cross-sectional prediction + direct-RL allocation.

This is the "optimal ensemble" the survey points to: the **transformer** is the
prediction lane (per-asset cost-aware directional score), the **direct-RL
allocator** (Differential Sharpe Ratio) is the portfolio lane, and they are
chained — the allocator sizes the transformer's cross-sectional signal under
risk and transaction costs.

Both lanes are walk-forward, nested and leakage-safe: in each fold the
transformer is trained on the *past* and predicts the *next* block out-of-sample;
those OOS scores form the test panel the allocator is evaluated on (the allocator
is fit on the train-block signals). OOS net returns are concatenated across folds
and scored on annualized + **Deflated** Sharpe (`src/stats.py`), head-to-head
against equal-weight beta and a transformer-signal long-short book without RL — so
we can see whether the RL lane adds anything over the prediction lane alone.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np

from .eve_data import EveSequenceSample, build_bar_sequences, read_symbol_bars
from .eve_ingest import bars_dataset
from .eve_labels import CostModel
from .eve_portfolio import (
    Panel,
    PortfolioReport,
    _zscore_cross_section,
    cash_weights,
    equal_weights,
    long_short_weights,
    portfolio_metrics,
    simulate,
    _turnover_series,
)

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")


def _symbol_samples(lake: Any, sym: str, *, asset_class: str, provider: str,
                    timeframe: str, window: int, horizon: int) -> list[EveSequenceSample]:
    bars = read_symbol_bars(
        lake, provider=provider, asset_class=asset_class, symbol=sym,
        dataset=bars_dataset(timeframe),
    )
    return build_bar_sequences(bars, window=window, horizon=horizon, threshold=0.0)


def _signal_panel(dates: list[str], symbols: list[str],
                  per_sym: dict[str, dict[str, EveSequenceSample]], model: Any) -> Panel:
    """Cross-sectional panel whose single feature is the transformer's score."""
    T, N = len(dates), len(symbols)
    signals = np.zeros((T, N, 1))
    fwd = np.zeros((T, N))
    for j, sym in enumerate(symbols):
        samples = [per_sym[sym][d] for d in dates]
        proba = model.predict_proba(samples)  # list of [P_down, P_flat, P_up]
        for i, p in enumerate(proba):
            signals[i, j, 0] = p[2] - p[0]            # up minus down
            fwd[i, j] = samples[i].future_return
    signals[:, :, 0] = _zscore_cross_section(signals[:, :, 0])
    return Panel(dates, symbols, signals, fwd)


def ensemble_walk_forward(
    lake: Any,
    symbols: Sequence[str],
    *,
    asset_class: str = "equity",
    provider: str = "yahoo",
    timeframe: str = "1d",
    window: int = 16,
    horizon: int = 1,
    cost: float = 0.0005,
    n_folds: int = 4,
    min_train_days: int = 504,
    transformer_kwargs: dict | None = None,
    allocator_kwargs: dict | None = None,
    seed: int = 0,
) -> PortfolioReport:
    """Nested walk-forward verdict for transformer→RL vs transformer-LS vs beta."""
    from .eve_transformer import TransformerModel
    from .eve_rl import DirectRLAllocator

    tk = {"epochs": 10, "device": None, "seed": seed, **(transformer_kwargs or {})}
    ak = {"epochs": 40, "cost": cost, "seed": seed, **(allocator_kwargs or {})}
    label_cost = CostModel().round_trip()

    per_sym: dict[str, dict[str, EveSequenceSample]] = {}
    for sym in symbols:
        samples = _symbol_samples(lake, sym, asset_class=asset_class, provider=provider,
                                  timeframe=timeframe, window=window, horizon=horizon)
        per_sym[sym] = {s.end_ts[:10]: s for s in samples}
    syms = [s for s in symbols if per_sym.get(s)]
    common = sorted(set.intersection(*[set(per_sym[s]) for s in syms])) if syms else []
    if len(common) < min_train_days + n_folds:
        raise ValueError(f"not enough overlapping dates ({len(common)}) for the ensemble")

    nets: dict[str, list[float]] = {k: [] for k in ("ensemble", "transformer_ls", "equal_weight", "cash")}
    turns: dict[str, list[float]] = {k: [] for k in nets}
    folds_used = 0

    edges = sorted(set(int(e) for e in np.linspace(min_train_days, len(common), n_folds + 1)))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        folds_used += 1
        train_dates, test_dates = common[:lo], common[lo:hi]

        pooled_train = [per_sym[s][d] for s in syms for d in train_dates]
        model = TransformerModel(**tk).fit(pooled_train, label_cost)

        train_panel = _signal_panel(train_dates, syms, per_sym, model)
        test_panel = _signal_panel(test_dates, syms, per_sym, model)

        alloc = DirectRLAllocator(**ak).fit(train_panel)
        books = {
            "ensemble": alloc.predict_weights(test_panel),
            "transformer_ls": long_short_weights(test_panel, signal_feature=0, top_frac=0.3),
            "equal_weight": equal_weights(test_panel),
            "cash": cash_weights(test_panel),
        }
        for name, W in books.items():
            nets[name].extend(simulate(W, test_panel.fwd_returns, cost).tolist())
            turns[name].extend(_turnover_series(W, test_panel.fwd_returns).tolist())

    n_trials = len(nets)
    metrics = [portfolio_metrics(name, np.asarray(nets[name]), np.asarray(turns[name]),
                                 n_trials=n_trials, seed=seed) for name in nets]
    learned = [m for m in metrics if m.name in ("ensemble", "transformer_ls") and m.n > 0]
    best = max(learned, key=lambda m: m.sharpe) if learned else max(metrics, key=lambda m: m.sharpe)
    base_sr = max((m.sharpe for m in metrics if m.name in ("cash", "equal_weight")), default=0.0)

    reasons: list[str] = []
    if best.sharpe <= 0:
        reasons.append("sharpe<=0")
    if best.sharpe <= base_sr:
        reasons.append("does_not_beat_beta")
    if best.deflated_sharpe < 0.95:
        reasons.append("deflated_sharpe<0.95")

    return PortfolioReport(cost=cost, n_folds=folds_used, metrics=metrics,
                           best_name=best.name, best_beats_baselines=not reasons,
                           gate_reasons=reasons)
