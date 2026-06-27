"""Eve portfolio lane — direct-RL allocation on a cross-sectional signal panel.

This is the second half of the ensemble the survey points to
(`docs/EVE_SHARPE_LITERATURE.md`): a transformer/GBDT predicts a *cross-sectional*
signal per asset (breadth — the regime where validated edge actually exists), and
an **RL allocator** turns those signals into portfolio weights, optimizing a
**transaction-cost-aware, risk-adjusted** objective.

The RL method is **direct / recurrent reinforcement learning with the
Differential Sharpe Ratio** (Moody & Saffell, NIPS 1998): the DSR is the marginal
contribution of each period's return to the overall Sharpe, so summing per-step
DSR ≈ maximizing the Sharpe online, and the whole rollout is differentiable, so we
train the policy by gradient ascent on cumulative DSR net of costs — recurrent (the
previous weights feed back in, so the policy learns to control turnover). This is
the stable variant of "RL for portfolio optimization": no value-function / Q-learning
instability.

Everything is judged the repo way: realized **post-cost** portfolio returns →
Sharpe and **Deflated Sharpe** (`src/stats.py`), penalized for the number of
strategies tried, and compared head-to-head against cash, equal-weight, and a
plain signal long–short book. The numpy core here needs no torch; only the
:class:`DirectRLAllocator` does (imported lazily).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import stats as _stats

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Differential Sharpe Ratio (Moody & Saffell)
# --------------------------------------------------------------------------- #
def differential_sharpe_series(
    returns: Sequence[float], eta: float = 0.02, warmup: int | None = None
) -> np.ndarray:
    """Per-step Differential Sharpe Ratio for a return series (numpy reference).

    A_t, B_t are EMAs of returns and squared returns; the DSR at t is the
    derivative of the Sharpe ratio w.r.t. the new observation. Summing the series
    approximates the realized Sharpe. The EMA variance is unreliable until ~1/eta
    samples have accrued, so DSR is emitted only after a ``warmup`` (defaults to
    1/eta) — otherwise a handful of early near-zero-variance steps dominate. The
    torch allocator uses the same recursion and warmup.
    """
    r = np.asarray(list(returns), dtype=float)
    out = np.zeros(r.size, dtype=float)
    if warmup is None:
        warmup = max(2, int(1.0 / eta)) if eta > 0 else 2
    A = B = 0.0
    for t in range(r.size):
        if t == 0:
            A, B = r[t], r[t] ** 2
            continue
        dA = r[t] - A
        dB = r[t] ** 2 - B
        var = B - A * A
        if t >= warmup and var > 1e-10:
            out[t] = (B * dA - 0.5 * A * dB) / (var ** 1.5)
        A += eta * dA
        B += eta * dB
    return out


# --------------------------------------------------------------------------- #
# Cross-sectional panel
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Panel:
    """Leakage-safe cross-sectional panel aligned on a common date axis.

    ``signals[t, i, :]`` uses only information available at date ``dates[t]``;
    ``fwd_returns[t, i]`` is the close-to-close return realized *after* t (from
    dates[t] to dates[t+1]). So a weight chosen from ``signals[t]`` earns
    ``fwd_returns[t]`` — no look-ahead.
    """

    dates: list[str]
    symbols: list[str]
    signals: np.ndarray      # (T, N, F)
    fwd_returns: np.ndarray  # (T, N)

    @property
    def n_features(self) -> int:
        return self.signals.shape[2]


def _zscore_cross_section(x: np.ndarray) -> np.ndarray:
    """Z-score each row (cross-section at one date), robust to zero variance."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return np.where(sd > 0, (x - mu) / sd, 0.0)


def build_panel(
    closes_by_symbol: dict[str, list[tuple[str, float]]],
    *,
    momentum_lookbacks: Sequence[int] = (5, 20, 60),
    vol_lookback: int = 20,
) -> Panel:
    """Build a panel from per-symbol (date, close) series.

    Features per asset (all cross-sectionally z-scored, leakage-safe): trailing
    returns over each momentum lookback plus trailing realized vol. Only dates
    present for *every* symbol are kept, so the matrices are dense.
    """
    symbols = sorted(closes_by_symbol)
    if not symbols:
        raise ValueError("no symbols")
    # Intersect the date axis across all symbols.
    common: set[str] | None = None
    maps: dict[str, dict[str, float]] = {}
    for sym in symbols:
        m = {d: c for d, c in closes_by_symbol[sym] if c > 0}
        maps[sym] = m
        common = set(m) if common is None else (common & set(m))
    dates = sorted(common or [])
    warm = max(list(momentum_lookbacks) + [vol_lookback])
    if len(dates) < warm + 2:
        raise ValueError("not enough overlapping dates to build a panel")

    closes = np.array([[maps[s][d] for s in symbols] for d in dates], dtype=float)  # (D, N)
    rets = np.zeros_like(closes)
    rets[1:] = closes[1:] / closes[:-1] - 1.0

    T_full = len(dates)
    feats_per_t: list[np.ndarray] = []
    fwd_per_t: list[np.ndarray] = []
    kept_dates: list[str] = []
    for t in range(warm, T_full - 1):
        cols = []
        for k in momentum_lookbacks:
            cols.append(closes[t] / closes[t - k] - 1.0)
        vol = rets[t - vol_lookback + 1:t + 1].std(axis=0)
        cols.append(vol)
        feat = np.stack(cols, axis=1)  # (N, F)
        feat = _zscore_cross_section(feat.T).T  # z-score each feature cross-sectionally
        feats_per_t.append(feat)
        fwd_per_t.append(closes[t + 1] / closes[t] - 1.0)
        kept_dates.append(dates[t])

    return Panel(
        dates=kept_dates,
        symbols=symbols,
        signals=np.stack(feats_per_t, axis=0),       # (T, N, F)
        fwd_returns=np.stack(fwd_per_t, axis=0),      # (T, N)
    )


def build_panel_from_lake(
    lake: Any,
    symbols: Sequence[str],
    *,
    asset_class: str = "equity",
    provider: str = "yahoo",
    timeframe: str = "1d",
    **panel_kwargs: Any,
) -> Panel:
    from .eve_data import read_symbol_bars
    from .eve_ingest import bars_dataset

    closes: dict[str, list[tuple[str, float]]] = {}
    for sym in symbols:
        bars = read_symbol_bars(
            lake, provider=provider, asset_class=asset_class, symbol=sym,
            dataset=bars_dataset(timeframe),
        )
        if bars:
            closes[sym] = [(b.ts[:10], float(b.close)) for b in bars]
    return build_panel(closes, **panel_kwargs)


# --------------------------------------------------------------------------- #
# Portfolio simulation + baselines
# --------------------------------------------------------------------------- #
def simulate(weights: np.ndarray, fwd_returns: np.ndarray, cost: float) -> np.ndarray:
    """Net per-step portfolio returns for a weight path, charging turnover cost.

    ``weights[t]`` is set using info up to date t and earns ``fwd_returns[t]``.
    Cost = ``cost`` * L1 turnover vs the previous weights drifted by their own
    realized returns (so holding still is free; only trading pays).
    """
    T, N = fwd_returns.shape
    w_prev_eff = np.zeros(N)
    nets = np.zeros(T)
    for t in range(T):
        w = weights[t]
        turnover = np.abs(w - w_prev_eff).sum()
        gross = float(w @ fwd_returns[t])
        nets[t] = gross - cost * turnover
        denom = 1.0 + gross
        w_prev_eff = (w * (1.0 + fwd_returns[t])) / denom if denom > 1e-9 else w
    return nets


def cash_weights(panel: Panel) -> np.ndarray:
    return np.zeros_like(panel.fwd_returns)


def equal_weights(panel: Panel) -> np.ndarray:
    T, N = panel.fwd_returns.shape
    return np.full((T, N), 1.0 / N)


def topk_long_short_weights(scores: np.ndarray, k: int, *, leverage: float = 1.0) -> np.ndarray:
    """Dollar-neutral book: long the top-k by score, short the bottom-k.

    qlib's TopkDropout family. ``scores`` is (T, N); gross leverage is normalized
    to ``leverage`` and the book is dollar-neutral.
    """
    T, N = scores.shape
    k = max(1, min(k, N // 2))
    W = np.zeros((T, N))
    for t in range(T):
        order = np.argsort(scores[t])
        W[t, order[-k:]] = 1.0
        W[t, order[:k]] = -1.0
        gross = np.abs(W[t]).sum()
        if gross > 0:
            W[t] *= leverage / gross
    return W


def long_short_weights(panel: Panel, *, signal_feature: int = 1, top_frac: float = 0.3,
                       leverage: float = 1.0) -> np.ndarray:
    """Dollar-neutral long–short book from one cross-sectional signal feature.

    Longs the top ``top_frac`` of assets by signal, shorts the bottom, sized so
    gross leverage is ``leverage`` and the book is dollar-neutral.
    """
    T, N = panel.fwd_returns.shape
    k = max(1, int(N * top_frac))
    W = np.zeros((T, N))
    for t in range(T):
        s = panel.signals[t, :, signal_feature]
        order = np.argsort(s)
        shorts, longs = order[:k], order[-k:]
        W[t, longs] = 1.0
        W[t, shorts] = -1.0
        gross = np.abs(W[t]).sum()
        if gross > 0:
            W[t] *= leverage / gross
    return W


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioMetrics:
    name: str
    n: int
    ann_return: float
    ann_vol: float
    sharpe: float            # annualized
    deflated_sharpe: float   # P(true SR>0), penalized for #strategies
    t_stat: float
    ci_lo: float
    ci_hi: float
    avg_turnover: float

    def to_row(self) -> dict[str, Any]:
        return {
            "name": self.name, "n": self.n,
            "ann_return": round(self.ann_return, 4), "ann_vol": round(self.ann_vol, 4),
            "sharpe": round(self.sharpe, 3), "deflated_sharpe": round(self.deflated_sharpe, 3),
            "t_stat": round(self.t_stat, 3),
            "ci_lo": round(self.ci_lo, 6), "ci_hi": round(self.ci_hi, 6),
            "avg_turnover": round(self.avg_turnover, 3),
        }


def portfolio_metrics(
    name: str, net_returns: np.ndarray, turnover: np.ndarray | None = None,
    *, n_trials: int = 1, periods_per_year: float = TRADING_DAYS, seed: int = 0,
) -> PortfolioMetrics:
    r = np.asarray(net_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return PortfolioMetrics(name, n, 0, 0, 0, 0, 0, 0, 0, 0)
    daily_sr = _stats.per_obs_sharpe(r)
    ann_sr = daily_sr * math.sqrt(periods_per_year)
    ci = _stats.block_bootstrap_ci(r, seed=seed)
    dsr = _stats.deflated_sharpe_ratio(
        daily_sr, n_trials=max(1, n_trials), n_obs=n, sr_std=max(abs(daily_sr), 0.05)
    )
    return PortfolioMetrics(
        name=name, n=n,
        ann_return=float(r.mean() * periods_per_year),
        ann_vol=float(r.std(ddof=1) * math.sqrt(periods_per_year)),
        sharpe=ann_sr, deflated_sharpe=dsr,
        t_stat=_stats.t_stat(r), ci_lo=ci.lo, ci_hi=ci.hi,
        avg_turnover=float(np.mean(turnover)) if turnover is not None and len(turnover) else 0.0,
    )


def _turnover_series(weights: np.ndarray, fwd_returns: np.ndarray) -> np.ndarray:
    T, N = fwd_returns.shape
    w_prev_eff = np.zeros(N)
    out = np.zeros(T)
    for t in range(T):
        w = weights[t]
        out[t] = np.abs(w - w_prev_eff).sum()
        gross = float(w @ fwd_returns[t])
        denom = 1.0 + gross
        w_prev_eff = (w * (1.0 + fwd_returns[t])) / denom if denom > 1e-9 else w
    return out


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioReport:
    cost: float
    n_folds: int
    metrics: list[PortfolioMetrics]
    best_name: str
    best_beats_baselines: bool
    gate_reasons: list[str] = field(default_factory=list)
    net_returns: dict[str, list[float]] = field(default_factory=dict)
    pbo: float = float("nan")  # probability of backtest overfitting (CSCV)

    def machine_line(self) -> str:
        best = next((m for m in self.metrics if m.name == self.best_name), None)
        sr = best.sharpe if best else 0.0
        dsr = best.deflated_sharpe if best else 0.0
        return (
            f"eve_portfolio: best={self.best_name} ann_sharpe={sr:+.2f} "
            f"deflated_sharpe={dsr:.2f} beats_baselines={'true' if self.best_beats_baselines else 'false'}"
        )


# A strategy is either a callable(panel)->weights (baselines) or a fittable
# allocator exposing fit(train_panel) and predict_weights(test_panel).
def _slice_panel(panel: Panel, lo: int, hi: int) -> Panel:
    return Panel(panel.dates[lo:hi], panel.symbols, panel.signals[lo:hi], panel.fwd_returns[lo:hi])


def walk_forward_portfolio(
    panel: Panel,
    strategies: dict[str, Any],
    *,
    cost: float = 0.0005,
    n_folds: int = 4,
    min_train: int = 252,
    embargo: int = 0,
    seed: int = 0,
) -> PortfolioReport:
    """Anchored walk-forward: train past, evaluate next block, concatenate OOS.

    ``strategies`` maps name -> callable(panel)->weights (stateless baselines) or
    an allocator with ``fit(train)`` + ``predict_weights(test)``. Each strategy's
    OOS net returns are concatenated across folds, then scored on annualized and
    deflated Sharpe with the selection penalty = number of strategies.
    """
    T = panel.fwd_returns.shape[0]
    nets: dict[str, list[float]] = {k: [] for k in strategies}
    turns: dict[str, list[float]] = {k: [] for k in strategies}
    folds_used = 0

    if T >= min_train + n_folds:
        edges = sorted(set(int(e) for e in np.linspace(min_train, T, n_folds + 1)))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if hi <= lo:
                continue
            # Embargo: purge the last `embargo` train rows so a label whose
            # horizon overlaps the test block can't leak into training.
            train = _slice_panel(panel, 0, max(1, lo - embargo))
            test = _slice_panel(panel, lo, hi)
            folds_used += 1
            for name, strat in strategies.items():
                if callable(strat):
                    W = strat(test)
                else:
                    strat.fit(train)
                    W = strat.predict_weights(test)
                nets[name].extend(simulate(W, test.fwd_returns, cost).tolist())
                turns[name].extend(_turnover_series(W, test.fwd_returns).tolist())

    n_trials = len(strategies)
    metrics = [
        portfolio_metrics(name, np.asarray(nets[name]), np.asarray(turns[name]),
                          n_trials=n_trials, seed=seed)
        for name in strategies
    ]
    baseline_names = {"cash", "equal_weight", "long_short"}
    base_sr = max((m.sharpe for m in metrics if m.name in baseline_names), default=0.0)
    learned = [m for m in metrics if m.name not in baseline_names and m.n > 0]
    best = max(learned, key=lambda m: m.sharpe) if learned else max(metrics, key=lambda m: m.sharpe)

    reasons: list[str] = []
    if best.sharpe <= 0:
        reasons.append("sharpe<=0")
    if best.sharpe <= base_sr:
        reasons.append("does_not_beat_baseline_sharpe")
    if best.deflated_sharpe < 0.95:
        reasons.append("deflated_sharpe<0.95")
    if not (best.ci_lo > 0):
        reasons.append("ci95_includes_0")

    # Probability of backtest overfitting across the competing strategies (CSCV).
    lengths = {len(v) for v in nets.values()}
    pbo = float("nan")
    if len(lengths) == 1 and lengths != {0} and len(nets) >= 2:
        matrix = np.column_stack([np.asarray(nets[name]) for name in strategies])
        pbo = _stats.probability_of_backtest_overfitting(matrix)

    return PortfolioReport(
        cost=cost, n_folds=folds_used, metrics=metrics,
        best_name=best.name, best_beats_baselines=not reasons, gate_reasons=reasons,
        net_returns={name: nets[name] for name in strategies}, pbo=pbo,
    )


def print_portfolio_report(report: PortfolioReport) -> None:
    print("\n=== Eve Portfolio Verdict (post-cost walk-forward, annualized) ===")
    print(f"cost/turnover: {report.cost * 1e4:.1f} bps | folds: {report.n_folds}")
    print(f"{'strategy':<14}{'n':>6}{'annRet':>9}{'annVol':>9}{'Sharpe':>9}"
          f"{'DSR':>7}{'t':>7}{'turn':>8}")
    for m in report.metrics:
        print(f"{m.name:<14}{m.n:>6}{m.ann_return:>9.3f}{m.ann_vol:>9.3f}{m.sharpe:>9.2f}"
              f"{m.deflated_sharpe:>7.2f}{m.t_stat:>7.2f}{m.avg_turnover:>8.2f}")
    print()
    if np.isfinite(report.pbo):
        print(f"PBO (prob. backtest overfitting, CSCV): {report.pbo:.2f}  (lower is better; <0.5 robust)")
    if report.gate_reasons:
        print(f"best '{report.best_name}' blocked by: {', '.join(report.gate_reasons)}")
    print(report.machine_line())
