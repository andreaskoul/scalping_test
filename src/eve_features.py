"""Eve feature factory — Alpha158-style cross-sectional factors.

The survey's verdict was blunt: *architecture is not the bottleneck, signal is.*
qlib's edge is not its models but its **feature library** (Alpha158/Alpha360) —
dozens of rolling price/volume factors, cross-sectionally rank-normalized. This
module is that lane: a leakage-safe per-symbol factor set, assembled into a
cross-sectional :class:`~src.eve_portfolio.Panel` and **rank-normalized per
date** (qlib's CSRankNorm), ready for a GBDT/transformer predictor and the RL
allocator.

Every factor at bar t uses only bars ≤ t; the panel's ``fwd_returns[t]`` is the
*next* period's return, so nothing leaks. numpy-only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _swv

from .eve_portfolio import Panel

_EPS = 1e-12
DEFAULT_WINDOWS = (5, 10, 20, 30, 60)


def _roll(a: np.ndarray, w: int, fn) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    if a.shape[0] >= w:
        out[w - 1:] = fn(_swv(a, w))
    return out


def _roll_corr(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    out = np.full(a.shape[0], np.nan)
    if a.shape[0] >= w:
        A, B = _swv(a, w), _swv(b, w)
        Am = A - A.mean(1, keepdims=True)
        Bm = B - B.mean(1, keepdims=True)
        num = (Am * Bm).sum(1)
        den = np.sqrt((Am ** 2).sum(1) * (Bm ** 2).sum(1)) + _EPS
        out[w - 1:] = num / den
    return out


def _roll_slope(a: np.ndarray, w: int) -> np.ndarray:
    """Per-window OLS slope of a vs time index, normalized by level."""
    out = np.full(a.shape[0], np.nan)
    if a.shape[0] >= w:
        t = np.arange(w, dtype=float)
        tc = t - t.mean()
        tvar = (tc ** 2).sum() + _EPS
        A = _swv(a, w)
        num = (tc[None, :] * (A - A.mean(1, keepdims=True))).sum(1)
        out[w - 1:] = (num / tvar) / (a[w - 1:] + _EPS)
    return out


def compute_features(bars: Sequence[Any], windows: Sequence[int] = DEFAULT_WINDOWS):
    """Per-bar leakage-safe factors for one symbol.

    Returns (dates, feature_matrix[T, F], feature_names). Rows whose deepest
    window is not yet available are NaN (dropped at panel assembly).
    """
    o = np.array([b.open for b in bars], dtype=float)
    h = np.array([b.high for b in bars], dtype=float)
    low = np.array([b.low for b in bars], dtype=float)
    c = np.array([b.close for b in bars], dtype=float)
    v = np.array([float(b.volume) for b in bars], dtype=float)
    dates = [b.ts[:10] for b in bars]
    T = c.shape[0]
    r = np.zeros(T)
    r[1:] = c[1:] / (c[:-1] + _EPS) - 1.0
    gains = np.where(r > 0, r, 0.0)
    losses = np.where(r < 0, -r, 0.0)
    dvol = c * v  # dollar volume — the liquidity base series (Amihud denominator)

    feats: dict[str, np.ndarray] = {}
    # K-line (intrabar) shape factors — no window.
    feats["kmid"] = (c - o) / (o + _EPS)
    feats["klen"] = (h - low) / (o + _EPS)
    feats["kup"] = (h - np.maximum(o, c)) / (o + _EPS)
    feats["klow"] = (np.minimum(o, c) - low) / (o + _EPS)
    feats["kmid2"] = (c - o) / (h - low + _EPS)

    for w in windows:
        roc = np.full(T, np.nan)
        if T > w:
            roc[w:] = c[w:] / (c[:-w] + _EPS) - 1.0
        feats[f"roc{w}"] = roc
        feats[f"ma{w}"] = _roll(c, w, lambda x: x.mean(1)) / (c + _EPS) - 1.0
        feats[f"std{w}"] = _roll(r, w, lambda x: x.std(1))
        max_c = _roll(c, w, lambda x: x.max(1))
        min_c = _roll(c, w, lambda x: x.min(1))
        feats[f"rsv{w}"] = (c - min_c) / (max_c - min_c + _EPS)  # stochastic %K
        avg_gain = _roll(gains, w, lambda x: x.mean(1))
        avg_loss = _roll(losses, w, lambda x: x.mean(1))
        feats[f"rsi{w}"] = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + _EPS))
        feats[f"vma{w}"] = _roll(v, w, lambda x: x.mean(1)) / (v + _EPS)
        feats[f"corr{w}"] = _roll_corr(c, v, w)
        feats[f"beta{w}"] = _roll_slope(c, w)
        # Liquidity (Gu-Kelly-Xiu top-3 family): log mean dollar-volume + Amihud
        # illiquidity (|return| per $ traded). Slow-decay, low-turnover signals.
        feats[f"dvol{w}"] = np.log1p(_roll(dvol, w, lambda x: x.mean(1)))
        feats[f"amihud{w}"] = _roll(np.abs(r) / (dvol + 1.0), w, lambda x: x.mean(1))

    names = list(feats.keys())
    mat = np.column_stack([feats[n] for n in names])
    return dates, mat, names


def _csrank_norm(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank-normalize each row to ~[-0.5, 0.5] (qlib CSRankNorm).

    Robust to outliers (ranks, not values). NaNs are treated as mid-rank.
    """
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        row = x[t]
        finite = np.isfinite(row)
        n = finite.sum()
        if n <= 1:
            continue
        ranks = np.full(row.shape[0], np.nan)
        order = np.argsort(np.where(finite, row, np.inf), kind="stable")
        ranks[order[:n]] = np.arange(n)
        # NaNs -> middle rank
        ranks[~finite] = (n - 1) / 2.0
        out[t] = ranks / (n - 1) - 0.5
    return out


def build_feature_panel(
    lake: Any,
    symbols: Sequence[str],
    *,
    asset_class: str = "equity",
    provider: str = "yahoo",
    timeframe: str = "1d",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    rank_normalize: bool = True,
    min_dates_frac: float = 0.9,
    rebalance: str = "1d",
) -> Panel:
    """Assemble a cross-sectional factor panel from lake bars (leakage-safe).

    Only dates present for every symbol are kept (dense matrices). Each factor is
    rank-normalized across symbols at each date. ``fwd_returns[t, i]`` is the
    close-to-close return from the panel date t to the next panel date.

    ``rebalance`` controls the holding/decision horizon. ``"1d"`` keeps every
    common trading day (daily next-day labels). ``"1m"`` keeps only each month's
    last common date, so ``fwd_returns`` is the *next-month* return and the panel
    rebalances monthly — the slow-decay / low-turnover horizon the literature
    (Gu-Kelly-Xiu) ties to capturable post-cost Sharpe. Features are still the
    daily-window factors snapshotted at month-end (only the label horizon changes).
    """
    from .eve_data import read_symbol_bars
    from .eve_ingest import bars_dataset

    # Memory-lean: one (n_valid, F) matrix + a date->row index per symbol, rather
    # than millions of tiny per-(symbol, date) arrays.
    per_sym_feat: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    per_sym_close: dict[str, dict[str, float]] = {}
    names: list[str] | None = None
    for sym in symbols:
        bars = read_symbol_bars(lake, provider=provider, asset_class=asset_class,
                                symbol=sym, dataset=bars_dataset(timeframe))
        if len(bars) <= max(windows) + 2:
            continue
        dates, mat, nm = compute_features(bars, windows)
        names = nm
        valid_idx = [i for i in range(len(dates)) if np.isfinite(mat[i]).all()]
        if not valid_idx:
            continue
        date_to_row = {dates[i]: r for r, i in enumerate(valid_idx)}
        per_sym_feat[sym] = (date_to_row, mat[valid_idx])
        per_sym_close[sym] = {b.ts[:10]: float(b.close) for b in bars}

    # Drop short-history symbols (late listers) so the common window stays long
    # — otherwise one recent IPO collapses the date intersection.
    if per_sym_feat:
        max_dates = max(len(v[0]) for v in per_sym_feat.values())
        per_sym_feat = {s: v for s, v in per_sym_feat.items()
                        if len(v[0]) >= min_dates_frac * max_dates}
    syms = sorted(per_sym_feat)
    if len(syms) < 2 or names is None:
        raise ValueError("need >=2 symbols with enough history to build a feature panel")
    common = sorted(set.intersection(*[set(per_sym_feat[s][0]) for s in syms]))
    if rebalance == "1m":
        # Keep the last common date of each calendar month (common is sorted asc),
        # so the panel rebalances monthly and fwd is the next month-end return.
        by_month: dict[str, str] = {}
        for d in common:
            by_month[d[:7]] = d
        common = sorted(by_month.values())
    elif rebalance != "1d":
        raise ValueError(f"unsupported rebalance {rebalance!r} (use '1d' or '1m')")
    if len(common) < 3:
        raise ValueError("not enough overlapping dates across symbols")

    F = len(names)
    T = len(common) - 1  # last date has no forward return
    signals = np.zeros((T, len(syms), F))
    fwd = np.zeros((T, len(syms)))
    for j, sym in enumerate(syms):
        d2r, m = per_sym_feat[sym]
        close = per_sym_close[sym]
        for t in range(T):
            signals[t, j] = m[d2r[common[t]]]
            fwd[t, j] = close[common[t + 1]] / (close[common[t]] + _EPS) - 1.0

    if rank_normalize:
        for k in range(F):
            signals[:, :, k] = _csrank_norm(signals[:, :, k])

    return Panel(dates=common[:T], symbols=syms, signals=signals, fwd_returns=fwd)


class _DictBar:
    """Lightweight bar wrapper so FMP dicts feed compute_features like AlpacaBars."""
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, d: dict):
        self.ts = d["ts"]
        self.open = float(d["open"]); self.high = float(d["high"])
        self.low = float(d["low"]); self.close = float(d["close"])
        self.volume = float(d.get("volume", 0.0) or 0.0)


def build_ragged_feature_panel(
    lake: Any,
    current_symbols: Sequence[str],
    delisted_bars: dict[str, Sequence[dict]],
    *,
    asset_class: str = "equity",
    provider: str = "yahoo",
    timeframe: str = "1d",
    windows: Sequence[int] = DEFAULT_WINDOWS,
    rank_normalize: bool = True,
    min_names: int = 30,
) -> Panel:
    """Survivorship-corrected MONTHLY panel over a *ragged* universe.

    Combines current names (from the lake) with delisted names (FMP EOD bar dicts,
    ``{ts,open,high,low,close,volume}``) whose lifespans differ. Instead of a dense
    common-date intersection (which would drop every dropped name), it aligns each
    symbol to its own month-end snapshots on a **union monthly calendar**, marks
    months where a name is not trading as NaN features (excluded from GBDT training)
    with a ``valid`` mask, and sets ``fwd_returns`` to next-month return where valid
    else 0. This is the only leak-safe way to include bankrupt/acquired names —
    closing the survivorship hole `eve_universe` quantified.
    """
    from .eve_data import read_symbol_bars
    from .eve_ingest import bars_dataset

    # Per-symbol month-end feature row + close, keyed by 'YYYY-MM'.
    per_sym: dict[str, tuple[dict[str, np.ndarray], dict[str, float]]] = {}
    names: list[str] | None = None
    sources = [(s.upper(), None) for s in current_symbols] + \
              [(s.upper(), delisted_bars[s]) for s in delisted_bars]
    for sym, fmp in sources:
        if fmp is None:
            bars = read_symbol_bars(lake, provider=provider, asset_class=asset_class,
                                    symbol=sym, dataset=bars_dataset(timeframe))
        else:
            bars = [_DictBar(d) for d in fmp]
        if len(bars) <= max(windows) + 2:
            continue
        dates, mat, nm = compute_features(bars, windows)
        names = nm
        feat_by_month: dict[str, np.ndarray] = {}
        close_by_month: dict[str, float] = {}
        closes = {b.ts[:10]: float(b.close) for b in bars}
        for i in range(len(dates)):
            if not np.isfinite(mat[i]).all():
                continue
            m = dates[i][:7]
            feat_by_month[m] = mat[i]          # last valid row in the month wins
            close_by_month[m] = closes[dates[i]]
        if feat_by_month:
            per_sym[sym] = (feat_by_month, close_by_month)

    if not per_sym or names is None:
        raise ValueError("no symbols with enough history for a ragged panel")
    syms = sorted(per_sym)
    # Union monthly calendar, keeping months with >= min_names valid symbols.
    month_count: dict[str, int] = {}
    for fbm, _ in per_sym.values():
        for m in fbm:
            month_count[m] = month_count.get(m, 0) + 1
    months = sorted(m for m, c in month_count.items() if c >= min_names)
    if len(months) < 3:
        raise ValueError("ragged calendar too short")
    T, N, F = len(months) - 1, len(syms), len(names)
    signals = np.full((T, N, F), np.nan)
    fwd = np.zeros((T, N))
    valid = np.zeros((T, N), dtype=bool)
    for j, sym in enumerate(syms):
        fbm, cbm = per_sym[sym]
        for t in range(T):
            m, mn = months[t], months[t + 1]
            if m in fbm:
                signals[t, j] = fbm[m]
                if m in cbm and mn in cbm:
                    fwd[t, j] = cbm[mn] / (cbm[m] + _EPS) - 1.0
                    valid[t, j] = True
    if rank_normalize:
        for k in range(F):
            col = _csrank_norm(signals[:, :, k])    # fills NaN -> mid-rank
            signals[:, :, k] = np.where(valid, col, np.nan)  # re-mask absent -> NaN
    else:
        signals = np.where(valid[:, :, None], np.nan_to_num(signals), np.nan)
    return Panel(dates=months[:T], symbols=syms, signals=signals,
                 fwd_returns=fwd, valid=valid)
