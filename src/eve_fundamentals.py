"""Fundamentals lane — slow-decay value/quality/profitability factors (FMP).

The multi-feature survey's #1 missing family (`docs/EVE_SHARPE_LITERATURE.md`):
Gu-Kelly-Xiu's dominant non-price signals are firm characteristics — value,
profitability, quality, investment. They decay over *quarters*, not days, so they
carry alpha at the monthly horizon at a fraction of the turnover — exactly the
lever that pushed the GBDT book past beta net of cost.

Source = FMP ``key-metrics`` (quarterly). Two pieces, kept separate so the alpha
logic is pure and unit-tested without any network:

* :func:`fetch_fundamentals` — cached per-symbol ingest (an injectable ``fetcher``
  does the actual FMP MCP call; results are JSON-cached so a symbol is pulled once).
* :func:`augment_panel_with_fundamentals` — pure: align quarterly records to a
  panel's month-end dates **with a filing lag** (a record dated D is usable only at
  D + ``lag_days``, so we never use numbers before they were public), forward-fill,
  cross-sectionally rank-normalize, and concatenate onto the panel's price factors.

PIT discipline: the lag is the whole point. 10-Qs land ~30-45 days after quarter
end; we default to a conservative 60-day buffer so nothing leaks.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .eve_features import _csrank_norm
from .eve_portfolio import Panel

# Slow-decay value / quality / profitability fields from FMP key-metrics that map
# to the Gu-Kelly-Xiu dominant non-price families. All ratios (scale-free), and
# rank-normalized cross-sectionally so absolute units don't matter.
DEFAULT_FUND_FIELDS: tuple[str, ...] = (
    "returnOnEquity", "returnOnInvestedCapital", "returnOnAssets",   # profitability
    "earningsYield", "freeCashFlowYield", "evToEBITDA", "evToSales",  # value
    "currentRatio", "incomeQuality", "netDebtToEBITDA",              # quality/leverage
)


def _iso(d: str) -> date:
    return date.fromisoformat(d[:10])


def fetch_fundamentals(
    symbols: Sequence[str],
    fetcher: Callable[[str], list[dict]],
    cache_dir: str | Path,
    *,
    refresh: bool = False,
) -> dict[str, list[dict]]:
    """Pull quarterly key-metrics per symbol, caching each to ``cache_dir`` once.

    ``fetcher(symbol)`` returns the raw FMP key-metrics list (newest-first). A
    symbol whose cache file exists is not re-fetched (the user's "store api calls
    and don't rerun the same" rule), so a full-universe pull can resume across runs.
    """
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    for sym in symbols:
        fp = cdir / f"{sym.upper()}.json"
        if fp.exists() and not refresh:
            out[sym.upper()] = json.loads(fp.read_text())
            continue
        recs = fetcher(sym) or []
        fp.write_text(json.dumps(recs))
        out[sym.upper()] = recs
    return out


def _pit_series(
    records: list[dict], dates: Sequence[str], fields: Sequence[str], lag_days: int
) -> np.ndarray:
    """(T, F) PIT-aligned, forward-filled field values for one symbol.

    For each panel date t, use the most recent record whose ``date + lag_days`` is
    on or before t (so only already-filed numbers are visible). Missing -> NaN.
    """
    # Sort records oldest-first by their *available* date (period end + lag).
    avail = []
    for r in records:
        if not r.get("date"):
            continue
        try:
            adate = _iso(r["date"]) + timedelta(days=lag_days)
        except ValueError:
            continue
        avail.append((adate, r))
    avail.sort(key=lambda x: x[0])

    T, F = len(dates), len(fields)
    out = np.full((T, F), np.nan)
    i = 0
    cur: dict | None = None
    panel_dates = [_iso(d) for d in dates]
    for t, pd in enumerate(panel_dates):
        while i < len(avail) and avail[i][0] <= pd:
            cur = avail[i][1]
            i += 1
        if cur is not None:
            for k, f in enumerate(fields):
                v = cur.get(f)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    out[t, k] = float(v)
    return out


def augment_panel_with_fundamentals(
    panel: Panel,
    records_by_symbol: dict[str, list[dict]],
    *,
    fields: Sequence[str] = DEFAULT_FUND_FIELDS,
    lag_days: int = 60,
    rank_normalize: bool = True,
) -> Panel:
    """Concatenate PIT-aligned fundamental factors onto a price-factor panel.

    Returns a new :class:`Panel` whose ``signals`` is the original price factors
    with ``len(fields)`` fundamental factors appended on the feature axis (same T,
    N, fwd_returns). Symbols absent from ``records_by_symbol`` get all-NaN
    fundamentals, which rank-normalize to mid-rank (neutral) — so partial coverage
    degrades gracefully instead of dropping names.
    """
    T, N, F = panel.signals.shape
    Ff = len(fields)
    fund = np.full((T, N, Ff), np.nan)
    for j, sym in enumerate(panel.symbols):
        recs = records_by_symbol.get(sym.upper())
        if recs:
            fund[:, j, :] = _pit_series(recs, panel.dates, fields, lag_days)

    if rank_normalize:
        for k in range(Ff):
            fund[:, :, k] = _csrank_norm(fund[:, :, k])  # NaN -> mid-rank
    else:
        fund = np.nan_to_num(fund, nan=0.0)

    signals = np.concatenate([panel.signals, fund], axis=2)
    return Panel(dates=panel.dates, symbols=panel.symbols,
                 signals=signals, fwd_returns=panel.fwd_returns)


def fmp_keymetrics_fetcher(call_tool: Callable[..., Any], *, limit: int = 60) -> Callable[[str], list[dict]]:
    """Adapt an FMP-MCP ``key-metrics`` caller into a ``fetch_fundamentals`` fetcher.

    ``call_tool(endpoint="key-metrics", symbol=sym, period="quarter", limit=limit)``
    must return the parsed JSON list. Kept as a tiny adapter so the network edge is
    injected (and mocked in tests); ``limit=60`` quarters ≈ 15 years of history.
    """
    def _fetch(symbol: str) -> list[dict]:
        res = call_tool(endpoint="key-metrics", symbol=symbol, period="quarter", limit=limit)
        return res if isinstance(res, list) else []
    return _fetch
