"""FMP HTTP ingest — fast, cached, full-universe fundamentals + delisted prices.

The FMP MCP connector is one call per symbol routed through the agent context, which
does not scale to 472 names. With an ``FMP_API_KEY`` in ``.env`` we hit the same
``stable`` endpoints directly over HTTP (no context cost) and cache pull-once — the
same pattern as :mod:`eve_ingest` / :mod:`eve_yahoo`. Two products:

* **Fundamentals** (``keymetrics_http_fetcher``): a fetcher that plugs straight into
  :func:`eve_fundamentals.fetch_fundamentals`, so the cache + PIT-alignment logic is
  unchanged — only the transport differs from the MCP path.
* **Delisted prices** (``fetch_eod_bars``): EOD OHLCV for *any* ticker incl. dropped
  S&P names — the data needed to actually close the survivorship hole
  (`eve_universe`: 178 members we lacked prices for).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Sequence

FMP_BASE = "https://financialmodelingprep.com/stable"


def load_fmp_key(env_path: str | Path = None) -> str:
    """Read ``FMP_API_KEY`` from a ``.env`` file (never logged)."""
    p = Path(env_path) if env_path else Path(__file__).resolve().parents[1] / ".env"
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("FMP_API_KEY=") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise KeyError("FMP_API_KEY not found in .env")


def _fmp_get(endpoint: str, params: dict, key: str, *, retries: int = 3, timeout: int = 30):
    q = dict(params)
    q["apikey"] = key
    url = f"{FMP_BASE}/{endpoint}?" + urllib.parse.urlencode(q)
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 — transient HTTP, retry with backoff
            last = e
            time.sleep(1.0 * (i + 1))
    raise last


def keymetrics_http_fetcher(
    key: str, *, period: str = "quarter", limit: int = 80
) -> Callable[[str], list[dict]]:
    """Fetcher for :func:`eve_fundamentals.fetch_fundamentals` (HTTP key-metrics).

    ``limit=80`` quarters ≈ 20 years; ``period='quarter'`` gives the quarterly cadence
    the PIT alignment expects.
    """
    def _fetch(symbol: str) -> list[dict]:
        res = _fmp_get("key-metrics", {"symbol": symbol, "period": period, "limit": limit}, key)
        return res if isinstance(res, list) else []
    return _fetch


def fetch_eod_bars(
    symbols: Sequence[str],
    key: str,
    cache_dir: str | Path,
    *,
    start: str,
    end: str,
    sleep: float = 0.0,
    refresh: bool = False,
) -> dict[str, list[dict]]:
    """Cached EOD OHLCV per symbol (incl. delisted) -> normalized bar dicts.

    Each bar: ``{ts, open, high, low, close, volume}`` (ts = ``YYYY-MM-DD``), oldest
    first — the shape :func:`eve_data.read_symbol_bars` consumers expect. Pull-once.
    """
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    for sym in symbols:
        fp = cdir / f"{sym.upper()}.json"
        if fp.exists() and not refresh:
            out[sym.upper()] = json.loads(fp.read_text())
            continue
        raw = _fmp_get("historical-price-eod/full",
                       {"symbol": sym, "from": start, "to": end}, key)
        rows = raw if isinstance(raw, list) else raw.get("historical", []) if isinstance(raw, dict) else []
        bars = [
            {"ts": r["date"][:10], "open": float(r.get("open", 0.0)),
             "high": float(r.get("high", 0.0)), "low": float(r.get("low", 0.0)),
             "close": float(r.get("close", 0.0)), "volume": float(r.get("volume", 0.0) or 0.0)}
            for r in rows if r.get("date") and r.get("close") is not None
        ]
        bars.sort(key=lambda b: b["ts"])
        fp.write_text(json.dumps(bars))
        out[sym.upper()] = bars
        if sleep:
            time.sleep(sleep)
    return out
