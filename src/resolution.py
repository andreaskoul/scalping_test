"""
Resolution-source awareness — anchor Up/Down strikes to the real reference.

The dominant bug in the old pipeline: it anchored each Up/Down strike to the
*first Binance mid we happened to observe*, a synthetic value uncorrelated
with how the market actually settles.  That is why `SKIP_UPDOWN` was on and
the whole 5m/15m/hourly Up/Down universe — the highest-volume segment — was
abandoned.

Reality (verified):
  - 5m / 15m / threshold markets settle on the **Chainlink BTC/USD** stream,
    read on-chain at the window-end timestamp.
  - Hourly Up/Down settles on the **Binance 1h candle** (close vs open).
  - The "Price to Beat" is the reference price at the window *open*, which
    Polymarket publishes.

We reconstruct the Price to Beat as the reference price at the window-open
timestamp (the open of the 1-minute bar containing it), which both Chainlink
and the Binance candle agree on at the boundary.  That replaces the bogus
first-observation strike with the value the market is actually judged against.
"""

from __future__ import annotations

import logging
import re
import time
from enum import Enum

import aiohttp

log = logging.getLogger(__name__)

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

# duration of the window from slug/question text
_DUR_RE = re.compile(r"\b(\d+)\s*(?:m|min|minute)s?\b", re.I)
_SLUG_DUR_RE = re.compile(r"updown[-_](\d+)m", re.I)


class ResolutionSource(str, Enum):
    CHAINLINK = "chainlink"          # 5m / 15m / threshold
    BINANCE_CANDLE = "binance_candle"  # hourly up/down
    UNKNOWN = "unknown"


def window_seconds(market) -> float:
    """Best-effort window length for an Up/Down market, in seconds."""
    slug = getattr(market, "slug", "") or ""
    m = _SLUG_DUR_RE.search(slug)
    if m:
        return float(m.group(1)) * 60.0
    q = (getattr(market, "question", "") or "")
    m = _DUR_RE.search(q)
    if m:
        return float(m.group(1)) * 60.0
    if re.search(r"hour", q, re.I) or re.search(r"updown[-_]60m", slug, re.I):
        return 3600.0
    return 0.0


def resolution_source(market) -> ResolutionSource:
    if getattr(market, "is_threshold", False):
        return ResolutionSource.CHAINLINK
    if getattr(market, "is_updown", False):
        w = window_seconds(market)
        if w and w >= 3600.0:
            return ResolutionSource.BINANCE_CANDLE
        return ResolutionSource.CHAINLINK
    return ResolutionSource.UNKNOWN


def window_start_ts(market) -> float:
    """Unix ts of the window open = expiry − window length."""
    exp = getattr(market, "expiry_ts", 0.0)
    w = window_seconds(market)
    if exp <= 0 or w <= 0:
        return 0.0
    return exp - w


class PriceToBeatCache:
    """Fetches and caches the window-open reference price per market."""

    def __init__(self):
        # condition_id → (anchored_strike, fetched_at)
        self._cache: dict[str, tuple[float, float]] = {}

    async def anchored_strike(
        self, session: aiohttp.ClientSession, market
    ) -> float | None:
        """Reference price at window open for an Up/Down market, or None."""
        cid = getattr(market, "condition_id", "")
        if cid in self._cache:
            return self._cache[cid][0]

        start = window_start_ts(market)
        if start <= 0:
            return None
        # Don't fetch for a window that hasn't opened yet.
        if start > time.time() + 5:
            return None

        sym = getattr(market, "symbol", "btcusdt")
        ref = await self._fetch_open(session, sym, start)
        if ref is not None and ref > 0:
            self._cache[cid] = (ref, time.monotonic())
            log.info(
                "Price-to-Beat anchored: %s K=%.2f (window open %d)",
                getattr(market, "question", "")[:48], ref, int(start),
            )
            return ref
        return None

    async def _fetch_open(
        self, session: aiohttp.ClientSession, symbol: str, start_ts: float
    ) -> float | None:
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": int(start_ts // 60 * 60) * 1000,
            "limit": 1,
        }
        try:
            async with session.get(
                BINANCE_KLINES, params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                if r.status != 200:
                    return None
                rows = await r.json()
        except Exception as exc:
            log.debug("Price-to-Beat fetch failed (%s @ %d): %s", symbol, int(start_ts), exc)
            return None
        if not rows:
            return None
        # kline row: [openTime, open, high, low, close, ...]
        try:
            return float(rows[0][1])
        except (IndexError, ValueError, TypeError):
            return None
