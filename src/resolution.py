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


# Candidate fields under which Polymarket may publish the real Chainlink
# "Price to Beat" / window-open reference on the Gamma market object. We read
# the real value when present and fall back to the Binance-kline proxy.
_PTB_FIELDS = (
    "priceToBeat", "price_to_beat", "startPrice", "startingPrice",
    "openPrice", "referencePrice", "strikePrice",
)


def extract_published_ptb(market_raw: dict) -> float:
    """Best-effort read of the published Price-to-Beat from a raw market dict.

    Returns 0.0 if no recognised field is present (caller then uses the proxy).
    Forward-compatible: when Polymarket exposes the field we anchor to the exact
    settlement reference instead of a Binance approximation."""
    for k in _PTB_FIELDS:
        v = market_raw.get(k)
        if v in (None, "", 0):
            continue
        try:
            f = float(v)
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


class ChainlinkBasis:
    """EWMA of (settlement reference − Binance mid) per symbol.

    Bootstrapped for free whenever we anchor an Up/Down strike from a *published*
    Price-to-Beat: the difference between that Chainlink value and the Binance
    1-minute open at the same instant is a direct basis observation. The live
    pricer then shifts spot by this basis so p* is computed against the feed the
    market actually settles on, not raw Binance. Zero until a published PTB
    diverges from Binance (i.e. a no-op until real data is present)."""

    def __init__(self, half_life_secs: float = 3600.0):
        self.half_life = max(1.0, half_life_secs)
        self._b: dict[str, tuple[float, float]] = {}  # symbol → (ewma, last_ts)

    def record(self, symbol: str, basis: float) -> None:
        now = time.monotonic()
        prev = self._b.get(symbol)
        if prev is None:
            self._b[symbol] = (basis, now)
            return
        ewma, last = prev
        import math
        lam = math.exp(-(now - last) * math.log(2.0) / self.half_life)
        self._b[symbol] = (lam * ewma + (1.0 - lam) * basis, now)

    def value(self, symbol: str) -> float:
        p = self._b.get(symbol)
        return p[0] if p else 0.0


class PriceToBeatCache:
    """Fetches and caches the window-open reference price per market.

    Prefers the real published Price-to-Beat (PolyMarket.price_to_beat); falls
    back to the Binance 1-minute open as a proxy. When both are available it
    records the difference as a Chainlink-vs-Binance basis sample."""

    def __init__(self, basis: "ChainlinkBasis | None" = None):
        # condition_id → (anchored_strike, fetched_at)
        self._cache: dict[str, tuple[float, float]] = {}
        self._basis = basis

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
        published = float(getattr(market, "price_to_beat", 0.0) or 0.0)
        open_ref = await self._fetch_open(session, sym, start)

        if published > 0:
            strike = published
            if open_ref and open_ref > 0 and self._basis is not None:
                self._basis.record(sym, published - open_ref)   # settlement − binance
            source = "published"
        elif open_ref and open_ref > 0:
            strike = open_ref
            source = "binance-proxy"
        else:
            return None

        self._cache[cid] = (strike, time.monotonic())
        log.info(
            "Price-to-Beat anchored (%s): %s K=%.2f (window open %d)",
            source, getattr(market, "question", "")[:42], strike, int(start),
        )
        return strike

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
