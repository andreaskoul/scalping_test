"""
Binance perpetual funding-rate poller → carry + smart-money tilt.

Two uses, both missing from the spot-only pricer:

  1. **Carry r for risk-neutral pricing.**  Black–Scholes binary value uses
     a drift r; for crypto the right r is the perpetual cost-of-carry, well
     approximated by the funding rate annualised.  Portnaya (2026) inverts
     Binance options at the exchange r rather than r=0.

  2. **Smart-money directional tilt (DRADIS "Basis").**  Funding is what
     perp longs pay shorts; persistently positive funding = crowded longs
     (lean short / fade), negative = crowded shorts.  Used as a small,
     capped drift that *fades* extreme funding.

Funding settles every 8h on Binance; the predicted/last rate is polled from
the public FAPI premium-index endpoint.  Annualisation: rate * 3 * 365.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

FAPI_PREMIUM = "https://fapi.binance.com/fapi/v1/premiumIndex"
REFRESH_SECS: float = 60.0
HTTP_TIMEOUT: float = 8.0

# Cap the annualised carry/drift the funding signal can inject, so a single
# 8h print can't dominate a short-horizon binary price.
MAX_CARRY_ANNUAL: float = 0.30
# How hard to fade extreme funding as a directional drift (annualised cap).
FADE_DRIFT_CAP: float = 0.15


@dataclass
class FundingSnapshot:
    symbol: str
    funding_rate_8h: float      # raw per-8h rate (e.g. +0.0001 = 1bp)
    carry_annual: float         # annualised, capped
    fade_drift_annual: float    # smart-money fade tilt, capped
    fetched_at: float           # monotonic


def _to_perp(symbol: str) -> str:
    """`btcusdt` spot → `BTCUSDT` perp (USDT-M futures share the ticker)."""
    return symbol.upper()


class FundingOracle:
    """Background-refreshes perp funding for a set of symbols."""

    def __init__(self, symbols: list[str], refresh_secs: float = REFRESH_SECS):
        self._symbols = [s.lower() for s in symbols]
        self._refresh = refresh_secs
        self._snaps: dict[str, FundingSnapshot] = {}
        self._stop = asyncio.Event()

    def snapshot(self, symbol: str) -> FundingSnapshot | None:
        s = self._snaps.get(symbol.lower())
        if s is None:
            return None
        if (time.monotonic() - s.fetched_at) > 5 * self._refresh:
            return None
        return s

    def carry(self, symbol: str) -> float:
        s = self.snapshot(symbol)
        return s.carry_annual if s else 0.0

    def fade_drift(self, symbol: str) -> float:
        s = self.snapshot(symbol)
        return s.fade_drift_annual if s else 0.0

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            log.info("Funding poller started (refresh=%.0fs, symbols=%s)",
                     self._refresh, self._symbols)
            while not self._stop.is_set():
                for sym in self._symbols:
                    try:
                        await self._fetch_one(session, sym)
                    except Exception as exc:
                        log.debug("Funding fetch failed for %s: %s", sym, exc)
                await asyncio.sleep(self._refresh)

    async def _fetch_one(self, session: aiohttp.ClientSession, sym: str) -> None:
        params = {"symbol": _to_perp(sym)}
        async with session.get(FAPI_PREMIUM, params=params) as r:
            if r.status != 200:
                return
            data = await r.json()
        # Endpoint returns a dict for a single symbol.
        row = data[0] if isinstance(data, list) else data
        try:
            rate = float(row.get("lastFundingRate", 0.0))
        except (TypeError, ValueError):
            return
        carry = max(-MAX_CARRY_ANNUAL, min(MAX_CARRY_ANNUAL, rate * 3.0 * 365.0))
        # Fade: opposite sign of funding, magnitude saturating in the rate.
        fade = -max(-1.0, min(1.0, rate / 0.0005)) * FADE_DRIFT_CAP
        self._snaps[sym] = FundingSnapshot(
            symbol=sym,
            funding_rate_8h=rate,
            carry_annual=carry,
            fade_drift_annual=fade,
            fetched_at=time.monotonic(),
        )
        log.debug("Funding %s: rate=%.6f carry=%.3f fade=%.3f", sym, rate, carry, fade)
