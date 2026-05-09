"""
Deribit options-implied volatility fetcher.

Pulls the at-the-money mark IV for the nearest-expiry BTC and ETH
options on Deribit and exposes it as a forward-looking σ estimate.
This is the same forward-vol signal the maker bots on Polymarket use
to price binary outcomes — using realised vol alone (which is purely
backward-looking) puts us at a structural information disadvantage.

Why a floor not a replacement:
  - When realised vol spikes above IV (regime break / news event), the
    realised signal is the right one and we don't want IV smoothing it
    away.
  - When realised is napping (last 60s was quiet), IV still represents
    the market's consensus on the next 5–60 min and stops the model
    from being overconfident.

Result: σ_used = max(σ_ewma, σ_iv, σ_floor)

Refresh cadence: every IV_REFRESH_SECS (default 60 s).  Deribit's
get_book_summary_by_currency returns ~1000 instruments per currency in
one call, so the polling cost is small.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

log = logging.getLogger(__name__)

DERIBIT_API = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
IV_REFRESH_SECS: float = 60.0
HTTP_TIMEOUT: float = 8.0

# How many ATM-nearest strikes to average to smooth out single-instrument noise.
ATM_AVERAGE_K: int = 4

# How many of the soonest-expiring expiries to consider ATM-near.
NEAREST_EXPIRIES: int = 1


@dataclass
class IVSnapshot:
    symbol: str           # "btcusdt" / "ethusdt"
    sigma_annual: float   # annualised IV, e.g. 0.55
    underlying: float
    expiry_unix: float
    n_strikes: int
    fetched_at: float     # monotonic


def _parse_instrument(name: str) -> tuple[str, float, str] | None:
    """`BTC-9MAY26-80000-C` → ('9MAY26', 80000.0, 'C')."""
    parts = name.split("-")
    if len(parts) != 4:
        return None
    try:
        return parts[1], float(parts[2]), parts[3]
    except ValueError:
        return None


def _expiry_to_unix(expiry: str) -> float:
    """`9MAY26` → unix ts at 08:00 UTC (Deribit settlement time)."""
    try:
        # Deribit format: D[D]MMMYY  (e.g., 9MAY26 or 12MAY26)
        dt = datetime.strptime(expiry, "%d%b%y").replace(
            hour=8, minute=0, tzinfo=timezone.utc,
        )
        return dt.timestamp()
    except ValueError:
        return 0.0


def _atm_iv_from_summary(rows: list[dict]) -> tuple[float, float, float, int] | None:
    """Return (iv, underlying, expiry_unix, n) using the nearest expiry,
    averaged across the K strikes closest to ATM."""
    by_expiry: dict[str, list[dict]] = {}
    for o in rows:
        name = o.get("instrument_name", "")
        parsed = _parse_instrument(name)
        if not parsed:
            continue
        exp, strike, kind = parsed
        try:
            iv = float(o.get("mark_iv") or 0.0) / 100.0
            under = float(o.get("underlying_price") or 0.0)
        except (TypeError, ValueError):
            continue
        if iv <= 0 or under <= 0:
            continue
        by_expiry.setdefault(exp, []).append({
            "strike": strike, "kind": kind, "iv": iv, "under": under,
        })

    if not by_expiry:
        return None

    # Pick the soonest expiry that is still in the future.
    now_ts = time.time()
    expiries = sorted(by_expiry.keys(), key=lambda e: _expiry_to_unix(e))
    chosen_exp = None
    for exp in expiries:
        if _expiry_to_unix(exp) > now_ts + 60:
            chosen_exp = exp
            break
    if chosen_exp is None:
        return None

    bucket = by_expiry[chosen_exp]
    underlying = sum(o["under"] for o in bucket) / len(bucket)
    bucket.sort(key=lambda o: abs(o["strike"] - underlying))
    nearest = bucket[: ATM_AVERAGE_K * 2]   # call + put × ATM_AVERAGE_K
    if not nearest:
        return None
    iv = sum(o["iv"] for o in nearest) / len(nearest)
    return iv, underlying, _expiry_to_unix(chosen_exp), len(nearest)


class DeribitIV:
    """Background-refreshes ATM IV for a list of symbols."""

    SYMBOL_TO_CCY = {
        "btcusdt": "BTC",
        "ethusdt": "ETH",
    }

    def __init__(self, symbols: list[str], refresh_secs: float = IV_REFRESH_SECS):
        self._symbols = [s.lower() for s in symbols]
        self._refresh = refresh_secs
        self._snapshots: dict[str, IVSnapshot] = {}
        self._stop = asyncio.Event()

    def snapshot(self, symbol: str) -> IVSnapshot | None:
        s = self._snapshots.get(symbol.lower())
        if s is None:
            return None
        # Stale check — if the fetcher has been silent for >5x refresh
        # interval, treat as unavailable.
        if (time.monotonic() - s.fetched_at) > 5 * self._refresh:
            return None
        return s

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            log.info("Deribit IV poller started (refresh=%.0fs, symbols=%s)",
                     self._refresh, self._symbols)
            while not self._stop.is_set():
                for sym in self._symbols:
                    ccy = self.SYMBOL_TO_CCY.get(sym)
                    if not ccy:
                        continue
                    try:
                        iv = await self._fetch_one(session, ccy)
                        if iv is not None:
                            self._snapshots[sym] = IVSnapshot(
                                symbol=sym,
                                sigma_annual=iv[0],
                                underlying=iv[1],
                                expiry_unix=iv[2],
                                n_strikes=iv[3],
                                fetched_at=time.monotonic(),
                            )
                            log.debug(
                                "Deribit IV %s: σ=%.4f S=%.2f n=%d",
                                sym, iv[0], iv[1], iv[3],
                            )
                    except Exception as exc:
                        log.debug("Deribit IV fetch failed for %s: %s", ccy, exc)
                await asyncio.sleep(self._refresh)

    async def _fetch_one(
        self, session: aiohttp.ClientSession, currency: str
    ) -> tuple[float, float, float, int] | None:
        params = {"currency": currency, "kind": "option"}
        async with session.get(DERIBIT_API, params=params) as r:
            if r.status != 200:
                return None
            data = await r.json()
        rows = data.get("result") or []
        if not rows:
            return None
        return _atm_iv_from_summary(rows)
