"""
Kalshi BTC market client + Polymarket↔Kalshi cross-venue arbitrage scan.

Backed by Gebele & Matthes (2026): ~6% of events are dual-listed across
venues with persistent 2-4% execution-aware price deviations — a near
risk-free edge the bot has zero exposure to today.

Same binary event listed on both venues lets us lock $1:

    buy YES on the cheaper venue + buy NO on the other
    cost = ask_yes_A + ask_no_B      (ask_no = 1 − bid_yes)
    edge = 1 − min(cost_AB, cost_BA) − fees

Event identity is the hard part (semantic non-fungibility); we match on
(symbol, strike±tol, expiry±tol) for BTC threshold/hourly markets, which is
exact enough for the price-defined crypto contracts.

Market data here uses Kalshi's **public** GET /markets (no auth).  Live
two-venue execution needs Kalshi credentials + a Kalshi executor, so the
scanner is detect-and-log by default (XVENUE_ENABLED=0).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

DEFAULT_BASE = "https://api.elections.kalshi.com/trade-api/v2"
# Kalshi crypto series tickers (hourly/threshold BTC & ETH price markets).
BTC_SERIES = ("KXBTCD", "KXBTC", "KXETHD", "KXETH")


@dataclass
class KalshiMarket:
    ticker: str
    symbol: str          # btcusdt / ethusdt
    strike: float
    expiry_ts: float
    yes_bid: float       # 0..1
    yes_ask: float       # 0..1


@dataclass
class XVenueArb:
    poly_market: object
    kalshi: KalshiMarket
    leg_desc: str
    cost: float
    edge: float          # 1 − cost − fees


class KalshiClient:
    def __init__(self, base: str = DEFAULT_BASE):
        self.base = base.rstrip("/")

    async def fetch_btc_markets(self, session: aiohttp.ClientSession) -> list[KalshiMarket]:
        out: list[KalshiMarket] = []
        for series in BTC_SERIES:
            try:
                out.extend(await self._fetch_series(session, series))
            except Exception as exc:
                log.debug("Kalshi series %s fetch failed: %s", series, exc)
        return out

    async def _fetch_series(
        self, session: aiohttp.ClientSession, series: str
    ) -> list[KalshiMarket]:
        url = f"{self.base}/markets"
        params = {"series_ticker": series, "status": "open", "limit": "200"}
        async with session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=8)
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
        rows = data.get("markets", []) if isinstance(data, dict) else []
        out: list[KalshiMarket] = []
        sym = "ethusdt" if "ETH" in series else "btcusdt"
        for m in rows:
            strike = _strike_of(m)
            expiry = _expiry_of(m)
            if strike <= 0 or expiry <= 0:
                continue
            # Kalshi prices are integer cents 1..99.
            yb = float(m.get("yes_bid", 0) or 0) / 100.0
            ya = float(m.get("yes_ask", 0) or 0) / 100.0
            if ya <= 0:
                continue
            out.append(KalshiMarket(
                ticker=str(m.get("ticker", "")),
                symbol=sym, strike=strike, expiry_ts=expiry,
                yes_bid=yb, yes_ask=ya,
            ))
        return out


def _strike_of(m: dict) -> float:
    for k in ("floor_strike", "cap_strike", "strike"):
        v = m.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def _expiry_of(m: dict) -> float:
    import datetime as _dt
    s = m.get("close_time") or m.get("expiration_time")
    if not s:
        return 0.0
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def find_xvenue_arbs(
    poly_markets: list,
    poly_ws,
    kalshi_markets: list[KalshiMarket],
    min_credit: float = 0.01,
    strike_tol: float = 1e-6,
    expiry_tol_secs: float = 120.0,
    fee: float = 0.02,
) -> list[XVenueArb]:
    """Match poly↔kalshi on identical events and find lock-$1 opportunities."""
    out: list[XVenueArb] = []
    # index kalshi by symbol for cheap lookup
    by_sym: dict[str, list[KalshiMarket]] = {}
    for k in kalshi_markets:
        by_sym.setdefault(k.symbol, []).append(k)

    for pm in poly_markets:
        if not getattr(pm, "is_threshold", False) or getattr(pm, "strike", 0) <= 0:
            continue
        yes_book = poly_ws.snapshot(pm.yes_token_id)
        no_book = poly_ws.snapshot(pm.no_token_id)
        if yes_book is None or no_book is None:
            continue
        for km in by_sym.get(pm.symbol, ()):
            if abs(km.strike - pm.strike) > max(strike_tol, pm.strike * 1e-4):
                continue
            if abs(km.expiry_ts - pm.expiry_ts) > expiry_tol_secs:
                continue
            # buy YES poly + NO kalshi  vs  buy NO poly + YES kalshi
            ask_yes_poly = yes_book.best_ask
            ask_no_poly = no_book.best_ask
            ask_no_kalshi = 1.0 - km.yes_bid
            ask_yes_kalshi = km.yes_ask
            cost_ab = ask_yes_poly + ask_no_kalshi
            cost_ba = ask_no_poly + ask_yes_kalshi
            cost = min(cost_ab, cost_ba)
            edge = 1.0 - cost - fee
            if edge >= min_credit:
                desc = ("YESpoly+NOkalshi" if cost_ab <= cost_ba
                        else "NOpoly+YESkalshi")
                out.append(XVenueArb(pm, km, desc, cost, edge))
                log.info(
                    "X-VENUE arb %s K=%.0f exp=%d %s cost=%.4f edge=%.4f",
                    pm.symbol, pm.strike, int(pm.expiry_ts), desc, cost, edge,
                )
    return out
