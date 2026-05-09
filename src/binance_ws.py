"""
Binance public WebSocket client.

Subscribes to <symbol>@bookTicker and <symbol>@aggTrade streams.
Maintains:
  - best bid/ask → fair mid-price
  - **EWMA realised volatility** (RiskMetrics-style, λ ≡ exp(-Δt/τ)) on
    tick-level log-returns → σ_annual.  Half-life τ defaults to 30 s,
    which adapts ~3× faster than the previous 60 s SMA window and is
    more responsive to vol regime shifts in crypto.
  - **Order-flow imbalance drift estimator** from aggTrade aggressor
    sign × dollar volume over a rolling 30 s window.  Maps net signed
    flow as a fraction of gross flow into a small annualised drift term
    (capped ±OFI_MAX_DRIFT) that the lognormal pricer can use instead
    of the naive zero-drift assumption.

Reconnects automatically with exponential back-off on disconnect.
Callers read state via .snapshot() which is always consistent.
"""

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import websockets
from websockets.exceptions import ConnectionClosed

from .pricing import SIGMA_MIN, SIGMA_MAX

log = logging.getLogger(__name__)

WS_BASE = "wss://stream.binance.com:443/stream"

# EWMA half-life for realised vol — 30s gives ~3x faster response
# than the prior 60s SMA and tracks crypto vol regime shifts better.
SIGMA_HALF_LIFE_SECS: float = 30.0

# Window for order-flow imbalance drift estimator (seconds).
OFI_WINDOW_SECS: float = 30.0
# Max annualised drift produced by OFI signal (caps the directional bias).
OFI_MAX_DRIFT: float = 0.20

PING_INTERVAL: float = 20.0      # match Binance server cadence
RECONNECT_BASE: float = 1.0
RECONNECT_MAX: float = 30.0
SECS_PER_YEAR: float = 365.25 * 86400.0


@dataclass
class BinanceTick:
    symbol: str
    bid: float
    ask: float
    mid: float
    sigma_annual: float
    ts: float             # monotonic seconds
    drift_annual: float = 0.0   # OFI-derived annualised drift, range ±OFI_MAX_DRIFT


@dataclass
class _State:
    bid: float = 0.0
    ask: float = 0.0
    last_trade_price: float = 0.0
    last_trade_ts: float = 0.0
    # EWMA variance per second (annualised in snapshot()).
    ewma_var_per_sec: float = 0.0
    n_returns_seen: int = 0
    # Rolling window of (ts, signed_dollar_volume, abs_dollar_volume) for OFI.
    flow: Deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=8192)
    )


class BinanceWS:
    """Subscribe to one symbol's public market data streams."""

    # OFI drift is mildly expensive (deque walk) — cache its result
    # since the underlying flow doesn't move on sub-second timescales.
    _DRIFT_CACHE_TTL: float = 0.25  # 250ms

    def __init__(self, symbol: str, stale_threshold_secs: float = 2.0):
        self.symbol = symbol.lower()
        self.stale_threshold = stale_threshold_secs
        self._state = _State()
        self._ts: float = 0.0
        self._lock = asyncio.Lock()
        # Most recent (computed_at, drift) — caller checks ttl.
        self._drift_cache: tuple[float, float] = (0.0, 0.0)
        # Set on every bookTicker arrival so the orchestrator can
        # wake an event-driven eval loop instead of polling.
        self.tick_event = asyncio.Event()

    def snapshot(self) -> BinanceTick | None:
        """Return latest tick or None if data is stale / not yet received."""
        now = time.monotonic()
        s = self._state
        if s.bid == 0 or (now - self._ts) > self.stale_threshold:
            return None
        mid = (s.bid + s.ask) / 2.0
        sigma = self._compute_sigma()
        drift = self._compute_drift(now)
        return BinanceTick(
            symbol=self.symbol,
            bid=s.bid,
            ask=s.ask,
            mid=mid,
            sigma_annual=sigma,
            drift_annual=drift,
            ts=self._ts,
        )

    def is_stale(self) -> bool:
        return (time.monotonic() - self._ts) > self.stale_threshold

    def _compute_sigma(self) -> float:
        if self._state.n_returns_seen < 5:
            return SIGMA_MIN
        var_annual = self._state.ewma_var_per_sec * SECS_PER_YEAR
        if var_annual <= 0:
            return SIGMA_MIN
        return max(SIGMA_MIN, min(SIGMA_MAX, math.sqrt(var_annual)))

    def _compute_drift(self, now: float) -> float:
        # TTL cache — same OFI signal serves multiple market evaluations
        # within a single eval-loop tick.
        last_at, last_drift = self._drift_cache
        if (now - last_at) < self._DRIFT_CACHE_TTL:
            return last_drift
        cutoff = now - OFI_WINDOW_SECS
        signed = 0.0
        abs_vol = 0.0
        flow = self._state.flow
        while flow and flow[0][0] < cutoff:
            flow.popleft()
        for _ts, signed_dv, abs_dv in flow:
            signed += signed_dv
            abs_vol += abs_dv
        drift = 0.0 if abs_vol <= 0 else max(-1.0, min(1.0, signed / abs_vol)) * OFI_MAX_DRIFT
        self._drift_cache = (now, drift)
        return drift

    async def run(self) -> None:
        """Loop forever, reconnecting with back-off."""
        delay = RECONNECT_BASE
        streams = f"{self.symbol}@bookTicker/{self.symbol}@aggTrade"
        url = f"{WS_BASE}?streams={streams}"
        while True:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=10,
                ) as ws:
                    log.info("Binance WS connected: %s", url)
                    delay = RECONNECT_BASE
                    async for raw in ws:
                        await self._handle(raw)
            except ConnectionClosed as exc:
                log.warning("Binance WS closed (%s), reconnecting in %.1fs", exc, delay)
            except Exception as exc:
                log.error("Binance WS error: %s, reconnecting in %.1fs", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        stream = (msg.get("stream") or "").lower()
        data = msg.get("data", msg)
        now = time.monotonic()

        # bookTicker has NO "e" field — identify by stream or shape.
        if "bookticker" in stream or ("b" in data and "a" in data and "u" in data and "e" not in data):
            try:
                self._state.bid = float(data["b"])
                self._state.ask = float(data["a"])
                self._ts = now
                # Wake the orchestrator. set() is a no-op if already set,
                # so the cost when nothing is awaiting is essentially nil.
                if not self.tick_event.is_set():
                    self.tick_event.set()
            except (KeyError, ValueError, TypeError):
                pass
            return

        if data.get("e") == "aggTrade":
            try:
                price = float(data["p"])
                qty = float(data["q"])
                # Binance: m=true means buyer is the market-maker, i.e. the
                # aggressor was a SELLER (so signed flow is negative).
                buyer_is_maker = bool(data.get("m"))
            except (KeyError, ValueError, TypeError):
                return

            if self._state.last_trade_price > 0 and self._state.last_trade_ts > 0:
                lr = math.log(price / self._state.last_trade_price)
                dt = max(1e-3, now - self._state.last_trade_ts)
                # EWMA on variance-per-second basis. Decay weight scales
                # with elapsed time so irregular trade arrivals are handled
                # correctly: λ_eff = exp(-dt / τ).
                lam = math.exp(-dt / SIGMA_HALF_LIFE_SECS)
                inst_var_per_sec = (lr * lr) / dt
                self._state.ewma_var_per_sec = (
                    lam * self._state.ewma_var_per_sec
                    + (1.0 - lam) * inst_var_per_sec
                )
                self._state.n_returns_seen += 1

            # Track signed dollar flow for OFI drift.
            sign = -1.0 if buyer_is_maker else +1.0
            dv = price * qty
            self._state.flow.append((now, sign * dv, dv))

            self._state.last_trade_price = price
            self._state.last_trade_ts = now
            self._ts = now
