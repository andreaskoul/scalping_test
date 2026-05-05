"""
Binance public WebSocket client.

Subscribes to <symbol>@bookTicker and <symbol>@aggTrade streams.
Maintains:
  - best bid/ask → fair mid-price
  - rolling log-returns over the last VOL_WINDOW_SECS seconds → σ_annual
  - monotonic timestamp of last received tick

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

from .pricing import realized_vol_annual, SIGMA_MIN

log = logging.getLogger(__name__)

WS_BASE = "wss://stream.binance.com:443/stream"
VOL_WINDOW_SECS: float = 60.0   # rolling window for realized σ
PING_INTERVAL: float = 20.0      # match Binance server cadence
RECONNECT_BASE: float = 1.0
RECONNECT_MAX: float = 30.0


@dataclass
class BinanceTick:
    symbol: str
    bid: float
    ask: float
    mid: float
    sigma_annual: float
    ts: float  # monotonic seconds


@dataclass
class _State:
    bid: float = 0.0
    ask: float = 0.0
    last_trade_price: float = 0.0
    last_trade_ts: float = 0.0
    log_returns: Deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=4096)
    )  # (timestamp, log_return)


class BinanceWS:
    """Subscribe to one symbol's public market data streams."""

    def __init__(self, symbol: str, stale_threshold_secs: float = 2.0):
        self.symbol = symbol.lower()
        self.stale_threshold = stale_threshold_secs
        self._state = _State()
        self._ts: float = 0.0
        self._lock = asyncio.Lock()

    def snapshot(self) -> BinanceTick | None:
        """Return latest tick or None if data is stale / not yet received."""
        now = time.monotonic()
        s = self._state
        if s.bid == 0 or (now - self._ts) > self.stale_threshold:
            return None
        mid = (s.bid + s.ask) / 2.0
        sigma = self._compute_sigma()
        return BinanceTick(
            symbol=self.symbol,
            bid=s.bid,
            ask=s.ask,
            mid=mid,
            sigma_annual=sigma,
            ts=self._ts,
        )

    def is_stale(self) -> bool:
        return (time.monotonic() - self._ts) > self.stale_threshold

    def _compute_sigma(self) -> float:
        cutoff = time.monotonic() - VOL_WINDOW_SECS
        recent = [r for ts, r in self._state.log_returns if ts >= cutoff]
        if len(recent) < 2:
            return SIGMA_MIN
        return realized_vol_annual(recent, VOL_WINDOW_SECS)

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
        data = msg.get("data", msg)
        event = data.get("e")
        now = time.monotonic()

        if event == "bookTicker":
            self._state.bid = float(data["b"])
            self._state.ask = float(data["a"])
            self._ts = now

        elif event == "aggTrade":
            price = float(data["p"])
            if self._state.last_trade_price > 0:
                lr = math.log(price / self._state.last_trade_price)
                self._state.log_returns.append((now, lr))
            self._state.last_trade_price = price
            self._state.last_trade_ts = now
            self._ts = now
