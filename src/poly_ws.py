"""
Polymarket CLOB WebSocket client.

Connects to wss://ws-subscriptions-clob.polymarket.com and subscribes to the
'market' channel for real-time book updates on a set of token_ids.

The 'user' channel (order fills) is subscribed separately and requires
L2 API credentials — it is optional and only opened when credentials
are present.

Maintains an in-memory order book (best bid/ask) per token_id.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
HEARTBEAT_INTERVAL = 10.0
RECONNECT_BASE = 1.0
RECONNECT_MAX = 30.0


@dataclass
class BookLevel:
    price: float
    size: float


@dataclass
class BookSnapshot:
    token_id: str
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    ts: float  # monotonic


class PolyWS:
    """Maintain live best-bid/ask for a set of Polymarket token_ids."""

    def __init__(
        self,
        token_ids: list[str],
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        stale_threshold_secs: float = 5.0,
    ):
        self._token_ids = list(token_ids)
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._stale = stale_threshold_secs
        # token_id → {bid: float, ask: float, bid_size: float, ask_size: float, ts: float}
        self._books: dict[str, dict] = {
            tid: {"bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0, "ts": 0.0}
            for tid in token_ids
        }
        self._fill_callbacks: list = []

    def snapshot(self, token_id: str) -> BookSnapshot | None:
        b = self._books.get(token_id)
        if b is None or b["bid"] == 0.0:
            return None
        if (time.monotonic() - b["ts"]) > self._stale:
            return None
        return BookSnapshot(
            token_id=token_id,
            best_bid=b["bid"],
            best_ask=b["ask"],
            bid_size=b["bid_size"],
            ask_size=b["ask_size"],
            ts=b["ts"],
        )

    def on_fill(self, callback) -> None:
        """Register a callback(token_id, side, price, size) for fill events."""
        self._fill_callbacks.append(callback)

    def update_tokens(self, token_ids: list[str]) -> None:
        """Hot-add tokens without reconnecting (next subscribe loop picks them up)."""
        for tid in token_ids:
            if tid not in self._books:
                self._books[tid] = {"bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0, "ts": 0.0}
        self._token_ids = list(self._books)

    async def run(self) -> None:
        delay = RECONNECT_BASE
        while True:
            try:
                async with websockets.connect(
                    WS_URL, ping_interval=HEARTBEAT_INTERVAL, ping_timeout=15
                ) as ws:
                    log.info("Polymarket WS connected")
                    delay = RECONNECT_BASE
                    await self._subscribe(ws)
                    async for raw in ws:
                        self._handle(raw)
            except ConnectionClosed as exc:
                log.warning("Polymarket WS closed (%s), retry in %.1fs", exc, delay)
            except Exception as exc:
                log.error("Polymarket WS error: %s, retry in %.1fs", exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)

    async def _subscribe(self, ws) -> None:
        sub = {
            "auth": None,
            "type": "subscribe",
            "channel": "market",
            "markets": self._token_ids,
        }
        await ws.send(json.dumps(sub))

        if self._api_key:
            # L2 HMAC auth for user channel (fills)
            # py-clob-client handles header generation; here we embed raw creds
            user_sub = {
                "auth": {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "passphrase": self._api_passphrase,
                },
                "type": "subscribe",
                "channel": "user",
                "markets": self._token_ids,
            }
            await ws.send(json.dumps(user_sub))

    def _handle(self, raw: str) -> None:
        try:
            events = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(events, list):
            events = [events]
        for event in events:
            etype = event.get("event_type") or event.get("type")
            if etype in ("book", "price_change"):
                self._update_book(event)
            elif etype in ("trade", "order_matched"):
                self._dispatch_fill(event)

    def _update_book(self, event: dict) -> None:
        tid = event.get("asset_id") or event.get("market")
        if not tid or tid not in self._books:
            return
        now = time.monotonic()
        b = self._books[tid]

        # Full book snapshot
        if "bids" in event or "asks" in event:
            bids = event.get("bids", [])
            asks = event.get("asks", [])
            if bids:
                best = max(bids, key=lambda x: float(x[0]))
                b["bid"] = float(best[0])
                b["bid_size"] = float(best[1])
            if asks:
                best = min(asks, key=lambda x: float(x[0]))
                b["ask"] = float(best[0])
                b["ask_size"] = float(best[1])

        # Delta / price_change event
        if "price" in event:
            side = event.get("side", "").upper()
            price = float(event["price"])
            size = float(event.get("size", 0))
            if side == "BUY":
                b["bid"] = max(b["bid"], price)
                b["bid_size"] = size
            elif side == "SELL":
                b["ask"] = min(b["ask"], price)
                b["ask_size"] = size

        b["ts"] = now

    def _dispatch_fill(self, event: dict) -> None:
        tid = event.get("asset_id") or event.get("market", "")
        side = event.get("side", "")
        price = float(event.get("price", 0))
        size = float(event.get("size", 0))
        for cb in self._fill_callbacks:
            try:
                cb(tid, side, price, size)
            except Exception as exc:
                log.error("Fill callback error: %s", exc)
