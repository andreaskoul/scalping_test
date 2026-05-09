"""
Polymarket CLOB book client — WebSocket primary, REST batch fallback.

Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market and
subscribes to the public `book` / `price_change` / `tick_size_change`
events for the configured set of asset (token) IDs.

Why WSS over REST:
  - REST polling at 0.5 s leaves us 0–500 ms stale on every fill;
    measured Polymarket book updates are often <100 ms apart, so the
    REST snapshot is mid-flight.
  - The latency-arb edge has been compressed to a few seconds total —
    burning 250 ms per cycle on poll lag is ~5–10 % of the entire
    available alpha.

Reliability:
  - WS reconnects with exponential back-off.
  - On any disconnect, falls back to a one-shot batch REST refresh of
    all tracked tokens so the book state stays usable.
  - Drops books that are silent for `stale_threshold_secs` (caller
    treats as "no data").

Public interface preserved: `update_tokens`, `snapshot`, `run`.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed

log = logging.getLogger(__name__)

CLOB_REST = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Re-subscribe whenever the universe changes; cap subscriptions per
# message to avoid hitting any payload limit (observed: server accepts
# 200+ ids per message but keep some headroom).
SUBSCRIBE_BATCH: int = 100

# REST fallback batch size when WS is down.
REST_BATCH_SIZE: int = 100
REST_TIMEOUT: float = 5.0

PING_INTERVAL: float = 10.0
RECONNECT_BASE: float = 1.0
RECONNECT_MAX: float = 30.0


@dataclass
class BookSnapshot:
    token_id: str
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    ts: float  # monotonic


class PolyWS:
    """Maintain Polymarket CLOB book state for a set of token_ids.

    Same surface as the previous REST poller — `update_tokens` accepts a
    list, `snapshot(token_id)` returns a BookSnapshot or None.
    """

    def __init__(
        self,
        token_ids: list[str],
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        stale_threshold_secs: float = 5.0,
    ):
        self._token_ids: set[str] = set(token_ids)
        self._stale = stale_threshold_secs
        # token_id → {bids: {price: size}, asks: {price: size}, ts}
        self._books: dict[str, dict] = {}
        self._fill_callbacks: list = []
        self._ws = None
        self._subscribed: set[str] = set()
        self._sub_lock = asyncio.Lock()

    # ---------- public surface ----------

    def snapshot(self, token_id: str) -> BookSnapshot | None:
        b = self._books.get(token_id)
        if b is None:
            return None
        ts = b.get("ts", 0.0)
        if ts <= 0 or (time.monotonic() - ts) > self._stale:
            return None
        bid, bsize = self._best(b.get("bids", {}), best="max")
        ask, asize = self._best(b.get("asks", {}), best="min")
        if bid <= 0:
            return None
        return BookSnapshot(
            token_id=token_id,
            best_bid=bid,
            best_ask=ask,
            bid_size=bsize,
            ask_size=asize,
            ts=ts,
        )

    def on_fill(self, callback) -> None:
        self._fill_callbacks.append(callback)

    def update_tokens(self, token_ids: list[str]) -> None:
        """Add new tokens to the tracked set. The run loop picks up the
        delta on its next iteration and sends a subscribe message."""
        new = set(token_ids) - self._token_ids
        self._token_ids.update(token_ids)
        for tid in new:
            self._books.setdefault(tid, {"bids": {}, "asks": {}, "ts": 0.0})

    async def run(self) -> None:
        """Connect, subscribe, dispatch events; reconnect on failure."""
        delay = RECONNECT_BASE
        while True:
            try:
                async with websockets.connect(
                    CLOB_WS,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=15,
                ) as ws:
                    self._ws = ws
                    log.info("Polymarket WS connected: %s", CLOB_WS)
                    delay = RECONNECT_BASE
                    self._subscribed.clear()
                    # First subscribe + a periodic resubscribe loop so
                    # newly-added tokens get hooked up.
                    sub_task = asyncio.create_task(self._subscriber_loop())
                    try:
                        async for raw in ws:
                            self._handle(raw)
                    finally:
                        sub_task.cancel()
                        self._ws = None
            except ConnectionClosed as exc:
                log.warning("Polymarket WS closed (%s), reconnecting in %.1fs", exc, delay)
            except Exception as exc:
                log.error("Polymarket WS error: %s, reconnecting in %.1fs", exc, delay)

            # Bridge the gap with a REST refresh while we're disconnected.
            try:
                await self._rest_refresh_all()
            except Exception as exc:
                log.debug("REST fallback refresh failed: %s", exc)

            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX)

    # ---------- internals ----------

    @staticmethod
    def _best(side: dict, best: str = "max") -> tuple[float, float]:
        if not side:
            return (0.0, 0.0) if best == "max" else (1.0, 0.0)
        if best == "max":
            price = max(side.keys())
        else:
            price = min(side.keys())
        size = side.get(price, 0.0)
        if size <= 0:
            # Stale level — fall back gracefully.
            return (0.0, 0.0) if best == "max" else (1.0, 0.0)
        return price, size

    async def _subscriber_loop(self) -> None:
        """Send subscribe messages whenever the tracked set grows."""
        while True:
            try:
                async with self._sub_lock:
                    pending = sorted(self._token_ids - self._subscribed)
                if pending and self._ws is not None:
                    for i in range(0, len(pending), SUBSCRIBE_BATCH):
                        chunk = pending[i:i + SUBSCRIBE_BATCH]
                        msg = {"type": "market", "assets_ids": chunk}
                        await self._ws.send(json.dumps(msg))
                        self._subscribed.update(chunk)
                        log.debug("Polymarket WS: subscribed %d new tokens", len(chunk))
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug("Subscriber loop error: %s", exc)
                await asyncio.sleep(1.0)

    def _handle(self, raw) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        # Server sends single events or a list of them.
        events = msg if isinstance(msg, list) else [msg]
        now = time.monotonic()
        for ev in events:
            etype = (ev.get("event_type") or ev.get("type") or "").lower()
            tid = str(ev.get("asset_id") or ev.get("token_id") or "")
            if not tid:
                continue
            book = self._books.setdefault(tid, {"bids": {}, "asks": {}, "ts": 0.0})

            if etype == "book":
                # Full snapshot — replace levels.
                book["bids"] = {
                    float(b["price"]): float(b["size"])
                    for b in (ev.get("bids") or [])
                    if float(b.get("size", 0)) > 0
                }
                book["asks"] = {
                    float(a["price"]): float(a["size"])
                    for a in (ev.get("asks") or [])
                    if float(a.get("size", 0)) > 0
                }
                book["ts"] = now
            elif etype == "price_change":
                # Incremental update — apply per-level deltas.
                for ch in ev.get("changes") or []:
                    try:
                        price = float(ch["price"])
                        size = float(ch["size"])
                        side = (ch.get("side") or "").upper()
                    except (KeyError, ValueError, TypeError):
                        continue
                    levels = book["bids"] if side in ("BUY", "BID") else book["asks"]
                    if size <= 0:
                        levels.pop(price, None)
                    else:
                        levels[price] = size
                book["ts"] = now
            elif etype == "tick_size_change":
                # No book impact for our purposes.
                book["ts"] = now

    # ---------- REST fallback ----------

    async def _rest_refresh_all(self) -> None:
        if not self._token_ids:
            return
        timeout = aiohttp.ClientTimeout(total=REST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tokens = list(self._token_ids)
            for i in range(0, len(tokens), REST_BATCH_SIZE):
                chunk = tokens[i:i + REST_BATCH_SIZE]
                body = [{"token_id": t} for t in chunk]
                try:
                    async with session.post(f"{CLOB_REST}/books", json=body) as r:
                        if r.status != 200:
                            continue
                        data = await r.json()
                except Exception:
                    continue
                now = time.monotonic()
                for entry in data if isinstance(data, list) else []:
                    tid = str(entry.get("asset_id") or entry.get("token_id") or "")
                    if not tid:
                        continue
                    bids = {
                        float(b["price"]): float(b["size"])
                        for b in (entry.get("bids") or [])
                        if float(b.get("size", 0)) > 0
                    }
                    asks = {
                        float(a["price"]): float(a["size"])
                        for a in (entry.get("asks") or [])
                        if float(a.get("size", 0)) > 0
                    }
                    self._books[tid] = {"bids": bids, "asks": asks, "ts": now}
