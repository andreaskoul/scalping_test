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


def _empty_book() -> dict:
    return {
        "bids": {}, "asks": {},
        "best_bid": 0.0, "best_bid_size": 0.0,
        "best_ask": 1.0, "best_ask_size": 0.0,
        "ts": 0.0,
    }


def _refresh_top(book: dict) -> bool:
    """Recompute and store top-of-book from the level dicts.

    Returns True iff the top-of-book moved (price or size changed on
    either side) — the caller uses this to decide whether to mark the
    token dirty for re-evaluation.
    """
    bids = book["bids"]
    asks = book["asks"]
    if bids:
        bp = max(bids)
        bs = bids[bp]
    else:
        bp, bs = 0.0, 0.0
    if asks:
        ap = min(asks)
        asz = asks[ap]
    else:
        ap, asz = 1.0, 0.0
    moved = (
        bp != book["best_bid"] or bs != book["best_bid_size"]
        or ap != book["best_ask"] or asz != book["best_ask_size"]
    )
    book["best_bid"] = bp
    book["best_bid_size"] = bs
    book["best_ask"] = ap
    book["best_ask_size"] = asz
    return moved


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
        event_sink=None,
        snapshot_interval_secs: float = 10.0,
    ):
        self._token_ids: set[str] = set(token_ids)
        self._stale = stale_threshold_secs
        # token_id → {
        #   bids: {price: size},  asks: {price: size},
        #   best_bid, best_bid_size, best_ask, best_ask_size,  # cached
        #   ts,
        # }
        # The cached top-of-book is updated on every event so `snapshot()`
        # is O(1). Without it, snapshot would do max/min across the whole
        # level dict — at ~3000 calls/s on liquid universes that was the
        # single largest CPU hot spot.
        self._books: dict[str, dict] = {}
        self._fill_callbacks: list = []
        self._event_sink = event_sink
        self._snapshot_interval = max(0.0, snapshot_interval_secs)
        self._last_event_snapshot: dict[str, float] = {}
        self._ws = None
        self._subscribed: set[str] = set()
        self._sub_lock = asyncio.Lock()
        # Set whenever ANY tracked book changes — the orchestrator can
        # await this to drive event-driven evaluation instead of polling.
        self._book_change = asyncio.Event()
        # Tokens whose best-of-book changed since last drain — lets the
        # orchestrator evaluate only the markets that could have moved.
        self._dirty_tokens: set[str] = set()

    # ---------- public surface ----------

    def snapshot(self, token_id: str) -> BookSnapshot | None:
        """O(1): reads the cached best-of-book updated by the WS handler."""
        b = self._books.get(token_id)
        if b is None:
            return None
        ts = b["ts"]
        if ts <= 0 or (time.monotonic() - ts) > self._stale:
            return None
        bid = b["best_bid"]
        if bid <= 0:
            return None
        return BookSnapshot(
            token_id=token_id,
            best_bid=bid,
            best_ask=b["best_ask"],
            bid_size=b["best_bid_size"],
            ask_size=b["best_ask_size"],
            ts=ts,
        )

    def walk_book(self, token_id: str, side: str, target_size: float) -> tuple[float, float]:
        """Volume-weighted average price for crossing `target_size` shares.

        Returns (vwap, total_available).
          - side="BUY": we cross the asks → consume from cheapest ask up.
          - side="SELL": we hit the bids → consume from highest bid down.

        If the book has less than target_size, returns (vwap_of_what_exists,
        total_available). Caller decides whether the partial fill is
        acceptable. If the book is empty on the chosen side, returns (0, 0).

        Used by the signal generator to compute the *effective* execution
        price when our desired size exceeds the depth at the top of book.
        Without this, the bot assumes the entire order fills at the
        displayed best price — a known-wrong assumption that can hide a
        few percent of slippage on thin books.
        """
        b = self._books.get(token_id)
        if b is None:
            return 0.0, 0.0
        if side.upper() == "BUY":
            levels = sorted(b["asks"].items())  # ascending price
        else:
            levels = sorted(b["bids"].items(), reverse=True)  # descending price
        remaining = target_size
        cost = 0.0
        filled = 0.0
        for price, size in levels:
            if remaining <= 0:
                break
            take = min(size, remaining)
            cost += price * take
            filled += take
            remaining -= take
        if filled <= 0:
            return 0.0, 0.0
        return cost / filled, filled

    def drain_dirty(self) -> set[str]:
        """Return tokens whose top-of-book changed since the last drain.

        The orchestrator calls this each loop iteration to know exactly
        which markets to re-evaluate. Far cheaper than scanning every
        market every tick.
        """
        if not self._dirty_tokens:
            return set()
        d = self._dirty_tokens
        self._dirty_tokens = set()
        self._book_change.clear()
        return d

    async def wait_change(self, timeout: float | None = None) -> bool:
        """Block until at least one tracked book changes."""
        if self._dirty_tokens:
            return True
        try:
            if timeout is None:
                await self._book_change.wait()
                return True
            await asyncio.wait_for(self._book_change.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def on_fill(self, callback) -> None:
        self._fill_callbacks.append(callback)

    def update_tokens(self, token_ids: list[str]) -> None:
        """Add new tokens to the tracked set. The run loop picks up the
        delta on its next iteration and sends a subscribe message."""
        new = set(token_ids) - self._token_ids
        self._token_ids.update(token_ids)
        for tid in new:
            self._books.setdefault(tid, _empty_book())

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

    def _mark_dirty(self, token_id: str) -> None:
        self._dirty_tokens.add(token_id)
        if not self._book_change.is_set():
            self._book_change.set()

    def _emit_event(self, token_id: str, etype: str, book: dict, now_mono: float, force_levels: bool = False) -> None:
        if self._event_sink is None:
            return
        include_levels = force_levels
        if self._snapshot_interval > 0:
            last = self._last_event_snapshot.get(token_id, 0.0)
            if now_mono - last >= self._snapshot_interval:
                include_levels = True
                self._last_event_snapshot[token_id] = now_mono
        event = {
            "ts_mono": now_mono,
            "ts_wall": time.time(),
            "token_id": token_id,
            "etype": etype,
            "best_bid": book["best_bid"],
            "best_ask": book["best_ask"],
            "bid_sz": book["best_bid_size"],
            "ask_sz": book["best_ask_size"],
        }
        if include_levels:
            event["levels"] = {
                "bids": [[p, s] for p, s in sorted(book["bids"].items(), reverse=True)],
                "asks": [[p, s] for p, s in sorted(book["asks"].items())],
            }
        try:
            self._event_sink(event)
        except Exception as exc:
            log.debug("Poly event sink failed: %s", exc)

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
            book = self._books.get(tid)
            if book is None:
                book = _empty_book()
                self._books[tid] = book

            if etype == "book":
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
                if _refresh_top(book):
                    self._mark_dirty(tid)
                    self._emit_event(tid, "book", book, now, force_levels=True)
            elif etype == "price_change":
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
                if _refresh_top(book):
                    self._mark_dirty(tid)
                    self._emit_event(tid, "price_change", book, now)
            elif etype == "tick_size_change":
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
                    book = self._books.get(tid) or _empty_book()
                    book["bids"] = {
                        float(b["price"]): float(b["size"])
                        for b in (entry.get("bids") or [])
                        if float(b.get("size", 0)) > 0
                    }
                    book["asks"] = {
                        float(a["price"]): float(a["size"])
                        for a in (entry.get("asks") or [])
                        if float(a.get("size", 0)) > 0
                    }
                    book["ts"] = now
                    self._books[tid] = book
                    if _refresh_top(book):
                        self._mark_dirty(tid)
