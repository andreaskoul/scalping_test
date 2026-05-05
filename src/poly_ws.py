"""
Polymarket CLOB book poller (REST-based).

<<<<<<< HEAD
Connects to wss://ws-subscriptions-clob.polymarket.com and subscribes to the
'market' channel for real-time book updates on a set of token_ids.
=======
Polls https://clob.polymarket.com/book?token_id=<id> every POLL_INTERVAL_SECS
for each tracked token. This is more reliable than the WebSocket endpoint
for development and paper-trading.
>>>>>>> 4f5c3882a968876bb5cb376913e17c4916a47ecf

Switch to the WS implementation for production co-located deployments
where sub-second latency matters.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

<<<<<<< HEAD
WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
HEARTBEAT_INTERVAL = 10.0
RECONNECT_BASE = 1.0
RECONNECT_MAX = 30.0


@dataclass
class BookLevel:
    price: float
    size: float
=======
CLOB_REST = "https://clob.polymarket.com"
POLL_INTERVAL_SECS: float = 0.5
REQUEST_TIMEOUT: float = 3.0
>>>>>>> 4f5c3882a968876bb5cb376913e17c4916a47ecf


@dataclass
class BookSnapshot:
    token_id: str
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    ts: float  # monotonic


class PolyWS:
    """Polls Polymarket CLOB REST book endpoint for a set of token_ids.

    Named PolyWS to keep the interface identical to the WS version —
    drop-in replacement, no changes needed in signal.py or main.py.
    """

    def __init__(
        self,
        token_ids: list[str],
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        stale_threshold_secs: float = 5.0,
    ):
        self._token_ids: list[str] = list(token_ids)
        self._stale = stale_threshold_secs
        self._books: dict[str, dict] = {}
        self._fill_callbacks: list = []
        self._session: aiohttp.ClientSession | None = None

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
        self._fill_callbacks.append(callback)

    def update_tokens(self, token_ids: list[str]) -> None:
        for tid in token_ids:
            if tid not in self._books:
                self._books[tid] = {"bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0, "ts": 0.0}
        self._token_ids = list(self._books)

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session
            log.info("Polymarket REST poller started (interval=%.1fs)", POLL_INTERVAL_SECS)
            while True:
                tokens = list(self._token_ids)
                tasks = [self._fetch_book(session, tid) for tid in tokens]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(POLL_INTERVAL_SECS)

    async def _fetch_book(self, session: aiohttp.ClientSession, token_id: str) -> None:
        url = f"{CLOB_REST}/book"
        try:
            async with session.get(url, params={"token_id": token_id}) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except Exception as exc:
            log.debug("Book fetch error for %s: %s", token_id[:12], exc)
            return

        now = time.monotonic()
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        best_bid = best_bid_size = 0.0
        best_ask = best_ask_size = 1.0

        if bids:
            top = max(bids, key=lambda x: float(x.get("price", 0)))
            best_bid = float(top.get("price", 0))
            best_bid_size = float(top.get("size", 0))

        if asks:
            top = min(asks, key=lambda x: float(x.get("price", 1)))
            best_ask = float(top.get("price", 1))
            best_ask_size = float(top.get("size", 0))

        if token_id not in self._books:
            self._books[token_id] = {}
        self._books[token_id].update({
            "bid": best_bid,
            "ask": best_ask,
            "bid_size": best_bid_size,
            "ask_size": best_ask_size,
            "ts": now,
        })
