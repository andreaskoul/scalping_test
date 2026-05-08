"""
Polymarket CLOB book poller (REST-based).

Polls https://clob.polymarket.com/book?token_id=<id> every POLL_INTERVAL_SECS
for each tracked token. This is more reliable than the WebSocket endpoint
for development and paper-trading.

Switch to the WS implementation for production co-located deployments
where sub-second latency matters.
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)

CLOB_REST = "https://clob.polymarket.com"
POLL_INTERVAL_SECS: float = 0.5
REQUEST_TIMEOUT: float = 5.0
BATCH_SIZE: int = 100   # max tokens per POST /books request


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

    Named PolyWS to keep the interface identical to the WS version --
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
        """Batch-poll books via POST /books to keep request count low.

        With ~150 tokens, the per-token GET approach hits ~300 req/s and
        triggers IP rate-limits within minutes. The batch endpoint accepts
        a JSON array of {token_id} and returns one book per entry, so a
        full universe refresh costs 2–3 requests instead of 150.
        """
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self._session = session
            log.info("Polymarket REST poller started (interval=%.1fs, batch /books)", POLL_INTERVAL_SECS)
            while True:
                tokens = list(self._token_ids)
                if tokens:
                    chunks = [tokens[i:i + BATCH_SIZE] for i in range(0, len(tokens), BATCH_SIZE)]
                    await asyncio.gather(
                        *[self._fetch_books_batch(session, c) for c in chunks],
                        return_exceptions=True,
                    )
                await asyncio.sleep(POLL_INTERVAL_SECS)

    async def _fetch_books_batch(
        self, session: aiohttp.ClientSession, token_ids: list[str]
    ) -> None:
        body = [{"token_id": tid} for tid in token_ids]
        try:
            async with session.post(f"{CLOB_REST}/books", json=body) as resp:
                if resp.status != 200:
                    log.debug("Batch /books status=%d", resp.status)
                    return
                data = await resp.json()
        except Exception as exc:
            log.debug("Batch /books error: %s", exc)
            return

        now = time.monotonic()
        for entry in data if isinstance(data, list) else []:
            tid = str(entry.get("asset_id") or entry.get("token_id") or "")
            if not tid:
                continue
            bids = entry.get("bids", []) or []
            asks = entry.get("asks", []) or []

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

            if tid not in self._books:
                self._books[tid] = {}
            self._books[tid].update({
                "bid": best_bid,
                "ask": best_ask,
                "bid_size": best_bid_size,
                "ask_size": best_ask_size,
                "ts": now,
            })
