"""
Order execution wrapper.

In paper-trade mode (default) every order is simulated: "filled" at
the requested price and size, PnL tracked in memory and logged.

In live mode, orders are sent via py-clob-client as FAK (Fill-And-Kill)
IOC orders. The CLOB matches on-chain; our fill is immediate or rejected.

Paper vs live is controlled by the PAPER_TRADE env var and the
--live CLI flag. Both paths log identical structured records to SQLite.
"""

import asyncio
import logging
import time
import aiosqlite
from dataclasses import dataclass

from .signal import Signal, Side

log = logging.getLogger(__name__)

DB_PATH = "fills.db"


@dataclass
class Fill:
    ts: float
    market_id: str
    token_id: str
    side: str
    price: float
    size: float
    fee: float
    p_star: float
    edge: float
    paper: bool
    order_id: str = ""
    # F1 enrichment — lets pnl_attribution slice by TTE / maker / source and
    # run a calibration audit (p_star vs realised outcome) without re-fetching.
    expiry_ts: float = 0.0
    tte_at_fill: float = 0.0
    is_maker: bool = False
    source: str = "model"
    question: str = ""


# Column order for the fills table (single source of truth for schema,
# migration, and INSERT).
FILL_COLUMNS = [
    "ts", "market_id", "token_id", "side", "price", "size", "fee",
    "p_star", "edge", "paper", "order_id",
    "expiry_ts", "tte_at_fill", "is_maker", "source", "question",
]


class Executor:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self._clob = None
        self._db: aiosqlite.Connection | None = None
        self._position: dict[str, float] = {}  # token_id → net shares
        self._realised_pnl: float = 0.0

    async def setup(self, private_key: str | None = None) -> None:
        self._db = await aiosqlite.connect(DB_PATH)
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS fills (
                ts REAL, market_id TEXT, token_id TEXT, side TEXT,
                price REAL, size REAL, fee REAL, p_star REAL, edge REAL,
                paper INTEGER, order_id TEXT,
                expiry_ts REAL, tte_at_fill REAL, is_maker INTEGER,
                source TEXT, question TEXT
            )"""
        )
        # Migrate pre-F1 databases: add any column the running schema expects
        # but an older file is missing. ALTER TABLE ADD COLUMN is cheap and
        # idempotent here because we only add what PRAGMA reports as absent.
        cur = await self._db.execute("PRAGMA table_info(fills)")
        existing = {row[1] for row in await cur.fetchall()}
        await cur.close()
        _coltypes = {
            "expiry_ts": "REAL", "tte_at_fill": "REAL", "is_maker": "INTEGER",
            "source": "TEXT", "question": "TEXT",
        }
        for col, typ in _coltypes.items():
            if col not in existing:
                await self._db.execute(f"ALTER TABLE fills ADD COLUMN {col} {typ}")
        await self._db.commit()

        if not self.paper and private_key:
            self._init_clob(private_key)

    def _init_clob(self, private_key: str) -> None:
        try:
            from py_clob_client.client import ClobClient  # type: ignore
            from py_clob_client.clob_types import ApiCreds  # type: ignore
            import os

            host = "https://clob.polymarket.com"
            chain_id = int(os.getenv("CHAIN_ID", "137"))
            # Derive L2 creds from L1 key (one-time on first call)
            client = ClobClient(host, key=private_key, chain_id=chain_id)
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)
            self._clob = client
            log.info("ClobClient initialised (live mode)")
        except ImportError:
            log.error("py-clob-client not installed — cannot run in live mode")
            raise

    async def execute(self, signal: Signal) -> Fill | None:
        from .pricing import taker_fee_per_share, maker_rebate_per_share, FEE_RATE_CRYPTO
        fee_rate = getattr(getattr(signal, "market", None), "fee_rate", FEE_RATE_CRYPTO)
        if getattr(signal, "is_maker", False):
            # Maker/post-only: pay no taker fee and accrue a rebate (negative
            # fee = credit).  Paper mode assumes the resting order fills, which
            # is optimistic on fill probability — see RUNBOOK.
            fee = -maker_rebate_per_share(signal.price) * signal.size
        else:
            fee = taker_fee_per_share(signal.price, fee_rate) * signal.size
        order_id = ""

        if self.paper:
            order_id = f"paper-{int(time.time()*1000)}"
            log.info(
                "[PAPER] %s %s %g shares @ %.4f (fee=%.4f edge=%.4f src=%s)",
                "MAKER" if getattr(signal, "is_maker", False) else "TAKER",
                signal.side.value, signal.size, signal.price, fee, signal.edge,
                getattr(signal, "source", "model"),
            )
        else:
            order_id = await self._send_live(signal)
            if not order_id:
                return None

        market = getattr(signal, "market", None)
        expiry_ts = float(getattr(market, "expiry_ts", 0.0) or 0.0)
        fill = Fill(
            ts=time.time(),
            market_id=signal.market.condition_id,
            token_id=signal.token_id,
            side=signal.side.value,
            price=signal.price,
            size=signal.size,
            fee=fee,
            p_star=signal.p_star,
            edge=signal.edge,
            paper=self.paper,
            order_id=order_id,
            expiry_ts=expiry_ts,
            tte_at_fill=(expiry_ts - time.time()) if expiry_ts else 0.0,
            is_maker=bool(getattr(signal, "is_maker", False)),
            source=getattr(signal, "source", "model"),
            question=str(getattr(market, "question", "") or ""),
        )
        await self._record(fill)
        self._update_position(fill)
        return fill

    async def _send_live(self, signal: Signal) -> str:
        if not self._clob:
            log.error("CLOB client not initialised")
            return ""
        loop = asyncio.get_event_loop()
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore
            args = OrderArgs(
                token_id=signal.token_id,
                price=signal.price,
                size=signal.size,
                side=signal.side.value,
            )
            # Maker → resting GTC (post-only price is non-crossing by
            # construction); taker → FAK (fill-and-kill / IOC).
            otype = OrderType.GTC if getattr(signal, "is_maker", False) else OrderType.FAK
            resp = await loop.run_in_executor(
                None,
                lambda: self._clob.create_and_post_order(args, otype),
            )
            order_id = resp.get("orderID", resp.get("id", ""))
            log.info("[LIVE] order %s sent, response: %s", order_id, resp)
            return str(order_id)
        except Exception as exc:
            log.error("Live order error: %s", exc)
            return ""

    def _update_position(self, fill: Fill) -> None:
        delta = fill.size if fill.side == "BUY" else -fill.size
        self._position[fill.token_id] = self._position.get(fill.token_id, 0.0) + delta

    async def _record(self, fill: Fill) -> None:
        if self._db:
            cols = ", ".join(FILL_COLUMNS)
            ph = ", ".join("?" for _ in FILL_COLUMNS)
            await self._db.execute(
                f"INSERT INTO fills ({cols}) VALUES ({ph})",
                (
                    fill.ts, fill.market_id, fill.token_id, fill.side,
                    fill.price, fill.size, fill.fee, fill.p_star, fill.edge,
                    int(fill.paper), fill.order_id,
                    fill.expiry_ts, fill.tte_at_fill, int(fill.is_maker),
                    fill.source, fill.question,
                ),
            )
            await self._db.commit()

    def positions(self) -> dict[str, float]:
        return dict(self._position)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
