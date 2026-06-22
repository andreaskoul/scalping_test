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
import random
import time
import aiosqlite
from dataclasses import dataclass, field, replace

from .signal import Signal, Side
from .storage import ensure_parent, env_db_path

log = logging.getLogger(__name__)

DB_PATH = env_db_path("FILL_DB_PATH", "fills.db")


@dataclass
class OrderLifecycle:
    """Tracks resting maker orders so GTD/GTC orders can be expired/cancelled.

    Pure bookkeeping (no I/O): the Executor registers a maker order with its
    expiry and calls `due()` to find ones to cancel. Live mode then issues the
    cancel; paper mode just drops them. Keeps order management out of the hot
    path and unit-testable."""
    _open: dict[str, tuple[str, float]] = field(default_factory=dict)  # id → (token, expiry_ts)

    def register(self, order_id: str, token_id: str, expiry_ts: float) -> None:
        if order_id:
            self._open[order_id] = (token_id, expiry_ts)

    def due(self, now: float) -> list[str]:
        return [oid for oid, (_t, exp) in self._open.items() if exp <= now]

    def drop(self, order_id: str) -> None:
        self._open.pop(order_id, None)

    def open_count(self) -> int:
        return len(self._open)


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
    def __init__(
        self,
        paper: bool = True,
        maker_fill_prob: float = 1.0,
        maker_gtd_secs: float = 12.0,
        seed: int | None = None,
    ):
        self.paper = paper
        self.maker_fill_prob = maker_fill_prob   # paper-mode realism for resting orders
        self.maker_gtd_secs = maker_gtd_secs
        self._fill_rng = random.Random(seed)
        self.lifecycle = OrderLifecycle()
        self._clob = None
        self._db: aiosqlite.Connection | None = None
        self._position: dict[str, float] = {}  # token_id → net shares
        self._realised_pnl: float = 0.0

    async def setup(self, private_key: str | None = None) -> None:
        ensure_parent(DB_PATH)
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

        # Paper-mode maker realism: a resting post-only order is only sometimes
        # hit. Skipping the fill models the order resting unfilled (the caller
        # may retry next tick). Live fills are decided by the exchange, not here.
        if self.paper and getattr(signal, "is_maker", False) and self.maker_fill_prob < 1.0:
            if self._fill_rng.random() > self.maker_fill_prob:
                log.debug("[PAPER] maker order rested unfilled (p=%.2f)", self.maker_fill_prob)
                return None

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
        if not self.paper and fill.is_maker and order_id:
            self.lifecycle.register(order_id, fill.token_id, time.time() + self.maker_gtd_secs)
        return fill

    async def execute_atomic(self, signals: list[Signal]) -> list[Fill] | None:
        """Execute a multi-leg combo as a unit; on a failed leg, flatten the
        already-filled legs (orphan protection) and return None.

        Paper mode fills every leg; live mode can partially fill, so the flatten
        path matters there. Returns the list of fills on success."""
        done: list[tuple[Signal, Fill]] = []
        for s in signals:
            f = await self.execute(s)
            if f is None:
                if done:
                    log.warning("Combo leg failed after %d fills — flattening orphans.", len(done))
                    await self._flatten(done)
                return None
            done.append((s, f))
        return [f for _, f in done]

    async def _flatten(self, done: list[tuple[Signal, Fill]]) -> None:
        for s, _f in done:
            opp = replace(
                s,
                side=Side.SELL if s.side == Side.BUY else Side.BUY,
                is_maker=False, source="flatten",
            )
            try:
                await self.execute(opp)
            except Exception as exc:
                log.error("Flatten leg failed: %s", exc)

    async def expire_maker_orders(self, now: float | None = None) -> int:
        """Cancel resting maker orders past their GTD. Returns count cancelled."""
        now = now if now is not None else time.time()
        due = self.lifecycle.due(now)
        for oid in due:
            if not self.paper and self._clob is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda o=oid: self._clob.cancel(o))
                except Exception as exc:
                    log.debug("Cancel %s failed: %s", oid, exc)
            self.lifecycle.drop(oid)
        return len(due)

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
