"""Decision telemetry for paper/live validation.

The recorder is intentionally best-effort: it uses a bounded queue and drops
records rather than blocking the trading hot path.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiosqlite

from .storage import default_db_path, ensure_parent, env_db_path


DECISION_COLUMNS = [
    "decision_id", "ts_wall", "ts_mono", "run_id", "stage",
    "market_id", "question", "token_id", "symbol", "strike", "expiry_ts", "tte",
    "is_updown", "is_threshold",
    "spot_bid", "spot_ask", "spot_mid", "sigma", "iv", "carry", "drift", "ofi",
    "poly_bid", "poly_ask", "poly_bid_sz", "poly_ask_sz", "book_age",
    "p_fair", "p_star", "wedge", "edge_buy", "edge_sell", "chosen_side",
    "chosen_price", "is_maker", "maker_price", "size",
    "signal", "reject_reason", "sampled",
    "resolution", "edge_at_5s", "edge_at_30s", "executable", "realized_pnl",
]


@dataclass
class DecisionTrace:
    decision_id: str = ""
    ts_wall: float = 0.0
    ts_mono: float = 0.0
    run_id: str = ""
    stage: str = "eval"

    market_id: str = ""
    question: str = ""
    token_id: str = ""
    symbol: str = ""
    strike: float = 0.0
    expiry_ts: float = 0.0
    tte: float = 0.0
    is_updown: int = 0
    is_threshold: int = 0

    spot_bid: float = 0.0
    spot_ask: float = 0.0
    spot_mid: float = 0.0
    sigma: float = 0.0
    iv: float = 0.0
    carry: float = 0.0
    drift: float = 0.0
    ofi: float = 0.0

    poly_bid: float = 0.0
    poly_ask: float = 0.0
    poly_bid_sz: float = 0.0
    poly_ask_sz: float = 0.0
    book_age: float = 0.0

    p_fair: float = 0.0
    p_star: float = 0.0
    wedge: float = 0.0
    edge_buy: float = 0.0
    edge_sell: float = 0.0
    chosen_side: str = ""
    chosen_price: float = 0.0
    is_maker: int = 0
    maker_price: float = 0.0
    size: float = 0.0

    signal: int = 0
    reject_reason: str = ""
    sampled: int = 0

    resolution: float | None = None
    edge_at_5s: float | None = None
    edge_at_30s: float | None = None
    executable: int | None = None
    realized_pnl: float | None = None

    @classmethod
    def new(cls, run_id: str, stage: str = "eval") -> "DecisionTrace":
        return cls(
            decision_id=str(uuid.uuid4()),
            ts_wall=time.time(),
            ts_mono=time.monotonic(),
            run_id=run_id,
            stage=stage,
        )

    def mark_reject(self, reason: str) -> None:
        self.signal = 0
        self.reject_reason = reason

    def mark_signal(self, side: str, is_maker: bool, price: float, size: float) -> None:
        self.signal = 1
        self.reject_reason = ""
        self.chosen_side = side
        self.chosen_price = price
        self.is_maker = int(is_maker)
        self.maker_price = price if is_maker else 0.0
        self.size = size

    def row(self) -> tuple[Any, ...]:
        data = asdict(self)
        return tuple(data.get(col) for col in DECISION_COLUMNS)


class Recorder:
    def __init__(
        self,
        db_path: str = default_db_path("decisions.db"),
        reject_sample: float = 0.02,
        queue_size: int = 10000,
        batch_size: int = 250,
        run_id: str | None = None,
        poly_events_dir: str = "poly_events",
        poly_events_enabled: bool = True,
        poly_event_queue_size: int = 20000,
        seed: int | None = None,
    ):
        self.run_id = run_id or f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.db_path = db_path
        self.reject_sample = max(0.0, min(1.0, reject_sample))
        self.queue: asyncio.Queue[DecisionTrace | None] = asyncio.Queue(maxsize=queue_size)
        self.poly_events_enabled = poly_events_enabled
        self.poly_events_dir = Path(poly_events_dir) / self.run_id
        self.poly_queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=poly_event_queue_size)
        self.batch_size = batch_size
        self.dropped = 0
        self.poly_dropped = 0
        self.recorded = 0
        self.poly_recorded = 0
        self._rng = random.Random(seed)
        self._db: aiosqlite.Connection | None = None
        self._task: asyncio.Task | None = None
        self._poly_task: asyncio.Task | None = None

    @classmethod
    def from_env(cls) -> "Recorder | None":
        if os.getenv("TELEMETRY_ENABLED", "1").strip().lower() in ("0", "false", "no", "off"):
            return None
        return cls(
            db_path=env_db_path("TELEMETRY_DB_PATH", "decisions.db"),
            reject_sample=float(os.getenv("TELEMETRY_REJECT_SAMPLE", "0.02")),
            queue_size=int(float(os.getenv("TELEMETRY_QUEUE_SIZE", "10000"))),
            batch_size=int(float(os.getenv("TELEMETRY_BATCH_SIZE", "250"))),
            run_id=os.getenv("RUN_ID") or None,
            poly_events_dir=os.getenv("POLY_EVENTS_DIR", "poly_events"),
            poly_events_enabled=os.getenv("POLY_EVENTS_ENABLED", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            poly_event_queue_size=int(float(os.getenv("POLY_EVENT_QUEUE_SIZE", "20000"))),
        )

    async def start(self) -> None:
        ensure_parent(self.db_path)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute(_create_table_sql())
        await self._migrate_decisions()
        await self._db.commit()
        self._task = asyncio.create_task(self._writer(), name="telemetry-writer")
        if self.poly_events_enabled:
            self.poly_events_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "run_id": self.run_id,
                "created_ts": time.time(),
                "format": "jsonl.gz",
                "event": "top_of_book_on_move; full levels on periodic snapshots",
            }
            (self.poly_events_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            self._poly_task = asyncio.create_task(self._poly_writer(), name="poly-event-writer")

    def should_record(self, trace: DecisionTrace) -> bool:
        if trace.signal:
            return True
        if trace.stage in ("risk", "exec"):
            return True
        return self._rng.random() < self.reject_sample

    def record(self, trace: DecisionTrace) -> None:
        if not self.should_record(trace):
            return
        trace.sampled = 1
        try:
            self.queue.put_nowait(trace)
        except asyncio.QueueFull:
            self.dropped += 1

    def record_poly_event(self, event: dict) -> None:
        if not self.poly_events_enabled:
            return
        event.setdefault("run_id", self.run_id)
        try:
            self.poly_queue.put_nowait(event)
        except asyncio.QueueFull:
            self.poly_dropped += 1

    async def close(self) -> None:
        if self._task is not None:
            try:
                self.queue.put_nowait(None)
            except asyncio.QueueFull:
                await self.queue.put(None)
            await self._task
        if self._poly_task is not None:
            try:
                self.poly_queue.put_nowait(None)
            except asyncio.QueueFull:
                await self.poly_queue.put(None)
            await self._poly_task
        if self._db is not None:
            await self._db.close()

    async def _writer(self) -> None:
        assert self._db is not None
        batch: list[DecisionTrace] = []
        while True:
            item = await self.queue.get()
            if item is None:
                break
            batch.append(item)
            if len(batch) >= self.batch_size:
                await self._flush(batch)
                batch.clear()
        if batch:
            await self._flush(batch)

    async def _flush(self, batch: list[DecisionTrace]) -> None:
        assert self._db is not None
        cols = ", ".join(DECISION_COLUMNS)
        ph = ", ".join("?" for _ in DECISION_COLUMNS)
        await self._db.executemany(
            f"INSERT OR REPLACE INTO decisions ({cols}) VALUES ({ph})",
            [trace.row() for trace in batch],
        )
        await self._db.commit()
        self.recorded += len(batch)

    async def _migrate_decisions(self) -> None:
        assert self._db is not None
        cur = await self._db.execute("PRAGMA table_info(decisions)")
        existing = {row[1] for row in await cur.fetchall()}
        await cur.close()
        coltypes = {
            "chosen_price": "REAL",
            "resolution": "REAL",
            "edge_at_5s": "REAL",
            "edge_at_30s": "REAL",
            "executable": "INTEGER",
            "realized_pnl": "REAL",
        }
        for col, typ in coltypes.items():
            if col not in existing:
                await self._db.execute(f"ALTER TABLE decisions ADD COLUMN {col} {typ}")

    async def _poly_writer(self) -> None:
        current_hour = ""
        handle = None
        try:
            while True:
                item = await self.poly_queue.get()
                if item is None:
                    break
                hour = time.strftime("%Y%m%dT%H", time.gmtime(float(item.get("ts_wall", time.time()))))
                if hour != current_hour:
                    if handle is not None:
                        handle.close()
                    current_hour = hour
                    path = self.poly_events_dir / f"{hour}.jsonl.gz"
                    handle = gzip.open(path, "at", encoding="utf-8")
                assert handle is not None
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
                self.poly_recorded += 1
        finally:
            if handle is not None:
                handle.close()


def _create_table_sql() -> str:
    return """
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id TEXT PRIMARY KEY,
        ts_wall REAL,
        ts_mono REAL,
        run_id TEXT,
        stage TEXT,
        market_id TEXT,
        question TEXT,
        token_id TEXT,
        symbol TEXT,
        strike REAL,
        expiry_ts REAL,
        tte REAL,
        is_updown INTEGER,
        is_threshold INTEGER,
        spot_bid REAL,
        spot_ask REAL,
        spot_mid REAL,
        sigma REAL,
        iv REAL,
        carry REAL,
        drift REAL,
        ofi REAL,
        poly_bid REAL,
        poly_ask REAL,
        poly_bid_sz REAL,
        poly_ask_sz REAL,
        book_age REAL,
        p_fair REAL,
        p_star REAL,
        wedge REAL,
        edge_buy REAL,
        edge_sell REAL,
        chosen_side TEXT,
        chosen_price REAL,
        is_maker INTEGER,
        maker_price REAL,
        size REAL,
        signal INTEGER,
        reject_reason TEXT,
        sampled INTEGER,
        resolution REAL,
        edge_at_5s REAL,
        edge_at_30s REAL,
        executable INTEGER,
        realized_pnl REAL
    )
    """