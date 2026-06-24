"""Shadow-maker canary schema and reporting.

The canary is the execution-truth tier: tiny live post-only probes used to
measure fill rate and adverse selection. This module intentionally starts with
schema/reporting primitives; order placement is gated by future live-safety work.

`plan_probe` is a *shadow* planner: it records what a tiny post-only probe WOULD
be, and (given recorded book replay) attaches a queue/fill label and signed
markouts — without ever touching a live venue. `CANARY_ENABLED` stays 0 by
default and no order placement exists in this module.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import uuid

from . import replay
from .storage import default_db_path, ensure_parent


# Default tiny probe notional ($). Kept at $1-2 so a planned/live probe can
# never matter to PnL — it only measures fill rate + adverse selection.
DEFAULT_PROBE_SIZE_USD: float = 1.0
DEFAULT_PROBE_GTD_SECS: float = 30.0


CANARY_COLUMNS = [
    "order_id", "ts_post", "ts_event", "token_id", "side", "price", "size",
    "filled", "ts_fill", "queue_ahead_est", "mark_5s", "mark_30s",
    "resolution", "adverse_bps", "rebate_earned", "planned",
]


def ensure_schema(db_path: str = default_db_path("canary.db")) -> None:
    ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS canary (
                order_id TEXT PRIMARY KEY,
                ts_post REAL,
                ts_event REAL,
                token_id TEXT,
                side TEXT,
                price REAL,
                size REAL,
                filled INTEGER,
                ts_fill REAL,
                queue_ahead_est REAL,
                mark_5s REAL,
                mark_30s REAL,
                resolution REAL,
                adverse_bps REAL,
                rebate_earned REAL,
                planned INTEGER
            )
            """
        )
        # Migrate older canary DBs that predate the `planned` column.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(canary)")}
        if "planned" not in existing:
            conn.execute("ALTER TABLE canary ADD COLUMN planned INTEGER")
        conn.commit()
    finally:
        conn.close()


def canary_enabled() -> bool:
    return os.getenv("CANARY_ENABLED", "0").strip().lower() not in ("0", "false", "no", "off")


def max_total_exposure() -> float:
    return float(os.getenv("CANARY_MAX_TOTAL_EXPOSURE", "25"))


def _attr(signal_like, key, default=None):
    """Read `key` from a dict or an object (DecisionTrace/SignalLike)."""
    if isinstance(signal_like, dict):
        return signal_like.get(key, default)
    return getattr(signal_like, key, default)


def _row_get(row, key, default=None):
    """Read a column from a sqlite3.Row that may not have it (pre-migration)."""
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def plan_probe(
    signal_like,
    events: list[replay.ReplayEvent] | None = None,
    *,
    size_usd: float = DEFAULT_PROBE_SIZE_USD,
    gtd_secs: float = DEFAULT_PROBE_GTD_SECS,
) -> dict:
    """Record what a tiny post-only probe WOULD be — never places an order.

    `signal_like` may be a dict or any object exposing token_id, chosen_side
    (or side), and a post price (maker_price / chosen_price / price). The probe
    posts on the passive side at the maker price. When recorded `events` are
    given, attaches a fill/queue label and signed markouts at t+5s / t+30s via
    `src.replay`, so we can read off the planned-probe fill rate and adverse
    selection offline. The returned dict is row-shaped for the canary table with
    `planned = 1`.
    """
    token_id = str(_attr(signal_like, "token_id", "") or "")
    side = str(_attr(signal_like, "chosen_side", None) or _attr(signal_like, "side", "") or "").upper()
    price = (
        _attr(signal_like, "maker_price", None)
        or _attr(signal_like, "chosen_price", None)
        or _attr(signal_like, "price", None)
    )
    price = float(price) if price else 0.0
    ts_post = float(_attr(signal_like, "ts_wall", 0.0) or 0.0)
    size = (size_usd / price) if price > 0 else 0.0

    probe: dict = {
        "order_id": f"probe-{uuid.uuid4().hex[:12]}",
        "ts_post": ts_post,
        "ts_event": None,
        "token_id": token_id,
        "side": side,
        "price": price,
        "size": size,
        "filled": 0,
        "ts_fill": None,
        "queue_ahead_est": None,
        "mark_5s": None,
        "mark_30s": None,
        "resolution": None,
        "adverse_bps": None,
        "rebate_earned": None,
        "planned": 1,
    }

    if events and token_id and side in ("BUY", "SELL") and price > 0:
        result = replay.estimate_maker_replay(events, token_id, side, price, ts_post, gtd_secs)
        probe["filled"] = int(result.filled)
        probe["ts_fill"] = result.fill_ts
        probe["ts_event"] = result.fill_ts
        probe["mark_5s"] = result.mark_5s
        probe["mark_30s"] = result.mark_30s
        # Queue-ahead proxy: passive depth resting ahead of us at post time.
        snap = replay.book_at_or_after(events, token_id, ts_post)
        if snap is not None:
            probe["queue_ahead_est"] = (
                float(snap.bid_size) if side == "BUY" else float(snap.ask_size)
            )
    return probe


def record_probe(probe: dict, db_path: str = default_db_path("canary.db")) -> None:
    """Persist a planned probe row. This is the only write path — still no order."""
    ensure_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cols = ", ".join(CANARY_COLUMNS)
        ph = ", ".join("?" for _ in CANARY_COLUMNS)
        conn.execute(
            f"INSERT OR REPLACE INTO canary ({cols}) VALUES ({ph})",
            tuple(probe.get(c) for c in CANARY_COLUMNS),
        )
        conn.commit()
    finally:
        conn.close()


def report(db_path: str = default_db_path("canary.db")) -> None:
    ensure_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(conn.execute("SELECT * FROM canary ORDER BY ts_post"))
    finally:
        conn.close()
    if not rows:
        print("No canary probes recorded yet.")
        return
    filled = [r for r in rows if r["filled"]]
    adverse = [float(r["adverse_bps"]) for r in rows if r["adverse_bps"] is not None]
    rebate = sum(float(r["rebate_earned"] or 0.0) for r in rows)
    planned = [r for r in rows if _row_get(r, "planned")]
    live = [r for r in rows if not _row_get(r, "planned")]
    print("\n=== Canary Report ===")
    print(f"Probes:       {len(rows)}  (planned={len(planned)} live={len(live)})")
    print(f"Filled:       {len(filled)} ({len(filled) / len(rows) * 100:.1f}%)")
    print(f"Rebate:       ${rebate:+.4f}")
    if adverse:
        print(f"Adverse bps:  mean={sum(adverse) / len(adverse):+.2f} n={len(adverse)}")

    if planned:
        p_filled = [r for r in planned if r["filled"]]
        fill_rate = len(p_filled) / len(planned) * 100
        # Adverse markout: for a maker probe, the post-fill mid moving against
        # the side shows as a negative signed markout. Report the mean over
        # filled probes (where adverse selection actually bites).
        m5 = [float(r["mark_5s"]) for r in p_filled if r["mark_5s"] is not None]
        m30 = [float(r["mark_30s"]) for r in p_filled if r["mark_30s"] is not None]
        print("\nPlanned probes (shadow, no live orders):")
        print(f"  Fill rate:        {fill_rate:.1f}% ({len(p_filled)}/{len(planned)})")
        if m5:
            print(f"  Mean markout@5s:  {sum(m5) / len(m5):+.4f} (n={len(m5)})")
        if m30:
            print(f"  Mean markout@30s: {sum(m30) / len(m30):+.4f} (n={len(m30)})")
        adverse_5 = [x for x in m5 if x < 0]
        if m5:
            print(
                f"  Adverse@5s:       {len(adverse_5)}/{len(m5)} filled probes "
                f"({len(adverse_5) / len(m5) * 100:.1f}%)"
            )


def cli() -> None:
    ap = argparse.ArgumentParser(description="Canary probe report")
    ap.add_argument("--db", default=default_db_path("canary.db"))
    args = ap.parse_args()
    report(args.db)


if __name__ == "__main__":
    cli()