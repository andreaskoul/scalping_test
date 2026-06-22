"""Shadow-maker canary schema and reporting.

The canary is the execution-truth tier: tiny live post-only probes used to
measure fill rate and adverse selection. This module intentionally starts with
schema/reporting primitives; order placement is gated by future live-safety work.
"""

from __future__ import annotations

import argparse
import os
import sqlite3

from .storage import default_db_path, ensure_parent


CANARY_COLUMNS = [
    "order_id", "ts_post", "ts_event", "token_id", "side", "price", "size",
    "filled", "ts_fill", "queue_ahead_est", "mark_5s", "mark_30s",
    "resolution", "adverse_bps", "rebate_earned",
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
                rebate_earned REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def canary_enabled() -> bool:
    return os.getenv("CANARY_ENABLED", "0").strip().lower() not in ("0", "false", "no", "off")


def max_total_exposure() -> float:
    return float(os.getenv("CANARY_MAX_TOTAL_EXPOSURE", "25"))


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
    print("\n=== Canary Report ===")
    print(f"Probes:       {len(rows)}")
    print(f"Filled:       {len(filled)} ({len(filled) / len(rows) * 100:.1f}%)")
    print(f"Rebate:       ${rebate:+.4f}")
    if adverse:
        print(f"Adverse bps:  mean={sum(adverse) / len(adverse):+.2f} n={len(adverse)}")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Canary probe report")
    ap.add_argument("--db", default=default_db_path("canary.db"))
    args = ap.parse_args()
    report(args.db)


if __name__ == "__main__":
    cli()