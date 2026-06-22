"""Backfill outcomes and realised PnL into decision telemetry.

This is intentionally conservative: it labels resolution and realised PnL for
signal rows where we know side, price, and size. Edge-persistence/executable
labels require recorded book replay and are filled by later replay/backfill
passes.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3

import aiohttp

from .pnl import _fetch_resolution
from .storage import default_db_path


def _pending_markets(db_path: str, limit: int) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT market_id
            FROM decisions
            WHERE market_id != '' AND resolution IS NULL
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def _apply_resolution(db_path: str, market_id: str, resolution: float) -> int:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE decisions SET resolution = ? WHERE market_id = ? AND resolution IS NULL",
            (resolution, market_id),
        )
        conn.execute(
            """
            UPDATE decisions
            SET realized_pnl = CASE
                WHEN signal = 1 AND chosen_side = 'BUY' AND chosen_price > 0 AND size > 0
                    THEN (resolution - chosen_price) * size
                WHEN signal = 1 AND chosen_side = 'SELL' AND chosen_price > 0 AND size > 0
                    THEN (chosen_price - resolution) * size
                ELSE realized_pnl
            END
            WHERE market_id = ?
            """,
            (market_id,),
        )
        n = conn.total_changes
        conn.commit()
        return n
    finally:
        conn.close()


async def backfill(db_path: str = default_db_path("decisions.db"), limit: int = 5000) -> None:
    markets = _pending_markets(db_path, limit)
    if not markets:
        print("No pending decision markets to backfill.")
        return

    updated = resolved = 0
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_fetch_resolution(session, mid) for mid in markets])
    for market_id, resolution in zip(markets, results):
        if resolution is None:
            continue
        resolved += 1
        updated += _apply_resolution(db_path, market_id, float(resolution))

    print(f"Backfill complete: {resolved}/{len(markets)} markets resolved, {updated} rows touched.")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Backfill decision telemetry outcomes")
    ap.add_argument("--db", default=default_db_path("decisions.db"))
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()
    asyncio.run(backfill(args.db, args.limit))


if __name__ == "__main__":
    cli()