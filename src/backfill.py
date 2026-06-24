"""Backfill outcomes and realised PnL into decision telemetry.

This is intentionally conservative: it labels resolution and realised PnL for
signal rows where we know side, price, and size. Edge-persistence/executable
labels are filled from recorded Polymarket book replay (`src.replay`): we mark
the signed midpoint markout at t+5s / t+30s and whether the posted maker level
would have actually traded through within a GTD window.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3

import aiohttp

from . import replay
from .pnl import _fetch_resolution
from .storage import default_db_path

# Default GTD window (seconds) for the executable label: did the book trade
# through the posted maker level before this deadline? Kept short to match the
# scalper's reaction horizon; tunable via the CLI.
DEFAULT_EXECUTABLE_GTD_SECS: float = 30.0


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


def _signal_rows_for_labels(db_path: str, limit: int) -> list[sqlite3.Row]:
    """Signal rows that still need at least one replay-derived label."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT decision_id, token_id, chosen_side, chosen_price, maker_price,
                   is_maker, ts_wall, edge_at_5s, edge_at_30s, executable
            FROM decisions
            WHERE signal = 1
              AND token_id != ''
              AND chosen_side IN ('BUY', 'SELL')
              AND chosen_price > 0
              AND (edge_at_5s IS NULL OR edge_at_30s IS NULL OR executable IS NULL)
            ORDER BY ts_wall
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return list(rows)
    finally:
        conn.close()


def compute_edge_labels(
    events: list[replay.ReplayEvent],
    token_id: str,
    side: str,
    chosen_price: float,
    posted_price: float,
    ts_wall: float,
    gtd_secs: float = DEFAULT_EXECUTABLE_GTD_SECS,
) -> tuple[float | None, float | None, int]:
    """Replay-derived (edge_at_5s, edge_at_30s, executable) for one signal.

    Edge is the signed midpoint markout vs the chosen entry price (positive is
    favourable to the side: BUY profits when the midpoint rises). Executable is
    1 when the posted maker level would have traded through within the GTD
    window, else 0. Markouts may be None when the recorded book has no snapshot
    at/after the horizon.
    """
    edge_5s = replay.markout(events, token_id, side, chosen_price, ts_wall, 5.0)
    edge_30s = replay.markout(events, token_id, side, chosen_price, ts_wall, 30.0)
    filled, _fill_ts, _reason = replay.estimate_maker_fill(
        events, token_id, side, posted_price, ts_wall, gtd_secs
    )
    return edge_5s, edge_30s, int(filled)


def backfill_edge_labels(
    db_path: str,
    events_dir: str,
    *,
    limit: int = 50000,
    gtd_secs: float = DEFAULT_EXECUTABLE_GTD_SECS,
) -> tuple[int, int]:
    """Fill edge_at_5s / edge_at_30s / executable for signal rows from replay.

    Loads the recorded Polymarket events once, then for each pending signal row
    computes the signed markout at t+5s / t+30s and the executable label. Only
    rows whose token appears in the replay window get updated. Returns
    (rows_considered, rows_updated).
    """
    events = replay.load_poly_events(events_dir)
    tokens_in_replay = {ev.token_id for ev in events}
    rows = _signal_rows_for_labels(db_path, limit)
    if not rows:
        print("No signal rows pending edge labels.")
        return 0, 0

    conn = sqlite3.connect(db_path)
    updated = 0
    try:
        for row in rows:
            token_id = str(row["token_id"])
            if token_id not in tokens_in_replay:
                continue
            side = str(row["chosen_side"])
            chosen_price = float(row["chosen_price"])
            # Maker probes post at maker_price; takers cross at chosen_price.
            posted_price = (
                float(row["maker_price"])
                if row["is_maker"] and row["maker_price"] and float(row["maker_price"]) > 0
                else chosen_price
            )
            ts_wall = float(row["ts_wall"])
            edge_5s, edge_30s, executable = compute_edge_labels(
                events, token_id, side, chosen_price, posted_price, ts_wall, gtd_secs
            )
            conn.execute(
                """
                UPDATE decisions
                SET edge_at_5s = ?, edge_at_30s = ?, executable = ?
                WHERE decision_id = ?
                """,
                (edge_5s, edge_30s, executable, str(row["decision_id"])),
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()

    print(
        f"Edge-label backfill: {updated}/{len(rows)} signal rows labelled "
        f"from {len(tokens_in_replay)} replayed tokens."
    )
    return len(rows), updated


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
    ap.add_argument(
        "--events",
        default=None,
        help="poly_events/<run_id> dir; when set, fills edge_at_5s/30s + "
        "executable labels from recorded book replay (no network needed).",
    )
    ap.add_argument(
        "--gtd-secs",
        type=float,
        default=DEFAULT_EXECUTABLE_GTD_SECS,
        help="GTD window (s) for the executable trade-through label.",
    )
    args = ap.parse_args()
    if args.events:
        backfill_edge_labels(
            args.db, args.events, limit=args.limit, gtd_secs=args.gtd_secs
        )
    else:
        asyncio.run(backfill(args.db, args.limit))


if __name__ == "__main__":
    cli()