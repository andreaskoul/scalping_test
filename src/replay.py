"""Replay helpers for recorded Polymarket event shards.

This initial replay module focuses on the non-refetchable part of the system:
rebuilding the Polymarket top-of-book timeline captured by `PolyWS`.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

from .poly_ws import BookSnapshot


@dataclass
class ReplayEvent:
    ts_wall: float
    ts_mono: float
    token_id: str
    etype: str
    snapshot: BookSnapshot
    levels: dict | None = None


def load_poly_events(run_dir: str | Path) -> list[ReplayEvent]:
    run_path = Path(run_dir)
    events: list[ReplayEvent] = []
    for shard in sorted(run_path.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                snap = BookSnapshot(
                    token_id=str(row["token_id"]),
                    best_bid=float(row.get("best_bid", 0.0)),
                    best_ask=float(row.get("best_ask", 1.0)),
                    bid_size=float(row.get("bid_sz", 0.0)),
                    ask_size=float(row.get("ask_sz", 0.0)),
                    ts=float(row.get("ts_mono", 0.0)),
                )
                events.append(ReplayEvent(
                    ts_wall=float(row.get("ts_wall", 0.0)),
                    ts_mono=float(row.get("ts_mono", 0.0)),
                    token_id=str(row["token_id"]),
                    etype=str(row.get("etype", "")),
                    snapshot=snap,
                    levels=row.get("levels"),
                ))
    events.sort(key=lambda ev: (ev.ts_wall, ev.ts_mono))
    return events


def latest_books(events: list[ReplayEvent]) -> dict[str, BookSnapshot]:
    books: dict[str, BookSnapshot] = {}
    for event in events:
        books[event.token_id] = event.snapshot
    return books


def report(run_dir: str | Path) -> None:
    events = load_poly_events(run_dir)
    books = latest_books(events)
    with_levels = sum(1 for ev in events if ev.levels)
    print(f"\n=== Poly Replay Shard Report ===")
    print(f"Run dir:       {run_dir}")
    print(f"Events:        {len(events)}")
    print(f"Tokens:        {len(books)}")
    print(f"Full snapshots:{with_levels}")
    if events:
        print(f"First ts:      {events[0].ts_wall:.3f}")
        print(f"Last ts:       {events[-1].ts_wall:.3f}")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Replay recorded Polymarket event shards")
    ap.add_argument("run_dir", help="poly_events/<run_id> directory")
    args = ap.parse_args()
    report(args.run_dir)


if __name__ == "__main__":
    cli()