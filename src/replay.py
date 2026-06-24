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


@dataclass
class MakerReplayResult:
    token_id: str
    side: str
    price: float
    posted_ts: float
    filled: bool
    fill_ts: float | None = None
    mark_5s: float | None = None
    mark_30s: float | None = None
    reason: str = ""


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


def events_for_token(events: list[ReplayEvent], token_id: str) -> list[ReplayEvent]:
    """Return one token's replay events in chronological order."""
    return sorted((ev for ev in events if ev.token_id == token_id), key=lambda ev: (ev.ts_wall, ev.ts_mono))


def book_at_or_after(events: list[ReplayEvent], token_id: str, ts_wall: float) -> BookSnapshot | None:
    """First snapshot at or after wall-clock ts for markout estimates."""
    for ev in events_for_token(events, token_id):
        if ev.ts_wall >= ts_wall:
            return ev.snapshot
    return None


def estimate_maker_fill(
    events: list[ReplayEvent],
    token_id: str,
    side: str,
    price: float,
    posted_ts: float,
    gtd_secs: float,
) -> tuple[bool, float | None, str]:
    """Conservative maker-fill estimate from top-of-book replay.

    BUY maker is considered filled only when a later ask trades/moves down to
    or through our bid. SELL maker is filled only when a later bid moves up to
    or through our ask. This intentionally undercounts fills when only top L2 is
    available; it is a validation label, not a paper-PnL optimism switch.
    """
    side_u = side.upper()
    if side_u not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if price <= 0:
        raise ValueError("price must be positive")
    deadline = posted_ts + max(0.0, gtd_secs)
    future = [ev for ev in events_for_token(events, token_id) if posted_ts < ev.ts_wall <= deadline]
    for ev in future:
        snap = ev.snapshot
        if side_u == "BUY" and snap.best_ask > 0 and snap.best_ask <= price:
            return True, ev.ts_wall, "ASK_THROUGH_PRICE"
        if side_u == "SELL" and snap.best_bid >= price:
            return True, ev.ts_wall, "BID_THROUGH_PRICE"
    return False, None, "GTD_EXPIRED"


def markout(
    events: list[ReplayEvent],
    token_id: str,
    side: str,
    entry_price: float,
    ts_wall: float,
    delay_secs: float,
) -> float | None:
    """Signed midpoint markout after a delay.

    Positive is favorable to the entry side: BUY profits when midpoint rises;
    SELL profits when midpoint falls.
    """
    snap = book_at_or_after(events, token_id, ts_wall + delay_secs)
    if snap is None or snap.best_bid <= 0 or snap.best_ask <= 0:
        return None
    mid = 0.5 * (snap.best_bid + snap.best_ask)
    if side.upper() == "BUY":
        return mid - entry_price
    if side.upper() == "SELL":
        return entry_price - mid
    raise ValueError("side must be BUY or SELL")


def estimate_maker_replay(
    events: list[ReplayEvent],
    token_id: str,
    side: str,
    price: float,
    posted_ts: float,
    gtd_secs: float,
) -> MakerReplayResult:
    filled, fill_ts, reason = estimate_maker_fill(events, token_id, side, price, posted_ts, gtd_secs)
    return MakerReplayResult(
        token_id=token_id,
        side=side.upper(),
        price=price,
        posted_ts=posted_ts,
        filled=filled,
        fill_ts=fill_ts,
        mark_5s=markout(events, token_id, side, price, posted_ts, 5.0),
        mark_30s=markout(events, token_id, side, price, posted_ts, 30.0),
        reason=reason,
    )


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