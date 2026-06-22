"""Decision telemetry analysis helpers.

Usage:
    python -m src.decision_analysis --db data/db/decisions.db --hours 8
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from collections import Counter

from .storage import default_db_path


def _rows(db_path: str, hours: float) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cutoff = time.time() - hours * 3600 if hours > 0 else 0.0
        return list(
            conn.execute(
                "SELECT * FROM decisions WHERE ts_wall >= ? ORDER BY ts_wall",
                (cutoff,),
            )
        )
    finally:
        conn.close()


def report(db_path: str, hours: float) -> None:
    try:
        rows = _rows(db_path, hours)
    except sqlite3.OperationalError as exc:
        print(f"No decision telemetry in {db_path}: {exc}")
        return

    total = len(rows)
    signals = sum(1 for row in rows if row["signal"])
    rejects = total - signals
    stages = Counter(row["stage"] or "unknown" for row in rows)
    reasons = Counter((row["reject_reason"] or "SIGNAL") for row in rows)
    fills_like = sum(1 for row in rows if row["stage"] == "exec" and row["signal"])
    unfilled = sum(1 for row in rows if row["reject_reason"] == "EXEC_UNFILLED")
    labeled = [row for row in rows if row["resolution"] is not None]
    signal_rows = [row for row in rows if row["signal"]]
    labeled_signals = [row for row in signal_rows if row["resolution"] is not None]
    pnl_rows = [row for row in labeled_signals if row["realized_pnl"] is not None]

    label = f"last {hours:g}h" if hours > 0 else "all time"
    print(f"\n=== Decision Funnel ({label}) ===")
    print(f"Rows recorded:    {total}")
    print("Note: reject rows may be sampled; signals/risk/exec rows are always recorded.")
    print(f"Signals:          {signals}")
    print(f"Rejects:          {rejects}")
    print(f"Exec fills rows:  {fills_like}")
    print(f"Exec unfilled:    {unfilled}")
    if total:
        print(f"Signal rate:      {signals / total * 100:.3f}% of recorded rows")
    if labeled:
        print(f"Resolved rows:    {len(labeled)}")
    if pnl_rows:
        total_pnl = sum(float(row["realized_pnl"] or 0.0) for row in pnl_rows)
        win_rate = sum(1 for row in pnl_rows if float(row["realized_pnl"] or 0.0) > 0) / len(pnl_rows)
        print(f"Signal PnL rows:  {len(pnl_rows)}")
        print(f"Signal PnL:       ${total_pnl:+.4f}")
        print(f"Signal win rate:  {win_rate * 100:.1f}%")

    print("\nBy stage:")
    for stage, count in stages.most_common():
        print(f"  {stage:12s} {count:8d}")

    print("\nTop reject/signal reasons:")
    for reason, count in reasons.most_common(20):
        print(f"  {reason:32s} {count:8d}")

    print("\nSignals by side/execution:")
    side_exec = Counter(
        (row["chosen_side"] or "unknown", "maker" if row["is_maker"] else "taker")
        for row in signal_rows
    )
    for (side, exec_type), count in side_exec.most_common():
        print(f"  {side:5s} {exec_type:5s} {count:8d}")

    persisted_5s = [row for row in signal_rows if row["edge_at_5s"] is not None]
    persisted_30s = [row for row in signal_rows if row["edge_at_30s"] is not None]
    if persisted_5s or persisted_30s:
        print("\nEdge persistence:")
        if persisted_5s:
            med5 = sorted(float(row["edge_at_5s"]) for row in persisted_5s)[len(persisted_5s) // 2]
            print(f"  median edge@5s:  {med5:+.4f} ({len(persisted_5s)} rows)")
        if persisted_30s:
            med30 = sorted(float(row["edge_at_30s"]) for row in persisted_30s)[len(persisted_30s) // 2]
            print(f"  median edge@30s: {med30:+.4f} ({len(persisted_30s)} rows)")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Decision telemetry funnel report")
    ap.add_argument("--db", default=default_db_path("decisions.db"))
    ap.add_argument("--hours", type=float, default=8.0)
    args = ap.parse_args()
    report(args.db, args.hours)


if __name__ == "__main__":
    cli()