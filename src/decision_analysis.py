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

    _edge_persistence_section(signal_rows)
    _executable_section(signal_rows)
    _calibration_section(labeled_signals)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _chosen_edge_at_decision(row: sqlite3.Row) -> float | None:
    """Edge@0: the live edge for the chosen side at decision time."""
    side = (row["chosen_side"] or "").upper()
    if side == "BUY" and row["edge_buy"] is not None:
        return float(row["edge_buy"])
    if side == "SELL" and row["edge_sell"] is not None:
        return float(row["edge_sell"])
    return None


def _edge_persistence_section(signal_rows: list[sqlite3.Row]) -> None:
    """Did the edge vanish after reaction lag? Median edge@0 / 5s / 30s."""
    edge0 = [e for e in (_chosen_edge_at_decision(r) for r in signal_rows) if e is not None]
    persisted_5s = [float(r["edge_at_5s"]) for r in signal_rows if r["edge_at_5s"] is not None]
    persisted_30s = [float(r["edge_at_30s"]) for r in signal_rows if r["edge_at_30s"] is not None]
    if not (edge0 or persisted_5s or persisted_30s):
        return
    print("\nEdge persistence (did the edge survive reaction lag?):")
    if edge0:
        print(f"  median edge@0:   {_median(edge0):+.4f} ({len(edge0)} rows)")
    if persisted_5s:
        print(f"  median edge@5s:  {_median(persisted_5s):+.4f} ({len(persisted_5s)} rows)")
    if persisted_30s:
        print(f"  median edge@30s: {_median(persisted_30s):+.4f} ({len(persisted_30s)} rows)")


def _executable_section(signal_rows: list[sqlite3.Row]) -> None:
    """Fraction of signals whose posted level would have traded through."""
    labeled = [r for r in signal_rows if r["executable"] is not None]
    if not labeled:
        return
    execu = sum(1 for r in labeled if int(r["executable"]))
    print(
        f"\nExecutable rate:   {execu / len(labeled) * 100:.1f}% "
        f"({execu}/{len(labeled)} labelled signals traded through)"
    )


def _calibration_section(labeled_signals: list[sqlite3.Row]) -> None:
    """p_fair vs realized resolution: Brier score + reliability gap.

    Brier = mean((p_fair - outcome)^2). The no-skill baseline is the Brier of
    always predicting the base rate, p_bar. A reliability gap < 0 (model Brier
    below baseline) means the probabilities carry information.
    """
    pairs = [
        (float(r["p_fair"]), float(r["resolution"]))
        for r in labeled_signals
        if r["p_fair"] is not None and r["resolution"] is not None
    ]
    # Only meaningful for binary {0,1} resolutions.
    pairs = [(p, o) for p, o in pairs if o in (0.0, 1.0)]
    if not pairs:
        return
    n = len(pairs)
    brier = sum((p - o) ** 2 for p, o in pairs) / n
    p_bar = sum(o for _, o in pairs) / n
    baseline = sum((p_bar - o) ** 2 for _, o in pairs) / n
    gap = brier - baseline
    print("\nCalibration (p_fair vs realized resolution):")
    print(f"  Brier:           {brier:.4f} (n={n})")
    print(f"  No-skill Brier:  {baseline:.4f} (base rate p={p_bar:.3f})")
    verdict = "informative" if gap < 0 else "no better than base rate"
    print(f"  Reliability gap: {gap:+.4f} ({verdict})")


def cli() -> None:
    ap = argparse.ArgumentParser(description="Decision telemetry funnel report")
    ap.add_argument("--db", default=default_db_path("decisions.db"))
    ap.add_argument("--hours", type=float, default=8.0)
    args = ap.parse_args()
    report(args.db, args.hours)


if __name__ == "__main__":
    cli()