"""Adam-scoped promotion verdict for the latency-arb scalper.

Reads the decision telemetry DB (and optionally a recorded poly-events dir) and
emits a machine-readable promotion verdict plus human-readable gate lines. The
verdict is intentionally conservative about Polymarket's 2026 cost reality:

  * Crypto markets DO charge a taker fee (since Jan 2026).
  * Maker rebates are a *pro-rata daily distribution* of taker fees (~20% share),
    NOT a guaranteed per-share credit.

So the `paper_ok` edge gate uses **taker-side, fee-inclusive edge as the
conservative floor** (`src.pricing.taker_fee_per_share`) and never books the
maker rebate as guaranteed income. Maker rebate is upside, not a credit.

Scope note: this is the *paper/replay* tier. `canary_ok` and `live_ok` are
ALWAYS False here because there is no live canary fill evidence yet — promoting
past paper requires real post-only probe fills, which this pass cannot produce.

Statistical hurdle: a positive mean edge is meaningless on autocorrelated,
overlapping per-event returns. We require a block-bootstrap t-stat >= 2 (via
`src.stats`) on the fee-inclusive taker edge, plus survival of reaction lag
(`edge5s_median > 0`) and non-degenerate calibration (finite Brier no worse
than the no-skill base-rate baseline).
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import stats
from .pricing import taker_fee_per_share
from .storage import default_db_path

# Default gates (tunable via cli flags).
DEFAULT_MIN_SIGNALS: int = 30
DEFAULT_T_STAT_MIN: float = 2.0


@dataclass(frozen=True)
class AdamVerdict:
    paper_ok: bool
    canary_ok: bool  # always False this pass
    live_ok: bool  # always False this pass

    oos_edge_per_event: float
    t_stat: float
    edge5s_median: float
    brier: float
    executable_rate: float  # fraction in [0, 1]; NaN when unlabelled
    n_signals: int

    # Diagnostics for the human lines / debugging.
    schema_ok: bool
    n_min: int
    t_stat_min: float
    edge_ci_lo: float
    edge_ci_hi: float
    brier_baseline: float
    gate_reasons: tuple[str, ...]

    def machine_line(self) -> str:
        exec_pct = (
            f"{self.executable_rate * 100:.1f}"
            if np.isfinite(self.executable_rate)
            else "n/a"
        )
        return (
            f"adam: paper_ok={self.paper_ok} canary_ok={self.canary_ok} live_ok={self.live_ok}\n"
            f"      oos_edge_per_event={self.oos_edge_per_event:.5f}  "
            f"t_stat={self.t_stat:.2f}  "
            f"edge5s_median={self.edge5s_median:.5f}  "
            f"brier={self.brier:.4f}  "
            f"executable_rate={exec_pct}%  "
            f"n_signals={self.n_signals}"
        )


REQUIRED_COLUMNS = {
    "signal", "chosen_side", "chosen_price", "resolution",
    "edge_at_5s", "p_fair",
}


def _schema_ok(conn: sqlite3.Connection) -> bool:
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
    except sqlite3.OperationalError:
        return False
    return REQUIRED_COLUMNS.issubset(cols)


def _taker_edge_per_event(side: str, chosen_price: float, resolution: float) -> float:
    """Fee-inclusive, taker-side realized edge per share for one resolved event.

    Gross is the resolved payoff vs entry; the Polymarket taker fee at the entry
    price is subtracted. Maker rebate is deliberately NOT credited (it is a
    pro-rata daily distribution, not a guaranteed per-share credit).
    """
    side_u = side.upper()
    if side_u == "BUY":
        gross = resolution - chosen_price
    elif side_u == "SELL":
        gross = chosen_price - resolution
    else:
        gross = 0.0
    return gross - taker_fee_per_share(chosen_price)


def compute_verdict(
    rows,
    *,
    min_signals: int = DEFAULT_MIN_SIGNALS,
    t_stat_min: float = DEFAULT_T_STAT_MIN,
    schema_ok: bool = True,
    seed: int = 0,
) -> AdamVerdict:
    """Pure function: gates -> AdamVerdict over an iterable of decision rows.

    `rows` may be sqlite3.Row objects or dicts. Only resolved signal rows
    (signal=1, resolution in {0,1}, BUY/SELL, chosen_price>0) feed the edge and
    calibration gates. `canary_ok`/`live_ok` are forced False.
    """
    def g(r, k):
        if isinstance(r, dict):
            return r.get(k)
        try:
            return r[k]
        except (IndexError, KeyError):
            return None

    signals = [r for r in rows if g(r, "signal")]
    resolved = [
        r for r in signals
        if g(r, "resolution") in (0.0, 1.0, 0, 1)
        and (g(r, "chosen_side") or "").upper() in ("BUY", "SELL")
        and (g(r, "chosen_price") or 0) and float(g(r, "chosen_price")) > 0
    ]

    edges = [
        _taker_edge_per_event(
            str(g(r, "chosen_side")), float(g(r, "chosen_price")), float(g(r, "resolution"))
        )
        for r in resolved
    ]
    n_signals = len(resolved)

    # Edge stats (block bootstrap + t-stat) over fee-inclusive taker edge.
    if edges:
        mean_edge = float(np.mean(edges))
        ci = stats.block_bootstrap_ci(edges, seed=seed)
        t = stats.t_stat(edges)
        ci_lo, ci_hi = ci.lo, ci.hi
    else:
        mean_edge = 0.0
        t = 0.0
        ci_lo = ci_hi = 0.0

    # Edge persistence at 5s (over all signals that have the label).
    edge5s = [
        float(g(r, "edge_at_5s")) for r in signals if g(r, "edge_at_5s") is not None
    ]
    edge5s_median = float(np.median(edge5s)) if edge5s else float("nan")

    # Executable rate over labelled signals.
    exec_labels = [
        int(g(r, "executable")) for r in signals if g(r, "executable") is not None
    ]
    executable_rate = (
        float(np.mean(exec_labels)) if exec_labels else float("nan")
    )

    # Calibration Brier vs no-skill base-rate baseline.
    cal = [
        (float(g(r, "p_fair")), float(g(r, "resolution")))
        for r in resolved
        if g(r, "p_fair") is not None
    ]
    if cal:
        brier = float(np.mean([(p - o) ** 2 for p, o in cal]))
        p_bar = float(np.mean([o for _, o in cal]))
        brier_baseline = float(np.mean([(p_bar - o) ** 2 for _, o in cal]))
    else:
        brier = float("nan")
        brier_baseline = float("nan")

    # --- Gates for paper_ok (ALL must pass) ---
    reasons: list[str] = []
    if not schema_ok:
        reasons.append("schema_missing")
    if n_signals < min_signals:
        reasons.append(f"n_signals<{min_signals}")
    if not (mean_edge > 0 and t >= t_stat_min):
        reasons.append(f"taker_edge_t<{t_stat_min:g}_or_<=0")
    if not (np.isfinite(edge5s_median) and edge5s_median > 0):
        reasons.append("edge5s_median<=0")
    if not (np.isfinite(brier) and brier <= brier_baseline):
        reasons.append("brier_worse_than_baseline")

    paper_ok = len(reasons) == 0

    return AdamVerdict(
        paper_ok=paper_ok,
        canary_ok=False,
        live_ok=False,
        oos_edge_per_event=mean_edge,
        t_stat=t,
        edge5s_median=edge5s_median,
        brier=brier,
        executable_rate=executable_rate,
        n_signals=n_signals,
        schema_ok=schema_ok,
        n_min=min_signals,
        t_stat_min=t_stat_min,
        edge_ci_lo=ci_lo,
        edge_ci_hi=ci_hi,
        brier_baseline=brier_baseline,
        gate_reasons=tuple(reasons),
    )


def verdict_from_db(
    db_path: str,
    *,
    min_signals: int = DEFAULT_MIN_SIGNALS,
    t_stat_min: float = DEFAULT_T_STAT_MIN,
    seed: int = 0,
) -> AdamVerdict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if not _schema_ok(conn):
            return compute_verdict(
                [], min_signals=min_signals, t_stat_min=t_stat_min,
                schema_ok=False, seed=seed,
            )
        rows = list(conn.execute("SELECT * FROM decisions WHERE signal = 1"))
    finally:
        conn.close()
    return compute_verdict(
        rows, min_signals=min_signals, t_stat_min=t_stat_min, schema_ok=True, seed=seed
    )


def _print_human(v: AdamVerdict) -> None:
    print("\n=== Adam Promotion Verdict (paper/replay tier) ===")
    print(f"Schema present:        {'yes' if v.schema_ok else 'NO'}")
    print(f"Resolved signals:      {v.n_signals} (gate: >= {v.n_min})")
    print(
        f"Taker fee-incl edge:   {v.oos_edge_per_event:+.5f}/event  "
        f"t={v.t_stat:.2f} (gate: edge>0 & t>={v.t_stat_min:g})  "
        f"CI95=[{v.edge_ci_lo:+.5f}, {v.edge_ci_hi:+.5f}]"
    )
    edge5 = f"{v.edge5s_median:+.5f}" if np.isfinite(v.edge5s_median) else "n/a"
    print(f"Edge@5s median:        {edge5} (gate: > 0 — survives reaction lag)")
    if np.isfinite(v.brier):
        print(
            f"Calibration Brier:     {v.brier:.4f} vs no-skill {v.brier_baseline:.4f} "
            f"(gate: <= baseline)"
        )
    else:
        print("Calibration Brier:     n/a (no resolved p_fair)")
    exec_pct = f"{v.executable_rate * 100:.1f}%" if np.isfinite(v.executable_rate) else "n/a"
    print(f"Executable rate:       {exec_pct} (diagnostic)")
    if v.gate_reasons:
        print(f"paper_ok BLOCKED by:   {', '.join(v.gate_reasons)}")
    print(
        "canary_ok/live_ok:     always false this pass — no live post-only "
        "probe fills exist yet (paper/replay evidence only)."
    )
    print()
    print(v.machine_line())


def cli() -> None:
    ap = argparse.ArgumentParser(description="Adam-scoped promotion verdict")
    ap.add_argument("--db", default=default_db_path("decisions.db"))
    ap.add_argument(
        "--events",
        default=None,
        help="poly_events/<run_id> dir. When set, edge_at_5s/30s + executable "
        "labels are backfilled from replay before the verdict is computed.",
    )
    ap.add_argument("--min-signals", type=int, default=DEFAULT_MIN_SIGNALS)
    ap.add_argument("--t-stat-min", type=float, default=DEFAULT_T_STAT_MIN)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.events:
        # Best-effort: fill replay labels so edge5s/executable gates have data.
        from .backfill import backfill_edge_labels

        try:
            backfill_edge_labels(args.db, args.events)
        except Exception as exc:  # pragma: no cover - operational guard
            print(f"(edge-label backfill skipped: {exc})")

    v = verdict_from_db(
        args.db,
        min_signals=args.min_signals,
        t_stat_min=args.t_stat_min,
        seed=args.seed,
    )
    _print_human(v)


if __name__ == "__main__":
    cli()
