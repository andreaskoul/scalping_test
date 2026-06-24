"""Adam promotion-verdict gates (Workstream D, adam_report)."""

import math
import sqlite3

from src.adam_report import (
    DEFAULT_MIN_SIGNALS,
    AdamVerdict,
    compute_verdict,
    verdict_from_db,
)
from src.telemetry import _create_table_sql


def _positive_rows(n=40):
    """Strong positive fee-inclusive taker edge with informative p_fair.

    80% of BUY-at-0.40 signals resolve to 1; p_fair is high for winners and low
    for losers, so the Brier beats the no-skill base-rate baseline. Variance is
    non-zero (some losers), so the block-bootstrap t-stat clears 2.
    """
    rows = []
    for i in range(n):
        win = (i % 5 != 0)  # 80% win rate -> positive after taker fee
        rows.append(
            {
                "signal": 1,
                "chosen_side": "BUY",
                "chosen_price": 0.40,
                "resolution": 1.0 if win else 0.0,
                "edge_at_5s": 0.015,  # edge survives reaction lag
                "p_fair": 0.75 if win else 0.35,
                "executable": 1,
            }
        )
    return rows


def test_positive_dataset_passes_paper_ok():
    v = compute_verdict(_positive_rows())
    assert isinstance(v, AdamVerdict)
    assert v.paper_ok is True
    assert v.gate_reasons == ()
    assert v.oos_edge_per_event > 0
    assert v.t_stat >= 2.0
    assert v.edge5s_median > 0
    assert math.isfinite(v.brier) and v.brier <= v.brier_baseline


def test_canary_and_live_always_false_even_on_positive():
    v = compute_verdict(_positive_rows())
    assert v.canary_ok is False
    assert v.live_ok is False


def test_zero_edge_fails():
    """Even-money outcomes around the entry leave no fee-inclusive edge."""
    rows = []
    for i in range(40):
        rows.append(
            {
                "signal": 1,
                "chosen_side": "BUY",
                "chosen_price": 0.50,
                "resolution": 1.0 if i % 2 == 0 else 0.0,  # ~50/50 around 0.50
                "edge_at_5s": 0.01,
                "p_fair": 0.50,
                "executable": 1,
            }
        )
    v = compute_verdict(rows)
    assert v.paper_ok is False
    assert "taker_edge_t<2_or_<=0" in v.gate_reasons


def test_too_few_signals_fails():
    v = compute_verdict(_positive_rows(n=5))
    assert v.paper_ok is False
    assert any("n_signals" in r for r in v.gate_reasons)


def test_edge5s_non_positive_fails():
    rows = _positive_rows()
    for r in rows:
        r["edge_at_5s"] = -0.02  # edge vanished after reaction lag
    v = compute_verdict(rows)
    assert v.paper_ok is False
    assert "edge5s_median<=0" in v.gate_reasons


def test_schema_missing_fails():
    v = compute_verdict([], schema_ok=False)
    assert v.paper_ok is False
    assert "schema_missing" in v.gate_reasons
    assert v.n_signals == 0


def test_machine_line_formats_correctly():
    v = compute_verdict(_positive_rows())
    line = v.machine_line()
    assert "adam: paper_ok=True canary_ok=False live_ok=False" in line
    assert "oos_edge_per_event=" in line
    assert "t_stat=" in line
    assert "edge5s_median=" in line
    assert "brier=" in line
    assert "executable_rate=" in line
    assert "n_signals=" in line
    # canary/live literally False in the machine line.
    assert "canary_ok=False" in line
    assert "live_ok=False" in line


def test_machine_line_handles_nan_executable_rate():
    rows = _positive_rows()
    for r in rows:
        r.pop("executable")  # no executable labels -> rate is NaN
    v = compute_verdict(rows)
    assert math.isnan(v.executable_rate)
    assert "executable_rate=n/a%" in v.machine_line()


def test_sell_side_edge_signs_correctly():
    """SELL at a high price that resolves to 0 is a winning fee-inclusive edge."""
    rows = []
    for i in range(40):
        win = (i % 5 != 0)
        rows.append(
            {
                "signal": 1,
                "chosen_side": "SELL",
                "chosen_price": 0.60,
                "resolution": 0.0 if win else 1.0,  # SELL wins when it resolves 0
                "edge_at_5s": 0.01,
                "p_fair": 0.25 if win else 0.65,
                "executable": 1,
            }
        )
    v = compute_verdict(rows)
    assert v.oos_edge_per_event > 0
    assert v.paper_ok is True


def test_verdict_from_db_roundtrip(tmp_path):
    db = tmp_path / "decisions.db"
    conn = sqlite3.connect(db)
    conn.execute(_create_table_sql())
    for i, r in enumerate(_positive_rows()):
        conn.execute(
            """
            INSERT INTO decisions (
                decision_id, ts_wall, ts_mono, run_id, stage, market_id,
                token_id, chosen_side, chosen_price, size, signal,
                resolution, edge_at_5s, executable, p_fair
            ) VALUES (?, 1, 1, 'r', 'exec', 'm', 't', ?, ?, 1.0, 1, ?, ?, ?, ?)
            """,
            (
                f"d{i}", r["chosen_side"], r["chosen_price"], r["resolution"],
                r["edge_at_5s"], r["executable"], r["p_fair"],
            ),
        )
    conn.commit()
    conn.close()

    v = verdict_from_db(str(db))
    assert v.paper_ok is True
    assert v.n_signals == 40
    assert v.canary_ok is False and v.live_ok is False


def test_verdict_from_db_missing_table(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()  # no decisions table
    v = verdict_from_db(str(db))
    assert v.paper_ok is False
    assert v.schema_ok is False
    assert "schema_missing" in v.gate_reasons


def test_default_min_signals_constant():
    assert DEFAULT_MIN_SIGNALS == 30
