"""Edge-persistence + executable label math (Workstream D, backfill)."""

import sqlite3

from src.backfill import (
    DEFAULT_EXECUTABLE_GTD_SECS,
    backfill_edge_labels,
    compute_edge_labels,
)
from src.canary import plan_probe
from src.poly_ws import BookSnapshot
from src.replay import ReplayEvent
from src.telemetry import _create_table_sql


def _ev(ts, bid, ask, token="tok"):
    return ReplayEvent(
        ts_wall=ts,
        ts_mono=ts,
        token_id=token,
        etype="book",
        snapshot=BookSnapshot(token, bid, ask, 100.0, 80.0, ts),
    )


def _rising_book():
    """Mid rises from 0.425 -> 0.51 -> 0.60 across t=1,6,31."""
    return [
        _ev(1.0, 0.40, 0.45),
        _ev(6.0, 0.50, 0.52),
        _ev(31.0, 0.58, 0.62),
    ]


def test_buy_profits_when_mid_rises():
    events = _rising_book()
    # Entered BUY at 0.45 at t=1. Mid@6=0.51 (+0.06), mid@31=0.60 (+0.15).
    edge5, edge30, executable = compute_edge_labels(
        events, "tok", "BUY", chosen_price=0.45, posted_price=0.45, ts_wall=1.0
    )
    assert edge5 > 0
    assert edge30 > edge5  # edge grows as mid keeps rising
    assert round(edge5, 4) == 0.06
    assert round(edge30, 4) == 0.15


def test_sell_signs_invert_buy():
    events = _rising_book()
    # Same rising book is adverse to a SELL.
    edge5, edge30, _ = compute_edge_labels(
        events, "tok", "SELL", chosen_price=0.45, posted_price=0.45, ts_wall=1.0
    )
    assert edge5 < 0
    assert edge30 < edge5  # gets worse for the seller as mid rises
    assert round(edge5, 4) == -0.06


def test_executable_label_buy_trades_through():
    events = _rising_book()
    # Posting a BUY maker at 0.46: the ask never drops to/through 0.46 here
    # (asks go 0.45->0.52->0.62), but the FIRST event at t=1 has ask 0.45<=0.46
    # is at posted_ts, not after, so it should NOT fill. Use a price the later
    # asks never reach -> not executable.
    _, _, executable = compute_edge_labels(
        events, "tok", "BUY", chosen_price=0.46, posted_price=0.46, ts_wall=1.0,
        gtd_secs=60.0,
    )
    assert executable == 0

    # A descending-ask book DOES trade through a resting BUY.
    desc = [_ev(1.0, 0.40, 0.45), _ev(5.0, 0.39, 0.43), _ev(8.0, 0.38, 0.41)]
    _, _, ex2 = compute_edge_labels(
        desc, "tok", "BUY", chosen_price=0.42, posted_price=0.42, ts_wall=1.0,
        gtd_secs=30.0,
    )
    assert ex2 == 1


def test_executable_label_respects_gtd():
    # Bid rises through a SELL only at t=10; a 5s GTD must not see it.
    events = [_ev(1.0, 0.40, 0.45), _ev(10.0, 0.50, 0.55)]
    _, _, ex_short = compute_edge_labels(
        events, "tok", "SELL", chosen_price=0.48, posted_price=0.48, ts_wall=1.0,
        gtd_secs=5.0,
    )
    assert ex_short == 0
    _, _, ex_long = compute_edge_labels(
        events, "tok", "SELL", chosen_price=0.48, posted_price=0.48, ts_wall=1.0,
        gtd_secs=30.0,
    )
    assert ex_long == 1


def test_markout_none_when_no_future_book():
    events = [_ev(1.0, 0.40, 0.45)]  # nothing at/after t+5
    edge5, edge30, _ = compute_edge_labels(
        events, "tok", "BUY", 0.45, 0.45, ts_wall=1.0
    )
    assert edge5 is None
    assert edge30 is None


def _make_db(tmp_path, rows):
    db = tmp_path / "decisions.db"
    conn = sqlite3.connect(db)
    conn.execute(_create_table_sql())
    for r in rows:
        conn.execute(
            """
            INSERT INTO decisions (
                decision_id, ts_wall, ts_mono, run_id, stage, market_id,
                token_id, chosen_side, chosen_price, maker_price, is_maker,
                size, signal, p_fair
            ) VALUES (?, ?, ?, 'r', 'exec', 'm', ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                r["decision_id"], r["ts_wall"], r["ts_wall"], r["token_id"],
                r["chosen_side"], r["chosen_price"], r.get("maker_price", 0.0),
                r.get("is_maker", 0), r.get("size", 1.0), r.get("p_fair", 0.5),
            ),
        )
    conn.commit()
    conn.close()
    return db


def _events_dir(tmp_path, events):
    import gzip
    import json

    run_dir = tmp_path / "poly_events" / "run"
    run_dir.mkdir(parents=True)
    with gzip.open(run_dir / "20260622T10.jsonl.gz", "wt", encoding="utf-8") as fh:
        for ev in events:
            fh.write(
                json.dumps(
                    {
                        "ts_wall": ev.ts_wall,
                        "ts_mono": ev.ts_mono,
                        "token_id": ev.token_id,
                        "etype": "book",
                        "best_bid": ev.snapshot.best_bid,
                        "best_ask": ev.snapshot.best_ask,
                        "bid_sz": ev.snapshot.bid_size,
                        "ask_sz": ev.snapshot.ask_size,
                    }
                )
                + "\n"
            )
    return run_dir


def test_backfill_edge_labels_updates_db(tmp_path):
    events = _rising_book()
    db = _make_db(
        tmp_path,
        [
            {
                "decision_id": "d1", "ts_wall": 1.0, "token_id": "tok",
                "chosen_side": "BUY", "chosen_price": 0.45,
            },
            # A token NOT in the replay window -> stays NULL.
            {
                "decision_id": "d2", "ts_wall": 1.0, "token_id": "missing",
                "chosen_side": "BUY", "chosen_price": 0.45,
            },
        ],
    )
    run_dir = _events_dir(tmp_path, events)

    considered, updated = backfill_edge_labels(str(db), str(run_dir))
    assert considered == 2
    assert updated == 1

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    d1 = conn.execute(
        "SELECT edge_at_5s, edge_at_30s, executable FROM decisions WHERE decision_id='d1'"
    ).fetchone()
    d2 = conn.execute(
        "SELECT edge_at_5s, executable FROM decisions WHERE decision_id='d2'"
    ).fetchone()
    conn.close()

    assert round(d1["edge_at_5s"], 4) == 0.06
    assert round(d1["edge_at_30s"], 4) == 0.15
    assert d1["executable"] in (0, 1)
    # Token absent from replay is left unlabelled.
    assert d2["edge_at_5s"] is None
    assert d2["executable"] is None


def test_backfill_uses_maker_price_for_executable(tmp_path):
    # is_maker=1 with a maker_price that the descending ask trades through.
    desc = [_ev(1.0, 0.40, 0.45), _ev(5.0, 0.39, 0.43), _ev(8.0, 0.38, 0.41)]
    db = _make_db(
        tmp_path,
        [
            {
                "decision_id": "m1", "ts_wall": 1.0, "token_id": "tok",
                "chosen_side": "BUY", "chosen_price": 0.45,
                "maker_price": 0.42, "is_maker": 1,
            }
        ],
    )
    run_dir = _events_dir(tmp_path, desc)
    backfill_edge_labels(str(db), str(run_dir), gtd_secs=30.0)
    conn = sqlite3.connect(db)
    executable = conn.execute(
        "SELECT executable FROM decisions WHERE decision_id='m1'"
    ).fetchone()[0]
    conn.close()
    # Posted at maker_price 0.42, the ask drops to 0.41 at t=8 -> trades through.
    assert executable == 1


def test_default_gtd_constant_is_positive():
    assert DEFAULT_EXECUTABLE_GTD_SECS > 0


def test_plan_probe_shadow_attaches_labels(tmp_path):
    """Canary shadow planner records a probe + replay labels, no live order."""
    events = _rising_book()
    signal_like = {
        "token_id": "tok", "chosen_side": "BUY", "maker_price": 0.45,
        "ts_wall": 1.0,
    }
    probe = plan_probe(signal_like, events)
    assert probe["planned"] == 1
    assert probe["side"] == "BUY"
    assert probe["size"] > 0  # $1 / 0.45
    assert probe["mark_5s"] is not None
    # Queue-ahead proxy = passive bid depth at post time.
    assert probe["queue_ahead_est"] == 100.0


def test_plan_probe_without_events_is_inert(tmp_path):
    probe = plan_probe({"token_id": "tok", "side": "SELL", "price": 0.5, "ts_wall": 2.0})
    assert probe["planned"] == 1
    assert probe["filled"] == 0
    assert probe["mark_5s"] is None
