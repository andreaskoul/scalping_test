import gzip
import json
import sqlite3

import pytest

from src.backfill import _apply_resolution
from src.backtest import _price_persisted
from src.poly_ws import BookSnapshot
from src.replay import (
    ReplayEvent,
    estimate_maker_fill,
    estimate_maker_replay,
    latest_books,
    load_poly_events,
    markout,
)
from src.telemetry import _create_table_sql


def test_backfill_resolution_and_pnl(tmp_path):
    db = tmp_path / "decisions.db"
    conn = sqlite3.connect(db)
    conn.execute(_create_table_sql())
    conn.execute(
        """
        INSERT INTO decisions (
            decision_id, ts_wall, ts_mono, run_id, stage, market_id, token_id,
            chosen_side, chosen_price, size, signal
        ) VALUES ('d1', 1, 1, 'r', 'exec', 'm1', 't', 'BUY', 0.25, 4.0, 1)
        """
    )
    conn.commit()
    conn.close()
    _apply_resolution(str(db), "m1", 1.0)
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT resolution, realized_pnl FROM decisions").fetchone()
    conn.close()
    assert row == (1.0, 3.0)


def test_replay_loads_poly_event_shards(tmp_path):
    run_dir = tmp_path / "poly_events" / "run"
    run_dir.mkdir(parents=True)
    with gzip.open(run_dir / "20260622T10.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts_wall": 1.0,
            "ts_mono": 2.0,
            "token_id": "tok",
            "etype": "book",
            "best_bid": 0.4,
            "best_ask": 0.45,
            "bid_sz": 10,
            "ask_sz": 9,
        }) + "\n")
    events = load_poly_events(run_dir)
    books = latest_books(events)
    assert len(events) == 1
    assert books["tok"].best_bid == 0.4
    assert books["tok"].ask_size == 9


def test_price_persisted_requires_neighbors_within_band():
    history = [(1, 0.10), (2, 0.11), (3, 0.115)]
    assert _price_persisted(history, 1, 0.02)
    assert not _price_persisted(history, 1, 0.001)
    assert not _price_persisted(history, 0, 0.02)


def _replay_event(ts, bid, ask):
    return ReplayEvent(
        ts_wall=ts,
        ts_mono=ts,
        token_id="tok",
        etype="book",
        snapshot=BookSnapshot("tok", bid, ask, 10.0, 10.0, ts),
    )


def test_estimate_maker_fill_buy_when_ask_moves_through_price():
    events = [
        _replay_event(1.0, 0.40, 0.45),
        _replay_event(3.0, 0.39, 0.42),
        _replay_event(5.0, 0.38, 0.405),
    ]

    filled, fill_ts, reason = estimate_maker_fill(events, "tok", "BUY", 0.41, 1.0, 10.0)

    assert filled
    assert fill_ts == 5.0
    assert reason == "ASK_THROUGH_PRICE"


def test_estimate_maker_fill_sell_when_bid_moves_through_price():
    events = [
        _replay_event(1.0, 0.40, 0.45),
        _replay_event(4.0, 0.44, 0.48),
        _replay_event(6.0, 0.46, 0.49),
    ]

    filled, fill_ts, reason = estimate_maker_fill(events, "tok", "SELL", 0.455, 1.0, 10.0)

    assert filled
    assert fill_ts == 6.0
    assert reason == "BID_THROUGH_PRICE"


def test_estimate_maker_fill_expires_without_trade_through():
    events = [_replay_event(1.0, 0.40, 0.45), _replay_event(4.0, 0.41, 0.44)]

    filled, fill_ts, reason = estimate_maker_fill(events, "tok", "BUY", 0.41, 1.0, 3.0)

    assert not filled
    assert fill_ts is None
    assert reason == "GTD_EXPIRED"


def test_markout_and_maker_replay_result_are_signed_to_side():
    events = [
        _replay_event(1.0, 0.40, 0.45),
        _replay_event(6.0, 0.50, 0.52),
        _replay_event(31.0, 0.30, 0.34),
    ]

    assert markout(events, "tok", "BUY", 0.45, 1.0, 5.0) == pytest.approx(0.06)
    assert markout(events, "tok", "SELL", 0.45, 1.0, 5.0) == pytest.approx(-0.06)
    result = estimate_maker_replay(events, "tok", "BUY", 0.52, 1.0, 10.0)
    assert result.filled
    assert result.fill_ts == 6.0
    assert result.mark_30s == pytest.approx(-0.20)