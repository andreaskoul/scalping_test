import gzip
import json
import sqlite3

from src.backfill import _apply_resolution
from src.backtest import _price_persisted
from src.replay import load_poly_events, latest_books
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