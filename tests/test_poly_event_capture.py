import asyncio
import gzip
import json

from src.poly_ws import PolyWS
from src.telemetry import Recorder


def test_polyws_event_sink_on_book_move():
    events = []
    ws = PolyWS(["tok"], event_sink=events.append, snapshot_interval_secs=10.0)
    ws._handle(json.dumps({
        "event_type": "book",
        "asset_id": "tok",
        "bids": [{"price": "0.40", "size": "12"}],
        "asks": [{"price": "0.45", "size": "9"}],
    }))
    assert len(events) == 1
    event = events[0]
    assert event["token_id"] == "tok"
    assert event["etype"] == "book"
    assert event["best_bid"] == 0.40
    assert event["best_ask"] == 0.45
    assert "levels" in event


def test_polyws_event_sink_on_price_change_move():
    events = []
    ws = PolyWS(["tok"], event_sink=events.append, snapshot_interval_secs=9999.0)
    ws._handle(json.dumps({
        "event_type": "book",
        "asset_id": "tok",
        "bids": [{"price": "0.40", "size": "12"}],
        "asks": [{"price": "0.45", "size": "9"}],
    }))
    ws._handle(json.dumps({
        "event_type": "price_change",
        "asset_id": "tok",
        "changes": [{"side": "BUY", "price": "0.41", "size": "3"}],
    }))
    assert len(events) == 2
    assert events[-1]["etype"] == "price_change"
    assert events[-1]["best_bid"] == 0.41
    assert "levels" not in events[-1]


def test_recorder_writes_poly_event_shard(tmp_path):
    async def run():
        rec = Recorder(
            db_path=str(tmp_path / "decisions.db"),
            run_id="run-poly-test",
            poly_events_dir=str(tmp_path / "poly_events"),
        )
        await rec.start()
        rec.record_poly_event({
            "ts_wall": 1_782_071_200.0,
            "ts_mono": 123.0,
            "token_id": "tok",
            "etype": "book",
            "best_bid": 0.4,
            "best_ask": 0.45,
            "bid_sz": 12.0,
            "ask_sz": 9.0,
        })
        await rec.close()
        return rec

    rec = asyncio.run(run())
    shard_dir = tmp_path / "poly_events" / "run-poly-test"
    shards = list(shard_dir.glob("*.jsonl.gz"))
    assert rec.poly_recorded == 1
    assert (shard_dir / "manifest.json").exists()
    assert len(shards) == 1
    with gzip.open(shards[0], "rt", encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    assert row["run_id"] == "run-poly-test"
    assert row["token_id"] == "tok"