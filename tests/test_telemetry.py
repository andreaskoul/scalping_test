import sqlite3
import time
import asyncio

from src.telemetry import DecisionTrace, Recorder
from src.signal import SignalGenerator, Side
from src.binance_ws import BinanceTick
from src.poly_ws import BookSnapshot
from src.poly_universe import PolyMarket


def _market(strike=80000.0, expiry_offset=3000.0):
    return PolyMarket(
        condition_id="c", question="Bitcoin above 80,000?",
        yes_token_id="y", no_token_id="n", yes_price=0.5, no_price=0.5,
        strike=strike, expiry_ts=time.time() + expiry_offset, tick_size=0.01,
        is_threshold=True,
    )


def _tick(mid=95000.0, sigma=0.5):
    return BinanceTick(
        symbol="btcusdt", bid=mid - 1, ask=mid + 1, mid=mid,
        sigma_annual=sigma, ts=time.monotonic(), drift_annual=0.0,
    )


def _book(bid=0.65, ask=0.70):
    return BookSnapshot(
        token_id="y", best_bid=bid, best_ask=ask,
        bid_size=100.0, ask_size=100.0, ts=time.monotonic(),
    )


def test_signal_trace_records_signal():
    gen = SignalGenerator(max_notional_per_trade=100, safety_eps=0.001)
    trace = DecisionTrace.new("run-test")
    sig = gen.evaluate(_market(), _tick(), _book(), trace=trace)
    assert sig is not None
    assert trace.signal == 1
    assert trace.chosen_side == Side.BUY.value
    assert trace.edge_buy > 0
    assert trace.reject_reason == ""
    assert trace.p_fair > 0


def test_signal_trace_records_reject_reason():
    gen = SignalGenerator(max_notional_per_trade=100, safety_eps=0.001)
    trace = DecisionTrace.new("run-test")
    sig = gen.evaluate(_market(expiry_offset=30.0), _tick(), _book(), trace=trace)
    assert sig is None
    assert trace.signal == 0
    assert trace.reject_reason == "TTE_TOO_SHORT"


def test_recorder_persists_signal(tmp_path):
    async def run():
        db = tmp_path / "decisions.db"
        rec = Recorder(db_path=str(db), reject_sample=1.0, queue_size=10, batch_size=2)
        await rec.start()
        trace = DecisionTrace.new("run-test")
        trace.market_id = "m"
        trace.mark_signal("BUY", False, 0.5, 10.0)
        rec.record(trace)
        await rec.close()
        return db

    db = asyncio.run(run())

    conn = sqlite3.connect(db)
    row = conn.execute("SELECT run_id, signal, chosen_side, size FROM decisions").fetchone()
    conn.close()
    assert row == ("run-test", 1, "BUY", 10.0)