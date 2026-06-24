import sqlite3

from src.category import _rebuild_from_fills, _resolve_fills_db
from src.poly_ws import PolyWS, RECONNECT_BASE, RECONNECT_MAX
from src.telemetry import DecisionTrace, Recorder


# --------- telemetry NO_BOOK / universe-reject cap ---------

def test_universe_reject_cap_limits_kept_rows():
    rec = Recorder(reject_sample=1.0, universe_reject_cap=5, seed=0)
    kept = 0
    for _ in range(100):
        t = DecisionTrace.new("r", stage="universe")
        t.mark_reject("NO_BOOK")
        if rec.should_record(t):
            kept += 1
    assert kept == 5  # hard cap, regardless of sampling passing every time


def test_signals_and_exec_never_capped():
    rec = Recorder(reject_sample=1.0, universe_reject_cap=0, seed=0)
    sig = DecisionTrace.new("r", stage="eval")
    sig.mark_signal("BUY", False, 0.5, 10.0)
    assert rec.should_record(sig) is True
    risk = DecisionTrace.new("r", stage="risk")
    risk.mark_reject("RISK_BLOCKED")
    assert rec.should_record(risk) is True
    # universe rejects are fully suppressed at cap=0
    nb = DecisionTrace.new("r", stage="universe")
    nb.mark_reject("NO_BOOK")
    assert rec.should_record(nb) is False


def test_reject_sample_zero_records_nothing():
    rec = Recorder(reject_sample=0.0, universe_reject_cap=10_000, seed=1)
    nb = DecisionTrace.new("r", stage="universe")
    nb.mark_reject("NO_BOOK")
    assert all(not rec.should_record(nb) for _ in range(50))


# --------- category reconcile path / robustness ---------

def test_resolve_fills_db_prefers_explicit_then_env(monkeypatch):
    assert _resolve_fills_db("/explicit/path.db") == "/explicit/path.db"
    monkeypatch.setenv("FILL_DB_PATH", "/tmp/fills_strict.db")
    assert _resolve_fills_db(None) == "/tmp/fills_strict.db"


async def test_reconcile_missing_file_does_not_crash(tmp_path):
    tracker = await _rebuild_from_fills(str(tmp_path / "absent.db"))
    assert tracker is not None  # empty tracker, no exception


async def test_reconcile_missing_table_does_not_crash(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()  # valid sqlite, no `fills` table
    tracker = await _rebuild_from_fills(str(db))
    assert tracker is not None


# --------- poly_ws counters + reconnect backoff invariant ---------

def test_polyws_counters_track_universe():
    ws = PolyWS(token_ids=["a", "b", "c"])
    assert ws.tracked_count == 3
    assert ws.subscribed_count == 0
    assert ws.reconnects == 0
    ws.update_tokens(["c", "d"])  # delta add
    assert ws.tracked_count == 4


def test_reconnect_backoff_is_bounded_and_monotonic():
    delay = RECONNECT_BASE
    seq = [delay]
    for _ in range(12):
        delay = min(delay * 2, RECONNECT_MAX)
        seq.append(delay)
    assert seq == sorted(seq)          # non-decreasing
    assert seq[-1] == RECONNECT_MAX    # saturates at the cap
    assert RECONNECT_BASE < RECONNECT_MAX
