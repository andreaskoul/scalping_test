"""Phase 1 (F1): enriched fills schema + attribution helpers."""

import sqlite3
import time

import src.execute as ex
from src.execute import Executor, FILL_COLUMNS
from src.signal import Signal, Side
from src.poly_universe import PolyMarket
from src.pnl_attribution import _price_bucket, _tte_bucket, _stats, _calibration_report


def _sig(is_maker=False, source="model"):
    m = PolyMarket(
        condition_id="c1", question="Bitcoin above 95,000?",
        yes_token_id="y", no_token_id="n", yes_price=0.5, no_price=0.5,
        strike=95000.0, expiry_ts=time.time() + 1200, tick_size=0.01,
    )
    return Signal(market=m, token_id="y", side=Side.BUY, price=0.5, size=10.0,
                  p_star=0.55, edge=0.02, sigma=0.5, is_maker=is_maker, source=source)


async def test_fill_records_f1_columns(tmp_path):
    ex.DB_PATH = str(tmp_path / "f.db")
    e = Executor(paper=True)
    await e.setup()
    f = await e.execute(_sig(is_maker=True, source="meanrev"))
    await e.close()
    assert f.is_maker and f.source == "meanrev" and f.tte_at_fill > 0 and f.question

    conn = sqlite3.connect(ex.DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM fills").fetchone()
    conn.close()
    for c in FILL_COLUMNS:
        assert c in row.keys()
    assert row["is_maker"] == 1 and row["source"] == "meanrev"
    assert abs(row["expiry_ts"] - row["tte_at_fill"] - row["ts"]) < 5.0


async def test_migration_adds_columns_to_old_db(tmp_path):
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE fills (ts REAL, market_id TEXT, token_id TEXT, side TEXT,"
        " price REAL, size REAL, fee REAL, p_star REAL, edge REAL, paper INTEGER,"
        " order_id TEXT)"
    )
    conn.execute("INSERT INTO fills VALUES (1,'c','t','BUY',0.5,10,0.1,0.5,0.01,1,'o')")
    conn.commit()
    conn.close()

    ex.DB_PATH = db
    e = Executor(paper=True)
    await e.setup()            # should ALTER in the missing columns
    await e.execute(_sig())    # and the new INSERT must work
    await e.close()

    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fills)")}
    n = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    conn.close()
    assert {"expiry_ts", "tte_at_fill", "is_maker", "source", "question"} <= cols
    assert n == 2  # legacy row preserved + new row


def test_buckets():
    assert _price_bucket(0.05) == "0.00-0.20"
    assert _price_bucket(0.5) == "0.40-0.60"
    assert _tte_bucket(0) == "unknown"
    assert _tte_bucket(300) == "<10m"
    assert _tte_bucket(7200) == ">=1h"


def test_stats():
    total, mean, t = _stats([2.0, -1.0, 1.0, 0.0])
    assert abs(total - 2.0) < 1e-9 and abs(mean - 0.5) < 1e-9
    assert math_isfinite(t)
    assert _stats([]) == (0.0, 0.0, 0.0)


def test_calibration_runs(capsys):
    calib = [(0.1 * (i % 10), float(i % 2)) for i in range(20)]
    _calibration_report(calib)
    out = capsys.readouterr().out
    assert "Brier" in out


def math_isfinite(x):
    import math
    return math.isfinite(x)
