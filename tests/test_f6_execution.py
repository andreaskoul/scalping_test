"""Phase 6: maker fill model, atomic combo + flatten, lifecycle, Kalshi exec."""

import time

import src.execute as ex_mod
from src.execute import Executor, OrderLifecycle
from src.signal import Signal, Side
from src.poly_universe import PolyMarket
from src.kalshi import KalshiExecutor


def _sig(side=Side.BUY, is_maker=False, source="model"):
    m = PolyMarket(condition_id="c", question="q", yes_token_id="y", no_token_id="n",
                   yes_price=0.5, no_price=0.5, strike=95000.0,
                   expiry_ts=time.time() + 600, tick_size=0.01)
    return Signal(market=m, token_id="y", side=side, price=0.5, size=10.0,
                  p_star=0.55, edge=0.02, sigma=0.5, is_maker=is_maker, source=source)


async def test_maker_fill_probability(tmp_path):
    ex_mod.DB_PATH = str(tmp_path / "a.db")
    e = Executor(paper=True, maker_fill_prob=0.0)
    await e.setup()
    assert await e.execute(_sig(is_maker=True)) is None    # never hit
    await e.close()

    ex_mod.DB_PATH = str(tmp_path / "b.db")
    e = Executor(paper=True, maker_fill_prob=1.0)
    await e.setup()
    f = await e.execute(_sig(is_maker=True))
    await e.close()
    assert f is not None and f.is_maker


async def test_execute_atomic_success(tmp_path):
    ex_mod.DB_PATH = str(tmp_path / "c.db")
    e = Executor(paper=True)
    await e.setup()
    fills = await e.execute_atomic([_sig(side=Side.BUY), _sig(side=Side.SELL)])
    await e.close()
    assert fills is not None and len(fills) == 2


async def test_execute_atomic_flattens_orphan(tmp_path):
    ex_mod.DB_PATH = str(tmp_path / "d.db")
    e = Executor(paper=True, maker_fill_prob=0.0)   # maker leg will never fill
    await e.setup()
    res = await e.execute_atomic([
        _sig(side=Side.BUY, is_maker=False),   # taker fills (+10)
        _sig(side=Side.BUY, is_maker=True),    # maker rests unfilled → flatten
    ])
    await e.close()
    assert res is None
    assert abs(e.positions().get("y", 0.0)) < 1e-9   # filled leg was flattened


def test_order_lifecycle():
    lc = OrderLifecycle()
    lc.register("o1", "t", 100.0)
    lc.register("o2", "t", 200.0)
    assert lc.open_count() == 2
    assert lc.due(150.0) == ["o1"]
    lc.drop("o1")
    assert lc.open_count() == 1 and lc.due(300.0) == ["o2"]


def test_kalshi_order_body():
    b = KalshiExecutor.order_body("KXBTC-X", "buy", "yes", 10, 45)
    assert b["yes_price"] == 45 and b["count"] == 10 and b["action"] == "buy" and b["type"] == "limit"
    b2 = KalshiExecutor.order_body("KXBTC-X", "sell", "no", 5, 55)
    assert b2["no_price"] == 55 and "yes_price" not in b2


def test_kalshi_headers(monkeypatch):
    ex = KalshiExecutor("KEYID", "--pem--")
    monkeypatch.setattr(KalshiExecutor, "_sign", lambda self, ts, m, p: "SIG")
    h = ex.headers("POST", "/portfolio/orders")
    assert h["KALSHI-ACCESS-KEY"] == "KEYID"
    assert h["KALSHI-ACCESS-SIGNATURE"] == "SIG"
    assert "KALSHI-ACCESS-TIMESTAMP" in h and h["Content-Type"] == "application/json"
