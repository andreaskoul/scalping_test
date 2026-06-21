"""Phase 3: mean-reversion TTE-gate + exit, and real Price-to-Beat + basis."""

import time

from src.meanrev import MeanReversionTracker
from src.resolution import (
    extract_published_ptb, ChainlinkBasis, PriceToBeatCache, window_seconds,
)
from src.poly_universe import PolyMarket


def _mr(**kw):
    base = dict(band=0.05, half_life_secs=600.0, max_hold_secs=3600.0,
               warmup_updates=2, z_entry=1.5, exit_z=0.5, tte_gate_mult=2.0)
    base.update(kw)
    return MeanReversionTracker(**base)


def test_meanrev_tte_gate_blocks_short_markets():
    mr = _mr(half_life_secs=14400.0)        # gate = 28800s
    for _ in range(5):
        mr.update("t", 300.0, 0.50, 0.49, 0.51)   # 5-min market
    a = mr.update("t", 300.0, 0.50, 0.69, 0.71)    # huge deviation
    assert a is None and not mr.has_position("t")   # gated: can't revert in 5min


def test_meanrev_enter_then_exit():
    mr = _mr()
    tte = 100_000.0
    for _ in range(5):
        mr.update("t", tte, 0.50, 0.49, 0.51)
    enter = mr.update("t", tte, 0.50, 0.69, 0.71)   # rich → SELL
    assert enter.kind == "ENTER" and enter.side == "SELL" and mr.has_position("t")
    # D reverts toward the mean → take-profit exit (flatten = BUY)
    ex = None
    for _ in range(3):
        ex = mr.update("t", tte, 0.50, 0.49, 0.51)
        if ex:
            break
    assert ex is not None and ex.kind == "EXIT" and ex.side == "BUY"
    assert not mr.has_position("t")


def test_extract_published_ptb():
    assert extract_published_ptb({"priceToBeat": "95010.5"}) == 95010.5
    assert extract_published_ptb({"startPrice": 94000}) == 94000.0
    assert extract_published_ptb({"price_to_beat": 0}) == 0.0
    assert extract_published_ptb({"unrelated": 1}) == 0.0


def test_chainlink_basis_ewma():
    b = ChainlinkBasis(half_life_secs=1e9)
    assert b.value("btcusdt") == 0.0
    b.record("btcusdt", 10.0)
    assert abs(b.value("btcusdt") - 10.0) < 1e-9
    b.record("btcusdt", 20.0)                # dt≈0 → barely moves off 10
    assert 9.5 < b.value("btcusdt") < 11.0


def _updown(ptb=0.0):
    return PolyMarket(
        condition_id="u1", question="Bitcoin Up or Down",
        yes_token_id="y", no_token_id="n", yes_price=0.5, no_price=0.5,
        strike=0.0, expiry_ts=time.time() + 10, tick_size=0.01,
        symbol="btcusdt", slug="btc-updown-5m-1773742200", is_updown=True,
        price_to_beat=ptb,
    )


async def test_pricetobeat_prefers_published_and_records_basis(monkeypatch):
    assert window_seconds(_updown()) == 300.0
    basis = ChainlinkBasis()
    cache = PriceToBeatCache(basis=basis)

    async def fake_open(self, session, sym, start):
        return 95000.0
    monkeypatch.setattr(PriceToBeatCache, "_fetch_open", fake_open)

    strike = await cache.anchored_strike(None, _updown(ptb=95010.0))
    assert strike == 95010.0                      # used the published value
    assert abs(basis.value("btcusdt") - 10.0) < 1e-9   # basis = published − binance


async def test_pricetobeat_falls_back_to_proxy(monkeypatch):
    basis = ChainlinkBasis()
    cache = PriceToBeatCache(basis=basis)

    async def fake_open(self, session, sym, start):
        return 95000.0
    monkeypatch.setattr(PriceToBeatCache, "_fetch_open", fake_open)

    strike = await cache.anchored_strike(None, _updown(ptb=0.0))
    assert strike == 95000.0                      # proxy
    assert basis.value("btcusdt") == 0.0          # no basis sample without published
