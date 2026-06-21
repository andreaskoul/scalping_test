"""Tests for the research-driven overlays added in the max-EV assembly."""

import time

from src.pricing import (
    implied_prob, wedge_estimate, skew_adjusted_sigma,
    physical_to_risk_neutral, maker_rebate_per_share, taker_fee_per_share,
)
from src.sizing import kelly_fraction_binary, kelly_size
from src.microstructure import (
    order_book_imbalance, DirectionalModel, MicroFeatures,
    momentum_bias, MicrostructureEngine,
)
from src.meanrev import MeanReversionTracker
from src.arbitrage import find_rebalance_arbs, find_bucket_arbs, scan_combos
from src.kalshi import KalshiMarket, find_xvenue_arbs
from src.signal import SignalGenerator, Side
from src.poly_universe import PolyMarket
from src.poly_ws import BookSnapshot
from src.binance_ws import BinanceTick
from src.config import Settings


# ---------------- builders / fakes ----------------

def _market(strike=95000.0, expiry_offset=3000.0, **kw):
    return PolyMarket(
        condition_id=kw.get("cid", "cond-x"), question=kw.get("q", "Bitcoin above 95,000?"),
        yes_token_id=kw.get("yes", "yes-tok"), no_token_id=kw.get("no", "no-tok"),
        yes_price=0.5, no_price=0.5, strike=strike,
        expiry_ts=time.time() + expiry_offset, tick_size=0.01,
        symbol=kw.get("symbol", "btcusdt"),
        is_threshold=kw.get("is_threshold", True), is_updown=kw.get("is_updown", False),
        fee_rate=0.07, fee_exponent=1.0,
    )


def _book(bid, ask, bid_size=1000.0, ask_size=1000.0, tok="yes-tok"):
    return BookSnapshot(token_id=tok, best_bid=bid, best_ask=ask,
                        bid_size=bid_size, ask_size=ask_size, ts=time.monotonic())


def _tick(mid=95000.0, sigma=0.5):
    return BinanceTick(symbol="btcusdt", bid=mid - 1, ask=mid + 1, mid=mid,
                       sigma_annual=sigma, ts=time.monotonic(), drift_annual=0.0)


class FakePolyWS:
    def __init__(self, books):
        self._books = books  # token_id -> BookSnapshot

    def snapshot(self, tid):
        return self._books.get(tid)

    def walk_book(self, tid, side, size):
        b = self._books.get(tid)
        if not b:
            return 0.0, 0.0
        return (b.best_ask, b.ask_size) if side == "BUY" else (b.best_bid, b.bid_size)


# ---------------- pricing ----------------

def test_wedge_sign_and_clamp():
    assert wedge_estimate(0.10, 1.0) > 0          # longshots rich
    assert wedge_estimate(0.90, 1.0) < 0          # favourites cheap
    assert abs(wedge_estimate(0.001, 10000.0)) <= 0.20 + 1e-9


def test_skew_raises_otm_downside():
    base = 0.5
    assert skew_adjusted_sigma(base, 95000, 90000, 0.15) > base   # K<S → higher σ
    assert skew_adjusted_sigma(base, 95000, 100000, 0.15) < base  # K>S → lower σ
    assert skew_adjusted_sigma(base, 95000, 90000, 0.0) == base   # disabled


def test_physical_to_rn_bounds():
    for p in (0.0, 0.3, 0.5, 0.8, 1.0):
        q = physical_to_risk_neutral(p, 0.5, 300.0, 0.0)
        assert 0.0 <= q <= 1.0


def test_maker_rebate_positive_peaks_mid():
    assert maker_rebate_per_share(0.5) > maker_rebate_per_share(0.1) > 0


def test_implied_prob_carry_and_discount():
    base = implied_prob(95000, 95000, 3600, 0.5)
    up = implied_prob(95000, 95000, 3600, 0.5, carry_annual=0.5)
    assert up > base                               # positive carry lifts P(up)
    disc = implied_prob(95000, 95000, 3600, 0.5, discount_rate=0.5)
    assert disc < base                             # discounting lowers the value


# ---------------- sizing ----------------

def test_kelly_zero_at_no_edge():
    assert kelly_fraction_binary(0.5, 0.5, "BUY") == 0.0
    assert kelly_fraction_binary(0.6, 0.5, "BUY") > 0.0
    assert kelly_size(0.5, 0.5, "BUY", 25.0) == 0.0


def test_kelly_size_scales_with_edge():
    small = kelly_size(0.55, 0.5, "BUY", 100.0, kelly_fraction=0.3)
    big = kelly_size(0.75, 0.5, "BUY", 100.0, kelly_fraction=0.3)
    assert big > small > 0


# ---------------- microstructure ----------------

def test_obi():
    assert order_book_imbalance(100, 0) == 1.0
    assert order_book_imbalance(0, 100) == -1.0
    assert order_book_imbalance(50, 50) == 0.0


def test_directional_model_follows_ofi():
    m = DirectionalModel()
    up = m.p_up(MicroFeatures(ofi=1.0))
    dn = m.p_up(MicroFeatures(ofi=-1.0))
    assert up > 0.5 > dn


def test_momentum_bias_window():
    f = MicroFeatures(move_window_usd=80.0, secs_to_expiry=120.0)
    assert momentum_bias(f, 60.0, 180.0) == 1
    f2 = MicroFeatures(move_window_usd=80.0, secs_to_expiry=600.0)
    assert momentum_bias(f2, 60.0, 180.0) == 0     # outside near-expiry window


def test_engine_obi_veto():
    eng = MicrostructureEngine(obi_veto=True, obi_veto_threshold=-0.60)
    assert eng.assess(MicroFeatures(), poly_obi=-0.7, side="BUY").vetoed
    assert not eng.assess(MicroFeatures(), poly_obi=-0.5, side="BUY").vetoed
    assert eng.assess(MicroFeatures(), poly_obi=0.7, side="SELL").vetoed


# ---------------- mean reversion ----------------

def test_meanrev_fires_on_deviation():
    mr = MeanReversionTracker(band=0.05, half_life_secs=600.0, max_hold_secs=3600.0,
                              warmup_updates=2, z_entry=1.5, tte_gate_mult=2.0)
    tte = 100_000.0  # well above the gate (2×600)
    for _ in range(5):
        assert mr.update("t", tte, 0.50, 0.49, 0.51) is None
    action = mr.update("t", tte, 0.50, 0.69, 0.71)  # mid 0.70 vs fair 0.50 → rich
    assert action is not None and action.kind == "ENTER" and action.side == "SELL"


# ---------------- combinatorial arbs ----------------

def test_rebalance_arb_detected():
    m = _market()
    ws = FakePolyWS({
        "yes-tok": _book(0.35, 0.40, tok="yes-tok"),
        "no-tok": _book(0.35, 0.40, tok="no-tok"),
    })
    arbs = find_rebalance_arbs([m], ws, min_credit=0.01, max_notional_usd=25)
    assert len(arbs) == 1
    a = arbs[0]
    assert a.kind == "rebalance" and len(a.legs) == 2
    assert all(leg.side == "BUY" for leg in a.legs)


def test_bucket_arb_detected():
    low = _market(strike=90000, cid="c1", yes="y1", no="n1", symbol="btcusdt")
    high = _market(strike=91000, cid="c2", yes="y2", no="n2", symbol="btcusdt")
    # same expiry so they form a ladder
    high.expiry_ts = low.expiry_ts
    ws = FakePolyWS({
        "y1": _book(0.25, 0.30, tok="y1"),
        "n2": _book(0.25, 0.30, tok="n2"),
    })
    arbs = find_bucket_arbs([low, high], ws, min_credit=0.01, max_notional_usd=25)
    assert len(arbs) == 1 and arbs[0].kind == "bucket"


def test_scan_combos_aggregates():
    m = _market()
    ws = FakePolyWS({"yes-tok": _book(0.35, 0.40, tok="yes-tok"),
                     "no-tok": _book(0.35, 0.40, tok="no-tok")})
    combos = scan_combos([m], ws, min_credit=0.01, max_notional_usd=25)
    assert any(c.kind == "rebalance" for c in combos)


# ---------------- cross-venue ----------------

def test_xvenue_arb_detected():
    pm = _market(strike=100000.0)
    ws = FakePolyWS({"yes-tok": _book(0.45, 0.50, tok="yes-tok"),
                     "no-tok": _book(0.45, 0.50, tok="no-tok")})
    km = KalshiMarket(ticker="KXBTC-X", symbol="btcusdt", strike=100000.0,
                      expiry_ts=pm.expiry_ts, yes_bid=0.40, yes_ask=0.45)
    arbs = find_xvenue_arbs([pm], ws, [km], min_credit=0.01, fee=0.0)
    assert len(arbs) == 1 and arbs[0].edge >= 0.01


# ---------------- signal overlays ----------------

def test_longshot_tilt_suppresses_buy():
    book = _book(0.13, 0.15)
    base = SignalGenerator(max_notional_per_trade=25, safety_eps=0.005, longshot_tilt_mult=0.0)
    tilt = SignalGenerator(max_notional_per_trade=25, safety_eps=0.005, longshot_tilt_mult=1.0)
    m = _market()
    s0 = base._check_leg(m, "yes-tok", Side.BUY, 0.20, 0.20, 1.0, book, 0.5, 0.0)
    s1 = tilt._check_leg(m, "yes-tok", Side.BUY, 0.20, 0.20, 1.0, book, 0.5, 0.0)
    assert s0 is not None and s0.side == Side.BUY and s0.edge > 0
    # the favourite-longshot haircut removes the BUY edge in the low tail
    assert s1 is None or s1.edge < s0.edge


def test_maker_path_selected_on_wide_spread():
    book = _book(0.45, 0.55)
    gen = SignalGenerator(max_notional_per_trade=25, safety_eps=0.005,
                          maker_enabled=True, maker_join_ticks=1)
    m = _market()
    s = gen._check_leg(m, "yes-tok", Side.BUY, 0.50, 0.50, 1.0, book, 0.5, 0.0)
    assert s is not None and s.is_maker is True
    assert 0.45 < s.price < 0.55          # posted inside the spread


def test_sell_tail_readmitted():
    # bid at 0.05 — below the BUY price_min (0.10) but allowed on SELL.
    book = _book(0.05, 0.07)
    gen = SignalGenerator(max_notional_per_trade=25, safety_eps=0.001,
                          price_min=0.10, sell_price_min=0.03)
    m = _market(strike=80000)  # deep ITM → p* high → selling the cheap YES is +EV
    s = gen._check_leg(m, "yes-tok", Side.SELL, 0.02, 0.02, 1.0, book, 0.5, 0.0)
    assert s is not None and s.side == Side.SELL


def test_obi_veto_blocks_taker():
    book = _book(0.45, 0.55)
    gen = SignalGenerator(max_notional_per_trade=25, obi_veto=True, obi_veto_threshold=-0.60)
    m = _market()
    assert gen._check_leg(m, "yes-tok", Side.BUY, 0.90, 0.90, 1.0, book, 0.5, -0.7) is None


# ---------------- config ----------------

def test_settings_maxev_defaults():
    cfg = Settings.from_env()
    assert cfg.longshot_tilt_mult == 1.0
    assert cfg.maker_enabled is True
    assert cfg.obi_veto is True
    assert cfg.kelly_enabled is True
    assert cfg.use_price_to_beat is True
    assert "maker=on" in cfg.summary()
