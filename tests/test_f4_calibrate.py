"""Phase 4: self-calibration fits + hot-load wiring."""

import json
import time

import numpy as np

from src.calibrate import fit_calibration, fit_wedge
from src.pricing import (
    load_wedge_coeffs, load_calibration, apply_calibration, wedge_estimate,
)
from src.signal import SignalGenerator, Side
from src.poly_universe import PolyMarket
from src.poly_ws import BookSnapshot


def test_fit_calibration_recovers_linear():
    rng = np.random.default_rng(0)
    ps = rng.uniform(0.1, 0.9, 3000)
    true = np.clip(0.1 + 0.8 * ps, 0, 1)
    y = (rng.uniform(size=3000) < true).astype(float)
    samples = [{"p_star": p, "price": p, "tte_hours": 1.0, "outcome": o}
               for p, o in zip(ps, y)]
    cal = fit_calibration(samples)
    assert abs(cal["b"] - 0.8) < 0.15 and abs(cal["a"] - 0.1) < 0.1
    assert cal["brier_after"] <= cal["brier_before"] + 1e-9


def test_fit_wedge_recovers_slope():
    rng = np.random.default_rng(1)
    ps = rng.uniform(0.05, 0.95, 3000)
    tte = rng.uniform(0.1, 5.0, 3000)
    rich = 0.06 - 0.3 * ps + 0.001 * tte + rng.normal(0, 0.01, 3000)
    price = ps + rich
    samples = [{"p_star": p, "price": pr, "tte_hours": t, "outcome": 1.0}
               for p, pr, t in zip(ps, price, tte)]
    w = fit_wedge(samples)
    assert abs(w["beta_pfair"] - (-0.3)) < 0.05
    assert abs(w["intercept"] - 0.06) < 0.03


def test_fit_returns_none_when_sparse():
    assert fit_calibration([{"p_star": 0.5, "price": 0.5, "tte_hours": 1, "outcome": 1}]) is None
    assert fit_wedge([{"p_star": 0.5, "price": 0.5, "tte_hours": 1, "outcome": 1}]) is None


def test_load_and_apply(tmp_path):
    wp = tmp_path / "w.json"
    wp.write_text(json.dumps({"intercept": 0.2, "beta_pfair": -0.4, "beta_tte_hr": 0.001}))
    w = load_wedge_coeffs(str(wp))
    assert w["beta_pfair"] == -0.4

    cp = tmp_path / "c.json"
    cp.write_text(json.dumps({"a": 0.1, "b": 0.8}))
    ab = load_calibration(str(cp))
    assert ab == (0.1, 0.8)
    assert abs(apply_calibration(0.5, ab) - 0.5) < 1e-9   # 0.1 + 0.8*0.5 = 0.5
    assert apply_calibration(0.5, None) == 0.5
    assert load_wedge_coeffs(str(tmp_path / "nope.json")) is None


def test_fitted_coeffs_flow_through_wedge():
    assert wedge_estimate(0.2, 1.0) != 0.0          # paper prior is non-zero here
    assert wedge_estimate(0.2, 1.0, 0.0, 0.0, 0.0) == 0.0


def test_signalgen_uses_fitted_wedge():
    book = BookSnapshot("y", 0.13, 0.15, 1000.0, 1000.0, time.monotonic())
    m = PolyMarket(condition_id="c", question="q", yes_token_id="y", no_token_id="n",
                   yes_price=0.5, no_price=0.5, strike=95000.0,
                   expiry_ts=time.time() + 3000, tick_size=0.01, is_threshold=True)
    # zeroed wedge coeffs → no haircut → BUY fires like tilt-off
    gen = SignalGenerator(max_notional_per_trade=25, safety_eps=0.005,
                          longshot_tilt_mult=1.0,
                          wedge_coeffs={"intercept": 0.0, "beta_pfair": 0.0, "beta_tte_hr": 0.0})
    s = gen._check_leg(m, "y", Side.BUY, 0.20, 0.20, 1.0, book, 0.5, 0.0)
    assert s is not None and s.side == Side.BUY and s.edge > 0


def test_calibration_applied_in_fair_prob():
    from src.binance_ws import BinanceTick
    m = PolyMarket(condition_id="c", question="q", yes_token_id="y", no_token_id="n",
                   yes_price=0.5, no_price=0.5, strike=95000.0,
                   expiry_ts=time.time() + 3000, tick_size=0.01, is_threshold=True)
    tick = BinanceTick("btcusdt", 94999, 95001, 95000.0, 0.5, time.monotonic(), 0.0)
    base = SignalGenerator(sigma_floor=0.05)
    # force calibration to a constant 0.42 (b=0) → fair_prob returns 0.42
    cal = SignalGenerator(sigma_floor=0.05, calib=(0.42, 0.0))
    p0, _ = base.fair_prob(m, tick)
    p1, _ = cal.fair_prob(m, tick)
    assert abs(p1 - 0.42) < 1e-9 and abs(p0 - 0.42) > 1e-6
