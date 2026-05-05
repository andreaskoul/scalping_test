"""Unit tests for src/pricing.py."""

import math
import pytest
from scipy.stats import norm

from src.pricing import (
    implied_prob,
    taker_fee_per_share,
    realized_vol_annual,
    SIGMA_MIN,
    SIGMA_MAX,
    FEE_RATE_CRYPTO,
)


class TestImpliedProb:
    def test_atm_converges_to_half(self):
        # At-the-money with long T → p ≈ 0.5
        p = implied_prob(spot=100, strike=100, time_to_expiry_secs=3600, sigma_annual=0.5)
        assert abs(p - 0.5) < 0.02

    def test_deep_itm(self):
        # Spot >> strike, long expiry → p → 1
        p = implied_prob(spot=200, strike=100, time_to_expiry_secs=86400, sigma_annual=0.3)
        assert p > 0.95

    def test_deep_otm(self):
        # Spot << strike, long expiry → p → 0
        p = implied_prob(spot=50, strike=100, time_to_expiry_secs=86400, sigma_annual=0.3)
        assert p < 0.05

    def test_matches_scipy_norm_cdf(self):
        spot, strike, T_secs, sigma = 95000.0, 94000.0, 1800.0, 0.5
        T = T_secs / (365.25 * 24 * 3600)
        d = (math.log(spot / strike) - 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        expected = float(norm.cdf(d))
        got = implied_prob(spot=spot, strike=strike, time_to_expiry_secs=T_secs, sigma_annual=sigma)
        assert abs(got - expected) < 1e-9

    def test_zero_expiry_returns_half(self):
        p = implied_prob(spot=100, strike=100, time_to_expiry_secs=0, sigma_annual=0.5)
        assert p == 0.5

    def test_negative_expiry_returns_half(self):
        p = implied_prob(spot=100, strike=100, time_to_expiry_secs=-1, sigma_annual=0.5)
        assert p == 0.5

    def test_sigma_clipped_at_min(self):
        p_clipped = implied_prob(spot=100, strike=100, time_to_expiry_secs=3600, sigma_annual=1e-9)
        p_min = implied_prob(spot=100, strike=100, time_to_expiry_secs=3600, sigma_annual=SIGMA_MIN)
        assert abs(p_clipped - p_min) < 1e-12

    def test_sigma_clipped_at_max(self):
        p_clipped = implied_prob(spot=100, strike=100, time_to_expiry_secs=3600, sigma_annual=999)
        p_max = implied_prob(spot=100, strike=100, time_to_expiry_secs=3600, sigma_annual=SIGMA_MAX)
        assert abs(p_clipped - p_max) < 1e-12

    def test_invalid_spot_raises(self):
        with pytest.raises(ValueError):
            implied_prob(spot=0, strike=100, time_to_expiry_secs=3600, sigma_annual=0.5)

    def test_invalid_strike_raises(self):
        with pytest.raises(ValueError):
            implied_prob(spot=100, strike=-1, time_to_expiry_secs=3600, sigma_annual=0.5)


class TestTakerFee:
    def test_peak_at_half(self):
        fee_half = taker_fee_per_share(0.5, FEE_RATE_CRYPTO)
        fee_90 = taker_fee_per_share(0.9, FEE_RATE_CRYPTO)
        fee_10 = taker_fee_per_share(0.1, FEE_RATE_CRYPTO)
        assert fee_half > fee_90
        assert fee_half > fee_10

    def test_parabolic_symmetry(self):
        assert abs(taker_fee_per_share(0.2) - taker_fee_per_share(0.8)) < 1e-12

    def test_crypto_at_ninety_cents(self):
        # 7.2% * 0.9 * 0.1 = 0.00648
        fee = taker_fee_per_share(0.9, FEE_RATE_CRYPTO)
        assert abs(fee - 0.00648) < 1e-10

    def test_approaches_zero_at_extremes(self):
        assert taker_fee_per_share(0.99) < 0.001
        assert taker_fee_per_share(0.01) < 0.001

    def test_always_positive(self):
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            assert taker_fee_per_share(p) > 0


class TestRealizedVol:
    def test_empty_returns_sigma_min(self):
        assert realized_vol_annual([], 60.0) == SIGMA_MIN

    def test_single_return_insufficient(self):
        assert realized_vol_annual([0.001], 60.0) == SIGMA_MIN

    def test_constant_returns_positive(self):
        returns = [0.0001] * 100
        sigma = realized_vol_annual(returns, 60.0)
        assert sigma > SIGMA_MIN

    def test_clipped_at_max(self):
        huge = [1.0] * 100
        sigma = realized_vol_annual(huge, 1.0)
        assert sigma == SIGMA_MAX

    def test_zero_window_returns_sigma_min(self):
        assert realized_vol_annual([0.001, 0.002], 0.0) == SIGMA_MIN
