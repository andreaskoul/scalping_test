"""Tests for the queue-aware maker-fill model (src.makerfill)."""

import pytest

from src.makerfill import (
    FillEstimate,
    estimate_fill,
    expected_maker_rebate,
    power_prob_fill,
    queue_ahead,
)
from src.pricing import maker_rebate_per_share, taker_fee_per_share
from src.poly_ws import BookSnapshot
from src.replay import ReplayEvent


# --------------------------------------------------------------------------- #
#  queue_ahead from full L2 levels                                            #
# --------------------------------------------------------------------------- #

def test_queue_ahead_buy_counts_bids_at_or_above_price():
    levels = {"bids": [[0.50, 10.0], [0.49, 5.0], [0.48, 3.0]], "asks": []}
    # BUY maker resting at 0.49: everyone paying >= 0.49 is ahead → 10 + 5 = 15.
    assert queue_ahead(levels, "BUY", 0.49) == pytest.approx(15.0)
    # At 0.50 only the 0.50 level qualifies.
    assert queue_ahead(levels, "BUY", 0.50) == pytest.approx(10.0)
    # Below the book → everything ahead.
    assert queue_ahead(levels, "BUY", 0.40) == pytest.approx(18.0)


def test_queue_ahead_sell_counts_asks_at_or_below_price():
    levels = {"bids": [], "asks": [[0.55, 8.0], [0.56, 4.0], [0.57, 2.0]]}
    # SELL maker resting at 0.56: asks quoting <= 0.56 are ahead → 8 + 4 = 12.
    assert queue_ahead(levels, "SELL", 0.56) == pytest.approx(12.0)
    assert queue_ahead(levels, "SELL", 0.55) == pytest.approx(8.0)
    assert queue_ahead(levels, "SELL", 0.60) == pytest.approx(14.0)


def test_queue_ahead_empty_or_none_is_zero():
    assert queue_ahead(None, "BUY", 0.5) == 0.0
    assert queue_ahead({}, "BUY", 0.5) == 0.0
    assert queue_ahead({"bids": [], "asks": []}, "SELL", 0.5) == 0.0


def test_queue_ahead_rejects_bad_side():
    with pytest.raises(ValueError):
        queue_ahead({"bids": [[0.5, 1]]}, "HOLD", 0.5)


# --------------------------------------------------------------------------- #
#  power_prob_fill monotonicity                                               #
# --------------------------------------------------------------------------- #

def test_power_prob_fill_zero_volume_is_zero():
    assert power_prob_fill(5.0, 0.0) == 0.0
    assert power_prob_fill(0.0, 0.0) == 0.0


def test_power_prob_fill_full_consumption_is_one():
    # No queue ahead → first trade fills us regardless of n.
    assert power_prob_fill(0.0, 100.0, n=1.0) == pytest.approx(1.0)
    assert power_prob_fill(0.0, 100.0, n=4.0) == pytest.approx(1.0)


def test_power_prob_fill_monotone_increasing_in_volume():
    # More volume traded through → higher fill probability (fixed queue/n).
    q = 10.0
    probs = [power_prob_fill(q, v, n=2.0) for v in (11.0, 20.0, 50.0, 100.0)]
    assert probs == sorted(probs)
    assert probs[0] < probs[-1]


def test_power_prob_fill_monotone_decreasing_in_n():
    # Higher n is more conservative → lower probability for partial consumption.
    q, v = 5.0, 10.0  # half consumed
    probs = [power_prob_fill(q, v, n=n) for n in (1.0, 2.0, 4.0, 8.0)]
    assert probs == sorted(probs, reverse=True)
    assert probs[-1] < probs[0]


def test_power_prob_fill_clipped_when_queue_exceeds_volume():
    # Queue ahead larger than traded volume → cannot have filled yet.
    assert power_prob_fill(100.0, 10.0, n=2.0) == 0.0


# --------------------------------------------------------------------------- #
#  expected_maker_rebate is the realism correction                           #
# --------------------------------------------------------------------------- #

def test_expected_rebate_below_naive_maker_rebate_per_share():
    # CRITICAL realism correction: pricing.maker_rebate_per_share treats the
    # rebate as a GUARANTEED per-share credit (implicitly fill_prob == 1). The
    # real rebate is a pro-rata distribution earned only WHEN we fill, and maker
    # fills are uncertain (queue position) — so at any realistic fill
    # probability the expected rebate is strictly below the naive credit. This
    # is the correction the spec asks us to prove.
    for price in (0.2, 0.35, 0.5, 0.65, 0.8):
        naive = maker_rebate_per_share(price)
        # Realistic maker fill probabilities are well below 1.
        for fp in (0.2, 0.5, 0.75):
            exp = expected_maker_rebate(price, fill_prob=fp)
            assert exp < naive, (price, fp, exp, naive)
        # And it scales linearly with fill probability.
        full = expected_maker_rebate(price, fill_prob=1.0)
        assert expected_maker_rebate(price, fill_prob=0.5) == pytest.approx(0.5 * full)


def test_expected_rebate_zero_fill_is_zero():
    assert expected_maker_rebate(0.5, fill_prob=0.0) == 0.0


def test_expected_rebate_is_share_of_taker_fee():
    price, share = 0.5, 0.20
    fee = taker_fee_per_share(price, 0.072, 1.0)
    assert expected_maker_rebate(price, 1.0, rebate_share=share, fee_rate=0.072) == \
        pytest.approx(share * fee)


# --------------------------------------------------------------------------- #
#  estimate_fill top-of-book fallback                                         #
# --------------------------------------------------------------------------- #

def _ev(ts, bid, ask):
    return ReplayEvent(
        ts_wall=ts, ts_mono=ts, token_id="tok", etype="book",
        snapshot=BookSnapshot("tok", bid, ask, 10.0, 10.0, ts),
    )


def test_estimate_fill_falls_back_to_top_of_book_without_levels():
    # No L2 levels and no traded_volume → conservative trade-through rule.
    events = [_ev(1.0, 0.40, 0.45), _ev(3.0, 0.39, 0.405)]
    # BUY at 0.41: ask later moves to 0.405 <= 0.41 → filled (hard label 1.0).
    est = estimate_fill(events, "tok", "BUY", 0.41, 1.0, 10.0)
    assert isinstance(est, FillEstimate)
    assert est.filled_prob == 1.0
    assert est.queue_ahead == 0.0
    # Expected rebate is present and scaled by the (now 1.0) fill prob.
    assert est.expected_rebate > 0.0


def test_estimate_fill_top_of_book_unfilled():
    events = [_ev(1.0, 0.40, 0.45), _ev(3.0, 0.41, 0.44)]
    # BUY at 0.41: ask never reaches <= 0.41 → not filled.
    est = estimate_fill(events, "tok", "BUY", 0.41, 1.0, 5.0)
    assert est.filled_prob == 0.0
    assert est.expected_rebate == 0.0


def test_estimate_fill_uses_queue_path_with_levels_and_volume():
    levels = {"bids": [[0.40, 20.0]], "asks": []}
    # BUY at 0.41 → no bids >= 0.41 ahead, queue 0, so any volume fills us.
    est = estimate_fill([], "tok", "BUY", 0.41, 0.0, 10.0,
                        levels=levels, traded_volume=5.0, n=2.0)
    assert est.queue_ahead == 0.0
    assert est.filled_prob == pytest.approx(1.0)


def test_estimate_fill_marks_adverse_on_negative_markout():
    est = estimate_fill([], "tok", "BUY", 0.41, 0.0, 10.0,
                        levels={"bids": [[0.40, 5.0]]}, traded_volume=10.0,
                        markout_5s=-0.01)
    assert est.adverse is True
    est_ok = estimate_fill([], "tok", "BUY", 0.41, 0.0, 10.0,
                           levels={"bids": [[0.40, 5.0]]}, traded_volume=10.0,
                           markout_5s=0.01)
    assert est_ok.adverse is False
