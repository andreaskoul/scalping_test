import math

import pytest

from src.eve_labels import (
    DOWN,
    FLAT,
    UP,
    CostModel,
    cost_aware_label,
    cost_aware_labels,
    label_distribution,
    post_cost_pnl,
    post_cost_pnls,
)


def test_cost_model_round_trip_is_two_sided():
    cm = CostModel(taker_fee_bps=2.0, half_spread_bps=1.0, slippage_bps=0.5)
    # per side = (2 + 1 + 0.5) bps; round trip crosses twice.
    assert cm.per_side() == pytest.approx(3.5 / 1e4)
    assert cm.round_trip() == pytest.approx(7.0 / 1e4)


def test_cost_aware_label_gates_on_round_trip_cost():
    c = 0.0030  # 30 bps round trip
    assert cost_aware_label(0.0050, c) == UP       # clears cost upward
    assert cost_aware_label(-0.0050, c) == DOWN     # clears cost downward
    assert cost_aware_label(0.0020, c) == FLAT      # inside the band -> no trade
    assert cost_aware_label(0.0030, c) == FLAT      # boundary is not strict edge
    assert cost_aware_label(-0.0030, c) == FLAT


def test_cost_aware_label_handles_nan_as_no_trade():
    assert cost_aware_label(float("nan"), 0.001) == FLAT


def test_cost_negative_rejected():
    with pytest.raises(ValueError):
        cost_aware_label(0.01, -0.001)


def test_post_cost_pnl_directions():
    c = 0.0030
    assert post_cost_pnl(UP, 0.0100, c) == pytest.approx(0.0070)
    assert post_cost_pnl(DOWN, -0.0100, c) == pytest.approx(0.0070)
    assert post_cost_pnl(FLAT, 0.0100, c) == 0.0
    # A wrong-way trade loses the move AND the cost.
    assert post_cost_pnl(UP, -0.0100, c) == pytest.approx(-0.0130)


def test_correctly_labeled_directional_sample_is_positive_ev():
    # By construction, any sample labeled UP/DOWN has positive post-cost PnL.
    c = 0.0030
    for r in (0.004, 0.01, -0.004, -0.02, 0.0031, -0.0031):
        a = cost_aware_label(r, c)
        if a != FLAT:
            assert post_cost_pnl(a, r, c) > 0
        else:
            assert post_cost_pnl(a, r, c) == 0.0


def test_vector_helpers_and_distribution():
    c = 0.0030
    frs = [0.01, -0.01, 0.0, 0.005, -0.005]
    labels = cost_aware_labels(frs, c)
    assert labels == [UP, DOWN, FLAT, UP, DOWN]
    pnls = post_cost_pnls(labels, frs, c)
    assert all(math.isfinite(p) for p in pnls)
    assert pnls[2] == 0.0
    assert label_distribution(labels) == {"down": 2, "flat": 1, "up": 2}


def test_post_cost_pnls_length_mismatch_raises():
    with pytest.raises(ValueError):
        post_cost_pnls([UP, DOWN], [0.01], 0.001)
