"""Unit tests for src/arbitrage.py — strike-monotonicity arb scanner."""

import time
import pytest
from unittest.mock import MagicMock

from src.arbitrage import find_strike_arbs, _ladder
from src.poly_universe import PolyMarket
from src.poly_ws import BookSnapshot


_FROZEN_NOW = time.time()  # captured at module load so siblings share an expiry


def _market(strike: float, token_id: str, expiry_offset: float = 1800.0,
            symbol: str = "btcusdt") -> PolyMarket:
    return PolyMarket(
        condition_id=f"cond-{strike}",
        question=f"Bitcoin above {strike:,.0f} at close?",
        yes_token_id=token_id,
        no_token_id=f"no-{token_id}",
        yes_price=0.5,
        no_price=0.5,
        strike=strike,
        expiry_ts=_FROZEN_NOW + expiry_offset,
        tick_size=0.01,
        symbol=symbol,
        is_threshold=True,
        fee_rate=0.07,
        fee_exponent=1.0,
    )


class _FakePolyWS:
    """Mock PolyWS that returns canned BookSnapshots."""
    def __init__(self, books: dict[str, tuple[float, float, float, float]]):
        # books: token_id -> (best_bid, best_ask, bid_size, ask_size)
        self._books = books

    def snapshot(self, token_id: str):
        b = self._books.get(token_id)
        if b is None:
            return None
        return BookSnapshot(
            token_id=token_id,
            best_bid=b[0], best_ask=b[1],
            bid_size=b[2], ask_size=b[3],
            ts=time.monotonic(),
        )


class TestLadder:
    def test_groups_by_symbol_and_expiry(self):
        m1 = _market(78000, "t1")
        m2 = _market(78500, "t2")
        m3 = _market(79000, "t3", symbol="ethusdt")
        groups = _ladder([m1, m2, m3])
        assert len(groups) == 2

    def test_sorted_by_strike(self):
        m_high = _market(79000, "th")
        m_low = _market(78000, "tl")
        m_mid = _market(78500, "tm")
        groups = _ladder([m_high, m_low, m_mid])
        ladder = list(groups.values())[0]
        assert [m.strike for m in ladder] == [78000, 78500, 79000]

    def test_skips_non_threshold(self):
        m_thresh = _market(78000, "t1")
        m_updown = _market(0, "t2")
        m_updown.is_threshold = False
        groups = _ladder([m_thresh, m_updown])
        assert sum(len(v) for v in groups.values()) == 1


class TestStrikeArbs:
    def test_no_arb_when_monotonicity_holds(self):
        # Lower strike has higher prices (proper P(K_low) > P(K_high))
        m_low = _market(78000, "t_low")
        m_high = _market(78500, "t_high")
        # Same expiry — they cluster
        m_high.expiry_ts = m_low.expiry_ts
        ws = _FakePolyWS({
            "t_low":  (0.85, 0.87, 100, 100),
            "t_high": (0.78, 0.80, 100, 100),
        })
        arbs = find_strike_arbs([m_low, m_high], ws)
        assert arbs == []

    def test_arb_when_monotonicity_violated(self):
        # bid(high) > ask(low) — clear violation
        m_low = _market(78000, "t_low")
        m_high = _market(78500, "t_high")
        m_high.expiry_ts = m_low.expiry_ts
        # ask_low=0.50, bid_high=0.55 → raw spread = 0.05
        # fee_low(0.50) = 0.07*0.5*0.5 = 0.0175
        # fee_high(0.55) = 0.07*0.55*0.45 = 0.017325
        # net_credit = 0.05 - 0.0175 - 0.017325 = 0.015175 > 0.01 default min
        ws = _FakePolyWS({
            "t_low":  (0.48, 0.50, 100, 100),
            "t_high": (0.55, 0.57, 100, 100),
        })
        arbs = find_strike_arbs([m_low, m_high], ws)
        assert len(arbs) == 1
        a = arbs[0]
        assert a.leg_low.side == "BUY" and a.leg_low.price == 0.50
        assert a.leg_high.side == "SELL" and a.leg_high.price == 0.55
        assert a.net_credit > 0.01
        assert a.leg_low.size == a.leg_high.size  # paired sizing

    def test_micro_violation_suppressed_by_min_credit(self):
        # Tiny violation that doesn't clear fees
        m_low = _market(78000, "t_low")
        m_high = _market(78500, "t_high")
        m_high.expiry_ts = m_low.expiry_ts
        # ask_low=0.50, bid_high=0.51 → fees alone eat ~3.5%
        ws = _FakePolyWS({
            "t_low":  (0.49, 0.50, 100, 100),
            "t_high": (0.51, 0.52, 100, 100),
        })
        arbs = find_strike_arbs([m_low, m_high], ws)
        assert arbs == []

    def test_size_limited_by_book_depth(self):
        m_low = _market(78000, "t_low")
        m_high = _market(78500, "t_high")
        m_high.expiry_ts = m_low.expiry_ts
        # Plenty of credit but only 5 shares on the bid side
        ws = _FakePolyWS({
            "t_low":  (0.10, 0.12, 100, 100),
            "t_high": (0.30, 0.32, 5,   100),
        })
        arbs = find_strike_arbs([m_low, m_high], ws, max_notional_usd=1000)
        assert len(arbs) == 1
        assert arbs[0].leg_low.size <= 5.0

    def test_skips_too_short_or_too_long_tte(self):
        m_low = _market(78000, "t_low", expiry_offset=60)  # under 3 min
        m_high = _market(78500, "t_high", expiry_offset=60)
        m_high.expiry_ts = m_low.expiry_ts
        ws = _FakePolyWS({
            "t_low":  (0.48, 0.50, 100, 100),
            "t_high": (0.55, 0.57, 100, 100),
        })
        arbs = find_strike_arbs([m_low, m_high], ws)
        assert arbs == []

    def test_only_adjacent_pairs_checked(self):
        # Three-strike ladder with violation between K1 and K3 only.
        # Adjacent pairs both consistent → no arb fired.
        m1 = _market(78000, "t1")
        m2 = _market(78500, "t2")
        m3 = _market(79000, "t3")
        for m in (m2, m3):
            m.expiry_ts = m1.expiry_ts
        ws = _FakePolyWS({
            "t1": (0.60, 0.62, 100, 100),
            "t2": (0.55, 0.57, 100, 100),
            "t3": (0.51, 0.53, 100, 100),  # well below K1's ask (0.62) — would be an arb if we checked non-adjacent
        })
        arbs = find_strike_arbs([m1, m2, m3], ws)
        # Adjacent pairs (1,2) and (2,3): consistent on both → no arb
        assert arbs == []

    def test_payoff_is_non_negative(self):
        """Sanity: in any actionable arb, the payoff is non-negative
        under all expiry paths.

        long YES_low + short YES_high yields:
          - S_T < K_low:        0 - 0 = 0
          - K_low <= S_T < K_high: 1 - 0 = +1
          - S_T >= K_high:      1 - 1 = 0
        Plus the net_credit collected upfront — strictly positive total.
        """
        m_low = _market(78000, "t_low")
        m_high = _market(78500, "t_high")
        m_high.expiry_ts = m_low.expiry_ts
        ws = _FakePolyWS({
            "t_low":  (0.48, 0.50, 100, 100),
            "t_high": (0.55, 0.57, 100, 100),
        })
        arbs = find_strike_arbs([m_low, m_high], ws)
        assert len(arbs) == 1
        a = arbs[0]
        # Test payoff under each scenario (per share, before fees which are
        # already netted into net_credit).
        for s_t, expected in [
            (m_low.strike - 1, 0.0),
            (m_low.strike + 1, 1.0),  # K_low ≤ S_T < K_high
            (m_high.strike + 1, 0.0),
        ]:
            yes_low = 1.0 if s_t > m_low.strike else 0.0
            yes_high = 1.0 if s_t > m_high.strike else 0.0
            payoff = yes_low - yes_high
            assert payoff == pytest.approx(expected)
            # Total return per share: payoff at expiry + upfront credit
            total = payoff + a.net_credit
            assert total >= 0, f"total payoff negative at S_T={s_t}"
