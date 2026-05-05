"""Unit tests for src/signal.py."""

import time
import pytest
from unittest.mock import MagicMock

from src.signal import SignalGenerator, Side
from src.binance_ws import BinanceTick
from src.poly_ws import BookSnapshot
from src.poly_universe import PolyMarket
from src.pricing import taker_fee_per_share, FEE_RATE_CRYPTO


def _market(strike: float = 94000.0, expiry_offset: float = 3600.0) -> PolyMarket:
    return PolyMarket(
        condition_id="cond-abc",
        question=f"Will Bitcoin go up or down? Reference: ${strike:,.0f}",
        yes_token_id="yes-token-001",
        no_token_id="no-token-001",
        yes_price=0.5,
        no_price=0.5,
        strike=strike,
        expiry_ts=time.time() + expiry_offset,
        tick_size=0.01,
    )


def _binance(mid: float = 95000.0, sigma: float = 0.5) -> BinanceTick:
    return BinanceTick(
        symbol="btcusdt",
        bid=mid - 1,
        ask=mid + 1,
        mid=mid,
        sigma_annual=sigma,
        ts=time.monotonic(),
    )


def _book(bid: float, ask: float, bid_size: float = 100.0, ask_size: float = 100.0) -> BookSnapshot:
    return BookSnapshot(
        token_id="yes-token-001",
        best_bid=bid,
        best_ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        ts=time.monotonic(),
    )


class TestEdgeCalc:
    def test_no_signal_when_edge_negative(self):
        # spot=94100, strike=94000, sigma=0.5, T=1h → p* ≈ 0.578
        # Set book so ask=0.56, bid=0.55 → both legs edge < 0
        # BUY edge  = 0.578 - 0.56 - fee(0.56) - 0.003 ≈ -0.003 < 0
        # SELL edge = 0.55 - 0.578 - fee(0.55) - 0.003 ≈ -0.049 < 0
        gen = SignalGenerator(max_notional_per_trade=100, safety_eps=0.003)
        market = _market(strike=94000)
        btc = _binance(mid=94100, sigma=0.5)
        book = _book(bid=0.55, ask=0.56)
        result = gen.evaluate(market, btc, book)
        assert result is None

    def test_buy_signal_fires_on_positive_edge(self):
        gen = SignalGenerator(max_notional_per_trade=100, safety_eps=0.001)
        market = _market(strike=80000)  # deep ITM — p* ≈ 0.98
        btc = _binance(mid=95000, sigma=0.5)
        fee = taker_fee_per_share(0.70, FEE_RATE_CRYPTO)
        # Set ask well below p* so edge > 0 after fee + eps
        book = _book(bid=0.65, ask=0.70)
        result = gen.evaluate(market, btc, book)
        assert result is not None
        assert result.side == Side.BUY
        assert result.edge > 0

    def test_sell_signal_fires_when_p_star_below_bid(self):
        gen = SignalGenerator(max_notional_per_trade=100, safety_eps=0.001)
        market = _market(strike=200000)  # deep OTM — p* ≈ 0.01
        btc = _binance(mid=95000, sigma=0.5)
        book = _book(bid=0.20, ask=0.25)  # bid way above p*
        result = gen.evaluate(market, btc, book)
        assert result is not None
        assert result.side == Side.SELL
        assert result.edge > 0

    def test_cooldown_suppresses_second_signal(self):
        gen = SignalGenerator(max_notional_per_trade=100, safety_eps=0.001, cooldown_secs=10.0)
        market = _market(strike=80000)
        btc = _binance(mid=95000, sigma=0.5)
        book = _book(bid=0.65, ask=0.70)
        first = gen.evaluate(market, btc, book)
        second = gen.evaluate(market, btc, book)
        if first is not None:
            assert second is None  # cooldown should suppress

    def test_no_signal_within_2min_of_expiry(self):
        gen = SignalGenerator(max_notional_per_trade=100, safety_eps=0.001)
        market = _market(strike=80000, expiry_offset=90)  # 90s to expiry
        btc = _binance(mid=95000, sigma=0.5)
        book = _book(bid=0.65, ask=0.70)
        result = gen.evaluate(market, btc, book)
        assert result is None

    def test_size_capped_by_notional(self):
        gen = SignalGenerator(max_notional_per_trade=10, safety_eps=0.001)
        market = _market(strike=80000)
        btc = _binance(mid=95000, sigma=0.5)
        book = _book(bid=0.65, ask=0.70, ask_size=1000)
        result = gen.evaluate(market, btc, book)
        if result is not None:
            assert result.price * result.size <= 10 + 0.01  # small rounding tolerance

    def test_size_capped_by_depth(self):
        gen = SignalGenerator(max_notional_per_trade=1000, safety_eps=0.001)
        market = _market(strike=80000)
        btc = _binance(mid=95000, sigma=0.5)
        book = _book(bid=0.65, ask=0.70, ask_size=3.0)  # only 3 shares available
        result = gen.evaluate(market, btc, book)
        if result is not None:
            assert result.size <= 3.0 + 0.01
