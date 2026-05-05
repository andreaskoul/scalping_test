"""Unit tests for src/risk.py."""

import time
import pytest
from pathlib import Path
from unittest.mock import patch

from src.risk import RiskManager, HALT_FILE
from src.signal import Signal, Side
from src.poly_universe import PolyMarket


def _signal(price: float = 0.70, size: float = 10.0) -> Signal:
    market = PolyMarket(
        condition_id="cond-abc",
        question="Will Bitcoin go up?",
        yes_token_id="yes-001",
        no_token_id="no-001",
        yes_price=price,
        no_price=1 - price,
        strike=94000.0,
        expiry_ts=time.time() + 3600,
        tick_size=0.01,
    )
    return Signal(
        market=market,
        token_id="yes-001",
        side=Side.BUY,
        price=price,
        size=size,
        p_star=0.80,
        edge=0.05,
        sigma=0.5,
    )


def _fresh_ts():
    return time.monotonic()


class TestRiskManager:
    def test_allows_valid_order(self):
        rm = RiskManager(max_notional_per_trade=100)
        allowed, reason = rm.check(_signal(0.70, 10), _fresh_ts(), _fresh_ts())
        assert allowed
        assert reason == ""

    def test_blocks_stale_binance(self):
        rm = RiskManager(binance_stale_secs=1.0)
        stale_ts = time.monotonic() - 5.0
        allowed, reason = rm.check(_signal(), stale_ts, _fresh_ts())
        assert not allowed
        assert "stale" in reason.lower()

    def test_blocks_stale_poly(self):
        rm = RiskManager(poly_stale_secs=1.0)
        stale_ts = time.monotonic() - 5.0
        allowed, reason = rm.check(_signal(), _fresh_ts(), stale_ts)
        assert not allowed
        assert "stale" in reason.lower()

    def test_blocks_oversized_trade(self):
        rm = RiskManager(max_notional_per_trade=5.0)
        # price=0.70, size=10 → notional=7.0 > 5.0
        allowed, reason = rm.check(_signal(0.70, 10), _fresh_ts(), _fresh_ts())
        assert not allowed
        assert "notional" in reason.lower()

    def test_per_minute_cap(self):
        rm = RiskManager(max_notional_per_trade=100, max_notional_per_minute=20)
        sig = _signal(0.70, 10)  # notional = 7.0
        # Fill three times to accumulate ~21 USD in the window
        rm.record_fill(7.0)
        rm.record_fill(7.0)
        rm.record_fill(7.0)
        allowed, reason = rm.check(sig, _fresh_ts(), _fresh_ts())
        assert not allowed
        assert "minute" in reason.lower()

    def test_daily_drawdown_stop(self):
        rm = RiskManager(daily_drawdown_stop=50.0)
        rm.record_loss(60.0)
        allowed, reason = rm.check(_signal(), _fresh_ts(), _fresh_ts())
        assert not allowed
        assert "drawdown" in reason.lower()

    def test_error_budget(self):
        rm = RiskManager(max_errors_per_minute=3)
        rm.record_error()
        rm.record_error()
        rm.record_error()
        allowed, reason = rm.check(_signal(), _fresh_ts(), _fresh_ts())
        assert not allowed
        assert "error" in reason.lower()

    def test_manual_halt(self):
        rm = RiskManager()
        rm.halt("test")
        assert rm.is_halted()
        allowed, reason = rm.check(_signal(), _fresh_ts(), _fresh_ts())
        assert not allowed

    def test_halt_file(self, tmp_path, monkeypatch):
        halt = tmp_path / "HALT"
        halt.touch()
        monkeypatch.setattr("src.risk.HALT_FILE", halt)
        rm = RiskManager()
        assert rm.is_halted()
        allowed, reason = rm.check(_signal(), _fresh_ts(), _fresh_ts())
        assert not allowed
        assert "HALT" in reason
