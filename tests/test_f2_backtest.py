"""Phase 2 (F2 residuals): OFI replay, combo PnL, backtest plumbing."""

import time

from src.microstructure import MicroReplay
from src.arbitrage import ArbLeg, ComboArb, combo_realized_pnl
from src.poly_universe import PolyMarket
from src.backtest import BacktestConfig, _build_signal_generator


def test_microreplay_ofi_window():
    # two buys (+) then a sell (-), each $10 notional
    trades = [(0.0, 100.0, 10.0, 10.0), (1.0, 100.0, 10.0, 10.0), (2.0, 100.0, -10.0, 10.0)]
    mr = MicroReplay(trades)
    assert abs(mr.ofi(2.0, window=100.0) - (10.0 / 30.0)) < 1e-9   # net +10 / abs 30
    assert abs(mr.ofi(2.0, window=0.5) - (-1.0)) < 1e-9            # only the sell


def test_microreplay_features():
    mr = MicroReplay([(float(t), 100.0 + 0.1 * t, 5.0, 5.0) for t in range(0, 200)])
    f = mr.features(190.0, spot=200.0, sigma=0.5, tte=120.0)
    assert f.ofi > 0.0                      # all buys
    assert f.ret_fast > 0.0                 # price rose
    assert f.rvol == 0.5 and f.secs_to_expiry == 120.0
    assert f.move_window_usd != 0.0


def _mkt(cid, yes, no):
    return PolyMarket(
        condition_id=cid, question="q", yes_token_id=yes, no_token_id=no,
        yes_price=0.5, no_price=0.5, strike=95000.0, expiry_ts=time.time() + 600,
        tick_size=0.01, is_threshold=True,
    )


def test_combo_realized_pnl_rebalance():
    m = _mkt("c", "y", "n")
    legs = [ArbLeg(m, "y", "BUY", 0.4, 10.0, 0.01),
            ArbLeg(m, "n", "BUY", 0.4, 10.0, 0.01)]
    combo = ComboArb(arb_id="a", kind="rebalance", legs=legs, net_credit=0.18, notional=8.0)
    # YES resolves true: YES pays 1, NO pays 0.
    #   YES leg: (1-0.4)*10 - 0.01*10 = 5.9 ; NO leg: (0-0.4)*10 - 0.1 = -4.1 → 1.8
    assert abs(combo_realized_pnl(combo, {"c": 1.0}) - 1.8) < 1e-9
    assert combo_realized_pnl(combo, {}) is None


def test_combo_pnl_is_credit_either_way():
    """A real rebalance arb (cost<1) must profit regardless of outcome."""
    m = _mkt("c", "y", "n")
    legs = [ArbLeg(m, "y", "BUY", 0.45, 10.0, 0.0),
            ArbLeg(m, "n", "BUY", 0.45, 10.0, 0.0)]
    combo = ComboArb(arb_id="a", kind="rebalance", legs=legs, net_credit=0.10, notional=9.0)
    assert abs(combo_realized_pnl(combo, {"c": 1.0}) - 1.0) < 1e-9
    assert abs(combo_realized_pnl(combo, {"c": 0.0}) - 1.0) < 1e-9


def test_backtest_config_f2_fields():
    cfg = BacktestConfig(carry_annual=0.1, maker_fill_prob=0.5, ofi_replay=True)
    assert cfg.carry_annual == 0.1 and cfg.maker_fill_prob == 0.5 and cfg.ofi_replay
    # generator builds with a micro_engine wired in
    from src.microstructure import MicrostructureEngine, DirectionalModel
    gen = _build_signal_generator(cfg, MicrostructureEngine(model=DirectionalModel()))
    assert gen.micro_engine is not None
