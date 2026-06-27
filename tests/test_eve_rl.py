import numpy as np
import pytest

from src.eve_portfolio import Panel, simulate, walk_forward_portfolio, equal_weights, cash_weights

pytest.importorskip("torch")
from src.eve_rl import DirectRLAllocator  # noqa: E402


def _predictive_panel(T=400, N=10, F=3, beta=0.03, noise=0.004, seed=0):
    """Panel where feature 0 genuinely predicts the next return (exploitable)."""
    rng = np.random.default_rng(seed)
    signals = rng.normal(0, 1, size=(T, N, F))
    fwd = beta * signals[:, :, 0] + rng.normal(0, noise, size=(T, N))
    dates = [f"d{t:04d}" for t in range(T)]
    return Panel(dates, [f"S{i}" for i in range(N)], signals, fwd)


def test_allocator_weights_are_dollar_neutral_and_levered():
    panel = _predictive_panel(T=120)
    alloc = DirectRLAllocator(epochs=8, leverage=1.0, seed=0).fit(panel)
    W = alloc.predict_weights(panel)
    assert W.shape == (120, 10)
    assert abs(W[5].sum()) < 1e-5                       # dollar neutral
    assert np.abs(W[5]).sum() <= 1.0 + 1e-6             # gross leverage <= 1


def test_allocator_learns_an_exploitable_signal_oos():
    panel = _predictive_panel(T=500, seed=1)
    train = Panel(panel.dates[:350], panel.symbols, panel.signals[:350], panel.fwd_returns[:350])
    test = Panel(panel.dates[350:], panel.symbols, panel.signals[350:], panel.fwd_returns[350:])
    alloc = DirectRLAllocator(epochs=40, lr=1e-2, cost=0.0002, seed=0).fit(train)
    W = alloc.predict_weights(test)
    nets = simulate(W, test.fwd_returns, cost=0.0002)
    assert nets.mean() > 0           # captures the signal out-of-sample
    # and it should beat equal-weight on this signal-driven panel
    ew = simulate(equal_weights(test), test.fwd_returns, cost=0.0002)
    assert nets.mean() > ew.mean()


def test_allocator_determinism_same_seed():
    panel = _predictive_panel(T=160)
    w1 = DirectRLAllocator(epochs=10, seed=3).fit(panel).predict_weights(panel)
    w2 = DirectRLAllocator(epochs=10, seed=3).fit(panel).predict_weights(panel)
    assert np.allclose(w1, w2)


def test_allocator_plugs_into_walk_forward():
    panel = _predictive_panel(T=500, seed=2)
    report = walk_forward_portfolio(
        panel,
        {"cash": cash_weights, "equal_weight": equal_weights,
         "rl_allocator": DirectRLAllocator(epochs=15, cost=0.0002, seed=0)},
        cost=0.0002, n_folds=3, min_train=200,
    )
    rl = next(m for m in report.metrics if m.name == "rl_allocator")
    assert rl.n > 0
    assert report.best_name == "rl_allocator"  # the learned strategy is the candidate
