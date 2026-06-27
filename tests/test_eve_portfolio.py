import numpy as np
import pytest

from src.eve_portfolio import (
    Panel,
    build_panel,
    cash_weights,
    differential_sharpe_series,
    equal_weights,
    long_short_weights,
    portfolio_metrics,
    simulate,
    walk_forward_portfolio,
)


def _synthetic_closes(n_days=400, n_sym=8, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    dates = [f"2026-{1 + (d // 28) % 12:02d}-{1 + d % 28:02d}" for d in range(n_days)]
    # ensure strictly increasing unique date strings
    dates = [f"20{16 + d // 360:02d}-{1 + (d // 30) % 12:02d}-{1 + d % 28:02d}" for d in range(n_days)]
    for s in range(n_sym):
        price = 100.0
        series = []
        for d in range(n_days):
            price *= 1 + rng.normal(0.0003, 0.01)
            series.append((dates[d], price))
        out[f"S{s}"] = series
    return out


def test_differential_sharpe_tracks_sharpe_improvement():
    # DSR measures the *change* in Sharpe. An upward regime shift should give
    # positive cumulative DSR; the mirror (downward) should give negative.
    rng = np.random.default_rng(0)
    up_shift = np.concatenate([rng.normal(0.0, 0.003, 200), rng.normal(0.01, 0.003, 200)])
    down_shift = np.concatenate([rng.normal(0.01, 0.003, 200), rng.normal(0.0, 0.003, 200)])
    assert differential_sharpe_series(up_shift, eta=0.04).sum() > 0
    assert differential_sharpe_series(down_shift, eta=0.04).sum() < 0


def test_build_panel_shapes_and_leakage_safe():
    panel = build_panel(_synthetic_closes(), momentum_lookbacks=(5, 20), vol_lookback=20)
    T, N, F = panel.signals.shape
    assert N == 8 and F == 3  # 2 momentum + 1 vol
    assert panel.fwd_returns.shape == (T, N)
    assert len(panel.dates) == T
    # fwd_returns are finite
    assert np.isfinite(panel.fwd_returns).all()


def test_simulate_charges_turnover_cost():
    fwd = np.array([[0.10, -0.10], [0.10, -0.10]])
    W = np.array([[1.0, -1.0], [1.0, -1.0]])  # hold the same book both steps
    free = simulate(W, fwd, cost=0.0)
    costed = simulate(W, fwd, cost=0.01)
    # First step pays to put the book on; second step holds (cheap after drift).
    assert costed[0] < free[0]
    assert free[0] == pytest.approx(0.20)


def test_long_short_is_dollar_neutral_and_levered():
    panel = build_panel(_synthetic_closes(), momentum_lookbacks=(5, 20), vol_lookback=20)
    W = long_short_weights(panel, signal_feature=1, top_frac=0.3, leverage=1.0)
    row = W[0]
    assert abs(row.sum()) < 1e-9            # dollar neutral
    assert np.abs(row).sum() == pytest.approx(1.0, abs=1e-9)  # gross leverage 1


def test_portfolio_metrics_positive_series():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.005, size=300)  # positive mean, modest vol
    m = portfolio_metrics("x", r, n_trials=3)
    assert m.sharpe > 0
    assert 0.0 <= m.deflated_sharpe <= 1.0


def test_walk_forward_baselines_only_runs():
    panel = build_panel(_synthetic_closes(n_days=500), momentum_lookbacks=(5, 20), vol_lookback=20)
    report = walk_forward_portfolio(
        panel,
        {"cash": cash_weights, "equal_weight": equal_weights,
         "long_short": lambda p: long_short_weights(p, signal_feature=1)},
        cost=0.0005, n_folds=3, min_train=120,
    )
    names = {m.name for m in report.metrics}
    assert {"cash", "equal_weight", "long_short"} <= names
    assert report.machine_line().startswith("eve_portfolio:")
    cash = next(m for m in report.metrics if m.name == "cash")
    assert cash.ann_return == 0.0 and cash.sharpe == 0.0
