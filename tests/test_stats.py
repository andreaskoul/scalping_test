import numpy as np

from src.stats import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    newey_west_se,
    per_obs_sharpe,
    t_stat,
)


def test_t_stat_degenerate_inputs():
    assert t_stat([]) == 0.0
    assert t_stat([1.0]) == 0.0
    assert t_stat([2.0, 2.0, 2.0]) == 0.0  # zero variance


def test_t_stat_positive_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, size=500)
    assert t_stat(x) > 2.0


def test_block_bootstrap_ci_brackets_mean_and_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(0.3, 1.0, size=400)
    a = block_bootstrap_ci(x, seed=42)
    b = block_bootstrap_ci(x, seed=42)
    assert a.lo < a.mean < a.hi
    assert (a.lo, a.hi) == (b.lo, b.hi)  # deterministic given seed


def test_block_bootstrap_ci_handles_tiny_input():
    assert block_bootstrap_ci([]).n == 0
    one = block_bootstrap_ci([1.5])
    assert one.mean == 1.5 and one.lo == 1.5 and one.hi == 1.5


def test_newey_west_se_nonnegative_and_widens_with_autocorr():
    rng = np.random.default_rng(2)
    iid = rng.normal(0, 1, size=600)
    # AR(1) positively autocorrelated series
    ar = np.zeros(600)
    e = rng.normal(0, 1, size=600)
    for i in range(1, 600):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    assert newey_west_se(iid) >= 0.0
    assert newey_west_se(ar) > newey_west_se(ar, lags=0) - 1e-9


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.1) == 0.0
    assert expected_max_sharpe(100, 0.1) > expected_max_sharpe(10, 0.1) > 0.0


def test_deflated_sharpe_decreases_with_more_trials():
    kw = dict(n_obs=500, sr_std=0.05)
    few = deflated_sharpe_ratio(0.15, n_trials=2, **kw)
    many = deflated_sharpe_ratio(0.15, n_trials=500, **kw)
    assert 0.0 <= many <= few <= 1.0


def test_per_obs_sharpe():
    assert per_obs_sharpe([1.0]) == 0.0
    rng = np.random.default_rng(3)
    x = rng.normal(0.2, 1.0, size=1000)
    assert per_obs_sharpe(x) > 0.0
