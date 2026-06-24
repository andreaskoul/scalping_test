"""Statistical validation primitives for strategy edge testing.

Shared by the replay sweep (`src.replay`) and the promotion verdict
(`src.adam_report`). The goal is to never report a raw mean edge as if it were
significant: per-event returns are autocorrelated and sweeps overfit, so we use
block bootstrap confidence intervals and a Deflated-Sharpe-style penalty for the
number of configurations tried.

References:
  - Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection
    Bias, Backtest Overfitting and Non-Normality" (2014).
  - Politis & Romano, "The Stationary Bootstrap" (1994) — moving-block variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as _sps


# Euler-Mascheroni constant (used in the expected-maximum-Sharpe estimator).
_EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    n: int
    block_size: int


def t_stat(values: np.ndarray | list[float]) -> float:
    """One-sample t-statistic of the mean against 0.

    Returns 0.0 for degenerate input (n < 2 or zero variance).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return 0.0
    sd = arr.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(arr.mean() / (sd / math.sqrt(n)))


def newey_west_se(values: np.ndarray | list[float], lags: int | None = None) -> float:
    """HAC (Newey-West) standard error of the mean, robust to autocorrelation.

    Per-event PnL is serially correlated (overlapping windows, regime runs), so
    the plain SE understates uncertainty. `lags` defaults to floor(n**(1/3)).
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return 0.0
    if lags is None:
        lags = max(1, int(n ** (1.0 / 3.0)))
    x = arr - arr.mean()
    gamma0 = float(np.dot(x, x) / n)
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(np.dot(x[k:], x[:-k]) / n)
        var += 2.0 * w * gamma_k
    if var <= 0:
        return 0.0
    return float(math.sqrt(var / n))


def block_bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    block_size: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> BootstrapCI:
    """Moving-block bootstrap CI for the mean of a serially correlated series.

    Resamples contiguous blocks to preserve short-range autocorrelation, so the
    interval is honest about how little independent information overlapping
    per-event returns actually carry.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, 0, 0)
    if n == 1:
        v = float(arr[0])
        return BootstrapCI(v, v, v, 1, 1)
    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = min(block_size, n)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_size))
    starts_max = n - block_size + 1
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()[:n]
        means[b] = arr[idx].mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return BootstrapCI(float(arr.mean()), lo, hi, n, block_size)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected maximum Sharpe under the null of zero true edge across trials.

    Bailey & López de Prado's estimator of E[max SR] when `n_trials` independent
    strategies with cross-trial Sharpe dispersion `sr_std` are tried. Grows with
    the number of configurations swept — this is the selection-bias hurdle.
    """
    if n_trials <= 1 or sr_std <= 0:
        return 0.0
    z1 = _sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = _sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_std * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    sr_observed: float,
    *,
    n_trials: int,
    n_obs: int,
    sr_std: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Probability the true Sharpe exceeds 0 after correcting for selection bias.

    `sr_observed`/`sr_std` are per-observation (non-annualized). Returns a
    probability in [0, 1]; values >= 0.95 are the usual "passes" threshold. The
    benchmark is `expected_max_sharpe(n_trials, sr_std)`, not 0, which is what
    penalizes large sweeps.
    """
    if n_obs <= 1:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sr_std)
    denom = 1.0 - skew * sr_observed + ((kurt - 1.0) / 4.0) * sr_observed ** 2
    if denom <= 0:
        return 0.0
    z = (sr_observed - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(_sps.norm.cdf(z))


def per_obs_sharpe(values: np.ndarray | list[float]) -> float:
    """Per-observation Sharpe (mean/std) of a return series; 0 if degenerate."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    sd = arr.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(arr.mean() / sd)
