"""
Binary-option pricing and Polymarket fee helpers.

For an "Up/Down" BTC market the payoff is 1 if S_T > K, else 0.
We use a lognormal (Black-Scholes) model with zero drift over the
short time horizons (minutes-to-hours) where the arb fires.

  p*(S, K, T, σ) = Φ( (ln(S/K) - 0.5·σ²·T) / (σ·√T) )

Polymarket taker fee per share (parabolic in price):
  fee = fee_rate × p × (1 − p)
where fee_rate = 0.072 for crypto markets.
"""

import math
from scipy.stats import norm  # type: ignore

# Polymarket category fee rates (taker, decimal)
FEE_RATE_CRYPTO: float = 0.072
FEE_RATE_SPORTS: float = 0.03
FEE_RATE_POLITICS: float = 0.04

# Vol guard rails — annualised σ clipped to these before scaling to T
SIGMA_MIN: float = 0.05   # 5 % annual  (~0.003 % per minute)
SIGMA_MAX: float = 5.00   # 500 % annual (crash guard)


def implied_prob(
    spot: float,
    strike: float,
    time_to_expiry_secs: float,
    sigma_annual: float,
    drift_annual: float = 0.0,
) -> float:
    """Return lognormal P(S_T > strike).

    Optional `drift_annual` shifts the mean-log-return term — useful when
    paired with an order-flow-imbalance signal so the pricer doesn't have
    to assume zero drift over windows where the book is clearly moving.

    Returns 0.5 when time_to_expiry_secs <= 0 (fair coin at expiry
    boundary — caller should not trade this close).
    """
    if time_to_expiry_secs <= 0:
        return 0.5
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")

    sigma_annual = float(
        max(SIGMA_MIN, min(SIGMA_MAX, sigma_annual))
    )
    T = time_to_expiry_secs / (365.25 * 24 * 3600)  # fraction of year
    sqrt_T = math.sqrt(T)
    d = (
        math.log(spot / strike) + (drift_annual - 0.5 * sigma_annual**2) * T
    ) / (sigma_annual * sqrt_T)
    return float(norm.cdf(d))


def taker_fee_per_share(
    price: float,
    fee_rate: float = FEE_RATE_CRYPTO,
    fee_exponent: float = 1.0,
) -> float:
    """Polymarket taker fee in dollars per share at a given price.

    Live Polymarket feeSchedule shape:
        fee = fee_rate × (price × (1-price))^fee_exponent

    price should be between 0 and 1 (exclusive).  With exponent=1 (the
    crypto/politics default) the fee peaks at price=0.5 → fee_rate/4 and
    collapses at the tails.  Higher exponents narrow the fee curve.

    The defaults match the crypto category as-of 2026-05; callers should
    pass the per-market values from PolyMarket.fee_rate / fee_exponent
    so changes in the live schedule propagate without a code change.
    """
    price = max(1e-6, min(1 - 1e-6, price))
    base = price * (1.0 - price)
    if fee_exponent == 1.0:
        return fee_rate * base
    return fee_rate * (base ** fee_exponent)


def realized_vol_annual(log_returns: list[float], window_secs: float) -> float:
    """Convert a list of per-tick log-returns to an annualised σ.

    log_returns should be ln(p_i / p_{i-1}) for recent ticks.
    window_secs is the total elapsed time covered by the returns.
    Returns SIGMA_MIN if the list is too short to compute.
    """
    n = len(log_returns)
    if n < 2 or window_secs <= 0:
        return SIGMA_MIN
    variance_per_sec = sum(r * r for r in log_returns) / (window_secs)
    sigma_annual = math.sqrt(variance_per_sec * 365.25 * 24 * 3600)
    return max(SIGMA_MIN, min(SIGMA_MAX, sigma_annual))
