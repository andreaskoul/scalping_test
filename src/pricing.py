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
    carry_annual: float = 0.0,
    discount_rate: float = 0.0,
) -> float:
    """Return lognormal risk-neutral P(S_T > strike).

    Optional `drift_annual` shifts the mean-log-return term — useful when
    paired with an order-flow-imbalance signal so the pricer doesn't have
    to assume zero drift over windows where the book is clearly moving.

    `carry_annual` is the risk-neutral carry/cost-of-carry term r.  For
    crypto the relevant r is not zero — it is tied to the perpetual
    funding rate / basis (Portnaya 2026 inverts Binance options at the
    exchange r).  `drift_annual` and `carry_annual` are summed into the
    d2 numerator so an OFI tilt and a funding carry can coexist.

    `discount_rate` applies the e^{-rτ} discount so the result is a
    *price* rather than a bare probability.  Default 0 keeps the
    probability interpretation (the discount is negligible at the
    minute-to-hour horizons where the arb fires, but matters on the
    hourly/daily threshold ladder).

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
    mu = drift_annual + carry_annual
    d = (
        math.log(spot / strike) + (mu - 0.5 * sigma_annual**2) * T
    ) / (sigma_annual * sqrt_T)
    p = float(norm.cdf(d))
    if discount_rate:
        p *= math.exp(-discount_rate * T)
    return p


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


# --------------------------------------------------------------------------- #
#  Maker rebate (CLOB v2, 2026)                                                #
# --------------------------------------------------------------------------- #
# Polymarket charges takers and *pays* makers.  Resting limit orders that get
# filled accrue a daily-settled USDC rebate funded by taker fees (~20% of
# collected crypto-market taker fees) plus the CLOB-v2 liquidity-rewards
# program for quotes resting near the midpoint.  Modelled as a per-share
# credit so the maker path can earn what the taker path pays.
MAKER_REBATE_RATE: float = 0.0125   # per-share credit at the parabolic peak


def maker_rebate_per_share(
    price: float,
    rebate_rate: float = MAKER_REBATE_RATE,
) -> float:
    """Per-share maker rebate (positive = credit to us) at a given price.

    Mirrors the parabolic shape of the taker fee so the rebate is largest
    mid-book and collapses at the tails, matching the observed schedule.
    Verify the live magnitude against docs.polymarket.com/changelog — the
    rate has moved several times in 2026; this is intentionally a knob.
    """
    price = max(1e-6, min(1 - 1e-6, price))
    return rebate_rate * price * (1.0 - price)


# --------------------------------------------------------------------------- #
#  Volatility skew (Portnaya 2026 §7.1: BTC smile is left-skewed)             #
# --------------------------------------------------------------------------- #
def skew_adjusted_sigma(
    sigma_atm: float,
    spot: float,
    strike: float,
    skew_coef: float = 0.0,
) -> float:
    """Bump ATM σ toward the smile for off-the-money strikes.

    BTC implied-vol smiles are left-skewed: low strikes (downside) carry
    higher IV.  Using ATM IV alone *understates* OTM fair value, which —
    per Portnaya — biases the measured Polymarket-vs-fair gap conservatively.
    A positive `skew_coef` raises σ for low strikes (K<S) and lowers it for
    high strikes, linear in log-moneyness m = ln(K/S).

    skew_coef = 0 (default) reproduces the flat-σ behaviour exactly.
    """
    if sigma_atm <= 0 or spot <= 0 or strike <= 0 or skew_coef == 0.0:
        return sigma_atm
    m = math.log(strike / spot)              # <0 for downside strikes
    adj = sigma_atm * (1.0 - skew_coef * m)  # m<0 → raises σ
    return max(SIGMA_MIN, min(SIGMA_MAX, adj))


# --------------------------------------------------------------------------- #
#  Favourite–longshot wedge (Portnaya 2026, Table 5)                          #
# --------------------------------------------------------------------------- #
# Empirically, Polymarket YES prices sit *above* the option-implied
# Black–Scholes value, by a gap D = P_poly - P_fair that is:
#   - largest at LOW fair probability (β on P_fair = -0.398),
#   - increasing in time-to-expiry (β = +0.0008 / hour),
#   - ~+6.3pp on average across the pooled BTC panel.
# The intercept below is set so the wedge ≈ +0.06 near the sample-mean
# fair prob (~0.35) and crosses zero around p≈0.5, turning mildly negative
# for high-prob YES (i.e. favourites are if anything slightly cheap).
WEDGE_INTERCEPT: float = 0.1945
WEDGE_BETA_PFAIR: float = -0.398
WEDGE_BETA_TTE_HR: float = 0.0008
WEDGE_CLAMP: float = 0.20      # cap |wedge| so a single term can't dominate


def load_wedge_coeffs(path: str = "wedge_coeffs.json") -> dict | None:
    """Load self-calibrated wedge coefficients (src.calibrate output), or None.

    The bot hot-loads these at startup so the favourite-longshot tilt is fit to
    your *own* resolved fills rather than the 2023-paper prior."""
    import json, os
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return {
            "intercept": float(d["intercept"]),
            "beta_pfair": float(d["beta_pfair"]),
            "beta_tte_hr": float(d["beta_tte_hr"]),
        }
    except Exception:
        return None


def load_calibration(path: str = "calib_coeffs.json") -> tuple[float, float] | None:
    """Load a linear recalibration (a, b) for p* → P(outcome), or None."""
    import json, os
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        return float(d["a"]), float(d["b"])
    except Exception:
        return None


def apply_calibration(p: float, ab: tuple[float, float] | None) -> float:
    """Recalibrate a probability with a fitted linear map, clamped to (0,1)."""
    if ab is None:
        return p
    a, b = ab
    return max(0.001, min(0.999, a + b * p))


def wedge_estimate(
    p_fair: float,
    tte_hours: float,
    intercept: float = WEDGE_INTERCEPT,
    beta_pfair: float = WEDGE_BETA_PFAIR,
    beta_tte_hr: float = WEDGE_BETA_TTE_HR,
    clamp: float = WEDGE_CLAMP,
) -> float:
    """Expected (P_poly - P_fair) favourite-longshot wedge in prob points.

    Positive → the market is expected to trade *rich* vs our fair p* (so a
    BUY-YES at the ask is fighting the wedge and should be discounted; a
    SELL-YES at the bid is riding it).  Used as an asymmetric edge haircut,
    not as a fair-value override.
    """
    w = intercept + beta_pfair * p_fair + beta_tte_hr * max(0.0, tte_hours)
    return max(-clamp, min(clamp, w))


# --------------------------------------------------------------------------- #
#  Physical → risk-neutral conversion (Deep et al. 2025, minimal-martingale)  #
# --------------------------------------------------------------------------- #
def physical_to_risk_neutral(
    p_phys: float,
    sigma_annual: float,
    horizon_secs: float,
    carry_annual: float = 0.0,
) -> float:
    """Map an ML/physical up-probability to a risk-neutral one.

    A directional classifier outputs a *physical* P(up); Deep et al. show
    physical and risk-neutral probabilities differ by ~21.7% on average, so
    trading the raw classifier output mis-sizes everything.  Using a one-
    step binomial with u = e^{σ√Δt}, d = 1/u and the minimal-martingale
    measure p* = (e^{rΔt} - d)/(u - d), we blend the physical signal toward
    p* by the same √Δt weight that governs how much a single step can move:

        p_rn = w·p* + (1-w)·p_phys ,   w = clip(σ√Δt · k)

    so over very short horizons (tiny √Δt) the physical signal dominates,
    and as the horizon grows the no-arbitrage anchor takes over.
    """
    p_phys = max(0.0, min(1.0, p_phys))
    if horizon_secs <= 0 or sigma_annual <= 0:
        return p_phys
    dt = horizon_secs / (365.25 * 24 * 3600)
    sqrt_dt = math.sqrt(dt)
    u = math.exp(sigma_annual * sqrt_dt)
    d = 1.0 / u
    if u == d:
        return p_phys
    p_star = (math.exp(carry_annual * dt) - d) / (u - d)
    p_star = max(0.0, min(1.0, p_star))
    w = max(0.0, min(1.0, sigma_annual * sqrt_dt * 4.0))
    return w * p_star + (1.0 - w) * p_phys
