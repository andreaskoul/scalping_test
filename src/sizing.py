"""
Fractional-Kelly position sizing for binary outcomes.

The existing sizer caps by notional and book depth only — it ignores the
*edge* and its *variance*, so a 0.5%-edge fill and a 5%-edge fill get the
same size.  Kelly scales stake with edge and shrinks it as the outcome
approaches a coin flip (max variance), which is exactly where binary bets
are most dangerous.

For a binary priced at `price` that we believe is worth `p_true`:

    BUY  (long YES): win (1-price) w.p. p_true, lose price w.p. (1-p_true)
    SELL (short YES): win price    w.p. (1-p_true), lose (1-price) w.p. p_true

The Kelly fraction of *bankroll-at-risk* is f* = edge / odds, clipped to
[0, 1] and then multiplied by `kelly_fraction` (fractional Kelly, default
0.30) for drawdown control.  We translate that into a share count against
the per-trade notional cap so it never exceeds existing risk limits.
"""

from __future__ import annotations


def kelly_fraction_binary(p_true: float, price: float, side: str) -> float:
    """Full-Kelly fraction of risk capital for a binary bet. 0 if no edge."""
    p_true = max(1e-6, min(1.0 - 1e-6, p_true))
    price = max(1e-6, min(1.0 - 1e-6, price))
    if side.upper() == "BUY":
        # profit if YES resolves: gain (1-price)/price per $ at risk
        b = (1.0 - price) / price
        p = p_true
    else:  # SELL YES == buy NO at (1-price)
        b = price / (1.0 - price)
        p = 1.0 - p_true
    # Kelly: f* = p - (1-p)/b  = (p*b - (1-p)) / b
    f = (p * b - (1.0 - p)) / b
    return max(0.0, min(1.0, f))


def kelly_size(
    p_true: float,
    price: float,
    side: str,
    max_notional: float,
    kelly_fraction: float = 0.30,
    min_shares: float = 1.0,
) -> float:
    """Shares to trade under fractional Kelly, capped by `max_notional`.

    Returns 0.0 when there is no Kelly edge (caller should then skip even
    if the raw edge filter passed — they can disagree at the margin).
    """
    if price <= 0:
        return 0.0
    f = kelly_fraction_binary(p_true, price, side) * max(0.0, kelly_fraction)
    if f <= 0:
        return 0.0
    notional = f * max_notional
    shares = notional / price
    if shares < min_shares:
        return 0.0
    return round(shares, 2)
