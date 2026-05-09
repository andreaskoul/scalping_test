"""
Static cross-market arbitrage scanner — strike-monotonicity violations.

Background
----------
Saguillo et al., "Unravelling the Probabilistic Forest: Arbitrage in
Prediction Markets" (arXiv:2508.03474, AFT 2025) classify Polymarket
arbitrage into two types:

  1. Market-Rebalancing Arbitrage — within a single market, the YES
     and NO prices fail to sum to $1.  *Structurally unreachable on
     the Polymarket CLOB*: the matching engine auto-mirrors the books
     (a buy-YES at $0.40 generates a synthetic sell-NO at $0.60), so
     ask_YES + ask_NO is always ≥ 1 by construction.  We do not scan
     for this type.

  2. Combinatorial Arbitrage — across multiple markets via logical
     dependencies.  This is where the persistent edge lives, because
     different maker bots run independently per market and don't
     always cross-link.  The cleanest, model-free instance is:

       **Strike monotonicity** within a same-expiry threshold ladder.

       For two markets at the same (symbol, expiry_ts) with strikes
       K_low < K_high:

           P(S_T > K_low) ≥ P(S_T > K_high)         (true by definition)

       If the observed quotes violate this — specifically if

           ask(K_low) + fee_low + fee_high < bid(K_high)

       — then buying YES at K_low and selling YES at K_high yields a
       net credit upfront with non-negative payoff at expiry under any
       price path:

           payoff = 1   if K_low  ≤  S_T  <  K_high
           payoff = 0   otherwise

       This is a pure (model-free) arbitrage: no σ, IV, or drift
       assumption required.  Profit = bid_high − ask_low − fees − ε.

Why we expect this to work
--------------------------
- Maker bots on Polymarket are run separately per token (different
  inventory accounts, different parameter sets).  They don't always
  re-link instantly across siblings, so brief strike-ladder
  inconsistencies do appear, especially during fast spot moves.
- Less-liquid strikes (far-OTM tails) often have less aggressive
  maker coverage and stale quotes.
- The literature (arXiv:2508.03474, arXiv:2601.01706) documents
  $40M+ extracted via combinatorial arbitrage in 2024-25.

Caveats
-------
- Two-legged execution is required.  In paper mode we always "fill",
  but live mode introduces leg-risk (one leg fills, the other doesn't,
  leaving us with directional exposure).  The scanner emits paired
  Signals tagged with a shared `arb_id`; the live executor would need
  atomic submission via Polymarket's batched-order endpoint.
- The arb has to clear the round-trip taker fee on both sides, which
  for crypto markets at p=0.5 is ~3.5% combined.  We require a hard
  minimum net credit (default 100 bps after fees) to filter noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .poly_universe import PolyMarket
from .poly_ws import PolyWS, BookSnapshot
from .pricing import taker_fee_per_share

log = logging.getLogger(__name__)

# Hard minimum credit after fees — prevents firing on micro-violations
# that get eaten by latency slippage between leg-1 and leg-2 fills.
DEFAULT_MIN_CREDIT: float = 0.01

# Cap notional on each leg.  The arb is symmetric so this is the
# notional on either side, not the gross.
DEFAULT_MAX_NOTIONAL_USD: float = 25.0


@dataclass
class ArbLeg:
    """One side of a multi-leg arbitrage trade."""
    market: PolyMarket
    token_id: str
    side: str                   # "BUY" or "SELL"
    price: float                # best ask (BUY) or best bid (SELL)
    size: float                 # shares
    fee_per_share: float


@dataclass
class StrikeArb:
    """A risk-free strike-monotonicity bull-spread."""
    symbol: str
    expiry_ts: float
    leg_low: ArbLeg             # BUY YES at the lower strike K_low
    leg_high: ArbLeg            # SELL YES at the higher strike K_high
    net_credit: float           # bid_high - ask_low - fees, > 0 to fire
    notional: float             # $ size of each leg
    arb_id: str                 # shared id for matched fills
    note: str = field(default="")


def _ladder(markets: list[PolyMarket]) -> dict[tuple[str, float], list[PolyMarket]]:
    """Group threshold markets by (symbol, expiry_ts), each list sorted
    by strike ascending."""
    out: dict[tuple[str, float], list[PolyMarket]] = {}
    for m in markets:
        if not m.is_threshold or m.strike <= 0 or m.expiry_ts <= 0:
            continue
        out.setdefault((m.symbol, m.expiry_ts), []).append(m)
    for k in out:
        out[k].sort(key=lambda x: x.strike)
    return out


def find_strike_arbs(
    markets: list[PolyMarket],
    poly_ws: PolyWS,
    min_credit: float = DEFAULT_MIN_CREDIT,
    max_notional_usd: float = DEFAULT_MAX_NOTIONAL_USD,
    min_tte_secs: float = 180.0,
    max_tte_secs: float = 3600.0,
) -> list[StrikeArb]:
    """Scan the threshold-market ladder for strike-monotonicity violations.

    Returns one StrikeArb per actionable pair, sized so each leg's
    notional is at most `max_notional_usd` and fits in the available
    book depth on both sides.
    """
    import time as _time
    out: list[StrikeArb] = []
    now_wall = _time.time()

    for (sym, exp_ts), ladder in _ladder(markets).items():
        tte = exp_ts - now_wall
        if tte < min_tte_secs or tte > max_tte_secs:
            continue
        if len(ladder) < 2:
            continue

        # For each pair (low, high) check the bull-spread arb.
        # We only check ADJACENT strikes — a non-adjacent violation
        # implies an adjacent violation somewhere on the ladder, and
        # adjacent pairs have the tightest credit threshold to clear.
        for i in range(len(ladder) - 1):
            m_low = ladder[i]
            m_high = ladder[i + 1]
            if m_low.strike >= m_high.strike:
                continue  # defensive — sorted earlier

            book_low = poly_ws.snapshot(m_low.yes_token_id)
            book_high = poly_ws.snapshot(m_high.yes_token_id)
            if book_low is None or book_high is None:
                continue

            ask_low = book_low.best_ask
            bid_high = book_high.best_bid
            if ask_low <= 0 or ask_low >= 1.0:
                continue
            if bid_high <= 0 or bid_high >= 1.0:
                continue

            fee_low = taker_fee_per_share(
                ask_low,
                fee_rate=getattr(m_low, "fee_rate", 0.07),
                fee_exponent=getattr(m_low, "fee_exponent", 1.0),
            )
            fee_high = taker_fee_per_share(
                bid_high,
                fee_rate=getattr(m_high, "fee_rate", 0.07),
                fee_exponent=getattr(m_high, "fee_exponent", 1.0),
            )

            net_credit = bid_high - ask_low - fee_low - fee_high
            if net_credit < min_credit:
                continue  # within fee-eaten noise; skip

            # Size each leg by the smaller of (notional cap / per-share
            # cost on the BUY leg, notional cap / per-share proceeds on
            # the SELL leg, available depth on each side).
            buy_per_share = ask_low + fee_low
            sell_per_share = bid_high - fee_high
            if buy_per_share <= 0 or sell_per_share <= 0:
                continue
            shares_by_buy_notional = max_notional_usd / buy_per_share
            shares_by_sell_notional = max_notional_usd / sell_per_share
            shares = round(min(
                shares_by_buy_notional,
                shares_by_sell_notional,
                book_low.ask_size,
                book_high.bid_size,
            ), 2)
            if shares < 1.0:
                continue

            arb_id = f"strike-{sym}-{int(exp_ts)}-{i}"
            leg_low = ArbLeg(
                market=m_low,
                token_id=m_low.yes_token_id,
                side="BUY",
                price=ask_low,
                size=shares,
                fee_per_share=fee_low,
            )
            leg_high = ArbLeg(
                market=m_high,
                token_id=m_high.yes_token_id,
                side="SELL",
                price=bid_high,
                size=shares,
                fee_per_share=fee_high,
            )
            out.append(StrikeArb(
                symbol=sym,
                expiry_ts=exp_ts,
                leg_low=leg_low,
                leg_high=leg_high,
                net_credit=net_credit,
                notional=shares * buy_per_share,
                arb_id=arb_id,
                note=(
                    f"BUY K={m_low.strike:.0f}@{ask_low:.4f} / "
                    f"SELL K={m_high.strike:.0f}@{bid_high:.4f} "
                    f"credit={net_credit:.4f}"
                ),
            ))
            log.info(
                "Strike-arb [%s exp=%d]: BUY K=%.0f @%.4f / SELL K=%.0f @%.4f "
                "credit=%.4f size=%.1f",
                sym, int(exp_ts),
                m_low.strike, ask_low, m_high.strike, bid_high,
                net_credit, shares,
            )

    return out
