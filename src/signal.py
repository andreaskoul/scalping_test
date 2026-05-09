"""
Signal generation — computes the Binance-implied probability for each
active Polymarket market and decides whether to fire a FAK/IOC order.

Edge formula (taker):
  edge_buy  = p* − best_ask − taker_fee(best_ask) − safety_eps
  edge_sell = best_bid − p* − taker_fee(best_bid) − safety_eps

Only fires when edge > 0 AND per-token cooldown has elapsed.

Sizing:
  size = min(
    max_notional_per_trade / price,   # notional cap per order
    depth_at_top,                     # available size at top of book
    kelly_size,                       # fractional Kelly (optional)
  )

Risk filters (all on by default; tunable via env vars):
  - PRICE_MIN/PRICE_MAX  — never trade tails (fee≈0, but $1.00 downside).
  - SIGMA_FLOOR          — clip realised σ up to a sane crypto baseline
                           (60s rolling vol systematically under-prices
                            tail risk on short windows).
  - SKIP_UPDOWN          — Up/Down markets have their strike anchored to
                           spot at *first observation*; for pre-listed
                           windows this is an entirely synthetic strike
                           uncorrelated with the actual resolution price.
  - MIN_TTE_SECS         — don't trade within N min of expiry.
  - MAX_TTE_SECS         — don't trade pre-listed markets > N hours away.
  - REQUIRE_FRESH_BOOK   — skip if Polymarket book older than `book_max_age`.
"""

import os
import time
import logging
from dataclasses import dataclass
from enum import Enum

from .pricing import implied_prob, taker_fee_per_share, FEE_RATE_CRYPTO
from .poly_universe import PolyMarket
from .binance_ws import BinanceTick
from .poly_ws import BookSnapshot

log = logging.getLogger(__name__)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Signal:
    market: PolyMarket
    token_id: str           # YES (Up) or NO (Down)
    side: Side              # BUY = take the ask; SELL = hit the bid
    price: float            # limit price to send
    size: float             # shares
    p_star: float           # Binance-implied probability
    edge: float             # edge in probability points (after fee)
    sigma: float
    arb_id: str = ""        # set on multi-leg static-arb signals so PnL
                            # accounting can pair the legs


class SignalGenerator:
    def __init__(
        self,
        max_notional_per_trade: float = 25.0,
        safety_eps: float = 0.005,           # base cushion for unmodeled costs (gas, etc.)
        cooldown_secs: float = 5.0,
        fee_rate: float = FEE_RATE_CRYPTO,   # default; per-market rate overrides
        price_min: float = 0.10,
        price_max: float = 0.90,
        sigma_floor: float = 0.40,
        skip_updown: bool = True,
        min_tte_secs: float = 180.0,
        max_tte_secs: float = 3600.0,
        book_max_age_secs: float = 2.0,
        iv_oracle=None,
        poly_ws=None,                        # for walk_book() VWAP lookup
        # Stoll/Huang-Stoll: effective spread is ~1.2-1.5× quoted spread
        # because of fleeting quotes, hidden liquidity, and adverse
        # selection. Default 1.3× is the median retail-venue estimate.
        effective_spread_mult: float = 1.3,
        # Cap on how far we'll walk the book before rejecting the trade,
        # measured as |VWAP - best| / best. 5% means a $0.50 ask can cost
        # at most $0.525 effective; beyond that we don't want the trade.
        max_walk_slippage: float = 0.05,
    ):
        self.max_notional = max_notional_per_trade
        self.safety_eps = safety_eps
        self.cooldown = cooldown_secs
        self.fee_rate = fee_rate
        self.price_min = price_min
        self.price_max = price_max
        self.sigma_floor = sigma_floor
        self.skip_updown = skip_updown
        self.min_tte_secs = min_tte_secs
        self.max_tte_secs = max_tte_secs
        self.book_max_age_secs = book_max_age_secs
        self.iv_oracle = iv_oracle
        self.poly_ws = poly_ws
        self.effective_spread_mult = effective_spread_mult
        self.max_walk_slippage = max_walk_slippage
        self._last_fire: dict[str, float] = {}  # token_id → monotonic ts

    def evaluate(
        self,
        market: PolyMarket,
        binance: BinanceTick,
        yes_book: BookSnapshot,
        no_book: BookSnapshot | None = None,
    ) -> Signal | None:
        """Return a Signal if an edge exists, else None."""
        now = time.monotonic()
        time_to_expiry = market.expiry_ts - time.time()

        # tte gates — skip near-expiry (model degeneracy + huge fee/edge ratio)
        # and far-expiry (pre-listed Up/Down or threshold markets we can't price).
        if time_to_expiry < self.min_tte_secs:
            return None
        if time_to_expiry > self.max_tte_secs:
            return None

        # Skip Up/Down markets — their strike is set at the official window-
        # open time on Polymarket, not at our first observation. Anchoring
        # to spot at first observation produces a synthetic strike that
        # systematically biases p* away from 0.5 on pre-listed markets.
        # Flag is precomputed at universe-load time → no per-tick string ops.
        if self.skip_updown and market.is_updown:
            return None

        # Skip if strike isn't anchored or spot is missing.
        if market.strike <= 0 or binance.mid <= 0:
            return None

        # σ blend: max of EWMA realised, options-implied (Deribit), and a
        # hard floor.  IV is the forward-vol consensus the maker bots use;
        # using only realised would put us at a structural info disadvantage
        # exactly when realised undershoots IV (quiet body of distribution
        # masking real tail risk).
        sigma_iv = 0.0
        if self.iv_oracle is not None:
            iv_snap = self.iv_oracle.snapshot(binance.symbol)
            if iv_snap is not None:
                sigma_iv = iv_snap.sigma_annual
        sigma_used = max(binance.sigma_annual, sigma_iv, self.sigma_floor)

        # Require a recent Polymarket print (the book might have moved
        # several ticks since the snapshot was taken).
        if (now - yes_book.ts) > self.book_max_age_secs:
            return None

        p_star = implied_prob(
            spot=binance.mid,
            strike=market.strike,
            time_to_expiry_secs=time_to_expiry,
            sigma_annual=sigma_used,
            drift_annual=getattr(binance, "drift_annual", 0.0),
        )

        # --- try to BUY the YES (Up) token ---
        signal = self._check_leg(
            market=market,
            token_id=market.yes_token_id,
            side=Side.BUY,
            p_star=p_star,
            book=yes_book,
            now=now,
            sigma=sigma_used,
        )
        if signal:
            return signal

        # --- try to SELL the YES token (equivalent to buying NO) ---
        signal = self._check_leg(
            market=market,
            token_id=market.yes_token_id,
            side=Side.SELL,
            p_star=p_star,
            book=yes_book,
            now=now,
            sigma=sigma_used,
        )
        return signal

    def _check_leg(
        self,
        market: PolyMarket,
        token_id: str,
        side: Side,
        p_star: float,
        book: BookSnapshot,
        now: float,
        sigma: float,
    ) -> Signal | None:
        if side == Side.BUY:
            top_price = book.best_ask
            top_size = book.ask_size
        else:
            top_price = book.best_bid
            top_size = book.bid_size

        # Tail filter: never sell at 0.01 or buy at 0.99. Tiny credit, $1 downside.
        if top_price < self.price_min or top_price > self.price_max:
            return None

        # ---- Sizing + walk-the-book VWAP ----
        # If the desired size exceeds top-of-book depth we have to consume
        # multiple levels, paying VWAP rather than the displayed best.
        # The previous code silently assumed the whole order filled at
        # the top — a Stoll-effective-spread error that hid 1-5% of cost
        # on thin books.
        if top_price <= 0:
            return None
        desired_shares = self.max_notional / top_price
        if top_size > 0 and desired_shares <= top_size:
            # Common case — fully fills at the top, VWAP == best price.
            exec_price = top_price
            size = round(min(desired_shares, top_size), 2)
        elif self.poly_ws is not None:
            vwap, available = self.poly_ws.walk_book(
                token_id, side.value, desired_shares,
            )
            if vwap <= 0 or available <= 0:
                return None
            # Reject if walking the book would cost more than the cap.
            slippage = abs(vwap - top_price) / top_price
            if slippage > self.max_walk_slippage:
                return None
            exec_price = vwap
            size = round(min(desired_shares, available), 2)
        else:
            # No level access — fall back to top-only (legacy behaviour).
            exec_price = top_price
            size = round(min(desired_shares, top_size if top_size > 0 else desired_shares), 2)

        if size < 1.0:
            return None

        # ---- Cost model ----
        # 1) Per-market parabolic taker fee.
        fee = taker_fee_per_share(
            exec_price,
            fee_rate=getattr(market, "fee_rate", self.fee_rate),
            fee_exponent=getattr(market, "fee_exponent", 1.0),
        )

        # 2) Effective-spread cushion (Stoll 1989 / Huang-Stoll 1997).
        # Quoted spread is already paid implicitly by crossing best bid/
        # ask; effective spread is typically 1.2-1.5× larger because of
        # fleeting quotes, hidden liquidity, and adverse selection. We
        # charge the "extra" portion as an explicit cushion so the same
        # safety_eps doesn't have to mean different things on tight vs
        # wide books.
        quoted_spread = max(0.0, book.best_ask - book.best_bid)
        eff_spread_extra = (self.effective_spread_mult - 1.0) * quoted_spread

        if side == Side.BUY:
            edge = p_star - exec_price - fee - eff_spread_extra - self.safety_eps
        else:
            edge = exec_price - p_star - fee - eff_spread_extra - self.safety_eps

        if edge <= 0:
            return None

        # Cooldown gate
        last = self._last_fire.get(token_id, 0.0)
        if now - last < self.cooldown:
            return None

        self._last_fire[token_id] = now
        log.info(
            "Signal %s %s token=%s px=%.4f (top=%.4f) p*=%.4f edge=%.4f sigma=%.3f size=%.1f",
            side.value, market.question[:60], token_id[:12],
            exec_price, top_price, p_star, edge, sigma, size,
        )
        return Signal(
            market=market,
            token_id=token_id,
            side=side,
            price=exec_price,
            size=size,
            p_star=p_star,
            edge=edge,
            sigma=sigma,
        )
