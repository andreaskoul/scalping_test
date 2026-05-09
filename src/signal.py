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


class SignalGenerator:
    def __init__(
        self,
        max_notional_per_trade: float = 25.0,
        safety_eps: float = 0.02,           # 200bps cushion (was 30bps)
        cooldown_secs: float = 5.0,         # raised from 1s to thin the firehose
        fee_rate: float = FEE_RATE_CRYPTO,
        price_min: float = 0.10,            # skip 0.01 tails (huge downside)
        price_max: float = 0.90,
        sigma_floor: float = 0.40,          # crypto realised vol baseline
        skip_updown: bool = True,           # strike anchoring is unreliable
        min_tte_secs: float = 180.0,        # 3 min — fee dominates closer than that
        max_tte_secs: float = 3600.0,       # 1 hour — pre-listed markets are noise
        book_max_age_secs: float = 2.0,     # require fresh book to cross
        iv_oracle=None,                     # optional .snapshot(symbol) -> IVSnapshot
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
        if self.skip_updown and ("up or down" in market.question.lower()):
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
            exec_price = book.best_ask
            available_size = book.ask_size
        else:
            exec_price = book.best_bid
            available_size = book.bid_size

        # Tail filter: never sell at 0.01 or buy at 0.99. Tiny credit, $1 downside.
        if exec_price < self.price_min or exec_price > self.price_max:
            return None

        fee = taker_fee_per_share(exec_price, self.fee_rate)
        if side == Side.BUY:
            edge = p_star - exec_price - fee - self.safety_eps
        else:
            edge = exec_price - p_star - fee - self.safety_eps

        if edge <= 0:
            return None

        # Cooldown gate
        last = self._last_fire.get(token_id, 0.0)
        if now - last < self.cooldown:
            return None

        # Size
        if exec_price <= 0:
            return None
        max_shares = self.max_notional / exec_price
        size = round(min(max_shares, available_size if available_size > 0 else max_shares), 2)
        if size < 1.0:
            return None

        self._last_fire[token_id] = now
        log.info(
            "Signal %s %s token=%s price=%.4f p*=%.4f edge=%.4f sigma=%.3f",
            side.value, market.question[:60], token_id[:12], exec_price, p_star, edge, sigma,
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
