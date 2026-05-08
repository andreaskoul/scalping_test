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
"""

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
        safety_eps: float = 0.003,
        cooldown_secs: float = 1.0,
        fee_rate: float = FEE_RATE_CRYPTO,
    ):
        self.max_notional = max_notional_per_trade
        self.safety_eps = safety_eps
        self.cooldown = cooldown_secs
        self.fee_rate = fee_rate
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

        # Don't trade in the last 2 minutes — model degeneracy near expiry
        if time_to_expiry < 120:
            return None

        # Skip if Binance σ hasn't warmed up — fewer than 5 trades in window
        from .pricing import SIGMA_MIN
        if binance.sigma_annual <= SIGMA_MIN:
            return None

        # Skip if strike isn't yet anchored (Up/Down markets need spot snapshot).
        if market.strike <= 0 or binance.mid <= 0:
            return None

        p_star = implied_prob(
            spot=binance.mid,
            strike=market.strike,
            time_to_expiry_secs=time_to_expiry,
            sigma_annual=binance.sigma_annual,
        )

        # --- try to BUY the YES (Up) token ---
        signal = self._check_leg(
            market=market,
            token_id=market.yes_token_id,
            side=Side.BUY,
            p_star=p_star,
            book=yes_book,
            now=now,
            sigma=binance.sigma_annual,
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
            sigma=binance.sigma_annual,
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
            fee = taker_fee_per_share(exec_price, self.fee_rate)
            edge = p_star - exec_price - fee - self.safety_eps
        else:
            exec_price = book.best_bid
            available_size = book.bid_size
            fee = taker_fee_per_share(exec_price, self.fee_rate)
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
