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

from .pricing import (
    implied_prob,
    taker_fee_per_share,
    maker_rebate_per_share,
    skew_adjusted_sigma,
    wedge_estimate,
    physical_to_risk_neutral,
    apply_calibration,
    FEE_RATE_CRYPTO,
)
from .poly_universe import PolyMarket
from .binance_ws import BinanceTick
from .poly_ws import BookSnapshot
from .microstructure import MicroFeatures, order_book_imbalance
from .sizing import kelly_size

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
    is_maker: bool = False  # post-only resting order → earns rebate, pays no taker fee
    source: str = "model"   # model | meanrev | arb | xvenue — for attribution


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
        # ---- research-driven overlays (all OFF by default for backward
        # compatibility; main.py turns them on from the max-EV config) ----
        longshot_tilt_mult: float = 0.0,      # Portnaya favourite-longshot haircut on BUY
        skew_coef: float = 0.0,               # smile bump for OTM strikes
        sell_price_min: float | None = None,  # re-admit the low tail on SELL only
        micro_engine=None,                    # MicrostructureEngine (OFI/OBI/momentum)
        ml_overlay: bool = False,
        ml_weight: float = 0.0,
        obi_veto: bool = False,
        obi_veto_threshold: float = -0.60,
        kelly_enabled: bool = False,
        kelly_fraction: float = 0.30,
        maker_enabled: bool = False,
        maker_join_ticks: int = 1,
        ml_horizon_secs: float = 300.0,
        nudge_cap: float = 0.10,
        # self-calibration (src.calibrate output); None → paper/module priors
        wedge_coeffs: dict | None = None,
        calib: tuple[float, float] | None = None,
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
        self.longshot_tilt_mult = longshot_tilt_mult
        self.skew_coef = skew_coef
        self.sell_price_min = sell_price_min if sell_price_min is not None else price_min
        self.micro_engine = micro_engine
        self.ml_overlay = ml_overlay
        self.ml_weight = ml_weight
        self.obi_veto = obi_veto
        self.obi_veto_threshold = obi_veto_threshold
        self.kelly_enabled = kelly_enabled
        self.kelly_fraction = kelly_fraction
        self.maker_enabled = maker_enabled
        self.maker_join_ticks = maker_join_ticks
        self.ml_horizon_secs = ml_horizon_secs
        self.nudge_cap = nudge_cap
        self.wedge_coeffs = wedge_coeffs
        self.calib = calib
        self._last_fire: dict[str, float] = {}  # token_id → monotonic ts

    def _sigma_used(self, market: PolyMarket, binance: BinanceTick) -> float:
        sigma_iv = 0.0
        if self.iv_oracle is not None:
            iv_snap = self.iv_oracle.snapshot(binance.symbol)
            if iv_snap is not None:
                sigma_iv = iv_snap.sigma_annual
        sigma_used = max(binance.sigma_annual, sigma_iv, self.sigma_floor)
        if self.skew_coef and market.strike > 0 and binance.mid > 0:
            sigma_used = skew_adjusted_sigma(sigma_used, binance.mid, market.strike, self.skew_coef)
        return sigma_used

    def fair_prob(
        self, market: PolyMarket, binance: BinanceTick, carry_annual: float = 0.0
    ) -> tuple[float, float]:
        """Option-implied terminal probability + the σ used. Shared with main's
        mean-reversion path so D_t is measured against the same fair value."""
        sigma_used = self._sigma_used(market, binance)
        tte = market.expiry_ts - time.time()
        if tte <= 0 or market.strike <= 0 or binance.mid <= 0:
            return 0.5, sigma_used
        p = implied_prob(
            spot=binance.mid, strike=market.strike, time_to_expiry_secs=tte,
            sigma_annual=sigma_used, drift_annual=getattr(binance, "drift_annual", 0.0),
            carry_annual=carry_annual,
        )
        return apply_calibration(p, self.calib), sigma_used

    def evaluate(
        self,
        market: PolyMarket,
        binance: BinanceTick,
        yes_book: BookSnapshot,
        no_book: BookSnapshot | None = None,
        features: MicroFeatures | None = None,
        carry_annual: float = 0.0,
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

        # Up/Down markets are now tradeable once the strike has been anchored
        # to the real Chainlink Price-to-Beat (see resolution.py + main.py);
        # only skip them if the legacy flag is set AND the strike is unanchored.
        if self.skip_updown and market.is_updown:
            return None

        # Skip if strike isn't anchored or spot is missing.
        if market.strike <= 0 or binance.mid <= 0:
            return None

        # σ blend: max of EWMA realised, options-implied (Deribit), and a
        # hard floor, then a smile bump for OTM strikes (Portnaya §7.1 — ATM
        # IV alone understates OTM fair value).
        sigma_used = self._sigma_used(market, binance)

        # Require a recent Polymarket print (the book might have moved
        # several ticks since the snapshot was taken).
        if (now - yes_book.ts) > self.book_max_age_secs:
            return None

        # Fair (option-implied) terminal probability with perp-funding carry.
        p_fair = implied_prob(
            spot=binance.mid,
            strike=market.strike,
            time_to_expiry_secs=time_to_expiry,
            sigma_annual=sigma_used,
            drift_annual=getattr(binance, "drift_annual", 0.0),
            carry_annual=carry_annual,
        )
        # Self-calibrated recalibration of the pricer (src.calibrate), if loaded.
        p_fair = apply_calibration(p_fair, self.calib)

        obi = order_book_imbalance(yes_book.bid_size, yes_book.ask_size)

        # Directional microstructure nudge (OFI/ML + momentum/fade), scaled by
        # threshold sensitivity p(1-p) and capped.  p_fair stays the truth
        # anchor for the wedge; p_star is the nudged trading probability.
        p_star = p_fair
        if self.micro_engine is not None and self.ml_overlay and features is not None:
            p_up = self.micro_engine.model.p_up(features)
            horizon = min(time_to_expiry, self.ml_horizon_secs)
            p_up_rn = physical_to_risk_neutral(p_up, sigma_used, horizon, carry_annual)
            assessment = self.micro_engine.assess(features, obi, side="NEUTRAL")
            eff_dir = 2.0 * (p_up_rn - 0.5) + 0.5 * assessment.direction_bias
            sens = p_fair * (1.0 - p_fair)
            nudge = max(-self.nudge_cap, min(self.nudge_cap, self.ml_weight * eff_dir * sens))
            p_star = min(0.999, max(0.001, p_fair + nudge))

        tte_hours = time_to_expiry / 3600.0

        sig_buy = self._check_leg(market, market.yes_token_id, Side.BUY,
                                  p_star, p_fair, tte_hours, yes_book, sigma_used, obi)
        sig_sell = self._check_leg(market, market.yes_token_id, Side.SELL,
                                   p_star, p_fair, tte_hours, yes_book, sigma_used, obi)

        candidates = [s for s in (sig_buy, sig_sell) if s is not None]
        if not candidates:
            return None
        best = max(candidates, key=lambda s: s.edge)

        # Per-token cooldown applied once, after picking the better leg.
        last = self._last_fire.get(best.token_id, 0.0)
        if now - last < self.cooldown:
            return None
        self._last_fire[best.token_id] = now

        log.info(
            "Signal %s %s %s token=%s px=%.4f p*=%.4f p_fair=%.4f edge=%.4f σ=%.3f size=%.1f",
            "MAKER" if best.is_maker else "TAKER", best.side.value,
            market.question[:48], best.token_id[:12],
            best.price, best.p_star, p_fair, best.edge, best.sigma, best.size,
        )
        return best

    def _check_leg(
        self,
        market: PolyMarket,
        token_id: str,
        side: Side,
        p_star: float,
        p_fair: float,
        tte_hours: float,
        book: BookSnapshot,
        sigma: float,
        obi: float,
    ) -> Signal | None:
        # ---- OBI toxicity veto (adverse-selection gate) ----
        if self.obi_veto:
            if side == Side.BUY and obi < self.obi_veto_threshold:
                return None
            if side == Side.SELL and obi > -self.obi_veto_threshold:
                return None

        if side == Side.BUY:
            top_price, top_size = book.best_ask, book.ask_size
            price_floor = self.price_min
        else:
            top_price, top_size = book.best_bid, book.bid_size
            price_floor = self.sell_price_min   # re-admit the longshot tail on SELL
        if top_price <= 0 or top_price < price_floor or top_price > self.price_max:
            return None

        # ---- Favourite-longshot wedge haircut (Portnaya Table 5; fitted coeffs
        # from src.calibrate when available, else the paper prior) ----
        if not self.longshot_tilt_mult:
            wedge = 0.0
        elif self.wedge_coeffs:
            wedge = wedge_estimate(
                p_fair, tte_hours,
                self.wedge_coeffs["intercept"],
                self.wedge_coeffs["beta_pfair"],
                self.wedge_coeffs["beta_tte_hr"],
            )
        else:
            wedge = wedge_estimate(p_fair, tte_hours)
        buy_pen = self.longshot_tilt_mult * max(0.0, wedge)
        sell_pen = self.longshot_tilt_mult * max(0.0, -wedge)
        quoted_spread = max(0.0, book.best_ask - book.best_bid)
        eff_spread_extra = (self.effective_spread_mult - 1.0) * quoted_spread

        candidates: list[tuple[float, float, float, bool]] = []  # edge, px, size, is_maker

        # ---- Taker variant (cross the book, pay fee + effective spread) ----
        desired = self.max_notional / top_price
        exec_price = None
        size = 0.0
        if top_size > 0 and desired <= top_size:
            exec_price, size = top_price, round(min(desired, top_size), 2)
        elif self.poly_ws is not None:
            vwap, available = self.poly_ws.walk_book(token_id, side.value, desired)
            if vwap > 0 and available > 0 and abs(vwap - top_price) / top_price <= self.max_walk_slippage:
                exec_price, size = vwap, round(min(desired, available), 2)
        else:
            exec_price = top_price
            size = round(min(desired, top_size if top_size > 0 else desired), 2)
        if exec_price is not None and size >= 1.0:
            fee = taker_fee_per_share(
                exec_price, getattr(market, "fee_rate", self.fee_rate),
                getattr(market, "fee_exponent", 1.0),
            )
            if side == Side.BUY:
                edge = p_star - exec_price - fee - eff_spread_extra - self.safety_eps - buy_pen
            else:
                edge = exec_price - p_star - fee - eff_spread_extra - self.safety_eps - sell_pen
            candidates.append((edge, exec_price, size, False))

        # ---- Maker variant (post inside the spread, earn rebate, no taker fee) ----
        if self.maker_enabled:
            tick = getattr(market, "tick_size", 0.01) or 0.01
            if side == Side.BUY:
                mprice = round(book.best_bid + self.maker_join_ticks * tick, 4)
                if mprice >= book.best_ask:
                    mprice = book.best_bid
            else:
                mprice = round(book.best_ask - self.maker_join_ticks * tick, 4)
                if mprice <= book.best_bid:
                    mprice = book.best_ask
            if mprice > 0 and price_floor <= mprice <= self.price_max:
                rebate = maker_rebate_per_share(mprice)
                msize = round(self.max_notional / mprice, 2)
                if msize >= 1.0:
                    if side == Side.BUY:
                        medge = p_star - mprice + rebate - self.safety_eps - buy_pen
                    else:
                        medge = mprice - p_star + rebate - self.safety_eps - sell_pen
                    candidates.append((medge, mprice, msize, True))

        if not candidates:
            return None
        edge, exec_price, size, is_maker = max(candidates, key=lambda c: c[0])
        if edge <= 0:
            return None

        # ---- Fractional-Kelly cap on size ----
        if self.kelly_enabled:
            ks = kelly_size(p_star, exec_price, side.value, self.max_notional, self.kelly_fraction)
            if ks <= 0:
                return None
            size = min(size, ks)
        if size < 1.0:
            return None

        return Signal(
            market=market, token_id=token_id, side=side,
            price=exec_price, size=round(size, 2),
            p_star=p_star, edge=edge, sigma=sigma,
            is_maker=is_maker, source="model",
        )
