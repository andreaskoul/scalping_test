"""
Central settings — the single source of truth for every knob in the bot.

All modules read their parameters from a `Settings` instance built here, so
the "final assembly" lives in one place.  Defaults below are the **max-EV
profile** distilled from the research pass (Portnaya 2026 favourite-longshot
wedge; Deep et al. 2025 OFI-dominant microstructure; Polymarket CLOB-v2 maker
rebates; Chainlink settlement).  Every value is overridable via env so the
profile can be detuned without touching code.

Load once at startup:

    from .config import Settings
    cfg = Settings.from_env()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _list(name: str, default: str) -> list[str]:
    return [s.strip().lower() for s in os.getenv(name, default).split(",") if s.strip()]


@dataclass
class Settings:
    # ---- venue / universe ----
    binance_symbols: list[str] = field(default_factory=lambda: _list("BINANCE_SYMBOLS", "btcusdt,ethusdt"))
    min_tte_secs: float = 0.0
    max_tte_secs: float = 0.0
    book_max_age_secs: float = 0.0

    # ---- sizing / risk ----
    max_notional_per_trade: float = 0.0
    max_notional_per_minute: float = 0.0
    daily_drawdown_stop: float = 0.0
    cooldown_secs: float = 0.0
    binance_stale_secs: float = 0.0
    poly_stale_secs: float = 0.0

    # ---- base edge model ----
    safety_eps: float = 0.0
    sigma_floor: float = 0.0
    effective_spread_mult: float = 0.0
    max_walk_slippage: float = 0.0
    # taker price band (buy side); sell side gets a wider low tail (longshot edge)
    price_min: float = 0.0
    price_max: float = 0.0
    sell_price_min: float = 0.0      # re-admit the low tail on SELL only

    # ---- favourite-longshot tilt (Portnaya) ----
    longshot_tilt_mult: float = 0.0   # 0 disables; 1.0 = full wedge haircut on BUY
    carry_from_funding: bool = True   # feed perp funding into pricing carry r
    skew_coef: float = 0.0            # smile bump for OTM strikes

    # ---- resolution-source awareness ----
    use_price_to_beat: bool = True    # anchor Up/Down strike to Chainlink open
    chainlink_basis_adj: bool = True  # price against settlement feed, not raw Binance

    # ---- microstructure / OFI / ML overlay ----
    obi_veto: bool = True
    obi_veto_threshold: float = 0.0   # block taker entries into toxic flow below this
    ml_overlay: bool = True           # blend directional model into p*
    ml_weight: float = 0.0            # max weight given to the ML tilt
    momentum_enabled: bool = True     # crowd-momentum continuation
    momentum_min_move_usd: float = 0.0
    momentum_window_secs: float = 0.0
    impulse_fade_enabled: bool = True

    # ---- mean-reversion overlay (Portnaya 4h half-life) ----
    meanrev_enabled: bool = True
    meanrev_band: float = 0.0         # |D - mean(D)| > band to fire
    meanrev_half_life_secs: float = 0.0
    meanrev_max_hold_secs: float = 0.0

    # ---- maker execution ----
    maker_enabled: bool = True        # post-only path earns rebate instead of paying fee
    maker_rebate_rate: float = 0.0
    maker_join_ticks: int = 0         # how many ticks inside the spread to post
    maker_gtd_secs: float = 0.0

    # ---- cross-venue arb (Polymarket <-> Kalshi) ----
    xvenue_enabled: bool = False      # off by default — needs Kalshi creds for live
    xvenue_min_credit: float = 0.0
    kalshi_api_base: str = ""

    # ---- static / combinatorial arb ----
    arb_enabled: bool = True
    arb_min_credit: float = 0.0
    rebalance_arb_enabled: bool = True   # YES+NO < 1 within a market
    bucket_arb_enabled: bool = True      # mutually-exclusive multi-outcome sum

    # ---- kelly ----
    kelly_enabled: bool = True
    kelly_fraction: float = 0.0          # fraction of full Kelly

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            binance_symbols=_list("BINANCE_SYMBOLS", "btcusdt,ethusdt"),
            min_tte_secs=_f("MIN_TTE_SECS", 60.0),       # max-EV: trade closer to expiry
            max_tte_secs=_f("MAX_TTE_SECS", 86400.0),
            book_max_age_secs=_f("BOOK_MAX_AGE_SECS", 2.0),

            max_notional_per_trade=_f("MAX_NOTIONAL_PER_TRADE", 25.0),
            max_notional_per_minute=_f("MAX_NOTIONAL_PER_MINUTE", 200.0),
            daily_drawdown_stop=_f("DAILY_DRAWDOWN_STOP", 500.0),
            cooldown_secs=_f("COOLDOWN_SECS", 5.0),
            binance_stale_secs=_f("BINANCE_STALE_SECS", 2.0),
            poly_stale_secs=_f("POLY_STALE_SECS", 5.0),

            safety_eps=_f("EDGE_SAFETY_EPS", 0.012),
            sigma_floor=_f("SIGMA_FLOOR", 0.40),
            effective_spread_mult=_f("EFFECTIVE_SPREAD_MULT", 1.3),
            max_walk_slippage=_f("MAX_WALK_SLIPPAGE", 0.05),
            price_min=_f("PRICE_MIN", 0.10),
            price_max=_f("PRICE_MAX", 0.90),
            sell_price_min=_f("SELL_PRICE_MIN", 0.03),

            longshot_tilt_mult=_f("LONGSHOT_TILT_MULT", 1.0),
            carry_from_funding=_b("CARRY_FROM_FUNDING", True),
            skew_coef=_f("SKEW_COEF", 0.15),

            use_price_to_beat=_b("USE_PRICE_TO_BEAT", True),
            chainlink_basis_adj=_b("CHAINLINK_BASIS_ADJ", True),

            obi_veto=_b("OBI_VETO", True),
            obi_veto_threshold=_f("OBI_VETO_THRESHOLD", -0.60),
            ml_overlay=_b("ML_OVERLAY", True),
            ml_weight=_f("ML_WEIGHT", 0.25),
            momentum_enabled=_b("MOMENTUM_ENABLED", True),
            momentum_min_move_usd=_f("MOMENTUM_MIN_MOVE_USD", 60.0),
            momentum_window_secs=_f("MOMENTUM_WINDOW_SECS", 180.0),
            impulse_fade_enabled=_b("IMPULSE_FADE_ENABLED", True),

            meanrev_enabled=_b("MEANREV_ENABLED", True),
            meanrev_band=_f("MEANREV_BAND", 0.05),
            meanrev_half_life_secs=_f("MEANREV_HALF_LIFE_SECS", 14400.0),
            meanrev_max_hold_secs=_f("MEANREV_MAX_HOLD_SECS", 12600.0),

            maker_enabled=_b("MAKER_ENABLED", True),
            maker_rebate_rate=_f("MAKER_REBATE_RATE", 0.0125),
            maker_join_ticks=_i("MAKER_JOIN_TICKS", 1),
            maker_gtd_secs=_f("MAKER_GTD_SECS", 12.0),

            xvenue_enabled=_b("XVENUE_ENABLED", False),
            xvenue_min_credit=_f("XVENUE_MIN_CREDIT", 0.01),
            kalshi_api_base=os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2"),

            arb_enabled=_b("ARB_ENABLED", True),
            arb_min_credit=_f("ARB_MIN_CREDIT", 0.01),
            rebalance_arb_enabled=_b("REBALANCE_ARB_ENABLED", True),
            bucket_arb_enabled=_b("BUCKET_ARB_ENABLED", True),

            kelly_enabled=_b("KELLY_ENABLED", True),
            kelly_fraction=_f("KELLY_FRACTION", 0.30),
        )

    def summary(self) -> str:
        on = lambda b: "on" if b else "off"
        return (
            f"longshot_tilt={self.longshot_tilt_mult:.2f} maker={on(self.maker_enabled)} "
            f"ml={on(self.ml_overlay)}({self.ml_weight:.2f}) obi_veto={on(self.obi_veto)}"
            f"({self.obi_veto_threshold:+.2f}) momentum={on(self.momentum_enabled)} "
            f"meanrev={on(self.meanrev_enabled)} kelly={on(self.kelly_enabled)}"
            f"({self.kelly_fraction:.2f}) price_to_beat={on(self.use_price_to_beat)} "
            f"xvenue={on(self.xvenue_enabled)} sell_tail>={self.sell_price_min:.2f}"
        )
