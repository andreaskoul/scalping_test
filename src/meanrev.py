"""
Mean-reversion overlay on the Polymarket-vs-fair gap.

Portnaya (2026) finds the gap D_t = P_poly − P_fair is persistent but
*mean-reverting*, with an AR(1) half-life of ~4 hours.  That is a slower,
higher-capacity edge than the latency arb: it does not need sub-100ms
execution, just the discipline to fade D_t when it stretches beyond its own
recent mean and to hold for a few hours.

We track an EWMA of D_t per token (half-life ≈ 4h to match the AR(1)) and
fire a fade when the *current* deviation from that mean exceeds a band:

    D_t − EWMA(D) >  band   → market unusually rich  → SELL YES
    D_t − EWMA(D) < −band   → market unusually cheap → BUY  YES

The trade's truth anchor is the option-implied p_fair, so sizing downstream
(Kelly) treats p_fair as P(win).  Entry is gated by a per-token cooldown set
to the max-hold horizon so we don't stack the same reversion repeatedly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class FadeIntent:
    token_id: str
    side: str           # "BUY" or "SELL" (of YES)
    p_fair: float
    deviation: float    # signed D_t − EWMA(D)
    price: float        # the book level we'd cross (bid for SELL, ask for BUY)


class MeanReversionTracker:
    def __init__(
        self,
        band: float = 0.05,
        half_life_secs: float = 14400.0,   # 4h
        max_hold_secs: float = 12600.0,    # 3.5h
        warmup_updates: int = 8,
    ):
        self.band = band
        self.half_life = max(1.0, half_life_secs)
        self.max_hold = max_hold_secs
        self.warmup = warmup_updates
        # token_id → (ewma_D, last_update_monotonic, n_updates)
        self._ewma: dict[str, tuple[float, float, int]] = {}
        self._last_fire: dict[str, float] = {}

    def update(
        self,
        token_id: str,
        p_fair: float,
        best_bid: float,
        best_ask: float,
    ) -> FadeIntent | None:
        """Feed a fresh observation; return a FadeIntent if one fires."""
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return None
        p_poly = 0.5 * (best_bid + best_ask)
        d = p_poly - p_fair

        now = time.monotonic()
        prev = self._ewma.get(token_id)
        if prev is None:
            self._ewma[token_id] = (d, now, 1)
            return None
        ewma, last_ts, n = prev
        dt = max(1e-3, now - last_ts)
        lam = math.exp(-dt * math.log(2.0) / self.half_life)
        ewma = lam * ewma + (1.0 - lam) * d
        self._ewma[token_id] = (ewma, now, n + 1)

        if n + 1 < self.warmup:
            return None

        # cooldown: one fade per max-hold horizon per token
        last_fire = self._last_fire.get(token_id, 0.0)
        if now - last_fire < self.max_hold:
            return None

        deviation = d - ewma
        if deviation > self.band:
            self._last_fire[token_id] = now
            return FadeIntent(token_id, "SELL", p_fair, deviation, best_bid)
        if deviation < -self.band:
            self._last_fire[token_id] = now
            return FadeIntent(token_id, "BUY", p_fair, deviation, best_ask)
        return None
