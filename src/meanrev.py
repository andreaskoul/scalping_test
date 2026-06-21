"""
Mean-reversion overlay on the Polymarket-vs-fair gap — now a round-trip.

Portnaya (2026): D_t = P_poly − P_fair is persistent but mean-reverting with an
AR(1) half-life of ~4h.  The earlier version had two defects this rewrite fixes:

  1. **No TTE gate** — it would arm a 4h-reversion thesis on a 5-minute market
     that resolves long before D_t reverts. We now require
     `tte > tte_gate_mult · half_life`, so it only trades markets with room to
     revert (hourly / threshold), never 5- or 15-minute windows.

  2. **No exit** — it was entry-only and held to resolution, i.e. a directional
     bet, not a reversion trade. We now track the open position and emit an
     EXIT when D_t reverts inside `exit_z·σ` of its mean (take-profit) or after
     `max_hold` (time stop).

The mean/variance of D_t are tracked as a time-decayed EWMA whose half-life is
the AR(1) half-life; entry is a z-score test `|D − μ| > z_entry·σ` (plus an
absolute `band` floor), which is the AR(1)-consistent "stretched beyond its own
recent range" signal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class MeanRevAction:
    kind: str            # "ENTER" or "EXIT"
    token_id: str
    side: str            # BUY or SELL of YES
    price: float         # book level to cross (bid for SELL, ask for BUY)
    p_fair: float
    z: float             # standardized deviation at the moment of action


@dataclass
class _Pos:
    side: str
    entry_ts: float
    size: float
    entry_price: float


class MeanReversionTracker:
    def __init__(
        self,
        band: float = 0.05,
        half_life_secs: float = 14400.0,   # 4h
        max_hold_secs: float = 12600.0,    # 3.5h
        warmup_updates: int = 8,
        tte_gate_mult: float = 2.0,        # need tte > 2× half-life to arm
        z_entry: float = 2.0,
        exit_z: float = 0.5,
        var_floor: float = 1e-6,
    ):
        self.band = band
        self.half_life = max(1.0, half_life_secs)
        self.max_hold = max_hold_secs
        self.warmup = warmup_updates
        self.tte_gate = tte_gate_mult * self.half_life
        self.z_entry = z_entry
        self.exit_z = exit_z
        self.var_floor = var_floor
        # token_id → (mean, var, last_ts, n)
        self._ewma: dict[str, tuple[float, float, float, int]] = {}
        self._open: dict[str, _Pos] = {}

    def has_position(self, token_id: str) -> bool:
        return token_id in self._open

    def update(
        self,
        token_id: str,
        tte: float,
        p_fair: float,
        best_bid: float,
        best_ask: float,
        size_hint: float = 0.0,
    ) -> MeanRevAction | None:
        """Feed an observation; return an ENTER or EXIT action, or None.

        `tte` (seconds to expiry) gates entries; exits are allowed regardless so
        an open position can always be closed."""
        if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
            return None
        p_poly = 0.5 * (best_bid + best_ask)
        d = p_poly - p_fair

        now = time.monotonic()
        prev = self._ewma.get(token_id)
        if prev is None:
            self._ewma[token_id] = (d, 0.0, now, 1)
            return None
        mean, var, last_ts, n = prev
        dt = max(1e-3, now - last_ts)
        lam = math.exp(-dt * math.log(2.0) / self.half_life)
        new_mean = lam * mean + (1.0 - lam) * d
        new_var = lam * var + (1.0 - lam) * (d - mean) * (d - new_mean)
        self._ewma[token_id] = (new_mean, max(0.0, new_var), now, n + 1)

        sigma = math.sqrt(max(new_var, self.var_floor))
        z = (d - new_mean) / sigma

        # ---- exit takes priority over a new entry on the same token ----
        pos = self._open.get(token_id)
        if pos is not None:
            reverted = abs(z) <= self.exit_z
            timed_out = (now - pos.entry_ts) >= self.max_hold
            if reverted or timed_out:
                del self._open[token_id]
                # flatten: opposite side of the entry
                if pos.side == "SELL":
                    return MeanRevAction("EXIT", token_id, "BUY", best_ask, p_fair, z)
                return MeanRevAction("EXIT", token_id, "SELL", best_bid, p_fair, z)
            return None

        if n + 1 < self.warmup:
            return None
        if tte < self.tte_gate:
            return None   # not enough time for the reversion to complete
        deviation = d - new_mean
        if abs(deviation) <= self.band or abs(z) < self.z_entry:
            return None

        size = round(size_hint, 2) if size_hint > 0 else 0.0
        if deviation > 0:   # market unusually rich → SELL YES at the bid
            self._open[token_id] = _Pos("SELL", now, size, best_bid)
            return MeanRevAction("ENTER", token_id, "SELL", best_bid, p_fair, z)
        # market unusually cheap → BUY YES at the ask
        self._open[token_id] = _Pos("BUY", now, size, best_ask)
        return MeanRevAction("ENTER", token_id, "BUY", best_ask, p_fair, z)
