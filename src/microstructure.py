"""
Microstructure overlay — OFI-driven direction, OBI veto, momentum & fade.

Grounded in Deep et al. (2025): on minute-level data a Random Forest predicts
the next move at 88% AUC, with **order-flow imbalance the dominant feature
(43%)** and lagged returns next (31%) — i.e. real short-horizon momentum, not
a random walk.  We can't ship their (computationally infeasible) binary tree,
and raw ML direction is unreliable on binaries (arXiv:2511.15960), so we use:

  1. A small, transparent **logistic directional model** over the features we
     can compute live (OFI, fast/slow returns, realised vol, spread).  Its
     output is a *physical* P(up); the caller converts it to risk-neutral via
     pricing.physical_to_risk_neutral before blending into p*.  Weights default
     to OFI-dominant and can be overridden by a trained `model_weights.json`.

  2. An **OBI toxicity veto** (DRADIS-style, default −0.60): block a taker that
     would cross into stacked opposing size — the classic adverse-selection
     trap where the quote is stale *because* informed flow is about to run it.

  3. **Crowd-momentum continuation** near expiry (the reference repo's edge) and
     **impulse-fade** mean-reversion away from expiry — regime-split by tte so
     they never fight each other.
"""

from __future__ import annotations

import bisect
import itertools
import json
import logging
import math
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

MODEL_WEIGHTS_PATH = "model_weights.json"


@dataclass
class MicroFeatures:
    ofi: float = 0.0            # signed-flow ratio in [-1, 1]
    ret_fast: float = 0.0       # log-return over ~5s
    ret_slow: float = 0.0       # log-return over ~60s
    rvol: float = 0.0           # annualised realised vol
    spread_rel: float = 0.0     # relative Binance spread
    move_window_usd: float = 0.0
    secs_to_expiry: float = 0.0


def order_book_imbalance(bid_size: float, ask_size: float) -> float:
    """(bid − ask) / (bid + ask) in [-1, 1]. >0 = buy pressure."""
    tot = bid_size + ask_size
    if tot <= 0:
        return 0.0
    return (bid_size - ask_size) / tot


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Logistic feature space. p_up = sigmoid(w · transform(features)). The trainer
# (src.ml_train) fits `w` on exactly this transform so train and inference agree.
FEATURE_KEYS = ["bias", "ofi", "ret_fast", "ret_slow"]

# Default weights — OFI dominant, momentum next, per Deep et al. importances.
# Used as the cold-start fallback until a trained model_weights.json exists.
_DEFAULT_WEIGHTS = {
    "bias": 0.0,
    "ofi": 2.5,
    "ret_fast": 1.2,
    "ret_slow": 0.8,
}


def transform_features(f: MicroFeatures) -> list[float]:
    """Map MicroFeatures to the logistic design vector (incl. bias term).

    Returns are squashed (a few bps → O(1)) so no single term saturates; OFI is
    already bounded to [-1, 1]. Order matches FEATURE_KEYS."""
    return [
        1.0,
        max(-1.0, min(1.0, f.ofi)),
        math.tanh(f.ret_fast * 400.0),
        math.tanh(f.ret_slow * 150.0),
    ]


class DirectionalModel:
    """Physical P(up) from microstructure features (logistic blend)."""

    def __init__(self, weights: dict | None = None):
        self.w = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)

    @classmethod
    def load(cls, path: str = MODEL_WEIGHTS_PATH) -> "DirectionalModel":
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                # Accept either a bare weights dict or {"weights": {...}, ...}.
                return cls(data.get("weights", data) if isinstance(data, dict) else None)
            except Exception as exc:
                log.warning("Failed to load %s: %s — using defaults.", path, exc)
        return cls()

    def p_up(self, f: MicroFeatures) -> float:
        x = transform_features(f)
        z = sum(self.w.get(k, 0.0) * xi for k, xi in zip(FEATURE_KEYS, x))
        return _sigmoid(z)


def momentum_bias(
    f: MicroFeatures,
    min_move_usd: float,
    near_expiry_secs: float,
) -> int:
    """+1 / −1 / 0 — continuation bias in the final window before expiry."""
    if f.secs_to_expiry > near_expiry_secs:
        return 0
    if abs(f.move_window_usd) < min_move_usd:
        return 0
    return 1 if f.move_window_usd > 0 else -1


def impulse_fade_bias(
    f: MicroFeatures,
    min_move_usd: float,
    near_expiry_secs: float,
) -> int:
    """+1 / −1 / 0 — fade an over-extension when there's ample time to revert."""
    if f.secs_to_expiry <= near_expiry_secs:
        return 0
    if abs(f.move_window_usd) < 2.0 * min_move_usd:
        return 0
    return -1 if f.move_window_usd > 0 else 1


class MicroReplay:
    """Rebuild MicroFeatures from a historical trade tape (Binance aggTrades).

    Lets the backtest exercise the OFI/ML/momentum overlays on real history —
    aggTrades are downloadable even though full L2 depth is not, and OFI is the
    dominant feature (Deep et al.), so this recovers most of the microstructure
    signal. Trades are (ts, price, signed_dollar_vol, abs_dollar_vol), sorted.
    """

    def __init__(self, trades: list[tuple[float, float, float, float]]):
        trades = sorted(trades, key=lambda t: t[0])
        self.ts = [t[0] for t in trades]
        self.px = [t[1] for t in trades]
        self._cum_s = list(itertools.accumulate(t[2] for t in trades))
        self._cum_a = list(itertools.accumulate(t[3] for t in trades))

    def _idx_le(self, t: float) -> int:
        return bisect.bisect_right(self.ts, t) - 1

    def price_at(self, t: float) -> float:
        i = self._idx_le(t)
        return self.px[i] if i >= 0 else 0.0

    def ofi(self, t: float, window: float = 30.0) -> float:
        hi = self._idx_le(t)
        if hi < 0:
            return 0.0
        lo = bisect.bisect_left(self.ts, t - window)
        if lo > hi:
            return 0.0
        signed = self._cum_s[hi] - (self._cum_s[lo - 1] if lo > 0 else 0.0)
        absv = self._cum_a[hi] - (self._cum_a[lo - 1] if lo > 0 else 0.0)
        return 0.0 if absv <= 0 else max(-1.0, min(1.0, signed / absv))

    def features(
        self, t: float, spot: float, sigma: float, tte: float,
        ofi_window: float = 30.0, mom_window: float = 180.0,
    ) -> MicroFeatures:
        p0, pf, ps = self.price_at(t), self.price_at(t - 5), self.price_at(t - 60)
        pm = self.price_at(t - mom_window)
        ret_fast = math.log(p0 / pf) if p0 > 0 and pf > 0 else 0.0
        ret_slow = math.log(p0 / ps) if p0 > 0 and ps > 0 else 0.0
        return MicroFeatures(
            ofi=self.ofi(t, ofi_window),
            ret_fast=ret_fast, ret_slow=ret_slow,
            rvol=sigma, spread_rel=0.0,
            move_window_usd=(spot - pm) if pm > 0 else 0.0,
            secs_to_expiry=tte,
        )


@dataclass
class MicroAssessment:
    p_up_physical: float
    obi: float
    direction_bias: int     # combined momentum/fade vote, +1/−1/0
    vetoed: bool            # OBI toxicity veto for the requested side


class MicrostructureEngine:
    def __init__(
        self,
        model: DirectionalModel | None = None,
        obi_veto: bool = True,
        obi_veto_threshold: float = -0.60,
        momentum_enabled: bool = True,
        impulse_fade_enabled: bool = True,
        momentum_min_move_usd: float = 60.0,
        momentum_near_expiry_secs: float = 180.0,
    ):
        self.model = model or DirectionalModel.load()
        self.obi_veto = obi_veto
        self.obi_thr = obi_veto_threshold
        self.momentum_enabled = momentum_enabled
        self.impulse_fade_enabled = impulse_fade_enabled
        self.min_move = momentum_min_move_usd
        self.near_expiry = momentum_near_expiry_secs

    def assess(self, f: MicroFeatures, poly_obi: float, side: str) -> MicroAssessment:
        p_up = self.model.p_up(f)

        bias = 0
        if self.momentum_enabled:
            bias += momentum_bias(f, self.min_move, self.near_expiry)
        if self.impulse_fade_enabled:
            bias += impulse_fade_bias(f, self.min_move, self.near_expiry)

        vetoed = False
        if self.obi_veto:
            # BUY crosses the ask: toxic if sellers are stacked (obi very −).
            # SELL hits the bid: toxic if buyers are stacked (obi very +).
            if side.upper() == "BUY" and poly_obi < self.obi_thr:
                vetoed = True
            elif side.upper() == "SELL" and poly_obi > -self.obi_thr:
                vetoed = True

        return MicroAssessment(
            p_up_physical=p_up,
            obi=poly_obi,
            direction_bias=max(-1, min(1, bias)),
            vetoed=vetoed,
        )
