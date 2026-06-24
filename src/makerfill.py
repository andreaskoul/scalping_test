"""Queue-aware maker-fill model for counterfactual replay evaluation.

Inspired by nkaz001/hftbacktest's queue models — in particular
``PowerProbQueueFunc`` and the risk-averse / probabilistic queue position
estimators (see github.com/nkaz001/hftbacktest, ``hftbacktest/models/queue.py``).
The dominant driver of whether a resting limit order fills is *queue position*:
how much size rests at or ahead of our price level when trades arrive. A fill
only happens once cumulative traded volume on our side has consumed everything
ahead of us in the queue.

These are LABELS, not optimism. Every function here is conservative by
construction (queue-ahead is counted from full L2 depth; the power exponent
``n`` defaults to 2, which under-counts fills relative to a naive
"any-trade-fills-us" rule). The ONLY ground truth for fills is the live canary;
this module exists to produce honest counterfactual fill *probabilities* for
backtest/replay scoring, never to inflate paper PnL.

Pure functions, numpy-only. No network, no asyncio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pricing import taker_fee_per_share
from .replay import estimate_maker_fill  # top-of-book fallback (conservative)


@dataclass
class FillEstimate:
    """Counterfactual maker-fill label for one resting order.

    ``filled_prob`` is a probability in [0, 1] (queue-aware). ``expected_rebate``
    is the *expected* per-share maker rebate (see ``expected_maker_rebate`` —
    it is an expectation over an uncertain pro-rata distribution, not a
    guaranteed credit). ``adverse`` flags an unfavorable markout (we filled and
    the market then moved against us — classic adverse selection).
    """

    filled_prob: float
    queue_ahead: float
    expected_rebate: float
    markout_5s: float | None = None
    markout_30s: float | None = None
    adverse: bool = False


def queue_ahead(levels: dict | None, side: str, price: float) -> float:
    """Resting size at or ahead of our price from a full-L2 ``levels`` dict.

    ``levels`` is the recorded shape ``{"bids": [[px, sz], ...], "asks": [...]}``.

    For a BUY maker resting at ``price``, everyone willing to pay >= ``price``
    is ahead of (or level with) us in the bid queue, so queue-ahead is the total
    bid size at prices >= ``price``. Symmetrically, a SELL maker at ``price`` is
    behind every ask quoting <= ``price``, so queue-ahead is the total ask size
    at prices <= ``price``.

    Returns 0.0 when ``levels`` is None / empty / malformed, or when no resting
    size qualifies. Size at exactly our price counts as ahead (pessimistic:
    assume we joined the back of that price level).
    """
    if not levels:
        return 0.0
    side_u = side.upper()
    if side_u == "BUY":
        book = levels.get("bids") or []
        ahead = 0.0
        for entry in book:
            try:
                px, sz = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if px >= price and sz > 0:
                ahead += sz
        return ahead
    if side_u == "SELL":
        book = levels.get("asks") or []
        ahead = 0.0
        for entry in book:
            try:
                px, sz = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if px <= price and sz > 0:
                ahead += sz
        return ahead
    raise ValueError("side must be BUY or SELL")


def power_prob_fill(queue_ahead: float, traded_volume: float, n: float = 2.0) -> float:
    """hftbacktest-style probability that volume consumed our queue position.

    Models the fill probability of a resting order whose queue-ahead is
    ``queue_ahead`` after ``traded_volume`` has traded through our side:

        p = clip(1 - queue_ahead / traded_volume, 0, 1) ** n

    The linear ``1 - queue_ahead/traded_volume`` term is the fraction of our
    queue position that volume has eaten; raising it to the power ``n`` makes
    the estimate progressively more conservative (``n > 1`` discounts partial
    consumption). This is the ``PowerProbQueueFunc`` shape from
    nkaz001/hftbacktest (``hftbacktest/models/queue.py``); ``n = 2`` is the
    repo's typical risk-averse default and the default here.

    Returns 0.0 when ``traded_volume <= 0`` (nothing traded → no fill) and 1.0
    only once cumulative volume has fully consumed the queue ahead.
    """
    if traded_volume <= 0:
        return 0.0
    if queue_ahead < 0:
        queue_ahead = 0.0
    consumed = 1.0 - (queue_ahead / traded_volume)
    consumed = float(np.clip(consumed, 0.0, 1.0))
    return float(consumed ** max(0.0, n))


def expected_maker_rebate(
    price: float,
    fill_prob: float,
    rebate_share: float = 0.20,
    fee_rate: float = 0.072,
    fee_exponent: float = 1.0,
) -> float:
    """EXPECTED (uncertain) per-share maker rebate — realism correction.

    CRITICAL: Polymarket maker rebates are NOT a guaranteed per-share credit.
    They are a *pro-rata daily distribution* of collected taker fees: makers
    share a pool (~20% of crypto-market taker fees as of 2026) split across all
    eligible resting liquidity that day. Our realized rebate therefore depends
    on (a) whether we actually fill and (b) our share of the daily pool, neither
    of which is known at quote time.

    We model it as an *expectation*:

        expected = rebate_share * taker_fee_per_share(price, ...) * fill_prob

    i.e. the taker-fee that our price level would generate, scaled by the maker
    pool share and by our probability of filling. ``pricing.maker_rebate_per_share``
    instead treats the rebate as a GUARANTEED per-share credit (implicitly
    fill_prob == 1 and a certain rate) and therefore OVERSTATES what we actually
    earn: at any realistic maker fill probability (queue position rarely
    guarantees a fill) this expectation is strictly below that naive credit.
    Use this function, not ``maker_rebate_per_share``, when scoring
    counterfactual maker PnL.

    Returns 0.0 for a zero fill probability.
    """
    rebate_share = max(0.0, rebate_share)
    fill_prob = float(np.clip(fill_prob, 0.0, 1.0))
    fee = taker_fee_per_share(price, fee_rate, fee_exponent)
    return rebate_share * fee * fill_prob


def estimate_fill(
    events: list,
    token_id: str,
    side: str,
    price: float,
    posted_ts: float,
    gtd_secs: float,
    *,
    levels: dict | None = None,
    traded_volume: float | None = None,
    n: float = 2.0,
    rebate_share: float = 0.20,
    fee_rate: float = 0.072,
    fee_exponent: float = 1.0,
    markout_5s: float | None = None,
    markout_30s: float | None = None,
) -> FillEstimate:
    """Combine the queue-aware and top-of-book paths into one FillEstimate.

    When full-L2 ``levels`` AND a ``traded_volume`` proxy are available, use the
    queue-aware ``power_prob_fill``. When only top-of-book is available (no
    ``levels`` or no volume proxy), fall back to the conservative trade-through
    rule in ``replay.estimate_maker_fill`` (a hard 0/1 label). Either way the
    rebate is the *expected* (uncertain) one.

    ``markout_5s``/``markout_30s`` are passed through if the caller computed them
    (via ``replay.markout``); ``adverse`` is set when the 5s markout is negative.
    """
    if levels and traded_volume is not None:
        q = queue_ahead(levels, side, price)
        prob = power_prob_fill(q, traded_volume, n=n)
    else:
        # Top-of-book fallback: conservative trade-through rule → hard label.
        filled, _fill_ts, _reason = estimate_maker_fill(
            events, token_id, side, price, posted_ts, gtd_secs
        )
        q = 0.0
        prob = 1.0 if filled else 0.0

    rebate = expected_maker_rebate(
        price, prob, rebate_share=rebate_share,
        fee_rate=fee_rate, fee_exponent=fee_exponent,
    )
    adverse = markout_5s is not None and markout_5s < 0.0
    return FillEstimate(
        filled_prob=prob,
        queue_ahead=q,
        expected_rebate=rebate,
        markout_5s=markout_5s,
        markout_30s=markout_30s,
        adverse=adverse,
    )
