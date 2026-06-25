"""Cost-aware labeling for Eve.

Raw-direction labels (``build_bar_sequences`` in :mod:`src.eve_data`) are a
research baseline only. The production-adjacent label per
``docs/EVE_TRANSFORMER_RESEARCH.md`` is *future tradable edge after fees and
slippage, with an explicit no-trade class*: a directional action is labeled
only when its **post-cost** edge is positive.

This is the cost-aware execution filter that recurs across the literature:

  - López de Prado, *Advances in Financial Machine Learning* (2018): the
    triple-barrier method labels by which barrier (profit-take / stop / time)
    is hit first, so the label embeds the cost of acting rather than the raw
    sign of a move.
  - Ntakaris et al. (FI-2010, arXiv:1705.03233) and the LOBCAST/benchmark
    studies: ternary up/stable/down labels use a threshold alpha; deep models
    that look strong on raw labels "fail to generate consistent profit" once a
    realistic cost is imposed.
  - "Machine Learning-Based Bitcoin Trading Under Transaction Costs"
    (arXiv:2606.00060): forecasts only become valuable when the trading rule
    permits a trade *only when the forecast magnitude exceeds a
    transaction-cost-based threshold*. Naive sign strategies die at ~10 bps.

The design consequence used here: **the no-trade boundary IS the round-trip
cost** ``c``, not an arbitrary threshold. A long is labeled only when the gross
forward return clears ``c``; a short only when it clears ``c`` to the downside;
everything in between is no-trade. ``post_cost_pnl`` uses the same ``c`` so a
correctly-labeled directional sample is positive-EV by construction and the
model's only job is to recover that label out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# Action / label encoding — identical to EveSequenceSample.label so the lake,
# baselines, and evaluator all speak the same integers.
DOWN = 0  # take the short side
FLAT = 1  # no trade
UP = 2    # take the long side

ACTION_NAMES = {DOWN: "down", FLAT: "flat", UP: "up"}


@dataclass(frozen=True)
class CostModel:
    """Round-trip trading cost as a fraction of notional.

    Fields are *per side* in basis points; a round trip crosses twice. Keeping
    the legs explicit makes the cost auditable per the research contract's
    accuracy-claims audit (every edge must record its cost treatment).

    - ``taker_fee_bps``: exchange/protocol fee charged per fill.
    - ``half_spread_bps``: half the quoted spread, i.e. the cost of crossing to
      the touch on one side.
    - ``slippage_bps``: extra adverse fill beyond the touch (impact, latency).

    Defaults are a deliberately *non-zero* liquid-equity-ish floor so that
    "edge" never means "edge before costs". Callers should override per venue
    (Polymarket taker fees, crypto maker rebates, etc.).
    """

    taker_fee_bps: float = 0.0
    half_spread_bps: float = 1.0
    slippage_bps: float = 0.5

    def per_side(self) -> float:
        return (self.taker_fee_bps + self.half_spread_bps + self.slippage_bps) / 1e4

    def round_trip(self) -> float:
        """Total cost (fraction) of opening and closing one unit position."""
        return 2.0 * self.per_side()


def cost_aware_label(future_return: float, cost: float) -> int:
    """Ternary action label gated by round-trip ``cost``.

    Returns UP/DOWN only when the gross move clears the cost of trading it;
    otherwise FLAT (no-trade). ``cost`` is the round-trip fraction (see
    :meth:`CostModel.round_trip`). A non-finite return is treated as no-trade.
    """
    if cost < 0:
        raise ValueError("cost must be non-negative")
    r = float(future_return)
    if r != r:  # NaN
        return FLAT
    if r > cost:
        return UP
    if r < -cost:
        return DOWN
    return FLAT


def post_cost_pnl(action: int, future_return: float, cost: float) -> float:
    """Realized post-cost PnL (fraction of notional) of taking ``action``.

    Long earns ``future_return`` minus the round-trip cost; short earns the
    negative move minus cost; no-trade earns zero. This is the *only* quantity
    the Verifier should gate on — accuracy on raw labels is not a substitute
    (FI-2010 critique). The cost charged here matches the labeling cost.
    """
    r = float(future_return)
    if r != r:  # NaN -> treat as flat, no PnL
        return 0.0
    if action == UP:
        return r - cost
    if action == DOWN:
        return -r - cost
    return 0.0


def cost_aware_labels(future_returns: Iterable[float], cost: float) -> list[int]:
    """Vectorless helper: cost-aware action label for each forward return."""
    return [cost_aware_label(r, cost) for r in future_returns]


def post_cost_pnls(
    actions: Sequence[int], future_returns: Sequence[float], cost: float
) -> list[float]:
    """Per-sample post-cost PnL for a sequence of actions vs realized returns."""
    if len(actions) != len(future_returns):
        raise ValueError("actions and future_returns must be the same length")
    return [post_cost_pnl(a, r, cost) for a, r in zip(actions, future_returns)]


def label_distribution(actions: Iterable[int]) -> dict[str, int]:
    """Count of down/flat/up actions, manifest-friendly."""
    counts = {"down": 0, "flat": 0, "up": 0}
    for a in actions:
        counts[ACTION_NAMES.get(int(a), "flat")] += 1
    return counts
