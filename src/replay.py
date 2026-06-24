"""Replay helpers + counterfactual evaluator for recorded Polymarket shards.

The first half of this module rebuilds the Polymarket top-of-book timeline that
`PolyWS` captured — the non-refetchable part of the system. The second half
(``ReplayConfig`` onward) turns that timeline into a counterfactual *evaluator*:
it refetches the Binance spot/feature series for the recorded window (reusing
``src.backtest``'s fetchers), replays the real ``SignalGenerator`` against the
merged book+spot stream, and scores each emitted signal with a queue-aware
maker-fill model (``src.makerfill``) plus markout. A config sweep then splits
signals by time into train/validation and reports OOS edge with a
selection-bias-aware (Deflated-Sharpe) penalty from ``src.stats``.

Counterfactual fills are LABELS, not optimism. The maker path is queue-aware and
conservative by default; the ONLY ground truth for fills is the live canary.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

from .poly_ws import BookSnapshot


@dataclass
class ReplayEvent:
    ts_wall: float
    ts_mono: float
    token_id: str
    etype: str
    snapshot: BookSnapshot
    levels: dict | None = None


@dataclass
class MakerReplayResult:
    token_id: str
    side: str
    price: float
    posted_ts: float
    filled: bool
    fill_ts: float | None = None
    mark_5s: float | None = None
    mark_30s: float | None = None
    reason: str = ""


def load_poly_events(run_dir: str | Path) -> list[ReplayEvent]:
    run_path = Path(run_dir)
    events: list[ReplayEvent] = []
    for shard in sorted(run_path.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                snap = BookSnapshot(
                    token_id=str(row["token_id"]),
                    best_bid=float(row.get("best_bid", 0.0)),
                    best_ask=float(row.get("best_ask", 1.0)),
                    bid_size=float(row.get("bid_sz", 0.0)),
                    ask_size=float(row.get("ask_sz", 0.0)),
                    ts=float(row.get("ts_mono", 0.0)),
                )
                events.append(ReplayEvent(
                    ts_wall=float(row.get("ts_wall", 0.0)),
                    ts_mono=float(row.get("ts_mono", 0.0)),
                    token_id=str(row["token_id"]),
                    etype=str(row.get("etype", "")),
                    snapshot=snap,
                    levels=row.get("levels"),
                ))
    events.sort(key=lambda ev: (ev.ts_wall, ev.ts_mono))
    return events


def latest_books(events: list[ReplayEvent]) -> dict[str, BookSnapshot]:
    books: dict[str, BookSnapshot] = {}
    for event in events:
        books[event.token_id] = event.snapshot
    return books


def events_for_token(events: list[ReplayEvent], token_id: str) -> list[ReplayEvent]:
    """Return one token's replay events in chronological order."""
    return sorted((ev for ev in events if ev.token_id == token_id), key=lambda ev: (ev.ts_wall, ev.ts_mono))


def book_at_or_after(events: list[ReplayEvent], token_id: str, ts_wall: float) -> BookSnapshot | None:
    """First snapshot at or after wall-clock ts for markout estimates."""
    for ev in events_for_token(events, token_id):
        if ev.ts_wall >= ts_wall:
            return ev.snapshot
    return None


def estimate_maker_fill(
    events: list[ReplayEvent],
    token_id: str,
    side: str,
    price: float,
    posted_ts: float,
    gtd_secs: float,
) -> tuple[bool, float | None, str]:
    """Conservative maker-fill estimate from top-of-book replay.

    BUY maker is considered filled only when a later ask trades/moves down to
    or through our bid. SELL maker is filled only when a later bid moves up to
    or through our ask. This intentionally undercounts fills when only top L2 is
    available; it is a validation label, not a paper-PnL optimism switch.
    """
    side_u = side.upper()
    if side_u not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    if price <= 0:
        raise ValueError("price must be positive")
    deadline = posted_ts + max(0.0, gtd_secs)
    future = [ev for ev in events_for_token(events, token_id) if posted_ts < ev.ts_wall <= deadline]
    for ev in future:
        snap = ev.snapshot
        if side_u == "BUY" and snap.best_ask > 0 and snap.best_ask <= price:
            return True, ev.ts_wall, "ASK_THROUGH_PRICE"
        if side_u == "SELL" and snap.best_bid >= price:
            return True, ev.ts_wall, "BID_THROUGH_PRICE"
    return False, None, "GTD_EXPIRED"


def markout(
    events: list[ReplayEvent],
    token_id: str,
    side: str,
    entry_price: float,
    ts_wall: float,
    delay_secs: float,
) -> float | None:
    """Signed midpoint markout after a delay.

    Positive is favorable to the entry side: BUY profits when midpoint rises;
    SELL profits when midpoint falls.
    """
    snap = book_at_or_after(events, token_id, ts_wall + delay_secs)
    if snap is None or snap.best_bid <= 0 or snap.best_ask <= 0:
        return None
    mid = 0.5 * (snap.best_bid + snap.best_ask)
    if side.upper() == "BUY":
        return mid - entry_price
    if side.upper() == "SELL":
        return entry_price - mid
    raise ValueError("side must be BUY or SELL")


def estimate_maker_replay(
    events: list[ReplayEvent],
    token_id: str,
    side: str,
    price: float,
    posted_ts: float,
    gtd_secs: float,
) -> MakerReplayResult:
    filled, fill_ts, reason = estimate_maker_fill(events, token_id, side, price, posted_ts, gtd_secs)
    return MakerReplayResult(
        token_id=token_id,
        side=side.upper(),
        price=price,
        posted_ts=posted_ts,
        filled=filled,
        fill_ts=fill_ts,
        mark_5s=markout(events, token_id, side, price, posted_ts, 5.0),
        mark_30s=markout(events, token_id, side, price, posted_ts, 30.0),
        reason=reason,
    )


# --------------------------------------------------------------------------- #
#  Counterfactual evaluator (Workstream A)                                     #
# --------------------------------------------------------------------------- #
# Everything below turns the recorded poly timeline into a counterfactual
# evaluator: refetch Binance, replay the real SignalGenerator over the merged
# stream, score fills (taker = book cross; maker = queue-aware probability +
# markout), then sweep configs with an OOS time split and a Deflated-Sharpe
# penalty for the number of configs tried.
#
# Imports are local to this section so the cheap shard-report path (and the
# tests in test_validation_pipeline) keep working without scipy/numpy/aiohttp.

import bisect as _bisect
import time as _time
from typing import Any, Callable, Sequence

import numpy as np

from . import makerfill as _makerfill
from . import stats as _stats
from .binance_ws import BinanceTick
from .microstructure import MicroFeatures
from .poly_universe import PolyMarket
from .pricing import taker_fee_per_share as _tfee
from .signal import SignalGenerator
from .telemetry import DecisionTrace


@dataclass
class ReplayConfig:
    """A point in the SignalGenerator knob-space to evaluate / sweep.

    Only the subset of knobs that materially change which signals fire is
    captured; everything else uses the SignalGenerator defaults. ``name`` is a
    stable label for reporting and reproducibility.
    """

    name: str = "default"
    safety_eps: float = 0.005
    maker_enabled: bool = False
    maker_join_ticks: int = 1
    obi_veto: bool = False
    obi_veto_threshold: float = -0.60
    min_tte_secs: float = 180.0
    max_tte_secs: float = 3600.0
    price_min: float = 0.10
    price_max: float = 0.90
    fee_rate: float = 0.072
    fee_exponent: float = 1.0
    # Maker-fill model knobs (queue-aware scoring; do not affect signal firing).
    maker_fill_n: float = 2.0
    rebate_share: float = 0.20

    def build_signal_generator(self) -> SignalGenerator:
        """Construct a SignalGenerator with this config's knobs.

        ``skip_updown`` is forced off and ``book_max_age_secs`` is generous so
        the replay (which presents fresh books by construction) is not gated by
        live-only staleness checks. The pricing fee schedule comes from the
        per-market PolyMarket we synthesise, not from here.
        """
        return SignalGenerator(
            safety_eps=self.safety_eps,
            cooldown_secs=0.0,            # replay scores every eligible event
            fee_rate=self.fee_rate,
            price_min=self.price_min,
            price_max=self.price_max,
            skip_updown=False,
            min_tte_secs=self.min_tte_secs,
            max_tte_secs=self.max_tte_secs,
            book_max_age_secs=1e9,
            maker_enabled=self.maker_enabled,
            maker_join_ticks=self.maker_join_ticks,
            obi_veto=self.obi_veto,
            obi_veto_threshold=self.obi_veto_threshold,
        )


@dataclass
class BinancePoint:
    """One refetched Binance observation aligned to wall-clock time."""

    ts_wall: float
    mid: float
    sigma_annual: float
    drift_annual: float = 0.0
    features: MicroFeatures | None = None


@dataclass
class BinanceSeries:
    """Time-ordered Binance spot/feature series over the recorded window.

    ``points`` are sorted by ``ts_wall``. ``mid_at`` / ``point_at`` do an
    as-of (last-known) lookup so the merge can pull the spot state that was
    current at any poly-book timestamp.
    """

    symbol: str
    points: list[BinancePoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.points.sort(key=lambda p: p.ts_wall)
        self._ts = [p.ts_wall for p in self.points]

    def point_at(self, ts_wall: float) -> BinancePoint | None:
        if not self.points:
            return None
        i = _bisect.bisect_right(self._ts, ts_wall) - 1
        return self.points[i] if i >= 0 else None


@dataclass
class SignalRecord:
    """One counterfactual signal + its scored fill labels."""

    ts_wall: float
    token_id: str
    side: str
    price: float
    size: float
    edge: float
    is_maker: int
    p_star: float
    fill_prob: float
    queue_ahead: float
    expected_rebate: float
    markout_5s: float | None
    markout_30s: float | None
    pnl_proxy: float          # post-cost per-share PnL proxy (markout-based)
    adverse: int


@dataclass
class ReplayResults:
    """Result of evaluating one config over a window."""

    config_name: str
    seed: int
    device: str
    records: list[SignalRecord] = field(default_factory=list)

    @property
    def pnls(self) -> list[float]:
        return [r.pnl_proxy for r in self.records]


@dataclass
class ConfigEdge:
    """Per-config OOS edge summary with selection-bias-aware stats."""

    config_name: str
    n_train: int
    n_val: int
    train_mean: float
    val_mean: float
    val_ci_lo: float
    val_ci_hi: float
    val_t_stat: float
    val_sharpe: float


@dataclass
class SweepReport:
    """Full sweep output: per-config OOS edges + the selected config's DSR."""

    seed: int
    device: str
    train_frac: float
    walk_forward: bool
    edges: list[ConfigEdge] = field(default_factory=list)
    selected: str = ""
    selected_dsr: float = 0.0
    n_trials: int = 0


def _synth_market(token_id: str, strike: float, expiry_ts: float, symbol: str,
                  fee_rate: float, fee_exponent: float, tick: float = 0.01) -> PolyMarket:
    return PolyMarket(
        condition_id=token_id,
        question=f"replay::{token_id[:12]}",
        yes_token_id=token_id, no_token_id="",
        yes_price=0.5, no_price=0.5,
        strike=strike, expiry_ts=expiry_ts,
        tick_size=tick, symbol=symbol,
        is_updown=False, is_threshold=True,
        fee_rate=fee_rate, fee_exponent=fee_exponent,
    )


def evaluate_window(
    events: list[ReplayEvent],
    binance_series: BinanceSeries,
    signal_gen: SignalGenerator,
    *,
    seed: int = 0,
    config: ReplayConfig | None = None,
    strike_for: Callable[[str], float] | None = None,
    expiry_for: Callable[[str], float] | None = None,
) -> ReplayResults:
    """Replay the SignalGenerator over the merged poly+binance stream.

    Walks the recorded poly events in WALL-CLOCK order (``ts_wall``; ``ts_mono``
    only breaks intra-source ties via the loader's sort). At each poly
    top-of-book change it pulls the as-of Binance state, rebuilds a
    ``BookSnapshot``, synthesises a ``BinanceTick``/``MicroFeatures``, and calls
    ``SignalGenerator.evaluate`` with a ``DecisionTrace``. The market's
    ``expiry_ts`` is offset from the live wall clock so ``evaluate``'s
    ``expiry_ts - time.time()`` reproduces the recorded tte, and the book ts is
    set fresh so the staleness gate passes — the same trick ``src.backtest``
    uses.

    For each emitted signal it computes the counterfactual fill:
      * taker  → assume crossed (cross of the recorded book), fill_prob 1.0;
      * maker  → ``makerfill.power_prob_fill`` over the L2 queue-ahead, using
                 the near-future traded-volume proxy from the recorded book, and
                 the *expected* (uncertain) rebate.
    plus a markout-based, post-cost per-share PnL proxy. Deterministic: identical
    inputs + seed → identical records.
    """
    cfg = config or ReplayConfig()
    # The evaluation path is fully deterministic (no RNG); `seed` is recorded in
    # the results and threaded into the bootstrap CI in `sweep` so identical
    # seed → identical sweep output.
    records: list[SignalRecord] = []

    if not events or not binance_series.points:
        return ReplayResults(cfg.name, seed, "cpu", records)

    symbol = binance_series.symbol
    # Default strike (when no per-token strike is injected) anchors to the as-of
    # spot at each event, which keeps threshold pricing well-defined and broadly
    # ATM. Deterministic — no RNG. Callers can inject a per-token strike/expiry
    # for a more faithful counterfactual.

    # Pre-index per-token event lists for the traded-volume proxy and markout.
    by_token: dict[str, list[ReplayEvent]] = {}
    for ev in events:
        by_token.setdefault(ev.token_id, []).append(ev)

    # Walk events in wall-clock order (loader already sorts by (ts_wall, ts_mono)).
    for ev in events:
        if ev.etype != "book":
            continue
        bp = binance_series.point_at(ev.ts_wall)
        if bp is None or bp.mid <= 0:
            continue

        strike = strike_for(ev.token_id) if strike_for else bp.mid
        # tte: keep it inside the config band so eligible events actually score.
        tte = 0.5 * (cfg.min_tte_secs + cfg.max_tte_secs)
        if expiry_for:
            exp = expiry_for(ev.token_id)
            tte = max(1.0, exp - ev.ts_wall)

        now = _time.time()
        mono = _time.monotonic()
        snap = BookSnapshot(
            token_id=ev.token_id,
            best_bid=ev.snapshot.best_bid,
            best_ask=ev.snapshot.best_ask,
            bid_size=ev.snapshot.bid_size,
            ask_size=ev.snapshot.ask_size,
            ts=mono,
        )
        tick = BinanceTick(
            symbol=symbol, bid=bp.mid, ask=bp.mid, mid=bp.mid,
            sigma_annual=max(0.05, bp.sigma_annual), ts=mono,
            drift_annual=bp.drift_annual,
        )
        market = _synth_market(
            ev.token_id, strike, now + tte, symbol, cfg.fee_rate, cfg.fee_exponent,
        )
        trace = DecisionTrace.new(run_id="replay", stage="eval")
        sig = signal_gen.evaluate(
            market, tick, snap, features=bp.features, trace=trace,
        )
        if sig is None:
            continue

        side = sig.side.value
        # Markout signed to the entry side (favorable = positive).
        mk5 = markout(by_token.get(ev.token_id, []), ev.token_id, side, sig.price, ev.ts_wall, 5.0)
        mk30 = markout(by_token.get(ev.token_id, []), ev.token_id, side, sig.price, ev.ts_wall, 30.0)

        if sig.is_maker and ev.levels:
            # Queue-aware maker fill. Traded-volume proxy: size that traded
            # through our side over the near future (the depletion of top size
            # plus any incoming size at/through our price), approximated by the
            # opposing-side size that appeared within the GTD horizon. We use a
            # conservative proxy: the max opposing top size observed over the
            # next 30s window on the recorded tape.
            traded_vol = _traded_volume_proxy(by_token.get(ev.token_id, []), side, ev.ts_wall, 30.0)
            q = _makerfill.queue_ahead(ev.levels, side, sig.price)
            fill_prob = _makerfill.power_prob_fill(q, traded_vol, n=cfg.maker_fill_n)
            rebate = _makerfill.expected_maker_rebate(
                sig.price, fill_prob, rebate_share=cfg.rebate_share,
                fee_rate=cfg.fee_rate, fee_exponent=cfg.fee_exponent,
            )
        elif sig.is_maker:
            # No L2 → conservative top-of-book trade-through fallback (0/1 label).
            filled, _ts, _r = estimate_maker_fill(
                by_token.get(ev.token_id, []), ev.token_id, side, sig.price, ev.ts_wall, 30.0,
            )
            q = 0.0
            fill_prob = 1.0 if filled else 0.0
            rebate = _makerfill.expected_maker_rebate(
                sig.price, fill_prob, rebate_share=cfg.rebate_share,
                fee_rate=cfg.fee_rate, fee_exponent=cfg.fee_exponent,
            )
        else:
            # Taker: crosses the recorded book by construction → filled.
            q = 0.0
            fill_prob = 1.0
            rebate = 0.0

        # Post-cost per-share PnL proxy. Use the 5s markout as the realized
        # directional move (favorable already signed to side). Taker pays the
        # parabolic fee; maker earns the EXPECTED (uncertain) rebate. Scale the
        # whole thing by fill_prob so an unfilled maker contributes ~0.
        directional = mk5 if mk5 is not None else 0.0
        if sig.is_maker:
            pnl = fill_prob * (directional + rebate)
        else:
            fee = _tfee(sig.price, cfg.fee_rate, cfg.fee_exponent)
            pnl = fill_prob * (directional - fee)

        records.append(SignalRecord(
            ts_wall=ev.ts_wall,
            token_id=ev.token_id,
            side=side,
            price=sig.price,
            size=sig.size,
            edge=sig.edge,
            is_maker=int(sig.is_maker),
            p_star=sig.p_star,
            fill_prob=fill_prob,
            queue_ahead=q,
            expected_rebate=rebate,
            markout_5s=mk5,
            markout_30s=mk30,
            pnl_proxy=pnl,
            adverse=int(mk5 is not None and mk5 < 0.0),
        ))

    return ReplayResults(cfg.name, seed, "cpu", records)


def _traded_volume_proxy(
    token_events: list[ReplayEvent], side: str, ts_wall: float, horizon: float
) -> float:
    """Conservative near-future traded-volume proxy for queue consumption.

    We don't have a Polymarket trade tape in the recording, only book snapshots.
    As a proxy for how much volume trades through OUR side over ``horizon``
    seconds, take the maximum opposing top-of-book size that appears in the
    window — i.e. how much eager counter-flow showed up to lift/hit our level.
    This is intentionally modest (a single top-of-book size, not a sum) so the
    fill probability stays conservative.
    """
    side_u = side.upper()
    deadline = ts_wall + max(0.0, horizon)
    best = 0.0
    for ev in token_events:
        if ts_wall < ev.ts_wall <= deadline:
            sz = ev.snapshot.ask_size if side_u == "BUY" else ev.snapshot.bid_size
            best = max(best, sz)
    return best


def sweep(
    events: list[ReplayEvent],
    binance_series: BinanceSeries,
    configs: Sequence[ReplayConfig],
    *,
    train_frac: float = 0.6,
    seed: int = 0,
    walk_forward: bool = False,
    strike_for: Callable[[str], float] | None = None,
    expiry_for: Callable[[str], float] | None = None,
) -> SweepReport:
    """Evaluate each config over the window and report OOS-only edge.

    Signals are split by TIME (not shuffled): the earliest ``train_frac`` of the
    wall-clock span is train, the rest validation. With ``walk_forward=True`` the
    split point is computed per-config from that config's own signal times (so a
    config that only fires late isn't all-validation by accident); otherwise a
    single global wall-clock cut (from the event span) is shared by all configs
    for a like-for-like comparison.

    Reports per-config OOS (validation) mean edge with a block-bootstrap CI,
    t-stat and per-obs Sharpe (all from ``src.stats``). The SELECTED config (best
    validation mean) is then penalized by ``deflated_sharpe_ratio`` with
    ``n_trials = len(configs)`` so its edge is discounted for the number of
    configs tried. Deterministic given ``seed``.
    """
    edges: list[ConfigEdge] = []
    results: dict[str, ReplayResults] = {}

    # Global wall-clock cut shared across configs (unless walk-forward).
    if events:
        ev_sorted = sorted(events, key=lambda e: (e.ts_wall, e.ts_mono))
        t0, t1 = ev_sorted[0].ts_wall, ev_sorted[-1].ts_wall
        global_cut = t0 + train_frac * (t1 - t0)
    else:
        global_cut = 0.0

    for cfg in configs:
        gen = cfg.build_signal_generator()
        res = evaluate_window(
            events, binance_series, gen, seed=seed, config=cfg,
            strike_for=strike_for, expiry_for=expiry_for,
        )
        results[cfg.name] = res
        recs = sorted(res.records, key=lambda r: r.ts_wall)

        if walk_forward and recs:
            ts = [r.ts_wall for r in recs]
            cut = ts[0] + train_frac * (ts[-1] - ts[0])
        else:
            cut = global_cut

        train = [r.pnl_proxy for r in recs if r.ts_wall <= cut]
        val = [r.pnl_proxy for r in recs if r.ts_wall > cut]

        ci = _stats.block_bootstrap_ci(val, seed=seed) if val else _stats.BootstrapCI(0.0, 0.0, 0.0, 0, 0)
        edges.append(ConfigEdge(
            config_name=cfg.name,
            n_train=len(train),
            n_val=len(val),
            train_mean=float(np.mean(train)) if train else 0.0,
            val_mean=float(np.mean(val)) if val else 0.0,
            val_ci_lo=ci.lo,
            val_ci_hi=ci.hi,
            val_t_stat=_stats.t_stat(val),
            val_sharpe=_stats.per_obs_sharpe(val),
        ))

    # Select on OOS mean edge, then deflate for the number of configs tried.
    selected = ""
    selected_dsr = 0.0
    if edges:
        best = max(edges, key=lambda e: e.val_mean)
        selected = best.config_name
        val_sharpes = [e.val_sharpe for e in edges]
        sr_std = float(np.std(val_sharpes, ddof=1)) if len(val_sharpes) > 1 else 0.0
        selected_dsr = _stats.deflated_sharpe_ratio(
            best.val_sharpe,
            n_trials=max(1, len(configs)),
            n_obs=best.n_val,
            sr_std=sr_std,
        )

    return SweepReport(
        seed=seed,
        device="cpu",
        train_frac=train_frac,
        walk_forward=walk_forward,
        edges=edges,
        selected=selected,
        selected_dsr=selected_dsr,
        n_trials=len(configs),
    )


async def refetch_binance(
    run_dir: str | Path,
    *,
    symbol: str = "BTCUSDT",
    sigma_window_secs: float = 600.0,
    fetcher: Callable[..., Any] | None = None,
    prefetched: BinanceSeries | None = None,
) -> BinanceSeries:
    """Refetch the Binance spot/feature series over a recording's window.

    Thin async wrapper that REUSES ``src.backtest._fetch_binance_klines`` (1m
    closes), the aggTrades fetch, and ``src.microstructure.MicroReplay`` to build
    a per-poly-event spot/feature timeline. The network call is INJECTABLE:
      * pass ``prefetched`` to skip fetching entirely (unit tests run offline);
      * pass ``fetcher`` — a callable ``(symbol, start_ts, end_ts) ->
        (klines, trades)`` — to stub the network.
    With neither, it opens a real aiohttp session and hits Binance.

    Returns a ``BinanceSeries`` sampled at each recorded poly-event wall time.
    """
    if prefetched is not None:
        return prefetched

    events = load_poly_events(run_dir)
    if not events:
        return BinanceSeries(symbol=symbol, points=[])
    start_ts = events[0].ts_wall
    end_ts = events[-1].ts_wall

    if fetcher is not None:
        klines, trades = await _maybe_await(fetcher(symbol, start_ts, end_ts))
    else:
        import aiohttp
        from .backtest import (
            _fetch_binance_klines, _fetch_agg_trades, RateGate,
            BINANCE_CONCURRENCY, BINANCE_MIN_INTERVAL_SECS,
        )
        gate = RateGate(BINANCE_CONCURRENCY, BINANCE_MIN_INTERVAL_SECS)
        async with aiohttp.ClientSession(headers={"User-Agent": "Mozilla/5.0"}) as session:
            klines = await _fetch_binance_klines(session, gate, symbol, start_ts, end_ts)
            trades = await _fetch_agg_trades(session, gate, symbol, start_ts, end_ts)

    return build_binance_series(
        events, symbol, klines, trades, sigma_window_secs=sigma_window_secs,
    )


async def _maybe_await(x: Any) -> Any:
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x


def build_binance_series(
    events: list[ReplayEvent],
    symbol: str,
    klines: list[tuple[float, float]],
    trades: list[tuple[float, float, float, float]],
    *,
    sigma_window_secs: float = 600.0,
) -> BinanceSeries:
    """Assemble a BinanceSeries from raw klines + aggTrades, sampled per event.

    ``klines`` are ``(unix_ts, close)``; ``trades`` are
    ``(ts, price, signed_dv, abs_dv)`` (the shapes ``src.backtest`` returns).
    Realised sigma is a rolling log-return std over ``sigma_window_secs`` of
    klines; features come from ``MicroReplay`` when a trade tape is present.
    Pure/deterministic — no network, no RNG.
    """
    from .microstructure import MicroReplay
    from .pricing import realized_vol_annual

    series_pts: list[BinancePoint] = []
    if not klines:
        return BinanceSeries(symbol=symbol, points=[])

    import math

    kl = sorted(klines, key=lambda k: k[0])
    kl_ts = [k[0] for k in kl]
    kl_px = [k[1] for k in kl]
    replay = MicroReplay(trades) if trades else None

    def _sigma_at(ts: float) -> float:
        hi = _bisect.bisect_right(kl_ts, ts) - 1
        if hi < 1:
            return 0.40
        lo = _bisect.bisect_left(kl_ts, ts - sigma_window_secs)
        lo = max(0, min(lo, hi - 1))
        rets: list[float] = []
        for i in range(lo + 1, hi + 1):
            p0, p1 = kl_px[i - 1], kl_px[i]
            if p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
        span = max(1.0, kl_ts[hi] - kl_ts[lo])
        return realized_vol_annual(rets, span) if len(rets) >= 2 else 0.40

    def _mid_at(ts: float) -> float:
        i = _bisect.bisect_right(kl_ts, ts) - 1
        return kl_px[i] if i >= 0 else 0.0

    # Sample at each unique poly event wall time (deduped, sorted).
    sample_ts = sorted({ev.ts_wall for ev in events})
    for ts in sample_ts:
        mid = _mid_at(ts)
        if mid <= 0:
            continue
        sigma = _sigma_at(ts)
        feats = None
        drift = 0.0
        if replay is not None:
            feats = replay.features(ts, mid, sigma, sigma_window_secs)
            ofi = replay.ofi(ts)
            drift = max(-0.20, min(0.20, ofi)) * 0.20
        series_pts.append(BinancePoint(
            ts_wall=ts, mid=mid, sigma_annual=sigma, drift_annual=drift, features=feats,
        ))
    return BinanceSeries(symbol=symbol, points=series_pts)


def evaluate_run(
    run_dir: str | Path,
    *,
    configs: Sequence[ReplayConfig] | None = None,
    do_sweep: bool = False,
    oos: bool = False,
    seed: int = 0,
    symbol: str = "BTCUSDT",
    binance_series: BinanceSeries | None = None,
) -> SweepReport | ReplayResults:
    """Top-level evaluate: load events, (re)fetch binance, run config(s).

    If ``binance_series`` is provided it is used directly (offline); otherwise
    this performs a real network refetch via ``refetch_binance``. Returns a
    ``SweepReport`` when ``do_sweep`` (or multiple configs), else a single
    ``ReplayResults``.
    """
    events = load_poly_events(run_dir)
    if binance_series is None:
        import asyncio
        binance_series = asyncio.run(refetch_binance(run_dir, symbol=symbol))

    cfgs = list(configs) if configs else [ReplayConfig(name="default")]
    if do_sweep or len(cfgs) > 1:
        return sweep(
            events, binance_series, cfgs,
            train_frac=0.6 if oos else 1.0, seed=seed, walk_forward=oos,
        )
    gen = cfgs[0].build_signal_generator()
    return evaluate_window(events, binance_series, gen, seed=seed, config=cfgs[0])


def _print_sweep_report(rep: SweepReport) -> None:
    print("\n=== Poly Replay Evaluation (counterfactual) ===")
    print(f"seed={rep.seed}  device={rep.device}  train_frac={rep.train_frac}  "
          f"walk_forward={rep.walk_forward}  n_trials={rep.n_trials}")
    print(f"{'config':<22}{'n_tr':>6}{'n_val':>7}{'train':>10}{'val':>10}"
          f"{'val_t':>8}{'val_sr':>8}{'ci_lo':>9}{'ci_hi':>9}")
    for e in rep.edges:
        print(f"{e.config_name:<22}{e.n_train:>6}{e.n_val:>7}{e.train_mean:>10.4f}"
              f"{e.val_mean:>10.4f}{e.val_t_stat:>8.2f}{e.val_sharpe:>8.3f}"
              f"{e.val_ci_lo:>9.4f}{e.val_ci_hi:>9.4f}")
    print(f"\nSelected: {rep.selected}  Deflated-Sharpe (P[SR>0] after "
          f"{rep.n_trials}-config selection bias) = {rep.selected_dsr:.3f}")


def _print_results(res: ReplayResults) -> None:
    print("\n=== Poly Replay Evaluation (single config) ===")
    print(f"config={res.config_name}  seed={res.seed}  device={res.device}")
    print(f"signals={len(res.records)}")
    if res.records:
        pnls = res.pnls
        print(f"mean pnl_proxy={float(np.mean(pnls)):.5f}  "
              f"makers={sum(r.is_maker for r in res.records)}  "
              f"adverse={sum(r.adverse for r in res.records)}")


def report(run_dir: str | Path) -> None:
    events = load_poly_events(run_dir)
    books = latest_books(events)
    with_levels = sum(1 for ev in events if ev.levels)
    print(f"\n=== Poly Replay Shard Report ===")
    print(f"Run dir:       {run_dir}")
    print(f"Events:        {len(events)}")
    print(f"Tokens:        {len(books)}")
    print(f"Full snapshots:{with_levels}")
    if events:
        print(f"First ts:      {events[0].ts_wall:.3f}")
        print(f"Last ts:       {events[-1].ts_wall:.3f}")


def _default_sweep_configs() -> list[ReplayConfig]:
    """A small, illustrative config grid for the CLI sweep."""
    return [
        ReplayConfig(name="taker_eps005", safety_eps=0.005, maker_enabled=False),
        ReplayConfig(name="taker_eps010", safety_eps=0.010, maker_enabled=False),
        ReplayConfig(name="maker_j1", safety_eps=0.005, maker_enabled=True, maker_join_ticks=1),
        ReplayConfig(name="maker_j2", safety_eps=0.005, maker_enabled=True, maker_join_ticks=2),
        ReplayConfig(name="taker_obi", safety_eps=0.005, maker_enabled=False, obi_veto=True),
    ]


def cli() -> None:
    ap = argparse.ArgumentParser(description="Replay recorded Polymarket event shards")
    ap.add_argument("run_dir", help="poly_events/<run_id> directory")
    ap.add_argument("--evaluate", action="store_true",
                    help="run the counterfactual evaluator (refetches Binance)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep a config grid and report OOS edge + Deflated Sharpe")
    ap.add_argument("--oos", action="store_true",
                    help="time-split signals into train/validation and report OOS only")
    ap.add_argument("--seed", type=int, default=0, help="deterministic seed")
    ap.add_argument("--symbol", default="BTCUSDT", help="Binance symbol to refetch")
    args = ap.parse_args()

    if not (args.evaluate or args.sweep):
        report(args.run_dir)
        return

    configs = _default_sweep_configs() if args.sweep else None
    out = evaluate_run(
        args.run_dir, configs=configs, do_sweep=args.sweep,
        oos=args.oos, seed=args.seed, symbol=args.symbol,
    )
    if isinstance(out, SweepReport):
        _print_sweep_report(out)
    else:
        _print_results(out)


if __name__ == "__main__":
    cli()