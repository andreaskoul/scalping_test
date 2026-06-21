"""
Orchestrator — wires Binance WS, Polymarket WS, signal generator,
risk manager, and executor into a single async event loop.

Usage:
  python -m src.main             # paper-trade (safe default)
  python -m src.main --live      # REAL orders — requires POLY_PRIVATE_KEY
  python -m src.main --help

Paper-trade mode runs the full stack against live feeds but never sends
a real order. Logs every signal to fills.db for post-mortem analysis.

Hot-path design:
  - The eval loop is **event-driven**: it awaits the OR of (any Binance
    bookTicker arrival, any Polymarket book change, idle timeout). On
    wake, it evaluates only the markets whose underlying data could
    have moved — Binance-symbol-dirty markets if a tick arrived,
    Polymarket-token-dirty markets if a book changed. This replaces the
    prior fixed 50 ms polling sleep and cuts mean reaction lag by ~25 ms.
  - The universe refresh runs as a background task with atomic swap so
    it doesn't block the event loop during the 1–3 s Gamma fetch.
  - Heartbeat counters are cumulative across the heartbeat window
    (previously they reset every iteration, which made the displayed
    numbers meaningless).
"""

import argparse
import asyncio
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

import aiohttp

from .arbitrage import scan_combos
from .binance_ws import BinanceWS
from .category import CategoryTracker
from .config import Settings
from .deribit_iv import DeribitIV
from .execute import Executor
from .funding import FundingOracle
from .kalshi import KalshiClient, find_xvenue_arbs
from .meanrev import MeanReversionTracker
from .microstructure import DirectionalModel, MicrostructureEngine, MicroFeatures
from .poly_universe import fetch_active_markets
from .poly_ws import PolyWS
from .pricing import implied_prob, SIGMA_MIN, load_wedge_coeffs, load_calibration
from .resolution import PriceToBeatCache, ChainlinkBasis
from .risk import RiskManager
from .signal import Side, Signal, SignalGenerator
from .sizing import kelly_size

log = logging.getLogger(__name__)

UNIVERSE_REFRESH_SECS = 30.0    # re-poll Gamma API for new markets
DIRTY_POLL_SECS = 0.005         # 5ms backoff when nothing has changed
FULL_SWEEP_SECS = 1.0           # backstop: re-evaluate everything at least once/sec
HEARTBEAT_SECS = 30.0


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def _compute_blacklist_set(tracker: CategoryTracker) -> set[str]:
    return {
        k for k, s in tracker.stats.items()
        if s.fills >= tracker.min_fills and s.realised_pnl <= tracker.threshold
    }


async def _universe_refresher(
    session: aiohttp.ClientSession,
    poly_ws: PolyWS,
    binance_clients: dict,
    state: dict,
    price_to_beat: "PriceToBeatCache | None" = None,
    use_ptb: bool = False,
) -> None:
    """Refresh the market universe in the background without blocking the
    eval loop. Writes the new market list and per-symbol partition into
    `state` atomically.

    When `use_ptb` is set, Up/Down strikes are anchored to the real Chainlink
    Price-to-Beat (window-open reference) here — off the hot path — replacing
    the old first-observation hack that forced SKIP_UPDOWN."""
    while True:
        try:
            markets = await fetch_active_markets(session)
            if use_ptb and price_to_beat is not None:
                for m in markets:
                    if m.is_updown and m.strike <= 0:
                        try:
                            k = await price_to_beat.anchored_strike(session, m)
                            if k:
                                m.strike = k
                        except Exception as exc:
                            log.debug("Price-to-Beat anchor failed: %s", exc)
            token_ids = []
            by_symbol: dict[str, list] = {sym: [] for sym in binance_clients}
            missing: dict[str, int] = {}
            for m in markets:
                token_ids.append(m.yes_token_id)
                token_ids.append(m.no_token_id)
                if m.symbol in by_symbol:
                    by_symbol[m.symbol].append(m)
                else:
                    missing[m.symbol] = missing.get(m.symbol, 0) + 1
            poly_ws.update_tokens(token_ids)
            # Atomic swap — readers see the old or new list, never partial.
            state["markets"] = markets
            state["by_symbol"] = by_symbol
            # Index from token_id → market, so book-change events can be
            # mapped to the relevant market in O(1).
            state["by_token"] = {m.yes_token_id: m for m in markets}
            log.info("Universe refreshed: %d markets", len(markets))
            if missing:
                log.warning(
                    "Skipping %d markets — no Binance feed for symbols: %s. "
                    "Add to BINANCE_SYMBOLS env var to enable.",
                    sum(missing.values()),
                    ",".join(f"{s}({n})" for s, n in missing.items()),
                )
        except Exception as exc:
            log.warning("Universe refresh failed: %s", exc)
        await asyncio.sleep(UNIVERSE_REFRESH_SECS)


async def _xvenue_loop(
    session: aiohttp.ClientSession,
    kalshi: KalshiClient,
    poly_ws: PolyWS,
    state: dict,
    min_credit: float,
    interval: float = 5.0,
) -> None:
    """Detect Polymarket↔Kalshi cross-venue lock-$1 arbs and log them.

    Detect-and-log only: two-venue *execution* needs Kalshi credentials and a
    Kalshi executor (see kalshi.py / RUNBOOK).  Surfacing the opportunity is
    the first, safe step."""
    while True:
        try:
            kalshi_markets = await kalshi.fetch_btc_markets(session)
            markets = state.get("markets", [])
            if kalshi_markets and markets:
                arbs = find_xvenue_arbs(markets, poly_ws, kalshi_markets, min_credit=min_credit)
                if arbs:
                    log.info("X-VENUE: %d cross-venue opportunities this scan", len(arbs))
        except Exception as exc:
            log.debug("X-venue scan failed: %s", exc)
        await asyncio.sleep(interval)


async def main(paper: bool, log_level: str = "INFO", duration_secs: float = 0.0) -> None:
    _setup_logging(log_level)

    private_key = os.getenv("POLY_PRIVATE_KEY")
    if not paper and not private_key:
        raise SystemExit("POLY_PRIVATE_KEY must be set in .env for live mode")

    symbols = [s.strip().lower() for s in os.getenv("BINANCE_SYMBOLS", "btcusdt,ethusdt").split(",") if s.strip()]
    max_notional = float(os.getenv("MAX_NOTIONAL_PER_TRADE", "25"))
    max_per_min = float(os.getenv("MAX_NOTIONAL_PER_MINUTE", "200"))
    drawdown_stop = float(os.getenv("DAILY_DRAWDOWN_STOP", "500"))
    safety_eps = float(os.getenv("EDGE_SAFETY_EPS", "0.02"))
    cooldown = float(os.getenv("COOLDOWN_SECS", "5.0"))
    binance_stale = float(os.getenv("BINANCE_STALE_SECS", "2.0"))
    poly_stale = float(os.getenv("POLY_STALE_SECS", "5.0"))
    price_min = float(os.getenv("PRICE_MIN", "0.10"))
    price_max = float(os.getenv("PRICE_MAX", "0.90"))
    sigma_floor = float(os.getenv("SIGMA_FLOOR", "0.40"))
    skip_updown = os.getenv("SKIP_UPDOWN", "1") not in ("0", "false", "False")
    min_tte_secs = float(os.getenv("MIN_TTE_SECS", "180"))
    max_tte_secs = float(os.getenv("MAX_TTE_SECS", "3600"))
    book_max_age_secs = float(os.getenv("BOOK_MAX_AGE_SECS", "2.0"))

    cfg = Settings.from_env()
    effective_skip_updown = skip_updown and not cfg.use_price_to_beat

    mode = "PAPER" if paper else "LIVE"
    log.info("=== Polymarket-vs-Binance arb bot starting [%s] ===", mode)
    log.info(
        "Binance symbols: %s | safety_eps=%.4f cooldown=%.1fs notional<=%.0f "
        "| price=[%.2f,%.2f] σ_floor=%.2f tte=[%.0f,%.0f]s skip_updown=%s",
        symbols, safety_eps, cooldown, max_notional,
        price_min, price_max, sigma_floor, min_tte_secs, max_tte_secs, effective_skip_updown,
    )
    if duration_secs > 0:
        log.info("Will stop after %.0f seconds and print PnL summary.", duration_secs)

    binance_clients = {sym: BinanceWS(sym, stale_threshold_secs=binance_stale) for sym in symbols}
    poly_ws = PolyWS(token_ids=[], stale_threshold_secs=poly_stale)
    iv_oracle = DeribitIV(symbols=symbols)
    effective_spread_mult = float(os.getenv("EFFECTIVE_SPREAD_MULT", "1.3"))
    max_walk_slippage = float(os.getenv("MAX_WALK_SLIPPAGE", "0.05"))
    arb_min_credit = float(os.getenv("ARB_MIN_CREDIT", "0.01"))
    arb_enabled = os.getenv("ARB_ENABLED", "1") not in ("0", "false", "False")

    # ---- research-driven overlays (max-EV config; see RUNBOOK.md) ----
    log.info("Overlays: %s", cfg.summary())
    funding_oracle = FundingOracle(symbols)
    micro_engine = MicrostructureEngine(
        model=DirectionalModel.load(),
        obi_veto=cfg.obi_veto,
        obi_veto_threshold=cfg.obi_veto_threshold,
        momentum_enabled=cfg.momentum_enabled,
        impulse_fade_enabled=cfg.impulse_fade_enabled,
        momentum_min_move_usd=cfg.momentum_min_move_usd,
        momentum_near_expiry_secs=cfg.momentum_window_secs,
    )
    chainlink_basis = ChainlinkBasis()
    price_to_beat = PriceToBeatCache(basis=chainlink_basis)
    meanrev = (
        MeanReversionTracker(
            band=cfg.meanrev_band,
            half_life_secs=cfg.meanrev_half_life_secs,
            max_hold_secs=cfg.meanrev_max_hold_secs,
        )
        if cfg.meanrev_enabled else None
    )
    kalshi_client = KalshiClient(cfg.kalshi_api_base) if cfg.xvenue_enabled else None

    # Self-calibration: hot-load wedge + recalibration fitted by src.calibrate
    # from our own resolved fills (falls back to the paper prior if absent).
    fitted_wedge = load_wedge_coeffs("wedge_coeffs.json")
    fitted_calib = load_calibration("calib_coeffs.json")
    if fitted_wedge:
        log.info("Loaded fitted wedge coeffs: %s", fitted_wedge)
    if fitted_calib:
        log.info("Loaded p* recalibration: a=%.4f b=%.4f", *fitted_calib)

    signal_gen = SignalGenerator(
        max_notional_per_trade=max_notional,
        safety_eps=safety_eps,
        cooldown_secs=cooldown,
        price_min=price_min,
        price_max=price_max,
        sigma_floor=sigma_floor,
        skip_updown=effective_skip_updown,
        min_tte_secs=min_tte_secs,
        max_tte_secs=max_tte_secs,
        book_max_age_secs=book_max_age_secs,
        iv_oracle=iv_oracle,
        poly_ws=poly_ws,
        effective_spread_mult=effective_spread_mult,
        max_walk_slippage=max_walk_slippage,
        longshot_tilt_mult=cfg.longshot_tilt_mult,
        skew_coef=cfg.skew_coef,
        sell_price_min=cfg.sell_price_min,
        micro_engine=micro_engine,
        ml_overlay=cfg.ml_overlay,
        ml_weight=cfg.ml_weight,
        obi_veto=cfg.obi_veto,
        obi_veto_threshold=cfg.obi_veto_threshold,
        kelly_enabled=cfg.kelly_enabled,
        kelly_fraction=cfg.kelly_fraction,
        maker_enabled=cfg.maker_enabled,
        maker_join_ticks=cfg.maker_join_ticks,
        wedge_coeffs=fitted_wedge,
        calib=fitted_calib,
    )
    risk = RiskManager(
        max_notional_per_trade=max_notional,
        max_notional_per_minute=max_per_min,
        daily_drawdown_stop=drawdown_stop,
        binance_stale_secs=binance_stale,
        poly_stale_secs=poly_stale,
    )
    executor = Executor(
        paper=paper,
        maker_fill_prob=cfg.maker_fill_prob,
        maker_gtd_secs=cfg.maker_gtd_secs,
    )
    await executor.setup(private_key=private_key)

    category_tracker = CategoryTracker.load()
    blacklist = _compute_blacklist_set(category_tracker)
    if category_tracker.stats:
        log.info(
            "Loaded category tracker: %d known, %d currently blacklisted",
            len(category_tracker.stats), len(blacklist),
        )

    # Atomic universe state — refreshed by the background task.
    universe_state: dict = {
        "markets": [], "by_symbol": {sym: [] for sym in symbols}, "by_token": {},
    }

    session = aiohttp.ClientSession()
    tasks = [
        asyncio.create_task(c.run(), name=f"binance-{sym}")
        for sym, c in binance_clients.items()
    ]
    tasks.append(asyncio.create_task(poly_ws.run(), name="poly-ws"))
    tasks.append(asyncio.create_task(iv_oracle.run(), name="deribit-iv"))
    tasks.append(asyncio.create_task(funding_oracle.run(), name="funding"))
    tasks.append(asyncio.create_task(
        _universe_refresher(
            session, poly_ws, binance_clients, universe_state,
            price_to_beat=price_to_beat, use_ptb=cfg.use_price_to_beat,
        ),
        name="universe",
    ))
    if kalshi_client is not None:
        tasks.append(asyncio.create_task(
            _xvenue_loop(session, kalshi_client, poly_ws, universe_state, cfg.xvenue_min_credit),
            name="xvenue",
        ))

    # Heartbeat-window cumulative counters.
    cum_eval = cum_no_spot = cum_no_book = cum_warmup = cum_signals = cum_fills = 0
    cum_arbs = cum_arb_fills = 0
    top_edge_seen = (-1.0, "")

    last_heartbeat = time.monotonic()
    last_full_sweep = 0.0
    deadline = (time.time() + duration_secs) if duration_secs > 0 else 0.0

    try:
        while True:
            if risk.is_halted():
                log.warning("Bot halted. Sleeping...")
                await asyncio.sleep(5)
                continue

            now_wall = time.time()
            if deadline and now_wall >= deadline:
                log.info("Duration reached (%.0fs). Shutting down.", duration_secs)
                break

            # ---- Drain dirty signals from Binance + Polymarket. ----
            dirty_symbols = set()
            for sym, c in binance_clients.items():
                if c.tick_event.is_set():
                    c.tick_event.clear()
                    dirty_symbols.add(sym)
            dirty_tokens = poly_ws.drain_dirty()

            now_mono = time.monotonic()
            had_event = bool(dirty_symbols or dirty_tokens)

            # If nothing happened recently, short-sleep instead of busy-looping.
            # Periodic full sweep catches σ/IV/drift drift even on quiet feeds.
            need_full_sweep = (now_mono - last_full_sweep) >= FULL_SWEEP_SECS
            if not had_event and not need_full_sweep:
                await asyncio.sleep(DIRTY_POLL_SECS)
                continue

            markets = universe_state["markets"]
            by_symbol = universe_state["by_symbol"]
            by_token = universe_state["by_token"]
            if not markets:
                # Universe not loaded yet — wait briefly and retry.
                await asyncio.sleep(DIRTY_POLL_SECS)
                continue

            # ---- Candidate set: dirty-driven on events, full sweep otherwise. ----
            if had_event and not need_full_sweep:
                seen: set = set()
                candidates: list = []
                for sym in dirty_symbols:
                    for m in by_symbol.get(sym, ()):
                        if m.condition_id not in seen:
                            seen.add(m.condition_id)
                            candidates.append(m)
                for tid in dirty_tokens:
                    m = by_token.get(tid)
                    if m is not None and m.condition_id not in seen:
                        seen.add(m.condition_id)
                        candidates.append(m)
            else:
                candidates = markets
                last_full_sweep = now_mono

            for market in candidates:
                bc = binance_clients.get(market.symbol)
                if bc is None:
                    continue
                tick = bc.snapshot()
                if tick is None:
                    cum_no_spot += 1
                    continue

                # Price against the settlement feed: shift Binance spot by the
                # tracked Chainlink basis so p* reflects what the market resolves
                # on. No-op until a published Price-to-Beat diverges from Binance.
                if cfg.chainlink_basis_adj:
                    _b = chainlink_basis.value(market.symbol)
                    if _b:
                        tick.mid += _b
                        tick.bid += _b
                        tick.ask += _b

                # Up/Down strike anchoring. With Price-to-Beat on, the
                # refresher anchors to the real Chainlink window-open ref; if
                # it hasn't yet, skip rather than fall back to the biased
                # first-observation hack. Threshold markets already have a
                # parsed strike, so this only gates unanchored Up/Down.
                if market.strike <= 0:
                    if market.is_updown and cfg.use_price_to_beat:
                        continue
                    market.strike = tick.mid

                yes_book = poly_ws.snapshot(market.yes_token_id)
                if yes_book is None:
                    cum_no_book += 1
                    continue

                if tick.sigma_annual <= SIGMA_MIN:
                    cum_warmup += 1
                    continue

                cum_eval += 1

                # Pre-evaluate blacklist by category (cheap set lookup).
                # Note: the exec_price isn't known yet, but the kind/tte
                # buckets dominate the key — we use the book mid as a
                # proxy.  If the bucket is blacklisted, skip without
                # paying for evaluate().
                tte = market.expiry_ts - now_wall
                proxy_px = (yes_book.best_bid + yes_book.best_ask) * 0.5
                cat_key = CategoryTracker.category_key(
                    market.question, market.symbol, tte, proxy_px,
                )
                if cat_key in blacklist:
                    continue

                # Microstructure features + perp-funding carry for this market.
                feats = MicroFeatures(
                    ofi=bc.ofi_ratio(),
                    ret_fast=bc.recent_return(5.0),
                    ret_slow=bc.recent_return(60.0),
                    rvol=tick.sigma_annual,
                    spread_rel=((tick.ask - tick.bid) / tick.mid) if tick.mid > 0 else 0.0,
                    move_window_usd=bc.price_move(cfg.momentum_window_secs),
                    secs_to_expiry=market.expiry_ts - now_wall,
                )
                carry = 0.0
                if cfg.carry_from_funding:
                    carry += funding_oracle.carry(market.symbol)
                carry += funding_oracle.fade_drift(market.symbol)

                signal = signal_gen.evaluate(
                    market, tick, yes_book, features=feats, carry_annual=carry,
                )

                # Mean-reversion overlay (Portnaya 4h half-life). Fed every tick
                # so the EWMA stays fresh and open fades can be exited; gated to
                # markets with tte >> half-life (never 5/15-min). ENTER only when
                # the model leg is quiet; EXIT always flattens.
                if meanrev is not None:
                    tte_sec = market.expiry_ts - now_wall
                    pf_mr, _ = signal_gen.fair_prob(market, tick, carry)
                    mid_px = 0.5 * (yes_book.best_bid + yes_book.best_ask)
                    size_hint = round(max_notional / mid_px, 2) if mid_px > 0 else 0.0
                    action = meanrev.update(
                        market.yes_token_id, tte_sec, pf_mr,
                        yes_book.best_bid, yes_book.best_ask, size_hint=size_hint,
                    )
                    if action is not None and action.price > 0 and (
                        signal is None or action.kind == "EXIT"
                    ):
                        if cfg.kelly_enabled and action.kind == "ENTER":
                            msz = kelly_size(action.p_fair, action.price, action.side,
                                             max_notional, cfg.kelly_fraction)
                        else:
                            msz = round(max_notional / action.price, 2)
                        if msz and msz >= 1.0:
                            signal = Signal(
                                market=market, token_id=market.yes_token_id,
                                side=Side(action.side), price=action.price, size=round(msz, 2),
                                p_star=action.p_fair, edge=0.0, sigma=0.0,
                                source=("meanrev" if action.kind == "ENTER" else "meanrev-exit"),
                            )

                if signal is None:
                    # Track the largest raw mispricing visible — diagnostic
                    # for the heartbeat. No extra p* call: re-derive from
                    # the inputs we already have. Keep this cheap by
                    # gating on the heartbeat being due.
                    if (now_mono - last_heartbeat) > (HEARTBEAT_SECS - 5):
                        if tte > min_tte_secs and market.strike > 0:
                            sigma_eff = max(tick.sigma_annual, sigma_floor)
                            p = implied_prob(
                                tick.mid, market.strike, tte, sigma_eff,
                                drift_annual=tick.drift_annual,
                            )
                            raw = abs(p - proxy_px)
                            if raw > top_edge_seen[0]:
                                top_edge_seen = (raw, market.question)
                    continue

                cum_signals += 1
                # Check blacklist again with actual fill price (may differ
                # from book mid by enough to land in a different bucket).
                cat_key = CategoryTracker.category_key(
                    market.question, market.symbol, tte, signal.price,
                )
                if cat_key in blacklist:
                    continue

                allowed, reason = risk.check(signal, binance_ts=tick.ts, poly_ts=yes_book.ts)
                if not allowed:
                    log.debug("Risk blocked: %s", reason)
                    continue

                fill = await executor.execute(signal)
                if fill:
                    cum_fills += 1
                    risk.record_fill(fill.price * fill.size)
                else:
                    risk.record_error()

            # ---- Static arbitrage scan (model-free) ----
            # Strike-monotonicity violations on the threshold ladder are
            # risk-free static arbs. They're rare but persistent because
            # different makers run the various strikes independently and
            # don't always cross-link instantly. Reference:
            # arXiv:2508.03474 (Saguillo et al, AFT 2025) — combinatorial
            # arbitrage extracted ~$40M from Polymarket in 2024–25.
            if cfg.arb_enabled:
                combos = scan_combos(
                    markets, poly_ws,
                    min_credit=cfg.arb_min_credit,
                    max_notional_usd=max_notional,
                    min_tte_secs=min_tte_secs,
                    max_tte_secs=max_tte_secs,
                    rebalance=cfg.rebalance_arb_enabled,
                    bucket=cfg.bucket_arb_enabled,
                )
                for combo in combos:
                    cum_arbs += 1
                    # One Signal per leg, all tagged with the shared arb_id so
                    # the PnL accountant can match them. Strike arbs have a
                    # BUY+SELL pair; rebalance/bucket arbs are BUY+BUY.
                    sigs = [
                        Signal(
                            market=leg.market, token_id=leg.token_id, side=Side(leg.side),
                            price=leg.price, size=leg.size, p_star=0.0,
                            edge=combo.net_credit, sigma=0.0,
                            arb_id=combo.arb_id, source="arb",
                        )
                        for leg in combo.legs
                    ]
                    if not all(
                        risk.check(s, binance_ts=now_mono, poly_ts=now_mono)[0] for s in sigs
                    ):
                        log.debug("Combo %s blocked by risk", combo.arb_id)
                        continue
                    # Atomic: a failed leg flattens the filled ones (orphan-safe).
                    combo_fills = await executor.execute_atomic(sigs)
                    if combo_fills is None:
                        risk.record_error()
                        continue
                    for f in combo_fills:
                        risk.record_fill(f.price * f.size)
                    cum_arb_fills += 1
                    log.info(
                        "ARB EXECUTED %s [%s] credit=%.4f notional=%.2f legs=%d",
                        combo.arb_id, combo.kind, combo.net_credit,
                        combo.notional, len(combo.legs),
                    )

            # ---- Heartbeat (cumulative over window) ----
            if (now_mono - last_heartbeat) > HEARTBEAT_SECS:
                sigma_str_parts = []
                for sym, bc in binance_clients.items():
                    snap = bc.snapshot()
                    s = snap.sigma_annual if snap else 0.0
                    d = snap.drift_annual if snap else 0.0
                    iv_snap = iv_oracle.snapshot(sym)
                    iv = iv_snap.sigma_annual if iv_snap else 0.0
                    sigma_str_parts.append(f"{sym}[σ={s:.3f} iv={iv:.3f} μ={d:+.3f}]")
                log.info(
                    "HEARTBEAT eval=%d signals=%d fills=%d arbs=%d arb-fills=%d "
                    "no-spot=%d no-book=%d warmup=%d markets=%d %s top-raw-gap=%.3f (%s)",
                    cum_eval, cum_signals, cum_fills,
                    cum_arbs, cum_arb_fills,
                    cum_no_spot, cum_no_book, cum_warmup, len(markets),
                    " ".join(sigma_str_parts),
                    top_edge_seen[0], top_edge_seen[1][:50],
                )
                cum_eval = cum_no_spot = cum_no_book = cum_warmup = cum_signals = cum_fills = 0
                cum_arbs = cum_arb_fills = 0
                top_edge_seen = (-1.0, "")
                last_heartbeat = now_mono

    finally:
        # Graceful shutdown — cancel background tasks, close DB, print PnL.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await session.close()
        await executor.close()

        # Reconcile category tracker against newly-resolved markets so the
        # next run inherits what this run learned.
        try:
            from .category import _rebuild_from_fills
            rebuilt = await _rebuild_from_fills()
            rebuilt.save()
            log.info("Category tracker reconciled and saved.")
        except Exception as exc:
            log.warning("Category tracker reconcile failed: %s", exc)
            try:
                category_tracker.save()
            except Exception:
                pass


def cli() -> None:
    parser = argparse.ArgumentParser(description="Polymarket-vs-Binance arb bot")
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Send real orders (default: paper-trade only)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N minutes and print PnL summary (default: run forever)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    paper = not args.live
    if not paper:
        print(
            "\n⚠️  LIVE MODE ENABLED — real orders will be sent to Polymarket.\n"
            "   Ensure PAPER_TRADE=false in .env and you have reviewed the runbook.\n"
            "   Press Ctrl-C within 5 seconds to abort...\n"
        )
        import time as _t
        _t.sleep(5)

    duration_secs = args.duration * 60.0
    try:
        asyncio.run(main(paper=paper, log_level=args.log_level, duration_secs=duration_secs))
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print("\n--- Final PnL Summary ---")
    try:
        from .pnl import report
        asyncio.run(report())
    except Exception as exc:
        print(f"PnL summary unavailable: {exc}")


if __name__ == "__main__":
    cli()
