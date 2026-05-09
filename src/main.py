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

from .binance_ws import BinanceWS
from .category import CategoryTracker
from .deribit_iv import DeribitIV
from .execute import Executor
from .poly_universe import fetch_active_markets
from .poly_ws import PolyWS
from .pricing import implied_prob, SIGMA_MIN
from .risk import RiskManager
from .signal import SignalGenerator

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
) -> None:
    """Refresh the market universe in the background without blocking the
    eval loop. Writes the new market list and per-symbol partition into
    `state` atomically."""
    while True:
        try:
            markets = await fetch_active_markets(session)
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

    mode = "PAPER" if paper else "LIVE"
    log.info("=== Polymarket-vs-Binance arb bot starting [%s] ===", mode)
    log.info(
        "Binance symbols: %s | safety_eps=%.4f cooldown=%.1fs notional<=%.0f "
        "| price=[%.2f,%.2f] σ_floor=%.2f tte=[%.0f,%.0f]s skip_updown=%s",
        symbols, safety_eps, cooldown, max_notional,
        price_min, price_max, sigma_floor, min_tte_secs, max_tte_secs, skip_updown,
    )
    if duration_secs > 0:
        log.info("Will stop after %.0f seconds and print PnL summary.", duration_secs)

    binance_clients = {sym: BinanceWS(sym, stale_threshold_secs=binance_stale) for sym in symbols}
    poly_ws = PolyWS(token_ids=[], stale_threshold_secs=poly_stale)
    iv_oracle = DeribitIV(symbols=symbols)
    signal_gen = SignalGenerator(
        max_notional_per_trade=max_notional,
        safety_eps=safety_eps,
        cooldown_secs=cooldown,
        price_min=price_min,
        price_max=price_max,
        sigma_floor=sigma_floor,
        skip_updown=skip_updown,
        min_tte_secs=min_tte_secs,
        max_tte_secs=max_tte_secs,
        book_max_age_secs=book_max_age_secs,
        iv_oracle=iv_oracle,
    )
    risk = RiskManager(
        max_notional_per_trade=max_notional,
        max_notional_per_minute=max_per_min,
        daily_drawdown_stop=drawdown_stop,
        binance_stale_secs=binance_stale,
        poly_stale_secs=poly_stale,
    )
    executor = Executor(paper=paper)
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
    tasks.append(asyncio.create_task(
        _universe_refresher(session, poly_ws, binance_clients, universe_state),
        name="universe",
    ))

    # Heartbeat-window cumulative counters.
    cum_eval = cum_no_spot = cum_no_book = cum_warmup = cum_signals = cum_fills = 0
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

                # Up/Down markets — anchor strike to live spot first time.
                if market.strike <= 0:
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

                signal = signal_gen.evaluate(market, tick, yes_book)
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
                    "HEARTBEAT eval=%d signals=%d fills=%d no-spot=%d no-book=%d warmup=%d "
                    "markets=%d %s top-raw-gap=%.3f (%s)",
                    cum_eval, cum_signals, cum_fills,
                    cum_no_spot, cum_no_book, cum_warmup, len(markets),
                    " ".join(sigma_str_parts),
                    top_edge_seen[0], top_edge_seen[1][:50],
                )
                cum_eval = cum_no_spot = cum_no_book = cum_warmup = cum_signals = cum_fills = 0
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
