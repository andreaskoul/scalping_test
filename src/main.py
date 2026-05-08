"""
Orchestrator — wires Binance WS, Polymarket WS, signal generator,
risk manager, and executor into a single async event loop.

Usage:
  python -m src.main             # paper-trade (safe default)
  python -m src.main --live      # REAL orders — requires POLY_PRIVATE_KEY
  python -m src.main --help

Paper-trade mode runs the full stack against live feeds but never sends
a real order. Logs every signal to fills.db for post-mortem analysis.
"""

import argparse
import asyncio
import logging
import os
import time
import aiohttp

from dotenv import load_dotenv

load_dotenv()

from .binance_ws import BinanceWS
from .execute import Executor
from .poly_universe import fetch_active_markets
from .poly_ws import PolyWS
from .risk import RiskManager
from .signal import SignalGenerator

log = logging.getLogger(__name__)

UNIVERSE_REFRESH_SECS = 30.0    # re-poll Gamma API for new markets
EVAL_INTERVAL_SECS = 0.05       # main loop tick (20 Hz)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


async def main(paper: bool, log_level: str = "INFO") -> None:
    _setup_logging(log_level)

    private_key = os.getenv("POLY_PRIVATE_KEY")
    if not paper and not private_key:
        raise SystemExit("POLY_PRIVATE_KEY must be set in .env for live mode")

    symbols = [s.strip().lower() for s in os.getenv("BINANCE_SYMBOLS", "btcusdt,ethusdt").split(",") if s.strip()]
    max_notional = float(os.getenv("MAX_NOTIONAL_PER_TRADE", "25"))
    max_per_min = float(os.getenv("MAX_NOTIONAL_PER_MINUTE", "200"))
    drawdown_stop = float(os.getenv("DAILY_DRAWDOWN_STOP", "500"))
    safety_eps = float(os.getenv("EDGE_SAFETY_EPS", "0.003"))
    cooldown = float(os.getenv("COOLDOWN_SECS", "1.0"))
    binance_stale = float(os.getenv("BINANCE_STALE_SECS", "2.0"))
    poly_stale = float(os.getenv("POLY_STALE_SECS", "5.0"))

    mode = "PAPER" if paper else "LIVE"
    log.info("=== Polymarket-vs-Binance arb bot starting [%s] ===", mode)
    log.info("Binance symbols: %s | safety_eps=%.4f cooldown=%.1fs notional<=%.0f",
             symbols, safety_eps, cooldown, max_notional)

    # Components
    binance_clients = {sym: BinanceWS(sym, stale_threshold_secs=binance_stale) for sym in symbols}
    poly_ws = PolyWS(token_ids=[], stale_threshold_secs=poly_stale)
    signal_gen = SignalGenerator(
        max_notional_per_trade=max_notional,
        safety_eps=safety_eps,
        cooldown_secs=cooldown,
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

    # Background tasks
    tasks = [
        asyncio.create_task(c.run(), name=f"binance-{sym}")
        for sym, c in binance_clients.items()
    ]
    tasks.append(asyncio.create_task(poly_ws.run(), name="poly-ws"))

    # Universe + eval loop
    markets = []
    last_refresh = 0.0
    last_heartbeat = 0.0
    HEARTBEAT_SECS = 30.0

    async with aiohttp.ClientSession() as session:
        while True:
            if risk.is_halted():
                log.warning("Bot halted. Sleeping...")
                await asyncio.sleep(5)
                continue

            now = time.time()

            # Refresh market universe periodically
            if now - last_refresh > UNIVERSE_REFRESH_SECS:
                markets = await fetch_active_markets(session)
                token_ids = []
                for m in markets:
                    token_ids += [m.yes_token_id, m.no_token_id]
                poly_ws.update_tokens(token_ids)
                last_refresh = now
                log.info("Universe refreshed: %d markets", len(markets))

            # Pipeline-state stats accumulated this iteration.
            n_evaluated = n_skipped_no_spot = n_skipped_no_book = n_skipped_warmup = 0
            top_edge_seen = (-1.0, "")  # (raw |p* - mid|, market_question)

            for market in markets:
                binance_client = binance_clients.get(market.symbol)
                if binance_client is None:
                    continue
                binance_tick = binance_client.snapshot()
                if binance_tick is None:
                    n_skipped_no_spot += 1
                    continue

                # Up/Down markets: use live Binance mid as the reference strike
                if market.strike <= 0:
                    market.strike = binance_tick.mid

                yes_book = poly_ws.snapshot(market.yes_token_id)
                if yes_book is None:
                    n_skipped_no_book += 1
                    continue

                from .pricing import implied_prob, SIGMA_MIN
                if binance_tick.sigma_annual <= SIGMA_MIN:
                    n_skipped_warmup += 1
                    continue

                # Track the largest raw mispricing visible right now —
                # gives the user signal even if no trade fires.
                tte = market.expiry_ts - now
                if tte > 120 and market.strike > 0:
                    p_star = implied_prob(binance_tick.mid, market.strike, tte, binance_tick.sigma_annual)
                    poly_mid = (yes_book.best_bid + yes_book.best_ask) / 2
                    raw = abs(p_star - poly_mid)
                    if raw > top_edge_seen[0]:
                        top_edge_seen = (raw, market.question)

                n_evaluated += 1
                signal = signal_gen.evaluate(market, binance_tick, yes_book)
                if signal is None:
                    continue

                allowed, reason = risk.check(
                    signal,
                    binance_ts=binance_tick.ts,
                    poly_ts=yes_book.ts,
                )
                if not allowed:
                    log.debug("Risk blocked: %s", reason)
                    continue

                fill = await executor.execute(signal)
                if fill:
                    risk.record_fill(fill.price * fill.size)
                else:
                    risk.record_error()

            # Heartbeat: state of the pipeline so the user sees progress
            # even when no signals fire.
            if now - last_heartbeat > HEARTBEAT_SECS:
                sigmas = {}
                for sym, bc in binance_clients.items():
                    snap = bc.snapshot()
                    sigmas[sym] = snap.sigma_annual if snap else 0.0
                sigma_str = " ".join(f"{s}σ={v:.3f}" for s, v in sigmas.items())
                log.info(
                    "HEARTBEAT eval=%d no-spot=%d no-book=%d warmup=%d markets=%d %s top-raw-gap=%.3f (%s)",
                    n_evaluated, n_skipped_no_spot, n_skipped_no_book, n_skipped_warmup,
                    len(markets), sigma_str,
                    top_edge_seen[0], top_edge_seen[1][:50],
                )
                last_heartbeat = now

            await asyncio.sleep(EVAL_INTERVAL_SECS)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Polymarket-vs-Binance arb bot")
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Send real orders (default: paper-trade only)",
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

    try:
        asyncio.run(main(paper=paper, log_level=args.log_level))
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    cli()
