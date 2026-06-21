"""
Offline paper-trade simulation through the REAL signal/exec/PnL stack.

Why this exists: the sandbox blocks all market-data hosts (Binance, Polymarket,
Kalshi, Deribit all return HTTP 403), so a live paper run pulls zero ticks.
This harness instead generates a synthetic BTC world whose *generating process*
embeds the documented favourite-longshot wedge (Portnaya 2026), then runs the
real `SignalGenerator.evaluate` + `Executor` (paper) and resolves every market
against the realized GBM path to compute true PnL.

It is a synthetic illustration, not live alpha: it shows whether the assembled
edges harvest the wedge when the wedge is present, run head-to-head against the
legacy taker-only config on *identical* price paths and books.

LIMITATIONS (read before trusting any number this prints):
  - This is a PLUMBING test, not a backtest. The bot is effectively handed
    near-oracle fair value (book mid = true fair + wedge + tiny noise; the
    bot's p* ≈ true fair), so win rates (~90%) and ROI are unrealistically
    high. Real markets do not reveal the true probability.
  - The A/B magnitudes are NOT a verdict on max-EV vs legacy: into a rigged
    free-money market, the taker baseline trades more (maker fills are gated
    to 50%) at full size (no Kelly shrink) and therefore books more nominal
    PnL. That is an artifact of betting maximally into an easy market.
  - The ONE robust, real signal here is the FEE LINE: the maker config earns
    net rebates (negative fees) where the taker baseline pays fees — the
    documented economics flip, independent of the synthetic edge.
  - Real PnL requires live/historical data (blocked in this sandbox by a
    network policy returning HTTP 403 for Binance/Polymarket/Kalshi/Deribit);
    run from an allowed region via `python -m src.main` or `src.backtest`.
"""

import asyncio
import math
import random
import sqlite3
import time

import src.execute as execmod
from src.binance_ws import BinanceTick
from src.execute import Executor
from src.poly_universe import PolyMarket
from src.poly_ws import BookSnapshot
from src.pricing import implied_prob, wedge_estimate
from src.signal import SignalGenerator

SECS_YEAR = 365.25 * 86400.0
TTE0 = 300          # 5-minute rounds
DT = 1              # 1s GBM steps
SIGMA = 0.50        # true annualised vol of the generating process
S0 = 95000.0
Z_GRID = (-2.5, -1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5, 2.5)  # strike spread
MAX_NOTIONAL = 25.0
MAKER_FILL_PROB = 0.5   # resting maker orders only sometimes get hit


def gbm_path(rng):
    s = S0
    out = [s]
    for _ in range(TTE0 // DT):
        z = rng.gauss(0, 1)
        s *= math.exp(-0.5 * SIGMA**2 * (DT / SECS_YEAR) + SIGMA * math.sqrt(DT / SECS_YEAR) * z)
        out.append(s)
    return out


def build_scenario(seed, n_rounds):
    """Materialise every path, strike, and book-noise draw ONCE so both
    configs see identical markets."""
    rng = random.Random(seed)
    rounds = []
    move_sd = SIGMA * math.sqrt(TTE0 / SECS_YEAR)
    for r in range(n_rounds):
        path = gbm_path(rng)
        s_t_final = path[-1]
        markets = []
        for z in Z_GRID:
            strike = round(path[0] * math.exp(z * move_sd), 2)
            cid = f"r{r}-z{z}"
            evals = []
            for t in range(0, TTE0 - 59, 30):
                evals.append((
                    path[t], TTE0 - t,
                    rng.gauss(0, 0.01),            # book mid noise
                    SIGMA * (1 + rng.gauss(0, 0.05)),  # bot's noisy σ estimate
                ))
            markets.append((cid, strike, s_t_final, evals))
        rounds.append(markets)
    return rounds


async def run_config(name, scenario, db_path, **cfg):
    execmod.DB_PATH = db_path
    gen = SignalGenerator(
        max_notional_per_trade=MAX_NOTIONAL, cooldown_secs=0.0,
        min_tte_secs=60.0, max_tte_secs=86400.0, sigma_floor=0.05,
        iv_oracle=None, poly_ws=None, **cfg,
    )
    ex = Executor(paper=True)
    await ex.setup()
    fill_rng = random.Random(98765)
    res_map = {}
    for markets in scenario:
        for cid, strike, s_t_final, evals in markets:
            res_map[cid] = (strike, s_t_final)
            for s_t, tte, noise, sigma_obs in evals:
                now, mono = time.time(), time.monotonic()
                m = PolyMarket(
                    condition_id=cid, question=f"BTC above {strike:.0f}?",
                    yes_token_id=cid + "-Y", no_token_id=cid + "-N",
                    yes_price=0.5, no_price=0.5, strike=strike,
                    expiry_ts=now + tte, tick_size=0.01, symbol="btcusdt",
                    is_threshold=True,
                )
                tick = BinanceTick("btcusdt", s_t - 1, s_t + 1, s_t, sigma_obs, mono, 0.0)
                # Honest fair value (generating σ) + the documented wedge + noise.
                p_fair = implied_prob(s_t, strike, tte, SIGMA)
                mid = min(0.99, max(0.01, p_fair + wedge_estimate(p_fair, tte / 3600.0) + noise))
                bid = round(max(0.01, mid - 0.01), 2)
                ask = round(min(0.99, mid + 0.01), 2)
                book = BookSnapshot(m.yes_token_id, bid, ask, 500.0, 500.0, mono)
                sig = gen.evaluate(m, tick, book)
                if sig is None:
                    continue
                if sig.is_maker and fill_rng.random() > MAKER_FILL_PROB:
                    continue  # resting order wasn't hit
                await ex.execute(sig)
    await ex.close()
    return res_map


def pnl_report(name, db_path, res_map):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT market_id, side, price, size, fee FROM fills").fetchall()
    conn.close()
    n = wins = maker = 0
    total = fees = gross = 0.0
    by_side = {"BUY": [0.0, 0], "SELL": [0.0, 0]}
    for mid, side, price, size, fee in rows:
        strike, s_t_final = res_map[mid]
        res = 1.0 if s_t_final > strike else 0.0
        directional = (res - price) * size if side == "BUY" else (price - res) * size
        pnl = directional - fee
        total += pnl
        fees += fee
        gross += price * size
        n += 1
        wins += 1 if pnl > 0 else 0
        maker += 1 if fee < 0 else 0
        by_side[side][0] += pnl
        by_side[side][1] += 1
    roi = (total / gross * 100.0) if gross else 0.0
    wr = (wins / n * 100.0) if n else 0.0
    print(f"\n[{name}]")
    print(f"  fills={n}  maker={maker}  taker={n-maker}  win%={wr:.0f}")
    print(f"  gross_notional=${gross:,.0f}  net_fees=${fees:+.2f}  realized_PnL=${total:+.2f}  ROI={roi:+.2f}%")
    print(f"  by side: BUY ${by_side['BUY'][0]:+.2f} ({by_side['BUY'][1]}) | SELL ${by_side['SELL'][0]:+.2f} ({by_side['SELL'][1]})")
    return total, n


async def main():
    import os
    scenario = build_scenario(seed=7, n_rounds=120)
    n_markets = sum(len(r) for r in scenario)
    print(f"Synthetic scenario: {len(scenario)} rounds x {len(Z_GRID)} strikes = "
          f"{n_markets} markets, wedge baked into book mids.")

    for f in ("/tmp/sim_maxev.db", "/tmp/sim_legacy.db"):
        if os.path.exists(f):
            os.remove(f)

    # A) Max-EV: longshot tilt + maker + Kelly + sell-tail.
    res = await run_config(
        "MAX-EV", scenario, "/tmp/sim_maxev.db",
        safety_eps=0.005, price_min=0.10, price_max=0.90, sell_price_min=0.03,
        longshot_tilt_mult=1.0, maker_enabled=True, maker_join_ticks=1,
        kelly_enabled=True, kelly_fraction=0.30,
    )
    a_total, a_n = pnl_report("MAX-EV (tilt+maker+kelly+sell-tail)", "/tmp/sim_maxev.db", res)

    # B) Legacy: taker-only, symmetric, no tilt/tail.
    res = await run_config(
        "LEGACY", scenario, "/tmp/sim_legacy.db",
        safety_eps=0.005, price_min=0.10, price_max=0.90,
    )
    b_total, b_n = pnl_report("LEGACY (taker-only baseline)", "/tmp/sim_legacy.db", res)

    print("\n=== DELTA (Max-EV - Legacy) ===")
    print(f"  PnL: ${a_total - b_total:+.2f}   (max-ev ${a_total:+.2f} vs legacy ${b_total:+.2f})")
    for f in ("/tmp/sim_maxev.db", "/tmp/sim_legacy.db"):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    asyncio.run(main())
