"""
Paper-trade PnL monitor.

Reads fills.db and computes:
  - per-position notional, unrealized mark-to-market PnL (using current
    Polymarket book mid),
  - realized PnL for any market that has resolved (price → 1.0 / 0.0),
  - aggregate fill stats: n_fills, gross notional, fee paid, win rate.

Usage:
  python -m src.pnl              # one-shot snapshot
  python -m src.pnl --watch 30   # refresh every 30s
"""

import argparse
import asyncio
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass

import aiohttp

DB_PATH = "fills.db"
GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_BOOK = "https://clob.polymarket.com/book"


@dataclass
class Position:
    token_id: str
    market_id: str
    side: str          # net "BUY" (long YES) or "SELL" (short YES)
    shares: float
    avg_price: float   # weighted avg entry price
    fee_paid: float
    n_fills: int


def _load_fills(db_path: str = DB_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ts, market_id, token_id, side, price, size, fee, p_star, edge, paper, order_id "
        "FROM fills ORDER BY ts ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _fold_positions(fills: list[dict]) -> dict[str, Position]:
    """Net out fills per token_id into a position with avg cost."""
    pos: dict[str, Position] = {}
    for f in fills:
        tid = f["token_id"]
        delta = f["size"] if f["side"] == "BUY" else -f["size"]
        if tid not in pos:
            pos[tid] = Position(
                token_id=tid,
                market_id=f["market_id"],
                side="BUY" if delta > 0 else "SELL",
                shares=delta,
                avg_price=f["price"],
                fee_paid=f["fee"],
                n_fills=1,
            )
            continue
        p = pos[tid]
        new_shares = p.shares + delta
        # Recompute avg only on adds in the same direction; reduce keeps avg.
        if (p.shares >= 0 and delta > 0) or (p.shares <= 0 and delta < 0):
            denom = abs(p.shares) + abs(delta)
            p.avg_price = (
                p.avg_price * abs(p.shares) + f["price"] * abs(delta)
            ) / denom if denom else f["price"]
        p.shares = new_shares
        p.side = "BUY" if new_shares >= 0 else "SELL"
        p.fee_paid += f["fee"]
        p.n_fills += 1
    return pos


async def _fetch_book_mid(session: aiohttp.ClientSession, token_id: str) -> float | None:
    try:
        async with session.get(
            CLOB_BOOK, params={"token_id": token_id},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            if r.status != 200:
                return None
            d = await r.json()
            bids, asks = d.get("bids", []), d.get("asks", [])
            if not bids or not asks:
                return None
            bb = max(float(b["price"]) for b in bids)
            ba = min(float(a["price"]) for a in asks)
            return (bb + ba) / 2
    except Exception:
        return None


async def _fetch_resolution(session: aiohttp.ClientSession, market_id: str) -> float | None:
    """Return 1.0 if YES won, 0.0 if NO won, None if unresolved."""
    try:
        async with session.get(
            f"{GAMMA_API}/{market_id}",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            if r.status != 200:
                return None
            m = await r.json()
            if not m.get("closed"):
                return None
            # outcomePrices ends up as ["1","0"] or ["0","1"] post-resolution
            prices = m.get("outcomePrices") or "[]"
            if isinstance(prices, str):
                import json as _j
                try:
                    prices = _j.loads(prices)
                except Exception:
                    return None
            if not prices or len(prices) < 1:
                return None
            return float(prices[0])  # YES price at resolution
    except Exception:
        return None


async def report(db_path: str = DB_PATH) -> None:
    fills = _load_fills(db_path)
    if not fills:
        print("No fills yet.")
        return

    positions = _fold_positions(fills)

    gross_notional = sum(f["price"] * f["size"] for f in fills)
    total_fees = sum(f["fee"] for f in fills)
    avg_edge = sum(f["edge"] for f in fills) / len(fills)
    paper_count = sum(f["paper"] for f in fills)

    print(f"\n=== Paper-Trade PnL Report ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
    print(f"Fills:           {len(fills)} ({paper_count} paper, {len(fills) - paper_count} live)")
    print(f"Gross notional:  ${gross_notional:,.2f}")
    print(f"Total fees:      ${total_fees:.4f}")
    print(f"Mean signal edge: {avg_edge*100:+.2f}%")
    print(f"Open positions:  {sum(1 for p in positions.values() if abs(p.shares) > 1e-6)}")

    # Mark-to-market + resolution lookup
    mtm_pnl = 0.0
    realized_pnl = 0.0
    win_count = lose_count = 0
    rows: list[tuple[str, Position, float | None, float | None]] = []

    async with aiohttp.ClientSession() as s:
        # Fetch mid + resolution for each unique market_id
        unique_markets = list({p.market_id for p in positions.values()})
        unique_tokens = list(positions.keys())
        mids = await asyncio.gather(*[_fetch_book_mid(s, tid) for tid in unique_tokens])
        resols = await asyncio.gather(*[_fetch_resolution(s, mid) for mid in unique_markets])
        mid_by_token = dict(zip(unique_tokens, mids))
        resol_by_market = dict(zip(unique_markets, resols))

    print(f"\n{'Market':<50} {'shares':>9} {'avg':>7} {'mark':>7} {'resol':>6} {'PnL':>9}")
    print("-" * 95)
    for tid, p in positions.items():
        if abs(p.shares) < 1e-6:
            continue
        resol = resol_by_market.get(p.market_id)
        mark = mid_by_token.get(tid)
        # YES position PnL = shares * (mark - avg). SELL = shorted YES.
        ref = resol if resol is not None else mark
        pnl = (ref - p.avg_price) * p.shares - p.fee_paid if ref is not None else 0.0
        if resol is not None:
            realized_pnl += pnl
            if pnl > 0:
                win_count += 1
            elif pnl < 0:
                lose_count += 1
        else:
            mtm_pnl += pnl
        line = (
            f"{p.market_id[:48]:<50} "
            f"{p.shares:>+9.2f} {p.avg_price:>7.4f} "
            f"{mark if mark is not None else 0:>7.4f} "
            f"{resol if resol is not None else float('nan'):>6.2f} "
            f"{pnl:>+9.4f}"
        )
        print(line)

    print("-" * 95)
    print(f"Realized PnL:    ${realized_pnl:+.4f}  ({win_count}W / {lose_count}L)")
    print(f"Unrealized MtM:  ${mtm_pnl:+.4f}")
    print(f"Total PnL:       ${realized_pnl + mtm_pnl:+.4f}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Paper-trade PnL monitor")
    parser.add_argument("--db", default=DB_PATH, help="path to fills.db")
    parser.add_argument("--watch", type=float, default=0.0, help="refresh every N seconds")
    args = parser.parse_args()

    async def loop():
        while True:
            await report(args.db)
            if args.watch <= 0:
                break
            await asyncio.sleep(args.watch)

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
