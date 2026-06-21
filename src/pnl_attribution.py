"""
PnL attribution — confirm the favourite-longshot wedge on our own fills.

Priority #1 from the research pass: before changing any model code, verify
Portnaya's claim against the money we actually put down.  Groups realised PnL
by **side × entry-price bucket** and by **side × model-probability (p*) bucket**.

The prediction to confirm: BUY-YES at low price / low p* (the OTM longshot)
is where losses concentrate, because the market is efficiently rich there and
our p* sits above the true risk-neutral value.  If so, the asymmetric tilt in
signal.py is justified empirically, not just from a 2023 paper.

Usage:
  python -m src.pnl_attribution            # reads fills.db, prints tables
  python -m src.pnl_attribution --db x.db
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections import defaultdict

import aiohttp

from .pnl import _load_fills, _fold_positions, _fetch_resolution


def _price_bucket(px: float) -> str:
    if px < 0.20:
        return "0.00-0.20"
    if px < 0.40:
        return "0.20-0.40"
    if px <= 0.60:
        return "0.40-0.60"
    if px <= 0.80:
        return "0.60-0.80"
    return "0.80-1.00"


async def attribute(db_path: str = "fills.db") -> None:
    try:
        fills = _load_fills(db_path)
    except sqlite3.OperationalError:
        print(f"No fills table in {db_path} — run the paper bot first.")
        return
    if not fills:
        print("No fills yet — run the paper bot first.")
        return
    positions = _fold_positions(fills)

    async with aiohttp.ClientSession() as s:
        markets = list({p.market_id for p in positions.values()})
        resols = await asyncio.gather(*[_fetch_resolution(s, m) for m in markets])
    resol_by_market = dict(zip(markets, resols))

    # bucket -> [realized_pnl, n, wins]
    by_price: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0, 0])
    by_pstar: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0, 0])

    # We need entry side + p* per token; fold loses p*, so re-walk fills for p*.
    pstar_by_token: dict[str, float] = {}
    for f in fills:
        pstar_by_token.setdefault(f["token_id"], f["p_star"])

    resolved = 0
    for tid, p in positions.items():
        if abs(p.shares) < 1e-6:
            continue
        ref = resol_by_market.get(p.market_id)
        if ref is None:
            continue
        resolved += 1
        pnl = (ref - p.avg_price) * p.shares - p.fee_paid
        side = "BUY" if p.shares > 0 else "SELL"
        win = 1 if pnl > 0 else 0

        b = by_price[(side, _price_bucket(p.avg_price))]
        b[0] += pnl; b[1] += 1; b[2] += win

        ps = pstar_by_token.get(tid, p.avg_price)
        b2 = by_pstar[(side, _price_bucket(ps))]
        b2[0] += pnl; b2[1] += 1; b2[2] += win

    print(f"\n=== PnL attribution ({resolved} resolved positions) ===")
    if resolved == 0:
        print("No resolved positions yet — attribution needs settled markets.")
        return

    def _dump(title: str, table: dict) -> None:
        print(f"\n{title}")
        print(f"  {'side':<5} {'bucket':<12} {'n':>4} {'win%':>6} {'PnL':>11} {'PnL/trade':>11}")
        print("  " + "-" * 56)
        for (side, bucket), (pnl, n, wins) in sorted(table.items()):
            wr = 100.0 * wins / n if n else 0.0
            print(f"  {side:<5} {bucket:<12} {n:>4} {wr:>5.0f}% {pnl:>+11.3f} {pnl/max(1,n):>+11.4f}")

    _dump("By entry price:", by_price)
    _dump("By model p* (moneyness proxy):", by_pstar)
    print(
        "\nWedge check: if BUY rows in the low buckets (0.00-0.40) are the worst, "
        "the favourite-longshot tilt is confirmed on your own fills."
    )


def cli() -> None:
    ap = argparse.ArgumentParser(description="Confirm the favourite-longshot wedge on fills.db")
    ap.add_argument("--db", default="fills.db")
    args = ap.parse_args()
    asyncio.run(attribute(args.db))


if __name__ == "__main__":
    cli()
