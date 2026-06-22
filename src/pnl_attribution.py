"""
PnL attribution + calibration audit — confirm the edges on our own fills.

Priority #1 from the research pass, now using the F1-enriched schema:
  - realised PnL by side × entry-price bucket
  - realised PnL by side × time-to-expiry bucket   (needs tte_at_fill)
  - maker vs taker, and by signal source (model / meanrev / arb)
  - per-bucket significance (mean ± s.e., t-stat) so a "winning" bucket with
    n=3 isn't mistaken for an edge
  - a calibration audit: is p* well-calibrated against realised outcomes?
    (Brier score + a reliability table). If p* is biased, every downstream
    edge is mismeasured — this is the EMOS/calibration check.

The prediction to confirm (Portnaya): BUY-YES at low price / low p* / long TTE
is where losses concentrate, because the market is efficiently rich there.

Usage:
  python -m src.pnl_attribution            # reads fills.db
  python -m src.pnl_attribution --db x.db
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sqlite3
from collections import defaultdict

import aiohttp

from .pnl import _fetch_resolution
from .storage import default_db_path


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


def _tte_bucket(tte: float) -> str:
    if tte <= 0:
        return "unknown"
    if tte < 600:
        return "<10m"
    if tte < 1800:
        return "10-30m"
    if tte < 3600:
        return "30-60m"
    return ">=1h"


def _load_fills(db_path: str) -> list[dict]:
    """Read fills with the F1 columns, defaulting any a pre-F1 db lacks."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute("PRAGMA table_info(fills)")
    cols = {row[1] for row in cur.fetchall()}
    base = ["ts", "market_id", "token_id", "side", "price", "size", "fee", "p_star"]
    opt = {"tte_at_fill": 0.0, "is_maker": 0, "source": "model"}
    select = base + [c for c in opt if c in cols]
    rows = conn.execute(f"SELECT {', '.join(select)} FROM fills ORDER BY ts ASC").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = {k: r[k] for k in select}
        for c, default in opt.items():
            d.setdefault(c, default)
        out.append(d)
    return out


def _stats(pnls: list[float]) -> tuple[float, float, float]:
    """(total, mean, t-stat) — t = mean / s.e.; |t|>2 ≈ significant."""
    n = len(pnls)
    if n == 0:
        return 0.0, 0.0, 0.0
    total = sum(pnls)
    mean = total / n
    if n < 2:
        return total, mean, 0.0
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    se = math.sqrt(var / n)
    return total, mean, (mean / se if se > 0 else 0.0)


async def attribute(db_path: str = default_db_path("fills.db")) -> None:
    try:
        fills = _load_fills(db_path)
    except sqlite3.OperationalError:
        print(f"No fills table in {db_path} — run the paper bot first.")
        return
    if not fills:
        print("No fills yet — run the paper bot first.")
        return

    async with aiohttp.ClientSession() as s:
        markets = list({f["market_id"] for f in fills})
        resols = await asyncio.gather(*[_fetch_resolution(s, m) for m in markets])
    resol = dict(zip(markets, resols))

    # per-fill realised pnl (each fill scored independently vs resolution)
    buckets_price: dict[tuple, list] = defaultdict(list)
    buckets_tte: dict[tuple, list] = defaultdict(list)
    buckets_maker: dict[str, list] = defaultdict(list)
    buckets_src: dict[str, list] = defaultdict(list)
    calib: list[tuple[float, float]] = []   # (p_star, outcome) for model fills
    resolved = 0

    for f in fills:
        ref = resol.get(f["market_id"])
        if ref is None:
            continue
        resolved += 1
        side, price, size, fee = f["side"], f["price"], f["size"], f["fee"]
        directional = (ref - price) * size if side == "BUY" else (price - ref) * size
        pnl = directional - fee
        buckets_price[(side, _price_bucket(price))].append(pnl)
        buckets_tte[(side, _tte_bucket(f["tte_at_fill"]))].append(pnl)
        buckets_maker["maker" if f["is_maker"] else "taker"].append(pnl)
        buckets_src[f.get("source", "model")].append(pnl)
        # calibration: p* predicts P(YES); outcome is the YES resolution.
        if 0.0 < f["p_star"] < 1.0 and f.get("source", "model") == "model":
            calib.append((f["p_star"], ref))

    print(f"\n=== PnL attribution ({resolved}/{len(fills)} fills resolved) ===")
    if resolved == 0:
        print("No resolved fills yet — attribution needs settled markets "
              "(resolution lookup may be network-blocked here).")
        return

    def _dump(title: str, table: dict, keyfmt) -> None:
        print(f"\n{title}")
        print(f"  {'group':<22} {'n':>4} {'win%':>5} {'PnL':>10} {'PnL/t':>9} {'t-stat':>7}")
        print("  " + "-" * 62)
        for key in sorted(table):
            pnls = table[key]
            total, mean, t = _stats(pnls)
            wr = 100.0 * sum(1 for p in pnls if p > 0) / len(pnls)
            flag = " *" if abs(t) >= 2.0 else ""
            print(f"  {keyfmt(key):<22} {len(pnls):>4} {wr:>4.0f}% "
                  f"{total:>+10.3f} {mean:>+9.4f} {t:>+7.2f}{flag}")

    _dump("By side × entry price:", buckets_price, lambda k: f"{k[0]} {k[1]}")
    _dump("By side × time-to-expiry:", buckets_tte, lambda k: f"{k[0]} {k[1]}")
    _dump("By execution type:", buckets_maker, lambda k: k)
    _dump("By signal source:", buckets_src, lambda k: k)

    _calibration_report(calib)
    print(
        "\nWedge check: if BUY rows in the low price / low-p* / long-TTE buckets "
        "are the worst (negative PnL/t, t-stat ≤ -2), the favourite-longshot "
        "tilt is confirmed on your own fills."
    )


def _calibration_report(calib: list[tuple[float, float]]) -> None:
    if len(calib) < 10:
        print("\nCalibration: too few model fills with a valid p* to assess.")
        return
    brier = sum((p - o) ** 2 for p, o in calib) / len(calib)
    print(f"\nCalibration audit ({len(calib)} model fills) — Brier={brier:.4f} "
          f"(lower is better; 0.25 = uninformative coin-flip)")
    print(f"  {'p* bin':<12} {'n':>4} {'pred':>6} {'realised':>9} {'gap':>7}")
    print("  " + "-" * 42)
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    for lo, hi in zip(edges, edges[1:]):
        b = [(p, o) for p, o in calib if lo <= p < hi]
        if not b:
            continue
        pred = sum(p for p, _ in b) / len(b)
        real = sum(o for _, o in b) / len(b)
        print(f"  [{lo:.1f},{hi:.1f})    {len(b):>4} {pred:>6.3f} "
              f"{real:>9.3f} {real - pred:>+7.3f}")
    print("  (gap > 0 ⇒ p* under-predicts; gap < 0 ⇒ p* over-predicts / too rich)")


def cli() -> None:
    ap = argparse.ArgumentParser(description="PnL attribution + calibration audit on fills.db")
    ap.add_argument("--db", default=default_db_path("fills.db"))
    args = ap.parse_args()
    asyncio.run(attribute(args.db))


if __name__ == "__main__":
    cli()
