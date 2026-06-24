"""
Per-category PnL tracker + auto-blacklist.

Markets are bucketed into categories by structural features
(symbol × kind × tte_bucket × price_bucket).  Cumulative realised PnL
is tracked per category in `category_state.json`.  When a category's
cumulative PnL drops below `BLACKLIST_LOSS_THRESHOLD`, future markets
in that category are skipped — the bot effectively learns from its own
losses without retraining.

This is a coarse defence against adverse selection: if the maker bots
on Polymarket consistently win against us in (e.g.) "ETH up/down 5min
ATM" markets, the running PnL on that bucket drops, the bucket gets
blacklisted, and we stop bleeding into it.

Usage:
  from .category import CategoryTracker
  tracker = CategoryTracker.load()
  if tracker.is_blacklisted(market, side, exec_price):
      skip
  ...
  tracker.record_fill(market, side, exec_price, pnl_per_share)
  tracker.save()

Or rebuild from history:
  python -m src.category rebuild

CLI:
  python -m src.category report     # show per-category PnL
  python -m src.category rebuild    # rebuild from fills.db + resolutions
  python -m src.category clear      # wipe state
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from .storage import default_db_path, env_db_path

log = logging.getLogger(__name__)

STATE_PATH = "category_state.json"
BLACKLIST_LOSS_THRESHOLD: float = -50.0   # $ — once a category is below this, skip it
MIN_FILLS_BEFORE_BLACKLIST: int = 5       # need a few samples before judging


@dataclass
class CategoryStats:
    fills: int = 0
    realised_pnl: float = 0.0
    wins: int = 0
    losses: int = 0


@dataclass
class CategoryTracker:
    stats: dict[str, CategoryStats] = field(default_factory=dict)
    threshold: float = BLACKLIST_LOSS_THRESHOLD
    min_fills: int = MIN_FILLS_BEFORE_BLACKLIST

    # ----- key derivation -----
    @staticmethod
    def category_key(
        market_question: str,
        symbol: str,
        time_to_expiry_secs: float,
        exec_price: float,
    ) -> str:
        """Bucket a market into a coarse, stable category key."""
        q = (market_question or "").lower()
        if "up or down" in q or "up-or-down" in q:
            kind = "updown"
        elif "above" in q or "reach" in q or "over" in q:
            kind = "threshold"
        else:
            kind = "other"

        if time_to_expiry_secs < 600:
            tte = "u10m"
        elif time_to_expiry_secs < 1800:
            tte = "u30m"
        elif time_to_expiry_secs < 3600:
            tte = "u1h"
        else:
            tte = "ge1h"

        if exec_price < 0.20:
            px = "tail_lo"
        elif exec_price < 0.40:
            px = "low"
        elif exec_price <= 0.60:
            px = "atm"
        elif exec_price <= 0.80:
            px = "high"
        else:
            px = "tail_hi"

        return f"{symbol}.{kind}.{tte}.{px}"

    # ----- state mutators -----
    def record_fill(self, key: str, pnl: float) -> None:
        s = self.stats.setdefault(key, CategoryStats())
        s.fills += 1
        s.realised_pnl += pnl
        if pnl > 0:
            s.wins += 1
        elif pnl < 0:
            s.losses += 1

    def is_blacklisted(self, key: str) -> bool:
        s = self.stats.get(key)
        if s is None or s.fills < self.min_fills:
            return False
        return s.realised_pnl <= self.threshold

    # ----- persistence -----
    @classmethod
    def load(cls, path: str = STATE_PATH) -> "CategoryTracker":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            stats = {k: CategoryStats(**v) for k, v in (data.get("stats") or {}).items()}
            return cls(
                stats=stats,
                threshold=float(data.get("threshold", BLACKLIST_LOSS_THRESHOLD)),
                min_fills=int(data.get("min_fills", MIN_FILLS_BEFORE_BLACKLIST)),
            )
        except Exception as exc:
            log.warning("Failed to load %s: %s — starting fresh.", path, exc)
            return cls()

    def save(self, path: str = STATE_PATH) -> None:
        payload = {
            "saved_at": time.time(),
            "threshold": self.threshold,
            "min_fills": self.min_fills,
            "stats": {k: asdict(v) for k, v in self.stats.items()},
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)

    # ----- reporting -----
    def report(self) -> str:
        if not self.stats:
            return "(no categories yet)"
        rows = sorted(self.stats.items(), key=lambda kv: kv[1].realised_pnl)
        out = [f"{'category':<32} {'fills':>6} {'wins':>5} {'losses':>5} {'PnL':>12} {'flag':>10}"]
        out.append("-" * 78)
        for k, s in rows:
            flag = "BLACKLIST" if (s.fills >= self.min_fills and s.realised_pnl <= self.threshold) else ""
            out.append(f"{k:<32} {s.fills:>6} {s.wins:>5} {s.losses:>5} {s.realised_pnl:>+12.2f} {flag:>10}")
        return "\n".join(out)


# --------- CLI: rebuild from fills.db + Gamma resolution lookup ---------

def _resolve_fills_db(db_path: str | None) -> str:
    """Resolve the fills DB the executor actually wrote.

    The executor uses `env_db_path("FILL_DB_PATH", "fills.db")`; the previous
    reconcile hard-coded `data/db/fills.db` and crashed at shutdown with
    'no such table: fills' whenever the run wrote e.g. fills_strict.db.
    """
    if db_path:
        return db_path
    return env_db_path("FILL_DB_PATH", "fills.db")


async def _rebuild_from_fills(db_path: str | None = None) -> CategoryTracker:
    import aiohttp
    from .pnl import _fetch_resolution, _fetch_book_mid

    db_path = _resolve_fills_db(db_path)
    if not os.path.exists(db_path):
        log.warning("Category reconcile: fills DB %s not found; skipping.", db_path)
        return CategoryTracker()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        fills = conn.execute(
            "SELECT ts, market_id, token_id, side, price, size, fee, p_star, edge "
            "FROM fills ORDER BY ts ASC"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.warning("Category reconcile: %s in %s; skipping.", exc, db_path)
        return CategoryTracker()
    finally:
        conn.close()

    if not fills:
        return CategoryTracker()

    # Group fills by token_id; we need entry side, avg price, total size, fee.
    positions: dict[str, dict] = {}
    for f in fills:
        p = positions.setdefault(f["token_id"], {
            "market_id": f["market_id"],
            "shares": 0.0, "weighted_px": 0.0, "fee": 0.0,
            "first_question": "",
        })
        delta = f["size"] if f["side"] == "BUY" else -f["size"]
        if (p["shares"] >= 0 and delta > 0) or (p["shares"] <= 0 and delta < 0):
            p["weighted_px"] = (
                p["weighted_px"] * abs(p["shares"]) + f["price"] * abs(delta)
            ) / max(1e-9, abs(p["shares"]) + abs(delta))
        p["shares"] += delta
        p["fee"] += f["fee"]

    # We don't have the question text in fills.db — need to look it up by
    # condition_id via Gamma. For categorisation purposes, treat unknown
    # as "other.atm".  Better: we record the question on each fill in the
    # main bot once this lands.
    tracker = CategoryTracker()
    async with aiohttp.ClientSession() as s:
        # Fetch resolutions in bulk.
        unique_markets = list({p["market_id"] for p in positions.values()})
        resolutions = await asyncio.gather(*[_fetch_resolution(s, m) for m in unique_markets])
        resol_by_market = dict(zip(unique_markets, resolutions))

    for token_id, p in positions.items():
        resol = resol_by_market.get(p["market_id"])
        if resol is None:
            # Infer from entry price for tail trades.
            avg = p["weighted_px"]
            if avg <= 0.05 and p["shares"] < 0:
                resol = 0.0   # sold deep OTM, assume NO won
            elif avg >= 0.95 and p["shares"] > 0:
                resol = 1.0
            else:
                continue       # skip unresolved
        ref = resol
        avg = p["weighted_px"]
        pnl = (ref - avg) * p["shares"] - p["fee"]
        # Without question text, lump into kind=unknown but bucket by price.
        key = CategoryTracker.category_key(
            market_question="", symbol="unknown",
            time_to_expiry_secs=0, exec_price=avg,
        )
        tracker.record_fill(key, pnl)
    return tracker


def main() -> int:
    parser = argparse.ArgumentParser(description="Category PnL tracker / blacklist manager")
    parser.add_argument("cmd", choices=["report", "rebuild", "clear"])
    parser.add_argument("--db", default=env_db_path("FILL_DB_PATH", "fills.db"))
    parser.add_argument("--state", default=STATE_PATH)
    args = parser.parse_args()

    if args.cmd == "report":
        t = CategoryTracker.load(args.state)
        print(t.report())
        print(f"\nthreshold: ${t.threshold:.2f}    min_fills: {t.min_fills}")
        return 0
    elif args.cmd == "rebuild":
        t = asyncio.run(_rebuild_from_fills(args.db))
        t.save(args.state)
        print(t.report())
        print(f"\nSaved to {args.state}")
        return 0
    elif args.cmd == "clear":
        if os.path.exists(args.state):
            os.remove(args.state)
            print(f"Removed {args.state}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
