"""Replay-based validator for combinatorial-arbitrage robustness.

The live scanner's `rebalance` "arbs" are phantom: Polymarket's UNIFIED order
book makes ask_YES + ask_NO = 1 + spread_YES >= 1 from any consistent snapshot,
so an observed sub-$1 sum is a stale cross-book artifact, not a fillable trade.
The genuine, model-free edge is the strike-monotonicity bull-spread (BUY YES at
K_low, SELL YES at K_high, same expiry → payoff >= credit >= 0 under any path).

This module measures that genuine edge on recorded L2 books:
  * detect every strike-monotonicity violation over the replay timeline,
  * aggregate consecutive detections into EPISODES and measure how long each
    PERSISTS — a violation that lasts < your fill latency is not capturable,
  * report credit distribution and the risk-free realized floor (credit*size).

Market metadata (symbol/strike/expiry per YES token) is recovered from a
decisions DB written by a live/paper run; books come from the captured
poly_events shards.
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass

from .poly_universe import PolyMarket
from .pricing import taker_fee_per_share
from .replay import ReplayEvent, events_for_token, load_poly_events


@dataclass
class ArbEpisode:
    """A maximal run over which one adjacent strike-pair stayed in violation."""
    arb_id: str
    symbol: str
    expiry_ts: float
    k_low: float
    k_high: float
    start_ts: float
    end_ts: float
    n_detections: int
    max_credit: float
    size_floor: float  # min depth seen across the episode → fillable shares

    @property
    def persistence_secs(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def realized_floor(self) -> float:
        """Risk-free lower bound: the strike spread pays >= credit per share."""
        return self.max_credit * self.size_floor


def load_token_markets(db_path: str) -> dict[str, PolyMarket]:
    """Recover YES-token → market metadata (symbol, strike, expiry) from a
    decisions DB. Only threshold markets with a parsed strike qualify."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT token_id, MAX(symbol) symbol, MAX(strike) strike, "
            "MAX(expiry_ts) expiry_ts, MAX(market_id) market_id, MAX(question) question "
            "FROM decisions WHERE token_id != '' AND strike > 0 AND is_threshold = 1 "
            "GROUP BY token_id"
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, PolyMarket] = {}
    for r in rows:
        out[str(r["token_id"])] = PolyMarket(
            condition_id=str(r["market_id"] or ""),
            question=str(r["question"] or ""),
            yes_token_id=str(r["token_id"]),
            no_token_id="",
            yes_price=0.5,
            no_price=0.5,
            strike=float(r["strike"]),
            expiry_ts=float(r["expiry_ts"]),
            tick_size=0.01,
            symbol=str(r["symbol"] or "btcusdt"),
            is_threshold=True,
        )
    return out


def _ladders(markets: list[PolyMarket]) -> dict[tuple[str, float], list[PolyMarket]]:
    out: dict[tuple[str, float], list[PolyMarket]] = {}
    for m in markets:
        out.setdefault((m.symbol, m.expiry_ts), []).append(m)
    for k in out:
        out[k].sort(key=lambda x: x.strike)
    return {k: v for k, v in out.items() if len(v) >= 2}


def _timeline(events: list[ReplayEvent], token_id: str) -> list[tuple[float, object]]:
    return [(ev.ts_wall, ev.snapshot) for ev in events_for_token(events, token_id)]


def _book_at(timeline: list[tuple[float, object]], t: float):
    """Latest snapshot at or before wall-clock t (timeline is sorted)."""
    res = None
    for ts, snap in timeline:
        if ts <= t:
            res = snap
        else:
            break
    return res


def detect_strike_episodes(
    events: list[ReplayEvent],
    markets: list[PolyMarket],
    *,
    min_credit: float = 0.0,
    fee_rate: float = 0.07,
) -> list[ArbEpisode]:
    """Walk the replay timeline; emit one ArbEpisode per contiguous violation
    of strike monotonicity on an adjacent same-expiry pair."""
    episodes: list[ArbEpisode] = []
    timelines = {m.yes_token_id: _timeline(events, m.yes_token_id) for m in markets}

    for (sym, exp), ladder in _ladders(markets).items():
        for i in range(len(ladder) - 1):
            m_low, m_high = ladder[i], ladder[i + 1]
            t_low = timelines.get(m_low.yes_token_id, [])
            t_high = timelines.get(m_high.yes_token_id, [])
            if not t_low or not t_high:
                continue
            times = sorted({ts for ts, _ in t_low} | {ts for ts, _ in t_high})
            arb_id = f"strike-{sym}-{int(exp)}-{i}"

            in_ep = False
            start = max_credit = n = 0.0
            size_floor = 0.0
            for t in times:
                bl = _book_at(t_low, t)
                bh = _book_at(t_high, t)
                credit = 0.0
                violated = False
                if (bl is not None and bh is not None
                        and 0.0 < bl.best_ask < 1.0 and 0.0 < bh.best_bid < 1.0):
                    fee_low = taker_fee_per_share(bl.best_ask, fee_rate)
                    fee_high = taker_fee_per_share(bh.best_bid, fee_rate)
                    credit = bh.best_bid - bl.best_ask - fee_low - fee_high
                    if credit > 0 and credit >= min_credit:
                        violated = True
                        size = min(bl.ask_size, bh.bid_size)
                if violated:
                    if not in_ep:
                        in_ep, start, max_credit, n, size_floor = True, t, credit, 1, size
                    else:
                        max_credit = max(max_credit, credit)
                        size_floor = min(size_floor, size)
                        n += 1
                elif in_ep:
                    episodes.append(ArbEpisode(
                        arb_id, sym, exp, m_low.strike, m_high.strike,
                        start, t, int(n), max_credit, size_floor))
                    in_ep = False
            if in_ep:
                episodes.append(ArbEpisode(
                    arb_id, sym, exp, m_low.strike, m_high.strike,
                    start, times[-1], int(n), max_credit, size_floor))
    return episodes


def pair_coverage(events: list[ReplayEvent], markets: list[PolyMarket]) -> tuple[int, int]:
    """(adjacent pairs total, pairs where BOTH legs have book data). A '0
    violations' result is only meaningful relative to how many pairs we could
    actually evaluate."""
    toks = {ev.token_id for ev in events}
    total = with_data = 0
    for ladder in _ladders(markets).values():
        for i in range(len(ladder) - 1):
            total += 1
            if ladder[i].yes_token_id in toks and ladder[i + 1].yes_token_id in toks:
                with_data += 1
    return total, with_data


def summarize(episodes: list[ArbEpisode], latencies=(0.5, 1.0, 2.0)) -> dict:
    n = len(episodes)
    by_latency = {
        lat: [e for e in episodes if e.persistence_secs >= lat] for lat in latencies
    }
    credits = sorted(e.max_credit for e in episodes)
    return {
        "episodes": n,
        "median_persistence_secs": (
            sorted(e.persistence_secs for e in episodes)[n // 2] if n else 0.0
        ),
        "max_persistence_secs": max((e.persistence_secs for e in episodes), default=0.0),
        "median_credit": credits[n // 2] if n else 0.0,
        "max_credit": credits[-1] if n else 0.0,
        "capturable": {
            lat: {
                "count": len(eps),
                "realized_floor_usd": round(sum(e.realized_floor for e in eps), 4),
            }
            for lat, eps in by_latency.items()
        },
    }


def report(run_dir: str, db_path: str, *, min_credit: float = 0.0) -> dict:
    events = load_poly_events(run_dir)
    markets = list(load_token_markets(db_path).values())
    episodes = detect_strike_episodes(events, markets, min_credit=min_credit)
    s = summarize(episodes)
    pairs_total, pairs_data = pair_coverage(events, markets)
    s["pairs_total"] = pairs_total
    s["pairs_evaluable"] = pairs_data
    print("\n=== Strike-Arb Replay Validator ===")
    print(f"Run dir:            {run_dir}")
    print(f"Threshold markets:  {len(markets)}  (ladders >=2 strikes)")
    print(f"Replay events:      {len(events)}")
    print(f"Adjacent pairs:     {pairs_data} evaluable (both legs have book data) / {pairs_total} total")
    print(f"min_credit:         {min_credit:.4f} (after taker fees)")
    print(f"Violation episodes: {s['episodes']}")
    if s["episodes"]:
        print(f"Persistence:        median={s['median_persistence_secs']:.2f}s "
              f"max={s['max_persistence_secs']:.2f}s")
        print(f"Credit (max/ep):    median={s['median_credit']:.4f} max={s['max_credit']:.4f}")
        print("Capturable (survives fill latency):")
        for lat, c in s["capturable"].items():
            print(f"  > {lat:>4.1f}s latency: {c['count']:>4} episodes  "
                  f"risk-free floor ${c['realized_floor_usd']:+.2f}")
    else:
        print("No strike-monotonicity violations found — the model-free arb did "
              "not appear in this window.")
    return s


def cli() -> None:
    ap = argparse.ArgumentParser(description="Strike-arb robustness replay validator")
    ap.add_argument("run_dir", help="poly_events/<run_id> directory")
    ap.add_argument("--db", required=True, help="decisions DB with market metadata")
    ap.add_argument("--min-credit", type=float, default=0.0)
    args = ap.parse_args()
    report(args.run_dir, args.db, min_credit=args.min_credit)


if __name__ == "__main__":
    cli()
