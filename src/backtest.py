"""
Historical backtest harness.

For each recently-resolved BTC/ETH threshold market:
  1. Pull the YES-token price history from Polymarket at 1-minute fidelity.
  2. Pull Binance 1-minute klines for the same window.
  3. For every minute, recompute p* with the same signal logic and check
     whether a buy/sell edge would have fired.
  4. Mark each hypothetical fill against the resolution outcome
     (YES → 1.0, NO → 0.0) and aggregate PnL.

This is an *upper bound* on profit: it assumes (a) we cross the displayed
last-trade price (no slippage past one tick), (b) we win the queue against
co-located bots, (c) realized vol used for sizing is the same as our σ
estimator would have produced live. Real performance is strictly worse.

Usage:
  python -m src.backtest --days 3
  python -m src.backtest --days 7 --max-markets 50 --safety-eps 0.005
"""

import argparse
import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from .pricing import implied_prob, taker_fee_per_share, FEE_RATE_CRYPTO

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

THRESHOLD_RE = re.compile(
    r"(?:bitcoin|btc|ethereum|eth)\s+(?:above|reach(?:es)?|over|>=?)", re.I
)
UPDOWN_RE = re.compile(
    r"(?:bitcoin|btc|ethereum|eth)\s+(?:up or down|up-or-down)", re.I
)
ETH_RE = re.compile(r"(?:ethereum|eth)", re.I)
PRICE_RE = re.compile(r"\$?([\d]{1,3}(?:,\d{3})+(?:\.\d+)?|\d{3,}(?:\.\d+)?)")


@dataclass
class BacktestConfig:
    days: int = 3
    max_markets: int = 100
    max_notional: float = 25.0
    safety_eps: float = 0.003
    fee_rate: float = FEE_RATE_CRYPTO
    sigma_window_secs: float = 60 * 60   # 1h trailing realized vol
    min_tte_secs: float = 120.0          # don't trade in last 2 min


@dataclass
class SimFill:
    market_question: str
    symbol: str
    ts: float
    side: str
    price: float
    size: float
    p_star: float
    edge: float
    fee: float
    resolution: float       # 1.0 or 0.0
    pnl: float


def _parse_strike(question: str) -> float:
    matches = PRICE_RE.findall(question)
    candidates = []
    for m in matches:
        try:
            candidates.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return max(candidates) if candidates else 0.0


def _parse_iso(s: str | None) -> float:
    if not s:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(s).rstrip("Z")).replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0


async def _fetch_resolved_markets(session: aiohttp.ClientSession, days: int) -> list[dict]:
    """Pull resolved BTC/ETH markets (threshold + Up/Down) from last `days`."""
    headers = {"User-Agent": "Mozilla/5.0"}
    now = time.time()
    cutoff = now - days * 86400
    now_iso = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[dict] = []
    offset = 0
    # closed=true alone returns long-dated future markets that closed early;
    # constraining end_date_max=now restricts to truly past-resolved markets.
    while offset < 5000:
        url = (
            f"{GAMMA_API}?closed=true&limit=500&offset={offset}"
            f"&end_date_max={now_iso}&order=endDate&ascending=false"
        )
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                break
            batch = await r.json(content_type=None)
        if not batch:
            break
        for m in batch:
            q = m.get("question", "")
            if not (THRESHOLD_RE.search(q) or UPDOWN_RE.search(q)):
                continue
            end_ts = _parse_iso(m.get("endDate"))
            if end_ts < cutoff:
                return out
            out.append(m)
        if len(batch) < 500:
            break
        offset += len(batch)
    return out


async def _fetch_token_history(
    session: aiohttp.ClientSession, token_id: str, start_ts: float, end_ts: float
) -> list[tuple[float, float]]:
    """Return [(unix_ts, price), ...] for a token over the window."""
    params = {
        "market": token_id,
        "startTs": int(start_ts),
        "endTs": int(end_ts),
        "fidelity": "1",   # 1-minute buckets
    }
    try:
        async with session.get(
            CLOB_HISTORY, params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            if r.status != 200:
                return []
            d = await r.json()
            return [(float(p["t"]), float(p["p"])) for p in d.get("history", [])]
    except Exception as exc:
        log.debug("History fetch failed for %s: %s", token_id[:12], exc)
        return []


async def _fetch_binance_klines(
    session: aiohttp.ClientSession, symbol: str, start_ts: float, end_ts: float
) -> list[tuple[float, float]]:
    """Return [(unix_ts, close_price), ...] from Binance 1-minute klines."""
    out: list[tuple[float, float]] = []
    cur = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)
    while cur < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            async with session.get(
                BINANCE_KLINES, params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status != 200:
                    return out
                rows = await r.json()
        except Exception:
            return out
        if not rows:
            break
        for row in rows:
            # row: [openTime, open, high, low, close, vol, closeTime, ...]
            out.append((row[0] / 1000.0, float(row[4])))
        last_open = rows[-1][0]
        if len(rows) < 1000:
            break
        cur = last_open + 60_000
    return out


def _rolling_sigma(closes: list[tuple[float, float]], lookback_secs: float) -> dict[float, float]:
    """For each timestamp, return annualised σ from prior log-returns within lookback."""
    out: dict[float, float] = {}
    for i in range(1, len(closes)):
        cutoff = closes[i][0] - lookback_secs
        sq = 0.0
        n = 0
        for j in range(i - 1, -1, -1):
            if closes[j][0] < cutoff:
                break
            r = math.log(closes[j + 1][1] / closes[j][1])
            sq += r * r
            n += 1
        if n >= 5:
            window = closes[i][0] - closes[max(0, i - n)][0]
            if window > 0:
                var_per_sec = sq / window
                out[closes[i][0]] = math.sqrt(var_per_sec * 365.25 * 86400)
    return out


def _resolution(market: dict) -> float | None:
    """Extract YES outcome (1.0 or 0.0) from a resolved market."""
    prices = market.get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except Exception:
            return None
    if not prices or len(prices) < 1:
        return None
    try:
        return float(prices[0])
    except (ValueError, TypeError):
        return None


async def backtest(cfg: BacktestConfig) -> list[SimFill]:
    fills: list[SimFill] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(headers=headers) as session:
        log.info("Fetching resolved markets from last %d day(s)...", cfg.days)
        markets = await _fetch_resolved_markets(session, cfg.days)
        log.info("Found %d resolved BTC/ETH threshold markets", len(markets))
        markets = markets[: cfg.max_markets]

        for i, m in enumerate(markets):
            q = m.get("question", "")
            is_threshold = bool(THRESHOLD_RE.search(q))
            strike = _parse_strike(q) if is_threshold else 0.0
            symbol = "ETHUSDT" if ETH_RE.search(q) else "BTCUSDT"
            sym_lower = symbol.lower()
            expiry = _parse_iso(m.get("endDate"))
            start_ts = _parse_iso(m.get("startDate")) or (expiry - 4 * 3600)
            if expiry <= 0 or start_ts >= expiry:
                continue

            resolution = _resolution(m)
            if resolution is None:
                continue

            # Modern shape: clobTokenIds is a JSON-encoded list aligned with outcomes.
            raw_ids = m.get("clobTokenIds")
            if not raw_ids:
                continue
            try:
                ids = raw_ids if isinstance(raw_ids, list) else json.loads(raw_ids)
            except Exception:
                continue
            outcomes = m.get("outcomes")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except Exception:
                    outcomes = []
            yes_token = ""
            for idx, oc in enumerate(outcomes or []):
                if str(oc).lower() in ("yes", "up", "above", "higher"):
                    yes_token = str(ids[idx]) if idx < len(ids) else ""
                    break
            if not yes_token:
                continue

            history = await _fetch_token_history(session, yes_token, start_ts, expiry)
            if len(history) < 5:
                continue
            klines = await _fetch_binance_klines(session, symbol, start_ts, expiry)
            if len(klines) < 5:
                continue
            sigmas = _rolling_sigma(klines, cfg.sigma_window_secs)
            kline_by_ts = {ts: c for ts, c in klines}
            kline_keys = sorted(kline_by_ts)

            # For Up/Down markets the strike is the Binance close at start_ts.
            if not is_threshold:
                idx = _bisect_le(kline_keys, start_ts)
                if idx < 0:
                    continue
                strike = kline_by_ts[kline_keys[idx]]

            for poly_ts, poly_price in history:
                tte = expiry - poly_ts
                if tte < cfg.min_tte_secs:
                    continue
                # Snap to closest binance close <= poly_ts
                idx = _bisect_le(kline_keys, poly_ts)
                if idx < 0:
                    continue
                kts = kline_keys[idx]
                spot = kline_by_ts[kts]
                sigma = sigmas.get(kts)
                if not sigma or sigma <= 0.05:
                    continue

                p_star = implied_prob(spot, strike, tte, sigma)

                # Approx: use last-trade price as both bid and ask (no spread info).
                # Apply taker fee at the price we cross.
                fee_per_share = taker_fee_per_share(poly_price, cfg.fee_rate)

                edge_buy = p_star - poly_price - fee_per_share - cfg.safety_eps
                edge_sell = poly_price - p_star - fee_per_share - cfg.safety_eps
                if max(edge_buy, edge_sell) <= 0:
                    continue

                if edge_buy > edge_sell:
                    side, edge = "BUY", edge_buy
                else:
                    side, edge = "SELL", edge_sell

                if poly_price <= 0 or poly_price >= 1:
                    continue
                size = round(cfg.max_notional / poly_price, 2)

                # Realized PnL at resolution.
                if side == "BUY":
                    pnl = (resolution - poly_price) * size - fee_per_share * size
                else:
                    pnl = (poly_price - resolution) * size - fee_per_share * size

                fills.append(SimFill(
                    market_question=q,
                    symbol=sym_lower,
                    ts=poly_ts,
                    side=side,
                    price=poly_price,
                    size=size,
                    p_star=p_star,
                    edge=edge,
                    fee=fee_per_share * size,
                    resolution=resolution,
                    pnl=pnl,
                ))

            if (i + 1) % 10 == 0:
                log.info("...processed %d/%d markets, %d sim-fills so far",
                         i + 1, len(markets), len(fills))

    return fills


def _bisect_le(sorted_list: list[float], target: float) -> int:
    """Return largest index i such that sorted_list[i] <= target, else -1."""
    lo, hi = 0, len(sorted_list) - 1
    if hi < 0 or sorted_list[0] > target:
        return -1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sorted_list[mid] <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _summarize(fills: list[SimFill]) -> None:
    if not fills:
        print("\nNo simulated fills — no edge above threshold in window.")
        return
    n = len(fills)
    total_notional = sum(f.price * f.size for f in fills)
    total_fee = sum(f.fee for f in fills)
    total_pnl = sum(f.pnl for f in fills)
    wins = [f for f in fills if f.pnl > 0]
    losses = [f for f in fills if f.pnl < 0]
    by_market = {}
    for f in fills:
        by_market.setdefault(f.market_question, []).append(f)

    print(f"\n=== Backtest summary ({n} sim-fills across {len(by_market)} markets) ===")
    print(f"Gross notional:  ${total_notional:,.2f}")
    print(f"Total fees:      ${total_fee:.4f}")
    print(f"Realized PnL:    ${total_pnl:+,.4f}")
    print(f"Win rate:        {len(wins)/n*100:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Mean edge fired: {sum(f.edge for f in fills)/n*100:+.2f}%")
    if wins:
        print(f"Avg win:         ${sum(f.pnl for f in wins)/len(wins):+.3f}")
    if losses:
        print(f"Avg loss:        ${sum(f.pnl for f in losses)/len(losses):+.3f}")
    print(f"Return on notional: {total_pnl/total_notional*100:+.2f}%" if total_notional else "")

    # Top winners / losers
    fills_by_pnl = sorted(fills, key=lambda f: f.pnl)
    print("\nWorst 5 fills:")
    for f in fills_by_pnl[:5]:
        print(f"  {f.side} @{f.price:.3f} → {f.resolution:.0f} pnl={f.pnl:+.3f} | {f.market_question[:55]}")
    print("Best 5 fills:")
    for f in fills_by_pnl[-5:][::-1]:
        print(f"  {f.side} @{f.price:.3f} → {f.resolution:.0f} pnl={f.pnl:+.3f} | {f.market_question[:55]}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Backtest the arb signal on resolved markets")
    parser.add_argument("--days", type=int, default=3, help="lookback window")
    parser.add_argument("--max-markets", type=int, default=100)
    parser.add_argument("--max-notional", type=float, default=25.0)
    parser.add_argument("--safety-eps", type=float, default=0.003)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    cfg = BacktestConfig(
        days=args.days,
        max_markets=args.max_markets,
        max_notional=args.max_notional,
        safety_eps=args.safety_eps,
    )
    fills = asyncio.run(backtest(cfg))
    _summarize(fills)


if __name__ == "__main__":
    cli()
