"""
Historical backtest harness.

For each recently-resolved BTC/ETH threshold market:
  1. Pull the YES-token price history from Polymarket at 1-minute fidelity.
  2. Pull Binance 1-minute klines for the same window.
  3. For every minute, synthesise a bid/ask book around the displayed price
     and drive the *real* `SignalGenerator.evaluate()` — the exact same code
     path the live bot runs — so every overlay (favourite-longshot wedge,
     vol-smile σ bump, funding carry, fractional-Kelly sizing, maker/rebate
     path, effective-spread + taker-fee costs) is exercised identically.
  4. Mark each hypothetical fill against the resolution outcome
     (YES → 1.0, NO → 0.0) and aggregate PnL.

What this models now (vs. the old last-trade-price harness):
  - Fills cross a synthesised spread (BUY pays the ask, SELL hits the bid),
    so the half-spread cost is no longer ignored.
  - Taker fee / maker rebate are applied by the live signal code.
  - By default at most ONE entry per market is taken (the first qualifying
    minute) so the headline win-rate is over *independent* markets, not the
    same outcome counted once per minute. Use --all-fills to see every minute.

Still optimistic — real performance is worse — because we cannot replay:
  - true historical book depth / queue position (we assume top-of-book fill),
  - the OFI/ML/OBI microstructure overlays (no historical L2 + trade tape),
  - live funding/IV oracles (carry and Deribit-IV σ default to off here).
These gaps are reported in the summary so the number is not over-trusted.

Usage:
  python -m src.backtest --days 3
  python -m src.backtest --days 7 --max-markets 50 --half-spread 0.01
  python -m src.backtest --days 7 --all-fills        # every minute, not 1/market
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from .pricing import (
    implied_prob,
    taker_fee_per_share,
    maker_rebate_per_share,
    load_wedge_coeffs,
    load_calibration,
    FEE_RATE_CRYPTO,
)
from .config import Settings
from .signal import SignalGenerator, Side
from .poly_universe import PolyMarket
from .poly_ws import BookSnapshot
from .binance_ws import BinanceTick
from .microstructure import MicroReplay, MicrostructureEngine, DirectionalModel

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/markets"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_AGGTRADES = "https://api.binance.com/api/v3/aggTrades"

# Market workers can run in parallel, but each upstream gets its own polite
# gate. Polymarket documents /prices-history at 1,000 req / 10s; use 80% of
# that ceiling so a deployed backtest has room for jitter and other clients.
CLOB_CONCURRENCY = 20
CLOB_MIN_INTERVAL_SECS = 0.0125
BINANCE_CONCURRENCY = 3
BINANCE_MIN_INTERVAL_SECS = 0.08
HTTP_RETRIES = 4
CLOB_USER_AGENT = "scalping-test-backtest/0.1 (+polite-rate-limited)"

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
    max_markets: int = 300         # sampled evenly across the window
    max_notional: float = 25.0
    safety_eps: float | None = None      # None → use the live Config value
    fee_rate: float = FEE_RATE_CRYPTO
    sigma_window_secs: float = 60 * 60   # 1h trailing realized vol
    # Synthesised half-spread (price units) applied either side of the
    # displayed Polymarket price, since prices-history gives no L2 book.
    # 0.01 = one cent each side (2c wide), a realistic crypto-market quote.
    half_spread: float = 0.01
    # One entry per market (independent samples) unless overridden.
    all_fills: bool = False
    trade_updown: bool = False           # Up/Down strikes are synthetic; off
    debug: bool = False                  # print why markets/minutes were dropped
    # F2 residuals:
    carry_annual: float = 0.0            # perp-funding carry fed to the pricer
    maker_fill_prob: float = 1.0         # P(resting maker order is hit); <1 = honest
    ofi_replay: bool = False             # replay Binance aggTrades → OFI/ML overlay
    dump_fills: bool = False
    skip_first_secs: float = 0.0
    nofill_tail: float = 0.0             # reject prices below x or above 1-x when >0
    persistence_cents: float = 0.0       # require prev/next print within x cents when >0
    short_tte_sigma_floor: float = 0.0   # additional sigma floor for short-TTE realism


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
    is_maker: bool = False
    expiry_ts: float = 0.0  # market resolution time — for correlation grouping


class RateGate:
    def __init__(self, concurrency: int, min_interval_secs: float):
        self.sem = asyncio.Semaphore(concurrency)
        self.min_interval_secs = min_interval_secs
        self.lock = asyncio.Lock()
        self.next_at = 0.0

    async def wait_turn(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_at = time.monotonic() + self.min_interval_secs


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


# The BTC/ETH hourly threshold markets live in dedicated Gamma "series".
# Fetching by series (server-side filtered) is the only way to reach real
# history: the unfiltered /markets feed returns thousands of all-category
# closed markets per day, so paging it never reaches yesterday. Each event in
# a series is one resolution window (e.g. "Bitcoin above ___ on June 21, 3PM
# ET?") and carries ~20 strike markets with full clobTokenIds/outcomePrices.
EVENTS_API = "https://gamma-api.polymarket.com/events"
_SERIES_SLUGS = [
    "bitcoin-multi-strikes-hourly",
    "ethereum-multi-strikes-hourly",
]
_GAMMA_PAGE = 100
_MAX_PAGES = 40          # per series; hourly events ≈ a few days/page
_FRESH_SKIP_SECS = 3600  # skip markets resolved <1h ago (CLOB history not ready)


async def _fetch_resolved_markets(
    session: aiohttp.ClientSession, days: int
) -> tuple[list[dict], bool]:
    """Pull resolved BTC/ETH threshold markets from the last `days` via the
    hourly multi-strike series.

    Returns (markets, truncated); `truncated` is True if the page budget ran
    out before reaching the `days` cutoff for some series — i.e. the realised
    window is shorter than requested.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    cutoff = time.time() - days * 86400
    out: list[dict] = []
    truncated = False
    for slug in _SERIES_SLUGS:
        reached_cutoff = False
        for page in range(_MAX_PAGES):
            url = (
                f"{EVENTS_API}?series_slug={slug}&closed=true&limit={_GAMMA_PAGE}"
                f"&offset={page * _GAMMA_PAGE}&order=endDate&ascending=false"
            )
            try:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as r:
                    if r.status != 200:
                        break
                    batch = await r.json(content_type=None)
            except Exception as exc:
                log.debug("Event fetch failed for %s p%d: %s", slug, page, exc)
                break
            if not batch:
                reached_cutoff = True
                break
            stop = False
            for ev in batch:
                if _parse_iso(ev.get("endDate")) < cutoff:
                    stop = True
                    break
                ev_end, ev_start = ev.get("endDate"), ev.get("startDate")
                for mk in ev.get("markets", []):
                    q = mk.get("question", "")
                    if not (THRESHOLD_RE.search(q) or UPDOWN_RE.search(q)):
                        continue
                    # Markets in a series event occasionally omit their own
                    # dates — inherit the event window so pricing still works.
                    mk.setdefault("endDate", ev_end)
                    mk.setdefault("startDate", ev_start)
                    out.append(mk)
            if stop or len(batch) < _GAMMA_PAGE:
                reached_cutoff = True
                break
        if not reached_cutoff:
            truncated = True
    return out, truncated


async def _fetch_token_history(
    session: aiohttp.ClientSession,
    gate: RateGate,
    token_id: str,
    start_ts: float,
    end_ts: float,
) -> list[tuple[float, float]] | None:
    """Return price history, [] if truly empty, None if fetch failed."""
    params = {
        "market": token_id,
        "startTs": int(start_ts),
        "endTs": int(end_ts),
        "fidelity": "1",   # 1-minute buckets
    }
    d = await _get_json_urllib(gate, CLOB_HISTORY, params, timeout_secs=15)
    if d is None:
        log.debug("History fetch failed for %s", token_id[:12])
        return None
    return [(float(p["t"]), float(p["p"])) for p in d.get("history", [])]


async def _fetch_binance_klines(
    session: aiohttp.ClientSession,
    gate: RateGate,
    symbol: str,
    start_ts: float,
    end_ts: float,
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
        rows = await _get_json(session, gate, BINANCE_KLINES, params, timeout_secs=10)
        if rows is None:
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


async def _fetch_agg_trades(
    session: aiohttp.ClientSession,
    gate: RateGate,
    symbol: str,
    start_ts: float,
    end_ts: float,
) -> list[tuple[float, float, float, float]]:
    """Return [(ts, price, signed_dollar_vol, abs_dollar_vol), ...].

    Binance aggTrades caps a query at 1h span / 1000 rows, so we chunk by hour
    and paginate within a chunk by advancing startTime past the last id.
    m=true means the buyer was the maker → the aggressor sold → signed flow is
    negative (matches binance_ws's live OFI sign convention)."""
    out: list[tuple[float, float, float, float]] = []
    chunk = 3600 * 1000
    cur = int(start_ts * 1000)
    end_ms = int(end_ts * 1000)
    while cur < end_ms:
        chunk_end = min(cur + chunk, end_ms)
        sub = cur
        while sub < chunk_end:
            params = {"symbol": symbol.upper(), "startTime": sub, "endTime": chunk_end, "limit": 1000}
            rows = await _get_json(session, gate, BINANCE_AGGTRADES, params, timeout_secs=10)
            if rows is None:
                return out
            if not rows:
                break
            for t in rows:
                try:
                    ts = t["T"] / 1000.0
                    price = float(t["p"])
                    qty = float(t["q"])
                    sign = -1.0 if t.get("m") else 1.0
                    dv = price * qty
                    out.append((ts, price, sign * dv, dv))
                except (KeyError, ValueError, TypeError):
                    continue
            if len(rows) < 1000:
                break
            sub = rows[-1]["T"] + 1
        cur = chunk_end
    return out


async def _get_json(
    session: aiohttp.ClientSession,
    gate: RateGate,
    url: str,
    params: dict,
    timeout_secs: float,
):
    delay = 0.0
    for attempt in range(HTTP_RETRIES):
        async with gate.sem:
            await gate.wait_turn()
            try:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout_secs),
                ) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    if r.status in (429, 500, 502, 503, 504):
                        delay = _retry_delay(attempt, r.headers.get("Retry-After"))
                        log.debug("HTTP %s from %s; retrying in %.2fs", r.status, url, delay)
                    else:
                        log.debug("HTTP %s from %s", r.status, url)
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                delay = _retry_delay(attempt, None)
                log.debug("HTTP fetch failed from %s: %s; retrying in %.2fs", url, exc, delay)
        await asyncio.sleep(delay)
    return None


async def _get_json_urllib(
    gate: RateGate,
    url: str,
    params: dict,
    timeout_secs: float,
):
    delay = 0.0
    for attempt in range(HTTP_RETRIES):
        async with gate.sem:
            await gate.wait_turn()
            data, retry_after, retryable = await asyncio.to_thread(
                _get_json_urllib_once, url, params, timeout_secs,
            )
            if data is not None:
                return data
            if not retryable:
                return None
            delay = _retry_delay(attempt, retry_after)
        await asyncio.sleep(delay)
    return None


def _get_json_urllib_once(url: str, params: dict, timeout_secs: float):
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": CLOB_USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_secs) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body), None, False
    except urllib.error.HTTPError as exc:
        status = exc.code
        retry_after = exc.headers.get("Retry-After")
        if status in (429, 500, 502, 503, 504):
            log.debug("HTTP %s from %s; retrying", status, url)
            return None, retry_after, True
        log.debug("HTTP %s from %s", status, url)
        return None, None, False
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        log.debug("HTTP fetch failed from %s: %s; retrying", url, exc)
        return None, None, True


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(30.0, max(0.5, float(retry_after)))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2 ** attempt))


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


def _build_signal_generator(cfg: BacktestConfig, micro_engine=None) -> SignalGenerator:
    """Construct a SignalGenerator from the live Config so the backtest fires
    on exactly the same overlay stack as `main.py`. With --ofi-replay a
    micro_engine is supplied and OFI/ML/momentum overlays activate off the
    replayed aggTrade tape; otherwise (no historical L2/tape) they no-op."""
    lc = Settings.from_env()
    return SignalGenerator(
        max_notional_per_trade=cfg.max_notional,
        safety_eps=cfg.safety_eps if cfg.safety_eps is not None else lc.safety_eps,
        cooldown_secs=0.0,                 # replay has no wall-clock gap; we
                                           # dedup per-market ourselves instead
        price_min=lc.price_min,
        price_max=lc.price_max,
        sigma_floor=lc.sigma_floor,
        skip_updown=not cfg.trade_updown,
        min_tte_secs=lc.min_tte_secs,
        max_tte_secs=lc.max_tte_secs,
        book_max_age_secs=lc.book_max_age_secs,
        iv_oracle=None,
        poly_ws=None,
        effective_spread_mult=lc.effective_spread_mult,
        max_walk_slippage=lc.max_walk_slippage,
        longshot_tilt_mult=lc.longshot_tilt_mult,
        skew_coef=lc.skew_coef,
        sell_price_min=lc.sell_price_min,
        micro_engine=micro_engine,
        ml_overlay=lc.ml_overlay,          # no-ops without micro_engine/features
        ml_weight=lc.ml_weight,
        obi_veto=lc.obi_veto,
        obi_veto_threshold=lc.obi_veto_threshold,
        kelly_enabled=lc.kelly_enabled,
        kelly_fraction=lc.kelly_fraction,
        maker_enabled=lc.maker_enabled,
        maker_join_ticks=lc.maker_join_ticks,
        wedge_coeffs=load_wedge_coeffs("wedge_coeffs.json"),
        calib=load_calibration("calib_coeffs.json"),
    )


async def backtest(cfg: BacktestConfig) -> list[SimFill]:
    fills: list[SimFill] = []
    # Diagnostics: where do markets/minutes get dropped? Separates
    # "no gross edge existed" from "edge eaten by costs/overlays/gates".
    diag = Counter()
    micro_engine = None
    if cfg.ofi_replay:
        lc = Settings.from_env()
        micro_engine = MicrostructureEngine(
            model=DirectionalModel.load(),
            obi_veto=lc.obi_veto, obi_veto_threshold=lc.obi_veto_threshold,
            momentum_enabled=lc.momentum_enabled,
            impulse_fade_enabled=lc.impulse_fade_enabled,
            momentum_min_move_usd=lc.momentum_min_move_usd,
            momentum_near_expiry_secs=lc.momentum_window_secs,
        )
    sig_gen = _build_signal_generator(cfg, micro_engine)
    headers = {"User-Agent": "Mozilla/5.0"}
    worker_count = _available_worker_count()
    sem = asyncio.Semaphore(worker_count)
    clob_gate = RateGate(CLOB_CONCURRENCY, CLOB_MIN_INTERVAL_SECS)
    binance_gate = RateGate(BINANCE_CONCURRENCY, BINANCE_MIN_INTERVAL_SECS)
    klines_cache: dict[tuple[str, int, int], asyncio.Task[list[tuple[float, float]]]] = {}
    trades_cache: dict[tuple[str, int, int], asyncio.Task[list[tuple[float, float, float, float]]]] = {}
    sigma_cache: dict[tuple[str, int, int], dict[float, float]] = {}
    replay_cache: dict[tuple[str, int, int], MicroReplay | None] = {}
    async with aiohttp.ClientSession(headers=headers) as session:
        log.info("Fetching resolved markets from last %d day(s)...", cfg.days)
        markets, truncated = await _fetch_resolved_markets(session, cfg.days)
        log.info("Found %d resolved BTC/ETH threshold markets", len(markets))
        if truncated:
            log.warning(
                "Coverage TRUNCATED: hit the %d-page-per-series budget before "
                "reaching the %d-day cutoff — realised window is shorter than "
                "requested.", _MAX_PAGES, cfg.days,
            )
        # The very newest markets (last ~hour) often have no CLOB price history
        # yet — drop them so they don't masquerade as "no edge".
        fresh_cutoff = time.time() - _FRESH_SKIP_SECS
        markets = [m for m in markets if _parse_iso(m.get("endDate")) <= fresh_cutoff]
        markets = _select_markets(markets, cfg.max_markets)
        if markets:
            first = datetime.fromtimestamp(_parse_iso(markets[0].get("endDate")), timezone.utc)
            last = datetime.fromtimestamp(_parse_iso(markets[-1].get("endDate")), timezone.utc)
            log.info(
                "Selected %d markets spread across %s → %s; workers=%d, clob=%dx/%.2fs, binance=%dx/%.2fs",
                len(markets), first.strftime("%Y-%m-%d %H:%M UTC"),
                last.strftime("%Y-%m-%d %H:%M UTC"), worker_count,
                CLOB_CONCURRENCY, CLOB_MIN_INTERVAL_SECS,
                BINANCE_CONCURRENCY, BINANCE_MIN_INTERVAL_SECS,
            )
        diag["markets_seen"] = len(markets)

        async def cached_klines(symbol: str, start_ts: float, expiry: float) -> list[tuple[float, float]]:
            key = (symbol, int(start_ts), int(expiry))
            task = klines_cache.get(key)
            if task is None:
                task = asyncio.create_task(_fetch_binance_klines(session, binance_gate, symbol, start_ts, expiry))
                klines_cache[key] = task
            return await task

        async def cached_replay(symbol: str, start_ts: float, expiry: float) -> MicroReplay | None:
            key = (symbol, int(start_ts), int(expiry))
            if key in replay_cache:
                return replay_cache[key]
            task = trades_cache.get(key)
            if task is None:
                task = asyncio.create_task(_fetch_agg_trades(session, binance_gate, symbol, start_ts, expiry))
                trades_cache[key] = task
            trades = await task
            replay = MicroReplay(trades) if trades else None
            replay_cache[key] = replay
            return replay

        async def process_market(i: int, m: dict) -> list[SimFill]:
            async with sem:
                return await _process_market(
                    i, m, cfg, session, clob_gate, sig_gen, diag, cached_klines,
                    cached_replay, sigma_cache,
                )

        tasks = [asyncio.create_task(process_market(i, m)) for i, m in enumerate(markets)]
        completed = 0
        for task in asyncio.as_completed(tasks):
            fills.extend(await task)
            completed += 1
            if completed % 10 == 0:
                log.info("...processed %d/%d markets, %d sim-fills so far",
                         completed, len(markets), len(fills))

    if cfg.debug or not fills:
        _print_diag(diag)
    return fills


async def _process_market(
    i: int,
    m: dict,
    cfg: BacktestConfig,
    session: aiohttp.ClientSession,
    clob_gate: RateGate,
    sig_gen: SignalGenerator,
    diag: Counter,
    cached_klines,
    cached_replay,
    sigma_cache: dict[tuple[str, int, int], dict[float, float]],
) -> list[SimFill]:
            out: list[SimFill] = []
            q = m.get("question", "")
            is_threshold = bool(THRESHOLD_RE.search(q))
            strike = _parse_strike(q) if is_threshold else 0.0
            symbol = "ETHUSDT" if ETH_RE.search(q) else "BTCUSDT"
            sym_lower = symbol.lower()
            expiry = _parse_iso(m.get("endDate"))
            start_ts = _parse_iso(m.get("startDate")) or (expiry - 4 * 3600)
            if expiry <= 0 or start_ts >= expiry:
                return out

            resolution = _resolution(m)
            if resolution is None:
                return out

            # Modern shape: clobTokenIds is a JSON-encoded list aligned with outcomes.
            raw_ids = m.get("clobTokenIds")
            if not raw_ids:
                return out
            try:
                ids = raw_ids if isinstance(raw_ids, list) else json.loads(raw_ids)
            except Exception:
                return out
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
                return out

            is_updown = not is_threshold
            if is_updown and not cfg.trade_updown:
                diag["mkt_skip_updown"] += 1
                return out

            history = await _fetch_token_history(session, clob_gate, yes_token, start_ts, expiry)
            if history is None:
                diag["mkt_history_fetch_failed"] += 1
                return out
            if len(history) < 5:
                diag["mkt_no_history"] += 1
                return out
            klines = await cached_klines(symbol, start_ts, expiry)
            if len(klines) < 5:
                diag["mkt_no_klines"] += 1
                return out
            window_key = (symbol, int(start_ts), int(expiry))
            sigmas = sigma_cache.get(window_key)
            if sigmas is None:
                sigmas = _rolling_sigma(klines, cfg.sigma_window_secs)
                sigma_cache[window_key] = sigmas
            kline_by_ts = {ts: c for ts, c in klines}
            kline_keys = sorted(kline_by_ts)

            replay = None
            if cfg.ofi_replay:
                replay = await cached_replay(symbol, start_ts, expiry)
                if replay is not None:
                    diag["mkt_with_tape"] += 1

            # For Up/Down markets the strike is the Binance close at start_ts.
            if is_updown:
                idx = _bisect_le(kline_keys, start_ts)
                if idx < 0:
                    return out
                strike = kline_by_ts[kline_keys[idx]]

            diag["mkt_evaluable"] += 1
            mkt_max_raw = 0.0     # best gross edge (no eps/wedge) seen this market
            mkt_fired = False
            fill_rng = random.Random(yes_token)
            for hist_idx, (poly_ts, poly_price) in enumerate(history):
                if poly_price <= 0 or poly_price >= 1:
                    continue
                if cfg.skip_first_secs > 0 and poly_ts < start_ts + cfg.skip_first_secs:
                    diag["min_skip_early"] += 1
                    continue
                if cfg.nofill_tail > 0 and (poly_price < cfg.nofill_tail or poly_price > 1.0 - cfg.nofill_tail):
                    diag["min_tail_nofill"] += 1
                    continue
                if cfg.persistence_cents > 0 and not _price_persisted(history, hist_idx, cfg.persistence_cents):
                    diag["min_not_persistent"] += 1
                    continue
                tte = expiry - poly_ts
                if tte <= 0:
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
                if cfg.short_tte_sigma_floor > 0 and tte < 1800:
                    sigma = max(sigma, cfg.short_tte_sigma_floor)

                diag["min_evaluable"] += 1
                half = cfg.half_spread
                best_bid = max(1e-4, round(poly_price - half, 4))
                best_ask = min(1 - 1e-4, round(poly_price + half, 4))

                # Reference gross edge with NO overlays/eps — just model vs the
                # crossed price and the fee. Tells us whether any edge existed
                # at all, independent of the live haircuts and price gates.
                p_raw = implied_prob(spot, strike, tte, sigma)
                raw_buy = p_raw - best_ask - taker_fee_per_share(best_ask, cfg.fee_rate)
                raw_sell = best_bid - p_raw - taker_fee_per_share(best_bid, cfg.fee_rate)
                raw_edge = max(raw_buy, raw_sell)
                if raw_edge > 0:
                    diag["min_raw_edge_pos"] += 1
                    mkt_max_raw = max(mkt_max_raw, raw_edge)
                if best_ask > sig_gen.price_max or best_bid < sig_gen.price_min:
                    diag["min_price_gated"] += 1

                # Synthesise a top-of-book around the displayed price and drive
                # the *real* signal generator. expiry_ts is offset from the
                # current wall clock so evaluate()'s `expiry_ts - time.time()`
                # reproduces the historical tte; book.ts is fresh so the
                # staleness gate passes.
                now = time.time()
                yes_book = BookSnapshot(
                    token_id=yes_token,
                    best_bid=best_bid, best_ask=best_ask,
                    bid_size=1e6, ask_size=1e6,   # deep enough to fill at top
                    ts=time.monotonic(),
                )
                tick = BinanceTick(
                    symbol=sym_lower, bid=spot, ask=spot, mid=spot,
                    sigma_annual=sigma, ts=time.monotonic(), drift_annual=0.0,
                )
                pm = PolyMarket(
                    condition_id=m.get("conditionId", ""),
                    question=q,
                    yes_token_id=yes_token, no_token_id="",
                    yes_price=poly_price, no_price=1.0 - poly_price,
                    strike=strike, expiry_ts=now + tte,
                    tick_size=0.01, symbol=sym_lower,
                    is_updown=is_updown, is_threshold=is_threshold,
                    fee_rate=cfg.fee_rate, fee_exponent=1.0,
                )

                feats = replay.features(poly_ts, spot, sigma, tte) if replay is not None else None
                sig = sig_gen.evaluate(pm, tick, yes_book, features=feats,
                                       carry_annual=cfg.carry_annual)
                if sig is None:
                    continue
                # Honest maker-fill model: a resting post-only order is only
                # sometimes hit. <1 probability skips the fill and tries the
                # next minute (the order rests until taken or the window moves).
                if sig.is_maker and fill_rng.random() > cfg.maker_fill_prob:
                    diag["min_maker_unfilled"] += 1
                    continue
                diag["min_fired"] += 1
                mkt_fired = True

                # Realised PnL at resolution. Maker fills earn the rebate;
                # taker fills pay the parabolic fee — mirror the live edge.
                size = sig.size
                if sig.is_maker:
                    cost = -maker_rebate_per_share(sig.price)   # negative = credit
                else:
                    cost = taker_fee_per_share(sig.price, cfg.fee_rate, 1.0)
                if sig.side == Side.BUY:
                    pnl = (resolution - sig.price) * size - cost * size
                else:
                    pnl = (sig.price - resolution) * size - cost * size

                out.append(SimFill(
                    market_question=q,
                    symbol=sym_lower,
                    ts=poly_ts,
                    side=sig.side.value,
                    price=sig.price,
                    size=size,
                    p_star=sig.p_star,
                    edge=sig.edge,
                    fee=cost * size,
                    resolution=resolution,
                    pnl=pnl,
                    is_maker=sig.is_maker,
                    expiry_ts=expiry,
                ))
                if cfg.dump_fills:
                    print(
                        "DUMP_FILL "
                        f"side={sig.side.value} px={sig.price:.4f} poly={poly_price:.4f} "
                        f"spot={spot:.2f} strike={strike:.2f} tte={tte:.1f} "
                        f"sigma={sigma:.4f} p_raw={p_raw:.4f} p_star={sig.p_star:.4f} "
                        f"res={resolution:.0f} q={q[:70]}"
                    )

                # One independent sample per market unless --all-fills.
                if not cfg.all_fills:
                    break

            if mkt_max_raw > 0:
                diag["mkt_with_raw_edge"] += 1
            if mkt_fired:
                diag["mkt_fired"] += 1
            elif mkt_max_raw > 0:
                # A gross edge existed but the live haircuts/gates rejected it.
                diag["mkt_edge_eaten"] += 1

            return out


def _print_diag(diag: "Counter") -> None:
    print("\n--- Diagnostics (where candidates were dropped) ---")
    print(f"  markets seen:            {diag['markets_seen']}")
    print(f"    skipped Up/Down:       {diag['mkt_skip_updown']}")
    print(f"    history fetch failed:  {diag['mkt_history_fetch_failed']}  "
          f"(rate-limit/network; retried)")
    print(f"    no Polymarket history: {diag['mkt_no_history']}")
    print(f"    no Binance klines:     {diag['mkt_no_klines']}  "
          f"(HTTP 451 if geo-blocked)")
    print(f"    evaluable:             {diag['mkt_evaluable']}")
    print(f"      had a gross edge:    {diag['mkt_with_raw_edge']}")
    print(f"      fired a fill:        {diag['mkt_fired']}")
    print(f"      edge eaten by costs/overlays/gates: {diag['mkt_edge_eaten']}")
    print(f"  minutes evaluable:       {diag['min_evaluable']}")
    print(f"    skipped early window:  {diag['min_skip_early']}")
    print(f"    tail non-fillable:     {diag['min_tail_nofill']}")
    print(f"    not price-persistent:  {diag['min_not_persistent']}")
    print(f"    raw gross edge > 0:    {diag['min_raw_edge_pos']}")
    print(f"    price-gated (ask>max or bid<min): {diag['min_price_gated']}")
    print(f"    actually fired:        {diag['min_fired']}")
    if diag["mkt_evaluable"] and not diag["mkt_fired"]:
        if diag["mkt_with_raw_edge"]:
            print("  → Verdict: gross edge existed but live costs/overlays/price"
                  " gates removed it (this is the realistic correction).")
        else:
            print("  → Verdict: no gross edge in the window even before costs.")


def _available_worker_count(reserve: int = 2) -> int:
    """Workers from CPU actually available/headroom, reserving cores for the OS.

    Linux exposes process affinity; macOS does not, so there we estimate idle
    headroom with 1-minute load average instead of blindly using core count.
    """
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            available = len(affinity(0))
            return max(1, available - reserve)
        except OSError:
            pass

    total = os.cpu_count() or 1
    try:
        load_1m = os.getloadavg()[0]
    except OSError:
        load_1m = 0.0
    available = max(1, math.floor(total - load_1m))
    return max(1, available - reserve)


def _price_persisted(history: list[tuple[float, float]], idx: int, max_move: float) -> bool:
    if idx <= 0 or idx >= len(history) - 1:
        return False
    px = history[idx][1]
    prev_px = history[idx - 1][1]
    next_px = history[idx + 1][1]
    return abs(px - prev_px) <= max_move and abs(next_px - px) <= max_move


def _select_markets(markets: list[dict], max_markets: int) -> list[dict]:
    """Deterministically spread capped runs across the full lookback window."""
    ordered = sorted(markets, key=lambda m: _parse_iso(m.get("endDate")))
    if max_markets <= 0 or len(ordered) <= max_markets:
        return ordered

    last = len(ordered) - 1
    chosen = []
    seen: set[int] = set()
    for i in range(max_markets):
        idx = round(i * last / (max_markets - 1))
        if idx not in seen:
            chosen.append(ordered[idx])
            seen.add(idx)
    return chosen


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

    n_maker = sum(1 for f in fills if f.is_maker)
    print(f"\n=== Backtest summary ({n} sim-fills across {len(by_market)} markets) ===")
    print(f"Gross notional:  ${total_notional:,.2f}")
    print(f"Total fees/rebate:${total_fee:+.4f}  (negative = net rebate)")
    print(f"Realized PnL:    ${total_pnl:+,.4f}")
    print(f"Win rate:        {len(wins)/n*100:.1f}% ({len(wins)}W / {len(losses)}L)  [per-fill]")
    print(f"Maker / taker:   {n_maker} maker / {n - n_maker} taker fills")
    print(f"Mean edge fired: {sum(f.edge for f in fills)/n*100:+.2f}%")
    if wins:
        print(f"Avg win:         ${sum(f.pnl for f in wins)/len(wins):+.3f}")
    if losses:
        print(f"Avg loss:        ${sum(f.pnl for f in losses)/len(losses):+.3f}")
    print(f"Return on notional: {total_pnl/total_notional*100:+.2f}%" if total_notional else "")

    # --- Per-market view: the statistically honest sample size ---
    mkt_pnls = [sum(f.pnl for f in fs) for fs in by_market.values()]
    m = len(mkt_pnls)
    mkt_wins = sum(1 for p in mkt_pnls if p > 0)
    mean_mkt = sum(mkt_pnls) / m
    var = sum((p - mean_mkt) ** 2 for p in mkt_pnls) / m if m > 1 else 0.0
    stderr = math.sqrt(var / m) if m > 0 else 0.0
    print(f"\n--- Per-market (N={m}) ---")
    print(f"Markets profitable: {mkt_wins}/{m} ({mkt_wins/m*100:.1f}%)")
    print(f"PnL per market:     ${mean_mkt:+.3f} ± ${stderr:.3f} (1 s.e.)")
    print(f"95% CI on mean:     [${mean_mkt - 1.96*stderr:+.3f}, ${mean_mkt + 1.96*stderr:+.3f}]")
    if mean_mkt - 1.96 * stderr <= 0 <= mean_mkt + 1.96 * stderr:
        print("  ⚠ CI straddles 0 — edge is NOT statistically distinguishable from noise.")

    # --- Correlation guard: markets resolving on the same 1h candle and the
    # same underlying are ONE bet, not many. The CI above assumes independence;
    # if the markets cluster into a few resolution windows it is overconfident. ---
    windows = set()
    for fs in by_market.values():
        f0 = fs[0]
        sym = "ETH" if "eth" in f0.symbol or "ethereum" in f0.market_question.lower() else "BTC"
        windows.add((sym, round(f0.expiry_ts / 3600.0)))   # underlying × resolution hour
    k = len(windows)
    print(f"Independent resolution windows (underlying × hour): {k}")
    if k < m:
        print(f"  ⚠ {m} markets collapse to ~{k} independent event(s) — the per-market")
        print(f"    CI is OVERCONFIDENT. Treat this as ≈{k} bet(s), not {m}. A 100% win")
        print(f"    rate over correlated same-hour strikes is one move, not an edge.")

    print("\nCaveats (real PnL is worse than this):")
    print("  - top-of-book fill assumed; no queue/latency model")
    print("  - OFI/ML/OBI microstructure overlays inactive (no historical L2/tape)")
    print("  - funding carry & Deribit-IV σ off (oracles not replayed)")

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
    parser.add_argument("--max-markets", type=int, default=300,
                        help="markets to simulate, sampled evenly across the window")
    parser.add_argument("--max-notional", type=float, default=25.0)
    parser.add_argument("--safety-eps", type=float, default=None,
                        help="override edge cushion (default: live Config value)")
    parser.add_argument("--half-spread", type=float, default=0.01,
                        help="synthesised half-spread each side of displayed price")
    parser.add_argument("--all-fills", action="store_true",
                        help="record every qualifying minute, not one entry per market")
    parser.add_argument("--trade-updown", action="store_true",
                        help="include Up/Down markets (synthetic strike; off by default)")
    parser.add_argument("--carry-annual", type=float, default=0.0,
                        help="perp-funding carry fed to the pricer (annualised)")
    parser.add_argument("--maker-fill-prob", type=float, default=1.0,
                        help="P(resting maker order is filled); <1 is the honest setting")
    parser.add_argument("--ofi-replay", action="store_true",
                        help="replay Binance aggTrades to activate the OFI/ML/momentum overlay")
    parser.add_argument("--dump-fills", action="store_true",
                        help="print per-fill internals for diagnosing backtest artifacts")
    parser.add_argument("--skip-first-secs", type=float, default=0.0,
                        help="skip the first N seconds after market start")
    parser.add_argument("--nofill-tail", type=float, default=0.0,
                        help="treat prices below x or above 1-x as non-fillable when >0")
    parser.add_argument("--persistence-cents", type=float, default=0.0,
                        help="require prev/current/next history prints within this price distance")
    parser.add_argument("--short-tte-sigma-floor", type=float, default=0.0,
                        help="extra annual sigma floor when tte < 30m")
    parser.add_argument("--debug", action="store_true",
                        help="always print the drop-reason diagnostics")
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
        half_spread=args.half_spread,
        all_fills=args.all_fills,
        trade_updown=args.trade_updown,
        carry_annual=args.carry_annual,
        maker_fill_prob=args.maker_fill_prob,
        ofi_replay=args.ofi_replay,
        dump_fills=args.dump_fills,
        skip_first_secs=args.skip_first_secs,
        nofill_tail=args.nofill_tail,
        persistence_cents=args.persistence_cents,
        short_tte_sigma_floor=args.short_tte_sigma_floor,
        debug=args.debug,
    )
    fills = asyncio.run(backtest(cfg))
    _summarize(fills)


if __name__ == "__main__":
    cli()
