"""
Polymarket market universe — fetches active BTC/ETH binary markets from
the Gamma API and parses strike + expiry for the pricing model.

Two market families are tradeable:

  1. **Threshold markets** (the bulk of crypto volume today):
       "Bitcoin above 77,800 on May 8, 7AM ET?"
       "Ethereum above 2,345 on May 8, 6AM ET?"
     YES = S_T > K, NO = S_T <= K. Strike K is parsed from the question.

  2. **Up/Down windows** (5-min and hourly):
       "Bitcoin Up or Down - May 9, 1:55AM-2:00AM ET"
     Strike K is the spot at window open; we anchor it to live Binance
     mid the first time we see the market (handled in main.py).

Token IDs and outcome prices come from `clobTokenIds` (a JSON-encoded
string list) + `outcomes` + `outcomePrices`. The legacy `tokens` array
is now usually `null`. We require `enableOrderBook=true` and
`acceptingOrders=true` to skip pre-listed markets that have no
tradeable book yet.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/markets"

# Match BTC/ETH Up/Down or "above $K" threshold markets.
THRESHOLD_RE = re.compile(
    r"(?:bitcoin|btc|ethereum|eth)\s+(?:above|reach(?:es)?|over|>=?)",
    re.I,
)
UPDOWN_RE = re.compile(
    r"(?:bitcoin|btc|ethereum|eth)\s+(?:up or down|up-or-down)",
    re.I,
)
BTC_RE = re.compile(r"(?:bitcoin|btc)", re.I)
ETH_RE = re.compile(r"(?:ethereum|eth)", re.I)

# Match a numeric strike. Accepts "$94,500", "94,500", "94500.50".
PRICE_RE = re.compile(r"\$?([\d]{1,3}(?:,\d{3})+(?:\.\d+)?|\d{3,}(?:\.\d+)?)")


@dataclass
class PolyMarket:
    condition_id: str
    question: str
    yes_token_id: str   # YES / Up token
    no_token_id: str    # NO / Down token
    yes_price: float    # current mid on Polymarket
    no_price: float
    strike: float       # reference price (parsed or anchored to spot)
    expiry_ts: float    # UTC unix timestamp
    tick_size: float    # from API; default 0.01
    symbol: str = "btcusdt"  # Binance feed to use
    # Cached structural flags computed once at universe-load time so the
    # signal generator doesn't have to re-derive them every tick.
    is_updown: bool = False
    is_threshold: bool = False


def _parse_strike(question: str) -> float:
    """Extract a numeric strike from question text.

    "Bitcoin above 77,800 on May 8" -> 77800.0
    "BTC reach $94,500 by Friday"    -> 94500.0
    """
    matches = PRICE_RE.findall(question)
    if not matches:
        return 0.0
    # Pick the largest numeric match — avoids picking up a date like "5"
    # when the strike is "77,800". Strikes are always the dominant number.
    candidates = []
    for m in matches:
        try:
            candidates.append(float(m.replace(",", "")))
        except ValueError:
            continue
    if not candidates:
        return 0.0
    return max(candidates)


def _parse_expiry(end_date) -> float:
    """Parse ISO8601 endDate to UTC unix timestamp."""
    if not end_date:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(end_date).rstrip("Z")).replace(
            tzinfo=timezone.utc
        )
        return dt.timestamp()
    except Exception:
        return 0.0


def _decode_tokens(market: dict) -> tuple[str, str, float, float]:
    """Return (yes_token_id, no_token_id, yes_price, no_price).

    Reads `clobTokenIds` (JSON-encoded string list) aligned by index with
    `outcomes` (["Yes","No"] or ["Up","Down"]) and `outcomePrices`
    (["0.52","0.48"]).

    Falls back to the legacy `tokens` array shape if present.
    """
    # --- Modern shape (May 2025+) ------------------------------------
    raw_ids = market.get("clobTokenIds")
    if raw_ids:
        try:
            ids = raw_ids if isinstance(raw_ids, list) else json.loads(raw_ids)
        except (json.JSONDecodeError, TypeError):
            ids = []
        outcomes = market.get("outcomes") or []
        prices = market.get("outcomePrices") or []
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                outcomes = []
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                prices = []
        if len(ids) >= 2 and len(outcomes) >= 2:
            yes_id = no_id = ""
            yes_price = no_price = 0.5
            for i, oc in enumerate(outcomes[:2]):
                low = str(oc).lower()
                tok = str(ids[i]) if i < len(ids) else ""
                try:
                    px = float(prices[i]) if i < len(prices) else 0.5
                except (ValueError, TypeError):
                    px = 0.5
                if low in ("yes", "up", "higher", "above"):
                    yes_id, yes_price = tok, px
                elif low in ("no", "down", "lower", "below"):
                    no_id, no_price = tok, px
            if yes_id and no_id:
                return yes_id, no_id, yes_price, no_price

    # --- Legacy shape ------------------------------------------------
    tokens = market.get("tokens") or []
    yes_id = no_id = ""
    yes_price = no_price = 0.5
    for t in tokens:
        outcome = (t.get("outcome") or "").lower()
        tok = str(t.get("token_id", ""))
        try:
            px = float(t.get("price", 0.5))
        except (ValueError, TypeError):
            px = 0.5
        if outcome in ("yes", "up", "higher", "above"):
            yes_id, yes_price = tok, px
        elif outcome in ("no", "down", "lower", "below"):
            no_id, no_price = tok, px
    return yes_id, no_id, yes_price, no_price


async def fetch_active_markets(
    session: aiohttp.ClientSession,
    min_time_to_expiry_secs: float = 60.0,
    max_time_to_expiry_secs: float = 86400.0,
) -> list[PolyMarket]:
    """Return active BTC/ETH binary markets expiring within the window."""
    headers = {"User-Agent": "Mozilla/5.0"}
    result: list[PolyMarket] = []
    offset = 0
    pages_fetched = 0
    seen_btc_eth = 0

    while offset < 5000:
        now_ts = time.time()
        # start_date_min 4h ago — skips the large backlog of
        # expired-but-unsettled markets while keeping live windows.
        start_min = datetime.fromtimestamp(now_ts - 4 * 3600, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        url = (
            f"{GAMMA_API}?active=true&closed=false&limit=500"
            f"&start_date_min={start_min}&offset={offset}&order=endDate&ascending=true"
        )
        try:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                batch: list[dict] = await resp.json(content_type=None)
        except Exception as exc:
            log.error("Gamma API fetch failed at offset %d: %s", offset, exc)
            break

        if not batch:
            break
        pages_fetched += 1

        now = time.time()
        for m in batch:
            question = m.get("question", "")
            is_threshold = bool(THRESHOLD_RE.search(question))
            is_updown = bool(UPDOWN_RE.search(question))
            if not (is_threshold or is_updown):
                continue
            seen_btc_eth += 1

            expiry_ts = _parse_expiry(m.get("endDate"))
            if not expiry_ts:
                continue
            tte = expiry_ts - now
            if not (min_time_to_expiry_secs <= tte <= max_time_to_expiry_secs):
                continue

            # Only markets with a tradeable book.
            if not m.get("enableOrderBook") or not m.get("acceptingOrders"):
                continue
            if m.get("closed") or m.get("archived"):
                continue

            yes_id, no_id, yes_price, no_price = _decode_tokens(m)
            if not yes_id or not no_id:
                continue

            strike = _parse_strike(question) if is_threshold else 0.0
            symbol = "ethusdt" if ETH_RE.search(question) else "btcusdt"
            tick = float(m.get("orderPriceMinTickSize") or m.get("minimum_tick_size") or 0.01)
            result.append(
                PolyMarket(
                    condition_id=m.get("conditionId", m.get("id", "")),
                    question=question,
                    yes_token_id=yes_id,
                    no_token_id=no_id,
                    yes_price=yes_price,
                    no_price=no_price,
                    strike=strike,
                    expiry_ts=expiry_ts,
                    tick_size=tick,
                    symbol=symbol,
                    is_updown=is_updown,
                    is_threshold=is_threshold,
                )
            )

        # Stop once the page's last item is past our max window.
        last_expiry = _parse_expiry(batch[-1].get("endDate", ""))
        if last_expiry and (last_expiry - now) > max_time_to_expiry_secs:
            break

        if len(batch) < 500:
            break  # final page

        offset += len(batch)

    if result:
        log.info(
            "Universe: %d tradeable BTC/ETH markets (%d page(s), %d BTC/ETH seen)",
            len(result),
            pages_fetched,
            seen_btc_eth,
        )
    else:
        log.info(
            "Universe: 0 tradeable markets (%d page(s), %d BTC/ETH seen) — "
            "pre-listed but not yet accepting orders, or outside window",
            pages_fetched,
            seen_btc_eth,
        )
    return result
