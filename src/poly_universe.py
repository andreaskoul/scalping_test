"""
Polymarket market universe — fetches active BTC hourly Up/Down markets
from the Gamma API and parses strike + expiry for the pricing model.

Hourly "Up or Down" markets encode the reference price at market open
as the strike K and have a fixed expiry timestamp.

Market structure returned by Gamma:
  {
    "id":           "...",
    "question":     "Will Bitcoin's price go up or down from 2:00 PM to 3:00 PM ET?",
    "conditionId":  "0x...",
    "endDate":      "2025-04-10T19:00:00Z",
    "active":       true,
    "closed":       false,
    "tokens": [
      {"token_id": "...", "outcome": "Up",   "price": 0.52},
      {"token_id": "...", "outcome": "Down", "price": 0.48},
    ],
    "description": "...",   # often encodes reference price in text
  }
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

log = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/markets"
# Match all BTC/Bitcoin Up/Down formats: "BTC Up or Down 5m", "Bitcoin Up or Down - ...", etc.
SLUG_RE = re.compile(r"(?:bitcoin|btc|ethereum|eth)\s+(?:up or down|up-or-down)", re.I)
BTC_RE = re.compile(r"(?:bitcoin|btc)", re.I)
ETH_RE = re.compile(r"(?:ethereum|eth)", re.I)
# Extract a dollar price from question text like "$94,500"
PRICE_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)")


@dataclass
class PolyMarket:
    condition_id: str
    question: str
    yes_token_id: str   # "Up" token
    no_token_id: str    # "Down" token
    yes_price: float    # current mid on Polymarket
    no_price: float
    strike: float       # reference price (parsed from question text)
    expiry_ts: float    # UTC unix timestamp
    tick_size: float    # from API; default 0.01
    symbol: str = "btcusdt"  # Binance feed to use


def _parse_strike(question: str) -> float:
    """Extract a numeric dollar reference price from question text."""
    matches = PRICE_RE.findall(question)
    if not matches:
        return 0.0
    # Take the last match (usually the reference price, not a bound)
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return 0.0


def _parse_expiry(end_date: str) -> float:
    """Parse ISO8601 endDate to UTC unix timestamp."""
    try:
        dt = datetime.fromisoformat(end_date.rstrip("Z")).replace(
            tzinfo=timezone.utc
        )
        return dt.timestamp()
    except Exception:
        return 0.0


def _token_ids(tokens: list[dict]) -> tuple[str, str, float, float]:
    """Return (yes_token_id, no_token_id, yes_price, no_price).

    "Up" is treated as the YES (long) leg.
    """
    yes_id = no_id = ""
    yes_price = no_price = 0.5
    for t in tokens:
        outcome = (t.get("outcome") or "").lower()
        if outcome in ("up", "yes", "higher"):
            yes_id = str(t["token_id"])
            yes_price = float(t.get("price", 0.5))
        elif outcome in ("down", "no", "lower"):
            no_id = str(t["token_id"])
            no_price = float(t.get("price", 0.5))
    return yes_id, no_id, yes_price, no_price


async def fetch_active_markets(
    session: aiohttp.ClientSession,
    min_time_to_expiry_secs: float = 60.0,
    max_time_to_expiry_secs: float = 86400.0,
) -> list[PolyMarket]:
    """Return active BTC Up/Down markets expiring within the window.

    Paginates through the Gamma API (sorted ascending by endDate) because
    many already-expired-but-unsettled markets sit at the front of the list
    and push live markets past the first page.  Stops as soon as the last
    item in a page expires beyond max_time_to_expiry_secs.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    result: list[PolyMarket] = []
    offset = 0
    pages_fetched = 0

    while offset < 5000:
        now_ts = time.time()
        # Use start_date_min 4 hours ago — skips the large backlog of
        # expired-but-unsettled markets while keeping all live windows.
        start_min = datetime.fromtimestamp(now_ts - 4 * 3600, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        url = (
            f"{GAMMA_API}?active=true&closed=false&limit=500"
            f"&start_date_min={start_min}&offset={offset}&order=endDate&ascending=true"
        )
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                batch: list[dict] = await resp.json(content_type=None)
        except Exception as exc:
            log.error("Gamma API fetch failed at offset %d: %s", offset, exc)
            break

        if not batch:
            break
        pages_fetched += 1

        now = time.time()
        if pages_fetched == 1:
            sample = [m.get("question", "")[:80] for m in batch[:5]]
            log.info("Gamma page 1 sample: %s", sample)
            crypto_qs = [m.get("question", "") for m in batch
                         if any(k in m.get("question", "").lower()
                                for k in ("bitcoin", "btc", "ethereum", "eth"))]
            log.info("Crypto questions in page 1: %d — %s",
                     len(crypto_qs), [q[:60] for q in crypto_qs[:5]])

        for m in batch:
            question = m.get("question", "")
            if not SLUG_RE.search(question):
                continue
            expiry_ts = _parse_expiry(m.get("endDate", ""))
            if not expiry_ts:
                continue
            tte = expiry_ts - now
            if not (min_time_to_expiry_secs <= tte <= max_time_to_expiry_secs):
                continue
            tokens = m.get("tokens") or []
            yes_id, no_id, yes_price, no_price = _token_ids(tokens)
            if not yes_id or not no_id:
                continue
            strike = _parse_strike(question)
            symbol = "ethusdt" if ETH_RE.search(question) else "btcusdt"
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
                    tick_size=float(m.get("minimum_tick_size") or 0.01),
                    symbol=symbol,
                )
            )

        # Once the last item in the page is past our max window, all
        # subsequent pages will be too — stop paginating.
        last_expiry = _parse_expiry(batch[-1].get("endDate", ""))
        if last_expiry and (last_expiry - now) > max_time_to_expiry_secs:
            break

        if len(batch) < 500:
            break  # final page

        offset += len(batch)

    log.info("Universe: %d tradeable BTC Up/Down markets (%d page(s))", len(result), pages_fetched)
    return result
