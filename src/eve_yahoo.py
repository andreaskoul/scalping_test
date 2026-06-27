"""Yahoo Finance historical bar ingest for Eve (free; forex + more).

Alpaca FX data is a paid add-on (the API returns ``403 not authorized for FX
data`` on the free plan), so Yahoo Finance is Eve's free source for **forex**
(and a fallback for equities/indices/crypto). It is historical-batch only, like
the Alpaca path, and writes into the same Eve lake in the same normalized shape.

Yahoo's public chart endpoint needs no auth or crumb — only a browser-like
``User-Agent``:

    GET https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
        ?period1={epoch}&period2={epoch}&interval={1d|1h|5m|1m}

Response: ``chart.result[0].timestamp[]`` (epoch seconds) aligned with
``chart.result[0].indicators.quote[0].{open,high,low,close,volume}``; gaps are
padded with nulls (skipped). Forex symbols look like ``EURUSD=X``. Intraday
history is range-limited by Yahoo (≈730 days for 1h, ≈30 for 1m); daily goes
back decades.

Reuses the Alpaca path's :class:`ResponseCache`, urllib transport, lake
partition helpers, and :func:`normalize_alpaca_bars` (with ``provider="yahoo"``)
so the same two idempotency layers apply: a lake partition skip and a persistent
per-request response cache. Transport + sleep are injectable for credential-free
tests.
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from .eve_data import EveLake, normalize_alpaca_bars, write_jsonl_partition
from .eve_ingest import (
    AlpacaIngestError,
    IngestResult,
    ResponseCache,
    Transport,
    _as_date,
    _daterange,
    _partition_done,
    _write_empty_partition,
    alpaca_bar_to_norm_row,
    bars_dataset,
    urllib_transport,
)

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"
DEFAULT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


class YahooBarClient:
    """Cached, rate-limited client for Yahoo Finance chart bars."""

    def __init__(
        self,
        cache: ResponseCache,
        *,
        transport: Transport | None = None,
        user_agent: str = DEFAULT_UA,
        max_per_min: float = 120.0,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache = cache
        self.transport = transport or urllib_transport
        self.headers = {"User-Agent": user_agent}
        self.min_interval = 60.0 / max_per_min if max_per_min > 0 else 0.0
        self.max_retries = int(max_retries)
        self._sleep = sleep
        self._clock = clock
        self._last_call = 0.0
        self.api_calls = 0

    def _rate_limit(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (self._clock() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = self._clock()

    def _request(self, url: str, params: dict) -> dict:
        key = ResponseCache.request_key(url, params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        attempt = 0
        while True:
            self._rate_limit()
            status, payload, _hdrs = self.transport(url, params, self.headers)
            self.api_calls += 1
            if status == 200 and isinstance(payload, dict):
                self.cache.put(key, url, params, payload)
                return payload
            retryable = status == 429 or 500 <= status < 600
            if not retryable or attempt >= self.max_retries:
                raise AlpacaIngestError(f"GET {url} -> HTTP {status}")
            self._sleep(min(30.0, 2.0 ** attempt))
            attempt += 1

    def iter_raw_bars(
        self, symbol: str, interval: str, start: str, end: str
    ) -> Iterator[dict]:
        """Yield compact bar dicts (symbol + t/o/h/l/c/v) for one chart range."""
        p1 = _epoch(start)
        p2 = _epoch(end) + 86_400  # make end-day inclusive
        url = YAHOO_CHART + urllib.parse.quote(symbol)
        params = {"period1": p1, "period2": p2, "interval": interval}
        payload = self._request(url, params)
        chart = (payload or {}).get("chart") or {}
        if chart.get("error"):
            raise AlpacaIngestError(f"yahoo error for {symbol}: {chart['error']}")
        results = chart.get("result") or []
        if not results:
            return
        result = results[0]
        stamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        opens, highs = quote.get("open") or [], quote.get("high") or []
        lows, closes = quote.get("low") or [], quote.get("close") or []
        vols = quote.get("volume") or []
        for i, ts in enumerate(stamps):
            o = _at(opens, i)
            h = _at(highs, i)
            lo = _at(lows, i)
            c = _at(closes, i)
            if None in (o, h, lo, c):  # Yahoo pads gaps with null
                continue
            yield {
                "symbol": symbol,
                "t": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "o": o, "h": h, "l": lo, "c": c,
                "v": _at(vols, i) or 0.0,
            }


def ingest_yahoo_bars(
    client: YahooBarClient,
    lake: EveLake,
    *,
    symbol: str,
    interval: str,
    start: date | str,
    end: date | str,
    asset_class: str = "forex",
    force: bool = False,
) -> IngestResult:
    """Download one Yahoo chart range into the lake as per-day partitions.

    Yahoo returns the whole range in one (cached) call; we then write per-day
    partitions so the lake layout and `read_symbol_bars` match the Alpaca path.
    Days already on disk are skipped unless ``force``.
    """
    start_d = _as_date(start)
    end_d = _as_date(end)
    if end_d < start_d:
        raise ValueError("end must be on or after start")

    days = list(_daterange(start_d, end_d))
    dataset = bars_dataset(interval)
    calls0 = client.api_calls
    hits0 = client.cache.stats.hits

    # Fast path: if every day already exists and not forcing, make no request.
    if not force and all(
        _partition_done(lake, symbol, asset_class, d, dataset, provider="yahoo") for d in days
    ):
        return IngestResult(
            symbol=symbol, timeframe=interval, asset_class=asset_class,
            days_requested=len(days), days_ingested=0, days_skipped=len(days),
            rows=0, api_calls=0, cache_hits=0, partitions=[],
        )

    raw = list(client.iter_raw_bars(symbol, interval, start_d.isoformat(), end_d.isoformat()))
    by_day: dict[str, list[dict]] = {}
    for bar in raw:
        by_day.setdefault(bar["t"][:10], []).append(bar)

    ingested = skipped = rows = 0
    partitions: list[str] = []
    for day in days:
        if not force and _partition_done(lake, symbol, asset_class, day, dataset, provider="yahoo"):
            skipped += 1
            continue
        day_bars = by_day.get(day.isoformat(), [])
        if not day_bars:
            _write_empty_partition(lake, symbol, asset_class, day, dataset, provider="yahoo")
            ingested += 1
            continue
        bars = normalize_alpaca_bars(
            (alpaca_bar_to_norm_row(b) for b in day_bars),
            asset_class=asset_class, feed="yahoo",
            timeframe=interval, provider="yahoo",
        )
        manifest = write_jsonl_partition(bars, lake, partition_date=day, dataset=dataset)
        partitions.append(manifest.path)
        rows += manifest.rows
        ingested += 1

    return IngestResult(
        symbol=symbol, timeframe=interval, asset_class=asset_class,
        days_requested=len(days), days_ingested=ingested, days_skipped=skipped,
        rows=rows, api_calls=client.api_calls - calls0,
        cache_hits=client.cache.stats.hits - hits0, partitions=partitions,
    )


def _epoch(day: str) -> int:
    d = datetime.fromisoformat(str(day).replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp())


def _at(seq: list, i: int):
    return seq[i] if i < len(seq) else None


def cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Eve: cached Yahoo Finance historical bar ingest")
    ap.add_argument("--symbols", required=True, help="comma-separated, e.g. 'EURUSD=X,GBPUSD=X'")
    ap.add_argument("--asset-class", default="forex")
    ap.add_argument("--interval", default="1h", help="1d, 1h, 5m, 1m (intraday is range-limited)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--lake", default=None)
    ap.add_argument("--max-per-min", type=float, default=120.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    lake = EveLake(Path(args.lake)) if args.lake else EveLake.from_env()
    cache = ResponseCache(lake.root / "raw" / "yahoo" / "_apicache")
    client = YahooBarClient(cache, max_per_min=args.max_per_min)
    for symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        res = ingest_yahoo_bars(
            client, lake, symbol=symbol, interval=args.interval,
            start=args.start, end=args.end, asset_class=args.asset_class, force=args.force,
        )
        print(res.summary())
    print(f"\nTotal: {client.api_calls} network calls, {cache.stats.hits} cache hits.")


if __name__ == "__main__":
    cli()
