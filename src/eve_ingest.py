"""Alpaca historical bar ingest for Eve (batch-only, cached).

Alpaca is a *historical* data source for Eve, never a live feed or execution
venue (see ``docs/EVE_TRANSFORMER_RESEARCH.md``). This module downloads
historical bars from Alpaca's Market Data API and writes them into the Eve lake
in the same normalized shape :mod:`src.eve_data` already understands.

Two idempotency layers so we never re-run the same API call (a request from the
user), and so a re-ingest is cheap:

1. **Lake partition check (coarse).** Before requesting a day we check whether
   its lake partition already exists; if so we skip it entirely — no request is
   even built. This is the normal "already have it" fast path.

2. **Response cache (fine).** Every HTTP GET we *do* make is keyed by its
   canonical URL+params (auth headers excluded) and persisted to disk. A repeat
   of the exact request — including a re-download with ``force=True``, or the
   same page during pagination — is served from disk with no network call.

Endpoints (Market Data API, host ``data.alpaca.markets`` — distinct from the
trading/paper host):
  - stocks: ``/v2/stocks/bars``           (feed defaults to ``iex`` on free)
  - crypto: ``/v1beta3/crypto/{loc}/bars`` (loc defaults to ``us``)
Pagination follows ``next_page_token`` -> ``page_token`` until null. Bars use
compact keys ``t,o,h,l,c,v,n,vw`` and are mapped onto the normalizer's schema.

The HTTP transport and the sleep function are injectable, so tests run with a
fake transport and zero real latency, and require no Alpaca credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .eve_data import AlpacaBar, EveLake, LakeManifest, normalize_alpaca_bars, write_jsonl_partition

DATA_HOST = "https://data.alpaca.markets"
STOCK_BARS_PATH = "/v2/stocks/bars"
CRYPTO_BARS_TPL = "/v1beta3/crypto/{loc}/bars"
KEY_HEADER = "APCA-API-KEY-ID"
SECRET_HEADER = "APCA-API-SECRET-KEY"

# transport(url, params, headers) -> (status_code, json_payload, response_headers)
Transport = Callable[[str, dict[str, Any], dict[str, str]], "tuple[int, dict, dict]"]


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AlpacaCreds:
    key: str
    secret: str
    data_host: str = DATA_HOST
    crypto_loc: str = "us"

    @classmethod
    def from_env(cls) -> "AlpacaCreds | None":
        """Load creds from env, or None if absent (offline/test path).

        ``ALPACA_ENDPOINT`` in .env is the *trading* host; market data lives on
        a separate host, overridable via ``ALPACA_DATA_ENDPOINT``.
        """
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            return None
        return cls(
            key=key,
            secret=secret,
            data_host=os.getenv("ALPACA_DATA_ENDPOINT", DATA_HOST).rstrip("/"),
            crypto_loc=os.getenv("ALPACA_CRYPTO_LOC", "us"),
        )

    def headers(self) -> dict[str, str]:
        return {KEY_HEADER: self.key, SECRET_HEADER: self.secret}


# --------------------------------------------------------------------------- #
# Response cache
# --------------------------------------------------------------------------- #
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0


class ResponseCache:
    """Disk-backed cache of raw API responses, keyed by canonical request.

    The key excludes auth headers so it survives key rotation, and includes
    every query param (so distinct pages/ranges are distinct entries). Stored as
    JSON under a two-char shard dir to avoid huge flat directories.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.stats = CacheStats()

    @staticmethod
    def request_key(url: str, params: dict[str, Any]) -> str:
        canon = url + "?" + urllib.parse.urlencode(sorted((str(k), str(v)) for k, v in params.items()))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            self.stats.misses += 1
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return payload.get("response")

    def put(self, key: str, url: str, params: dict[str, Any], response: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "url": url,
            "params": {k: v for k, v in params.items()},
            "fetched_ts": time.time(),
            "response": response,
        }
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
        self.stats.writes += 1


# --------------------------------------------------------------------------- #
# HTTP transport (real)
# --------------------------------------------------------------------------- #
def urllib_transport(url: str, params: dict[str, Any], headers: dict[str, str], *, timeout: float = 30.0):
    """Default real transport over urllib. Returns (status, json, headers)."""
    query = urllib.parse.urlencode(params)
    full = f"{url}?{query}" if query else url
    req = urllib.request.Request(full, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}, dict(resp.headers)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"message": raw}
        return exc.code, payload, dict(exc.headers or {})


# --------------------------------------------------------------------------- #
# Bar client
# --------------------------------------------------------------------------- #
class AlpacaBarClient:
    """Cached, rate-limited, paginating client for Alpaca historical bars."""

    def __init__(
        self,
        creds: AlpacaCreds,
        cache: ResponseCache,
        *,
        transport: Transport | None = None,
        max_per_min: float = 180.0,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.creds = creds
        self.cache = cache
        self.transport = transport or urllib_transport
        self.min_interval = 60.0 / max_per_min if max_per_min > 0 else 0.0
        self.max_retries = int(max_retries)
        self._sleep = sleep
        self._clock = clock
        self._last_call = 0.0
        self.api_calls = 0  # genuine network calls (cache misses)

    def _rate_limit(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (self._clock() - self._last_call)
        if wait > 0:
            self._sleep(wait)
        self._last_call = self._clock()

    def _request(self, url: str, params: dict[str, Any]) -> dict:
        """One GET with caching, rate-limiting, and 429/5xx backoff."""
        key = ResponseCache.request_key(url, params)
        cached = self.cache.get(key)
        if cached is not None:
            return cached  # no network, no rate-limit wait

        attempt = 0
        while True:
            self._rate_limit()
            status, payload, resp_headers = self.transport(url, params, self.creds.headers())
            self.api_calls += 1
            if status == 200:
                self.cache.put(key, url, params, payload)
                return payload
            retryable = status == 429 or 500 <= status < 600
            if not retryable or attempt >= self.max_retries:
                msg = payload.get("message") if isinstance(payload, dict) else payload
                raise AlpacaIngestError(f"GET {url} -> HTTP {status}: {msg}")
            backoff = self._retry_after(resp_headers, attempt)
            self._sleep(backoff)
            attempt += 1

    @staticmethod
    def _retry_after(headers: dict[str, str], attempt: int) -> float:
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra:
            try:
                return max(0.0, float(ra))
            except ValueError:
                pass
        return min(30.0, (2.0 ** attempt))  # exponential, capped

    def _bars_url_params(
        self, symbol: str, timeframe: str, start: str, end: str, *, asset_class: str,
        limit: int, feed: str | None, page_token: str | None,
    ) -> "tuple[str, dict]":
        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": limit,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        if asset_class == "crypto":
            url = self.creds.data_host + CRYPTO_BARS_TPL.format(loc=self.creds.crypto_loc)
        else:
            url = self.creds.data_host + STOCK_BARS_PATH
            params["feed"] = feed or "iex"  # free stock data feed
        return url, params

    def iter_raw_bars(
        self, symbol: str, timeframe: str, start: str, end: str, *,
        asset_class: str, limit: int = 10000, feed: str | None = None,
    ) -> Iterator[dict]:
        """Yield raw Alpaca bar dicts (with symbol attached) across all pages."""
        page_token: str | None = None
        while True:
            url, params = self._bars_url_params(
                symbol, timeframe, start, end,
                asset_class=asset_class, limit=limit, feed=feed, page_token=page_token,
            )
            payload = self._request(url, params)
            bars_by_symbol = (payload or {}).get("bars") or {}
            for bar in bars_by_symbol.get(symbol, []) or []:
                row = dict(bar)
                row["symbol"] = symbol
                yield row
            page_token = (payload or {}).get("next_page_token")
            if not page_token:
                return


class AlpacaIngestError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Normalization + ingest orchestration
# --------------------------------------------------------------------------- #
def alpaca_bar_to_norm_row(bar: dict) -> dict[str, Any]:
    """Map a compact Alpaca bar (t/o/h/l/c/v/n/vw + symbol) to normalizer input."""
    return {
        "symbol": bar.get("symbol"),
        "timestamp": bar.get("t"),
        "open": bar.get("o"),
        "high": bar.get("h"),
        "low": bar.get("l"),
        "close": bar.get("c"),
        "volume": bar.get("v"),
        "trade_count": bar.get("n"),
        "vwap": bar.get("vw"),
    }


@dataclass(frozen=True)
class IngestResult:
    symbol: str
    timeframe: str
    asset_class: str
    days_requested: int
    days_ingested: int
    days_skipped: int
    rows: int
    api_calls: int
    cache_hits: int
    partitions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe} [{self.asset_class}]: "
            f"{self.days_ingested} ingested, {self.days_skipped} skipped "
            f"({self.rows} rows, {self.api_calls} api calls, {self.cache_hits} cache hits)"
        )


def _daterange(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _partition_done(lake: EveLake, symbol: str, asset_class: str, day: date) -> bool:
    part = lake.raw_partition("alpaca", asset_class, "bars", symbol, day)
    return (part / "manifest.json").exists()


def ingest_bars(
    client: AlpacaBarClient,
    lake: EveLake,
    *,
    symbol: str,
    timeframe: str,
    start: date | str,
    end: date | str,
    asset_class: str,
    feed: str | None = None,
    force: bool = False,
) -> IngestResult:
    """Download bars day-by-day into the lake, skipping days already present.

    Day granularity makes idempotency clean: each day is a lake partition and a
    cacheable request range, so re-ingesting a span only fetches the missing
    days, and ``force=True`` rebuilds from the response cache without network.
    """
    start_d = _as_date(start)
    end_d = _as_date(end)
    if end_d < start_d:
        raise ValueError("end must be on or after start")

    days = list(_daterange(start_d, end_d))
    hits0 = client.cache.stats.hits
    calls0 = client.api_calls
    ingested = skipped = rows = 0
    partitions: list[str] = []

    for day in days:
        if not force and _partition_done(lake, symbol, asset_class, day):
            skipped += 1
            continue
        day_start = f"{day.isoformat()}T00:00:00Z"
        day_end = f"{day.isoformat()}T23:59:59.999999Z"
        raw = list(
            client.iter_raw_bars(
                symbol, timeframe, day_start, day_end,
                asset_class=asset_class, feed=feed,
            )
        )
        if not raw:
            # Nothing for this day (holiday/no trades); record an empty marker so
            # we don't re-query it. Write a zero-row manifest sentinel.
            _write_empty_partition(lake, symbol, asset_class, day)
            ingested += 1
            continue
        bars = normalize_alpaca_bars(
            (alpaca_bar_to_norm_row(b) for b in raw),
            asset_class=asset_class,
            feed=feed or ("iex" if asset_class != "crypto" else "alpaca"),
            timeframe=timeframe,
        )
        manifest = write_jsonl_partition(bars, lake, partition_date=day)
        partitions.append(manifest.path)
        rows += manifest.rows
        ingested += 1

    return IngestResult(
        symbol=symbol,
        timeframe=timeframe,
        asset_class=asset_class,
        days_requested=len(days),
        days_ingested=ingested,
        days_skipped=skipped,
        rows=rows,
        api_calls=client.api_calls - calls0,
        cache_hits=client.cache.stats.hits - hits0,
        partitions=partitions,
    )


def _write_empty_partition(lake: EveLake, symbol: str, asset_class: str, day: date) -> None:
    part = lake.raw_partition("alpaca", asset_class, "bars", symbol, day)
    part.mkdir(parents=True, exist_ok=True)
    (part / "bars.jsonl").write_text("", encoding="utf-8")
    manifest = LakeManifest(
        schema_version="eve-lake-v1",
        provider="alpaca",
        asset_class=asset_class,
        dataset="bars",
        symbol=symbol,
        partition_date=day.isoformat(),
        rows=0,
        path=str(part / "bars.jsonl"),
        created_ts=time.time(),
    )
    (part / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Eve: cached Alpaca historical bar ingest")
    ap.add_argument("--symbols", required=True, help="comma-separated, e.g. 'AAPL,MSFT' or 'BTC/USD'")
    ap.add_argument("--asset-class", required=True, choices=["equity", "crypto"])
    ap.add_argument("--timeframe", default="1Min")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--feed", default=None, help="stock feed (default iex on free plan)")
    ap.add_argument("--lake", default=None, help="lake root (default $EVE_LAKE_DIR or data/lake)")
    ap.add_argument("--max-per-min", type=float, default=180.0)
    ap.add_argument("--force", action="store_true", help="re-ingest even if partitions exist (uses cache)")
    args = ap.parse_args()

    creds = AlpacaCreds.from_env()
    if creds is None:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment/.env")

    lake = EveLake(Path(args.lake)) if args.lake else EveLake.from_env()
    cache = ResponseCache(lake.root / "raw" / "alpaca" / "_apicache")
    client = AlpacaBarClient(creds, cache, max_per_min=args.max_per_min)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for symbol in symbols:
        result = ingest_bars(
            client, lake,
            symbol=symbol, timeframe=args.timeframe,
            start=args.start, end=args.end,
            asset_class=args.asset_class, feed=args.feed, force=args.force,
        )
        print(result.summary())
    print(
        f"\nTotal: {client.api_calls} network calls, "
        f"{cache.stats.hits} cache hits, {cache.stats.writes} cache writes."
    )


if __name__ == "__main__":
    cli()
