import json
from datetime import date

import pytest

from src.eve_data import EveLake
from src.eve_ingest import (
    AlpacaBarClient,
    AlpacaCreds,
    AlpacaIngestError,
    ResponseCache,
    alpaca_bar_to_norm_row,
    ingest_bars,
)


def _bar(day: str, minute: int, price: float) -> dict:
    hh, mm = divmod(minute, 60)
    return {
        "t": f"{day}T{hh:02d}:{mm:02d}:00Z",
        "o": price,
        "h": price + 1,
        "l": price - 1,
        "c": price + 0.5,
        "v": 100 + minute,
        "n": 5,
        "vw": price + 0.2,
    }


class FakeAlpaca:
    """Deterministic stand-in for the Alpaca data API with pagination."""

    def __init__(self, counts: dict, *, page_size: int = 10000, fail_status: int | None = None):
        # counts: {(symbol, "YYYY-MM-DD"): n_bars}
        self.counts = counts
        self.page_size = page_size
        self.fail_status = fail_status
        self.calls: list[tuple[str, dict]] = []

    def _bars(self, symbol: str, day: str) -> list[dict]:
        n = self.counts.get((symbol, day), 0)
        return [_bar(day, i, 100.0 + i) for i in range(n)]

    def __call__(self, url, params, headers):
        self.calls.append((url, dict(params)))
        assert headers.get("APCA-API-KEY-ID")  # auth always attached
        if self.fail_status is not None:
            return self.fail_status, {"message": "boom"}, {}
        symbol = params["symbols"]
        day = params["start"][:10]
        bars = self._bars(symbol, day)
        offset = int(params.get("page_token", 0))
        page = bars[offset:offset + self.page_size]
        nxt = offset + self.page_size
        token = str(nxt) if nxt < len(bars) else None
        body = {"bars": {symbol: page}}
        if token:
            body["next_page_token"] = token
        return 200, body, {}


def _client(transport):
    creds = AlpacaCreds(key="k", secret="s")
    cache = ResponseCache(transport._cache_root)  # set below
    return AlpacaBarClient(
        creds, cache, transport=transport, max_per_min=0.0, sleep=lambda _s: None
    )


@pytest.fixture
def lake(tmp_path):
    return EveLake(tmp_path / "lake")


def _make(transport, tmp_path):
    transport._cache_root = tmp_path / "cache"
    return _client(transport)


def test_alpaca_bar_to_norm_row_maps_compact_keys():
    row = alpaca_bar_to_norm_row({"symbol": "AAPL", "t": "2026-06-01T00:00:00Z",
                                  "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 9, "n": 3, "vw": 1.2})
    assert row == {
        "symbol": "AAPL", "timestamp": "2026-06-01T00:00:00Z",
        "open": 1, "high": 2, "low": 0.5, "close": 1.5,
        "volume": 9, "trade_count": 3, "vwap": 1.2,
    }


def test_ingest_writes_normalized_partitions(lake, tmp_path):
    transport = FakeAlpaca({("AAPL", "2026-06-01"): 3, ("AAPL", "2026-06-02"): 2})
    client = _make(transport, tmp_path)
    res = ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                      start="2026-06-01", end="2026-06-02", asset_class="equity")
    assert res.days_requested == 2
    assert res.days_ingested == 2
    assert res.rows == 5
    assert res.api_calls == 2  # one page per day
    # partition content is normalized AlpacaBar rows
    part = lake.raw_partition("alpaca", "equity", "bars", "AAPL", date(2026, 6, 1))
    lines = (part / "bars.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["provider"] == "alpaca" and first["symbol"] == "AAPL"
    assert first["open"] == 100.0 and first["timeframe"] == "1Min"


def test_pagination_stitches_pages(lake, tmp_path):
    transport = FakeAlpaca({("BTC/USD", "2026-06-01"): 5}, page_size=2)
    client = _make(transport, tmp_path)
    res = ingest_bars(client, lake, symbol="BTC/USD", timeframe="1Min",
                      start="2026-06-01", end="2026-06-01", asset_class="crypto")
    assert res.rows == 5
    assert res.api_calls == 3  # ceil(5/2) pages
    # crypto hits the v1beta3 endpoint, no stock feed param
    url, params = transport.calls[0]
    assert "/v1beta3/crypto/us/bars" in url
    assert "feed" not in params


def test_partition_skip_avoids_refetch(lake, tmp_path):
    transport = FakeAlpaca({("AAPL", "2026-06-01"): 2, ("AAPL", "2026-06-02"): 2})
    client = _make(transport, tmp_path)
    first = ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                        start="2026-06-01", end="2026-06-02", asset_class="equity")
    assert first.api_calls == 2
    # Second non-forced run: every day already on disk -> zero network calls.
    second = ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                         start="2026-06-01", end="2026-06-02", asset_class="equity")
    assert second.days_skipped == 2
    assert second.days_ingested == 0
    assert second.api_calls == 0


def test_force_reingest_uses_response_cache_not_network(lake, tmp_path):
    transport = FakeAlpaca({("AAPL", "2026-06-01"): 4}, page_size=2)
    client = _make(transport, tmp_path)
    ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                start="2026-06-01", end="2026-06-01", asset_class="equity")
    calls_after_first = client.api_calls
    assert calls_after_first == 2

    # force=True bypasses the partition skip, but the identical requests are
    # served from the response cache -> still no new network calls.
    forced = ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                         start="2026-06-01", end="2026-06-01", asset_class="equity", force=True)
    assert forced.api_calls == 0
    assert forced.cache_hits == 2
    assert client.api_calls == calls_after_first  # unchanged


def test_empty_day_is_recorded_and_not_requeried(lake, tmp_path):
    transport = FakeAlpaca({})  # no bars for any day
    client = _make(transport, tmp_path)
    res = ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                      start="2026-06-01", end="2026-06-01", asset_class="equity")
    assert res.rows == 0
    assert res.days_ingested == 1  # empty marker written
    part = lake.raw_partition("alpaca", "equity", "bars", "AAPL", date(2026, 6, 1))
    assert (part / "manifest.json").exists()
    manifest = json.loads((part / "manifest.json").read_text())
    assert manifest["rows"] == 0
    # Re-running skips it without a network call.
    res2 = ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                       start="2026-06-01", end="2026-06-01", asset_class="equity")
    assert res2.days_skipped == 1 and res2.api_calls == 0


def test_request_key_is_param_sensitive_and_auth_independent():
    k1 = ResponseCache.request_key("u", {"symbols": "AAPL", "start": "a"})
    k2 = ResponseCache.request_key("u", {"symbols": "AAPL", "start": "b"})
    k3 = ResponseCache.request_key("u", {"start": "a", "symbols": "AAPL"})  # order-independent
    assert k1 != k2
    assert k1 == k3


def test_http_error_raises_after_retries(lake, tmp_path):
    transport = FakeAlpaca({("AAPL", "2026-06-01"): 1}, fail_status=500)
    client = _make(transport, tmp_path)
    with pytest.raises(AlpacaIngestError):
        ingest_bars(client, lake, symbol="AAPL", timeframe="1Min",
                    start="2026-06-01", end="2026-06-01", asset_class="equity")
    assert client.api_calls == client.max_retries + 1  # initial + retries


def test_creds_from_env(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert AlpacaCreds.from_env() is None
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "sec")
    creds = AlpacaCreds.from_env()
    assert creds is not None
    assert creds.headers() == {"APCA-API-KEY-ID": "key", "APCA-API-SECRET-KEY": "sec"}
    assert creds.data_host == "https://data.alpaca.markets"
