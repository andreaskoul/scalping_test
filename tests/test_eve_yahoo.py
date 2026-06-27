import json
from datetime import date, datetime, timezone

import pytest

from src.eve_data import EveLake, read_symbol_bars
from src.eve_ingest import ResponseCache, bars_dataset
from src.eve_yahoo import YahooBarClient, ingest_yahoo_bars
from src.eve_ingest import AlpacaIngestError


def _epoch(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp())


class FakeYahoo:
    """Deterministic stand-in for the Yahoo chart endpoint."""

    def __init__(self, *, days: list[str], with_null_gap: bool = False, error: bool = False):
        self.days = days
        self.with_null_gap = with_null_gap
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, params, headers):
        self.calls.append((url, dict(params)))
        assert "User-Agent" in headers  # Yahoo needs a browser UA, no auth
        if self.error:
            return 200, {"chart": {"result": None, "error": {"code": "Not Found"}}}, {}
        stamps, opens, highs, lows, closes, vols = [], [], [], [], [], []
        for i, d in enumerate(self.days):
            stamps.append(_epoch(d))
            price = 1.10 + i * 0.001
            opens.append(price)
            highs.append(price + 0.002)
            lows.append(price - 0.002)
            closes.append(price + 0.001)
            vols.append(0)
        if self.with_null_gap:  # Yahoo pads missing slots with null
            stamps.append(_epoch(self.days[-1]) + 86_400)
            for arr in (opens, highs, lows, closes, vols):
                arr.append(None)
        body = {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": params and "EURUSD=X", "currency": "USD"},
                        "timestamp": stamps,
                        "indicators": {"quote": [{
                            "open": opens, "high": highs, "low": lows,
                            "close": closes, "volume": vols,
                        }]},
                    }
                ],
                "error": None,
            }
        }
        return 200, body, {}


def _client(transport, tmp_path):
    cache = ResponseCache(tmp_path / "ycache")
    return YahooBarClient(cache, transport=transport, max_per_min=0.0, sleep=lambda _s: None)


@pytest.fixture
def lake(tmp_path):
    return EveLake(tmp_path / "lake")


def test_ingest_yahoo_writes_normalized_forex_partitions(lake, tmp_path):
    transport = FakeYahoo(days=["2026-06-20", "2026-06-21", "2026-06-22"])
    client = _client(transport, tmp_path)
    res = ingest_yahoo_bars(client, lake, symbol="EURUSD=X", interval="1d",
                            start="2026-06-20", end="2026-06-22", asset_class="forex")
    assert res.rows == 3
    assert res.api_calls == 1  # one chart call covers the whole range
    bars = read_symbol_bars(lake, provider="yahoo", asset_class="forex",
                            symbol="EURUSD=X", dataset=bars_dataset("1d"))
    assert len(bars) == 3
    assert bars[0].provider == "yahoo" and bars[0].feed == "yahoo"
    assert bars[0].symbol == "EURUSD=X"


def test_null_gap_rows_are_skipped(lake, tmp_path):
    transport = FakeYahoo(days=["2026-06-20", "2026-06-21"], with_null_gap=True)
    client = _client(transport, tmp_path)
    res = ingest_yahoo_bars(client, lake, symbol="EURUSD=X", interval="1d",
                            start="2026-06-20", end="2026-06-23", asset_class="forex")
    # 2 real bars; the null-padded slot is dropped; remaining days are empty.
    assert res.rows == 2


def test_partition_skip_then_cache_no_refetch(lake, tmp_path):
    transport = FakeYahoo(days=["2026-06-20", "2026-06-21"])
    client = _client(transport, tmp_path)
    first = ingest_yahoo_bars(client, lake, symbol="EURUSD=X", interval="1d",
                              start="2026-06-20", end="2026-06-21", asset_class="forex")
    assert first.api_calls == 1
    # All days present -> fast path makes no request at all.
    second = ingest_yahoo_bars(client, lake, symbol="EURUSD=X", interval="1d",
                               start="2026-06-20", end="2026-06-21", asset_class="forex")
    assert second.days_skipped == 2 and second.api_calls == 0
    # force=True re-requests but the chart call is served from the response cache.
    forced = ingest_yahoo_bars(client, lake, symbol="EURUSD=X", interval="1d",
                               start="2026-06-20", end="2026-06-21", asset_class="forex", force=True)
    assert forced.api_calls == 0 and forced.cache_hits == 1


def test_yahoo_error_payload_raises(lake, tmp_path):
    transport = FakeYahoo(days=[], error=True)
    client = _client(transport, tmp_path)
    with pytest.raises(AlpacaIngestError):
        ingest_yahoo_bars(client, lake, symbol="BADSYM=X", interval="1d",
                          start="2026-06-20", end="2026-06-20", asset_class="forex")
