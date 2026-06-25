import json
from datetime import datetime, timezone

import pytest

from src.eve_data import (
    EveLake,
    build_bar_sequences,
    chronological_split,
    dataset_summary,
    normalize_alpaca_bars,
    select_torch_device,
    write_sequence_dataset,
    write_jsonl_partition,
)


def test_normalize_alpaca_bars_sorts_and_normalizes_timestamps():
    rows = [
        {
            "symbol": "BTC/USD",
            "timestamp": datetime(2026, 6, 22, 10, 1, tzinfo=timezone.utc),
            "open": "101",
            "high": 102,
            "low": 100,
            "close": 101.5,
            "volume": 7,
            "trade_count": 3,
            "vwap": 101.2,
        },
        {
            "symbol": "BTC/USD",
            "timestamp": "2026-06-22T10:00:00Z",
            "open": 100,
            "high": 101,
            "low": 99,
            "close": 100.5,
            "volume": 5,
        },
    ]

    bars = normalize_alpaca_bars(rows, asset_class="crypto", feed="alpaca", timeframe="1Min")

    assert [bar.ts for bar in bars] == ["2026-06-22T10:00:00Z", "2026-06-22T10:01:00Z"]
    assert bars[0].symbol == "BTC/USD"
    assert bars[0].trade_count is None
    assert bars[1].vwap == 101.2
    assert bars[1].feed == "alpaca"


def test_normalize_alpaca_bars_requires_symbol_and_timestamp():
    with pytest.raises(ValueError, match="symbol and timestamp"):
        normalize_alpaca_bars([
            {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
        ], asset_class="equity")


def test_write_jsonl_partition_writes_manifest_and_rows(tmp_path):
    lake = EveLake(tmp_path / "lake")
    bars = normalize_alpaca_bars([
        {
            "symbol": "SPY",
            "timestamp": "2026-06-22T14:30:00Z",
            "open": 500,
            "high": 501,
            "low": 499,
            "close": 500.5,
            "volume": 1000,
        }
    ], asset_class="equity", feed="iex", timeframe="1Min")

    manifest = write_jsonl_partition(bars, lake)
    data_path = tmp_path / manifest.path
    manifest_path = data_path.parent / "manifest.json"

    assert manifest.rows == 1
    assert "provider=alpaca" not in manifest.path  # path is local, provider sits in directories below raw
    assert data_path.exists()
    assert manifest_path.exists()
    row = json.loads(data_path.read_text().strip())
    meta = json.loads(manifest_path.read_text())
    assert row["symbol"] == "SPY"
    assert row["asset_class"] == "equity"
    assert meta["schema_version"] == "eve-lake-v1"
    assert meta["rows"] == 1


def test_write_jsonl_partition_rejects_mixed_symbols(tmp_path):
    lake = EveLake(tmp_path / "lake")
    bars = normalize_alpaca_bars([
        {"symbol": "SPY", "timestamp": "2026-06-22T14:30:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"symbol": "QQQ", "timestamp": "2026-06-22T14:30:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ], asset_class="equity")

    with pytest.raises(ValueError, match="share provider"):
        write_jsonl_partition(bars, lake)


class _Backend:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _TorchLike:
    def __init__(self, *, mps=False, cuda=False):
        self.backends = type("Backends", (), {"mps": _Backend(mps)})()
        self.cuda = _Backend(cuda)


def test_select_torch_device_prefers_mps_then_cuda_then_cpu():
    assert select_torch_device(_TorchLike(mps=True, cuda=True)) == "mps"
    assert select_torch_device(_TorchLike(mps=False, cuda=True)) == "cuda"
    assert select_torch_device(_TorchLike(mps=False, cuda=False)) == "cpu"


def _bars_for_sequences():
    rows = []
    for i, close in enumerate([100.0, 101.0, 102.0, 104.0, 103.0, 99.0, 100.0, 101.0]):
        rows.append({
            "symbol": "SPY",
            "timestamp": f"2026-06-22T14:{30 + i:02d}:00Z",
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000 + i,
            "trade_count": 10 + i,
            "vwap": close + 0.25,
        })
    return normalize_alpaca_bars(rows, asset_class="equity", feed="iex", timeframe="1Min")


def test_build_bar_sequences_uses_past_window_and_future_label():
    samples = build_bar_sequences(_bars_for_sequences(), window=3, horizon=2, threshold=0.01)

    assert len(samples) == 4
    first = samples[0]
    assert first.start_ts == "2026-06-22T14:30:00Z"
    assert first.end_ts == "2026-06-22T14:32:00Z"
    assert first.label_ts == "2026-06-22T14:34:00Z"
    assert len(first.features) == 3
    assert first.features[-1][3] == 102.0  # feature close at end_ts
    assert all(row[3] != 103.0 for row in first.features)  # future close is not in features
    assert first.label == 1  # 102 -> 103 is inside +/-1% no-trade band

    second = samples[1]
    assert second.label == 0  # 104 -> 99 breaches down threshold


def test_build_bar_sequences_rejects_mixed_symbols():
    bars = _bars_for_sequences()
    mixed = list(bars)
    mixed[0] = type(mixed[0])(**{**mixed[0].row(), "symbol": "QQQ"})

    with pytest.raises(ValueError, match="one provider"):
        build_bar_sequences(mixed, window=3, horizon=1)


def test_chronological_split_preserves_time_order():
    samples = build_bar_sequences(_bars_for_sequences(), window=3, horizon=1, threshold=0.0)
    split = chronological_split(samples, train_frac=0.5, val_frac=0.25)

    assert len(split.train) == 2
    assert len(split.val) == 1
    assert len(split.test) == 2
    assert split.train[-1].end_ts < split.val[0].end_ts < split.test[0].end_ts


def test_dataset_summary_counts_labels_and_ranges():
    samples = build_bar_sequences(_bars_for_sequences(), window=3, horizon=2, threshold=0.01)
    summary = dataset_summary(samples)

    assert summary["schema_version"] == "eve-lake-v1"
    assert summary["rows"] == len(samples)
    assert summary["symbols"] == ["SPY"]
    assert summary["start_ts"] == "2026-06-22T14:30:00Z"
    assert summary["end_ts"] == "2026-06-22T14:37:00Z"
    assert sum(summary["labels"].values()) == len(samples)


def test_write_sequence_dataset_writes_splits_and_manifest(tmp_path):
    lake = EveLake(tmp_path / "lake")
    samples = build_bar_sequences(_bars_for_sequences(), window=3, horizon=1, threshold=0.0)
    split = chronological_split(samples, train_frac=0.5, val_frac=0.25)

    manifest = write_sequence_dataset(split, lake, dataset_version="SPY 1Min v1")
    base = tmp_path / manifest.path

    assert manifest.train_rows == 2
    assert manifest.val_rows == 1
    assert manifest.test_rows == 2
    assert (base / "train.jsonl").exists()
    assert (base / "val.jsonl").exists()
    assert (base / "test.jsonl").exists()
    meta = json.loads((base / "manifest.json").read_text())
    train_rows = [json.loads(line) for line in (base / "train.jsonl").read_text().splitlines()]
    assert meta["dataset_version"] == "SPY 1Min v1"
    assert meta["summary"]["rows"] == len(samples)
    assert train_rows[0]["features"][0][3] == 100.0


def test_write_sequence_dataset_requires_version(tmp_path):
    split = chronological_split(build_bar_sequences(_bars_for_sequences(), window=3, horizon=1))
    with pytest.raises(ValueError, match="dataset_version"):
        write_sequence_dataset(split, EveLake(tmp_path / "lake"), dataset_version="")
