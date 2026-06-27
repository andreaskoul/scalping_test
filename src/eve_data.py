"""Historical data helpers for Eve transformer research.

This module is intentionally batch-only. Alpaca is used as a historical data
source for Eve training datasets, not as a live market-data or execution path.
The pure helpers below avoid requiring alpaca-py in tests; network downloaders
can be layered on top once credentials and plan limits are configured.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, fields as dataclass_fields
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "eve-lake-v1"


@dataclass(frozen=True)
class EveLake:
    """Rooted local lake for immutable raw and derived Eve datasets."""

    root: Path

    @classmethod
    def from_env(cls) -> "EveLake":
        return cls(Path(os.getenv("EVE_LAKE_DIR", "data/lake")))

    def raw_partition(
        self,
        provider: str,
        asset_class: str,
        dataset: str,
        symbol: str,
        partition_date: date | str,
    ) -> Path:
        day = partition_date.isoformat() if isinstance(partition_date, date) else str(partition_date)
        return (
            self.root
            / "raw"
            / _slug(provider)
            / f"asset_class={_slug(asset_class)}"
            / f"dataset={_slug(dataset)}"
            / f"symbol={_slug(symbol)}"
            / f"date={day}"
        )


@dataclass(frozen=True)
class LakeManifest:
    schema_version: str
    provider: str
    asset_class: str
    dataset: str
    symbol: str
    partition_date: str
    rows: int
    path: str
    created_ts: float
    source: str = "historical_batch"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


@dataclass(frozen=True)
class AlpacaBar:
    """Normalized Alpaca historical bar record for Eve training."""

    provider: str
    asset_class: str
    symbol: str
    ts: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: float | None = None
    vwap: float | None = None
    feed: str = "unknown"
    timeframe: str = "unknown"
    source: str = "historical_batch"

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EveSequenceSample:
    """Leakage-safe fixed-window sample for Eve model training."""

    provider: str
    asset_class: str
    symbol: str
    start_ts: str
    end_ts: str
    label_ts: str
    horizon_steps: int
    features: list[list[float]]
    future_return: float
    label: int  # 0=down, 1=no-trade/stable, 2=up

    def row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChronologicalSplit:
    train: list[EveSequenceSample]
    val: list[EveSequenceSample]
    test: list[EveSequenceSample]


@dataclass(frozen=True)
class SequenceDatasetManifest:
    schema_version: str
    dataset_version: str
    path: str
    train_rows: int
    val_rows: int
    test_rows: int
    summary: dict[str, Any]
    created_ts: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def normalize_alpaca_bars(
    rows: Iterable[Mapping[str, Any]],
    *,
    asset_class: str,
    feed: str = "unknown",
    timeframe: str = "unknown",
    provider: str = "alpaca",
) -> list[AlpacaBar]:
    """Normalize bar-like mappings into deterministic records.

    Accepts dict-like rows from alpaca-py DataFrame exports, test fixtures, the
    Alpaca downloader, or other providers (``provider`` overrides the source
    tag, e.g. "yahoo"). Required fields are symbol, timestamp, OHLC, and volume.
    Extra fields are ignored.
    """
    out: list[AlpacaBar] = []
    for row in rows:
        symbol = str(_first_present(row, "symbol", "tic") or "").strip()
        ts = _normalize_ts(_first_present(row, "timestamp", "ts", "date"))
        if not symbol or not ts:
            raise ValueError("bar row requires symbol and timestamp")
        out.append(
            AlpacaBar(
                provider=provider,
                asset_class=asset_class,
                symbol=symbol,
                ts=ts,
                open=_float_field(row, "open"),
                high=_float_field(row, "high"),
                low=_float_field(row, "low"),
                close=_float_field(row, "close"),
                volume=_float_field(row, "volume"),
                trade_count=_optional_float(row, "trade_count"),
                vwap=_optional_float(row, "vwap"),
                feed=feed,
                timeframe=timeframe,
            )
        )
    out.sort(key=lambda bar: (bar.symbol, bar.ts))
    return out


def write_jsonl_partition(
    records: Iterable[AlpacaBar],
    lake: EveLake,
    *,
    partition_date: date | str | None = None,
    dataset: str = "bars",
) -> LakeManifest:
    """Write normalized records to one raw JSONL partition and manifest.

    ``dataset`` names the partition family; ingest scopes it by timeframe
    (e.g. ``bars-1min``) so different timeframes for the same date never collide
    in one partition.
    """
    rows = list(records)
    if not rows:
        raise ValueError("cannot write an empty Eve lake partition")
    first = rows[0]
    if any((r.provider, r.asset_class, r.symbol) != (first.provider, first.asset_class, first.symbol) for r in rows):
        raise ValueError("records in one partition must share provider, asset_class, and symbol")

    part_day = partition_date or first.ts[:10]
    part = lake.raw_partition(first.provider, first.asset_class, dataset, first.symbol, part_day)
    part.mkdir(parents=True, exist_ok=True)
    data_path = part / "bars.jsonl"
    with data_path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record.row(), sort_keys=True, separators=(",", ":")) + "\n")

    manifest = LakeManifest(
        schema_version=SCHEMA_VERSION,
        provider=first.provider,
        asset_class=first.asset_class,
        dataset=dataset,
        symbol=first.symbol,
        partition_date=str(part_day),
        rows=len(rows),
        path=str(data_path),
        created_ts=time.time(),
    )
    (part / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def read_symbol_bars(
    lake: EveLake,
    *,
    provider: str,
    asset_class: str,
    symbol: str,
    dataset: str = "bars",
) -> list[AlpacaBar]:
    """Read every written bar partition for one symbol back into AlpacaBars.

    Inverse of :func:`write_jsonl_partition`: walks the symbol's ``date=*``
    partitions, parses each ``bars.jsonl`` row, and returns them sorted by
    timestamp. Empty (holiday/no-trade) partitions contribute nothing. Unknown
    extra keys are ignored so the reader survives forward schema additions.
    """
    symbol_dir = (
        lake.root
        / "raw"
        / _slug(provider)
        / f"asset_class={_slug(asset_class)}"
        / f"dataset={_slug(dataset)}"
        / f"symbol={_slug(symbol)}"
    )
    if not symbol_dir.exists():
        return []
    fields = {f.name for f in dataclass_fields(AlpacaBar)}
    out: list[AlpacaBar] = []
    for part in sorted(symbol_dir.glob("date=*")):
        data_path = part / "bars.jsonl"
        if not data_path.exists():
            continue
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out.append(AlpacaBar(**{k: v for k, v in row.items() if k in fields}))
    out.sort(key=lambda bar: (bar.symbol, bar.ts))
    return out


def build_bar_sequences(
    bars: Iterable[AlpacaBar],
    *,
    window: int,
    horizon: int,
    threshold: float = 0.0,
) -> list[EveSequenceSample]:
    """Build fixed-window samples from one symbol's chronological bars.

    Features use only bars from [t-window+1, t]. The label uses close[t+horizon]
    and is therefore kept outside the feature window. Labels are {-1,0,+1}
    encoded as {0,1,2}: down, stable/no-trade, up.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    ordered = sorted(list(bars), key=lambda bar: (bar.symbol, bar.ts))
    if not ordered:
        return []
    first = ordered[0]
    if any((bar.provider, bar.asset_class, bar.symbol) != (first.provider, first.asset_class, first.symbol) for bar in ordered):
        raise ValueError("sequence builder expects one provider, asset_class, and symbol")

    out: list[EveSequenceSample] = []
    last_i = len(ordered) - horizon
    for end_i in range(window - 1, last_i):
        feature_window = ordered[end_i - window + 1:end_i + 1]
        now = ordered[end_i]
        fut = ordered[end_i + horizon]
        if now.close <= 0 or fut.close <= 0:
            continue
        future_return = (fut.close / now.close) - 1.0
        if future_return > threshold:
            label = 2
        elif future_return < -threshold:
            label = 0
        else:
            label = 1
        out.append(
            EveSequenceSample(
                provider=first.provider,
                asset_class=first.asset_class,
                symbol=first.symbol,
                start_ts=feature_window[0].ts,
                end_ts=now.ts,
                label_ts=fut.ts,
                horizon_steps=horizon,
                features=[_bar_features(bar) for bar in feature_window],
                future_return=float(future_return),
                label=label,
            )
        )
    return out


def chronological_split(
    samples: Iterable[EveSequenceSample],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> ChronologicalSplit:
    """Split samples by time order, never randomly."""
    ordered = sorted(list(samples), key=lambda sample: (sample.symbol, sample.end_ts, sample.label_ts))
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return ChronologicalSplit(
        train=ordered[:train_end],
        val=ordered[train_end:val_end],
        test=ordered[val_end:],
    )


def dataset_summary(samples: Iterable[EveSequenceSample]) -> dict[str, Any]:
    """Small manifest-friendly summary for generated Eve samples."""
    rows = list(samples)
    labels = {"down": 0, "stable": 0, "up": 0}
    for sample in rows:
        if sample.label == 0:
            labels["down"] += 1
        elif sample.label == 1:
            labels["stable"] += 1
        elif sample.label == 2:
            labels["up"] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "symbols": sorted({sample.symbol for sample in rows}),
        "start_ts": min((sample.start_ts for sample in rows), default=""),
        "end_ts": max((sample.label_ts for sample in rows), default=""),
        "labels": labels,
    }


def write_sequence_dataset(
    split: ChronologicalSplit,
    lake: EveLake,
    *,
    dataset_version: str,
) -> SequenceDatasetManifest:
    """Write Eve train/val/test sequence samples with a manifest."""
    if not dataset_version or not str(dataset_version).strip():
        raise ValueError("dataset_version is required")
    base = lake.root / "features" / "eve_transformer" / _slug(dataset_version)
    base.mkdir(parents=True, exist_ok=True)
    _write_samples(base / "train.jsonl", split.train)
    _write_samples(base / "val.jsonl", split.val)
    _write_samples(base / "test.jsonl", split.test)

    all_samples = [*split.train, *split.val, *split.test]
    manifest = SequenceDatasetManifest(
        schema_version=SCHEMA_VERSION,
        dataset_version=str(dataset_version),
        path=str(base),
        train_rows=len(split.train),
        val_rows=len(split.val),
        test_rows=len(split.test),
        summary=dataset_summary(all_samples),
        created_ts=time.time(),
    )
    (base / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def select_torch_device(torch_module: Any) -> str:
    """Return Eve's preferred torch device name: mps, cuda, then cpu."""
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return "mps"
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda"
    return "cpu"


def _bar_features(bar: AlpacaBar) -> list[float]:
    mid = 0.5 * (bar.high + bar.low)
    range_rel = 0.0 if mid <= 0 else (bar.high - bar.low) / mid
    close_open = 0.0 if bar.open <= 0 else (bar.close / bar.open) - 1.0
    vwap_gap = 0.0 if bar.vwap is None or bar.close <= 0 else (bar.vwap / bar.close) - 1.0
    return [
        float(bar.open),
        float(bar.high),
        float(bar.low),
        float(bar.close),
        float(bar.volume),
        float(bar.trade_count or 0.0),
        float(vwap_gap),
        float(range_rel),
        float(close_open),
    ]


def _write_samples(path: Path, samples: Iterable[EveSequenceSample]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.row(), sort_keys=True, separators=(",", ":")) + "\n")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(value).strip())
    return cleaned.strip("-").lower() or "unknown"


def _first_present(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _normalize_ts(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return ""
        text = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return str(value).strip()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _float_field(row: Mapping[str, Any], name: str) -> float:
    value = _first_present(row, name)
    if value is None:
        raise ValueError(f"alpaca bar row requires {name}")
    return float(value)


def _optional_float(row: Mapping[str, Any], name: str) -> float | None:
    value = _first_present(row, name)
    if value is None or value == "":
        return None
    return float(value)
