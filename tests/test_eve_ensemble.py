from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.eve_data import EveLake, normalize_alpaca_bars, write_jsonl_partition

pytest.importorskip("torch")
from src.eve_ensemble import ensemble_walk_forward  # noqa: E402


def _write_daily_lake(tmp_path, symbols, n_days=120, seed=0):
    lake = EveLake(tmp_path / "lake")
    rng = np.random.default_rng(seed)
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for si, sym in enumerate(symbols):
        price, r = 100.0, 0.0
        rows = []
        for d in range(n_days):
            r = 0.5 * r + rng.normal(0.0003, 0.01)
            prev, price = price, price * (1 + r)
            rows.append({
                "symbol": sym, "timestamp": t0 + timedelta(days=d),
                "open": prev, "high": max(prev, price) * 1.001,
                "low": min(prev, price) * 0.999, "close": price, "volume": 1000,
            })
        bars = normalize_alpaca_bars(rows, asset_class="equity", feed="yahoo",
                                     timeframe="1d", provider="yahoo")
        # one partition per day (mirrors ingest)
        by_day = {}
        for b in bars:
            by_day.setdefault(b.ts[:10], []).append(b)
        for day, day_bars in by_day.items():
            write_jsonl_partition(day_bars, lake, partition_date=day, dataset="bars-1d")
    return lake


def test_ensemble_walk_forward_smoke(tmp_path):
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    lake = _write_daily_lake(tmp_path, symbols, n_days=110)
    report = ensemble_walk_forward(
        lake, symbols, asset_class="equity", provider="yahoo", timeframe="1d",
        window=6, horizon=1, cost=0.0005, n_folds=2, min_train_days=50,
        transformer_kwargs={"epochs": 2, "device": "cpu"},
        allocator_kwargs={"epochs": 3},
    )
    names = {m.name for m in report.metrics}
    assert {"ensemble", "transformer_ls", "equal_weight", "cash"} == names
    assert report.machine_line().startswith("eve_portfolio:")
    ens = next(m for m in report.metrics if m.name == "ensemble")
    assert ens.n > 0  # produced out-of-sample allocations across folds
