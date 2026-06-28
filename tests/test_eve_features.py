from datetime import datetime, timedelta, timezone

import numpy as np

from src.eve_data import EveLake, normalize_alpaca_bars, write_jsonl_partition
from src.eve_features import (
    DEFAULT_WINDOWS,
    _csrank_norm,
    build_feature_panel,
    build_ragged_feature_panel,
    compute_features,
)


def _bars(sym="AAA", n=200, seed=0):
    rng = np.random.default_rng(seed)
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    price, r = 100.0, 0.0
    rows = []
    for d in range(n):
        r = 0.4 * r + rng.normal(0.0004, 0.012)
        prev, price = price, price * (1 + r)
        rows.append({"symbol": sym, "timestamp": t0 + timedelta(days=d),
                     "open": prev, "high": max(prev, price) * 1.003,
                     "low": min(prev, price) * 0.997, "close": price,
                     "volume": 1e6 * (1 + 0.2 * rng.standard_normal())})
    return normalize_alpaca_bars(rows, asset_class="equity", feed="yahoo",
                                 timeframe="1d", provider="yahoo")


def test_compute_features_shape_and_leakage():
    bars = _bars(n=200)
    dates, mat, names = compute_features(bars)
    # 5 K-line + 10 rolling factors per window (8 price/vol + dvol + amihud liquidity)
    assert len(names) == 5 + 10 * len(DEFAULT_WINDOWS)
    assert "dvol20" in names and "amihud20" in names
    assert mat.shape == (200, len(names))
    # Deepest window is 60 -> rows before that are not all-finite (leakage-safe).
    assert not np.isfinite(mat[10]).all()
    assert np.isfinite(mat[120]).all()


def test_csrank_norm_range_and_nan_handling():
    x = np.array([[3.0, 1.0, 2.0, np.nan], [10.0, 20.0, 30.0, 40.0]])
    out = _csrank_norm(x)
    assert out.min() >= -0.5 - 1e-9 and out.max() <= 0.5 + 1e-9
    # monotonic: largest finite value gets the top rank in row 1
    assert np.argmax(out[1]) == 3
    # NaN mapped to the middle, not an extreme
    assert abs(out[0, 3]) < 0.5


def _write_lake(tmp_path, symbols, n=160):
    lake = EveLake(tmp_path / "lake")
    for i, s in enumerate(symbols):
        bars = _bars(s, n=n, seed=i)
        by_day = {}
        for b in bars:
            by_day.setdefault(b.ts[:10], []).append(b)
        for day, db in by_day.items():
            write_jsonl_partition(db, lake, partition_date=day, dataset="bars-1d")
    return lake


def test_build_feature_panel_dense_and_ranknormed(tmp_path):
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    lake = _write_lake(tmp_path, syms, n=160)
    panel = build_feature_panel(lake, syms, asset_class="equity", provider="yahoo", timeframe="1d")
    T, N, F = panel.signals.shape
    assert N == 5 and F == 5 + 10 * len(DEFAULT_WINDOWS)
    assert panel.fwd_returns.shape == (T, N)
    assert np.isfinite(panel.signals).all()           # dense, no NaN after assembly
    assert np.isfinite(panel.fwd_returns).all()
    # rank-normalized features live in [-0.5, 0.5]
    assert panel.signals.min() >= -0.5 - 1e-9 and panel.signals.max() <= 0.5 + 1e-9


def _delisted_dict_bars(n=180, seed=99):
    rng = np.random.default_rng(seed)
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    price, r, out = 100.0, 0.0, []
    for d in range(n):
        r = 0.4 * r + rng.normal(0.0004, 0.012)
        prev, price = price, price * (1 + r)
        out.append({"ts": (t0 + timedelta(days=d)).strftime("%Y-%m-%d"),
                    "open": prev, "high": max(prev, price) * 1.003,
                    "low": min(prev, price) * 0.997, "close": price,
                    "volume": 1e6})
    return out


def test_ragged_panel_includes_delisted_with_valid_mask(tmp_path):
    syms = ["AAA", "BBB", "CCC", "DDD"]
    lake = _write_lake(tmp_path, syms, n=400)          # ~13 months, all alive
    dead = _delisted_dict_bars(n=180)                  # ~6 months then stops
    panel = build_ragged_feature_panel(
        lake, syms, {"DEAD": dead}, min_names=2)
    assert panel.valid is not None
    assert "DEAD" in panel.symbols
    j = panel.symbols.index("DEAD")
    # DEAD is valid early then goes absent; absent rows have NaN features.
    assert panel.valid[:, j].any() and not panel.valid[:, j].all()
    absent = ~panel.valid[:, j]
    assert np.isnan(panel.signals[absent, j]).all()
    # Where absent, fwd return is 0 (weight will be 0 there) and survivors stay valid.
    assert (panel.fwd_returns[absent, j] == 0.0).all()
    # Valid (present) entries have finite features.
    pres = panel.valid[:, j]
    assert np.isfinite(panel.signals[pres, j]).all()


def test_monthly_rebalance_subsamples_to_month_ends(tmp_path):
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    lake = _write_lake(tmp_path, syms, n=400)  # ~13 months of daily bars
    daily = build_feature_panel(lake, syms, asset_class="equity", provider="yahoo",
                                timeframe="1d", rebalance="1d")
    monthly = build_feature_panel(lake, syms, asset_class="equity", provider="yahoo",
                                  timeframe="1d", rebalance="1m")
    # Far fewer rows (one per month, not per day) but same symbols/features.
    assert monthly.signals.shape[0] < daily.signals.shape[0] / 10
    assert monthly.signals.shape[1:] == daily.signals.shape[1:]
    # Panel dates are unique month-ends (distinct YYYY-MM prefixes).
    months = [d[:7] for d in monthly.dates]
    assert len(months) == len(set(months))
    assert np.isfinite(monthly.signals).all() and np.isfinite(monthly.fwd_returns).all()
