from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.eve_data import build_bar_sequences, normalize_alpaca_bars, write_sequence_dataset, chronological_split, EveLake, write_jsonl_partition
from src.eve_labels import CostModel, FLAT
from src.eve_baselines import (
    LogisticBaseline,
    NoTradeBaseline,
    PersistenceBaseline,
    WalkForwardResult,
    build_report,
    build_report_from_lake,
    evaluate_result,
    load_samples_from_dataset,
    walk_forward,
    _last_step_return,
    _window_momentum,
)


def _ar1_bars(n, *, phi, noise, seed, symbol="BTC/USD"):
    """Synthetic 1-min bars whose returns follow an AR(1) momentum process."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    price = 100.0
    r = 0.0
    rows = []
    for i in range(n):
        r = phi * r + rng.normal(0.0, noise)
        prev = price
        price = prev * (1.0 + r)
        rows.append(
            {
                "symbol": symbol,
                "timestamp": t0 + timedelta(minutes=i),
                "open": prev,
                "high": max(prev, price) * 1.0001,
                "low": min(prev, price) * 0.9999,
                "close": price,
                "volume": 10.0,
            }
        )
    return normalize_alpaca_bars(rows, asset_class="crypto", feed="test", timeframe="1Min")


def _samples(phi, noise, seed, n=1600, window=8, horizon=1):
    bars = _ar1_bars(n, phi=phi, noise=noise, seed=seed)
    return build_bar_sequences(bars, window=window, horizon=horizon, threshold=0.0)


def test_feature_helpers_recover_window_returns():
    bars = _ar1_bars(40, phi=0.5, noise=0.002, seed=1)
    samples = build_bar_sequences(bars, window=6, horizon=1, threshold=0.0)
    s = samples[10]
    feats = s.features
    assert _last_step_return(s) == pytest.approx(feats[-1][3] / feats[-2][3] - 1.0)
    assert _window_momentum(s) == pytest.approx(feats[-1][3] / feats[0][3] - 1.0)


def test_notrade_baseline_is_always_flat_zero_expectancy():
    samples = _samples(phi=0.6, noise=0.003, seed=2)
    res = walk_forward(samples, [NoTradeBaseline()], cost=0.0003, n_folds=4, min_train=100)
    m = evaluate_result(res[0], cost=0.0003, n_trials=4)
    assert m.coverage == 0.0
    assert m.post_cost_expectancy == 0.0
    assert m.n > 0


def test_walk_forward_is_anchored_no_leakage():
    """Each fold's test block must be strictly after its training block."""
    samples = _samples(phi=0.6, noise=0.003, seed=3)

    seen = []

    class Spy:
        name = "spy"

        def fit(self, train, cost):
            self._train_max = max(s.end_ts for s in train)
            return self

        def predict(self, test):
            test_min = min(s.end_ts for s in test)
            seen.append((self._train_max, test_min))
            return [FLAT] * len(test)

    res = walk_forward(samples, [Spy()], cost=0.0003, n_folds=5, min_train=100)
    assert res[0].n_folds == len(seen) >= 1
    for train_max, test_min in seen:
        assert train_max < test_min  # no overlap, strictly forward


def test_momentum_signal_lets_a_baseline_beat_no_trade_after_costs():
    # Strong positive autocorrelation -> persistence/momentum should clear cost.
    samples = _samples(phi=0.6, noise=0.004, seed=7)
    cm = CostModel(taker_fee_bps=0.0, half_spread_bps=1.0, slippage_bps=0.5)  # 3 bps round trip
    report = build_report(samples, cost_model=cm, n_folds=5, min_train=100)

    assert report.best_beats_notrade, report.machine_line()
    assert report.gate_reasons == []
    best = next(m for m in report.metrics if m.name == report.best_name)
    assert best.name != "no_trade"
    assert best.post_cost_expectancy > 0
    assert best.t_stat >= 2.0
    assert best.ci_lo > 0
    assert "beats_notrade=true" in report.machine_line()


def test_pure_noise_does_not_beat_no_trade():
    # No autocorrelation -> directional baselines pay cost with no edge.
    samples = _samples(phi=0.0, noise=0.003, seed=11)
    report = build_report(samples, n_folds=5, min_train=100)
    assert not report.best_beats_notrade
    assert report.gate_reasons  # blocked for a stated reason
    assert "beats_notrade=false" in report.machine_line()


def test_logistic_emits_calibration_brier():
    samples = _samples(phi=0.6, noise=0.004, seed=5)
    res = walk_forward(samples, [LogisticBaseline(iters=150)], cost=0.0003, n_folds=4, min_train=120)
    m = evaluate_result(res[0], cost=0.0003, n_trials=4)
    assert not np.isnan(m.brier)
    assert 0.0 <= m.brier <= 2.0  # multiclass Brier upper bound is 2


def test_empty_samples_report_is_safe():
    report = build_report([], n_folds=5, min_train=50)
    assert report.best_name in {"no_trade", "persistence", "momentum_rule", "logistic"}
    assert not report.best_beats_notrade
    assert all(m.n == 0 for m in report.metrics)


def test_build_report_from_lake_reads_real_partitions(tmp_path):
    # Write AR(1) bars to the lake as timeframe-scoped partitions, then verdict.
    bars = _ar1_bars(600, phi=0.6, noise=0.004, seed=4)
    lake = EveLake(tmp_path / "lake")
    # group by day and write each as its own partition (mirrors ingest)
    by_day: dict[str, list] = {}
    for b in bars:
        by_day.setdefault(b.ts[:10], []).append(b)
    for day, day_bars in by_day.items():
        write_jsonl_partition(day_bars, lake, partition_date=day, dataset="bars-1min")

    report, n_bars = build_report_from_lake(
        lake, symbol="BTC/USD", asset_class="crypto", timeframe="1Min",
        window=8, horizon=1, n_folds=4, min_train=100,
    )
    assert n_bars == 600
    assert report.machine_line().startswith("eve_baseline:")
    assert sum(report.label_dist.values()) > 0


def test_round_trip_through_lake_dataset(tmp_path):
    samples = _samples(phi=0.5, noise=0.003, seed=9, n=400)
    split = chronological_split(samples)
    lake = EveLake(tmp_path / "lake")
    manifest = write_sequence_dataset(split, lake, dataset_version="eve-test-v1")
    loaded = load_samples_from_dataset(manifest.path)
    assert len(loaded) == len(samples)
    # Loaded samples reconstruct into a runnable report.
    report = build_report(loaded, n_folds=3, min_train=80)
    assert report.machine_line().startswith("eve_baseline:")
