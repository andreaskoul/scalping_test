from datetime import datetime, timedelta, timezone

import pytest

from src.eve_data import build_bar_sequences, normalize_alpaca_bars
from src.eve_labels import DOWN, FLAT, UP, CostModel

torch = pytest.importorskip("torch")
from src.eve_transformer import TransformerModel  # noqa: E402
from src.eve_baselines import build_report  # noqa: E402


def _bars(n=300, seed=0):
    import random

    rng = random.Random(seed)
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    price, r = 100.0, 0.0
    rows = []
    for i in range(n):
        r = 0.6 * r + rng.gauss(0, 0.004)  # AR(1) momentum
        prev, price = price, price * (1 + r)
        rows.append({"symbol": "BTC/USD", "timestamp": t0 + timedelta(minutes=i),
                     "open": prev, "high": max(prev, price) * 1.0001,
                     "low": min(prev, price) * 0.9999, "close": price, "volume": 10.0})
    return normalize_alpaca_bars(rows, asset_class="crypto")


def _samples(n=300, seed=0, window=8):
    return build_bar_sequences(_bars(n, seed), window=window, horizon=1, threshold=0.0)


def _tiny(**kw):
    return TransformerModel(d_model=16, n_heads=2, n_layers=1, dim_ff=32,
                            epochs=2, batch_size=64, device="cpu", **kw)


def test_fit_predict_shapes_and_valid_proba():
    samples = _samples(300)
    model = _tiny().fit(samples, cost=0.0003)
    proba = model.predict_proba(samples[:20])
    assert len(proba) == 20
    for row in proba:
        assert len(row) == 3
        assert abs(sum(row) - 1.0) < 1e-4
    preds = model.predict(samples[:20])
    assert set(preds) <= {DOWN, FLAT, UP}
    assert model.history["epochs_run"] >= 1


def test_cpu_determinism_same_seed():
    samples = _samples(300)
    p1 = _tiny(seed=7).fit(samples, 0.0003).predict(samples[:30])
    p2 = _tiny(seed=7).fit(samples, 0.0003).predict(samples[:30])
    assert p1 == p2


def test_empty_train_is_safe():
    model = _tiny().fit([], cost=0.0003)
    assert model.predict_proba(_samples(20)[:5]) == [[1 / 3, 1 / 3, 1 / 3]] * 5


def test_transformer_plugs_into_build_report_as_extra_model():
    samples = _samples(600, seed=1)
    report = build_report(
        samples, cost_model=CostModel(), n_folds=2, min_train=150,
        extra_models=[_tiny(seed=1)],
    )
    names = {m.name for m in report.metrics}
    assert "transformer" in names
    tr = next(m for m in report.metrics if m.name == "transformer")
    assert tr.n > 0  # produced out-of-sample predictions across folds
    # Brier is finite because the transformer exposes predict_proba.
    assert tr.brier == tr.brier  # not NaN
