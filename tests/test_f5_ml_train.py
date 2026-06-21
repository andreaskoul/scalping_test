"""Phase 5: directional-model training (Rank 5)."""

import numpy as np

from src.ml_train import (
    auc, train_logistic, walk_forward, build_dataset, weights_dict, train_from_tape,
)
from src.microstructure import MicroReplay, DirectionalModel, MicroFeatures, FEATURE_KEYS


def test_auc_extremes():
    y = np.array([0, 0, 1, 1.0])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0      # perfect
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0      # reversed
    assert abs(auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_train_logistic_separable():
    rng = np.random.default_rng(0)
    n = 400
    ofi = rng.uniform(-1, 1, n)
    y = (ofi > 0).astype(float)
    X = np.column_stack([np.ones(n), ofi, np.zeros(n), np.zeros(n)])
    w = train_logistic(X, y, epochs=2000)
    assert w[1] > 0                       # positive OFI weight learned
    assert auc(y, X @ w) > 0.95


def _regime_tape():
    """Alternating up/down regimes where OFI sign predicts the next move."""
    trades = []
    price, t = 100.0, 0.0
    for block in range(12):
        d = 1.0 if block % 2 == 0 else -1.0
        for _ in range(120):
            price += d * 0.05
            trades.append((t, price, d * 5.0, 5.0))
            t += 1.0
    return trades


def test_build_dataset_shapes():
    X, y = build_dataset(MicroReplay(_regime_tape()), horizon=30.0, step=10.0)
    assert X.shape[1] == len(FEATURE_KEYS)
    assert len(y) == X.shape[0] > 50
    assert set(np.unique(y)).issubset({0.0, 1.0})


def test_train_from_tape_learns_signal():
    res = train_from_tape(_regime_tape(), horizon=30.0, step=10.0, folds=4)
    assert res["weights"] is not None
    assert res["weights"]["ofi"] > 0.0
    assert res["auc"] > 0.65              # walk-forward, out-of-sample


def test_walk_forward_range():
    X, y = build_dataset(MicroReplay(_regime_tape()), horizon=30.0, step=10.0)
    a = walk_forward(X, y, folds=4)
    assert 0.0 <= a <= 1.0


def test_weights_dict_loads_into_model():
    w = weights_dict(np.array([0.0, 3.0, 1.0, 0.5]))
    assert w["ofi"] == 3.0 and w["bias"] == 0.0
    m = DirectionalModel(w)
    assert m.p_up(MicroFeatures(ofi=1.0)) > 0.5
    assert m.p_up(MicroFeatures(ofi=-1.0)) < 0.5
