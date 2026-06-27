import numpy as np
import pytest

from src.eve_portfolio import Panel, simulate, walk_forward_portfolio, cash_weights, equal_weights

pytest.importorskip("lightgbm")
from src.eve_features import _csrank_norm  # noqa: E402
from src.eve_gbdt import GBDTPredictor, GBDTTopK  # noqa: E402


def _predictive_feature_panel(T=400, N=24, F=10, seed=0):
    """Feature panel where features 0 (+) and 1 (-) predict the forward return."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(0, 1, (T, N, F))
    fwd = 0.02 * sig[:, :, 0] - 0.012 * sig[:, :, 1] + rng.normal(0, 0.004, (T, N))
    for k in range(F):
        sig[:, :, k] = _csrank_norm(sig[:, :, k])
    dates = [f"d{t:04d}" for t in range(T)]
    return Panel(dates, [f"S{i}" for i in range(N)], sig, fwd)


def _fast_pred():
    return {"n_estimators": 80, "num_leaves": 15, "min_child_samples": 20}


def test_gbdt_predict_signal_shape():
    panel = _predictive_feature_panel(T=150)
    gb = GBDTPredictor(**_fast_pred()).fit(panel)
    scores = gb.predict_signal(panel)
    assert scores.shape == (150, 24)
    assert np.isfinite(scores).all()


def test_gbdt_topk_captures_signal_oos():
    panel = _predictive_feature_panel(T=500, seed=1)
    train = Panel(panel.dates[:350], panel.symbols, panel.signals[:350], panel.fwd_returns[:350])
    test = Panel(panel.dates[350:], panel.symbols, panel.signals[350:], panel.fwd_returns[350:])
    strat = GBDTTopK(k=6, predictor_kwargs=_fast_pred()).fit(train)
    W = strat.predict_weights(test)
    nets = simulate(W, test.fwd_returns, cost=0.0002)
    assert nets.mean() > 0  # GBDT recovers the predictive features out-of-sample


def test_gbdt_topk_plugs_into_walk_forward():
    panel = _predictive_feature_panel(T=500, seed=2)
    report = walk_forward_portfolio(
        panel,
        {"cash": cash_weights, "equal_weight": equal_weights,
         "gbdt_topk": GBDTTopK(k=6, predictor_kwargs=_fast_pred())},
        cost=0.0002, n_folds=3, min_train=200,
    )
    gb = next(m for m in report.metrics if m.name == "gbdt_topk")
    assert gb.n > 0 and gb.sharpe > 0
    assert report.best_name == "gbdt_topk"
