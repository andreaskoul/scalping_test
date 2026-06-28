"""GBDT prediction lane — LightGBM on the Alpha-style factor panel.

qlib's leaderboard winners are gradient-boosted trees (LightGBM, DoubleEnsemble),
not transformers (`docs/EVE_SHARPE_LITERATURE.md`). This is that lane: a LightGBM
regressor trained cross-sectionally on the `eve_features` factor panel to predict
the **rank-normalized forward return**, i.e. relative strength. Its per-asset
prediction is the signal; portfolio construction (Top-k long-short, or the RL
allocator) turns scores into a book.

Both wrappers expose the `fit(train_panel)` / `predict_weights(test_panel)`
portfolio protocol, so they drop straight into `walk_forward_portfolio` and are
judged on the same post-cost, deflated-Sharpe OOS series as every baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .eve_features import _csrank_norm
from .eve_portfolio import (Panel, aim_weights_from_scores, partial_adjust_path,
                            topk_long_short_weights)


class GBDTPredictor:
    """LightGBM cross-sectional return-rank predictor."""

    def __init__(
        self,
        *,
        num_leaves: int = 31,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 100,
        reg_lambda: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        # Native LightGBM params (avoids the sklearn dependency).
        self.params = dict(
            objective="regression", num_leaves=num_leaves, learning_rate=learning_rate,
            max_depth=max_depth, bagging_fraction=subsample, bagging_freq=1,
            feature_fraction=colsample_bytree, min_data_in_leaf=min_child_samples,
            lambda_l2=reg_lambda, seed=seed, verbosity=-1, num_threads=0,
        )
        self._model = None

    def fit(self, panel: Panel) -> "GBDTPredictor":
        import lightgbm as lgb

        T, N, F = panel.signals.shape
        X = panel.signals.reshape(T * N, F)
        # Target: cross-sectional rank of the forward return (relative strength).
        y = _csrank_norm(panel.fwd_returns).reshape(T * N)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        dtrain = lgb.Dataset(X[mask], label=y[mask], free_raw_data=False)
        self._model = lgb.train(self.params, dtrain, num_boost_round=self.n_estimators)
        return self

    def predict_signal(self, panel: Panel) -> np.ndarray:
        T, N, F = panel.signals.shape
        if self._model is None:
            return np.zeros((T, N))
        pred = self._model.predict(panel.signals.reshape(T * N, F))
        return np.asarray(pred, dtype=float).reshape(T, N)


class GBDTTopK:
    """GBDT predictor → Top-k long-short book (no RL)."""

    name = "gbdt_topk"

    def __init__(self, *, k: int = 50, leverage: float = 1.0, predictor_kwargs: dict | None = None):
        self.k = k
        self.leverage = leverage
        self.predictor = GBDTPredictor(**(predictor_kwargs or {}))

    def fit(self, panel: Panel) -> "GBDTTopK":
        self.predictor.fit(panel)
        return self

    def predict_weights(self, panel: Panel) -> np.ndarray:
        scores = self.predictor.predict_signal(panel)
        scores = np.where(panel.valid_mask(), scores, np.nan)  # absent names excluded
        return topk_long_short_weights(scores, self.k, leverage=self.leverage)


class GBDTPartialAdjust:
    """GBDT signal -> Gârleanu-Pedersen partial-adjustment book (cost-aware, deterministic).

    The robust, non-overfit answer to the turnover wall: track a continuous aim
    portfolio built from the GBDT score, trading only ``rate`` of the gap each day.
    Sweeping ``rate`` traces the optimal net-Sharpe frontier. No torch -> no
    LightGBM/torch OpenMP conflict.
    """

    def __init__(self, *, rate: float = 0.2, leverage: float = 1.0,
                 predictor_kwargs: dict | None = None):
        self.rate = rate
        self.leverage = leverage
        self.name = f"gbdt_pa{int(round(rate * 100)):02d}"
        self.predictor = GBDTPredictor(**(predictor_kwargs or {}))

    def fit(self, panel: Panel) -> "GBDTPartialAdjust":
        self.predictor.fit(panel)
        return self

    def predict_weights(self, panel: Panel) -> np.ndarray:
        scores = self.predictor.predict_signal(panel)
        scores = np.where(panel.valid_mask(), scores, np.nan)  # absent names excluded
        aim = aim_weights_from_scores(scores, leverage=self.leverage)
        return partial_adjust_path(aim, self.rate, panel.fwd_returns)


class GBDTAllocator:
    """GBDT predictor → RL allocator (the ensemble's portfolio lane on GBDT alpha)."""

    name = "gbdt_rl"

    def __init__(self, *, predictor_kwargs: dict | None = None, allocator_kwargs: dict | None = None):
        self.predictor = GBDTPredictor(**(predictor_kwargs or {}))
        self._alloc_kwargs = allocator_kwargs or {}
        self._alloc = None

    def _signal_panel(self, panel: Panel) -> Panel:
        scores = self.predictor.predict_signal(panel)
        sig = _csrank_norm(scores)[:, :, None]  # (T, N, 1)
        return Panel(panel.dates, panel.symbols, sig, panel.fwd_returns)

    def fit(self, panel: Panel) -> "GBDTAllocator":
        from .eve_rl import DirectRLAllocator

        self.predictor.fit(panel)
        self._alloc = DirectRLAllocator(**self._alloc_kwargs).fit(self._signal_panel(panel))
        return self

    def predict_weights(self, panel: Panel) -> np.ndarray:
        if self._alloc is None:
            return np.zeros(panel.fwd_returns.shape)
        return self._alloc.predict_weights(self._signal_panel(panel))
