"""Elastic Net model wrapper for tabular return prediction."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet

from mlfinance.models.base import r_squared_oos


class ElasticNetModel:
    """Validation-selected Elastic Net wrapper."""

    name = "elastic_net"

    def __init__(
        self,
        alpha_grid: list[float],
        l1_ratio_grid: list[float],
        *,
        max_iter: int = 10000,
        random_state: int = 1,
    ):
        if not alpha_grid or not l1_ratio_grid:
            raise ValueError("alpha_grid and l1_ratio_grid must be non-empty.")
        self.alpha_grid = list(alpha_grid)
        self.l1_ratio_grid = list(l1_ratio_grid)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.model_: ElasticNet | None = None
        self.best_params_: dict[str, float] | None = None
        self.val_scores_: list[dict[str, float]] = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):  # noqa: ANN001
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")
        best_model = None
        best_score = -np.inf
        best_params = None
        self.val_scores_ = []
        for alpha in self.alpha_grid:
            for l1_ratio in self.l1_ratio_grid:
                model = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                model.fit(X_train, y_train)
                if X_val is not None:
                    pred = model.predict(X_val)
                    score = r_squared_oos(y_val, pred)
                else:
                    score = 0.0
                rec = {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "val_oos_r2": float(score),
                }
                self.val_scores_.append(rec)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = rec
        if best_model is None or best_params is None:
            raise RuntimeError("ElasticNetModel failed to fit any model.")
        self.model_ = best_model
        self.best_params_ = best_params
        return self

    def predict(self, X):  # noqa: ANN001
        if self.model_ is None:
            raise RuntimeError("ElasticNetModel.predict called before fit.")
        return self.model_.predict(X)
