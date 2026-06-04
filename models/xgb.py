"""XGBoost and LightGBM wrappers sharing the BaseModel fit/predict signature."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - dependency declared in pyproject.toml
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore[assignment]


class XGBoostModel:
    """XGBoost regressor wrapper with built-in early stopping.

    Uses the 2.x constructor-baked ``early_stopping_rounds`` (the legacy
    ``fit(... early_stopping_rounds=...)`` argument was removed).
    """

    name: str = "xgboost"

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        tree_method: str = "hist",
        device: str = "auto",
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        seed: int = 1,
        **extra_xgb_params: Any,
    ):
        if xgb is None:
            raise ImportError("xgboost is required to instantiate XGBoostModel.")
        self.n_estimators: int = int(n_estimators)
        self.max_depth: int = int(max_depth)
        self.learning_rate: float = float(learning_rate)
        self.early_stopping_rounds: int = int(early_stopping_rounds)
        self.tree_method: str = str(tree_method)
        self.device: str = self._resolve_device(str(device))
        self.subsample: float = float(subsample)
        self.colsample_bytree: float = float(colsample_bytree)
        self.reg_lambda: float = float(reg_lambda)
        self.reg_alpha: float = float(reg_alpha)
        self.seed: int = int(seed)
        self.extra_xgb_params: dict[str, Any] = dict(extra_xgb_params)

        self._model: Any = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Use CUDA if torch detects a GPU; xgboost's PyPI wheel since 2.0 has CUDA built in."""
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _make_model(self, with_early_stopping: bool) -> Any:
        params: dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            tree_method=self.tree_method,
            device=self.device,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.seed,
            objective="reg:squarederror",
            verbosity=0,
        )
        if with_early_stopping:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        params.update(self.extra_xgb_params)
        return xgb.XGBRegressor(**params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> XGBoostModel:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        with_es = X_val is not None
        self._model = self._make_model(with_early_stopping=with_es)
        if with_es:
            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self._model.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoostModel.predict called before fit.")
        return np.asarray(self._model.predict(X)).reshape(-1)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("feature_importances_ queried before fit.")
        return np.asarray(self._model.feature_importances_)


class LightGBMModel:
    """LightGBM mirror of XGBoostModel using the v4+ callbacks API for early stopping."""

    name: str = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        device: str = "cpu",
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        min_child_samples: int = 20,
        seed: int = 1,
        **extra_lgb_params: Any,
    ):
        if lgb is None:
            raise ImportError("lightgbm is required to instantiate LightGBMModel.")
        self.n_estimators: int = int(n_estimators)
        self.max_depth: int = int(max_depth)
        self.num_leaves: int = int(num_leaves)
        self.learning_rate: float = float(learning_rate)
        self.early_stopping_rounds: int = int(early_stopping_rounds)
        self.device: str = str(device)
        self.subsample: float = float(subsample)
        self.colsample_bytree: float = float(colsample_bytree)
        self.reg_lambda: float = float(reg_lambda)
        self.reg_alpha: float = float(reg_alpha)
        self.min_child_samples: int = int(min_child_samples)
        self.seed: int = int(seed)
        self.extra_lgb_params: dict[str, Any] = dict(extra_lgb_params)

        self._model: Any = None

    def _make_model(self) -> Any:
        params: dict[str, Any] = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            min_child_samples=self.min_child_samples,
            device_type=self.device,
            random_state=self.seed,
            objective="regression",
            verbosity=-1,
        )
        params.update(self.extra_lgb_params)
        return lgb.LGBMRegressor(**params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> LightGBMModel:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        self._model = self._make_model()
        callbacks: list[Any] = []
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False)
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if eval_set is not None:
                self._model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    callbacks=callbacks,
                )
            else:
                self._model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LightGBMModel.predict called before fit.")
        return np.asarray(self._model.predict(X)).reshape(-1)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("feature_importances_ queried before fit.")
        return np.asarray(self._model.feature_importances_)
