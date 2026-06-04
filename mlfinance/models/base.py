"""Base model protocol and shared evaluation helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseModel(Protocol):
    """Structural protocol every model wrapper satisfies."""

    name: str

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaseModel:
        """Fit in-place and return self. X_val/y_val must be passed together."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return one scalar prediction per row of X."""
        ...


def r_squared_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """GKX OOS R²: ``1 - MSE / E[y^2]`` (raw second moment, NOT variance).

    Punishes models that fail to beat the predict-zero benchmark. This is the
    standard convention for monthly excess returns in asset pricing.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape."
        )
    if y_true.size == 0:
        return float("nan")

    mse = float(np.mean((y_true - y_pred) ** 2))
    denom = float(np.mean(y_true**2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - mse / denom
