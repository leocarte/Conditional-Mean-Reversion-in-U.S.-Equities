"""Out-of-sample evaluation metrics (Campbell-Thompson / GKX convention)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = [
    "oos_r_squared",
    "fold_aggregated_r2",
    "feature_target_corr",
]


def oos_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Campbell-Thompson / GKX OOS R²: ``1 - sum((y-y_hat)^2) / sum(y^2)``.

    Denominator is the raw second moment, NOT the demeaned SS. That makes it
    the "predict zero" benchmark appropriate for excess returns.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} must match.")
    if y_true.size == 0:
        return float("nan")
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        raise ValueError("y_true and y_pred must be finite.")

    sse = float(np.sum((y_true - y_pred) ** 2))
    denom = float(np.sum(y_true**2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - sse / denom


def fold_aggregated_r2(per_fold_results: list[dict]) -> dict:
    """Aggregate OOS R² across folds: pooled and equal-weighted mean +/- std."""
    if not per_fold_results:
        raise ValueError("per_fold_results is empty.")

    per_fold_r2: list[float] = []
    y_true_pool: list[np.ndarray] = []
    y_pred_pool: list[np.ndarray] = []
    for entry in per_fold_results:
        y_t = np.asarray(entry["y_true"], dtype=np.float64).ravel()
        y_p = np.asarray(entry["y_pred"], dtype=np.float64).ravel()
        per_fold_r2.append(oos_r_squared(y_t, y_p))
        y_true_pool.append(y_t)
        y_pred_pool.append(y_p)

    y_true_all = np.concatenate(y_true_pool)
    y_pred_all = np.concatenate(y_pred_pool)
    pooled = oos_r_squared(y_true_all, y_pred_all)
    arr = np.array(per_fold_r2, dtype=np.float64)
    return {
        "pooled_r2": pooled,
        "mean_r2": float(np.mean(arr)),
        "std_r2": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
        "n_folds": int(arr.size),
        "per_fold_r2": per_fold_r2,
    }


def feature_target_corr(
    features: pd.DataFrame,
    target: pd.Series,
    method: str = "spearman",
) -> pd.Series:
    """Per-feature correlation with the target. Defaults to Spearman for long-tailed inputs."""
    supported: Iterable[str] = ("spearman", "pearson", "kendall")
    if method not in supported:
        raise ValueError(f"method must be one of {tuple(supported)}, got {method!r}.")
    if len(features) != len(target):
        raise ValueError(f"features ({len(features)}) and target ({len(target)}) length mismatch.")
    target = pd.Series(target).reset_index(drop=True)
    features = features.reset_index(drop=True)

    out = features.apply(lambda col: col.corr(target, method=method))
    out.name = f"corr_{method}"
    return out.sort_values(ascending=False)
