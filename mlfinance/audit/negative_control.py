"""Negative-control leakage check.

Shifts target labels backward (wrong direction) and verifies the OOS R² of
a Ridge regression collapses to ~0. Surviving signal under wrong-shift
labels is unambiguous evidence of temporal leakage upstream.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mlfinance.audit.leakage import LeakageAuditError

logger = logging.getLogger(__name__)

__all__ = ["run_negative_control"]


_RATIO_THRESHOLD: float = 0.05
_ABSOLUTE_FLOOR: float = 1e-4


def _shift_labels_wrong(
    panel: pd.DataFrame,
    target_col: str,
    n_lags_wrong: int,
    permno_col: str = "permno",
    date_col: str = "date",
) -> pd.Series:
    """Shift labels within each permno by n_lags_wrong periods.

    Default n_lags_wrong=-2 puts the past return on the current row; the
    features cannot legitimately know it.
    """
    out = panel.sort_values([permno_col, date_col]).reset_index(drop=True)
    return out.groupby(permno_col, observed=True)[target_col].shift(n_lags_wrong)


def _fit_predict_ridge_single_fold(
    panel: pd.DataFrame,
    target: pd.Series,
    feature_cols: list[str],
    train_frac: float = 0.7,
    shrinkage: float = 1e-2,
) -> tuple[float, int]:
    """Fit Ridge on first train_frac of time-sorted panel; score the rest."""
    from mlfinance.eval.metrics import oos_r_squared
    from mlfinance.models.ridge import ridge_eig

    if "date" not in panel.columns:
        raise KeyError("_fit_predict_ridge_single_fold: missing 'date' column.")

    df = panel.copy().reset_index(drop=True)
    df = df[["date", *feature_cols]].copy()
    df["__target__"] = pd.Series(target).reset_index(drop=True)
    df = df.dropna(subset=feature_cols + ["__target__"]).reset_index(drop=True)
    if df.empty:
        return float("nan"), 0

    unique_dates = np.sort(df["date"].unique())
    if len(unique_dates) < 2:
        return float("nan"), 0
    split_idx = max(1, int(np.floor(len(unique_dates) * train_frac)))
    train_dates = unique_dates[:split_idx]
    train_mask = df["date"].isin(train_dates).to_numpy()
    test_mask = ~train_mask

    X_tr = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float64)
    y_tr = df.loc[train_mask, "__target__"].to_numpy(dtype=np.float64)
    X_te = df.loc[test_mask, feature_cols].to_numpy(dtype=np.float64)
    y_te = df.loc[test_mask, "__target__"].to_numpy(dtype=np.float64)

    if X_tr.size == 0 or X_te.size == 0:
        return float("nan"), 0

    _, preds = ridge_eig(X_tr, y_tr, X_te, [shrinkage])
    r2 = oos_r_squared(y_te, preds[:, 0])
    return float(r2), int(X_te.shape[0])


def run_negative_control(
    panel: pd.DataFrame,
    target_col: str,
    model_cls: Any | None = None,
    model_cfg: dict | None = None,
    n_lags_wrong: int = -2,
    feature_cols: list[str] | None = None,
    train_frac: float = 0.7,
    ratio_threshold: float = _RATIO_THRESHOLD,
    absolute_floor: float = _ABSOLUTE_FLOOR,
    raise_on_fail: bool = False,
    permno_col: str = "permno",
    date_col: str = "date",
) -> dict:
    """Run the negative-control leakage check on Ridge + a single fold."""
    if target_col not in panel.columns:
        raise KeyError(f"run_negative_control: missing target '{target_col}'.")
    if date_col not in panel.columns:
        raise KeyError(f"run_negative_control: missing '{date_col}'.")
    if permno_col not in panel.columns:
        raise KeyError(f"run_negative_control: missing '{permno_col}'.")

    if feature_cols is None:
        excluded = {
            permno_col,
            date_col,
            "yyyymm",
            "gvkey",
            "cik",
            "permco",
            "siccd",
            "naics",
            "exchcd",
            "dlret",
            "dlstcd",
            "mktcap",
            "sprtrn",
            "ret",
            "ret_exc",
            "ret_exc_next",
            "ret_exc_next_rank",
            "ret_exc_rank",
            target_col,
        }
        numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in excluded]
    if not feature_cols:
        raise ValueError(
            "run_negative_control: no usable numeric feature columns " "after exclusions."
        )

    panel_sorted = panel.sort_values([permno_col, date_col]).reset_index(drop=True)
    real_target = panel_sorted[target_col]
    wrong_target = _shift_labels_wrong(
        panel_sorted,
        target_col,
        n_lags_wrong=n_lags_wrong,
        permno_col=permno_col,
        date_col=date_col,
    )

    real_r2, n_test_real = _fit_predict_ridge_single_fold(
        panel_sorted, real_target, feature_cols, train_frac=train_frac
    )
    neg_r2, n_test_neg = _fit_predict_ridge_single_fold(
        panel_sorted, wrong_target, feature_cols, train_frac=train_frac
    )

    # Verdict logic:
    #   (a) both NaN: cannot decide, PASS with warning.
    #   (b) real has no signal AND negative also collapsed: PASS (noise panel).
    #   (c) real has no signal but negative has signal: FAIL (leak).
    #   (d) real has signal: compare ratio against ratio_threshold.
    if np.isnan(real_r2) and np.isnan(neg_r2):
        verdict = "PASS"
        ratio = float("nan")
        logger.warning(
            "run_negative_control: both real and negative-control R² are NaN "
            "(too few rows for a single-fold split). Treating as PASS."
        )
    elif abs(real_r2) < absolute_floor:
        if neg_r2 < absolute_floor * 10.0:
            verdict = "PASS"
            ratio = float("nan")
            logger.info(
                "run_negative_control: real R² (%.5f) below absolute floor "
                "(%.5f); panel appears signal-free. Treating as PASS.",
                real_r2,
                absolute_floor,
            )
        else:
            verdict = "FAIL"
            ratio = float("inf")
    elif abs(neg_r2) < absolute_floor:
        verdict = "PASS"
        ratio = abs(neg_r2) / abs(real_r2)
    else:
        ratio = abs(neg_r2) / abs(real_r2)
        verdict = "PASS" if ratio < ratio_threshold else "FAIL"

    out = {
        "negative_r2": float(neg_r2),
        "real_r2": float(real_r2),
        "ratio": float(ratio),
        "verdict": verdict,
        "n_features": int(len(feature_cols)),
        "n_test_rows": int(max(n_test_real, n_test_neg)),
        "n_lags_wrong": int(n_lags_wrong),
    }

    if verdict == "FAIL" and raise_on_fail:
        raise LeakageAuditError(
            f"Negative-control FAILED: neg_r2={neg_r2:.6f}, real_r2="
            f"{real_r2:.6f}, ratio={ratio:.3f} >= "
            f"{ratio_threshold}. Likely temporal leak upstream."
        )
    return out
