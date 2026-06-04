"""Temporal-leakage audits for the model panel and prediction pipeline."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "FeatureProvenance",
    "LeakageAuditError",
    "audit_feature_provenance",
    "audit_prediction_dates",
    "audit_preprocessor_fit_train_only",
    "audit_target_alignment",
]


class FeatureProvenance(NamedTuple):
    """Provenance triple: column, information_date, signal_date."""

    column_name: str
    information_date: pd.Timestamp
    signal_date: pd.Timestamp


class LeakageAuditError(AssertionError):
    """Raised whenever an audit detects a temporal leak."""


# Publication lag in days per family prefix. Lag is the minimum gap between
# information_date and signal_date; 0 means known the same day.
_DEFAULT_LAG_DAYS: dict[str, int] = {
    "jkp_": 30,
    "cz_": 30,
    "compustat_": 120,
    "saleq": 120,
    "atq": 120,
    "niq": 120,
    "ltq": 120,
    "ceqq": 120,
}


def _resolve_lag_days(column: str, overrides: dict[str, int] | None) -> int:
    """Return publication lag for column. Longest prefix wins; default 0."""
    merged: dict[str, int] = dict(_DEFAULT_LAG_DAYS)
    if overrides:
        for k, v in overrides.items():
            normalised = k.rstrip("*")
            merged[normalised] = int(v)

    for prefix in sorted(merged.keys(), key=len, reverse=True):
        if column.startswith(prefix) or column == prefix:
            return int(merged[prefix])
    return 0


def audit_feature_provenance(
    panel: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "date",
    as_of_lag_days: dict[str, int] | None = None,
    raise_on_fail: bool = False,
) -> pd.DataFrame:
    """Check every feature cell satisfies information_date <= signal_date.

    information_date is taken from an embedded ``<col>_availability_date``
    column when present, otherwise derived as ``signal_date - lag_days``
    using the column's family prefix.
    """
    if date_col not in panel.columns:
        raise KeyError(f"audit_feature_provenance: missing '{date_col}'.")

    signal_dates = pd.to_datetime(panel[date_col]).reset_index(drop=True)
    n = len(panel)
    violations_chunks: list[pd.DataFrame] = []

    permno_present = "permno" in panel.columns
    permno = (
        panel["permno"].reset_index(drop=True)
        if permno_present
        else pd.Series(np.full(n, -1), name="permno")
    )

    for col in feature_cols:
        if col not in panel.columns:
            continue
        lag_days = _resolve_lag_days(col, as_of_lag_days)
        lag_offset = pd.Timedelta(days=lag_days)

        embedded = f"{col}_availability_date"
        if embedded in panel.columns:
            info_dates = pd.to_datetime(panel[embedded]).reset_index(drop=True)
        else:
            info_dates = signal_dates - lag_offset

        observed = panel[col].notna().reset_index(drop=True)
        bad_mask = observed & (info_dates > signal_dates)
        if bad_mask.any():
            chunk = pd.DataFrame(
                {
                    "column": col,
                    "row_index": np.where(bad_mask.to_numpy())[0],
                    "permno": permno[bad_mask].to_numpy(),
                    "signal_date": signal_dates[bad_mask].to_numpy(),
                    "information_date": info_dates[bad_mask].to_numpy(),
                    "lag_days": lag_days,
                }
            )
            violations_chunks.append(chunk)

    if violations_chunks:
        violations = pd.concat(violations_chunks, ignore_index=True)
    else:
        violations = pd.DataFrame(
            columns=[
                "column",
                "row_index",
                "permno",
                "signal_date",
                "information_date",
                "lag_days",
            ]
        )

    if raise_on_fail and not violations.empty:
        raise LeakageAuditError(
            f"audit_feature_provenance found {len(violations)} violations "
            f"across {violations['column'].nunique()} columns."
        )
    return violations


def audit_target_alignment(
    panel: pd.DataFrame,
    target_col: str,
    return_col: str = "ret_exc",
    date_col: str = "date",
    permno_col: str = "permno",
    raise_on_fail: bool = False,
) -> pd.DataFrame:
    """Assert target_col equals the next period's return_col within permno.

    If target_col is a monotone transform (e.g. cross-sectional rank), the
    audit compares per-date ranks instead of raw values. Catches the
    off-by-one bug where target == contemporaneous return.
    """
    for c in (target_col, return_col, date_col, permno_col):
        if c not in panel.columns:
            raise KeyError(f"audit_target_alignment: missing column '{c}'.")

    sorted_panel = panel.sort_values([permno_col, date_col]).reset_index(drop=True).copy()
    expected_next = sorted_panel.groupby(permno_col, observed=True)[return_col].shift(-1)

    target_values = sorted_panel[target_col].astype("float64")
    expected_values = expected_next.astype("float64")

    # Detect monotone transforms via median per-date Spearman rho ~ 1.
    both = pd.DataFrame(
        {
            "t": target_values,
            "x": expected_values,
            "d": sorted_panel[date_col],
        }
    ).dropna(subset=["t", "x"])

    is_monotone = False
    if not both.empty:

        def _rho(s: pd.DataFrame) -> float:
            if len(s) < 3:
                return 1.0
            return float(s["t"].corr(s["x"], method="spearman"))

        rhos = both.groupby("d", observed=True).apply(_rho, include_groups=False)
        is_monotone = bool(np.nanmedian(rhos.values) >= 0.999)

    if is_monotone:
        target_rank = sorted_panel.groupby(date_col, observed=True)[target_col].rank(
            pct=True, method="average"
        )
        expected_rank = (
            sorted_panel.assign(_x=expected_next)
            .groupby(date_col, observed=True)["_x"]
            .rank(pct=True, method="average")
        )
        diff = (target_rank - expected_rank).abs()
        tol = 1e-6
    else:
        diff = (target_values - expected_values).abs()
        tol = 1e-9

    # Last row per permno has expected_next == NaN; skip.
    has_expected = expected_next.notna()
    bad_mask = (diff > tol) & has_expected

    if bad_mask.any():
        violations = pd.DataFrame(
            {
                "row_index": np.where(bad_mask.to_numpy())[0],
                "permno": sorted_panel.loc[bad_mask, permno_col].to_numpy(),
                "date": sorted_panel.loc[bad_mask, date_col].to_numpy(),
                "target_value": target_values[bad_mask].to_numpy(),
                "expected_next_return": expected_values[bad_mask].to_numpy(),
                "diff": diff[bad_mask].to_numpy(),
            }
        )
    else:
        violations = pd.DataFrame(
            columns=[
                "row_index",
                "permno",
                "date",
                "target_value",
                "expected_next_return",
                "diff",
            ]
        )

    if raise_on_fail and not violations.empty:
        raise LeakageAuditError(
            f"audit_target_alignment found {len(violations)} misaligned rows "
            f"(target '{target_col}' != next({return_col}))."
        )
    return violations


def audit_preprocessor_fit_train_only(
    preprocessor: Any,
    train_mask: np.ndarray,
    panel: pd.DataFrame,
    raise_on_fail: bool = True,
) -> bool:
    """Assert preprocessor.fit saw only rows in train_mask.

    Looks at ``_audit_n_fit_samples`` (set by :func:`fit_with_audit`) or
    sklearn's ``n_samples_seen_``. Fails closed when neither is exposed.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape[0] != len(panel):
        raise ValueError(
            f"train_mask length {train_mask.shape[0]} != panel length " f"{len(panel)}."
        )
    expected_n = int(train_mask.sum())

    seen_n: int | None = None
    for attr in ("_audit_n_fit_samples", "n_samples_seen_"):
        if hasattr(preprocessor, attr):
            val = getattr(preprocessor, attr)
            # SimpleImputer stores per-feature; take the max.
            seen_n = int(np.max(val)) if isinstance(val, np.ndarray) else int(val)
            break

    if seen_n is None:
        msg = (
            "audit_preprocessor_fit_train_only: preprocessor "
            f"{type(preprocessor).__name__} does not expose a recognised "
            "fit-sample-count attribute. Wrap your fit call with "
            "fit_with_audit(...) to make the audit deterministic."
        )
        if raise_on_fail:
            raise LeakageAuditError(msg)
        return False

    ok = seen_n == expected_n
    if not ok and raise_on_fail:
        raise LeakageAuditError(
            "audit_preprocessor_fit_train_only: "
            f"preprocessor saw {seen_n} samples but train_mask has "
            f"{expected_n}. Did .fit get the full panel by mistake?"
        )
    return ok


def fit_with_audit(
    preprocessor: Any,
    X: np.ndarray | pd.DataFrame,
    train_mask: np.ndarray,
    y: np.ndarray | pd.Series | None = None,
) -> Any:
    """Fit preprocessor on X[train_mask] and stamp _audit_n_fit_samples."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if isinstance(X, pd.DataFrame):
        X_train = X.loc[train_mask].to_numpy()
    else:
        X_train = np.asarray(X)[train_mask]

    if y is None:
        preprocessor.fit(X_train)
    else:
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
        preprocessor.fit(X_train, y_arr[train_mask])

    preprocessor._audit_n_fit_samples = int(X_train.shape[0])
    return preprocessor


def audit_prediction_dates(
    predictions: pd.DataFrame,
    signal_date_col: str = "date",
    return_date_col: str = "return_date",
    raise_on_fail: bool = False,
) -> pd.DataFrame:
    """Assert signal_date < return_date for every prediction row."""
    for c in (signal_date_col, return_date_col):
        if c not in predictions.columns:
            raise KeyError(f"audit_prediction_dates: missing column '{c}'.")

    sig = pd.to_datetime(predictions[signal_date_col]).reset_index(drop=True)
    ret = pd.to_datetime(predictions[return_date_col]).reset_index(drop=True)
    bad_mask = sig >= ret

    if bad_mask.any():
        gap = (ret - sig).dt.days
        violations = pd.DataFrame(
            {
                "row_index": np.where(bad_mask.to_numpy())[0],
                "signal_date": sig[bad_mask].to_numpy(),
                "return_date": ret[bad_mask].to_numpy(),
                "gap_days": gap[bad_mask].to_numpy(),
            }
        )
    else:
        violations = pd.DataFrame(columns=["row_index", "signal_date", "return_date", "gap_days"])

    if raise_on_fail and not violations.empty:
        raise LeakageAuditError(
            f"audit_prediction_dates found {len(violations)} rows where "
            "signal_date >= return_date."
        )
    return violations
