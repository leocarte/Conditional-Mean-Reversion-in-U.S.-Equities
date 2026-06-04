"""Cross-validation fold audits: disjointness, expansion, contiguity."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from mlfinance.audit.leakage import LeakageAuditError

__all__ = [
    "audit_cv_disjointness",
    "audit_no_inner_shuffle",
    "audit_walk_forward_expansion",
]


def _coerce_mask(arr: Any, n_expected: int | None = None) -> np.ndarray:
    out = np.asarray(arr, dtype=bool).ravel()
    if n_expected is not None and out.size != n_expected:
        raise ValueError(f"Mask length {out.size} != expected {n_expected}.")
    return out


def _dates_for_mask(
    dates: pd.Series | np.ndarray | None, mask: np.ndarray
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if dates is None:
        return (None, None)
    arr = pd.to_datetime(np.asarray(dates))
    if mask.sum() == 0:
        return (None, None)
    sub = arr[mask]
    return (pd.Timestamp(sub.min()), pd.Timestamp(sub.max()))


def _is_contiguous(mask: np.ndarray, dates: pd.Series | np.ndarray | None) -> bool:
    """Check mask selects a contiguous block of unique dates.

    Each unique date must be fully in or fully out, and selected dates
    must form an uninterrupted run between min and max. Falls back to
    row-index contiguity when dates is None.
    """
    if mask.sum() == 0:
        return True

    if dates is None:
        idx = np.flatnonzero(mask)
        return bool((idx[-1] - idx[0] + 1) == len(idx))

    dt = pd.to_datetime(np.asarray(dates))
    df = pd.DataFrame({"d": dt, "m": mask})
    per_date = df.groupby("d", observed=True)["m"].agg(n="size", k="sum")
    all_in = per_date["k"] == per_date["n"]
    none_in = per_date["k"] == 0
    if not (all_in | none_in).all():
        return False

    selected = per_date.index[all_in].sort_values()
    if len(selected) == 0:
        return True
    all_dates_sorted = pd.Index(sorted(per_date.index))
    start_pos = all_dates_sorted.get_loc(selected[0])
    end_pos = all_dates_sorted.get_loc(selected[-1])
    return bool(all_dates_sorted[start_pos : end_pos + 1].equals(selected))


def audit_cv_disjointness(
    folds: Sequence[dict],
    dates: pd.Series | np.ndarray | None = None,
    raise_on_fail: bool = False,
) -> dict:
    """Assert pairwise mask disjointness and train < val < test date ordering."""
    violations: list[dict] = []
    for k, fold in enumerate(folds, start=1):
        fold_idx = int(fold.get("fold_index", k))
        tr = _coerce_mask(fold["train_mask"])
        va = _coerce_mask(fold["val_mask"], n_expected=tr.size)
        te = _coerce_mask(fold["test_mask"], n_expected=tr.size)

        if np.any(tr & va):
            violations.append(
                {
                    "fold_index": fold_idx,
                    "type": "train_val_overlap",
                    "n_overlap": int(np.sum(tr & va)),
                }
            )
        if np.any(tr & te):
            violations.append(
                {
                    "fold_index": fold_idx,
                    "type": "train_test_overlap",
                    "n_overlap": int(np.sum(tr & te)),
                }
            )
        if np.any(va & te):
            violations.append(
                {
                    "fold_index": fold_idx,
                    "type": "val_test_overlap",
                    "n_overlap": int(np.sum(va & te)),
                }
            )

        if dates is not None:
            tr_min, tr_max = _dates_for_mask(dates, tr)
            va_min, va_max = _dates_for_mask(dates, va)
            te_min, te_max = _dates_for_mask(dates, te)
            if tr_max is not None and va_min is not None and tr_max >= va_min:
                violations.append(
                    {
                        "fold_index": fold_idx,
                        "type": "train_after_val_start",
                        "train_max": str(tr_max.date()),
                        "val_min": str(va_min.date()),
                    }
                )
            if va_max is not None and te_min is not None and va_max >= te_min:
                violations.append(
                    {
                        "fold_index": fold_idx,
                        "type": "val_after_test_start",
                        "val_max": str(va_max.date()),
                        "test_min": str(te_min.date()),
                    }
                )

    out = {
        "verdict": "FAIL" if violations else "PASS",
        "n_violations": len(violations),
        "details": violations,
    }
    if violations and raise_on_fail:
        raise LeakageAuditError(
            f"audit_cv_disjointness found {len(violations)} violations: " f"{violations}"
        )
    return out


def audit_walk_forward_expansion(
    folds: Sequence[dict],
    dates: pd.Series | np.ndarray | None = None,
    raise_on_fail: bool = False,
) -> dict:
    """Assert each fold's train_mask strictly expands relative to the previous."""
    if len(folds) < 2:
        return {
            "verdict": "PASS",
            "n_violations": 0,
            "details": [],
        }

    violations: list[dict] = []
    prev_mask: np.ndarray | None = None
    prev_train_max: pd.Timestamp | None = None
    for k, fold in enumerate(folds, start=1):
        fold_idx = int(fold.get("fold_index", k))
        cur = _coerce_mask(fold["train_mask"])

        if prev_mask is not None:
            shrank = prev_mask & ~cur
            if shrank.any():
                violations.append(
                    {
                        "fold_index": fold_idx,
                        "type": "train_lost_rows",
                        "n_rows_dropped": int(shrank.sum()),
                    }
                )
            if int(cur.sum()) <= int(prev_mask.sum()):
                violations.append(
                    {
                        "fold_index": fold_idx,
                        "type": "train_did_not_grow",
                        "n_prev": int(prev_mask.sum()),
                        "n_cur": int(cur.sum()),
                    }
                )

            if dates is not None:
                _, cur_max = _dates_for_mask(dates, cur)
                if prev_train_max is not None and cur_max is not None and cur_max <= prev_train_max:
                    violations.append(
                        {
                            "fold_index": fold_idx,
                            "type": "train_end_did_not_advance",
                            "prev_train_max": str(prev_train_max.date()),
                            "cur_train_max": str(cur_max.date()),
                        }
                    )

        if dates is not None:
            _, prev_train_max = _dates_for_mask(dates, cur)
        prev_mask = cur

    out = {
        "verdict": "FAIL" if violations else "PASS",
        "n_violations": len(violations),
        "details": violations,
    }
    if violations and raise_on_fail:
        raise LeakageAuditError(
            f"audit_walk_forward_expansion found {len(violations)} " f"violations: {violations}"
        )
    return out


def audit_no_inner_shuffle(
    folds: Sequence[dict],
    dates: pd.Series | np.ndarray | None = None,
    raise_on_fail: bool = False,
) -> dict:
    """Assert each mask selects a contiguous time window (no shuffling)."""
    violations: list[dict] = []
    for k, fold in enumerate(folds, start=1):
        fold_idx = int(fold.get("fold_index", k))
        for split in ("train_mask", "val_mask", "test_mask"):
            mask = _coerce_mask(fold[split])
            if not _is_contiguous(mask, dates):
                violations.append(
                    {
                        "fold_index": fold_idx,
                        "split": split,
                        "type": "non_contiguous_or_partial_date",
                    }
                )

    out = {
        "verdict": "FAIL" if violations else "PASS",
        "n_violations": len(violations),
        "details": violations,
    }
    if violations and raise_on_fail:
        raise LeakageAuditError(
            f"audit_no_inner_shuffle found {len(violations)} non-contiguous " f"masks: {violations}"
        )
    return out
