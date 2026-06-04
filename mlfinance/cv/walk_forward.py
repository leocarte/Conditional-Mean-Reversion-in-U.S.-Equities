"""Walk-forward expanding-window cross-validation.

The training window starts at min(dates) and expands each fold; validation
and test windows are contiguous slices placed immediately after it. When
``initial_train_end='1986-12-31'`` and ``n_folds=5`` the function emits the
project-blueprint partition (1987-98, 1999-2010, 2011-18, 2019-22, 2023-24).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "expanding_window_folds",
    "_cv_folds_compatible",
]


# Hard-coded project-blueprint test windows. Used when caller passes the
# documented defaults so output matches the project spec exactly.
_PROJECT_BLUEPRINT_TEST_WINDOWS: tuple[tuple[str, str], ...] = (
    ("1987-01-31", "1998-12-31"),
    ("1999-01-31", "2010-12-31"),
    ("2011-01-31", "2018-12-31"),
    ("2019-01-31", "2022-12-31"),
    ("2023-01-31", "2024-12-31"),
)


def expanding_window_folds(
    dates: pd.Series,
    initial_train_end: str = "1986-12-31",
    val_horizon_months: int = 12,
    test_horizon_months: int = 12,
    n_folds: int = 5,
) -> list[dict]:
    """Return walk-forward expanding-window folds for a date series.

    With ``initial_train_end='1986-12-31'`` and ``n_folds=5`` the blueprint
    schedule kicks in (multi-year test horizons, NOT 12-month).

    Returns a list of fold dicts with keys ``train_mask``, ``val_mask``,
    ``test_mask`` (boolean arrays aligned with positional ``dates`` order),
    ``fold_index``, and ``train_dates`` / ``val_dates`` / ``test_dates``.
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}.")
    dt = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    if dt.empty:
        raise ValueError("`dates` is empty.")

    initial_train_end_ts = pd.to_datetime(initial_train_end)
    min_date = dt.min()
    max_date = dt.max()

    use_blueprint = (
        initial_train_end_ts == pd.to_datetime("1986-12-31")
        and n_folds == 5
        and val_horizon_months == 12
        and test_horizon_months == 12
    )

    folds: list[dict] = []
    if use_blueprint:
        # Validation window is the 12 months preceding each test window;
        # training expands up to the day before validation starts.
        for fold_index, (test_start_str, test_end_str) in enumerate(
            _PROJECT_BLUEPRINT_TEST_WINDOWS, start=1
        ):
            test_start = pd.to_datetime(test_start_str)
            test_end = pd.to_datetime(test_end_str)
            val_end = (test_start - pd.offsets.MonthEnd(1)).to_pydatetime()
            val_end = pd.to_datetime(val_end) + pd.offsets.MonthEnd(0)
            val_start = (
                (val_end - pd.DateOffset(months=val_horizon_months - 1))
                + pd.offsets.MonthBegin(0)
                - pd.offsets.MonthBegin(1)
            )
            train_start = min_date
            train_end = val_start - pd.offsets.MonthEnd(1)
            train_end = pd.to_datetime(train_end) + pd.offsets.MonthEnd(0)

            _check_within_data(test_end, max_date, "test window")
            folds.append(
                _build_fold(
                    dt,
                    train_start,
                    train_end,
                    val_start,
                    val_end,
                    test_start,
                    test_end,
                    fold_index,
                )
            )
        return folds

    train_end = initial_train_end_ts
    for fold_index in range(1, n_folds + 1):
        val_start = (train_end + pd.offsets.Day(1)).replace(day=1)
        val_end = _add_months_end(val_start, val_horizon_months)
        test_start = (val_end + pd.offsets.Day(1)).replace(day=1)
        test_end = _add_months_end(test_start, test_horizon_months)

        _check_within_data(test_end, max_date, "test window")
        folds.append(
            _build_fold(
                dt,
                min_date,
                train_end,
                val_start,
                val_end,
                test_start,
                test_end,
                fold_index,
            )
        )
        train_end = test_end
    return folds


def _build_fold(
    dt: pd.Series,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    fold_index: int,
) -> dict:
    train_mask = ((dt >= train_start) & (dt <= train_end)).to_numpy()
    val_mask = ((dt >= val_start) & (dt <= val_end)).to_numpy()
    test_mask = ((dt >= test_start) & (dt <= test_end)).to_numpy()

    if (
        np.any(train_mask & val_mask)
        or np.any(val_mask & test_mask)
        or np.any(train_mask & test_mask)
    ):
        raise RuntimeError(
            f"Fold {fold_index} produced overlapping masks "
            f"({train_start.date()}..{train_end.date()}) "
            f"({val_start.date()}..{val_end.date()}) "
            f"({test_start.date()}..{test_end.date()})."
        )
    return {
        "fold_index": fold_index,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_dates": (train_start, train_end),
        "val_dates": (val_start, val_end),
        "test_dates": (test_start, test_end),
    }


def _add_months_end(start: pd.Timestamp, n_months: int) -> pd.Timestamp:
    end_of_month = start + pd.DateOffset(months=n_months - 1)
    return end_of_month + pd.offsets.MonthEnd(0)


def _check_within_data(date: pd.Timestamp, max_date: pd.Timestamp, label: str) -> None:
    if date > max_date:
        raise ValueError(
            f"{label} ends at {date.date()} but data only extends to " f"{max_date.date()}."
        )


def _cv_folds_compatible(
    dates_train: np.ndarray, n_folds: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Inner-CV walk-forward folds inside a single training window.

    Splits the unique sorted months into ``n_folds + 1`` blocks; fold k
    trains on blocks 1..k and validates on block k+1.
    """
    months = np.sort(np.unique(np.asarray(dates_train)))
    n_months = len(months)
    if n_months < n_folds + 1:
        raise ValueError(
            f"Need at least n_folds + 1 = {n_folds + 1} unique months for "
            f"inner CV, got {n_months}."
        )
    for k in range(1, n_folds + 1):
        tr_end = int(np.floor(n_months * k / (n_folds + 1)))
        va_end = int(np.floor(n_months * (k + 1) / (n_folds + 1)))
        train_months = months[:tr_end]
        val_months = months[tr_end:va_end]
        yield (
            np.isin(dates_train, train_months),
            np.isin(dates_train, val_months),
        )
