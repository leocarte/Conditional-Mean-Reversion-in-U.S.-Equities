"""Tests for the walk-forward expanding-window CV utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfinance.cv.walk_forward import _cv_folds_compatible, expanding_window_folds


def _monthly_panel_dates(start: str, end: str, stocks_per_month: int = 20) -> pd.Series:
    months = pd.date_range(start=start, end=end, freq="ME")
    dates = np.repeat(months.values, stocks_per_month)
    return pd.Series(pd.to_datetime(dates))


def test_expanding_window_no_overlap():
    """Masks disjoint, val/test after train, training window strictly expands."""
    dates = _monthly_panel_dates("1957-01-31", "2024-12-31")
    folds = expanding_window_folds(
        dates=dates,
        initial_train_end="1986-12-31",
        n_folds=5,
    )
    assert len(folds) == 5

    prev_train_end = None
    for fold in folds:
        tr, va, te = fold["train_mask"], fold["val_mask"], fold["test_mask"]
        assert not np.any(tr & va), f"train ∩ val non-empty for fold {fold['fold_index']}"
        assert not np.any(va & te), f"val ∩ test non-empty for fold {fold['fold_index']}"
        assert not np.any(tr & te), f"train ∩ test non-empty for fold {fold['fold_index']}"

        train_max = dates[tr].max()
        val_min = dates[va].min()
        val_max = dates[va].max()
        test_min = dates[te].min()
        assert train_max < val_min
        assert val_max < test_min

        train_end = fold["train_dates"][1]
        if prev_train_end is not None:
            assert train_end > prev_train_end
        prev_train_end = train_end


def test_5_folds_default_windows():
    """Project-blueprint test windows with initial_train_end=1986-12-31, 1957-2024 data."""
    dates = _monthly_panel_dates("1957-01-31", "2024-12-31")
    folds = expanding_window_folds(
        dates=dates,
        initial_train_end="1986-12-31",
        n_folds=5,
    )
    expected_test_windows = [
        ("1987-01-31", "1998-12-31"),
        ("1999-01-31", "2010-12-31"),
        ("2011-01-31", "2018-12-31"),
        ("2019-01-31", "2022-12-31"),
        ("2023-01-31", "2024-12-31"),
    ]
    for fold, (exp_start, exp_end) in zip(folds, expected_test_windows, strict=False):
        test_start, test_end = fold["test_dates"]
        assert test_start == pd.to_datetime(
            exp_start
        ), f"fold {fold['fold_index']} test_start {test_start} != {exp_start}"
        assert test_end == pd.to_datetime(
            exp_end
        ), f"fold {fold['fold_index']} test_end {test_end} != {exp_end}"

    min_dt = pd.to_datetime("1957-01-31")
    for fold in folds:
        assert fold["train_dates"][0] == min_dt


def test_expanding_window_rejects_bad_args():
    dates = _monthly_panel_dates("2000-01-31", "2005-12-31")
    with pytest.raises(ValueError):
        expanding_window_folds(dates=dates, initial_train_end="2000-12-31", n_folds=0)
    with pytest.raises(ValueError):
        expanding_window_folds(
            dates=pd.Series([], dtype="datetime64[ns]"),
            initial_train_end="2000-12-31",
        )


def test_expanding_window_generic_schedule_outside_blueprint():
    """Non-default initial_train_end triggers generic horizon arithmetic."""
    dates = _monthly_panel_dates("2000-01-31", "2010-12-31")
    folds = expanding_window_folds(
        dates=dates,
        initial_train_end="2003-12-31",
        val_horizon_months=12,
        test_horizon_months=12,
        n_folds=3,
    )
    assert len(folds) == 3
    assert folds[0]["val_dates"][0] == pd.to_datetime("2004-01-01")
    assert folds[0]["val_dates"][1] == pd.to_datetime("2004-12-31")
    assert folds[0]["test_dates"][0] == pd.to_datetime("2005-01-01")
    assert folds[0]["test_dates"][1] == pd.to_datetime("2005-12-31")
    assert folds[1]["train_dates"][1] == pd.to_datetime("2005-12-31")


def test_cv_folds_compatible_disjoint_and_chronological():
    months = np.array([200001 + m for m in range(60)])
    rows = np.tile(months, 5)
    folds = list(_cv_folds_compatible(rows, n_folds=5))
    assert len(folds) == 5
    for tr, va in folds:
        assert not np.any(tr & va)
        tr_months = np.unique(rows[tr])
        va_months = np.unique(rows[va])
        assert tr_months.max() < va_months.min()
    sizes = [tr.sum() for tr, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)


def test_cv_folds_compatible_too_few_months():
    with pytest.raises(ValueError):
        list(_cv_folds_compatible(np.array([200001, 200002]), n_folds=5))
