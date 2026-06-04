"""Tests for CRSP return-feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfinance.features.return_features import add_return_features


def test_add_return_features_aligns_lags_momentum_and_targets() -> None:
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    raw = pd.DataFrame(
        {
            "permno": 10001,
            "date": dates,
            "ret": np.linspace(0.01, 0.15, num=len(dates)),
            "sprtrn": np.linspace(0.001, 0.015, num=len(dates)),
        }
    )

    features = add_return_features(raw)
    row = features.loc[features["date"] == pd.Timestamp("2021-02-28")].iloc[0]

    assert row["ret_1m"] == raw.loc[raw["date"] == pd.Timestamp("2021-02-28"), "ret"].iloc[0]
    assert row["ret_2m"] == raw.loc[raw["date"] == pd.Timestamp("2021-01-31"), "ret"].iloc[0]
    assert row["ret_3m"] == raw.loc[raw["date"] == pd.Timestamp("2020-12-31"), "ret"].iloc[0]

    expected_mom_2_12 = float(np.prod(1.0 + raw.iloc[1:12]["ret"].to_numpy()) - 1.0)
    expected_mom_1_12 = float(np.prod(1.0 + raw.iloc[1:14]["ret"].to_numpy()) - 1.0)
    # The vectorized cumulative-return implementation uses log1p+sum+expm1
    # instead of prod(1+r)-1; the two differ by ~1 ULP (~1e-15 relative), well
    # below any economically meaningful precision.
    assert row["mom_2_12"] == pytest.approx(expected_mom_2_12, rel=1e-12)
    assert row["mom_1_12"] == pytest.approx(expected_mom_1_12, rel=1e-12)

    expected_target = raw.loc[raw["date"] == pd.Timestamp("2021-03-31"), "ret"].iloc[0]
    expected_target_excess = (
        expected_target - raw.loc[raw["date"] == pd.Timestamp("2021-03-31"), "sprtrn"].iloc[0]
    )
    assert row["target_ret_fwd_1m"] == expected_target
    assert row["target_excess_ret_fwd_1m"] == expected_target_excess


def test_add_return_features_never_uses_future_stock_returns_in_lagged_columns() -> None:
    dates = pd.date_range("2021-01-31", periods=6, freq="ME")
    raw = pd.DataFrame(
        {
            "permno": [10001] * len(dates),
            "date": dates,
            "ret": [0.10, 0.20, 0.30, 0.40, 9.99, 0.60],
            "sprtrn": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        }
    )

    features = add_return_features(raw)
    row = features.loc[features["date"] == pd.Timestamp("2021-04-30")].iloc[0]

    assert row["ret_1m"] == 0.40
    assert row["ret_2m"] == 0.30
    assert row["ret_3m"] == 0.20
    assert row["target_ret_fwd_1m"] == 9.99


def test_add_return_features_respects_missing_stock_months() -> None:
    raw = pd.DataFrame(
        {
            "permno": [10001, 10001, 10001],
            "date": pd.to_datetime(["2020-01-31", "2020-03-31", "2020-04-30"]),
            "ret": [0.10, 0.30, 0.40],
            "sprtrn": [0.01, 0.03, 0.04],
        }
    )

    features = add_return_features(raw).sort_values("date").reset_index(drop=True)
    january = features.loc[features["date"] == pd.Timestamp("2020-01-31")].iloc[0]
    march = features.loc[features["date"] == pd.Timestamp("2020-03-31")].iloc[0]
    april = features.loc[features["date"] == pd.Timestamp("2020-04-30")].iloc[0]

    assert pd.isna(january["target_ret_fwd_1m"])
    assert pd.isna(january["target_excess_ret_fwd_1m"])

    assert march["target_ret_fwd_1m"] == 0.40
    assert march["target_excess_ret_fwd_1m"] == 0.40 - 0.04

    assert pd.isna(march["ret_2m"])
    assert march["ret_3m"] == 0.10
    assert pd.isna(march["vol_3m"])

    assert april["ret_2m"] == 0.30
    assert pd.isna(april["ret_3m"])


def test_add_return_features_collapses_exact_duplicate_rows() -> None:
    raw = pd.DataFrame(
        {
            "permno": [10001, 10001, 10001],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29"]),
            "ret": [0.10, 0.10, 0.20],
            "sprtrn": [0.01, 0.01, 0.02],
            "ticker": ["AAA", "AAA", "AAA"],
            "permco": [1, 1, 1],
        }
    )

    features = add_return_features(raw).sort_values("date").reset_index(drop=True)

    assert len(features) == 2
    assert list(features["date"]) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    january = features.loc[features["date"] == pd.Timestamp("2020-01-31")].iloc[0]
    assert january["target_ret_fwd_1m"] == 0.20


def test_add_return_features_collapses_identifier_only_duplicate_differences() -> None:
    raw = pd.DataFrame(
        {
            "permno": [10001, 10001, 10001],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29"]),
            "ret": [0.10, 0.10, 0.20],
            "sprtrn": [0.01, 0.01, 0.02],
            "ticker": ["AAA", "BBB", "CCC"],
            "siccd": [1111, 2222, 3333],
            "naics": [4444, 5555, 6666],
        }
    )

    features = add_return_features(raw).sort_values("date").reset_index(drop=True)

    assert len(features) == 2
    january = features.loc[features["date"] == pd.Timestamp("2020-01-31")].iloc[0]
    assert january["ticker"] == "AAA"
    assert january["siccd"] == 1111
    assert january["naics"] == 4444
    assert january["target_ret_fwd_1m"] == 0.20


def test_add_return_features_raises_on_conflicting_duplicate_ret_rows() -> None:
    raw = pd.DataFrame(
        {
            "permno": [10001, 10001, 10001],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29"]),
            "ret": [0.10, 0.99, 0.20],
            "sprtrn": [0.01, 0.01, 0.02],
            "ticker": ["AAA", "AAA", "AAA"],
            "permco": [1, 1, 1],
        }
    )

    with np.testing.assert_raises_regex(
        ValueError,
        r"Conflicting duplicate \(permno, date\) groups.*1 groups.*10001.*2020-01-31",
    ):
        add_return_features(raw)


def test_add_return_features_raises_on_conflicting_duplicate_sprtrn_rows() -> None:
    raw = pd.DataFrame(
        {
            "permno": [10001, 10001, 10001],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-29"]),
            "ret": [0.10, 0.10, 0.20],
            "sprtrn": [0.01, 0.99, 0.02],
            "ticker": ["AAA", "BBB", "CCC"],
            "permco": [1, 2, 1],
        }
    )

    with np.testing.assert_raises_regex(
        ValueError,
        r"Conflicting duplicate \(permno, date\) groups.*1 groups.*10001.*2020-01-31",
    ):
        add_return_features(raw)


def test_momentum_preserves_missing_month_nan_semantics() -> None:
    dates = pd.date_range("2020-01-31", periods=16, freq="ME")
    raw = pd.DataFrame(
        {
            "permno": [1] * len(dates),
            "date": dates,
            "ret": [0.01] * len(dates),
            "sprtrn": [0.001] * len(dates),
        }
    )

    # Remove an internal observed month. Calendar expansion will insert it with
    # missing stock return. The old rolling.apply(_cumret) path produced NaN for
    # momentum windows containing that missing month; the vectorized path must
    # preserve that behavior rather than treating the missing return as zero.
    raw = raw.loc[raw["date"] != pd.Timestamp("2020-07-31")].reset_index(drop=True)

    out = add_return_features(raw)
    row = out.loc[out["date"] == pd.Timestamp("2021-02-28")].iloc[0]

    assert pd.isna(row["mom_2_12"])
    assert pd.isna(row["mom_1_12"])
    assert row["target_ret_fwd_1m"] == 0.01
