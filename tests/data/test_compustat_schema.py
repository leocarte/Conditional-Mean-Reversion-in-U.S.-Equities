"""Synthetic tests for the lightweight feature-audit Compustat audit scaffold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlfinance.data.compustat_schema import (
    audit_compustat_point_in_time_merge,
    audit_compustat_schema,
    merge_compustat_point_in_time,
    validate_compustat_schema_or_raise,
)


def _compustat_like_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gvkey": ["001000", "001000", "002000"],
            "datadate": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31"]),
            "fyear": [2020, 2020, 2020],
            "at": [100.0, 110.0, np.nan],
            "seq": [60.0, 61.0, 70.0],
            "txditc": [5.0, np.nan, 2.0],
            "pstk": [1.0, 1.0, np.nan],
            "ib": [10.0, 11.0, 12.0],
            "sale": [50.0, 51.0, 52.0],
            "cogs": [20.0, 21.0, np.nan],
            "dltt": [30.0, 31.0, 32.0],
            "dlc": [3.0, 4.0, 5.0],
            "oancf": [8.0, 9.0, 10.0],
            "capx": [4.0, np.nan, 6.0],
            "cusip": ["12345678", "12345678", None],
            "cik": ["1000", "1000", "2000"],
        }
    )


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_audit_compustat_schema_reads_csv_and_parquet(
    tmp_path: Path,
    extension: str,
) -> None:
    frame = _compustat_like_frame()
    path = tmp_path / f"compustat_like.{extension}"
    if extension == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)

    report = audit_compustat_schema(path)

    assert report.source == str(path)
    assert report.row_count == 3
    assert report.column_count == len(frame.columns)
    assert report.min_datadate == pd.Timestamp("2020-03-31")
    assert report.max_datadate == pd.Timestamp("2020-06-30")
    assert report.parsable_datadate_count == 3


def test_audit_compustat_schema_reports_field_and_merge_key_coverage() -> None:
    report = audit_compustat_schema(_compustat_like_frame())

    field_coverage = report.field_coverage.set_index("field")
    assert bool(field_coverage.loc["seq", "present"]) is True
    assert bool(field_coverage.loc["ceq", "present"]) is False
    assert field_coverage.loc["at", "non_null_count"] == 2
    assert field_coverage.loc["capx", "non_null_count"] == 2

    required = report.required_group_status.set_index("requirement")
    assert bool(required.loc["common_equity", "satisfied"]) is True
    assert required.loc["common_equity", "usable_alternatives"] == ("seq",)
    assert bool(required.loc["preferred_stock", "satisfied"]) is True
    assert required.loc["preferred_stock", "usable_alternatives"] == ("pstk",)
    assert report.unsatisfied_required_groups == []

    merge_keys = report.merge_key_coverage.set_index("field")
    assert bool(merge_keys.loc["gvkey", "present"]) is True
    assert merge_keys.loc["gvkey", "merge_ready_count"] == 3
    assert merge_keys.loc["cusip", "non_null_count"] == 2
    assert bool(merge_keys.loc["cusip8", "present"]) is False


def test_validate_compustat_schema_or_raise_reports_missing_required_groups() -> None:
    incomplete = pd.DataFrame(
        {
            "gvkey": ["001000"],
            "sale": [10.0],
            "cusip": ["12345678"],
        }
    )

    report = audit_compustat_schema(incomplete)
    assert "datadate" in report.unsatisfied_required_groups
    assert "common_equity" in report.unsatisfied_required_groups
    assert "earnings" in report.unsatisfied_required_groups

    with pytest.raises(
        ValueError,
        match=(
            r"Compustat schema audit failed; unsatisfied required groups: "
            r".*datadate \(datadate\).*common_equity \(ceq\|seq\).*earnings \(ni\|ib\)"
        ),
    ):
        validate_compustat_schema_or_raise(incomplete)


def test_merge_compustat_point_in_time_applies_conservative_lag() -> None:
    formation = pd.DataFrame(
        {
            "permno": [10001, 10001],
            "gvkey": ["001000", "001000"],
            "date": pd.to_datetime(["2020-06-30", "2020-10-31"]),
        }
    )
    compustat = pd.DataFrame(
        {
            "gvkey": ["001000"],
            "datadate": [pd.Timestamp("2020-03-31")],
            "at": [100.0],
        }
    )

    merged = merge_compustat_point_in_time(
        formation,
        compustat,
        merge_key="gvkey",
        lag_months=6,
        compustat_cols=["at"],
    )

    assert pd.isna(merged.loc[0, "compustat_at"])
    assert merged.loc[1, "compustat_at"] == 100.0
    assert merged.loc[1, "compustat_datadate"] == pd.Timestamp("2020-03-31")
    assert merged.loc[1, "compustat_availability_date"] == pd.Timestamp("2020-09-30")
    assert audit_compustat_point_in_time_merge(merged).empty


def test_merge_compustat_point_in_time_never_exposes_future_datadate() -> None:
    formation = pd.DataFrame(
        {
            "gvkey": ["001000", "001000"],
            "date": pd.to_datetime(["2020-07-31", "2021-01-31"]),
        }
    )
    compustat = pd.DataFrame(
        {
            "gvkey": ["001000", "001000"],
            "datadate": [pd.Timestamp("2020-03-31"), pd.Timestamp("2020-12-31")],
            "availability_date": [pd.Timestamp("2020-04-30"), pd.Timestamp("2021-01-15")],
            "at": [100.0, 999.0],
        }
    )

    merged = merge_compustat_point_in_time(
        formation,
        compustat,
        merge_key="gvkey",
        lag_months=6,
        compustat_cols=["at"],
    )

    assert pd.isna(merged.loc[0, "compustat_at"])
    assert merged.loc[1, "compustat_at"] == 100.0
    assert merged.loc[1, "compustat_datadate"] == pd.Timestamp("2020-03-31")
    assert merged.loc[1, "compustat_availability_date"] == pd.Timestamp("2020-09-30")


def test_merge_compustat_point_in_time_respects_gvkeys_with_overlapping_dates() -> None:
    formation = pd.DataFrame(
        {
            "permno": [20002, 10001, 20002, 10001],
            "gvkey": ["002000", "001000", "002000", "001000"],
            "date": pd.to_datetime(["2020-10-31", "2020-10-31", "2021-01-31", "2021-01-31"]),
        }
    )
    compustat = pd.DataFrame(
        {
            "gvkey": ["001000", "002000", "002000"],
            "datadate": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-06-30"]),
            "availability_date": pd.to_datetime(["2020-04-15", "2020-04-15", "2020-08-15"]),
            "at": [100.0, 200.0, 220.0],
        }
    )

    merged = merge_compustat_point_in_time(
        formation,
        compustat,
        merge_key="gvkey",
        lag_months=6,
        compustat_cols=["at"],
    )

    expected = pd.DataFrame(
        {
            "gvkey": ["002000", "001000", "002000", "001000"],
            "date": pd.to_datetime(["2020-10-31", "2020-10-31", "2021-01-31", "2021-01-31"]),
            "compustat_at": [200.0, 100.0, 220.0, 100.0],
            "compustat_datadate": pd.to_datetime(
                ["2020-03-31", "2020-03-31", "2020-06-30", "2020-03-31"]
            ),
            "compustat_availability_date": pd.to_datetime(
                ["2020-09-30", "2020-09-30", "2020-12-30", "2020-09-30"]
            ),
        }
    )

    assert merged["gvkey"].tolist() == expected["gvkey"].tolist()
    pd.testing.assert_series_equal(merged["date"], expected["date"], check_names=False)
    pd.testing.assert_series_equal(
        merged["compustat_datadate"], expected["compustat_datadate"], check_names=False
    )
    pd.testing.assert_series_equal(
        merged["compustat_availability_date"],
        expected["compustat_availability_date"],
        check_names=False,
    )
    assert merged["compustat_at"].tolist() == expected["compustat_at"].tolist()
    assert audit_compustat_point_in_time_merge(merged).empty
    matched = merged["compustat_availability_date"].notna()
    assert (merged.loc[matched, "compustat_availability_date"] <= merged.loc[matched, "date"]).all()


def test_audit_compustat_point_in_time_merge_flags_future_rows() -> None:
    merged = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-30"]),
            "compustat_datadate": pd.to_datetime(["2020-07-31"]),
            "compustat_availability_date": pd.to_datetime(["2020-06-30"]),
        }
    )

    violations = audit_compustat_point_in_time_merge(merged)
    assert len(violations) == 1
    assert violations.loc[0, "violation_type"] == "future_datadate"

    with pytest.raises(ValueError, match=r"future_datadate"):
        audit_compustat_point_in_time_merge(merged, raise_on_fail=True)
