"""Synthetic tests for the reviewed flow-only Compustat feature scaffold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlfinance.data.compustat_schema import merge_compustat_point_in_time
from mlfinance.features.compustat_flow_features import (
    build_compustat_flow_features,
    normalize_cusip8,
    prepare_compustat_flow_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_flow_only_feature_construction_works_without_balance_sheet_fields() -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000", "001000"],
            "datadate": pd.to_datetime(["2020-03-31", "2021-03-31"]),
            "fyearq": [2020, 2021],
            "fqtr": [1, 1],
            "saley": [100.0, 120.0],
            "cogsy": [60.0, 70.0],
            "niy": [10.0, 12.0],
            "oancfy": [8.0, 9.0],
            "capxy": [3.0, 4.0],
            "xrdy": [2.0, 3.0],
            "xsgay": [12.0, 13.0],
        }
    )

    features = build_compustat_flow_features(frame)

    assert "availability_date" in features.columns
    assert features.loc[0, "compustat_gross_margin"] == pytest.approx(0.4)
    assert features.loc[0, "compustat_net_margin"] == pytest.approx(0.1)
    assert features.loc[0, "compustat_operating_cash_flow_margin"] == pytest.approx(0.08)
    assert features.loc[0, "compustat_capex_intensity"] == pytest.approx(0.03)
    assert features.loc[0, "compustat_rd_intensity"] == pytest.approx(0.02)
    assert features.loc[0, "compustat_sga_intensity"] == pytest.approx(0.12)


def test_flow_field_fallbacks_are_row_wise() -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000", "001001"],
            "datadate": pd.to_datetime(["2020-03-31", "2020-03-31"]),
            "fyearq": [2020, 2020],
            "fqtr": [1, 1],
            "saley": [100.0, np.nan],
            "revty": [999.0, 200.0],
            "niy": [np.nan, 20.0],
            "iby": [10.0, 999.0],
            "oiadpy": [np.nan, 30.0],
            "oibdpy": [15.0, 999.0],
            "oancfy": [8.0, 16.0],
            "capxy": [3.0, 4.0],
        }
    )

    prepared = prepare_compustat_flow_frame(frame)

    assert prepared.loc[0, "sales"] == 100.0
    assert prepared.loc[1, "sales"] == 200.0
    assert prepared.loc[0, "earnings"] == 10.0
    assert prepared.loc[1, "earnings"] == 20.0
    assert prepared.loc[0, "operating_income"] == 15.0
    assert prepared.loc[1, "operating_income"] == 30.0


def test_flow_only_block_does_not_create_balance_sheet_style_features() -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000"],
            "datadate": [pd.Timestamp("2020-03-31")],
            "fyearq": [2020],
            "fqtr": [1],
            "saley": [100.0],
            "cogsy": [60.0],
            "niy": [10.0],
            "oancfy": [8.0],
            "capxy": [3.0],
            "xrdy": [2.0],
            "xsgay": [12.0],
        }
    )

    features = build_compustat_flow_features(frame)

    forbidden = {"roa", "leverage", "book_to_market", "asset_growth"}
    assert forbidden.isdisjoint(features.columns)


def test_blank_gvkeys_remain_missing_not_zero_filled() -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000", "", "   ", None],
            "datadate": pd.to_datetime(["2020-03-31"] * 4),
            "fyearq": [2020] * 4,
            "fqtr": [1] * 4,
            "saley": [100.0, 200.0, 300.0, 400.0],
        }
    )

    prepared = prepare_compustat_flow_frame(frame)

    assert prepared["gvkey"].tolist()[0] == "001000"
    assert prepared["gvkey"].notna().sum() == 1
    assert "000000" not in set(prepared["gvkey"].dropna())


def test_prepare_compustat_flow_frame_uses_conservative_six_month_lag() -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000"],
            "datadate": [pd.Timestamp("2020-03-31")],
            "fyearq": [2020],
            "fqtr": [1],
            "saley": [100.0],
        }
    )

    prepared = prepare_compustat_flow_frame(frame, lag_months=6)

    assert prepared.loc[0, "availability_date"] == pd.Timestamp("2020-09-30")
    assert bool((prepared["availability_date"] <= pd.Timestamp("2020-08-31")).iloc[0]) is False
    assert bool((prepared["availability_date"] <= pd.Timestamp("2020-09-30")).iloc[0]) is True


def test_yoy_growth_uses_same_firm_and_same_fiscal_quarter_only() -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000", "001000", "001000", "002000", "002000"],
            "datadate": pd.to_datetime(
                ["2020-03-31", "2020-06-30", "2021-03-31", "2020-03-31", "2021-03-31"]
            ),
            "fyearq": [2020, 2020, 2021, 2020, 2021],
            "fqtr": [1, 2, 1, 1, 1],
            "saley": [100.0, 500.0, 130.0, 80.0, 100.0],
            "niy": [10.0, 50.0, 13.0, 8.0, 10.0],
            "oancfy": [9.0, 40.0, 11.0, 7.0, 9.0],
            "capxy": [2.0, 20.0, 3.0, 1.0, 2.0],
        }
    )

    features = build_compustat_flow_features(frame)
    lookup = features.set_index(["gvkey", "datadate"])

    assert lookup.loc[
        ("001000", pd.Timestamp("2021-03-31")), "compustat_sales_yoy_growth"
    ] == pytest.approx(0.3)
    assert lookup.loc[
        ("001000", pd.Timestamp("2021-03-31")), "compustat_earnings_yoy_growth"
    ] == pytest.approx(0.3)
    assert lookup.loc[
        ("002000", pd.Timestamp("2021-03-31")), "compustat_sales_yoy_growth"
    ] == pytest.approx(0.25)


def test_safe_denominators_produce_nan_not_inf_and_raise_missing_flags() -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000", "001001"],
            "datadate": pd.to_datetime(["2020-03-31", "2020-03-31"]),
            "fyearq": [2020, 2020],
            "fqtr": [1, 1],
            "saley": [0.0, np.nan],
            "cogsy": [1.0, 1.0],
            "niy": [2.0, 2.0],
            "oancfy": [1.0, 1.0],
            "capxy": [1.0, 1.0],
        }
    )

    features = build_compustat_flow_features(frame)

    for column in (
        "compustat_gross_margin",
        "compustat_net_margin",
        "compustat_capex_intensity",
    ):
        assert features[column].isna().all()
        assert not np.isinf(features[column].fillna(np.nan).to_numpy()).any()
        assert features[f"{column}_missing"].tolist() == [1, 1]


def test_cusip8_normalization_and_missing_links_never_fabricate_values() -> None:
    normalized = normalize_cusip8(pd.Series(["123456789", "", "   ", None, "123", "abCDefgh9"]))

    assert normalized.iloc[0] == "12345678"
    assert pd.isna(normalized.iloc[1])
    assert pd.isna(normalized.iloc[2])
    assert pd.isna(normalized.iloc[3])
    assert pd.isna(normalized.iloc[4])
    assert normalized.iloc[5] == "ABCDEFGH"

    compustat = pd.DataFrame(
        {
            "cusip": ["123456789", "", "123"],
            "datadate": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-03-31"]),
            "fyearq": [2020, 2020, 2020],
            "fqtr": [1, 1, 1],
            "saley": [100.0, 200.0, 300.0],
        }
    )
    features = build_compustat_flow_features(compustat, merge_key="cusip8")

    assert features["cusip8"].notna().sum() == 1

    formation = pd.DataFrame(
        {
            "cusip8": ["12345678", "87654321"],
            "date": pd.to_datetime(["2020-10-31", "2020-10-31"]),
        }
    )
    merged = merge_compustat_point_in_time(
        formation,
        features,
        merge_key="cusip8",
        lag_months=6,
        compustat_cols=["compustat_sales_log1p"],
        prefix="",
    )

    assert merged.loc[0, "compustat_sales_log1p"] == pytest.approx(np.log1p(100.0))
    assert pd.isna(merged.loc[1, "compustat_sales_log1p"])


def test_compustat_flow_block_remains_disabled_in_active_config() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "main.yaml")

    assert cfg.data.use_compustat is False
    assert cfg.features.compustat_features.enabled is False
