"""Synthetic integration tests for the disabled-by-default feature-audit flow block."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlfinance.data.compustat_schema import audit_compustat_point_in_time_merge
from mlfinance.features.compustat_flow_features import add_flow_compustat_features_to_panel
from mlfinance.run.baseline_pipeline import build_model_panel
from mlfinance.run.linear_benchmarks import locked_split_masks

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_COMPUSTAT_TOKENS = (
    "roa",
    "leverage",
    "book_to_market",
    "asset_growth",
    "debt_to_assets",
    "assets_scaled",
    "equity_scaled",
)


def _write_synthetic_crsp(path: Path) -> None:
    dates = pd.date_range("2020-01-31", "2020-11-30", freq="ME")
    rows: list[dict[str, object]] = []
    for permno, cusip_root, base_ret in (
        (10001, "11111111", 0.010),
        (10002, "22222222", 0.020),
    ):
        for step, date in enumerate(dates):
            rows.append(
                {
                    "PERMNO": permno,
                    "PERMCO": permno,
                    "HdrCUSIP": f"{cusip_root}9",
                    "Ticker": f"T{permno}",
                    "SICCD": 3571,
                    "NAICS": 334111,
                    "MthCalDt": date,
                    "MthRet": base_ret + 0.001 * step,
                    "sprtrn": 0.001 + 0.0001 * step,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_synthetic_compustat(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "gvkey": ["001000", "001000", "002000", "002000"],
            "cusip": ["111111119", "111111119", "222222229", "222222229"],
            "datadate": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
            "fyearq": [2020, 2020, 2020, 2020],
            "fqtr": [1, 2, 1, 2],
            "saley": [100.0, 999.0, 200.0, 888.0],
            "cogsy": [60.0, 555.0, 120.0, 444.0],
            "niy": [10.0, 111.0, 20.0, 222.0],
            "oancfy": [8.0, 88.0, 16.0, 176.0],
            "capxy": [3.0, 33.0, 4.0, 44.0],
            "xrdy": [2.0, 22.0, 5.0, 55.0],
            "xsgay": [12.0, 144.0, 18.0, 216.0],
        }
    )
    frame.to_parquet(path, index=False)


def _integration_cfg(
    *,
    raw_crsp_path: Path,
    panel_path: Path,
    use_compustat: bool,
    compustat_enabled: bool,
    compustat_path: Path | None = None,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "panel_path": str(panel_path),
                "target_col": "target_ret_fwd_1m",
                "robustness_target_col": "target_excess_ret_fwd_1m",
                "date_col": "date",
                "permno_col": "permno",
                "ret_col": "ret",
                "market_col": "sprtrn",
                "use_compustat": use_compustat,
                "use_regime_features": False,
                "paths": {
                    "crsp_monthly": str(raw_crsp_path),
                    "compustat_quarterly": (
                        str(compustat_path)
                        if compustat_path is not None
                        else str(raw_crsp_path.parent / "missing_compustat.parquet")
                    ),
                },
            },
            "splits": {
                "locked": {
                    "train_start": "2020-03-31",
                    "train_end": "2020-06-30",
                    "val_start": "2020-07-31",
                    "val_end": "2020-08-31",
                    "test_start": "2020-09-30",
                    "test_end": "2020-10-31",
                }
            },
            "features": {
                "return_features": {
                    "enabled": True,
                    "min_history_months": 1,
                },
                "compustat_features": {
                    "enabled": compustat_enabled,
                    "conservative_lag_months": 6,
                },
            },
        }
    )


def test_compustat_default_config_keeps_compustat_disabled_and_panel_build_does_not_require_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "main.yaml")

    assert cfg.data.use_compustat is False
    assert cfg.features.compustat_features.enabled is False
    assert cfg.features.compustat_features.conservative_lag_months == 6

    crsp_path = tmp_path / "monthly_crsp.parquet"
    _write_synthetic_crsp(crsp_path)
    synthetic_cfg = _integration_cfg(
        raw_crsp_path=crsp_path,
        panel_path=tmp_path / "panel.parquet",
        use_compustat=False,
        compustat_enabled=False,
    )

    def _unexpected_compustat_load(*args, **kwargs):
        raise AssertionError("Compustat loading should remain disabled by default.")

    monkeypatch.setattr(
        "mlfinance.run.baseline_pipeline.load_compustat_quarterly", _unexpected_compustat_load
    )

    panel = build_model_panel(synthetic_cfg)

    assert not panel.empty
    assert not any(column.startswith("compustat_") for column in panel.columns)


def test_compustat_enabled_panel_build_merges_flow_only_compustat_after_six_month_lag(
    tmp_path: Path,
) -> None:
    crsp_path = tmp_path / "monthly_crsp.parquet"
    compustat_path = tmp_path / "CompFirmCharac.parquet"
    _write_synthetic_crsp(crsp_path)
    _write_synthetic_compustat(compustat_path)

    disabled_cfg = _integration_cfg(
        raw_crsp_path=crsp_path,
        panel_path=tmp_path / "panel_return_only.parquet",
        use_compustat=False,
        compustat_enabled=False,
        compustat_path=compustat_path,
    )
    enabled_cfg = _integration_cfg(
        raw_crsp_path=crsp_path,
        panel_path=tmp_path / "panel_with_compustat.parquet",
        use_compustat=True,
        compustat_enabled=True,
        compustat_path=compustat_path,
    )

    panel_disabled = (
        build_model_panel(disabled_cfg).sort_values(["permno", "date"]).reset_index(drop=True)
    )
    panel_enabled = (
        build_model_panel(enabled_cfg).sort_values(["permno", "date"]).reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        panel_enabled[["permno", "date"]],
        panel_disabled[["permno", "date"]],
    )
    pd.testing.assert_series_equal(
        panel_enabled["target_ret_fwd_1m"],
        panel_disabled["target_ret_fwd_1m"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        panel_enabled["target_excess_ret_fwd_1m"],
        panel_disabled["target_excess_ret_fwd_1m"],
        check_names=False,
    )

    disabled_masks = locked_split_masks(panel_disabled, enabled_cfg)
    enabled_masks = locked_split_masks(panel_enabled, enabled_cfg)
    for split_name in ("train", "validation", "test"):
        pd.testing.assert_series_equal(
            enabled_masks[split_name].reset_index(drop=True),
            disabled_masks[split_name].reset_index(drop=True),
            check_names=False,
        )

    june = panel_enabled.loc[panel_enabled["date"] == pd.Timestamp("2020-06-30")]
    assert june["compustat_sales_log1p"].isna().all()
    assert (june["compustat_sales_log1p_missing"] == 1).all()
    assert june["compustat_datadate"].isna().all()
    assert june["compustat_availability_date"].isna().all()

    october = panel_enabled.loc[panel_enabled["date"] == pd.Timestamp("2020-10-31")]
    assert len(october) == 2

    firm_10001 = october.loc[october["permno"] == 10001].iloc[0]
    firm_10002 = october.loc[october["permno"] == 10002].iloc[0]

    assert firm_10001["compustat_sales_log1p"] == pytest.approx(np.log1p(100.0))
    assert firm_10002["compustat_sales_log1p"] == pytest.approx(np.log1p(200.0))
    assert firm_10001["compustat_datadate"] == pd.Timestamp("2020-03-31")
    assert firm_10002["compustat_datadate"] == pd.Timestamp("2020-03-31")
    assert firm_10001["compustat_availability_date"] == pd.Timestamp("2020-09-30")
    assert firm_10002["compustat_availability_date"] == pd.Timestamp("2020-09-30")

    violations = audit_compustat_point_in_time_merge(panel_enabled, raise_on_fail=True)
    assert violations.empty

    assert not any(
        token in column for token in FORBIDDEN_COMPUSTAT_TOKENS for column in panel_enabled.columns
    )


def test_compustat_missing_compustat_links_stay_present_with_nan_features() -> None:
    panel = pd.DataFrame(
        {
            "permno": [10001, 10002],
            "date": pd.to_datetime(["2020-10-31", "2020-10-31"]),
            "cusip": ["111111119", "123"],
            "ret_1m": [0.01, 0.02],
        }
    )
    compustat = pd.DataFrame(
        {
            "cusip": ["111111119"],
            "datadate": [pd.Timestamp("2020-03-31")],
            "fyearq": [2020],
            "fqtr": [1],
            "saley": [100.0],
            "cogsy": [60.0],
            "niy": [10.0],
            "oancfy": [8.0],
            "capxy": [3.0],
        }
    )

    merged = add_flow_compustat_features_to_panel(
        panel,
        compustat,
        merge_key="cusip8",
        lag_months=6,
    ).sort_values("permno")

    matched = merged.loc[merged["permno"] == 10001].iloc[0]
    unmatched = merged.loc[merged["permno"] == 10002].iloc[0]

    assert len(merged) == 2
    assert matched["compustat_sales_log1p"] == pytest.approx(np.log1p(100.0))
    assert matched["cusip8"] == "11111111"
    assert pd.isna(unmatched["cusip8"])
    assert pd.isna(unmatched["compustat_sales_log1p"])
    assert unmatched["compustat_sales_log1p_missing"] == 1
    assert pd.isna(unmatched["compustat_datadate"])
    assert pd.isna(unmatched["compustat_availability_date"])


def test_compustat_compustat_pit_audit_rejects_future_availability_dates() -> None:
    panel = pd.DataFrame(
        {
            "permno": [10001],
            "date": [pd.Timestamp("2020-10-31")],
            "cusip8": ["11111111"],
        }
    )
    compustat = pd.DataFrame(
        {
            "cusip": ["111111119"],
            "datadate": [pd.Timestamp("2020-03-31")],
            "fyearq": [2020],
            "fqtr": [1],
            "saley": [100.0],
            "cogsy": [60.0],
            "niy": [10.0],
            "oancfy": [8.0],
            "capxy": [3.0],
        }
    )

    merged = add_flow_compustat_features_to_panel(panel, compustat, merge_key="cusip8")
    broken = merged.copy()
    broken.loc[0, "compustat_availability_date"] = pd.Timestamp("2020-11-30")

    with pytest.raises(ValueError, match=r"future_availability_date"):
        audit_compustat_point_in_time_merge(broken, raise_on_fail=True)
