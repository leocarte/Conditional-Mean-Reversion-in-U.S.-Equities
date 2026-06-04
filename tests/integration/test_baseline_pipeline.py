"""Synthetic end-to-end smoke test for the CRSP-only baseline pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlfinance.features.return_features import add_return_features
from mlfinance.run.baseline_pipeline import run_baseline_pipeline
from mlfinance.run.convert_crsp_to_parquet import convert_crsp_csv_to_parquet


def _write_synthetic_crsp(path: Path) -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2017-01-31", "2021-12-31", freq="ME")
    rows: list[dict] = []
    for permno in range(10001, 10031):
        permno_effect = (permno - 10000) * 0.0003
        for i, date in enumerate(dates):
            market = 0.002 * np.sin(i / 4) + 0.001 * np.cos(i / 7)
            ret = -0.35 * market + permno_effect + rng.normal(0.0, 0.03)
            rows.append(
                {
                    "permno": permno,
                    "permco": permno,
                    "hdrcusip": f"{permno:08d}",
                    "ticker": f"T{permno}",
                    "siccd": 3571,
                    "naics": 334111,
                    "mthcaldt": date,
                    "mthret": ret,
                    "sprtrn": market,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_synthetic_crsp_csv(path: Path) -> None:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    rows: list[dict] = []
    for permno in [10001, 10002]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "PERMNO": permno,
                    "PERMCO": permno,
                    "HdrCUSIP": f"{permno:08d}",
                    "Ticker": f"T{permno}",
                    "SICCD": 3571,
                    "NAICS": 334111,
                    "MthCalDt": date,
                    "MthRet": float(rng.normal(0.01 * (i + 1), 0.01)),
                    "sprtrn": float(rng.normal(0.002, 0.001)),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_baseline_pipeline_runs_end_to_end_on_synthetic_crsp(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    outputs_dir = tmp_path / "outputs"
    report_dir = tmp_path / "report"
    raw_dir.mkdir()

    crsp_path = raw_dir / "monthly_crsp.parquet"
    _write_synthetic_crsp(crsp_path)

    cfg = OmegaConf.create(
        {
            "data": {
                "panel_path": str(processed_dir / "model_panel.parquet"),
                "target_col": "target_ret_fwd_1m",
                "robustness_target_col": "target_excess_ret_fwd_1m",
                "date_col": "date",
                "permno_col": "permno",
                "ret_col": "ret",
                "market_col": "sprtrn",
                "use_compustat": False,
                "use_regime_features": False,
                "paths": {
                    "crsp_monthly": str(crsp_path),
                },
            },
            "splits": {
                "locked": {
                    "train_start": "2018-01-31",
                    "train_end": "2019-12-31",
                    "val_start": "2020-01-31",
                    "val_end": "2020-12-31",
                    "test_start": "2021-01-31",
                    "test_end": "2021-12-31",
                }
            },
            "features": {
                "return_features": {
                    "enabled": True,
                    "min_history_months": 12,
                }
            },
            "active_models": ["zero", "reversal", "zscore_reversal", "momentum"],
            "portfolio": {
                "n_buckets": 10,
                "long_bucket": 10,
                "short_bucket": 1,
                "weighting": "EW",
            },
            "costs": {
                "main_bps": 10,
                "bps_grid": [0, 5, 10, 25, 50],
            },
            "paths": {
                "predictions": str(outputs_dir / "predictions"),
                "backtests": str(outputs_dir / "backtests"),
                "metrics": str(outputs_dir / "metrics"),
                "report_tables": str(report_dir / "tables"),
            },
        }
    )

    outputs = run_baseline_pipeline(cfg, rebuild_panel=True)

    assert outputs["panel"].exists()
    assert outputs["backtests"].exists()
    assert outputs["first_table_csv"].exists()

    prediction_dir = outputs["predictions_dir"]
    prediction_files = sorted(prediction_dir.glob("*.parquet"))
    assert len(prediction_files) == 4

    all_predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_files],
        ignore_index=True,
    )
    assert set(all_predictions["model_name"]) == {"zero", "reversal", "zscore_reversal", "momentum"}
    assert all_predictions["date"].min() >= pd.Timestamp("2021-01-31")
    assert all_predictions["date"].max() == pd.Timestamp("2021-11-30")
    assert all_predictions["realized_date"].min() == pd.Timestamp("2021-02-28")
    assert all_predictions["realized_date"].max() == pd.Timestamp("2021-12-31")
    zero_preds = all_predictions.loc[all_predictions["model_name"] == "zero", "prediction"]
    assert (zero_preds == 0.0).all()

    backtests = pd.read_parquet(outputs["backtests"])
    assert set(backtests["cost_bps"].unique()) == {0, 5, 10, 25, 50}
    assert set(backtests["model_name"].unique()) == {
        "zero",
        "reversal",
        "zscore_reversal",
        "momentum",
    }
    zero_backtests = backtests.loc[backtests["model_name"] == "zero"].reset_index(drop=True)
    assert (zero_backtests["gross_return"] == 0.0).all()
    assert (zero_backtests["net_return"] == 0.0).all()
    assert (zero_backtests["turnover"] == 0.0).all()
    assert (zero_backtests["cost"] == 0.0).all()
    assert (zero_backtests["long_return"] == 0.0).all()
    assert (zero_backtests["short_return"] == 0.0).all()
    assert (zero_backtests["n_long"] == 0).all()
    assert (zero_backtests["n_short"] == 0).all()

    first_table = pd.read_csv(outputs["first_table_csv"])
    assert list(first_table["model_name"]) == ["momentum", "reversal", "zero", "zscore_reversal"]
    assert set(
        ["cost_bps", "n_months", "realized_start", "realized_end", "test_start", "test_end"]
    ).issubset(first_table.columns)
    assert set(first_table["cost_bps"]) == {10}
    assert set(first_table["n_months"]) == {11}
    assert set(first_table["realized_start"]) == {"2021-02-28"}
    assert set(first_table["realized_end"]) == {"2021-12-31"}
    assert set(first_table["test_start"]) == {"2021-01-31"}
    assert set(first_table["test_end"]) == {"2021-12-31"}
    non_zero_rows = first_table.loc[first_table["model_name"] != "zero"]
    assert not non_zero_rows["net_sharpe_10bps"].isna().any()
    zero_row = first_table.loc[first_table["model_name"] == "zero"].iloc[0]
    assert zero_row["gross_ann_return"] == 0.0
    assert zero_row["net_ann_return_10bps"] == 0.0
    assert zero_row["avg_turnover"] == 0.0
    assert pd.isna(zero_row["net_sharpe_10bps"])
    assert pd.isna(zero_row["top_minus_bottom_mean"])


def test_convert_crsp_csv_to_parquet_keeps_source_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "monthly_crsp.csv"
    parquet_path = tmp_path / "monthly_crsp.parquet"
    _write_synthetic_crsp_csv(csv_path)

    out = convert_crsp_csv_to_parquet(csv_path, parquet_path=parquet_path)

    assert out == parquet_path
    assert csv_path.exists()
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 6
    assert {"PERMNO", "MthCalDt", "MthRet", "sprtrn"}.issubset(df.columns)


def test_baseline_can_regenerate_outputs_from_existing_panel_without_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed_dir = tmp_path / "processed"
    outputs_dir = tmp_path / "outputs"
    report_dir = tmp_path / "report"
    processed_dir.mkdir()

    dates = pd.date_range("2017-01-31", "2021-12-31", freq="ME")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(19)
    for permno in range(10001, 10031):
        permno_effect = (permno - 10000) * 0.0002
        for i, date in enumerate(dates):
            market = 0.002 * np.sin(i / 5) + 0.001 * np.cos(i / 9)
            ret = permno_effect + rng.normal(0.0, 0.03) - 0.20 * market
            rows.append(
                {
                    "permno": permno,
                    "permco": permno,
                    "cusip": f"{permno:08d}",
                    "ticker": f"T{permno}",
                    "siccd": 3571,
                    "naics": 334111,
                    "date": date,
                    "ret": ret,
                    "sprtrn": market,
                }
            )
    panel = add_return_features(pd.DataFrame(rows))
    panel = panel.dropna(subset=["target_ret_fwd_1m"]).reset_index(drop=True)
    panel_path = processed_dir / "model_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    cfg = OmegaConf.create(
        {
            "data": {
                "panel_path": str(panel_path),
                "target_col": "target_ret_fwd_1m",
                "robustness_target_col": "target_excess_ret_fwd_1m",
                "date_col": "date",
                "permno_col": "permno",
                "ret_col": "ret",
                "market_col": "sprtrn",
                "use_compustat": False,
                "use_regime_features": False,
                "paths": {
                    "crsp_monthly": str(tmp_path / "raw" / "unused_monthly_crsp.parquet"),
                },
            },
            "splits": {
                "locked": {
                    "train_start": "2018-01-31",
                    "train_end": "2019-12-31",
                    "val_start": "2020-01-31",
                    "val_end": "2020-12-31",
                    "test_start": "2021-01-31",
                    "test_end": "2021-12-31",
                }
            },
            "features": {
                "return_features": {
                    "enabled": True,
                    "min_history_months": 12,
                }
            },
            "active_models": ["zero", "reversal", "zscore_reversal", "momentum"],
            "portfolio": {
                "n_buckets": 10,
                "long_bucket": 10,
                "short_bucket": 1,
                "weighting": "EW",
            },
            "costs": {
                "main_bps": 10,
                "bps_grid": [0, 5, 10, 25, 50],
            },
            "paths": {
                "predictions": str(outputs_dir / "predictions"),
                "backtests": str(outputs_dir / "backtests"),
                "metrics": str(outputs_dir / "metrics"),
                "report_tables": str(report_dir / "tables"),
            },
        }
    )

    def _raise_if_raw_load(*args, **kwargs):
        raise AssertionError("raw CRSP loading should not be invoked when rebuild_panel=False")

    monkeypatch.setattr("mlfinance.run.baseline_pipeline.load_crsp_monthly", _raise_if_raw_load)

    outputs = run_baseline_pipeline(cfg, rebuild_panel=False)

    assert outputs["panel"] == panel_path
    assert outputs["backtests"].exists()
    assert outputs["first_table_csv"].exists()
    assert outputs["first_table_tex"].exists()

    backtests = pd.read_parquet(outputs["backtests"])
    zero_backtests = backtests.loc[backtests["model_name"] == "zero"]
    assert (zero_backtests["gross_return"] == 0.0).all()
    assert (zero_backtests["net_return"] == 0.0).all()
    assert (zero_backtests["turnover"] == 0.0).all()
    assert (zero_backtests["n_long"] == 0).all()
    assert (zero_backtests["n_short"] == 0).all()

    first_table = pd.read_csv(outputs["first_table_csv"])
    zero_row = first_table.loc[first_table["model_name"] == "zero"].iloc[0]
    assert zero_row["gross_ann_return"] == 0.0
    assert zero_row["net_ann_return_10bps"] == 0.0
    assert zero_row["avg_turnover"] == 0.0
    assert pd.isna(zero_row["net_sharpe_10bps"])
    assert pd.isna(zero_row["top_minus_bottom_mean"])
