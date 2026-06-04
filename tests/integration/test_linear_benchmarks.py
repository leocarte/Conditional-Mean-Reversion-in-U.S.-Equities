"""Synthetic smoke tests for linear benchmark return-only linear models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlfinance.features.return_features import add_return_features
from mlfinance.run.linear_benchmarks import (
    PREDICTION_REQUIRED_COLUMNS,
    RETURN_ONLY_FEATURE_COLUMNS,
    run_linear_benchmarks,
)


def _write_synthetic_model_panel(path: Path) -> None:
    rng = np.random.default_rng(23)
    dates = pd.date_range("2017-01-31", "2021-12-31", freq="ME")
    rows: list[dict[str, object]] = []

    for permno in range(10001, 10041):
        prev_ret = 0.0
        prev_prev_ret = 0.0
        permno_effect = (permno - 10000) * 0.00015
        for i, date in enumerate(dates):
            market = 0.004 * np.sin(i / 4.0) + 0.002 * np.cos(i / 9.0)
            noise = rng.normal(0.0, 0.02)
            ret = permno_effect - 0.35 * prev_ret + 0.15 * prev_prev_ret + 0.20 * market + noise
            rows.append(
                {
                    "permno": permno,
                    "permco": permno,
                    "cusip": f"{permno:08d}",
                    "cusip8": f"{permno:08d}",
                    "ticker": f"T{permno}",
                    "siccd": 3571,
                    "naics": 334111,
                    "date": date,
                    "ret": ret,
                    "sprtrn": market,
                    "ret_market_adj": ret - market,
                }
            )
            prev_prev_ret = prev_ret
            prev_ret = ret

    panel = add_return_features(pd.DataFrame(rows))
    panel = panel.dropna(subset=["target_ret_fwd_1m"]).reset_index(drop=True)
    panel.to_parquet(path, index=False)


def _smoke_cfg(panel_path: Path, tmp_path: Path):
    outputs_dir = tmp_path / "outputs"
    report_dir = tmp_path / "report"
    return OmegaConf.create(
        {
            "data": {
                "panel_path": str(panel_path),
                "target_col": "target_ret_fwd_1m",
                "date_col": "date",
                "permno_col": "permno",
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
            "models": {
                "ridge": {
                    "shrinkage_grid": [1.0e-4, 1.0e-2, 1.0],
                },
                "elastic_net": {
                    "alpha_grid": [1.0e-4, 1.0e-2],
                    "l1_ratio_grid": [0.1, 0.9],
                    "max_iter": 10000,
                },
            },
            "portfolio": {
                "n_buckets": 10,
                "long_bucket": 10,
                "short_bucket": 1,
                "weighting": "EW",
            },
            "costs": {
                "main_bps": 10,
                "bps_grid": [0, 10, 25],
            },
            "eval": {
                "newey_west_lag_primary": 3,
            },
            "paths": {
                "predictions": str(outputs_dir / "predictions"),
                "backtests": str(outputs_dir / "backtests"),
                "metrics": str(outputs_dir / "metrics"),
                "report_tables": str(report_dir / "tables"),
            },
        }
    )


def _run_smoke_pipeline(tmp_path: Path):
    panel_path = tmp_path / "processed" / "model_panel.parquet"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    _write_synthetic_model_panel(panel_path)
    cfg = _smoke_cfg(panel_path, tmp_path)
    outputs = run_linear_benchmarks(cfg)
    return cfg, outputs


def test_linear_benchmarks_smoke_run_produces_oos_test_outputs(tmp_path: Path) -> None:
    cfg, outputs = _run_smoke_pipeline(tmp_path)

    assert outputs["validation_diagnostics"].exists()
    assert outputs["selected_hyperparameters"].exists()
    assert outputs["backtests"].exists()
    assert outputs["test_predictive_summary"].exists()
    assert outputs["test_cost_summary"].exists()
    assert outputs["test_summary"].exists()

    prediction_files = sorted(outputs["predictions_dir"].glob("*.parquet"))
    assert [path.name for path in prediction_files] == [
        "elastic_net__locked_test_202101_202112.parquet",
        "ridge__locked_test_202101_202112.parquet",
    ]

    all_predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_files],
        ignore_index=True,
    )
    assert set(PREDICTION_REQUIRED_COLUMNS).issubset(all_predictions.columns)
    assert set(all_predictions["model_name"]) == {"ridge", "elastic_net"}
    assert set(all_predictions["model"]) == {"ridge", "elastic_net"}
    assert set(all_predictions["split"]) == {"test"}
    assert set(all_predictions["target"]) == {"target_ret_fwd_1m"}
    assert all_predictions["date"].min() >= pd.Timestamp(cfg.splits.locked.test_start)
    assert all_predictions["date"].max() == pd.Timestamp("2021-11-30")
    assert all_predictions["realized_date"].min() == pd.Timestamp("2021-02-28")
    assert all_predictions["realized_date"].max() == pd.Timestamp("2021-12-31")

    backtests = pd.read_parquet(outputs["backtests"])
    assert set(backtests["model_name"]) == {"ridge", "elastic_net"}
    assert set(backtests["cost_bps"]) == {0, 10, 25}


def test_linear_benchmarks_never_saves_train_predictions(tmp_path: Path) -> None:
    cfg, outputs = _run_smoke_pipeline(tmp_path)

    prediction_files = sorted(outputs["predictions_dir"].glob("*.parquet"))
    assert len(prediction_files) == 2
    for path in prediction_files:
        frame = pd.read_parquet(path)
        assert (frame["split"] == "test").all()
        assert (
            frame["date"]
            .between(
                pd.Timestamp(cfg.splits.locked.test_start),
                pd.Timestamp(cfg.splits.locked.test_end),
            )
            .all()
        )
        assert (
            not frame["date"]
            .between(
                pd.Timestamp(cfg.splits.locked.train_start),
                pd.Timestamp(cfg.splits.locked.train_end),
            )
            .any()
        )
        assert (
            not frame["date"]
            .between(
                pd.Timestamp(cfg.splits.locked.val_start),
                pd.Timestamp(cfg.splits.locked.val_end),
            )
            .any()
        )


def test_linear_benchmarks_selection_metadata_tracks_train_only_and_refit_windows(
    tmp_path: Path,
) -> None:
    _, outputs = _run_smoke_pipeline(tmp_path)

    payload = json.loads(outputs["selected_hyperparameters"].read_text(encoding="utf-8"))

    assert payload["feature_columns"] == RETURN_ONLY_FEATURE_COLUMNS
    assert payload["selection_metric"] == "validation_net_sharpe_at_10bps"
    assert (
        payload["validation"]["train_only_preprocessing"]["ridge"]["preprocessor"]["fit_start"]
        == "2018-01-31"
    )
    assert (
        payload["validation"]["train_only_preprocessing"]["ridge"]["preprocessor"]["fit_end"]
        == "2019-12-31"
    )
    assert payload["test_refit"]["ridge"]["refit_sample"] == "train_plus_validation"
    assert payload["test_refit"]["ridge"]["refit_start"] == "2018-01-31"
    assert payload["test_refit"]["ridge"]["refit_end"] == "2020-12-31"
    assert payload["test_refit"]["elastic_net"]["refit_sample"] == "train_plus_validation"
