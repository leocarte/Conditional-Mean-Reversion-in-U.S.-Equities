"""Synthetic smoke tests for the nonlinear benchmark return-only models.

Exercises the full XGBoost + compact MLP pipeline end-to-end on a tiny
synthetic panel (CPU-friendly: 1-2 epochs, tens of trees). Requires xgboost,
torch, and hydra-core, which live in the cluster environment.

On macOS conda, torch and xgboost sharing one process is unsafe: the segfault
fires when xgboost builds a DMatrix, not at import. Prefer
``make nonlinear-benchmarks-smoke``, whose wrapper probes for this and skips
cleanly. When running pytest directly, set
``SKIP_NONLINEAR_BENCHMARK_SMOKE=1`` before the stack loads.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[2] / "scripts" / "_nonlinear_smoke_probe.py"

if os.environ.get("SKIP_NONLINEAR_BENCHMARK_SMOKE"):
    pytest.skip(
        "nonlinear benchmark nonlinear smoke skipped via SKIP_NONLINEAR_BENCHMARK_SMOKE "
        "(macOS conda torch+xgboost OpenMP clash); run on cluster for the strict path.",
        allow_module_level=True,
    )

import json

probe = subprocess.run(
    [sys.executable, str(PROBE)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
)
if probe.returncode != 0:
    pytest.skip(
        "nonlinear benchmark nonlinear tests skipped because torch + xgboost cannot share this "
        "process on the current stack.",
        allow_module_level=True,
    )

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlfinance.features.return_features import add_return_features
from mlfinance.run.linear_benchmarks import (
    PREDICTION_REQUIRED_COLUMNS,
    RETURN_ONLY_FEATURE_COLUMNS,
)
from mlfinance.run.nonlinear_benchmarks import run_nonlinear_benchmarks


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
                "xgb": {
                    "n_estimators_grid": [10, 20],
                    "max_depth_grid": [2],
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_lambda": 1.0,
                    "reg_alpha": 0.0,
                },
                "mlp": {
                    "hidden_dims": [8],
                    "activation": "relu",
                    "dropout": 0.0,
                    "lr": 1e-3,
                    "wd": 1e-4,
                    "batch_size": 256,
                    "epochs": 2,
                    "patience": 2,
                    "dropout_grid": [0.0],
                    "n_seeds": 1,
                    "use_layer_norm": True,
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
            },
        }
    )


def _run_smoke_pipeline(tmp_path: Path):
    panel_path = tmp_path / "processed" / "model_panel.parquet"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    _write_synthetic_model_panel(panel_path)
    cfg = _smoke_cfg(panel_path, tmp_path)
    return cfg, run_nonlinear_benchmarks(cfg)


def test_nonlinear_benchmarks_smoke_run_produces_oos_test_outputs(tmp_path: Path) -> None:
    cfg, outputs = _run_smoke_pipeline(tmp_path)

    for key in (
        "validation_diagnostics",
        "selected_hyperparameters",
        "backtests",
        "test_predictive_summary",
        "test_cost_summary",
        "test_summary",
        "xgb_feature_importance",
    ):
        assert outputs[key].exists(), key

    prediction_files = sorted(outputs["predictions_dir"].glob("*.parquet"))
    assert [path.name for path in prediction_files] == [
        "mlp__locked_test_202101_202112.parquet",
        "xgboost__locked_test_202101_202112.parquet",
    ]

    all_predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_files],
        ignore_index=True,
    )
    assert set(PREDICTION_REQUIRED_COLUMNS).issubset(all_predictions.columns)
    assert set(all_predictions["model_name"]) == {"xgboost", "mlp"}
    assert set(all_predictions["model"]) == {"xgboost", "mlp"}
    assert set(all_predictions["split"]) == {"test"}
    assert set(all_predictions["target"]) == {"target_ret_fwd_1m"}
    assert all_predictions["date"].min() >= pd.Timestamp(cfg.splits.locked.test_start)
    assert all_predictions["date"].max() == pd.Timestamp("2021-11-30")
    assert all_predictions["realized_date"].min() == pd.Timestamp("2021-02-28")
    assert all_predictions["realized_date"].max() == pd.Timestamp("2021-12-31")

    backtests = pd.read_parquet(outputs["backtests"])
    assert set(backtests["model_name"]) == {"xgboost", "mlp"}
    assert set(backtests["cost_bps"]) == {0, 10, 25}


def test_nonlinear_benchmarks_never_saves_train_or_validation_predictions(tmp_path: Path) -> None:
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
                pd.Timestamp(cfg.splits.locked.val_end),
            )
            .any()
        )


def test_nonlinear_benchmarks_feature_importance_and_refit_metadata(tmp_path: Path) -> None:
    _, outputs = _run_smoke_pipeline(tmp_path)

    importance = pd.read_csv(outputs["xgb_feature_importance"])
    assert list(importance.columns) == ["feature", "importance"]
    # One importance per preprocessed column: features + their missing-flag columns.
    assert len(importance) == 2 * len(RETURN_ONLY_FEATURE_COLUMNS)

    payload = json.loads(outputs["selected_hyperparameters"].read_text(encoding="utf-8"))
    assert payload["feature_columns"] == RETURN_ONLY_FEATURE_COLUMNS
    assert payload["selection_metric"] == "validation_net_sharpe_at_10bps"
    assert set(payload["test_refit"].keys()) == {"xgboost", "mlp"}
    assert payload["test_refit"]["xgboost"]["refit_sample"] == "train_plus_validation"
    assert payload["test_refit"]["mlp"]["refit_sample"] == "train_plus_validation"
    assert payload["test_refit"]["xgboost"]["refit_start"] == "2018-01-31"
    assert payload["test_refit"]["xgboost"]["refit_end"] == "2020-12-31"
