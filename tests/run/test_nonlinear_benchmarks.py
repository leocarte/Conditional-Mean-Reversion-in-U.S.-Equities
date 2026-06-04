"""Focused tests for the nonlinear benchmark pipeline helpers.

On macOS conda, torch and xgboost can segfault when they share one process.
Prefer ``make nonlinear-benchmarks-smoke``; for a direct pytest run set
``SKIP_NONLINEAR_BENCHMARK_SMOKE=1``.
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

import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlfinance.run.nonlinear_benchmarks import (
    MODEL_FAMILIES,
    _jsonable,
    _mlp_candidates,
    _xgb_candidates,
    select_best_candidates,
)


def _grid_cfg():
    return OmegaConf.create(
        {
            "models": {
                "xgb": {
                    "n_estimators_grid": [200, 500],
                    "max_depth_grid": [3, 6],
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "reg_alpha": 0.0,
                },
                "mlp": {
                    "hidden_dims": [64, 32],
                    "activation": "relu",
                    "dropout": 0.1,
                    "lr": 1e-3,
                    "wd": 1e-4,
                    "batch_size": 4096,
                    "epochs": 30,
                    "patience": 5,
                    "dropout_grid": [0.1, 0.2],
                    "n_seeds": 1,
                    "use_layer_norm": True,
                },
            }
        }
    )


def test_model_families_are_xgboost_and_mlp() -> None:
    assert MODEL_FAMILIES == ("xgboost", "mlp")


def test_xgb_candidate_grid_is_cartesian_product() -> None:
    candidates = _xgb_candidates(_grid_cfg())
    assert len(candidates) == 4
    assert {(c["n_estimators"], c["max_depth"]) for c in candidates} == {
        (200, 3),
        (200, 6),
        (500, 3),
        (500, 6),
    }
    assert all(c["learning_rate"] == 0.05 for c in candidates)
    assert all(c["reg_alpha"] == 0.0 for c in candidates)


def test_mlp_candidate_grid_varies_dropout_and_maps_weight_decay() -> None:
    candidates = _mlp_candidates(_grid_cfg())
    assert len(candidates) == 2
    assert sorted(c["dropout"] for c in candidates) == [0.1, 0.2]
    assert all(c["hidden_dims"] == [64, 32] for c in candidates)
    assert all(c["weight_decay"] == 1e-4 for c in candidates)
    assert all(c["use_layer_norm"] is True for c in candidates)


def test_select_best_candidates_picks_top_net_sharpe_with_tie_break() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "model_family": "xgboost",
                "model_name": "xgboost__cand00",
                "candidate_rank": 0,
                "validation_net_sharpe": 0.5,
                "rank_ic_mean": 0.01,
                "oos_r2": -1.0,
                "rmse": 0.20,
            },
            {
                "model_family": "xgboost",
                "model_name": "xgboost__cand01",
                "candidate_rank": 1,
                "validation_net_sharpe": 1.2,
                "rank_ic_mean": 0.02,
                "oos_r2": -0.9,
                "rmse": 0.19,
            },
            {
                "model_family": "mlp",
                "model_name": "mlp__cand00",
                "candidate_rank": 0,
                "validation_net_sharpe": 0.8,
                "rank_ic_mean": 0.03,
                "oos_r2": -0.8,
                "rmse": 0.18,
            },
            {
                "model_family": "mlp",
                "model_name": "mlp__cand01",
                "candidate_rank": 1,
                "validation_net_sharpe": 0.8,
                "rank_ic_mean": 0.01,
                "oos_r2": -0.85,
                "rmse": 0.21,
            },
        ]
    )
    winners = select_best_candidates(diagnostics)
    chosen = dict(zip(winners["model_family"], winners["model_name"], strict=True))
    assert len(winners) == 2
    assert chosen["xgboost"] == "xgboost__cand01"  # strictly higher net Sharpe
    assert chosen["mlp"] == "mlp__cand00"  # net-Sharpe tie broken by higher rank IC


def test_jsonable_coerces_numpy_and_nan() -> None:
    out = _jsonable(
        {
            "a": np.float64(1.5),
            "b": np.int64(3),
            "c": np.nan,
            "d": [np.float32(2.0)],
            "e": np.bool_(True),
        }
    )
    assert out == {"a": 1.5, "b": 3, "c": None, "d": [2.0], "e": True}
    json.dumps(out)  # must be serializable
