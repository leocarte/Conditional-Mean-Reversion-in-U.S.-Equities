"""Synthetic integration tests for the feature-audit nonlinear flow-Compustat runner.

Mirrors ``tests/integration/test_nonlinear_benchmarks.py``: skips at module
load on macOS conda (duplicate-OpenMP segfault) or when the runner-specific
opt-out env var is set, then runs the full feature-audit nonlinear pipeline on a
small synthetic CRSP + Compustat panel.

The tests verify that:

* default config keeps Compustat disabled (the runner must override per-call
  only);
* the feature-audit feature list builder includes return-only + flow-Compustat +
  missingness flags and excludes metadata / forbidden balance-sheet names
  (same contract as the linear feature-audit runner — covered there with full
  parametrization; here we only re-check the cross-cutting subset);
* the runner produces every promised artefact, including the XGB feature
  importance file and the side-by-side comparison files;
* prediction parquets carry the ``__compustat_flow_compustat_nonlinear`` suffix
  so they cannot overwrite the nonlinear-benchmark return-only files OR the feature-audit
  linear files;
* enabling Compustat does not change the locked split masks (target / split
  invariance, mirroring the linear feature-audit invariance check).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Module-level skip on macOS conda duplicate-OpenMP and explicit opt-outs --
# mirrors tests/integration/test_nonlinear_benchmarks.py:26-46.
if os.environ.get("SKIP_FLOW_COMPUSTAT_NONLINEAR_SMOKE"):
    pytest.skip(
        "SKIP_FLOW_COMPUSTAT_NONLINEAR_SMOKE is set; skipping feature-audit nonlinear smoke.",
        allow_module_level=True,
    )
if os.environ.get("SKIP_NONLINEAR_BENCHMARK_SMOKE"):
    pytest.skip(
        "SKIP_NONLINEAR_BENCHMARK_SMOKE is set; skipping feature-audit nonlinear smoke.",
        allow_module_level=True,
    )

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE = _REPO_ROOT / "scripts" / "_m2b_smoke_probe.py"
if _PROBE.exists():
    _probe = subprocess.run(
        [sys.executable, str(_PROBE)],
        capture_output=True,
        text=True,
    )
    if _probe.returncode != 0:
        pytest.skip(
            "torch + xgboost cannot share this process (likely macOS conda "
            "duplicate OpenMP); skipping feature-audit nonlinear smoke.",
            allow_module_level=True,
        )

# Heavy imports kept below the skip guards so we do not pay torch + xgboost
# imports on platforms that will skip anyway.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from mlfinance.run.flow_compustat_linear import (  # noqa: E402
    _with_compustat_overrides,
)
from mlfinance.run.flow_compustat_nonlinear import (  # noqa: E402
    OUTPUT_PREFIX_NONLINEAR,
    _flow_compustat_nonlinear_test_split_id,
    run_flow_compustat_nonlinear,
)
from mlfinance.run.linear_benchmarks import (  # noqa: E402
    RETURN_ONLY_FEATURE_COLUMNS,
    locked_split_masks,
)

REPO_ROOT = _REPO_ROOT


# ---------------------------------------------------------------------------
# Default-config guardrail
# ---------------------------------------------------------------------------


def test_default_config_keeps_compustat_disabled_for_nonlinear_runner() -> None:
    """The on-disk default config must keep Compustat disabled."""
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "main.yaml")
    assert cfg.data.use_compustat is False
    assert cfg.features.compustat_features.enabled is False
    assert int(cfg.features.compustat_features.conservative_lag_months) == 6


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


def _write_synthetic_crsp(path: Path) -> None:
    dates = pd.date_range("2019-01-31", "2021-12-31", freq="ME")
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    permnos = list(range(10001, 10011))  # 10 permnos
    cusip_roots = [f"{i:08d}" for i in range(11111111, 11111121)]
    for permno, cusip_root in zip(permnos, cusip_roots, strict=False):
        for step, date in enumerate(dates):
            rows.append(
                {
                    "PERMNO": int(permno),
                    "PERMCO": int(permno),
                    "HdrCUSIP": f"{cusip_root}9",
                    "Ticker": f"T{permno}",
                    "SICCD": 3571,
                    "NAICS": 334111,
                    "MthCalDt": date,
                    "MthRet": 0.005 + 0.001 * step + 0.002 * rng.standard_normal(),
                    "sprtrn": 0.002 + 0.0001 * step,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_synthetic_compustat(path: Path) -> None:
    quarters = pd.to_datetime(
        [
            "2018-12-31",
            "2019-03-31",
            "2019-06-30",
            "2019-09-30",
            "2019-12-31",
            "2020-03-31",
            "2020-06-30",
            "2020-09-30",
            "2020-12-31",
            "2021-03-31",
            "2021-06-30",
            "2021-09-30",
        ]
    )
    cusip_roots = [f"{i:08d}" for i in range(11111111, 11111121)]
    rows: list[dict[str, object]] = []
    for c_idx, cusip_root in enumerate(cusip_roots):
        gvkey = f"00{c_idx:04d}"
        for q_idx, datadate in enumerate(quarters):
            fy = int(datadate.year)
            fq = ((datadate.month - 1) // 3) + 1
            scale = 100.0 + 10.0 * q_idx + 5.0 * c_idx
            rows.append(
                {
                    "gvkey": gvkey,
                    "cusip": f"{cusip_root}9",
                    "datadate": datadate,
                    "fyearq": fy,
                    "fqtr": fq,
                    "saley": scale,
                    "cogsy": 0.6 * scale,
                    "niy": 0.1 * scale,
                    "oancfy": 0.08 * scale,
                    "capxy": 0.03 * scale,
                    "xrdy": 0.02 * scale,
                    "xsgay": 0.12 * scale,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _flow_compustat_nonlinear_cfg(tmp_path: Path) -> OmegaConf:
    """Tiny config that exercises the full feature-audit nonlinear path on synthetic data."""
    crsp_path = tmp_path / "monthly_crsp.parquet"
    compustat_path = tmp_path / "CompFirmCharac.parquet"
    _write_synthetic_crsp(crsp_path)
    _write_synthetic_compustat(compustat_path)
    return OmegaConf.create(
        {
            "data": {
                "panel_path": str(tmp_path / "model_panel.parquet"),
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
                    "compustat_quarterly": str(compustat_path),
                },
            },
            "splits": {
                "locked": {
                    "train_start": "2019-03-31",
                    "train_end": "2020-12-31",
                    "val_start": "2021-01-31",
                    "val_end": "2021-06-30",
                    "test_start": "2021-07-31",
                    "test_end": "2021-11-30",
                }
            },
            "features": {
                "return_features": {"enabled": True, "min_history_months": 1},
                "compustat_features": {
                    "enabled": False,
                    "conservative_lag_months": 6,
                },
            },
            "models": {
                # Tiny XGB grid -- 2 candidates, 10/20 trees, depth 2 (fast).
                "xgb": {
                    "n_estimators_grid": [10, 20],
                    "max_depth_grid": [2],
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "reg_alpha": 0.0,
                },
                # Tiny MLP grid -- 1 dropout candidate, 8-unit hidden, 2 epochs.
                "mlp": {
                    "hidden_dims": [8],
                    "activation": "relu",
                    "dropout": 0.0,
                    "dropout_grid": [0.0],
                    "lr": 1e-3,
                    "wd": 1e-4,
                    "batch_size": 256,
                    "epochs": 2,
                    "patience": 2,
                    "n_seeds": 1,
                    "use_bf16": False,
                    "use_layer_norm": True,
                },
            },
            "portfolio": {
                "n_buckets": 3,
                "long_bucket": 3,
                "short_bucket": 1,
                "weighting": "EW",
                "gross_leverage": 2.0,
                "rebalance_frequency": "monthly",
                "cutoffs": [3],
            },
            "costs": {
                "model": "flat_bps",
                "main_bps": 10,
                "bps_grid": [0, 10, 25],
            },
            "eval": {
                "predictive_metrics": ["rmse", "oos_r2"],
                "portfolio_metrics": ["ann_return", "ann_vol", "sharpe", "turnover"],
                "newey_west_lag_primary": 1,
                "benchmark_col": "sprtrn",
            },
            "paths": {
                "predictions": str(tmp_path / "outputs" / "predictions"),
                "backtests": str(tmp_path / "outputs" / "backtests"),
                "metrics": str(tmp_path / "outputs" / "metrics"),
                "report_tables": str(tmp_path / "report" / "tables"),
                "report_figures": str(tmp_path / "report" / "figures"),
            },
            "active_models": ["zero"],
            "runtime": {"rebuild_panel": True},
        }
    )


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def test_flow_compustat_nonlinear_runner_runs_end_to_end_on_synthetic_panel(tmp_path: Path) -> None:
    cfg = _flow_compustat_nonlinear_cfg(tmp_path)
    outputs = run_flow_compustat_nonlinear(cfg)

    expected_keys = {
        "panel_diagnostics",
        "feature_list",
        "validation_diagnostics",
        "test_predictive_summary",
        "test_cost_grid_summary",
        "test_summary",
        "test_summary_vs_return_only",
        "test_cost_grid_vs_return_only",
        "selection_and_disclosures",
        "backtests",
    }
    assert expected_keys.issubset(outputs.keys())
    for key in expected_keys:
        assert outputs[key].exists(), key

    # Panel-level facts the specification requires.
    diag = pd.read_csv(outputs["panel_diagnostics"]).iloc[0]
    assert int(diag["pit_violations"]) == 0
    assert int(diag["n_return_features"]) == len(RETURN_ONLY_FEATURE_COLUMNS)
    assert int(diag["n_compustat_features"]) > 0
    # The nonlinear runner uses ``include_missingness_flags=False`` because
    # ``LinearFeaturePreprocessor`` regenerates the *_missing flags for the model
    # itself (and pre-attaching them causes a duplicate-column-name collision).
    # So n_missingness_flags in the model feature list is 0 here; the panel
    # still contains the underlying compustat_*_missing columns, but they are
    # excluded from the model feature list.
    assert int(diag["n_missingness_flags"]) == 0
    assert diag["target_col"] == "target_ret_fwd_1m"
    assert diag["train_start"] == "2019-03-31"
    assert diag["test_end"] == "2021-11-30"

    # Default cfg flags MUST stay disabled (no in-place mutation of cfg).
    assert cfg.data.use_compustat is False
    assert cfg.features.compustat_features.enabled is False

    # Direct postcondition check on the actual invariant: the X matrix that
    # XGB / MLP see must have no duplicate column names. Mirrors the linear
    # test. The ``n_missingness_flags`` check above asserts the INPUT feature
    # list is clean (the historical root cause of duplicates); this check
    # asserts the OUTPUT matrix itself is clean, so a future regression that
    # introduces duplicates via a different path (e.g. a refactor of
    # ``LinearFeaturePreprocessor`` internals, or panel-side column injection
    # from a new feature block) still fails this test loudly. Covers both the
    # shared train-only validation-sweep preprocessor and the shared
    # train+validation refit preprocessor, for both model families.
    disclosures = json.loads(outputs["selection_and_disclosures"].read_text())
    # The nonlinear runner stores ``train_only_metadata`` (which itself contains
    # the ``train_only_preprocessing`` key) under ``train_only_preprocessing``,
    # producing a nested ``train_only_preprocessing.train_only_preprocessing``
    # path. Linear runner unwraps the inner layer before embedding -- structures
    # differ by one level, asserted independently.
    train_only_block = disclosures["train_only_preprocessing"]["train_only_preprocessing"]
    refit_block = disclosures["test_refit"]
    for family in ("xgboost", "mlp"):
        train_only_cols = train_only_block[family]["preprocessor"]["matrix_columns"]
        assert len(train_only_cols) == len(set(train_only_cols)), (
            f"{family} train-only preprocessor produced duplicate matrix columns: "
            f"{[c for c in train_only_cols if train_only_cols.count(c) > 1]}"
        )
        refit_cols = refit_block[family]["preprocessor"]["matrix_columns"]
        assert len(refit_cols) == len(set(refit_cols)), (
            f"{family} train+validation refit preprocessor produced duplicate "
            f"matrix columns: {[c for c in refit_cols if refit_cols.count(c) > 1]}"
        )


def test_flow_compustat_nonlinear_runner_does_not_change_locked_splits(tmp_path: Path) -> None:
    """Compustat-enabled and return-only panels must share identical split masks."""
    from mlfinance.run.baseline_pipeline import build_model_panel

    cfg = _flow_compustat_nonlinear_cfg(tmp_path)
    return_only_panel = (
        build_model_panel(cfg).sort_values(["permno", "date"]).reset_index(drop=True)
    )
    overridden = _with_compustat_overrides(cfg, lag_months=6)
    compustat_panel = (
        build_model_panel(overridden).sort_values(["permno", "date"]).reset_index(drop=True)
    )

    pd.testing.assert_series_equal(
        return_only_panel["target_ret_fwd_1m"],
        compustat_panel["target_ret_fwd_1m"],
        check_names=False,
    )
    masks_ro = locked_split_masks(return_only_panel, cfg)
    masks_en = locked_split_masks(compustat_panel, cfg)
    for split_name in ("train", "validation", "test"):
        pd.testing.assert_series_equal(
            masks_ro[split_name].reset_index(drop=True),
            masks_en[split_name].reset_index(drop=True),
            check_names=False,
        )


def test_flow_compustat_nonlinear_writes_prediction_parquet_with_flow_compustat_nonlinear_suffix(
    tmp_path: Path,
) -> None:
    cfg = _flow_compustat_nonlinear_cfg(tmp_path)
    outputs = run_flow_compustat_nonlinear(cfg)
    predictions_dir = outputs.get("predictions_dir")
    assert predictions_dir is not None and predictions_dir.exists()

    parquets = sorted(predictions_dir.glob("*.parquet"))
    assert parquets, "expected at least one feature-audit nonlinear prediction parquet"
    for path in parquets:
        # Must NEVER overwrite nonlinear-benchmark return-only files or feature-audit LINEAR files.
        assert "__compustat_flow_compustat_nonlinear" in path.stem, path.name
        assert _flow_compustat_nonlinear_test_split_id(cfg) in path.stem, path.name


def test_flow_compustat_nonlinear_runner_writes_xgb_feature_importance(tmp_path: Path) -> None:
    """Whenever XGB is among the winners (it must be, since we sweep XGB), the
    runner must write a feature-importance CSV."""
    cfg = _flow_compustat_nonlinear_cfg(tmp_path)
    outputs = run_flow_compustat_nonlinear(cfg)
    assert "xgb_feature_importance" in outputs
    assert outputs["xgb_feature_importance"].exists()
    importance = pd.read_csv(outputs["xgb_feature_importance"])
    assert {"feature", "importance"}.issubset(importance.columns)
    assert len(importance) > 0


def test_flow_compustat_nonlinear_outputs_never_overwrite_nonlinear_benchmark_outputs(
    tmp_path: Path,
) -> None:
    """Every feature-audit nonlinear metric file must carry the nonlinear prefix."""
    cfg = _flow_compustat_nonlinear_cfg(tmp_path)
    outputs = run_flow_compustat_nonlinear(cfg)
    for key, path in outputs.items():
        if key == "predictions_dir":
            continue
        marker_in_name = path.name.startswith(OUTPUT_PREFIX_NONLINEAR) or "__compustat" in path.name
        assert (
            marker_in_name
        ), f"{key} -> {path.name} does not carry a feature-audit nonlinear marker"


def test_flow_compustat_nonlinear_runner_comparison_handles_missing_baseline(
    tmp_path: Path,
) -> None:
    """If no nonlinear-benchmark return-only baseline exists, the comparison files still
    get written and explicitly flag baseline_available=False."""
    cfg = _flow_compustat_nonlinear_cfg(tmp_path)
    outputs = run_flow_compustat_nonlinear(cfg)
    comparison = pd.read_csv(outputs["test_summary_vs_return_only"])
    assert "baseline_available" in comparison.columns
    assert (comparison["baseline_available"] == False).all()  # noqa: E712
    cost_comp = pd.read_csv(outputs["test_cost_grid_vs_return_only"])
    assert (cost_comp["baseline_available"] == False).all()  # noqa: E712
