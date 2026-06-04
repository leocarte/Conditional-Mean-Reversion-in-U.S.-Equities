"""Synthetic tests for the feature-audit controlled flow-Compustat model-run path.

These tests cover the contract spelled out in the feature-audit-model-run brief:

* the on-disk default config never enables Compustat;
* ``build_flow_compustat_feature_list`` includes return-only + flow Compustat +
  (optional) missingness flags;
* the same builder excludes Compustat metadata / join keys / raw accounting
  columns, and raises on any balance-sheet / future-information token;
* the runner does not change ``target_col`` / ``robustness_target_col`` /
  locked split dates / locked split masks;
* a tiny synthetic panel runs the runner end-to-end and writes every promised
  artefact, including the side-by-side comparison against the saved return-only
  linear-benchmark baseline when present (and a graceful "baseline_available=False"
  table when it isn't).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlfinance.run.flow_compustat_linear import (
    EXCLUDED_COMPUSTAT_METADATA,
    FORBIDDEN_FEATURE_TOKENS,
    OUTPUT_PREFIX,
    _compustat_test_split_id,
    _with_compustat_overrides,
    build_flow_compustat_feature_list,
    run_flow_compustat_linear,
)
from mlfinance.run.linear_benchmarks import RETURN_ONLY_FEATURE_COLUMNS, locked_split_masks

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Default-config guardrail
# ---------------------------------------------------------------------------


def test_default_config_keeps_compustat_disabled() -> None:
    """The on-disk default config must keep Compustat disabled."""
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "main.yaml")
    assert cfg.data.use_compustat is False
    assert cfg.features.compustat_features.enabled is False
    # Conservative reporting lag is still wired up so the override below has
    # something to merge over.
    assert int(cfg.features.compustat_features.conservative_lag_months) == 6


def test_with_compustat_overrides_does_not_mutate_input() -> None:
    """``_with_compustat_overrides`` must produce a NEW cfg, leaving the input alone."""
    base = OmegaConf.create(
        {
            "data": {"use_compustat": False},
            "features": {
                "compustat_features": {"enabled": False, "conservative_lag_months": 6},
            },
        }
    )
    overridden = _with_compustat_overrides(base, lag_months=6)
    # Input untouched.
    assert base.data.use_compustat is False
    assert base.features.compustat_features.enabled is False
    # Output flipped.
    assert overridden.data.use_compustat is True
    assert overridden.features.compustat_features.enabled is True
    assert int(overridden.features.compustat_features.conservative_lag_months) == 6


# ---------------------------------------------------------------------------
# Feature-list builder
# ---------------------------------------------------------------------------


def _minimal_compustat_panel() -> pd.DataFrame:
    """Synthetic panel that carries every column the feature-list builder cares about."""
    n = 6
    rows: dict[str, list] = {col: [0.0] * n for col in RETURN_ONLY_FEATURE_COLUMNS}
    # A couple of representative flow features + their missingness flags. The
    # builder must pick up everything that exists, not only these.
    rows["compustat_sales_log1p"] = [1.0] * n
    rows["compustat_gross_margin"] = [0.2] * n
    rows["compustat_net_margin"] = [0.05] * n
    rows["compustat_sales_log1p_missing"] = [0] * n
    rows["compustat_gross_margin_missing"] = [0] * n
    rows["compustat_net_margin_missing"] = [0] * n
    # Metadata + raw accounting columns that must be excluded by name.
    rows["compustat_datadate"] = pd.to_datetime(["2020-03-31"] * n)
    rows["compustat_availability_date"] = pd.to_datetime(["2020-09-30"] * n)
    rows["cusip8"] = ["12345678"] * n
    rows["gvkey"] = ["001000"] * n
    rows["saley"] = [100.0] * n
    rows["cogsy"] = [60.0] * n
    # Required panel scaffolding (date / permno / target) so anything that wants
    # to query the panel still finds the usual columns.
    rows["permno"] = list(range(10001, 10001 + n))
    rows["date"] = pd.date_range("2020-03-31", periods=n, freq="ME")
    rows["target_ret_fwd_1m"] = np.linspace(0.01, 0.06, n)
    return pd.DataFrame(rows)


def test_compustat_feature_list_includes_return_compustat_and_missingness_flags() -> None:
    panel = _minimal_compustat_panel()
    features = build_flow_compustat_feature_list(panel, include_missingness_flags=True)

    assert set(RETURN_ONLY_FEATURE_COLUMNS).issubset(features)
    assert {"compustat_sales_log1p", "compustat_gross_margin", "compustat_net_margin"}.issubset(
        features
    )
    assert {
        "compustat_sales_log1p_missing",
        "compustat_gross_margin_missing",
        "compustat_net_margin_missing",
    }.issubset(features)
    # Return-only block must come first (the ordering is part of the contract
    # for downstream feature-coefficient inspection).
    assert features[: len(RETURN_ONLY_FEATURE_COLUMNS)] == list(RETURN_ONLY_FEATURE_COLUMNS)


def test_compustat_feature_list_can_disable_missingness_flags() -> None:
    panel = _minimal_compustat_panel()
    features = build_flow_compustat_feature_list(panel, include_missingness_flags=False)
    assert not any(c.endswith("_missing") for c in features)
    # Flow features still present.
    assert "compustat_sales_log1p" in features


def test_compustat_feature_list_excludes_metadata_and_raw_accounting() -> None:
    panel = _minimal_compustat_panel()
    features = build_flow_compustat_feature_list(panel, include_missingness_flags=True)
    for forbidden_meta in EXCLUDED_COMPUSTAT_METADATA:
        assert forbidden_meta not in features, forbidden_meta


@pytest.mark.parametrize("forbidden_name", list(FORBIDDEN_FEATURE_TOKENS))
def test_compustat_feature_list_raises_on_balance_sheet_token(
    forbidden_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For every forbidden balance-sheet / scaled token, a column carrying that
    token in its substring must trip the guard if it ever appears in the
    Compustat flow whitelist."""
    from mlfinance.run import flow_compustat_linear as runner_module

    fake_col = f"compustat_{forbidden_name}_proxy"
    monkeypatch.setattr(
        runner_module,
        "COMPUSTAT_FLOW_FEATURE_COLUMNS",
        list(runner_module.COMPUSTAT_FLOW_FEATURE_COLUMNS) + [fake_col],
    )

    panel = _minimal_compustat_panel()
    panel[fake_col] = 0.0

    with pytest.raises(ValueError, match="Forbidden balance-sheet"):
        build_flow_compustat_feature_list(panel, include_missingness_flags=False)


def test_compustat_feature_list_raises_when_future_info_token_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future-information names (target / fwd / future / label / prediction /
    realized) must trip the guard if they sneak into the flow whitelist."""
    from mlfinance.run import flow_compustat_linear as runner_module

    monkeypatch.setattr(
        runner_module,
        "COMPUSTAT_FLOW_FEATURE_COLUMNS",
        list(runner_module.COMPUSTAT_FLOW_FEATURE_COLUMNS) + ["compustat_target_proxy"],
    )

    panel = _minimal_compustat_panel()
    panel["compustat_target_proxy"] = 0.0

    with pytest.raises(ValueError, match="Future-information"):
        build_flow_compustat_feature_list(panel, include_missingness_flags=False)


def test_compustat_feature_list_raises_when_no_flow_features_present() -> None:
    """If the Compustat block did not actually attach, the builder must refuse
    to silently fall back to return-only -- that would defeat the experiment."""
    panel = _minimal_compustat_panel().drop(
        columns=[c for c in _minimal_compustat_panel().columns if c.startswith("compustat_")]
    )
    with pytest.raises(ValueError, match="no Compustat flow features"):
        build_flow_compustat_feature_list(panel, include_missingness_flags=True)


def test_compustat_feature_list_raises_on_missing_return_features() -> None:
    """Removing a required return-only feature is an error -- the
    Compustat-enabled panel must still contain the full return block."""
    panel = _minimal_compustat_panel().drop(columns=["mom_2_12"])
    with pytest.raises(KeyError, match="return-only features"):
        build_flow_compustat_feature_list(panel, include_missingness_flags=True)


# ---------------------------------------------------------------------------
# Synthetic end-to-end smoke
# ---------------------------------------------------------------------------


def _write_synthetic_crsp(path: Path) -> None:
    dates = pd.date_range("2020-01-31", "2021-12-31", freq="ME")
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    permnos = [10001, 10002, 10003]
    cusip_roots = ["11111111", "22222222", "33333333"]
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
    rows: list[dict[str, object]] = []
    for cusip_root in ("11111111", "22222222", "33333333"):
        gvkey = "001" + cusip_root[-3:]
        for q_idx, datadate in enumerate(quarters):
            fy = int(datadate.year)
            fq = ((datadate.month - 1) // 3) + 1
            scale = 100.0 + 10.0 * q_idx + int(cusip_root[0]) * 20.0
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


def _compustat_cfg(tmp_path: Path) -> OmegaConf:
    """Tiny config that exercises the full feature-audit model-run path on synthetic data."""
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
                    "train_start": "2020-03-31",
                    "train_end": "2021-06-30",
                    "val_start": "2021-07-31",
                    "val_end": "2021-09-30",
                    "test_start": "2021-10-31",
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
                "ridge": {"shrinkage_grid": [0.1, 1.0, 10.0]},
                "elastic_net": {
                    "alpha_grid": [0.01, 0.1],
                    "l1_ratio_grid": [0.5],
                    "max_iter": 1000,
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


def test_compustat_runner_runs_end_to_end_on_synthetic_panel(tmp_path: Path) -> None:
    cfg = _compustat_cfg(tmp_path)
    outputs = run_flow_compustat_linear(cfg)

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

    # Panel diagnostics carry the panel-level facts the brief requires.
    diag = pd.read_csv(outputs["panel_diagnostics"]).iloc[0]
    assert int(diag["pit_violations"]) == 0
    assert int(diag["n_return_features"]) == len(RETURN_ONLY_FEATURE_COLUMNS)
    assert int(diag["n_compustat_features"]) > 0
    # The linear runner uses ``include_missingness_flags=False`` so the model
    # feature list itself does not carry the *_missing flags; the
    # ``LinearFeaturePreprocessor`` regenerates them automatically before the
    # Ridge / EN fit. Panel diagnostics report 0 flags in the MODEL feature
    # list (the panel still contains the underlying compustat_*_missing
    # columns; they are simply excluded from the model matrix).
    assert int(diag["n_missingness_flags"]) == 0
    assert diag["target_col"] == "target_ret_fwd_1m"
    assert diag["train_start"] == "2020-03-31"
    assert diag["test_end"] == "2021-11-30"

    # The runner must NOT mutate the input config flags.
    assert cfg.data.use_compustat is False
    assert cfg.features.compustat_features.enabled is False

    # Direct postcondition check on the actual invariant: the X matrix that
    # Ridge / EN see must have no duplicate column names. The ``n_missingness_flags``
    # check above asserts the INPUT feature list is clean (the historical root
    # cause of duplicates); this check asserts the OUTPUT matrix itself is
    # clean, so a future regression that introduces duplicates via a different
    # path (e.g. a refactor of ``LinearFeaturePreprocessor`` internals) still
    # fails this test loudly. Covers both the train-only validation-sweep
    # preprocessor and the train+validation refit preprocessor, for both model
    # families.
    disclosures = json.loads(outputs["selection_and_disclosures"].read_text())
    for family in ("ridge", "elastic_net"):
        train_only_cols = disclosures["train_only_preprocessing"][family]["preprocessor"][
            "matrix_columns"
        ]
        assert len(train_only_cols) == len(set(train_only_cols)), (
            f"{family} train-only preprocessor produced duplicate matrix columns: "
            f"{[c for c in train_only_cols if train_only_cols.count(c) > 1]}"
        )
        refit_cols = disclosures["test_refit"][family]["preprocessor"]["matrix_columns"]
        assert len(refit_cols) == len(set(refit_cols)), (
            f"{family} train+validation refit preprocessor produced duplicate "
            f"matrix columns: {[c for c in refit_cols if refit_cols.count(c) > 1]}"
        )


def test_compustat_runner_does_not_change_locked_splits(tmp_path: Path) -> None:
    """The Compustat-enabled panel must have the same (permno, date) split
    universe as a return-only panel built off the same CRSP. Tests that
    enabling Compustat is invariant for splits at the row level."""
    from mlfinance.run.baseline_pipeline import build_model_panel

    cfg = _compustat_cfg(tmp_path)
    return_only_panel = (
        build_model_panel(cfg).sort_values(["permno", "date"]).reset_index(drop=True)
    )
    overridden = _with_compustat_overrides(cfg, lag_months=6)
    compustat_panel = (
        build_model_panel(overridden).sort_values(["permno", "date"]).reset_index(drop=True)
    )

    # Targets identical.
    pd.testing.assert_series_equal(
        return_only_panel["target_ret_fwd_1m"],
        compustat_panel["target_ret_fwd_1m"],
        check_names=False,
    )
    # Locked split masks identical.
    masks_ro = locked_split_masks(return_only_panel, cfg)
    masks_en = locked_split_masks(compustat_panel, cfg)
    for split_name in ("train", "validation", "test"):
        pd.testing.assert_series_equal(
            masks_ro[split_name].reset_index(drop=True),
            masks_en[split_name].reset_index(drop=True),
            check_names=False,
        )


def test_compustat_runner_writes_prediction_parquet_with_compustat_suffix(tmp_path: Path) -> None:
    cfg = _compustat_cfg(tmp_path)
    outputs = run_flow_compustat_linear(cfg)
    predictions_dir = outputs.get("predictions_dir")
    assert predictions_dir is not None and predictions_dir.exists()

    parquets = sorted(predictions_dir.glob("*.parquet"))
    assert parquets, "expected at least one feature-audit prediction parquet"
    for path in parquets:
        # The feature-audit model name suffix must be on every emitted prediction.
        assert "__compustat_flow_compustat" in path.stem, path.name
        assert _compustat_test_split_id(cfg) in path.stem


def test_compustat_runner_comparison_handles_missing_baseline(tmp_path: Path) -> None:
    """If no linear-benchmark return-only baseline exists yet (e.g. fresh repo),
    the comparison files still get written and explicitly flag
    ``baseline_available=False`` -- the runner must not crash on a missing
    return-only file."""
    cfg = _compustat_cfg(tmp_path)
    outputs = run_flow_compustat_linear(cfg)
    comparison = pd.read_csv(outputs["test_summary_vs_return_only"])
    assert "baseline_available" in comparison.columns
    assert (comparison["baseline_available"] == False).all()  # noqa: E712
    cost_comp = pd.read_csv(outputs["test_cost_grid_vs_return_only"])
    assert (cost_comp["baseline_available"] == False).all()  # noqa: E712


def test_compustat_runner_comparison_merges_existing_baseline(tmp_path: Path) -> None:
    """When a return-only linear-benchmark test summary already exists, the
    comparison table must merge it onto the feature-audit results and surface deltas."""
    cfg = _compustat_cfg(tmp_path)
    metrics_dir = Path(cfg.paths.metrics)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Fake baseline that exactly mirrors the columns build_test_summary emits.
    main_bps = int(cfg.costs.main_bps)
    baseline = pd.DataFrame(
        {
            "model_name": ["ridge", "elastic_net"],
            "rmse": [0.1, 0.1],
            "mae": [0.08, 0.08],
            "oos_r2": [-0.01, -0.02],
            "rank_ic_mean": [0.01, 0.0],
            "rank_ic_hac_tstat": [0.5, 0.0],
            "top_minus_bottom_mean": [0.005, 0.0],
            "n_predictions": [6, 6],
            "n_months": [2, 2],
            "realized_start": ["2021-11-30"] * 2,
            "realized_end": ["2021-12-31"] * 2,
            "split": ["test", "test"],
            "gross_ann_return": [0.05, 0.04],
            "gross_sharpe": [0.5, 0.4],
            f"net_ann_return_{main_bps}bps": [0.03, 0.02],
            f"net_sharpe_{main_bps}bps": [0.3, 0.2],
            "avg_turnover": [0.4, 0.4],
            "shrinkage": [1.0, np.nan],
            "alpha": [np.nan, 0.01],
            "l1_ratio": [np.nan, 0.5],
        }
    )
    baseline.to_csv(metrics_dir / "linear_benchmarks_test_summary.csv", index=False)

    # Minimal baseline cost grid -- one row per (model, cost_bps).
    cost_rows = []
    for model_name in ("ridge", "elastic_net"):
        for cost_bps in cfg.costs.bps_grid:
            cost_rows.append(
                {
                    "model_name": model_name,
                    "cost_bps": int(cost_bps),
                    "gross_sharpe": 0.4,
                    "net_sharpe": 0.2,
                    "avg_turnover": 0.4,
                }
            )
    pd.DataFrame(cost_rows).to_csv(
        metrics_dir / "linear_benchmarks_test_cost_grid_summary.csv", index=False
    )

    outputs = run_flow_compustat_linear(cfg)
    comparison = pd.read_csv(outputs["test_summary_vs_return_only"])
    assert (comparison["baseline_available"] == True).all()  # noqa: E712
    assert "compustat_oos_r2" in comparison.columns
    assert "return_only_oos_r2" in comparison.columns
    assert "delta_oos_r2" in comparison.columns

    cost_comp = pd.read_csv(outputs["test_cost_grid_vs_return_only"])
    assert "compustat_net_sharpe" in cost_comp.columns
    assert "return_only_net_sharpe" in cost_comp.columns
    assert "delta_net_sharpe" in cost_comp.columns


def test_compustat_runner_does_not_overwrite_linear_benchmarks_outputs(tmp_path: Path) -> None:
    """Every feature-audit output must carry the feature-audit prefix and never clobber
    a linear-benchmark return-only file with the same role."""
    cfg = _compustat_cfg(tmp_path)
    outputs = run_flow_compustat_linear(cfg)
    for key, path in outputs.items():
        if key == "predictions_dir":
            continue
        # Either prefixed with OUTPUT_PREFIX (metrics + backtests) or under the
        # tmp-path predictions directory with the feature-audit suffix on its stem.
        assert (
            path.name.startswith(OUTPUT_PREFIX) or "__compustat" in path.name
        ), f"{key} -> {path.name} does not carry a feature-audit marker"
