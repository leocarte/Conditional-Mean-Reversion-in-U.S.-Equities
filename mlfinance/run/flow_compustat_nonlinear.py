"""feature-audit controlled NONLINEAR model runs with the flow-only Compustat block.

Sibling of ``flow_compustat_linear.py``: trains and evaluates the
nonlinear-benchmark nonlinear families (XGBoost + compact MLP) on the integrated panel
with ``data.use_compustat=true`` and ``features.compustat_features.enabled=true``
(``conservative_lag_months=6``), and produces side-by-side outputs against the
already-saved return-only nonlinear-benchmark baseline so reviewers can read the
incremental effect of the flow-Compustat block directly.

Shares the feature-audit helpers (panel build override, feature-list builder, panel
diagnostics, comparison tables) with the LINEAR feature-audit runner -- imports them
from ``mlfinance.run.flow_compustat_linear`` rather than duplicating.

Constraints honoured (same as the linear feature-audit runner):
  - default config flags untouched (in-runner OmegaConf override only);
  - ``target_col`` / ``robustness_target_col`` unchanged;
  - locked train/validation/test dates unchanged;
  - target construction / split logic / backtest logic / cost logic unchanged;
  - no balance-sheet / ROA / leverage / book-to-market / asset-growth /
    debt-to-assets / assets-scaled / equity-scaled features (asserted by the
    shared feature-list builder);
  - no future information; no test-set tuning;
  - no generated outputs committed;
  - no tradability / microcap / causal / anomaly claim.

GPU note: XGBoost auto-selects ``device=cuda`` when ``torch.cuda.is_available()``
is True (via ``XGBoostModel._resolve_device``); the compact MLP places its tensors
on CUDA in the same condition. The cluster stage requires ``GPU >= 1`` so the
fit times stay reasonable on the real ~3.6 M-row panel.

Outputs land under git-ignored ``data/outputs/`` with a
``flow_compustat_nonlinear_`` prefix on metric files and a
``__compustat_flow_compustat_nonlinear`` suffix on prediction parquets so they
cannot collide with the nonlinear-benchmark return-only artefacts or the feature-audit
LINEAR artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlfinance.data.compustat_schema import audit_compustat_point_in_time_merge
from mlfinance.features.compustat_flow_features import COMPUSTAT_FLOW_FEATURE_COLUMNS
from mlfinance.run.baseline_pipeline import build_model_panel, summarize_backtests_by_cost
from mlfinance.run.flow_compustat_linear import (
    EXCLUDED_COMPUSTAT_METADATA,
    FORBIDDEN_FEATURE_TOKENS,
    FORBIDDEN_FUTURE_INFO_TOKENS,
    OUTPUT_PREFIX,
    _build_cost_grid_comparison,
    _build_test_summary_comparison,
    _panel_diagnostics,
    _read_optional_csv,
    _with_compustat_overrides,
    build_flow_compustat_feature_list,
)
from mlfinance.run.linear_benchmarks import (
    PREDICTION_REQUIRED_COLUMNS,
    RETURN_ONLY_FEATURE_COLUMNS,
    _write_csv,
    _write_json,
    run_prediction_backtests,
    split_panel,
    summarize_prediction_outputs,
)
from mlfinance.run.nonlinear_benchmarks import (
    MODEL_FAMILIES,
    _jsonable,
    build_test_predictions,
    build_validation_diagnostics,
    select_best_candidates,
)
from mlfinance.utils.io import write_parquet
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

OUTPUT_PREFIX_NONLINEAR = f"{OUTPUT_PREFIX}_nonlinear"


def _nonlinear_metric_path(cfg: DictConfig, stem: str) -> Path:
    return Path(cfg.paths.metrics) / f"{OUTPUT_PREFIX_NONLINEAR}_{stem}"


def _nonlinear_backtest_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.backtests) / f"{OUTPUT_PREFIX_NONLINEAR}_backtests.parquet"


def _flow_compustat_nonlinear_test_split_id(cfg: DictConfig) -> str:
    test_start = pd.to_datetime(cfg.splits.locked.test_start)
    test_end = pd.to_datetime(cfg.splits.locked.test_end)
    return f"flow_compustat_nonlinear_locked_test_{test_start:%Y%m}_{test_end:%Y%m}"


def _nonlinear_prediction_path(cfg: DictConfig, model_name: str) -> Path:
    return (
        Path(cfg.paths.predictions)
        / f"{model_name}__{_flow_compustat_nonlinear_test_split_id(cfg)}.parquet"
    )


def _save_flow_compustat_nonlinear_predictions(
    predictions: pd.DataFrame, cfg: DictConfig
) -> list[Path]:
    """Write XGB/MLP test-prediction parquets with the feature-audit nonlinear suffix.

    Mirrors ``flow_compustat_linear._save_compustat_predictions`` but uses a
    distinct ``__compustat_flow_compustat_nonlinear`` suffix so the files cannot
    overwrite either the nonlinear-benchmark return-only predictions OR the feature-audit
    LINEAR predictions in the same ``data/outputs/predictions/`` directory.
    """
    paths: list[Path] = []
    annotated = predictions.copy()
    if not annotated.empty:
        annotated["model_name"] = (
            annotated["model_name"].astype("string") + "__compustat_flow_compustat_nonlinear"
        )
        annotated["model"] = annotated["model_name"]
    for model_name, frame in annotated.groupby("model_name", sort=True):
        path = _nonlinear_prediction_path(cfg, str(model_name))
        write_parquet(
            frame.reset_index(drop=True),
            path,
            required_columns=PREDICTION_REQUIRED_COLUMNS,
        )
        paths.append(path)
    return paths


def _build_nonlinear_test_summary(
    predictive_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    winners: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Mirror ``linear_benchmarks.build_test_summary`` for nonlinear winners.

    The nonlinear-benchmark winners dataframe carries different hyperparameter columns
    (``params_json``, ``best_epoch``) than the linear winners (``shrinkage``,
    ``alpha``, ``l1_ratio``), so we keep only the columns that actually exist
    and merge them onto the headline metric table.
    """
    main_bps = int(cfg.costs.main_bps)
    main_cost = cost_summary.loc[cost_summary["cost_bps"] == main_bps].copy()
    main_cost = main_cost.rename(
        columns={
            "net_ann_return": f"net_ann_return_{main_bps}bps",
            "net_sharpe": f"net_sharpe_{main_bps}bps",
        }
    )

    selected = winners.copy()
    selected["model_name"] = selected["model_family"]
    candidate_cols = [
        c for c in ("model_name", "params_json", "best_epoch") if c in selected.columns
    ]

    return (
        predictive_summary.merge(
            main_cost[
                [
                    "model_name",
                    "gross_ann_return",
                    "gross_sharpe",
                    f"net_ann_return_{main_bps}bps",
                    f"net_sharpe_{main_bps}bps",
                    "avg_turnover",
                ]
            ],
            on="model_name",
            how="left",
            validate="one_to_one",
        )
        .merge(selected[candidate_cols], on="model_name", how="left", validate="one_to_one")
        .sort_values("model_name")
        .reset_index(drop=True)
    )


def _baseline_nonlinear_test_summary(cfg: DictConfig) -> pd.DataFrame | None:
    """Read the nonlinear-benchmark (return-only) test summary if present."""
    return _read_optional_csv(Path(cfg.paths.metrics) / "nonlinear_benchmarks_test_summary.csv")


def _baseline_nonlinear_cost_summary(cfg: DictConfig) -> pd.DataFrame | None:
    """Read the nonlinear-benchmark (return-only) cost-grid summary if present."""
    return _read_optional_csv(
        Path(cfg.paths.metrics) / "nonlinear_benchmarks_test_cost_grid_summary.csv"
    )


def run_flow_compustat_nonlinear(cfg: DictConfig) -> dict[str, Path]:
    """Train XGBoost + compact MLP with flow-Compustat features; compare to baseline."""
    lag_months = int(
        OmegaConf.select(cfg, "features.compustat_features.conservative_lag_months", default=6)
    )
    enabled_cfg = _with_compustat_overrides(cfg, lag_months=lag_months)

    panel = build_model_panel(enabled_cfg).copy()
    panel = panel.sort_values([cfg.data.date_col, cfg.data.permno_col]).reset_index(drop=True)

    # ``include_missingness_flags=False``: the explicit ``compustat_*_missing``
    # columns carry the same information as what ``LinearFeaturePreprocessor``
    # generates automatically for every input column, and passing both into
    # the preprocessor creates duplicate column names ('compustat_*_missing'
    # appears in both the original and the regenerated set). The model still
    # sees the missingness indicator -- it is just generated by the
    # preprocessor instead of being pre-attached on the panel.
    feature_cols = build_flow_compustat_feature_list(panel, include_missingness_flags=False)

    pit_frame = audit_compustat_point_in_time_merge(
        panel, formation_date_col=cfg.data.date_col, raise_on_fail=False
    )
    if not pit_frame.empty:
        raise ValueError(
            f"PIT audit found {len(pit_frame)} violation(s) before model training: "
            f"{sorted(pit_frame['violation_type'].unique())}."
        )

    diagnostics_frame = _panel_diagnostics(
        panel,
        feature_cols=feature_cols,
        cfg=cfg,
        pit_violations=int(len(pit_frame)),
    )
    diagnostics_path = _write_csv(
        diagnostics_frame, _nonlinear_metric_path(cfg, "panel_diagnostics.csv")
    )

    feature_list_path = _write_json(
        {
            "return_only_features": list(RETURN_ONLY_FEATURE_COLUMNS),
            "compustat_flow_features": [
                c for c in feature_cols if c.startswith("compustat_") and not c.endswith("_missing")
            ],
            "compustat_missingness_flags": [c for c in feature_cols if c.endswith("_missing")],
            "excluded_compustat_metadata": sorted(EXCLUDED_COMPUSTAT_METADATA),
            "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
            "forbidden_future_info_tokens": list(FORBIDDEN_FUTURE_INFO_TOKENS),
            "all_features": feature_cols,
            "n_total_features": len(feature_cols),
            "model_families": list(MODEL_FAMILIES),
        },
        _nonlinear_metric_path(cfg, "feature_list.json"),
    )

    splits = split_panel(panel, cfg)
    for split_name, frame in splits.items():
        if frame.empty:
            raise ValueError(
                f"feature-audit nonlinear panel produced an empty {split_name} split -- locked "
                "dates are incompatible with the Compustat-enabled panel."
            )

    validation_diagnostics, train_only_metadata = build_validation_diagnostics(
        splits, cfg, feature_cols
    )
    winners = select_best_candidates(validation_diagnostics)
    validation_path = _write_csv(
        validation_diagnostics,
        _nonlinear_metric_path(cfg, "validation_diagnostics.csv"),
    )

    test_predictions, refit_metadata, feature_importance = build_test_predictions(
        splits,
        winners,
        cfg=cfg,
        feature_cols=feature_cols,
    )
    prediction_paths = _save_flow_compustat_nonlinear_predictions(test_predictions, cfg)

    monthly_backtests = run_prediction_backtests(
        test_predictions, cfg, split_id=_flow_compustat_nonlinear_test_split_id(cfg)
    )
    backtest_path = _nonlinear_backtest_path(cfg)
    write_parquet(monthly_backtests, backtest_path)

    predictive_summary = summarize_prediction_outputs(test_predictions, cfg)
    predictive_summary_path = _write_csv(
        predictive_summary, _nonlinear_metric_path(cfg, "test_predictive_summary.csv")
    )

    cost_summary = summarize_backtests_by_cost(monthly_backtests)
    cost_summary_path = _write_csv(
        cost_summary, _nonlinear_metric_path(cfg, "test_cost_grid_summary.csv")
    )

    compustat_test_summary = _build_nonlinear_test_summary(
        predictive_summary, cost_summary, winners, cfg
    )
    test_summary_path = _write_csv(
        compustat_test_summary, _nonlinear_metric_path(cfg, "test_summary.csv")
    )

    # Side-by-side vs the nonlinear-benchmark return-only baseline (if it exists on disk).
    test_summary_comparison = _build_test_summary_comparison(
        compustat_test_summary, _baseline_nonlinear_test_summary(cfg), cfg=cfg
    )
    test_summary_comparison_path = _write_csv(
        test_summary_comparison,
        _nonlinear_metric_path(cfg, "test_summary_vs_return_only.csv"),
    )

    cost_grid_comparison = _build_cost_grid_comparison(
        cost_summary, _baseline_nonlinear_cost_summary(cfg)
    )
    cost_grid_comparison_path = _write_csv(
        cost_grid_comparison,
        _nonlinear_metric_path(cfg, "test_cost_grid_vs_return_only.csv"),
    )

    # XGBoost feature importance -- only written when XGB is among the winners.
    feature_importance_path: Path | None = None
    if feature_importance is not None and not feature_importance.empty:
        feature_importance_path = _write_csv(
            feature_importance,
            _nonlinear_metric_path(cfg, "xgb_feature_importance.csv"),
        )

    caution_payload: dict[str, Any] = {
        "scope": "feature-audit controlled flow-only Compustat nonlinear model run",
        "model_families": list(MODEL_FAMILIES),
        "constraints_honoured": {
            "default_config_unchanged": True,
            "target_col_unchanged": True,
            "robustness_target_col_unchanged": True,
            "split_dates_unchanged": True,
            "backtest_logic_unchanged": True,
            "transaction_cost_logic_unchanged": True,
            "no_balance_sheet_features": True,
            "no_future_information": True,
        },
        "disclosures": [
            "Compustat <-> CRSP link is keyed on CUSIP8 only (no CCM linktable on the Drive).",
            "Roughly one third of panel rows have no Compustat feature ("
            "~14% no CUSIP8 link, ~19% linked firms whose Compustat history "
            "starts after the panel row's date).",
            "The provided Monthly CRSP still lacks prc / shrout / shrcd / exchcd, "
            "so no true price / common-share / exchange / microcap screen is "
            "applied -- this is not a tradable-anomaly claim.",
            "No causal interpretation. The runner reports incremental "
            "predictive and (flat-bps) backtest metrics only.",
            "XGBoost feature importance is reported as a diagnostic, NOT as a "
            "causal attribution; tree-based importance is biased by feature "
            "scale and multicollinearity.",
        ],
        "selection_metric": f"validation_net_sharpe_at_{int(cfg.costs.main_bps)}bps",
        "selected_models": _jsonable(winners.to_dict(orient="records")),
        "train_only_preprocessing": train_only_metadata,
        "test_refit": refit_metadata,
    }
    caution_path = _write_json(
        caution_payload, _nonlinear_metric_path(cfg, "selection_and_disclosures.json")
    )

    logger.info(
        "feature-audit flow-Compustat nonlinear: %d rows, %d total features (%d return + "
        "%d Compustat + %d missingness flags); PIT violations=%d; winners=%s",
        len(panel),
        len(feature_cols),
        sum(c in RETURN_ONLY_FEATURE_COLUMNS for c in feature_cols),
        sum(c in COMPUSTAT_FLOW_FEATURE_COLUMNS for c in feature_cols),
        sum(c.endswith("_missing") for c in feature_cols),
        int(len(pit_frame)),
        list(winners["model_family"]) if not winners.empty else [],
    )

    outputs: dict[str, Path] = {
        "panel_diagnostics": diagnostics_path,
        "feature_list": feature_list_path,
        "validation_diagnostics": validation_path,
        "test_predictive_summary": predictive_summary_path,
        "test_cost_grid_summary": cost_summary_path,
        "test_summary": test_summary_path,
        "test_summary_vs_return_only": test_summary_comparison_path,
        "test_cost_grid_vs_return_only": cost_grid_comparison_path,
        "selection_and_disclosures": caution_path,
        "backtests": backtest_path,
    }
    if feature_importance_path is not None:
        outputs["xgb_feature_importance"] = feature_importance_path
    if prediction_paths:
        outputs["predictions_dir"] = Path(cfg.paths.predictions)
    return outputs


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    configure_logging(cfg)
    logger.info(
        "Running feature-audit flow-Compustat NONLINEAR model run with config:\n%s",
        OmegaConf.to_yaml(cfg),
    )
    outputs = run_flow_compustat_nonlinear(cfg)
    logger.info("feature-audit flow-Compustat nonlinear outputs:")
    for label, path in outputs.items():
        logger.info("  %s -> %s", label, path)


if __name__ == "__main__":  # pragma: no cover
    main()
