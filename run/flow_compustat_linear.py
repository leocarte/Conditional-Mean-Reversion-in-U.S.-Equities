"""feature-audit controlled model runs with the flow-only Compustat feature block.

This runner is the **model-run** half of Compustat integration: it trains and evaluates the
linear-benchmark linear families (Ridge + Elastic Net) on the integrated panel with
``data.use_compustat=true`` and ``features.compustat_features.enabled=true``
(``conservative_lag_months=6``), and produces side-by-side outputs against the
already-saved return-only linear-benchmark baseline so reviewers can read the
incremental effect of the flow-Compustat block directly.

It does NOT:

* change ``cfg.data.target_col`` or ``cfg.data.robustness_target_col``;
* change the locked train / validation / test dates;
* change target construction (``target_ret_fwd_1m`` shift logic stays);
* change split logic, backtest logic, or transaction-cost logic;
* change the on-disk default config flags (Compustat stays disabled by default);
* introduce balance-sheet, ROA, leverage, book-to-market, asset-growth,
  debt-to-assets, assets-scaled or equity-scaled fields (all forbidden by name
  and asserted);
* use any future information or fit on the test split;
* tune on the final test set;
* claim tradability, microcap robustness, causal discovery, or a discovered
  anomaly.

Compustat is enabled via an in-runner OmegaConf override only. The on-disk
``configs/main.yaml`` stays return-only.

Outputs land under git-ignored ``data/outputs/`` paths with a
``compustat_flow_compustat_`` prefix so they cannot collide with the return-only
linear-benchmark artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlfinance.data.compustat_schema import audit_compustat_point_in_time_merge
from mlfinance.features.compustat_flow_features import (
    COMPUSTAT_FLOW_FEATURE_COLUMNS,
    compustat_flow_feature_columns,
)
from mlfinance.run.baseline_pipeline import build_model_panel, summarize_backtests_by_cost
from mlfinance.run.linear_benchmarks import (
    PREDICTION_REQUIRED_COLUMNS,
    RETURN_ONLY_FEATURE_COLUMNS,
    _write_csv,
    _write_json,
    build_test_predictions,
    build_test_summary,
    build_validation_diagnostics,
    run_prediction_backtests,
    select_best_validation_models,
    split_panel,
    summarize_prediction_outputs,
)
from mlfinance.utils.io import write_parquet
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

# Balance-sheet / scaled tokens that this feature-audit-flow runner must refuse to
# include in the model matrix. Mirrors the feature-audit smoke guardrail; tripping
# this list means out-of-scope balance-sheet variables have slipped into the
# reviewed flow pipeline.
FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "roa",
    "leverage",
    "book_to_market",
    "asset_growth",
    "debt_to_assets",
    "assets_scaled",
    "equity_scaled",
)

# Future-information tokens reused from the linear-benchmark runner -- catches any
# accidental inclusion of targets, realized returns, or prediction columns in
# the feature matrix.
FORBIDDEN_FUTURE_INFO_TOKENS: tuple[str, ...] = (
    "target",
    "fwd",
    "future",
    "label",
    "prediction",
    "realized",
)

# Compustat columns that must NEVER enter the model matrix: PIT metadata,
# join keys, and raw accounting fields that the flow features were derived
# from. Anything in this set is filtered out before the forbidden-token check.
EXCLUDED_COMPUSTAT_METADATA: frozenset[str] = frozenset(
    {
        "compustat_datadate",
        "compustat_availability_date",
        "gvkey",
        "cusip",
        "cusip8",
        "hdrcusip",
        "datadate",
        "availability_date",
        "fyearq",
        "fqtr",
        "saley",
        "cogsy",
        "niy",
        "iby",
        "oancfy",
        "capxy",
        "xrdy",
        "xsgay",
        "xintp",
        "xinty",
        "dvy",
        "dpy",
        "oiadpy",
        "oibdpy",
        "piy",
        "txty",
        "revty",
    }
)

# Prefix every feature-audit output with this stem so the artefacts can never be
# mistaken for the linear-benchmark return-only files in the same directories.
OUTPUT_PREFIX = "flow_compustat_linear"


def _with_compustat_overrides(cfg: DictConfig, *, lag_months: int = 6) -> DictConfig:
    """Return ``cfg`` merged with the feature-audit Compustat-enabled overrides.

    The override is in-runner only -- it never touches ``configs/main.yaml`` --
    so the default repo state stays return-only and reviewers can flip the
    feature-audit block on for this specific runner without affecting other stages.
    """
    override = OmegaConf.create(
        {
            "data": {"use_compustat": True},
            "features": {
                "compustat_features": {
                    "enabled": True,
                    "conservative_lag_months": int(lag_months),
                }
            },
        }
    )
    return OmegaConf.merge(cfg, override)


def build_flow_compustat_feature_list(
    panel: pd.DataFrame,
    *,
    include_missingness_flags: bool = True,
) -> list[str]:
    """Build the feature-audit model feature list with explicit allow/deny rules.

    The list is the concatenation of:

    * the return-only linear-benchmark feature whitelist (``RETURN_ONLY_FEATURE_COLUMNS``);
    * the flow-only Compustat features that actually exist on the panel
      (``COMPUSTAT_FLOW_FEATURE_COLUMNS``);
    * if ``include_missingness_flags`` is True, the matching ``*_missing`` flags.

    Then we *explicitly* filter:

    * remove Compustat metadata / join keys / raw accounting columns
      (``EXCLUDED_COMPUSTAT_METADATA``);
    * raise on any column whose name carries a forbidden balance-sheet /
      scaled token (``FORBIDDEN_FEATURE_TOKENS``);
    * raise on any column whose name carries a future-information token
      (``FORBIDDEN_FUTURE_INFO_TOKENS``).

    The function never re-orders the return-only features and never silently
    drops a return-only feature: a missing return-only column is an error,
    because the Compustat-enabled panel must always contain them.
    """
    available = set(panel.columns)

    missing_return = [c for c in RETURN_ONLY_FEATURE_COLUMNS if c not in available]
    if missing_return:
        raise KeyError(
            "feature-audit panel is missing required return-only features: " f"{missing_return}"
        )

    flow_features = [c for c in COMPUSTAT_FLOW_FEATURE_COLUMNS if c in available]
    if not flow_features:
        raise ValueError(
            "feature-audit panel contains no Compustat flow features -- did the "
            "build_model_panel override actually enable use_compustat / "
            "features.compustat_features.enabled?"
        )

    missingness_flags: list[str] = []
    if include_missingness_flags:
        candidate_flags = [
            c
            for c in compustat_flow_feature_columns(add_missingness_flags=True)
            if c.endswith("_missing")
        ]
        missingness_flags = [c for c in candidate_flags if c in available]

    features = list(RETURN_ONLY_FEATURE_COLUMNS) + flow_features + missingness_flags

    metadata_in_list = [c for c in features if c in EXCLUDED_COMPUSTAT_METADATA]
    if metadata_in_list:
        raise ValueError(
            f"Compustat metadata columns must not enter the model matrix: {metadata_in_list}"
        )

    forbidden = [c for c in features if any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if forbidden:
        raise ValueError(
            f"Forbidden balance-sheet / scaled feature columns in feature list: {forbidden}"
        )

    future_info = [
        c for c in features if any(tok in c.lower() for tok in FORBIDDEN_FUTURE_INFO_TOKENS)
    ]
    if future_info:
        raise ValueError(f"Future-information feature columns in feature list: {future_info}")

    return features


def _panel_diagnostics(
    panel: pd.DataFrame,
    *,
    feature_cols: list[str],
    cfg: DictConfig,
    pit_violations: int,
) -> pd.DataFrame:
    """One-row diagnostic frame about the integrated panel.

    Records exactly what the specification asks for: rows, date range, feature counts,
    Compustat feature coverage, missingness rates, PIT result. Reviewers can
    diff this against the same fields from the smoke summary to confirm the
    runner is on the same (permno, date) universe as the smoke.
    """
    date_col = cfg.data.date_col
    n = len(panel)
    return_features = [c for c in feature_cols if c in RETURN_ONLY_FEATURE_COLUMNS]
    compustat_features = [c for c in feature_cols if c in COMPUSTAT_FLOW_FEATURE_COLUMNS]
    missingness_flags = [c for c in feature_cols if c.endswith("_missing")]

    if compustat_features:
        any_compustat = panel[compustat_features].notna().any(axis=1)
    else:
        any_compustat = pd.Series(False, index=panel.index)

    flag_means = {
        f"share_{flag}": (
            round(float(panel[flag].mean()), 4) if flag in panel.columns and n else 0.0
        )
        for flag in missingness_flags
    }

    dates = pd.to_datetime(panel[date_col], errors="coerce").dropna()
    row = {
        "rows": int(n),
        "date_min": dates.min().date().isoformat() if not dates.empty else None,
        "date_max": dates.max().date().isoformat() if not dates.empty else None,
        "n_return_features": len(return_features),
        "n_compustat_features": len(compustat_features),
        "n_missingness_flags": len(missingness_flags),
        "n_total_features": len(feature_cols),
        "rows_with_any_compustat_feature": int(any_compustat.sum()),
        "share_with_any_compustat_feature": (round(float(any_compustat.mean()), 4) if n else 0.0),
        "rows_missing_all_compustat_features": int((~any_compustat).sum()),
        "pit_violations": int(pit_violations),
        "target_col": str(cfg.data.target_col),
        "robustness_target_col": str(
            OmegaConf.select(cfg, "data.robustness_target_col", default="")
        ),
        "train_start": str(cfg.splits.locked.train_start),
        "train_end": str(cfg.splits.locked.train_end),
        "val_start": str(cfg.splits.locked.val_start),
        "val_end": str(cfg.splits.locked.val_end),
        "test_start": str(cfg.splits.locked.test_start),
        "test_end": str(cfg.splits.locked.test_end),
    }
    row.update(flag_means)
    return pd.DataFrame([row])


def _compustat_test_split_id(cfg: DictConfig) -> str:
    test_start = pd.to_datetime(cfg.splits.locked.test_start)
    test_end = pd.to_datetime(cfg.splits.locked.test_end)
    return f"compustat_locked_test_{test_start:%Y%m}_{test_end:%Y%m}"


def _compustat_prediction_path(cfg: DictConfig, model_name: str) -> Path:
    return Path(cfg.paths.predictions) / f"{model_name}__{_compustat_test_split_id(cfg)}.parquet"


def _compustat_metric_path(cfg: DictConfig, stem: str) -> Path:
    return Path(cfg.paths.metrics) / f"{OUTPUT_PREFIX}_{stem}"


def _compustat_backtest_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.backtests) / f"{OUTPUT_PREFIX}_backtests.parquet"


def _save_compustat_predictions(predictions: pd.DataFrame, cfg: DictConfig) -> list[Path]:
    """Write per-family parquets with a ``__compustat`` suffix so they cannot
    overwrite the linear-benchmark return-only prediction parquets."""
    paths: list[Path] = []
    annotated = predictions.copy()
    if not annotated.empty:
        annotated["model_name"] = (
            annotated["model_name"].astype("string") + "__compustat_flow_compustat"
        )
        annotated["model"] = annotated["model_name"]
    for model_name, frame in annotated.groupby("model_name", sort=True):
        path = _compustat_prediction_path(cfg, str(model_name))
        write_parquet(
            frame.reset_index(drop=True),
            path,
            required_columns=PREDICTION_REQUIRED_COLUMNS,
        )
        paths.append(path)
    return paths


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    """Return the CSV at ``path`` if it exists, otherwise None."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):  # pragma: no cover
            logger.warning("Could not parse return-only baseline file: %s", path)
            return None
    return None


def _baseline_test_summary(cfg: DictConfig) -> pd.DataFrame | None:
    """Read the linear-benchmark test summary (return-only) if present."""
    return _read_optional_csv(Path(cfg.paths.metrics) / "linear_benchmarks_test_summary.csv")


def _baseline_cost_summary(cfg: DictConfig) -> pd.DataFrame | None:
    """Read the linear-benchmark cost-grid summary (return-only) if present."""
    return _read_optional_csv(
        Path(cfg.paths.metrics) / "linear_benchmarks_test_cost_grid_summary.csv"
    )


def _build_test_summary_comparison(
    compustat_test_summary: pd.DataFrame,
    baseline_test_summary: pd.DataFrame | None,
    *,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Side-by-side comparison of feature-audit and return-only headline metrics."""
    main_bps = int(cfg.costs.main_bps)
    keep = [
        "model_name",
        "rmse",
        "mae",
        "oos_r2",
        "rank_ic_mean",
        "rank_ic_hac_tstat",
        "top_minus_bottom_mean",
        "gross_sharpe",
        f"net_sharpe_{main_bps}bps",
        f"net_ann_return_{main_bps}bps",
        "avg_turnover",
    ]
    if baseline_test_summary is None or baseline_test_summary.empty:
        comparison = compustat_test_summary[
            [c for c in keep if c in compustat_test_summary.columns]
        ].copy()
        comparison.columns = [
            "model_name" if c == "model_name" else f"compustat_{c}" for c in comparison.columns
        ]
        comparison["baseline_available"] = False
        return comparison.sort_values("model_name").reset_index(drop=True)

    base = baseline_test_summary[[c for c in keep if c in baseline_test_summary.columns]].copy()
    base.columns = ["model_name" if c == "model_name" else f"return_only_{c}" for c in base.columns]
    compustat_summary = compustat_test_summary[
        [c for c in keep if c in compustat_test_summary.columns]
    ].copy()
    compustat_summary.columns = [
        "model_name" if c == "model_name" else f"compustat_{c}" for c in compustat_summary.columns
    ]

    merged = base.merge(compustat_summary, on="model_name", how="outer", validate="one_to_one")
    for metric in keep:
        if metric == "model_name":
            continue
        base_col = f"return_only_{metric}"
        compustat_col = f"compustat_{metric}"
        if base_col in merged.columns and compustat_col in merged.columns:
            merged[f"delta_{metric}"] = merged[compustat_col] - merged[base_col]
    merged["baseline_available"] = True
    return merged.sort_values("model_name").reset_index(drop=True)


def _build_cost_grid_comparison(
    compustat_cost_summary: pd.DataFrame,
    baseline_cost_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compare net Sharpe / avg turnover across the full cost grid."""
    keep = ["model_name", "cost_bps", "gross_sharpe", "net_sharpe", "avg_turnover"]
    compustat_summary = compustat_cost_summary[
        [c for c in keep if c in compustat_cost_summary.columns]
    ].copy()
    compustat_summary.columns = [
        c if c in ("model_name", "cost_bps") else f"compustat_{c}"
        for c in compustat_summary.columns
    ]
    if baseline_cost_summary is None or baseline_cost_summary.empty:
        compustat_summary["baseline_available"] = False
        return compustat_summary.sort_values(["model_name", "cost_bps"]).reset_index(drop=True)

    base = baseline_cost_summary[[c for c in keep if c in baseline_cost_summary.columns]].copy()
    base.columns = [
        c if c in ("model_name", "cost_bps") else f"return_only_{c}" for c in base.columns
    ]
    merged = base.merge(
        compustat_summary, on=["model_name", "cost_bps"], how="outer", validate="one_to_one"
    )
    for metric in ("gross_sharpe", "net_sharpe", "avg_turnover"):
        base_col = f"return_only_{metric}"
        compustat_col = f"compustat_{metric}"
        if base_col in merged.columns and compustat_col in merged.columns:
            merged[f"delta_{metric}"] = merged[compustat_col] - merged[base_col]
    merged["baseline_available"] = True
    return merged.sort_values(["model_name", "cost_bps"]).reset_index(drop=True)


def run_flow_compustat_linear(cfg: DictConfig) -> dict[str, Path]:
    """Train Ridge + Elastic Net with flow-Compustat features and compare to baseline."""
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
    # appears in both the original and the regenerated set). The duplicate
    # names cause ``out[matrix_columns_]`` to return more columns than
    # ``matrix_columns_`` lists (pandas duplicate-name lookup returns ALL
    # matching columns), inflating the actual X matrix to ~145 columns
    # instead of the nominal 122 and slowing Ridge eig + Elastic Net
    # coordinate descent by ~1.4-2x. The model still sees the missingness
    # indicator -- it is just generated by the preprocessor instead of being
    # pre-attached on the panel. Mirrors the nonlinear runner.
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
        diagnostics_frame, _compustat_metric_path(cfg, "panel_diagnostics.csv")
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
        },
        _compustat_metric_path(cfg, "feature_list.json"),
    )

    splits = split_panel(panel, cfg)
    for split_name, frame in splits.items():
        if frame.empty:
            raise ValueError(
                f"feature-audit panel produced an empty {split_name} split -- locked dates "
                "are incompatible with the Compustat-enabled panel."
            )

    validation_diagnostics, train_only_metadata = build_validation_diagnostics(
        splits, cfg, feature_cols
    )
    winners = select_best_validation_models(validation_diagnostics)
    validation_path = _write_csv(
        validation_diagnostics,
        _compustat_metric_path(cfg, "validation_diagnostics.csv"),
    )

    test_predictions, refit_metadata = build_test_predictions(
        splits,
        winners,
        cfg=cfg,
        feature_cols=feature_cols,
    )
    prediction_paths = _save_compustat_predictions(test_predictions, cfg)

    monthly_backtests = run_prediction_backtests(
        test_predictions, cfg, split_id=_compustat_test_split_id(cfg)
    )
    backtest_path = _compustat_backtest_path(cfg)
    write_parquet(monthly_backtests, backtest_path)

    predictive_summary = summarize_prediction_outputs(test_predictions, cfg)
    predictive_summary_path = _write_csv(
        predictive_summary, _compustat_metric_path(cfg, "test_predictive_summary.csv")
    )

    cost_summary = summarize_backtests_by_cost(monthly_backtests)
    cost_summary_path = _write_csv(
        cost_summary, _compustat_metric_path(cfg, "test_cost_grid_summary.csv")
    )

    compustat_test_summary = build_test_summary(predictive_summary, cost_summary, winners, cfg)
    test_summary_path = _write_csv(
        compustat_test_summary, _compustat_metric_path(cfg, "test_summary.csv")
    )

    test_summary_comparison = _build_test_summary_comparison(
        compustat_test_summary, _baseline_test_summary(cfg), cfg=cfg
    )
    test_summary_comparison_path = _write_csv(
        test_summary_comparison,
        _compustat_metric_path(cfg, "test_summary_vs_return_only.csv"),
    )

    cost_grid_comparison = _build_cost_grid_comparison(cost_summary, _baseline_cost_summary(cfg))
    cost_grid_comparison_path = _write_csv(
        cost_grid_comparison,
        _compustat_metric_path(cfg, "test_cost_grid_vs_return_only.csv"),
    )

    caution_payload: dict[str, Any] = {
        "scope": "feature-audit controlled flow-only Compustat model run",
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
        ],
        "selection_metric": f"validation_net_sharpe_at_{int(cfg.costs.main_bps)}bps",
        "selected_models": winners.to_dict(orient="records"),
        "train_only_preprocessing": train_only_metadata["train_only_preprocessing"],
        "test_refit": refit_metadata,
    }
    caution_path = _write_json(
        caution_payload, _compustat_metric_path(cfg, "selection_and_disclosures.json")
    )

    logger.info(
        "feature-audit flow-Compustat models: %d rows, %d total features (%d return + "
        "%d Compustat + %d missingness flags); PIT violations=%d",
        len(panel),
        len(feature_cols),
        sum(c in RETURN_ONLY_FEATURE_COLUMNS for c in feature_cols),
        sum(c in COMPUSTAT_FLOW_FEATURE_COLUMNS for c in feature_cols),
        sum(c.endswith("_missing") for c in feature_cols),
        int(len(pit_frame)),
    )

    outputs = {
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
    if prediction_paths:
        outputs["predictions_dir"] = Path(cfg.paths.predictions)
    return outputs


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    configure_logging(cfg)
    logger.info(
        "Running feature-audit flow-Compustat controlled model run with config:\n%s",
        OmegaConf.to_yaml(cfg),
    )
    outputs = run_flow_compustat_linear(cfg)
    logger.info("feature-audit flow-Compustat outputs:")
    for label, path in outputs.items():
        logger.info("  %s -> %s", label, path)


if __name__ == "__main__":  # pragma: no cover
    main()
