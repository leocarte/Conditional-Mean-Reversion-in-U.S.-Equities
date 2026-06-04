"""nonlinear benchmark locked-split return-only XGBoost and compact MLP pipeline.

This runner conforms to the shared benchmark contract established by
``linear_benchmarks.py``: the same return-only feature list, the same locked
train/validation/test split, the same prediction/backtest schemas, and the
same selection metric (validation long-short net Sharpe at the default cost).
It reuses those helpers directly rather than re-implementing them, and only
adds the two non-linear model families:

* **Model A - XGBoost** (``mlfinance.models.xgb.XGBoostModel``): a small grid
  over depth x tree-count is swept on validation; no early stopping is used in
  the sweep so validation is purely a selection signal.
* **Model B - compact MLP** (``mlfinance.models.mlp_tabular.MLPRegressor``):
  a small grid over dropout, with validation-MSE early stopping during the
  sweep; the validated best-epoch count is then reused to refit on
  train+validation without peeking at the test window.

Both families share one ``LinearFeaturePreprocessor`` fit (per split), and the
panel is loaded through the shared benchmark projected loader so the heavy
CRSP string identifier columns are never read. Final-test predictions are
produced only after validation selection and are the only predictions saved to
disk.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlfinance.models.linear_preprocessing import LinearFeaturePreprocessor
from mlfinance.models.mlp_tabular import MLPRegressor
from mlfinance.models.xgb import XGBoostModel
from mlfinance.run.baseline_pipeline import summarize_backtests_by_cost
from mlfinance.run.linear_benchmarks import (
    PREDICTION_REQUIRED_COLUMNS,
    _as_date_string,
    _prediction_frame,
    _split_id,
    _validation_split_id,
    build_return_only_feature_list,
    load_model_panel,
    run_prediction_backtests,
    split_panel,
    summarize_prediction_outputs,
)
from mlfinance.utils.io import read_parquet_schema_names, write_parquet
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

MODEL_FAMILIES = ("xgboost", "mlp")
DEFAULT_SEED = 1


# ---------------------------------------------------------------------------
# Output paths (mirror the linear benchmark naming, with a nonlinear_benchmarks_*
# prefix so the two runners never collide on disk).
# ---------------------------------------------------------------------------
def _prediction_output_path(cfg: DictConfig, model_name: str) -> Path:
    return Path(cfg.paths.predictions) / f"{model_name}__{_split_id(cfg)}.parquet"


def _backtest_output_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.backtests) / "nonlinear_benchmarks_backtests.parquet"


def _validation_diagnostics_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "nonlinear_benchmarks_validation_diagnostics.csv"


def _test_predictive_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "nonlinear_benchmarks_test_predictive_summary.csv"


def _test_cost_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "nonlinear_benchmarks_test_cost_grid_summary.csv"


def _test_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "nonlinear_benchmarks_test_summary.csv"


def _feature_list_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "nonlinear_benchmarks_feature_list.json"


def _selection_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "nonlinear_benchmarks_selected_hyperparameters.json"


def _feature_importance_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "nonlinear_benchmarks_xgb_feature_importance.csv"


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy scalars / NaN into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) else value
    return obj


# ---------------------------------------------------------------------------
# Hyperparameter grids (read additively from configs/main.yaml).
# ---------------------------------------------------------------------------
def _xgb_candidates(cfg: DictConfig) -> list[dict[str, Any]]:
    xgb_cfg = cfg.models.xgb
    n_estimators_grid = [int(x) for x in xgb_cfg.n_estimators_grid]
    max_depth_grid = [int(x) for x in xgb_cfg.max_depth_grid]
    base = {
        "learning_rate": float(xgb_cfg.learning_rate),
        "subsample": float(xgb_cfg.subsample),
        "colsample_bytree": float(xgb_cfg.colsample_bytree),
        "reg_lambda": float(xgb_cfg.reg_lambda),
        "reg_alpha": float(OmegaConf.select(cfg, "models.xgb.reg_alpha", default=0.0)),
    }
    candidates: list[dict[str, Any]] = []
    for n_estimators in n_estimators_grid:
        for max_depth in max_depth_grid:
            candidates.append({"n_estimators": n_estimators, "max_depth": max_depth, **base})
    return candidates


def _mlp_candidates(cfg: DictConfig) -> list[dict[str, Any]]:
    mlp_cfg = cfg.models.mlp
    dropout_grid = [
        float(x)
        for x in OmegaConf.select(cfg, "models.mlp.dropout_grid", default=[float(mlp_cfg.dropout)])
    ]
    base = {
        "hidden_dims": [int(x) for x in mlp_cfg.hidden_dims],
        "activation": str(mlp_cfg.activation),
        "use_layer_norm": bool(OmegaConf.select(cfg, "models.mlp.use_layer_norm", default=True)),
        "use_bf16": bool(OmegaConf.select(cfg, "models.mlp.use_bf16", default=True)),
        "lr": float(mlp_cfg.lr),
        "weight_decay": float(mlp_cfg.wd),
        "batch_size": int(mlp_cfg.batch_size),
        "epochs": int(mlp_cfg.epochs),
        "patience": int(mlp_cfg.patience),
        "n_seeds": int(OmegaConf.select(cfg, "models.mlp.n_seeds", default=1)),
    }
    return [{**base, "dropout": dropout} for dropout in dropout_grid]


# ---------------------------------------------------------------------------
# Validation sweep: fit each candidate on the shared train matrix, score on val.
# The preprocessor is fit once in build_validation_diagnostics and the
# transformed matrices are reused here for both families.
# ---------------------------------------------------------------------------
def _xgb_validation_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    validation_frame: pd.DataFrame,
    *,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    for idx, params in enumerate(_xgb_candidates(cfg)):
        model = XGBoostModel(seed=DEFAULT_SEED, **params)
        model.fit(x_train, y_train)
        model_name = f"xgboost__cand{idx:02d}"
        outputs.append(
            _prediction_frame(
                validation_frame,
                model.predict(x_val),
                cfg=cfg,
                model_name=model_name,
                split="validation",
                split_id=_validation_split_id(cfg),
            )
        )
        meta_rows.append(
            {
                "candidate_rank": idx,
                "model_family": "xgboost",
                "model_name": model_name,
                "params_json": json.dumps(params, sort_keys=True),
                "best_epoch": np.nan,
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(meta_rows)


def _mlp_validation_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    validation_frame: pd.DataFrame,
    *,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    for idx, params in enumerate(_mlp_candidates(cfg)):
        model = MLPRegressor(seed=DEFAULT_SEED, **params)
        model.fit(x_train, y_train, x_val, y_val)
        model_name = f"mlp__cand{idx:02d}"
        outputs.append(
            _prediction_frame(
                validation_frame,
                model.predict(x_val),
                cfg=cfg,
                model_name=model_name,
                split="validation",
                split_id=_validation_split_id(cfg),
            )
        )
        meta_rows.append(
            {
                "candidate_rank": idx,
                "model_family": "mlp",
                "model_name": model_name,
                "params_json": json.dumps(params, sort_keys=True),
                "best_epoch": float(model.best_epoch_ if model.best_epoch_ is not None else 0),
            }
        )
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(meta_rows)


def build_validation_diagnostics(
    splits: dict[str, pd.DataFrame],
    cfg: DictConfig,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_frame = splits["train"]
    validation_frame = splits["validation"]

    # Fit the shared preprocessor ONCE on train and reuse the transformed
    # matrices for both the XGBoost and the MLP sweeps (no duplicate fits).
    preprocessor = LinearFeaturePreprocessor(feature_cols, date_col=cfg.data.date_col)
    x_train = preprocessor.fit_transform(train_frame).to_numpy()
    x_val = preprocessor.transform(validation_frame).to_numpy()
    y_train = train_frame[cfg.data.target_col].to_numpy(dtype="float64")
    y_val = validation_frame[cfg.data.target_col].to_numpy(dtype="float64")
    fit_meta = {"preprocessor": preprocessor.to_dict()}

    xgb_preds, xgb_meta = _xgb_validation_predictions(
        x_train, y_train, x_val, validation_frame, cfg=cfg
    )
    mlp_preds, mlp_meta = _mlp_validation_predictions(
        x_train, y_train, x_val, y_val, validation_frame, cfg=cfg
    )

    validation_predictions = pd.concat([xgb_preds, mlp_preds], ignore_index=True)
    candidate_meta = pd.concat([xgb_meta, mlp_meta], ignore_index=True)
    if validation_predictions.empty:
        raise ValueError(
            "No validation predictions were produced; check that the validation window "
            "in configs/main.yaml overlaps the model-panel date range."
        )

    predictive = summarize_prediction_outputs(validation_predictions, cfg)
    backtests = run_prediction_backtests(
        validation_predictions, cfg, split_id=_validation_split_id(cfg)
    )
    cost_summary = summarize_backtests_by_cost(backtests)
    main_bps = int(cfg.costs.main_bps)
    selection_metrics = cost_summary.loc[cost_summary["cost_bps"] == main_bps].rename(
        columns={
            "net_sharpe": "validation_net_sharpe",
            "gross_sharpe": "validation_gross_sharpe",
            "net_ann_return": "validation_net_ann_return",
            "avg_turnover": "validation_avg_turnover",
        }
    )[
        [
            "model_name",
            "validation_net_sharpe",
            "validation_gross_sharpe",
            "validation_net_ann_return",
            "validation_avg_turnover",
        ]
    ]

    diagnostics = (
        candidate_meta.merge(predictive, on="model_name", how="left", validate="one_to_one")
        .merge(selection_metrics, on="model_name", how="left", validate="one_to_one")
        .sort_values(["model_family", "candidate_rank"], kind="mergesort")
        .reset_index(drop=True)
    )
    diagnostics["selection_metric"] = "validation_net_sharpe"
    diagnostics["selection_cost_bps"] = main_bps

    metadata = {"train_only_preprocessing": {"xgboost": fit_meta, "mlp": fit_meta}}
    return diagnostics, metadata


def select_best_candidates(diagnostics: pd.DataFrame) -> pd.DataFrame:
    winners: list[pd.Series] = []
    for family in MODEL_FAMILIES:
        subset = diagnostics.loc[diagnostics["model_family"] == family].copy()
        if subset.empty:
            raise ValueError(f"No validation diagnostics available for model family {family!r}.")
        subset = subset.sort_values(
            ["validation_net_sharpe", "rank_ic_mean", "oos_r2", "rmse", "candidate_rank"],
            ascending=[False, False, False, True, True],
            na_position="last",
            kind="mergesort",
        )
        winners.append(subset.iloc[0])
    return pd.DataFrame(winners).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final refit on train+validation, predict the locked test window only.
# The preprocessor is fit once in build_test_predictions; the selected XGBoost
# and MLP both reuse the same transformed train+val / test matrices.
# ---------------------------------------------------------------------------
def _fit_selected_xgb(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_pred: np.ndarray,
    feature_names: list[str],
    *,
    params: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    model = XGBoostModel(seed=DEFAULT_SEED, **params)
    model.fit(x_fit, y_fit)
    importances = np.asarray(model.feature_importances_, dtype="float64")
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return model.predict(x_pred), feature_importance


def _fit_selected_mlp(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_pred: np.ndarray,
    *,
    params: dict[str, Any],
    best_epoch: int,
) -> tuple[np.ndarray, int]:
    refit_params = dict(params)
    refit_params["epochs"] = max(int(best_epoch), 1)
    model = MLPRegressor(seed=DEFAULT_SEED, **refit_params)
    model.fit(x_fit, y_fit)  # no validation frame -> trains the validated epoch budget
    return model.predict(x_pred), refit_params["epochs"]


def build_test_predictions(
    splits: dict[str, pd.DataFrame],
    winners: pd.DataFrame,
    *,
    cfg: DictConfig,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame | None]:
    fit_frame = (
        pd.concat([splits["train"], splits["validation"]], ignore_index=True)
        .sort_values([cfg.data.date_col, cfg.data.permno_col])
        .reset_index(drop=True)
    )
    test_frame = splits["test"].copy()

    # Fit the shared preprocessor ONCE on train+validation; reuse for both refits.
    preprocessor = LinearFeaturePreprocessor(feature_cols, date_col=cfg.data.date_col)
    x_fit = preprocessor.fit_transform(fit_frame).to_numpy()
    x_pred = preprocessor.transform(test_frame).to_numpy()
    y_fit = fit_frame[cfg.data.target_col].to_numpy(dtype="float64")
    feature_names = list(preprocessor.matrix_columns_ or [])
    preprocessor_meta = preprocessor.to_dict()

    outputs: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    feature_importance: pd.DataFrame | None = None

    for _, row in winners.iterrows():
        family = str(row["model_family"])
        params = json.loads(row["params_json"])
        model_meta: dict[str, Any] = {
            "selected_candidate": _jsonable(row.to_dict()),
            "refit_sample": "train_plus_validation",
            "refit_n_rows": int(len(fit_frame)),
            "refit_start": _as_date_string(fit_frame[cfg.data.date_col].min()),
            "refit_end": _as_date_string(fit_frame[cfg.data.date_col].max()),
            "preprocessor": preprocessor_meta,
        }
        if family == "xgboost":
            preds, feature_importance = _fit_selected_xgb(
                x_fit, y_fit, x_pred, feature_names, params=params
            )
            model_name = "xgboost"
        elif family == "mlp":
            preds, refit_epochs = _fit_selected_mlp(
                x_fit, y_fit, x_pred, params=params, best_epoch=int(row["best_epoch"])
            )
            model_meta["refit_epochs"] = refit_epochs
            model_name = "mlp"
        else:  # pragma: no cover - defensive only
            raise ValueError(f"Unsupported model family {family!r}.")

        outputs.append(
            _prediction_frame(
                test_frame,
                preds,
                cfg=cfg,
                model_name=model_name,
                split="test",
                split_id=_split_id(cfg),
            )
        )
        metadata[model_name] = _jsonable(model_meta)

    test_predictions = pd.concat(outputs, ignore_index=True)
    if test_predictions.empty:
        raise ValueError(
            "No test predictions were produced; check that the test window in "
            "configs/main.yaml overlaps the model-panel date range."
        )
    return test_predictions, metadata, feature_importance


def save_prediction_outputs(predictions: pd.DataFrame, cfg: DictConfig) -> list[Path]:
    paths: list[Path] = []
    for model_name, frame in predictions.groupby("model_name", sort=True):
        path = _prediction_output_path(cfg, str(model_name))
        write_parquet(
            frame.reset_index(drop=True),
            path,
            required_columns=PREDICTION_REQUIRED_COLUMNS,
        )
        paths.append(path)
    return paths


def build_test_summary(
    predictive_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    main_bps = int(cfg.costs.main_bps)
    main_cost = cost_summary.loc[cost_summary["cost_bps"] == main_bps].rename(
        columns={
            "net_ann_return": f"net_ann_return_{main_bps}bps",
            "net_sharpe": f"net_sharpe_{main_bps}bps",
        }
    )
    summary = predictive_summary.merge(
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
    return summary.sort_values("model_name").reset_index(drop=True)


def run_nonlinear_benchmarks(cfg: DictConfig) -> dict[str, Path]:
    feature_cols = build_return_only_feature_list(read_parquet_schema_names(cfg.data.panel_path))
    feature_list_path = _write_json(feature_cols, _feature_list_path(cfg))
    logger.info("nonlinear benchmark return-only feature list: %s", feature_cols)

    panel = load_model_panel(cfg, feature_cols)
    splits = split_panel(panel, cfg)

    diagnostics, train_only_metadata = build_validation_diagnostics(splits, cfg, feature_cols)
    winners = select_best_candidates(diagnostics)
    validation_diagnostics_path = _write_csv(diagnostics, _validation_diagnostics_path(cfg))

    test_predictions, refit_metadata, feature_importance = build_test_predictions(
        splits, winners, cfg=cfg, feature_cols=feature_cols
    )
    prediction_paths = save_prediction_outputs(test_predictions, cfg)

    monthly_backtests = run_prediction_backtests(test_predictions, cfg, split_id=_split_id(cfg))
    backtest_path = _backtest_output_path(cfg)
    write_parquet(monthly_backtests, backtest_path)

    predictive_summary = summarize_prediction_outputs(test_predictions, cfg)
    predictive_summary_path = _write_csv(predictive_summary, _test_predictive_summary_path(cfg))

    cost_summary = summarize_backtests_by_cost(monthly_backtests)
    cost_summary_path = _write_csv(cost_summary, _test_cost_summary_path(cfg))

    test_summary = build_test_summary(predictive_summary, cost_summary, cfg)
    test_summary_path = _write_csv(test_summary, _test_summary_path(cfg))

    outputs = {
        "feature_list": feature_list_path,
        "validation_diagnostics": validation_diagnostics_path,
        "backtests": backtest_path,
        "test_predictive_summary": predictive_summary_path,
        "test_cost_summary": cost_summary_path,
        "test_summary": test_summary_path,
    }

    if feature_importance is not None:
        outputs["xgb_feature_importance"] = _write_csv(
            feature_importance, _feature_importance_path(cfg)
        )

    selection_payload = {
        "feature_columns": feature_cols,
        "selection_metric": f"validation_net_sharpe_at_{int(cfg.costs.main_bps)}bps",
        "validation": {
            "train_only_preprocessing": train_only_metadata["train_only_preprocessing"],
            "selected_models": winners.to_dict(orient="records"),
        },
        "test_refit": _jsonable(refit_metadata),
    }
    outputs["selected_hyperparameters"] = _write_json(
        _jsonable(selection_payload), _selection_path(cfg)
    )

    if prediction_paths:
        outputs["predictions_dir"] = Path(cfg.paths.predictions)
    return outputs


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg)
    logger.info(
        "Running nonlinear benchmark nonlinear pipeline with config:\n%s", OmegaConf.to_yaml(cfg)
    )
    outputs = run_nonlinear_benchmarks(cfg)
    logger.info("nonlinear benchmark outputs:")
    for label, path in outputs.items():
        logger.info("  %s -> %s", label, path)


if __name__ == "__main__":  # pragma: no cover
    main()
