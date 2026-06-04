"""linear benchmark locked-split return-only Ridge and Elastic Net pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import ElasticNet

from mlfinance.backtest.metrics import turnover
from mlfinance.backtest.portfolio import (
    compute_realized_returns,
    decile_sort,
    long_short_weights,
)
from mlfinance.eval.newey_west import newey_west_se
from mlfinance.eval.predictive_metrics import decile_spread, information_coefficient, mae, rmse
from mlfinance.models.base import r_squared_oos
from mlfinance.models.linear_preprocessing import LinearFeaturePreprocessor
from mlfinance.models.ridge import ridge_eig
from mlfinance.run.baseline_pipeline import summarize_backtests_by_cost
from mlfinance.utils.io import read_parquet, read_parquet_schema_names, write_parquet
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

RETURN_ONLY_FEATURE_COLUMNS = [
    "ret_1m",
    "ret_2m",
    "ret_3m",
    "mom_2_12",
    "mom_1_12",
    "vol_3m",
    "vol_6m",
    "vol_12m",
    "drawdown_12m",
    "beta_12m",
    "idio_vol_12m",
    "ret_1m_rank",
    "mom_2_12_rank",
    "vol_12m_rank",
    "beta_12m_rank",
]
FORBIDDEN_FEATURE_TOKENS = ("target", "fwd", "future", "label", "prediction", "realized", "test")
PREDICTION_REQUIRED_COLUMNS = [
    "permno",
    "date",
    "month",
    "realized_date",
    "y_true",
    "y_pred",
    "prediction",
    "model",
    "model_name",
    "target",
    "split",
    "split_id",
]


def _as_date_string(value: pd.Timestamp | str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _window_mask(
    panel: pd.DataFrame,
    *,
    date_col: str,
    start: str,
    end: str,
) -> pd.Series:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    dates = pd.to_datetime(panel[date_col])
    return (dates >= start_ts) & (dates <= end_ts)


def locked_split_masks(panel: pd.DataFrame, cfg: DictConfig) -> dict[str, pd.Series]:
    date_col = cfg.data.date_col
    locked = cfg.splits.locked
    return {
        "train": _window_mask(
            panel,
            date_col=date_col,
            start=locked.train_start,
            end=locked.train_end,
        ),
        "validation": _window_mask(
            panel,
            date_col=date_col,
            start=locked.val_start,
            end=locked.val_end,
        ),
        "test": _window_mask(
            panel,
            date_col=date_col,
            start=locked.test_start,
            end=locked.test_end,
        ),
    }


def _available_column_names(columns_or_panel: pd.DataFrame | Iterable[str]) -> list[str]:
    if isinstance(columns_or_panel, pd.DataFrame):
        return list(columns_or_panel.columns)
    return list(columns_or_panel)


def build_return_only_feature_list(columns_or_panel: pd.DataFrame | Iterable[str]) -> list[str]:
    available_columns = _available_column_names(columns_or_panel)
    missing = [col for col in RETURN_ONLY_FEATURE_COLUMNS if col not in available_columns]
    if missing:
        raise KeyError(f"linear benchmark panel is missing return features: {missing}")

    forbidden = [
        col
        for col in RETURN_ONLY_FEATURE_COLUMNS
        if any(token in col.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Forbidden feature tokens detected in explicit feature list: {forbidden}")

    return list(RETURN_ONLY_FEATURE_COLUMNS)


def _split_id(cfg: DictConfig) -> str:
    test_start = pd.to_datetime(cfg.splits.locked.test_start)
    test_end = pd.to_datetime(cfg.splits.locked.test_end)
    return f"locked_test_{test_start:%Y%m}_{test_end:%Y%m}"


def _validation_split_id(cfg: DictConfig) -> str:
    val_start = pd.to_datetime(cfg.splits.locked.val_start)
    val_end = pd.to_datetime(cfg.splits.locked.val_end)
    return f"locked_validation_{val_start:%Y%m}_{val_end:%Y%m}"


def _prediction_output_path(cfg: DictConfig, model_name: str) -> Path:
    return Path(cfg.paths.predictions) / f"{model_name}__{_split_id(cfg)}.parquet"


def _backtest_output_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.backtests) / "linear_benchmarks_backtests.parquet"


def _validation_diagnostics_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "linear_benchmarks_validation_diagnostics.csv"


def _test_predictive_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "linear_benchmarks_test_predictive_summary.csv"


def _test_cost_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "linear_benchmarks_test_cost_grid_summary.csv"


def _test_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "linear_benchmarks_test_summary.csv"


def _feature_list_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "linear_benchmarks_feature_list.json"


def _selection_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "linear_benchmarks_selected_hyperparameters.json"


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_json(payload: dict[str, Any] | list[Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_model_panel(cfg: DictConfig, feature_cols: list[str]) -> pd.DataFrame:
    required = [
        cfg.data.permno_col,
        cfg.data.date_col,
        cfg.data.target_col,
        *feature_cols,
    ]
    panel = read_parquet(cfg.data.panel_path, columns=required, required_columns=required)
    panel[cfg.data.date_col] = pd.to_datetime(panel[cfg.data.date_col]) + pd.offsets.MonthEnd(0)
    panel = panel.dropna(
        subset=[cfg.data.permno_col, cfg.data.date_col, cfg.data.target_col]
    ).copy()
    panel = panel.sort_values([cfg.data.date_col, cfg.data.permno_col]).reset_index(drop=True)
    return panel


def split_panel(panel: pd.DataFrame, cfg: DictConfig) -> dict[str, pd.DataFrame]:
    masks = locked_split_masks(panel, cfg)
    splits = {
        name: panel.loc[mask]
        .copy()
        .sort_values([cfg.data.date_col, cfg.data.permno_col])
        .reset_index(drop=True)
        for name, mask in masks.items()
    }
    return splits


def _prediction_frame(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    cfg: DictConfig,
    model_name: str,
    split: str,
    split_id: str,
) -> pd.DataFrame:
    target_col = cfg.data.target_col
    date_col = cfg.data.date_col
    permno_col = cfg.data.permno_col

    out = frame[[permno_col, date_col, target_col]].copy()
    out["month"] = out[date_col]
    out["realized_date"] = out[date_col] + pd.offsets.MonthEnd(1)
    out["y_true"] = out[target_col].astype("float64")
    out["prediction"] = np.asarray(y_pred, dtype="float64")
    out["y_pred"] = out["prediction"]
    out["model_name"] = model_name
    out["model"] = model_name
    out["target"] = target_col
    out["split"] = split
    out["split_id"] = split_id
    return out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)


def _realized_return_panel(
    prediction_frame: pd.DataFrame,
    *,
    permno_col: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    return prediction_frame.pivot_table(
        index=date_col,
        columns=permno_col,
        values=target_col,
        aggfunc="first",
    ).sort_index()


def _leg_statistics(
    deciles: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    date_col: str,
    permno_col: str,
    target_col: str,
    long_bucket: int,
    short_bucket: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    realized = prediction_frame[[permno_col, date_col, target_col]]
    merged = deciles.merge(realized, on=[permno_col, date_col], how="left", validate="one_to_one")

    long_leg = merged.loc[merged["decile"] == long_bucket]
    short_leg = merged.loc[merged["decile"] == short_bucket]

    long_return = long_leg.groupby(date_col, sort=True)[target_col].mean()
    short_return = short_leg.groupby(date_col, sort=True)[target_col].mean()
    n_long = long_leg.groupby(date_col, sort=True)[permno_col].count()
    n_short = short_leg.groupby(date_col, sort=True)[permno_col].count()
    return long_return, short_return, n_long, n_short


def run_prediction_backtests(
    predictions: pd.DataFrame,
    cfg: DictConfig,
    *,
    split_id: str,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "month",
                "model_name",
                "split_id",
                "cost_bps",
                "gross_return",
                "turnover",
                "cost",
                "net_return",
                "long_return",
                "short_return",
                "n_long",
                "n_short",
            ]
        )

    date_col = cfg.data.date_col
    permno_col = cfg.data.permno_col
    target_col = cfg.data.target_col
    top_bucket = int(cfg.portfolio.long_bucket)
    bottom_bucket = int(cfg.portfolio.short_bucket)
    weighting = str(cfg.portfolio.weighting)
    cost_grid = [int(x) for x in cfg.costs.bps_grid]

    monthly_outputs: list[pd.DataFrame] = []
    for model_name, frame in predictions.groupby("model_name", sort=True):
        frame = frame.sort_values([date_col, permno_col]).reset_index(drop=True)
        pred_panel = frame.rename(columns={"prediction": "y_hat"})[[permno_col, date_col, "y_hat"]]
        deciles = decile_sort(pred_panel, n_deciles=int(cfg.portfolio.n_buckets))
        weights = long_short_weights(
            deciles,
            scheme=weighting,
            top_decile=top_bucket,
            bottom_decile=bottom_bucket,
        )
        returns_panel = _realized_return_panel(
            frame,
            permno_col=permno_col,
            date_col=date_col,
            target_col=target_col,
        )
        weights = weights.reindex(returns_panel.index).fillna(0.0)
        gross_return = compute_realized_returns(weights, returns_panel)
        turn = turnover(weights)

        long_return, short_return, n_long, n_short = _leg_statistics(
            deciles,
            frame,
            date_col=date_col,
            permno_col=permno_col,
            target_col=target_col,
            long_bucket=top_bucket,
            short_bucket=bottom_bucket,
        )

        gross_index = gross_return.index
        base = pd.DataFrame(
            {
                "date": gross_index,
                "month": gross_index,
                "model_name": model_name,
                "split_id": split_id,
                "gross_return": gross_return.to_numpy(),
                "turnover": turn.reindex(gross_index).fillna(0.0).to_numpy(),
                "long_return": long_return.reindex(gross_index).fillna(0.0).to_numpy(),
                "short_return": short_return.reindex(gross_index).fillna(0.0).to_numpy(),
                "n_long": n_long.reindex(gross_index).fillna(0).astype(int).to_numpy(),
                "n_short": n_short.reindex(gross_index).fillna(0).astype(int).to_numpy(),
            }
        )

        for cost_bps in cost_grid:
            out = base.copy()
            out["cost_bps"] = int(cost_bps)
            out["cost"] = out["turnover"] * (float(cost_bps) / 10_000.0)
            out["net_return"] = out["gross_return"] - out["cost"]
            monthly_outputs.append(out)

    return pd.concat(monthly_outputs, ignore_index=True)


def _rank_ic_hac_tstat(ic_series: pd.Series, lag: int) -> float:
    clean = ic_series.dropna().astype("float64")
    if clean.size < 2:
        return float("nan")
    lag = max(0, min(int(lag), int(clean.size) - 1))
    _, _, t_stat = newey_west_se(clean.to_numpy(), lag=lag)
    return float(t_stat)


def summarize_prediction_outputs(
    predictions: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    hac_lag = int(cfg.eval.newey_west_lag_primary)

    rows: list[dict[str, Any]] = []
    for model_name, frame in predictions.groupby("model_name", sort=True):
        clean = frame.dropna(subset=[target_col, "prediction"]).copy()
        renamed = clean.rename(columns={"prediction": "y_hat"})
        ic = information_coefficient(
            renamed,
            date_col=date_col,
            y_col=target_col,
            pred_col="y_hat",
            method="spearman",
        )
        spread = decile_spread(
            renamed,
            date_col=date_col,
            y_col=target_col,
            pred_col="y_hat",
            n_deciles=int(cfg.portfolio.n_buckets),
        )
        rows.append(
            {
                "model_name": model_name,
                "rmse": rmse(clean[target_col], clean["prediction"]),
                "mae": mae(clean[target_col], clean["prediction"]),
                "oos_r2": float(
                    r_squared_oos(
                        clean[target_col].to_numpy(),
                        clean["prediction"].to_numpy(),
                    )
                ),
                "rank_ic_mean": float(ic.mean()),
                "rank_ic_hac_tstat": _rank_ic_hac_tstat(ic, lag=hac_lag),
                "top_minus_bottom_mean": float(
                    spread.attrs.get("top_minus_bottom_mean", float("nan"))
                ),
                "n_predictions": int(len(clean)),
                "n_months": int(clean[date_col].nunique()),
                "realized_start": _as_date_string(clean["realized_date"].min()),
                "realized_end": _as_date_string(clean["realized_date"].max()),
                "split": str(clean["split"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def _ridge_validation_predictions(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    cfg: DictConfig,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    preprocessor = LinearFeaturePreprocessor(feature_cols, date_col=cfg.data.date_col)
    x_train = preprocessor.fit_transform(train_frame).to_numpy()
    x_val = preprocessor.transform(validation_frame).to_numpy()
    y_train = train_frame[cfg.data.target_col].to_numpy(dtype="float64")

    y_mean = float(np.mean(y_train))
    grid = [float(x) for x in cfg.models.ridge.shrinkage_grid]
    _, val_pred_matrix = ridge_eig(x_train, y_train - y_mean, x_val, grid)
    val_pred_matrix = val_pred_matrix + y_mean

    outputs: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    for idx, shrinkage in enumerate(grid):
        model_name = f"ridge__cand{idx:02d}"
        outputs.append(
            _prediction_frame(
                validation_frame,
                val_pred_matrix[:, idx],
                cfg=cfg,
                model_name=model_name,
                split="validation",
                split_id=_validation_split_id(cfg),
            )
        )
        meta_rows.append(
            {
                "candidate_rank": idx,
                "model_family": "ridge",
                "model_name": model_name,
                "shrinkage": float(shrinkage),
                "alpha": np.nan,
                "l1_ratio": np.nan,
            }
        )

    return (
        pd.concat(outputs, ignore_index=True),
        pd.DataFrame(meta_rows),
        {"preprocessor": preprocessor.to_dict()},
    )


def _elastic_net_validation_predictions(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    cfg: DictConfig,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    preprocessor = LinearFeaturePreprocessor(feature_cols, date_col=cfg.data.date_col)
    x_train = preprocessor.fit_transform(train_frame).to_numpy()
    x_val = preprocessor.transform(validation_frame).to_numpy()
    y_train = train_frame[cfg.data.target_col].to_numpy(dtype="float64")

    outputs: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    alpha_grid = [float(x) for x in cfg.models.elastic_net.alpha_grid]
    l1_ratio_grid = [float(x) for x in cfg.models.elastic_net.l1_ratio_grid]
    max_iter = int(cfg.models.elastic_net.max_iter)

    idx = 0
    for alpha in alpha_grid:
        for l1_ratio in l1_ratio_grid:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                random_state=1,
                fit_intercept=True,
            )
            model.fit(x_train, y_train)
            model_name = f"elastic_net__cand{idx:02d}"
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
                    "model_family": "elastic_net",
                    "model_name": model_name,
                    "shrinkage": np.nan,
                    "alpha": alpha,
                    "l1_ratio": l1_ratio,
                }
            )
            idx += 1

    return (
        pd.concat(outputs, ignore_index=True),
        pd.DataFrame(meta_rows),
        {"preprocessor": preprocessor.to_dict()},
    )


def build_validation_diagnostics(
    splits: dict[str, pd.DataFrame],
    cfg: DictConfig,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_frame = splits["train"]
    validation_frame = splits["validation"]

    ridge_preds, ridge_meta, ridge_fit_meta = _ridge_validation_predictions(
        train_frame,
        validation_frame,
        cfg=cfg,
        feature_cols=feature_cols,
    )
    enet_preds, enet_meta, enet_fit_meta = _elastic_net_validation_predictions(
        train_frame,
        validation_frame,
        cfg=cfg,
        feature_cols=feature_cols,
    )

    validation_predictions = pd.concat([ridge_preds, enet_preds], ignore_index=True)
    candidate_meta = pd.concat([ridge_meta, enet_meta], ignore_index=True)

    predictive = summarize_prediction_outputs(validation_predictions, cfg)
    backtests = run_prediction_backtests(
        validation_predictions,
        cfg,
        split_id=_validation_split_id(cfg),
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

    metadata = {
        "train_only_preprocessing": {
            "ridge": ridge_fit_meta,
            "elastic_net": enet_fit_meta,
        }
    }
    return diagnostics, metadata


def select_best_validation_models(diagnostics: pd.DataFrame) -> pd.DataFrame:
    winners: list[pd.Series] = []
    for family in ("ridge", "elastic_net"):
        subset = diagnostics.loc[diagnostics["model_family"] == family].copy()
        subset = subset.sort_values(
            [
                "validation_net_sharpe",
                "rank_ic_mean",
                "oos_r2",
                "rmse",
                "candidate_rank",
            ],
            ascending=[False, False, False, True, True],
            na_position="last",
            kind="mergesort",
        )
        if subset.empty:
            raise ValueError(f"No validation diagnostics available for model family {family!r}.")
        winners.append(subset.iloc[0])
    return pd.DataFrame(winners).reset_index(drop=True)


def _fit_selected_ridge(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    *,
    cfg: DictConfig,
    feature_cols: list[str],
    shrinkage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessor = LinearFeaturePreprocessor(feature_cols, date_col=cfg.data.date_col)
    x_fit = preprocessor.fit_transform(fit_frame).to_numpy()
    x_pred = preprocessor.transform(predict_frame).to_numpy()
    y_fit = fit_frame[cfg.data.target_col].to_numpy(dtype="float64")
    y_mean = float(np.mean(y_fit))
    _, preds = ridge_eig(x_fit, y_fit - y_mean, x_pred, [float(shrinkage)])
    return y_mean + preds[:, 0], {"preprocessor": preprocessor.to_dict()}


def _fit_selected_elastic_net(
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    *,
    cfg: DictConfig,
    feature_cols: list[str],
    alpha: float,
    l1_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    preprocessor = LinearFeaturePreprocessor(feature_cols, date_col=cfg.data.date_col)
    x_fit = preprocessor.fit_transform(fit_frame).to_numpy()
    x_pred = preprocessor.transform(predict_frame).to_numpy()
    y_fit = fit_frame[cfg.data.target_col].to_numpy(dtype="float64")

    model = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        max_iter=int(cfg.models.elastic_net.max_iter),
        random_state=1,
        fit_intercept=True,
    )
    model.fit(x_fit, y_fit)
    return model.predict(x_pred), {"preprocessor": preprocessor.to_dict()}


def build_test_predictions(
    splits: dict[str, pd.DataFrame],
    winners: pd.DataFrame,
    *,
    cfg: DictConfig,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit_frame = (
        pd.concat([splits["train"], splits["validation"]], ignore_index=True)
        .sort_values([cfg.data.date_col, cfg.data.permno_col])
        .reset_index(drop=True)
    )
    test_frame = splits["test"].copy()

    outputs: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for _, row in winners.iterrows():
        family = str(row["model_family"])
        if family == "ridge":
            preds, fit_meta = _fit_selected_ridge(
                fit_frame,
                test_frame,
                cfg=cfg,
                feature_cols=feature_cols,
                shrinkage=float(row["shrinkage"]),
            )
            model_name = "ridge"
        elif family == "elastic_net":
            preds, fit_meta = _fit_selected_elastic_net(
                fit_frame,
                test_frame,
                cfg=cfg,
                feature_cols=feature_cols,
                alpha=float(row["alpha"]),
                l1_ratio=float(row["l1_ratio"]),
            )
            model_name = "elastic_net"
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
        metadata[model_name] = {
            "selected_candidate": row.to_dict(),
            "refit_sample": "train_plus_validation",
            "refit_n_rows": int(len(fit_frame)),
            "refit_start": _as_date_string(fit_frame[cfg.data.date_col].min()),
            "refit_end": _as_date_string(fit_frame[cfg.data.date_col].max()),
            "preprocessor": fit_meta["preprocessor"],
        }

    return pd.concat(outputs, ignore_index=True), metadata


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
    winners: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    main_bps = int(cfg.costs.main_bps)
    main_cost = cost_summary.loc[cost_summary["cost_bps"] == main_bps].copy()
    main_cost = main_cost.rename(
        columns={
            "net_ann_return": f"net_ann_return_{main_bps}bps",
            "net_sharpe": f"net_sharpe_{main_bps}bps",
            "avg_turnover": "avg_turnover",
        }
    )

    selected = winners.copy()
    selected["model_name"] = selected["model_family"]
    keep = ["model_name", "shrinkage", "alpha", "l1_ratio"]

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
    ).merge(selected[keep], on="model_name", how="left", validate="one_to_one")
    return summary.sort_values("model_name").reset_index(drop=True)


def run_linear_benchmarks(cfg: DictConfig) -> dict[str, Path]:
    feature_cols = build_return_only_feature_list(read_parquet_schema_names(cfg.data.panel_path))
    feature_list_path = _write_json(feature_cols, _feature_list_path(cfg))
    logger.info("linear benchmark return-only feature list: %s", feature_cols)

    panel = load_model_panel(cfg, feature_cols)
    splits = split_panel(panel, cfg)
    diagnostics, train_only_metadata = build_validation_diagnostics(splits, cfg, feature_cols)
    winners = select_best_validation_models(diagnostics)
    validation_diagnostics_path = _write_csv(diagnostics, _validation_diagnostics_path(cfg))

    test_predictions, refit_metadata = build_test_predictions(
        splits,
        winners,
        cfg=cfg,
        feature_cols=feature_cols,
    )
    prediction_paths = save_prediction_outputs(test_predictions, cfg)

    monthly_backtests = run_prediction_backtests(test_predictions, cfg, split_id=_split_id(cfg))
    backtest_path = _backtest_output_path(cfg)
    write_parquet(monthly_backtests, backtest_path)

    predictive_summary = summarize_prediction_outputs(test_predictions, cfg)
    predictive_summary_path = _write_csv(predictive_summary, _test_predictive_summary_path(cfg))

    cost_summary = summarize_backtests_by_cost(monthly_backtests)
    cost_summary_path = _write_csv(cost_summary, _test_cost_summary_path(cfg))

    test_summary = build_test_summary(predictive_summary, cost_summary, winners, cfg)
    test_summary_path = _write_csv(test_summary, _test_summary_path(cfg))

    selection_payload = {
        "feature_columns": feature_cols,
        "selection_metric": f"validation_net_sharpe_at_{int(cfg.costs.main_bps)}bps",
        "validation": {
            "train_only_preprocessing": train_only_metadata["train_only_preprocessing"],
            "selected_models": winners.to_dict(orient="records"),
        },
        "test_refit": refit_metadata,
    }
    selection_path = _write_json(selection_payload, _selection_path(cfg))

    outputs = {
        "feature_list": feature_list_path,
        "validation_diagnostics": validation_diagnostics_path,
        "selected_hyperparameters": selection_path,
        "backtests": backtest_path,
        "test_predictive_summary": predictive_summary_path,
        "test_cost_summary": cost_summary_path,
        "test_summary": test_summary_path,
    }
    if prediction_paths:
        outputs["predictions_dir"] = Path(cfg.paths.predictions)
    return outputs


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg)
    logger.info("Running linear benchmark linear pipeline with config:\n%s", OmegaConf.to_yaml(cfg))
    outputs = run_linear_benchmarks(cfg)
    logger.info("linear benchmark outputs:")
    for label, path in outputs.items():
        logger.info("  %s -> %s", label, path)


if __name__ == "__main__":  # pragma: no cover
    main()
