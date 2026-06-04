"""CRSP-only baseline pipeline for conditional mean reversion.

Builds:
1. ``data/processed/model_panel.parquet`` with return features and targets.
2. OOS baseline predictions for the locked final-test window only.
3. Equal-weight long-short decile backtests with flat bps transaction costs.
4. A first baseline performance table for reversal, z-score reversal, and momentum.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlfinance.backtest.metrics import turnover
from mlfinance.backtest.portfolio import (
    compute_realized_returns,
    decile_sort,
    long_short_weights,
)
from mlfinance.data.loaders import load_compustat_quarterly, load_crsp_monthly
from mlfinance.eval.portfolio_metrics import summarize_backtest, summarize_monthly_returns
from mlfinance.eval.predictive_metrics import decile_spread, summarize_predictions
from mlfinance.features.compustat_flow_features import add_flow_compustat_features_to_panel
from mlfinance.features.return_features import add_return_features
from mlfinance.models.baselines import baseline_predictions_from_panel
from mlfinance.utils.io import read_parquet, write_parquet
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

SUPPORTED_BASELINE_MODELS = {
    "zero",
    "reversal",
    "zscore_reversal",
    "vol_scaled_reversal",
    "momentum",
    "reversal_plus_momentum",
}


def _panel_start_date(cfg: DictConfig) -> str:
    train_start = pd.to_datetime(cfg.splits.locked.train_start)
    min_history = int(cfg.features.return_features.min_history_months)
    return str(
        (train_start - pd.DateOffset(months=min_history)).to_period("M").to_timestamp("M").date()
    )


def _locked_test_mask(panel: pd.DataFrame, cfg: DictConfig) -> pd.Series:
    test_start = pd.to_datetime(cfg.splits.locked.test_start)
    test_end = pd.to_datetime(cfg.splits.locked.test_end)
    return (panel[cfg.data.date_col] >= test_start) & (panel[cfg.data.date_col] <= test_end)


def _split_id(cfg: DictConfig) -> str:
    test_start = pd.to_datetime(cfg.splits.locked.test_start)
    test_end = pd.to_datetime(cfg.splits.locked.test_end)
    return f"locked_test_{test_start:%Y%m}_{test_end:%Y%m}"


def _as_date_string(value: pd.Timestamp | str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _prediction_output_path(cfg: DictConfig, model_name: str) -> Path:
    split_id = _split_id(cfg)
    return Path(cfg.paths.predictions) / f"{model_name}__{split_id}.parquet"


def _backtest_output_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.backtests) / "baseline_backtests.parquet"


def _predictive_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "baseline_predictive_summary.csv"


def _cost_summary_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.metrics) / "baseline_cost_grid_summary.csv"


def _first_table_csv_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.report_tables) / "table_baseline_performance.csv"


def _first_table_tex_path(cfg: DictConfig) -> Path:
    return Path(cfg.paths.report_tables) / "table_baseline_performance.tex"


def runtime_rebuild_panel(cfg: DictConfig) -> bool:
    """Return the runtime panel rebuild flag, defaulting to ``True``."""
    value = OmegaConf.select(cfg, "runtime.rebuild_panel", default=True)
    return bool(value)


def build_model_panel(cfg: DictConfig) -> pd.DataFrame:
    """Load Monthly CRSP and build the return-feature model panel."""
    use_compustat = bool(OmegaConf.select(cfg, "data.use_compustat", default=False))
    compustat_features_enabled = bool(
        OmegaConf.select(cfg, "features.compustat_features.enabled", default=False)
    )
    if use_compustat and not compustat_features_enabled:
        logger.warning(
            "data.use_compustat is true but features.compustat_features.enabled is false; "
            "skipping flow-only Compustat integration."
        )
    if compustat_features_enabled and not use_compustat:
        logger.warning(
            "features.compustat_features.enabled is true but data.use_compustat is false; "
            "skipping flow-only Compustat integration."
        )
    if bool(cfg.data.use_regime_features):
        logger.warning("baseline ignores regime features even if the config leaves them enabled.")

    crsp = load_crsp_monthly(
        cfg.data.paths.crsp_monthly,
        start_date=_panel_start_date(cfg),
    )
    panel = add_return_features(
        crsp,
        permno_col=cfg.data.permno_col,
        date_col=cfg.data.date_col,
        ret_col=cfg.data.ret_col,
        market_col=cfg.data.market_col,
    )
    if use_compustat and compustat_features_enabled:
        lag_months = int(
            OmegaConf.select(
                cfg,
                "features.compustat_features.conservative_lag_months",
                default=6,
            )
        )
        compustat = load_compustat_quarterly(cfg.data.paths.compustat_quarterly)
        panel = add_flow_compustat_features_to_panel(
            panel,
            compustat,
            merge_key="cusip8",
            date_col=cfg.data.date_col,
            lag_months=lag_months,
        )
    panel = panel.dropna(subset=[cfg.data.target_col]).reset_index(drop=True)
    return panel


def save_model_panel(panel: pd.DataFrame, cfg: DictConfig) -> Path:
    path = Path(cfg.data.panel_path)
    write_parquet(panel, path)
    return path


def load_model_panel(cfg: DictConfig, *, rebuild: bool = True) -> pd.DataFrame:
    path = Path(cfg.data.panel_path)
    if rebuild or not path.exists():
        panel = build_model_panel(cfg)
        save_model_panel(panel, cfg)
        return panel
    return read_parquet(path)


def _active_baseline_models(cfg: DictConfig) -> list[str]:
    models = [str(name) for name in cfg.active_models]
    unsupported = [name for name in models if name not in SUPPORTED_BASELINE_MODELS]
    if unsupported:
        raise ValueError(
            "baseline only supports baseline models. "
            f"Unsupported active_models entries: {unsupported}"
        )
    if not models:
        raise ValueError("cfg.active_models is empty.")
    return models


def generate_oos_baseline_predictions(panel: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Generate locked-test predictions only, never training-period rows."""
    split_id = _split_id(cfg)
    test_panel = panel.loc[_locked_test_mask(panel, cfg)].copy()
    models = _active_baseline_models(cfg)
    target_col = cfg.data.target_col
    robust_target_col = cfg.data.robustness_target_col
    date_col = cfg.data.date_col

    outputs: list[pd.DataFrame] = []
    for model_name in models:
        preds = baseline_predictions_from_panel(
            test_panel,
            model=model_name,
            ret_1m_col="ret_1m",
            mom_col="mom_2_12",
            vol_col="vol_12m",
            date_col=date_col,
        )
        out = test_panel[[cfg.data.permno_col, date_col, target_col, robust_target_col]].copy()
        out["month"] = out[date_col]
        out["realized_date"] = out[date_col] + pd.offsets.MonthEnd(1)
        out["prediction"] = preds.astype(float)
        out["model_name"] = model_name
        out["split_id"] = split_id
        out = out.dropna(subset=["prediction", target_col]).reset_index(drop=True)
        outputs.append(out)

    if not outputs:
        return pd.DataFrame(
            columns=[
                cfg.data.permno_col,
                date_col,
                "month",
                "realized_date",
                target_col,
                robust_target_col,
                "prediction",
                "model_name",
                "split_id",
            ]
        )
    return pd.concat(outputs, ignore_index=True)


def save_prediction_outputs(predictions: pd.DataFrame, cfg: DictConfig) -> list[Path]:
    paths: list[Path] = []
    for model_name, frame in predictions.groupby("model_name", sort=True):
        path = _prediction_output_path(cfg, str(model_name))
        write_parquet(frame.reset_index(drop=True), path)
        paths.append(path)
    return paths


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


def run_baseline_backtests(predictions: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Backtest each baseline across the flat bps grid."""
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
    split_id = _split_id(cfg)

    monthly_outputs: list[pd.DataFrame] = []
    for model_name, frame in predictions.groupby("model_name", sort=True):
        frame = frame.sort_values([date_col, permno_col]).reset_index(drop=True)
        if model_name == "zero":
            no_trade_dates = pd.Index(sorted(frame[date_col].drop_duplicates()))
            base = pd.DataFrame(
                {
                    "date": no_trade_dates,
                    "month": no_trade_dates,
                    "model_name": model_name,
                    "split_id": split_id,
                    "gross_return": 0.0,
                    "turnover": 0.0,
                    "long_return": 0.0,
                    "short_return": 0.0,
                    "n_long": 0,
                    "n_short": 0,
                }
            )
            for cost_bps in cost_grid:
                out = base.copy()
                out["cost_bps"] = int(cost_bps)
                out["cost"] = 0.0
                out["net_return"] = 0.0
                monthly_outputs.append(out)
            continue

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
        long_return = long_return.reindex(gross_index).fillna(0.0)
        short_return = short_return.reindex(gross_index).fillna(0.0)
        n_long = n_long.reindex(gross_index).fillna(0).astype(int)
        n_short = n_short.reindex(gross_index).fillna(0).astype(int)

        base = pd.DataFrame(
            {
                "date": gross_index,
                "month": gross_index,
                "model_name": model_name,
                "split_id": split_id,
                "gross_return": gross_return.to_numpy(),
                "turnover": turn.reindex(gross_index).fillna(0.0).to_numpy(),
                "long_return": long_return.to_numpy(),
                "short_return": short_return.to_numpy(),
                "n_long": n_long.to_numpy(),
                "n_short": n_short.to_numpy(),
            }
        )

        for cost_bps in cost_grid:
            out = base.copy()
            out["cost_bps"] = int(cost_bps)
            out["cost"] = out["turnover"] * (float(cost_bps) / 10_000.0)
            out["net_return"] = out["gross_return"] - out["cost"]
            monthly_outputs.append(out)

    return pd.concat(monthly_outputs, ignore_index=True)


def summarize_predictive_outputs(predictions: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    target_col = cfg.data.target_col
    test_start = _as_date_string(pd.to_datetime(cfg.splits.locked.test_start))
    test_end = _as_date_string(pd.to_datetime(cfg.splits.locked.test_end))
    rows: list[dict[str, Any]] = []
    for model_name, frame in predictions.groupby("model_name", sort=True):
        metrics = summarize_predictions(
            frame.rename(columns={"prediction": "y_hat"}),
            y_col=target_col,
            pred_col="y_hat",
            date_col=cfg.data.date_col,
        )
        spread = decile_spread(
            frame.rename(columns={"prediction": "y_hat"}),
            date_col=cfg.data.date_col,
            y_col=target_col,
            pred_col="y_hat",
            n_deciles=int(cfg.portfolio.n_buckets),
        )
        metrics["model_name"] = model_name
        metrics["n_predictions"] = int(len(frame))
        metrics["n_months"] = int(frame["realized_date"].nunique())
        metrics["realized_start"] = _as_date_string(frame["realized_date"].min())
        metrics["realized_end"] = _as_date_string(frame["realized_date"].max())
        metrics["test_start"] = test_start
        metrics["test_end"] = test_end
        metrics["top_minus_bottom_mean"] = (
            np.nan
            if model_name == "zero"
            else float(spread.attrs.get("top_minus_bottom_mean", float("nan")))
        )
        if model_name == "zero":
            metrics["oos_r2"] = 0.0
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)


def summarize_backtests_by_cost(monthly_backtests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, cost_bps), frame in monthly_backtests.groupby(
        ["model_name", "cost_bps"], sort=True
    ):
        net_summary = summarize_backtest(frame, return_col="net_return")
        gross_summary = summarize_monthly_returns(frame["gross_return"])
        row: dict[str, Any] = {
            "model_name": model_name,
            "cost_bps": int(cost_bps),
            "n_months": int(frame["date"].nunique()),
            "gross_ann_return": gross_summary["ann_return"],
            "gross_ann_vol": gross_summary["ann_vol"],
            "gross_sharpe": gross_summary["sharpe"],
            "net_ann_return": net_summary["ann_return"],
            "net_ann_vol": net_summary["ann_vol"],
            "net_sharpe": net_summary["sharpe"],
            "net_sortino": net_summary["sortino"],
            "net_max_drawdown": net_summary["max_drawdown"],
            "net_calmar": net_summary["calmar"],
            "avg_turnover": net_summary.get("avg_turnover"),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model_name", "cost_bps"]).reset_index(drop=True)


def build_first_performance_table(
    predictive_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    main_bps = int(cfg.costs.main_bps)
    cost_main = cost_summary.loc[cost_summary["cost_bps"] == main_bps].copy()
    cost_main = cost_main.rename(
        columns={
            "net_ann_return": f"net_ann_return_{main_bps}bps",
            "net_sharpe": f"net_sharpe_{main_bps}bps",
            "net_max_drawdown": f"net_max_drawdown_{main_bps}bps",
        }
    )
    cols = [
        "model_name",
        "gross_ann_return",
        "gross_sharpe",
        f"net_ann_return_{main_bps}bps",
        f"net_sharpe_{main_bps}bps",
        f"net_max_drawdown_{main_bps}bps",
        "avg_turnover",
    ]
    table = predictive_summary.merge(
        cost_main[cols], on="model_name", how="left", validate="one_to_one"
    )
    table["cost_bps"] = main_bps
    order = [
        "model_name",
        "cost_bps",
        "n_months",
        "realized_start",
        "realized_end",
        "test_start",
        "test_end",
        "n_predictions",
        "rmse",
        "mae",
        "oos_r2",
        "rank_ic_mean",
        "rank_ic_tstat",
        "top_minus_bottom_mean",
        "gross_ann_return",
        "gross_sharpe",
        f"net_ann_return_{main_bps}bps",
        f"net_sharpe_{main_bps}bps",
        f"net_max_drawdown_{main_bps}bps",
        "avg_turnover",
    ]
    return table[order].sort_values("model_name").reset_index(drop=True)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_tex(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, float_format=lambda x: f"{x:.6f}"), encoding="utf-8")
    return path


def run_baseline_pipeline(cfg: DictConfig, *, rebuild_panel: bool = True) -> dict[str, Path]:
    """Run the full baseline pipeline and persist all outputs."""
    panel = load_model_panel(cfg, rebuild=rebuild_panel)
    panel_path = Path(cfg.data.panel_path)

    predictions = generate_oos_baseline_predictions(panel, cfg)
    prediction_paths = save_prediction_outputs(predictions, cfg)

    monthly_backtests = run_baseline_backtests(predictions, cfg)
    backtest_path = _backtest_output_path(cfg)
    write_parquet(monthly_backtests, backtest_path)

    predictive_summary = summarize_predictive_outputs(predictions, cfg)
    predictive_summary_path = _write_csv(predictive_summary, _predictive_summary_path(cfg))

    cost_summary = summarize_backtests_by_cost(monthly_backtests)
    cost_summary_path = _write_csv(cost_summary, _cost_summary_path(cfg))

    first_table = build_first_performance_table(predictive_summary, cost_summary, cfg)
    first_table_csv_path = _write_csv(first_table, _first_table_csv_path(cfg))
    first_table_tex_path = _write_tex(first_table, _first_table_tex_path(cfg))

    outputs = {
        "panel": panel_path,
        "backtests": backtest_path,
        "predictive_summary": predictive_summary_path,
        "cost_summary": cost_summary_path,
        "first_table_csv": first_table_csv_path,
        "first_table_tex": first_table_tex_path,
    }
    if prediction_paths:
        outputs["predictions_dir"] = Path(cfg.paths.predictions)
    return outputs


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg)
    logger.info("Running baseline pipeline with config:\n%s", OmegaConf.to_yaml(cfg))
    outputs = run_baseline_pipeline(cfg, rebuild_panel=runtime_rebuild_panel(cfg))
    logger.info("baseline outputs:")
    for label, path in outputs.items():
        logger.info("  %s -> %s", label, path)


if __name__ == "__main__":  # pragma: no cover
    main()
