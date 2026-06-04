"""Same-protocol walk-forward classical baseline evidence (no retraining).

Builds deterministic classical baseline signals directly from the canonical
``data/processed/model_panel.parquet`` and scores them under the accepted walk-forward
OOS protocol:

* reads only the columns needed for the requested baseline signals;
* filters to the 2017-2024 OOS window before any backtest work;
* reuses the accepted walk-forward cutoff/cost backtest grid helper;
* compares the accepted walk-forward ML monthly net returns to same-protocol
  classical baselines at the main 10% / 10 bps comparison point.

Outputs (all git-ignored under ``data/outputs/``)::

    predictions/classical_baseline_predictions.parquet
    backtests/classical_baseline_backtests.parquet
    metrics/classical_baseline_summary.csv
    metrics/classical_baseline_difference_tests.csv
    metrics/classical_baseline_preview.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlfinance.models.baselines import baseline_predictions_from_panel
from mlfinance.run.linear_benchmarks import _prediction_frame
from mlfinance.run.walkforward import OUTPUT_PREFIX as WALKFORWARD_PREFIX
from mlfinance.run.walkforward import _backtest_oos_grid
from mlfinance.run.walkforward_robustness import MAIN_CUTOFF, _nw_mean_tstat
from mlfinance.utils.io import read_parquet, read_parquet_schema_names, write_parquet
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

OUTPUT_PREFIX = "classical_baseline"
FEATURE_SET = "classical_baseline"
BASELINE_SOURCE = "classical_baseline"
DEFAULT_BASELINE_MODELS: tuple[str, ...] = (
    "reversal",
    "zscore_reversal",
    "momentum",
    "reversal_plus_momentum",
)
DEFAULT_OOS_START = "2017-01-31"
DEFAULT_OOS_END = "2024-12-31"
DIFF_COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    ("flow_compustat__mlp", "reversal"),
    ("flow_compustat__xgboost", "reversal"),
    ("flow_compustat__mlp", "zscore_reversal"),
    ("flow_compustat__xgboost", "zscore_reversal"),
    ("flow_compustat__mlp", "momentum"),
    ("flow_compustat__xgboost", "momentum"),
)
DIFF_COLUMNS = [
    "pair",
    "strategy_a",
    "strategy_b",
    "n_months",
    "mean_diff_monthly",
    "mean_diff_annualized",
    "nw_tstat_diff",
    "baseline_source",
]

BASELINE_SIGNAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "zero": (),
    "reversal": ("ret_1m",),
    "zscore_reversal": ("ret_1m",),
    "momentum": ("mom_2_12",),
    "reversal_plus_momentum": ("ret_1m", "mom_2_12"),
    "vol_scaled_reversal": ("ret_1m", "vol_12m"),
}


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def required_model_panel_columns(
    *,
    models: tuple[str, ...],
    date_col: str,
    permno_col: str,
    target_col: str,
    available_columns: list[str] | None = None,
) -> list[str]:
    """Smallest model-panel projection needed for the requested baseline models."""
    required = {date_col, permno_col, target_col}
    for model_name in models:
        if model_name not in BASELINE_SIGNAL_COLUMNS:
            raise ValueError(f"Unsupported classical baseline model: {model_name!r}")
        required.update(BASELINE_SIGNAL_COLUMNS[model_name])

    ordered = [date_col, permno_col, target_col, "ret_1m", "mom_2_12", "vol_12m"]
    projected = [column for column in ordered if column in required]

    if available_columns is not None:
        available = set(available_columns)
        missing = [column for column in projected if column not in available]
        if missing:
            raise KeyError(
                "model_panel.parquet is missing required baseline columns "
                f"{missing}; available columns start with {available_columns[:25]}"
            )
    return projected


def load_model_panel_slice(
    cfg: DictConfig,
    *,
    models: tuple[str, ...] = DEFAULT_BASELINE_MODELS,
    panel_path: str | Path | None = None,
    start: str = DEFAULT_OOS_START,
    end: str = DEFAULT_OOS_END,
) -> tuple[pd.DataFrame, list[str]]:
    """Read the minimal classical-baseline panel slice for the walk-forward OOS window."""
    path = Path(panel_path or cfg.data.panel_path)
    schema = read_parquet_schema_names(path)
    columns = required_model_panel_columns(
        models=models,
        date_col=str(cfg.data.date_col),
        permno_col=str(cfg.data.permno_col),
        target_col=str(cfg.data.target_col),
        available_columns=schema,
    )
    panel = read_parquet(path, columns=columns, required_columns=columns)
    date_col = str(cfg.data.date_col)
    permno_col = str(cfg.data.permno_col)
    target_col = str(cfg.data.target_col)

    panel[date_col] = pd.to_datetime(panel[date_col]) + pd.offsets.MonthEnd(0)
    panel = panel.dropna(subset=[permno_col, date_col, target_col]).copy()
    mask = panel[date_col].between(pd.Timestamp(start), pd.Timestamp(end))
    panel = panel.loc[mask].sort_values([date_col, permno_col]).reset_index(drop=True)
    return panel, columns


def build_classical_baseline_predictions(
    panel: pd.DataFrame,
    cfg: DictConfig,
    *,
    models: tuple[str, ...] = DEFAULT_BASELINE_MODELS,
    feature_set: str = FEATURE_SET,
    split_id: str = "walkforward_same_protocol_oos_201701_202412",
) -> pd.DataFrame:
    """Deterministic OOS baseline predictions from the already-built model panel."""
    date_col = str(cfg.data.date_col)
    outputs: list[pd.DataFrame] = []
    for model_name in models:
        preds = baseline_predictions_from_panel(
            panel,
            model=model_name,
            ret_1m_col="ret_1m",
            mom_col="mom_2_12",
            vol_col="vol_12m",
            date_col=date_col,
        )
        out = _prediction_frame(
            panel,
            preds.to_numpy(dtype="float64"),
            cfg=cfg,
            model_name=model_name,
            split="test",
            split_id=split_id,
        )
        out["feature_set"] = feature_set
        out["target_col"] = str(cfg.data.target_col)
        outputs.append(out)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def load_walkforward_backtests(
    cfg: DictConfig,
    *,
    backtests_path: str | Path | None = None,
) -> pd.DataFrame:
    """Accepted walk-forward walk-forward monthly backtests needed for paired differences."""
    path = Path(
        backtests_path or Path(cfg.paths.backtests) / f"{WALKFORWARD_PREFIX}_backtests.parquet"
    )
    required = ["date", "model_name", "cutoff", "cost_bps", "net_return"]
    return read_parquet(path, columns=required, required_columns=required)


def _split_strategy_name(model_name: str) -> tuple[str, str]:
    feature_set, _, model = str(model_name).partition("__")
    return feature_set, model


def _net_series_by_model(
    backtests: pd.DataFrame,
    *,
    cutoff: float,
    cost_bps: float,
    simplify_baseline_names: bool = False,
) -> dict[str, pd.Series]:
    """Monthly net-return series keyed by strategy/model name."""
    sub = backtests[
        (
            np.isclose(pd.to_numeric(backtests["cutoff"], errors="coerce"), float(cutoff))
            & np.isclose(pd.to_numeric(backtests["cost_bps"], errors="coerce"), float(cost_bps))
        )
    ].copy()
    if sub.empty:
        return {}
    sub["date"] = pd.to_datetime(sub["date"])
    series_by_model: dict[str, pd.Series] = {}
    for model_name, frame in sub.groupby("model_name", sort=True):
        key = model_name
        if simplify_baseline_names:
            _, key = _split_strategy_name(str(model_name))
        series_by_model[str(key)] = frame.set_index("date")["net_return"].sort_index()
    return series_by_model


def same_protocol_model_difference_tests(
    *,
    ml_backtests: pd.DataFrame,
    baseline_backtests: pd.DataFrame,
    cutoff: float = MAIN_CUTOFF,
    cost_bps: float = 10.0,
    hac_lag: int = 11,
) -> pd.DataFrame:
    """Paired HAC mean-difference tests at the accepted main comparison point."""
    ml_returns = _net_series_by_model(ml_backtests, cutoff=cutoff, cost_bps=cost_bps)
    baseline_returns = _net_series_by_model(
        baseline_backtests,
        cutoff=cutoff,
        cost_bps=cost_bps,
        simplify_baseline_names=True,
    )

    rows: list[dict[str, Any]] = []
    for strategy_a, strategy_b in DIFF_COMPARISON_PAIRS:
        if strategy_a not in ml_returns or strategy_b not in baseline_returns:
            continue
        diff = (ml_returns[strategy_a] - baseline_returns[strategy_b]).dropna()
        if len(diff) < 3:
            continue
        nw = _nw_mean_tstat(diff, hac_lag)
        rows.append(
            {
                "pair": f"{strategy_a} - {strategy_b}",
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "n_months": int(len(diff)),
                "mean_diff_monthly": float(diff.mean()),
                "mean_diff_annualized": float(diff.mean() * 12.0),
                "nw_tstat_diff": nw["tstat"],
                "baseline_source": BASELINE_SOURCE,
            }
        )
    if not rows:
        return pd.DataFrame(columns=DIFF_COLUMNS)
    return (
        pd.DataFrame(rows, columns=DIFF_COLUMNS)
        .sort_values(["strategy_a", "strategy_b"])
        .reset_index(drop=True)
    )


def _preview_markdown(
    *,
    summary: pd.DataFrame,
    difference_tests: pd.DataFrame,
    projected_columns: list[str],
    prediction_rows: int,
    oos_start: str,
    oos_end: str,
) -> str:
    def _head(df: pd.DataFrame, rows: int = 12) -> str:
        if df.empty:
            return "_(empty)_"
        with pd.option_context("display.width", 220, "display.max_columns", 40):
            return "```\n" + df.head(rows).round(6).to_string(index=False) + "\n```"

    main_summary = summary[
        (
            np.isclose(pd.to_numeric(summary["cutoff"], errors="coerce"), MAIN_CUTOFF)
            & np.isclose(pd.to_numeric(summary["cost_bps"], errors="coerce"), 10.0)
        )
    ].copy()
    main_summary = main_summary.sort_values("net_sharpe", ascending=False).reset_index(drop=True)

    lines = [
        "# walk-forward same-protocol classical baselines",
        "",
        "Generated by `mlfinance.run.classical_baselines` (no retraining).",
        "",
        f"- OOS window: `{oos_start}` to `{oos_end}`.",
        f"- Projected model-panel columns: `{projected_columns}`.",
        f"- Prediction rows written: `{prediction_rows}`.",
        "",
        "## Main summary (10% / 10 bps)",
        _head(main_summary),
        "",
        "## Same-protocol model-difference tests",
        _head(difference_tests),
        "",
    ]
    return "\n".join(lines)


def run_classical_baselines(
    cfg: DictConfig,
    *,
    models: tuple[str, ...] = DEFAULT_BASELINE_MODELS,
    output_prefix: str = OUTPUT_PREFIX,
    oos_start: str = DEFAULT_OOS_START,
    oos_end: str = DEFAULT_OOS_END,
) -> dict[str, Path]:
    """Run the same-protocol classical baseline package and persist all outputs."""
    panel, projected_columns = load_model_panel_slice(
        cfg,
        models=models,
        start=oos_start,
        end=oos_end,
    )
    if panel.empty:
        raise ValueError(f"No model-panel rows in the requested OOS window {oos_start}..{oos_end}.")

    predictions = build_classical_baseline_predictions(panel, cfg, models=models)
    if predictions.empty:
        raise ValueError("Classical baseline generation produced no OOS predictions.")

    predictions_path = Path(cfg.paths.predictions) / f"{output_prefix}_predictions.parquet"
    write_parquet(predictions, predictions_path)

    backtests, summary = _backtest_oos_grid(predictions, cfg)
    backtests_path = Path(cfg.paths.backtests) / f"{output_prefix}_backtests.parquet"
    write_parquet(backtests, backtests_path)

    metrics_dir = Path(cfg.paths.metrics)
    summary_path = _write_csv(summary, metrics_dir / f"{output_prefix}_summary.csv")

    walkforward_main_backtests = load_walkforward_backtests(cfg)
    diff = same_protocol_model_difference_tests(
        ml_backtests=walkforward_main_backtests,
        baseline_backtests=backtests,
        cutoff=MAIN_CUTOFF,
        cost_bps=float(cfg.costs.main_bps),
        hac_lag=int(cfg.eval.newey_west_lag_primary),
    )
    diff_path = _write_csv(diff, metrics_dir / f"{output_prefix}_difference_tests.csv")

    preview_path = metrics_dir / f"{output_prefix}_preview.md"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(
        _preview_markdown(
            summary=summary,
            difference_tests=diff,
            projected_columns=projected_columns,
            prediction_rows=len(predictions),
            oos_start=oos_start,
            oos_end=oos_end,
        ),
        encoding="utf-8",
    )

    outputs = {
        "predictions": predictions_path,
        "backtests": backtests_path,
        "summary": summary_path,
        "difference_tests": diff_path,
        "preview": preview_path,
    }
    logger.info(
        "walk-forward classical baselines done: %d OOS rows across %d deterministic strategies.",
        len(predictions),
        predictions.groupby(["feature_set", "model_name"]).ngroups,
    )
    return outputs


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    configure_logging()
    outputs = run_classical_baselines(cfg)
    for name, path in outputs.items():
        logger.info("Wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
