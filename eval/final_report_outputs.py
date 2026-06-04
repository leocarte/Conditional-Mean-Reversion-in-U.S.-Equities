"""Final report table and figure generation for accepted walk-forward outputs.

This module is strictly post-processing. It reads accepted canonical files
under ``data/outputs/`` and writes report-ready tables and figures under
``report/tables/`` and ``report/figures/`` when the needed inputs are present.

Guardrails:

* It never reads archived / historical / pre-1985 paths.
* It never touches model-training code, configs, targets, or split logic.
* Missing optional robustness files are skipped with an explicit note.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from mlfinance.eval.figures.style import apply_style
from mlfinance.eval.predictive_metrics import decile_spread

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_CUTOFF = 0.10
MAIN_COST_BPS = 10.0
ARCHIVE_TOKENS = ("archive", "historical", "pre-1985", "pre1985")

MAIN_SUMMARY_FILENAME = "walkforward_model_summary.csv"
COST_GRID_FILENAME = "walkforward_cost_grid_metrics.csv"
BREAKEVEN_FILENAME = "walkforward_breakeven_costs.csv"
REBALANCE_FILENAME = "walkforward_rebalance_robustness.csv"
REGIME_FILENAME = "walkforward_regime_robustness.csv"
ALPHA_BETA_FILENAME = "walkforward_alpha_beta_stats.csv"
DIFFERENCE_TESTS_FILENAME = "walkforward_model_difference_tests.csv"
CLASSICAL_BASELINE_SUMMARY_FILENAME = "classical_baseline_summary.csv"
CLASSICAL_BASELINE_DIFFERENCE_FILENAME = "classical_baseline_difference_tests.csv"
EXCESS_TARGET_SUMMARY_FILENAME = "walkforward_excess_model_summary.csv"
MLP_MULTISEED_SUMMARY_FILENAME = "walkforward_mlp_multiseed_summary.csv"
REGIME_ABLATION_FILENAME = "walkforward_regime_feature_ablation_comparison.csv"
BACKTESTS_FILENAME = "walkforward_backtests.parquet"
PREDICTIONS_FILENAME = "walkforward_predictions.parquet"
MODEL_PANEL_FILENAME = "model_panel.parquet"
WALKFORWARD_XGB_IMPORTANCE_CANDIDATES = (
    "walkforward_xgb_feature_importance.csv",
    "xgb_feature_importance.csv",
)

MAIN_TABLE_FILENAME = "table_walkforward_model_summary.csv"
COST_TABLE_FILENAME = "table_cost_breakeven.csv"
REGIME_TABLE_FILENAME = "table_regime_robustness.csv"
ALPHA_BETA_TABLE_FILENAME = "table_alpha_beta.csv"
REBALANCE_TABLE_FILENAME = "table_rebalance_robustness.csv"
DIFFERENCE_TESTS_TABLE_FILENAME = "table_model_incremental_tests.csv"
CLASSICAL_BASELINE_TABLE_FILENAME = "table_classical_baselines.csv"
CLASSICAL_BASELINE_DIFFERENCE_TABLE_FILENAME = "table_classical_model_difference_tests.csv"
SUPPLEMENTARY_ROBUSTNESS_TABLE_FILENAME = "table_supplementary_robustness.csv"
FINAL_MODEL_HIERARCHY_TABLE_FILENAME = "table_final_model_hierarchy.csv"
DECILE_TABLE_FILENAME = "table_prediction_deciles.csv"
DATA_SUMMARY_TABLE_FILENAME = "table_data_summary.csv"
FEATURE_IMPORTANCE_TABLE_FILENAME = "table_walkforward_xgb_feature_importance.csv"

COST_FIGURE_FILENAME = "figure_net_sharpe_vs_cost.png"
CUMULATIVE_FIGURE_FILENAME = "figure_cumulative_net_returns.png"
REGIME_FIGURE_FILENAME = "figure_regime_net_sharpe.png"
DECILE_FIGURE_FILENAME = "figure_prediction_deciles.png"

MODEL_COLORS = {
    "ridge": "#4C78A8",
    "elastic_net": "#F58518",
    "xgboost": "#54A24B",
    "mlp": "#E45756",
}
FEATURE_SET_LINESTYLES = {
    "flow_compustat": "-",
    "return_only": "--",
}
COMMON_COLUMN_ALIASES = {
    "gross_SR": "gross_sharpe",
    "gross_sr": "gross_sharpe",
    "net_SR": "net_sharpe",
    "net_sr": "net_sharpe",
    "net_ann": "net_ann_return",
    "turnover": "avg_turnover",
    "rank_IC": "rank_ic_mean",
    "rank_ic": "rank_ic_mean",
    "IC_t_HAC": "rank_ic_hac_tstat",
    "ic_t_hac": "rank_ic_hac_tstat",
    "portfolio_cutoff": "cutoff",
}
SUPPLEMENTARY_ROBUSTNESS_COLUMNS = [
    "experiment",
    "feature_set",
    "model",
    "target",
    "cutoff",
    "cost_bps",
    "net_sharpe",
    "gross_sharpe",
    "net_ann_return",
    "breakeven_bps",
    "rank_ic_mean",
    "rank_ic_hac_tstat",
    "reference_feature_set",
    "reference_model",
    "reference_net_sharpe",
    "delta_net_sharpe",
    "seed_count",
    "mean_net_sharpe",
    "std_net_sharpe",
    "min_net_sharpe",
    "max_net_sharpe",
    "interpretation",
    "source_file",
]
FINAL_MODEL_HIERARCHY_COLUMNS = [
    "strategy",
    "role",
    "feature_set",
    "model",
    "cutoff",
    "cost_bps",
    "net_sharpe",
    "gross_sharpe",
    "net_ann_return",
    "turnover",
    "breakeven_bps",
    "rank_ic_mean",
    "rank_ic_hac_tstat",
    "source_table",
    "interpretation",
]


@dataclass(frozen=True)
class FinalReportPaths:
    """Repository-local input/output locations for the final report layer."""

    repo_root: Path
    metrics_dir: Path
    backtests_dir: Path
    predictions_dir: Path
    processed_dir: Path
    report_tables_dir: Path
    report_figures_dir: Path


@dataclass
class GenerationResult:
    """Summary of what the final-output pass wrote or skipped."""

    written: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def note_skip(self, message: str) -> None:
        self.skipped.append(message)
        LOGGER.info(message)


def build_final_report_paths(
    repo_root: Path | None = None,
    *,
    metrics_dir: Path | None = None,
    backtests_dir: Path | None = None,
    predictions_dir: Path | None = None,
    processed_dir: Path | None = None,
    report_tables_dir: Path | None = None,
    report_figures_dir: Path | None = None,
) -> FinalReportPaths:
    """Construct default final report input/output paths under ``repo_root``."""

    root = (repo_root or REPO_ROOT).resolve()
    return FinalReportPaths(
        repo_root=root,
        metrics_dir=(metrics_dir or root / "data" / "outputs" / "metrics").resolve(),
        backtests_dir=(backtests_dir or root / "data" / "outputs" / "backtests").resolve(),
        predictions_dir=(predictions_dir or root / "data" / "outputs" / "predictions").resolve(),
        processed_dir=(processed_dir or root / "data" / "processed").resolve(),
        report_tables_dir=(report_tables_dir or root / "report" / "tables").resolve(),
        report_figures_dir=(report_figures_dir or root / "report" / "figures").resolve(),
    )


def validate_report_input_path(path: Path, *, allowed_root: Path) -> Path:
    """Reject non-canonical or archive-like input paths before reading them."""

    resolved = path.resolve(strict=False)
    root = allowed_root.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Input path must stay under {root}: {resolved}")

    lowered = resolved.as_posix().lower()
    token = next((item for item in ARCHIVE_TOKENS if item in lowered), None)
    if token is not None:
        raise ValueError(f"Refusing to read archive-like path {resolved} (matched {token!r}).")
    return resolved


def _read_optional_csv(
    path: Path, *, allowed_root: Path, result: GenerationResult
) -> pd.DataFrame | None:
    resolved = validate_report_input_path(path, allowed_root=allowed_root)
    if not resolved.exists():
        result.note_skip(f"Skipped missing optional input: {resolved.name}")
        return None
    return pd.read_csv(resolved)


def _read_optional_parquet(
    path: Path, *, allowed_root: Path, result: GenerationResult
) -> pd.DataFrame | None:
    resolved = validate_report_input_path(path, allowed_root=allowed_root)
    if not resolved.exists():
        result.note_skip(f"Skipped missing optional input: {resolved.name}")
        return None
    return pd.read_parquet(resolved)


def _read_first_present_csv(
    directory: Path,
    candidates: tuple[str, ...],
    *,
    result: GenerationResult,
) -> pd.DataFrame | None:
    for filename in candidates:
        path = directory / filename
        resolved = validate_report_input_path(path, allowed_root=directory)
        if resolved.exists():
            return pd.read_csv(resolved)
    result.note_skip("Skipped missing optional input: " + " / ".join(candidates))
    return None


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _normalize_common_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for alias, canonical in COMMON_COLUMN_ALIASES.items():
        if alias not in out.columns:
            continue
        if canonical in out.columns:
            out = out.drop(columns=[alias])
        else:
            out = out.rename(columns={alias: canonical})
    return out


def _split_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_common_aliases(df)
    if "model_name" not in out.columns and {"feature_set", "model"}.issubset(out.columns):
        out["model_name"] = out["feature_set"].astype(str) + "__" + out["model"].astype(str)

    if "model_name" in out.columns and (
        "feature_set" not in out.columns or "model" not in out.columns
    ):
        parts = out["model_name"].astype(str).str.split("__", n=1, expand=True)
        if "feature_set" not in out.columns:
            out["feature_set"] = parts[0]
        if "model" not in out.columns:
            if parts.shape[1] > 1:
                out["model"] = parts[1]
            else:
                out["model"] = out["model_name"].astype(str)
    return out


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _filter_main_spec(df: pd.DataFrame) -> pd.DataFrame:
    out = _coerce_numeric(df, ["cutoff", "cost_bps"])
    if "cutoff" in out.columns:
        out = out.loc[np.isclose(out["cutoff"], MAIN_CUTOFF)]
    if "cost_bps" in out.columns:
        out = out.loc[np.isclose(out["cost_bps"], MAIN_COST_BPS)]
    return out.reset_index(drop=True)


def _cost_label(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return f"{text}bps"


def _strategy_label(feature_set: str, model: str) -> str:
    return f"{feature_set} x {model}"


def _strategy_color(model: str) -> str:
    return MODEL_COLORS.get(model, "#4C78A8")


def _strategy_linestyle(feature_set: str) -> str:
    return FEATURE_SET_LINESTYLES.get(feature_set, "-.")


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    selected = [column for column in columns if column in available]
    if not selected:
        raise ValueError(f"None of the requested columns exist in {path.name}: {columns}")
    return pd.read_parquet(path, columns=selected)


def build_main_walkforward_table(model_summary: pd.DataFrame) -> pd.DataFrame:
    """Create the report-ready main walk-forward table for 10% cutoff / 10 bps."""

    summary = _split_strategy_columns(model_summary)
    summary = _filter_main_spec(summary)
    required = {
        "feature_set",
        "model",
        "gross_sharpe",
        "net_sharpe",
        "net_ann_return",
        "avg_turnover",
        "breakeven_bps",
        "rank_ic_mean",
        "rank_ic_hac_tstat",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Main summary is missing required columns: {sorted(missing)}")

    table = summary[
        [
            "feature_set",
            "model",
            "gross_sharpe",
            "net_sharpe",
            "net_ann_return",
            "avg_turnover",
            "breakeven_bps",
            "rank_ic_mean",
            "rank_ic_hac_tstat",
        ]
    ].rename(
        columns={
            "gross_sharpe": "gross_SR",
            "net_sharpe": "net_SR",
            "net_ann_return": "net_ann",
            "avg_turnover": "turnover",
            "rank_ic_mean": "rank_IC",
            "rank_ic_hac_tstat": "IC_t_HAC",
        }
    )
    return table.sort_values(
        ["net_SR", "gross_SR", "feature_set", "model"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _derive_cost_grid_from_main_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    summary = _split_strategy_columns(model_summary)
    summary = _coerce_numeric(summary, ["cutoff", "cost_bps", "net_sharpe", "breakeven_bps"])
    if "cutoff" in summary.columns:
        summary = summary.loc[np.isclose(summary["cutoff"], MAIN_CUTOFF)].copy()
    keep = ["feature_set", "model", "model_name", "cost_bps", "net_sharpe", "breakeven_bps"]
    available = [column for column in keep if column in summary.columns]
    return summary[available].drop_duplicates().reset_index(drop=True)


def _merge_cost_and_breakeven(
    cost_grid: pd.DataFrame | None,
    breakeven: pd.DataFrame | None,
    *,
    model_summary: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if cost_grid is None and model_summary is None:
        return None

    if cost_grid is None:
        merged = _derive_cost_grid_from_main_summary(model_summary)
    else:
        merged = _split_strategy_columns(cost_grid)
        merged = _coerce_numeric(merged, ["cutoff", "cost_bps", "net_sharpe", "breakeven_bps"])
        if "cutoff" in merged.columns:
            merged = merged.loc[np.isclose(merged["cutoff"], MAIN_CUTOFF)].copy()

    if breakeven is not None:
        breakeven_clean = _split_strategy_columns(breakeven)
        breakeven_clean = _coerce_numeric(breakeven_clean, ["cutoff", "breakeven_bps"])
        if "cutoff" in breakeven_clean.columns:
            breakeven_clean = breakeven_clean.loc[
                np.isclose(breakeven_clean["cutoff"], MAIN_CUTOFF)
            ]

        keep = ["feature_set", "model", "model_name", "breakeven_bps"]
        breakeven_clean = breakeven_clean[
            [column for column in keep if column in breakeven_clean.columns]
        ].drop_duplicates()
        join_keys = [
            column for column in ("feature_set", "model", "model_name") if column in merged.columns
        ]
        if "breakeven_bps" in merged.columns:
            merged = merged.drop(columns=["breakeven_bps"])
        merged = merged.merge(breakeven_clean, on=join_keys, how="left")

    return _split_strategy_columns(merged)


def build_cost_breakeven_table(
    cost_grid: pd.DataFrame | None,
    breakeven: pd.DataFrame | None,
    *,
    model_summary: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Create the cost-sensitivity / breakeven table for the main cutoff."""

    merged = _merge_cost_and_breakeven(cost_grid, breakeven, model_summary=model_summary)
    if merged is None or merged.empty:
        return None

    required = {"feature_set", "model", "cost_bps", "net_sharpe"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Cost grid is missing required columns: {sorted(missing)}")

    cost_values = sorted(merged["cost_bps"].dropna().unique().tolist())
    pivot = merged.pivot_table(
        index=["feature_set", "model"],
        columns="cost_bps",
        values="net_sharpe",
        aggfunc="first",
    ).reset_index()
    pivot.columns = [
        (
            "feature_set"
            if column == "feature_set"
            else "model" if column == "model" else f"net_SR_{_cost_label(float(column))}"
        )
        for column in pivot.columns
    ]

    if "breakeven_bps" in merged.columns:
        breakeven_table = (
            merged[["feature_set", "model", "breakeven_bps"]]
            .drop_duplicates(subset=["feature_set", "model"])
            .reset_index(drop=True)
        )
        pivot = pivot.merge(
            breakeven_table,
            on=["feature_set", "model"],
            how="left",
            validate="one_to_one",
        )

    sort_column = f"net_SR_{_cost_label(MAIN_COST_BPS)}"
    if sort_column not in pivot.columns and cost_values:
        sort_column = f"net_SR_{_cost_label(float(cost_values[0]))}"
    return pivot.sort_values(
        [sort_column, "feature_set", "model"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _plot_cost_sensitivity(cost_grid: pd.DataFrame, output_path: Path) -> Path:
    apply_style()
    grid = _split_strategy_columns(cost_grid)
    grid = _coerce_numeric(grid, ["cutoff", "cost_bps", "net_sharpe"])
    if "cutoff" in grid.columns:
        grid = grid.loc[np.isclose(grid["cutoff"], MAIN_CUTOFF)].copy()
    grid = grid.sort_values(["feature_set", "model", "cost_bps"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (feature_set, model), frame in grid.groupby(["feature_set", "model"], sort=False):
        ax.plot(
            frame["cost_bps"],
            frame["net_sharpe"],
            marker="o",
            linewidth=1.6,
            markersize=4.2,
            color=_strategy_color(model),
            linestyle=_strategy_linestyle(feature_set),
            label=_strategy_label(feature_set, model),
        )

    ax.set_xlabel("Transaction cost (bps)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Net Sharpe versus transaction cost")
    ax.legend(loc="best", ncol=2, fontsize=7.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path


def _plot_cumulative_net_returns(backtests: pd.DataFrame, output_path: Path) -> Path:
    apply_style()
    books = _split_strategy_columns(backtests)
    books = _coerce_numeric(books, ["cutoff", "cost_bps", "net_return"])
    if "cutoff" in books.columns:
        books = books.loc[np.isclose(books["cutoff"], MAIN_CUTOFF)]
    if "cost_bps" in books.columns:
        books = books.loc[np.isclose(books["cost_bps"], MAIN_COST_BPS)]
    if books.empty:
        raise ValueError("Backtests do not contain any 10% / 10bps rows for cumulative plotting.")

    date_col = "date" if "date" in books.columns else "month"
    books[date_col] = pd.to_datetime(books[date_col])

    monthly = (
        books.groupby(["feature_set", "model", date_col], as_index=False)["net_return"]
        .mean()
        .sort_values(date_col)
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (feature_set, model), frame in monthly.groupby(["feature_set", "model"], sort=False):
        wealth = (1.0 + frame["net_return"].fillna(0.0)).cumprod() - 1.0
        ax.plot(
            frame[date_col],
            wealth,
            linewidth=1.6,
            color=_strategy_color(model),
            linestyle=_strategy_linestyle(feature_set),
            label=_strategy_label(feature_set, model),
        )

    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return")
    ax.set_title("Cumulative net returns")
    ax.legend(loc="best", ncol=2, fontsize=7.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path


def build_prediction_decile_table(
    predictions: pd.DataFrame, *, n_deciles: int = 10
) -> pd.DataFrame:
    """Average realized next-month returns by prediction decile for each strategy."""

    preds = _split_strategy_columns(predictions)
    date_col = "date" if "date" in preds.columns else "month"
    target_col = _pick_first_present(
        preds, ("target_ret_fwd_1m", "target_excess_ret_fwd_1m", "realized_ret")
    )
    if target_col is None or "prediction" not in preds.columns:
        raise ValueError(
            "Predictions must contain a date/month column, prediction, and realized target column."
        )

    rows: list[pd.DataFrame] = []
    for (feature_set, model), frame in preds.groupby(["feature_set", "model"], sort=True):
        deciles = decile_spread(
            frame.rename(columns={"prediction": "y_hat"}),
            date_col=date_col,
            y_col=target_col,
            pred_col="y_hat",
            n_deciles=n_deciles,
        )
        deciles.insert(0, "model", model)
        deciles.insert(0, "feature_set", feature_set)
        deciles["top_minus_bottom_mean"] = float(
            deciles.attrs.get("top_minus_bottom_mean", float("nan"))
        )
        rows.append(deciles)

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_set",
                "model",
                "decile",
                "mean_realized_return",
                "top_minus_bottom_mean",
            ]
        )
    table = pd.concat(rows, ignore_index=True)
    return table.sort_values(
        ["top_minus_bottom_mean", "feature_set", "model", "decile"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def _plot_prediction_deciles(deciles: pd.DataFrame, output_path: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (feature_set, model), frame in deciles.groupby(["feature_set", "model"], sort=False):
        ordered = frame.sort_values("decile")
        ax.plot(
            ordered["decile"],
            ordered["mean_realized_return"],
            marker="o",
            linewidth=1.6,
            markersize=4.2,
            color=_strategy_color(model),
            linestyle=_strategy_linestyle(feature_set),
            label=_strategy_label(feature_set, model),
        )

    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Prediction decile")
    ax.set_ylabel("Mean realized next-month return")
    ax.set_title("Prediction-sort deciles")
    ax.legend(loc="best", ncol=2, fontsize=7.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path


def build_data_summary_table(panel_path: Path) -> pd.DataFrame:
    """One-row data summary from an existing processed panel parquet."""

    parquet = pq.ParquetFile(panel_path)
    columns = parquet.schema.names
    date_col = "date" if "date" in columns else "month"
    target_col = (
        "target_ret_fwd_1m"
        if "target_ret_fwd_1m" in columns
        else "target_excess_ret_fwd_1m" if "target_excess_ret_fwd_1m" in columns else None
    )
    if target_col is None or "permno" not in columns:
        raise ValueError(
            f"Processed panel must contain {date_col!r}, 'permno', and a target column."
        )

    panel = _read_parquet_columns(panel_path, [date_col, "permno", target_col])
    panel[date_col] = pd.to_datetime(panel[date_col])
    monthly_counts = panel.groupby(date_col)["permno"].nunique()
    excluded = {
        date_col,
        "permno",
        "ret",
        "sprtrn",
        "mkt_ret",
        "ret_market_adj",
        "target_ret_fwd_1m",
        "target_excess_ret_fwd_1m",
        "cusip",
        "cusip8",
        "ticker",
        "permco",
        "siccd",
        "naics",
    }
    feature_columns = [column for column in columns if column not in excluded]
    return pd.DataFrame(
        [
            {
                "sample_start": panel[date_col].min().date().isoformat(),
                "sample_end": panel[date_col].max().date().isoformat(),
                "stock_months": int(parquet.metadata.num_rows),
                "unique_stocks": int(panel["permno"].nunique()),
                "avg_stocks_per_month": float(monthly_counts.mean()),
                "n_features": int(len(feature_columns)),
                "target_col": target_col,
                "target_mean": float(panel[target_col].mean()),
                "target_vol": float(panel[target_col].std(ddof=1)),
            }
        ]
    )


def _select_strategy_row(
    df: pd.DataFrame,
    *,
    feature_set: str,
    model: str,
    target: str | None = None,
) -> pd.Series | None:
    frame = _split_strategy_columns(df)
    frame = _coerce_numeric(
        frame,
        [
            "cutoff",
            "cost_bps",
            "gross_sharpe",
            "net_sharpe",
            "net_ann_return",
            "breakeven_bps",
            "rank_ic_mean",
            "rank_ic_hac_tstat",
            "avg_turnover",
        ],
    )
    if "target" in frame.columns and target is not None:
        frame = frame.loc[frame["target"].astype(str) == target].copy()
    if "target_col" in frame.columns and target is not None:
        frame = frame.loc[frame["target_col"].astype(str) == target].copy()
    frame = _filter_main_spec(frame)
    selected = frame.loc[
        (frame["feature_set"].astype(str) == feature_set) & (frame["model"].astype(str) == model)
    ]
    if selected.empty:
        return None
    return selected.iloc[0]


def _build_excess_target_row(excess_summary: pd.DataFrame) -> dict[str, object] | None:
    row = _select_strategy_row(
        excess_summary,
        feature_set="flow_compustat",
        model="mlp",
        target="target_excess_ret_fwd_1m",
    )
    if row is None:
        row = _select_strategy_row(excess_summary, feature_set="flow_compustat", model="mlp")
    if row is None:
        return None
    return {
        "experiment": "excess_target",
        "feature_set": "flow_compustat",
        "model": "mlp",
        "target": "target_excess_ret_fwd_1m",
        "cutoff": float(row.get("cutoff", MAIN_CUTOFF)),
        "cost_bps": float(row.get("cost_bps", MAIN_COST_BPS)),
        "net_sharpe": float(row.get("net_sharpe", float("nan"))),
        "gross_sharpe": float(row.get("gross_sharpe", float("nan"))),
        "net_ann_return": float(row.get("net_ann_return", float("nan"))),
        "breakeven_bps": float(row.get("breakeven_bps", float("nan"))),
        "rank_ic_mean": float(row.get("rank_ic_mean", float("nan"))),
        "rank_ic_hac_tstat": float(row.get("rank_ic_hac_tstat", float("nan"))),
        "reference_feature_set": pd.NA,
        "reference_model": pd.NA,
        "reference_net_sharpe": pd.NA,
        "delta_net_sharpe": pd.NA,
        "seed_count": pd.NA,
        "mean_net_sharpe": pd.NA,
        "std_net_sharpe": pd.NA,
        "min_net_sharpe": pd.NA,
        "max_net_sharpe": pd.NA,
        "interpretation": (
            "supplementary excess-target robustness; does not replace the accepted "
            "raw-target main result"
        ),
        "source_file": EXCESS_TARGET_SUMMARY_FILENAME,
    }


def _build_multiseed_row(multiseed_summary: pd.DataFrame) -> dict[str, object] | None:
    frame = _split_strategy_columns(multiseed_summary)
    frame = _coerce_numeric(
        frame,
        [
            "n_seeds",
            "seed_count",
            "net_sharpe",
            "net_sharpe_mean",
            "mean_net_sharpe",
            "net_sharpe_std",
            "std_net_sharpe",
            "net_sharpe_min",
            "min_net_sharpe",
            "net_sharpe_max",
            "max_net_sharpe",
        ],
    )
    frame = frame.loc[
        (frame["feature_set"].astype(str) == "flow_compustat")
        & (frame["model"].astype(str) == "mlp")
    ].copy()
    if frame.empty:
        return None

    aggregate_row: pd.Series | None = None
    if "seed" in frame.columns:
        aggregate = frame.loc[frame["seed"].astype(str).str.contains("ALL", na=False)]
        if not aggregate.empty:
            aggregate_row = aggregate.iloc[0]
    if aggregate_row is None and "n_seeds" in frame.columns:
        aggregate = frame.loc[frame["n_seeds"].notna()]
        if not aggregate.empty:
            aggregate_row = aggregate.iloc[0]

    if aggregate_row is not None:
        seed_count = aggregate_row.get("n_seeds", aggregate_row.get("seed_count", float("nan")))
        mean_net_sharpe = aggregate_row.get(
            "net_sharpe_mean", aggregate_row.get("mean_net_sharpe", float("nan"))
        )
        std_net_sharpe = aggregate_row.get(
            "net_sharpe_std", aggregate_row.get("std_net_sharpe", float("nan"))
        )
        min_net_sharpe = aggregate_row.get(
            "net_sharpe_min", aggregate_row.get("min_net_sharpe", float("nan"))
        )
        max_net_sharpe = aggregate_row.get(
            "net_sharpe_max", aggregate_row.get("max_net_sharpe", float("nan"))
        )
    else:
        per_seed = frame.loc[frame["net_sharpe"].notna()].copy()
        if per_seed.empty:
            return None
        seed_count = (
            int(per_seed["seed"].nunique()) if "seed" in per_seed.columns else len(per_seed)
        )
        mean_net_sharpe = float(per_seed["net_sharpe"].mean())
        std_net_sharpe = float(per_seed["net_sharpe"].std(ddof=1))
        min_net_sharpe = float(per_seed["net_sharpe"].min())
        max_net_sharpe = float(per_seed["net_sharpe"].max())

    return {
        "experiment": "mlp_multiseed",
        "feature_set": "flow_compustat",
        "model": "mlp",
        "target": "target_ret_fwd_1m",
        "cutoff": MAIN_CUTOFF,
        "cost_bps": MAIN_COST_BPS,
        "net_sharpe": pd.NA,
        "gross_sharpe": pd.NA,
        "net_ann_return": pd.NA,
        "breakeven_bps": pd.NA,
        "rank_ic_mean": pd.NA,
        "rank_ic_hac_tstat": pd.NA,
        "reference_feature_set": pd.NA,
        "reference_model": pd.NA,
        "reference_net_sharpe": pd.NA,
        "delta_net_sharpe": pd.NA,
        "seed_count": int(seed_count) if pd.notna(seed_count) else pd.NA,
        "mean_net_sharpe": float(mean_net_sharpe),
        "std_net_sharpe": float(std_net_sharpe),
        "min_net_sharpe": float(min_net_sharpe),
        "max_net_sharpe": float(max_net_sharpe),
        "interpretation": (
            "supplementary MLP protocol-seed stability check; does not replace the "
            "accepted raw-target main result"
        ),
        "source_file": MLP_MULTISEED_SUMMARY_FILENAME,
    }


def _regime_ablation_interpretation(feature_set: str, model: str, delta_net_sharpe: float) -> str:
    if feature_set == "flow_compustat_regime":
        return (
            "supplementary direct regime-feature ablation reduces the accepted "
            f"flow_compustat x {model} specification"
        )
    if delta_net_sharpe > 0:
        return (
            "supplementary direct regime-feature ablation modestly helps this "
            "return-only specification, but it is not the accepted main result"
        )
    return (
        "supplementary direct regime-feature ablation is roughly flat to weaker "
        "than the return-only baseline"
    )


def _build_regime_ablation_rows(regime_ablation: pd.DataFrame) -> list[dict[str, object]]:
    frame = _split_strategy_columns(regime_ablation)
    frame = _coerce_numeric(
        frame,
        [
            "cutoff",
            "cost_bps",
            "gross_sharpe",
            "net_sharpe",
            "net_ann_return",
            "breakeven_bps",
            "rank_ic_mean",
            "rank_ic_hac_tstat",
            "baseline_net_sharpe",
            "reference_net_sharpe",
            "delta_net_sharpe_vs_baseline",
            "delta_net_sharpe",
        ],
    )
    frame = _filter_main_spec(frame)
    baseline_map = {
        "flow_compustat_regime": "flow_compustat",
        "return_regime": "return_only",
    }
    order = [
        ("flow_compustat_regime", "mlp"),
        ("flow_compustat_regime", "xgboost"),
        ("return_regime", "mlp"),
        ("return_regime", "xgboost"),
    ]
    rows: list[dict[str, object]] = []
    for feature_set, model in order:
        selected = frame.loc[
            (frame["feature_set"].astype(str) == feature_set)
            & (frame["model"].astype(str) == model)
        ]
        if selected.empty:
            continue
        row = selected.iloc[0]
        reference_feature_set = str(
            row.get(
                "baseline_feature_set", row.get("reference_feature_set", baseline_map[feature_set])
            )
        )
        reference_net_sharpe = row.get(
            "baseline_net_sharpe", row.get("reference_net_sharpe", float("nan"))
        )
        delta_net_sharpe = row.get(
            "delta_net_sharpe_vs_baseline", row.get("delta_net_sharpe", float("nan"))
        )
        rows.append(
            {
                "experiment": "regime_feature_ablation",
                "feature_set": feature_set,
                "model": model,
                "target": "target_ret_fwd_1m",
                "cutoff": float(row.get("cutoff", MAIN_CUTOFF)),
                "cost_bps": float(row.get("cost_bps", MAIN_COST_BPS)),
                "net_sharpe": float(row.get("net_sharpe", float("nan"))),
                "gross_sharpe": float(row.get("gross_sharpe", float("nan"))),
                "net_ann_return": float(row.get("net_ann_return", float("nan"))),
                "breakeven_bps": float(row.get("breakeven_bps", float("nan"))),
                "rank_ic_mean": float(row.get("rank_ic_mean", float("nan"))),
                "rank_ic_hac_tstat": float(row.get("rank_ic_hac_tstat", float("nan"))),
                "reference_feature_set": reference_feature_set,
                "reference_model": model,
                "reference_net_sharpe": float(reference_net_sharpe),
                "delta_net_sharpe": float(delta_net_sharpe),
                "seed_count": pd.NA,
                "mean_net_sharpe": pd.NA,
                "std_net_sharpe": pd.NA,
                "min_net_sharpe": pd.NA,
                "max_net_sharpe": pd.NA,
                "interpretation": _regime_ablation_interpretation(
                    feature_set, model, float(delta_net_sharpe)
                ),
                "source_file": REGIME_ABLATION_FILENAME,
            }
        )
    return rows


def build_supplementary_robustness_table(
    excess_summary: pd.DataFrame | None,
    multiseed_summary: pd.DataFrame | None,
    regime_ablation: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if excess_summary is not None and not excess_summary.empty:
        excess_row = _build_excess_target_row(excess_summary)
        if excess_row is not None:
            rows.append(excess_row)
    if multiseed_summary is not None and not multiseed_summary.empty:
        multiseed_row = _build_multiseed_row(multiseed_summary)
        if multiseed_row is not None:
            rows.append(multiseed_row)
    if regime_ablation is not None and not regime_ablation.empty:
        rows.extend(_build_regime_ablation_rows(regime_ablation))

    table = pd.DataFrame(rows, columns=SUPPLEMENTARY_ROBUSTNESS_COLUMNS)
    if table.empty:
        return table

    experiment_order = {
        "excess_target": 0,
        "mlp_multiseed": 1,
        "regime_feature_ablation": 2,
    }
    feature_order = {
        "flow_compustat": 0,
        "flow_compustat_regime": 1,
        "return_regime": 2,
    }
    model_order = {"mlp": 0, "xgboost": 1}
    table = table.assign(
        _experiment_order=table["experiment"].map(experiment_order).fillna(99),
        _feature_order=table["feature_set"].map(feature_order).fillna(99),
        _model_order=table["model"].map(model_order).fillna(99),
    )
    table = table.sort_values(
        ["_experiment_order", "_feature_order", "_model_order"],
        ascending=[True, True, True],
    ).drop(columns=["_experiment_order", "_feature_order", "_model_order"])
    return table.reset_index(drop=True)


def build_final_model_hierarchy_table(
    main_summary: pd.DataFrame, classical_baseline_summary: pd.DataFrame
) -> pd.DataFrame:
    main = _split_strategy_columns(main_summary)
    main = _coerce_numeric(
        main,
        [
            "cutoff",
            "cost_bps",
            "gross_sharpe",
            "net_sharpe",
            "net_ann_return",
            "avg_turnover",
            "breakeven_bps",
            "rank_ic_mean",
            "rank_ic_hac_tstat",
        ],
    )
    main = _filter_main_spec(main)

    classical = _split_strategy_columns(classical_baseline_summary)
    classical = _coerce_numeric(
        classical,
        [
            "cutoff",
            "cost_bps",
            "gross_sharpe",
            "net_sharpe",
            "net_ann_return",
            "avg_turnover",
            "breakeven_bps",
            "rank_ic_mean",
            "rank_ic_hac_tstat",
        ],
    )
    classical = _filter_main_spec(classical)

    rows: list[dict[str, object]] = []
    classical_specs = [
        ("reversal", "classical one-month reversal benchmark"),
        ("zscore_reversal", "classical z-score reversal benchmark"),
        ("momentum", "strongest classical same-protocol benchmark"),
        ("reversal_plus_momentum", "combined classical reversal plus momentum benchmark"),
    ]
    for model, interpretation in classical_specs:
        selected = classical.loc[classical["model"].astype(str) == model]
        if selected.empty:
            continue
        row = selected.iloc[0]
        rows.append(
            {
                "strategy": model,
                "role": "classical_baseline",
                "feature_set": str(row["feature_set"]),
                "model": model,
                "cutoff": float(row.get("cutoff", MAIN_CUTOFF)),
                "cost_bps": float(row.get("cost_bps", MAIN_COST_BPS)),
                "net_sharpe": float(row.get("net_sharpe", float("nan"))),
                "gross_sharpe": float(row.get("gross_sharpe", float("nan"))),
                "net_ann_return": float(row.get("net_ann_return", float("nan"))),
                "turnover": float(row.get("avg_turnover", float("nan"))),
                "breakeven_bps": float(row.get("breakeven_bps", float("nan"))),
                "rank_ic_mean": float(row.get("rank_ic_mean", float("nan"))),
                "rank_ic_hac_tstat": float(row.get("rank_ic_hac_tstat", float("nan"))),
                "source_table": CLASSICAL_BASELINE_TABLE_FILENAME,
                "interpretation": interpretation,
            }
        )

    ml_specs = [
        ("return_only", "xgboost", "return_only_ml", "accepted tree-based return-only comparison"),
        ("return_only", "mlp", "return_only_ml", "accepted nonlinear return-only comparison"),
        (
            "flow_compustat",
            "xgboost",
            "flow_compustat_ml",
            "accepted tree-based nonlinear comparison",
        ),
        (
            "flow_compustat",
            "mlp",
            "flow_compustat_ml",
            "accepted main nonlinear specification",
        ),
    ]
    for feature_set, model, role, interpretation in ml_specs:
        selected = main.loc[
            (main["feature_set"].astype(str) == feature_set) & (main["model"].astype(str) == model)
        ]
        if selected.empty:
            continue
        row = selected.iloc[0]
        rows.append(
            {
                "strategy": f"{feature_set} x {model}",
                "role": role,
                "feature_set": feature_set,
                "model": model,
                "cutoff": float(row.get("cutoff", MAIN_CUTOFF)),
                "cost_bps": float(row.get("cost_bps", MAIN_COST_BPS)),
                "net_sharpe": float(row.get("net_sharpe", float("nan"))),
                "gross_sharpe": float(row.get("gross_sharpe", float("nan"))),
                "net_ann_return": float(row.get("net_ann_return", float("nan"))),
                "turnover": float(row.get("avg_turnover", float("nan"))),
                "breakeven_bps": float(row.get("breakeven_bps", float("nan"))),
                "rank_ic_mean": float(row.get("rank_ic_mean", float("nan"))),
                "rank_ic_hac_tstat": float(row.get("rank_ic_hac_tstat", float("nan"))),
                "source_table": MAIN_TABLE_FILENAME,
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows, columns=FINAL_MODEL_HIERARCHY_COLUMNS)


def _prepare_optional_table(df: pd.DataFrame, *, main_metric: str | None = None) -> pd.DataFrame:
    out = _split_strategy_columns(df)
    out = _coerce_numeric(out, ["cutoff", "cost_bps"])
    if "cutoff" in out.columns:
        out = out.loc[np.isclose(out["cutoff"], MAIN_CUTOFF)].copy()
    if "cost_bps" in out.columns:
        exact = out.loc[np.isclose(out["cost_bps"], MAIN_COST_BPS)].copy()
        if not exact.empty:
            out = exact
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if main_metric and main_metric in out.columns:
        sort_cols.append(main_metric)
        ascending.append(False)
    for column in ("feature_set", "model", "regime"):
        if column in out.columns:
            sort_cols.append(column)
            ascending.append(True)
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=ascending)
    return out.reset_index(drop=True)


def _pick_first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _plot_regime_net_sharpe(regime_table: pd.DataFrame, output_path: Path) -> Path:
    apply_style()
    regimes = _split_strategy_columns(regime_table)
    regimes = _coerce_numeric(regimes, ["cutoff", "cost_bps"])
    if "cutoff" in regimes.columns:
        regimes = regimes.loc[np.isclose(regimes["cutoff"], MAIN_CUTOFF)]
    if "cost_bps" in regimes.columns:
        exact = regimes.loc[np.isclose(regimes["cost_bps"], MAIN_COST_BPS)]
        if not exact.empty:
            regimes = exact

    regime_col = _pick_first_present(
        regimes, ("regime", "regime_name", "state", "bucket", "segment", "split_name")
    )
    metric_col = _pick_first_present(regimes, ("net_sharpe", "net_SR", "net_sharpe_10bps"))
    if regime_col is None or metric_col is None:
        raise ValueError(
            "Regime robustness file must contain a regime label column and a net-Sharpe column."
        )

    regimes = regimes.rename(columns={regime_col: "regime_label", metric_col: "net_sharpe_value"})
    regimes = regimes.dropna(subset=["regime_label", "net_sharpe_value"]).copy()
    regimes["strategy_label"] = regimes.apply(
        lambda row: _strategy_label(str(row["feature_set"]), str(row["model"])), axis=1
    )

    strategy_order = (
        regimes.groupby(["feature_set", "model"], as_index=False)["net_sharpe_value"]
        .mean()
        .sort_values("net_sharpe_value", ascending=False)
    )
    top_pairs = list(strategy_order[["feature_set", "model"]].itertuples(index=False, name=None))
    if len(top_pairs) > 4:
        keep = set(top_pairs[:4])
        regimes = regimes.loc[
            regimes.apply(lambda row: (row["feature_set"], row["model"]) in keep, axis=1)
        ].copy()

    plot_data = regimes.pivot_table(
        index="regime_label",
        columns="strategy_label",
        values="net_sharpe_value",
        aggfunc="first",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(plot_data.index))
    n_series = max(len(plot_data.columns), 1)
    width = 0.8 / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * width
    for idx, label in enumerate(plot_data.columns):
        ax.bar(x + offsets[idx], plot_data[label].to_numpy(), width=width, label=label)

    ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, rotation=20, ha="right")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Regime robustness: net Sharpe")
    ax.legend(loc="best", ncol=2, fontsize=7.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)
    return output_path


def generate_final_report_outputs(paths: FinalReportPaths | None = None) -> GenerationResult:
    """Generate report tables/figures from accepted canonical walk-forward outputs."""

    resolved_paths = paths or build_final_report_paths()
    result = GenerationResult()

    main_summary = _read_optional_csv(
        resolved_paths.metrics_dir / MAIN_SUMMARY_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    cost_grid = _read_optional_csv(
        resolved_paths.metrics_dir / COST_GRID_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    breakeven = _read_optional_csv(
        resolved_paths.metrics_dir / BREAKEVEN_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    rebalance = _read_optional_csv(
        resolved_paths.metrics_dir / REBALANCE_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    regime = _read_optional_csv(
        resolved_paths.metrics_dir / REGIME_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    alpha_beta = _read_optional_csv(
        resolved_paths.metrics_dir / ALPHA_BETA_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    difference_tests = _read_optional_csv(
        resolved_paths.metrics_dir / DIFFERENCE_TESTS_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    classical_baseline_summary = _read_optional_csv(
        resolved_paths.metrics_dir / CLASSICAL_BASELINE_SUMMARY_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    classical_baseline_difference_tests = _read_optional_csv(
        resolved_paths.metrics_dir / CLASSICAL_BASELINE_DIFFERENCE_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    excess_target_summary = _read_optional_csv(
        resolved_paths.metrics_dir / EXCESS_TARGET_SUMMARY_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    mlp_multiseed_summary = _read_optional_csv(
        resolved_paths.metrics_dir / MLP_MULTISEED_SUMMARY_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    regime_ablation = _read_optional_csv(
        resolved_paths.metrics_dir / REGIME_ABLATION_FILENAME,
        allowed_root=resolved_paths.metrics_dir,
        result=result,
    )
    feature_importance = _read_first_present_csv(
        resolved_paths.metrics_dir,
        WALKFORWARD_XGB_IMPORTANCE_CANDIDATES,
        result=result,
    )
    backtests = _read_optional_parquet(
        resolved_paths.backtests_dir / BACKTESTS_FILENAME,
        allowed_root=resolved_paths.backtests_dir,
        result=result,
    )

    predictions_path = validate_report_input_path(
        resolved_paths.predictions_dir / PREDICTIONS_FILENAME,
        allowed_root=resolved_paths.predictions_dir,
    )
    panel_path = validate_report_input_path(
        resolved_paths.processed_dir / MODEL_PANEL_FILENAME,
        allowed_root=resolved_paths.processed_dir,
    )

    if main_summary is not None:
        main_table = build_main_walkforward_table(main_summary)
        main_path = _write_csv(main_table, resolved_paths.report_tables_dir / MAIN_TABLE_FILENAME)
        result.written.append(main_path)
    else:
        result.note_skip(f"Skipped {MAIN_TABLE_FILENAME}: {MAIN_SUMMARY_FILENAME} is absent.")

    cost_table = build_cost_breakeven_table(cost_grid, breakeven, model_summary=main_summary)
    if cost_table is not None and not cost_table.empty:
        cost_table_path = _write_csv(
            cost_table, resolved_paths.report_tables_dir / COST_TABLE_FILENAME
        )
        result.written.append(cost_table_path)

        long_cost_grid = _merge_cost_and_breakeven(cost_grid, breakeven, model_summary=main_summary)
        if long_cost_grid is not None and not long_cost_grid.empty:
            cost_fig_path = _plot_cost_sensitivity(
                long_cost_grid, resolved_paths.report_figures_dir / COST_FIGURE_FILENAME
            )
            result.written.append(cost_fig_path)
    else:
        result.note_skip(
            f"Skipped {COST_TABLE_FILENAME} and {COST_FIGURE_FILENAME}: no accepted cost grid data."
        )

    if backtests is not None and not backtests.empty:
        cumulative_path = _plot_cumulative_net_returns(
            backtests, resolved_paths.report_figures_dir / CUMULATIVE_FIGURE_FILENAME
        )
        result.written.append(cumulative_path)
    else:
        result.note_skip(
            f"Skipped {CUMULATIVE_FIGURE_FILENAME}: {BACKTESTS_FILENAME} is absent or empty."
        )

    if regime is not None and not regime.empty:
        regime_table = _prepare_optional_table(regime, main_metric="net_sharpe")
        regime_table_path = _write_csv(
            regime_table, resolved_paths.report_tables_dir / REGIME_TABLE_FILENAME
        )
        result.written.append(regime_table_path)
        regime_fig_path = _plot_regime_net_sharpe(
            regime_table, resolved_paths.report_figures_dir / REGIME_FIGURE_FILENAME
        )
        result.written.append(regime_fig_path)
    else:
        result.note_skip(
            f"Skipped {REGIME_TABLE_FILENAME} and {REGIME_FIGURE_FILENAME}: "
            f"{REGIME_FILENAME} is absent."
        )

    if alpha_beta is not None and not alpha_beta.empty:
        alpha_beta_table = _prepare_optional_table(
            alpha_beta,
            main_metric=_pick_first_present(alpha_beta, ("alpha_tstat", "alpha", "net_sharpe")),
        )
        alpha_beta_path = _write_csv(
            alpha_beta_table, resolved_paths.report_tables_dir / ALPHA_BETA_TABLE_FILENAME
        )
        result.written.append(alpha_beta_path)
    else:
        result.note_skip(f"Skipped {ALPHA_BETA_TABLE_FILENAME}: {ALPHA_BETA_FILENAME} is absent.")

    if rebalance is not None and not rebalance.empty:
        rebalance_table = _prepare_optional_table(rebalance, main_metric="net_sharpe")
        rebalance_path = _write_csv(
            rebalance_table, resolved_paths.report_tables_dir / REBALANCE_TABLE_FILENAME
        )
        result.written.append(rebalance_path)
    else:
        result.note_skip(f"Skipped {REBALANCE_TABLE_FILENAME}: {REBALANCE_FILENAME} is absent.")

    if difference_tests is not None and not difference_tests.empty:
        if "baseline_source" in difference_tests.columns:
            difference_tests = difference_tests.loc[
                ~difference_tests["baseline_source"]
                .astype("string")
                .str.contains("different protocol", case=False, na=False)
            ].copy()
        diff_table = _prepare_optional_table(
            difference_tests,
            main_metric=_pick_first_present(
                difference_tests, ("difference_tstat", "p_value", "mean_difference")
            ),
        )
        if not diff_table.empty:
            diff_path = _write_csv(
                diff_table, resolved_paths.report_tables_dir / DIFFERENCE_TESTS_TABLE_FILENAME
            )
            result.written.append(diff_path)
        else:
            result.note_skip(
                f"Skipped {DIFFERENCE_TESTS_TABLE_FILENAME}: no same-protocol rows remained."
            )
    else:
        result.note_skip(
            f"Skipped {DIFFERENCE_TESTS_TABLE_FILENAME}: {DIFFERENCE_TESTS_FILENAME} is absent."
        )

    if classical_baseline_summary is not None and not classical_baseline_summary.empty:
        classical_baseline_table = build_main_walkforward_table(classical_baseline_summary)
        classical_baseline_path = _write_csv(
            classical_baseline_table,
            resolved_paths.report_tables_dir / CLASSICAL_BASELINE_TABLE_FILENAME,
        )
        result.written.append(classical_baseline_path)
    else:
        result.note_skip(
            f"Skipped {CLASSICAL_BASELINE_TABLE_FILENAME}: "
            f"{CLASSICAL_BASELINE_SUMMARY_FILENAME} is absent."
        )

    if (
        classical_baseline_difference_tests is not None
        and not classical_baseline_difference_tests.empty
    ):
        classical_baseline_diff_table = _prepare_optional_table(
            classical_baseline_difference_tests,
            main_metric=_pick_first_present(
                classical_baseline_difference_tests,
                ("difference_tstat", "nw_tstat_diff", "p_value", "mean_difference"),
            ),
        )
        classical_baseline_diff_path = _write_csv(
            classical_baseline_diff_table,
            resolved_paths.report_tables_dir / CLASSICAL_BASELINE_DIFFERENCE_TABLE_FILENAME,
        )
        result.written.append(classical_baseline_diff_path)
    else:
        result.note_skip(
            f"Skipped {CLASSICAL_BASELINE_DIFFERENCE_TABLE_FILENAME}: "
            f"{CLASSICAL_BASELINE_DIFFERENCE_FILENAME} is absent."
        )

    supplementary_table = build_supplementary_robustness_table(
        excess_target_summary,
        mlp_multiseed_summary,
        regime_ablation,
    )
    if not supplementary_table.empty:
        supplementary_path = _write_csv(
            supplementary_table,
            resolved_paths.report_tables_dir / SUPPLEMENTARY_ROBUSTNESS_TABLE_FILENAME,
        )
        result.written.append(supplementary_path)
    else:
        result.note_skip(
            f"Skipped {SUPPLEMENTARY_ROBUSTNESS_TABLE_FILENAME}: no supplementary "
            "robustness inputs were present."
        )

    if (
        main_summary is not None
        and not main_summary.empty
        and classical_baseline_summary is not None
        and not classical_baseline_summary.empty
    ):
        final_hierarchy = build_final_model_hierarchy_table(
            main_summary, classical_baseline_summary
        )
        if not final_hierarchy.empty:
            final_hierarchy_path = _write_csv(
                final_hierarchy,
                resolved_paths.report_tables_dir / FINAL_MODEL_HIERARCHY_TABLE_FILENAME,
            )
            result.written.append(final_hierarchy_path)
        else:
            result.note_skip(
                f"Skipped {FINAL_MODEL_HIERARCHY_TABLE_FILENAME}: required hierarchy rows "
                "were not available."
            )
    else:
        result.note_skip(
            f"Skipped {FINAL_MODEL_HIERARCHY_TABLE_FILENAME}: "
            "main or classical baseline summaries are absent."
        )

    if predictions_path.exists():
        prediction_cols = _read_parquet_columns(
            predictions_path,
            [
                "date",
                "month",
                "prediction",
                "target_ret_fwd_1m",
                "target_excess_ret_fwd_1m",
                "realized_ret",
                "feature_set",
                "model_name",
            ],
        )
        deciles = build_prediction_decile_table(prediction_cols)
        decile_table_path = _write_csv(
            deciles, resolved_paths.report_tables_dir / DECILE_TABLE_FILENAME
        )
        result.written.append(decile_table_path)
        decile_fig_path = _plot_prediction_deciles(
            deciles, resolved_paths.report_figures_dir / DECILE_FIGURE_FILENAME
        )
        result.written.append(decile_fig_path)
    else:
        result.note_skip(
            f"Skipped {DECILE_TABLE_FILENAME} and {DECILE_FIGURE_FILENAME}: "
            f"{PREDICTIONS_FILENAME} is absent."
        )

    if panel_path.exists():
        data_summary = build_data_summary_table(panel_path)
        data_summary_path = _write_csv(
            data_summary, resolved_paths.report_tables_dir / DATA_SUMMARY_TABLE_FILENAME
        )
        result.written.append(data_summary_path)
    else:
        result.note_skip(
            f"Skipped {DATA_SUMMARY_TABLE_FILENAME}: {MODEL_PANEL_FILENAME} is absent."
        )

    if feature_importance is not None and not feature_importance.empty:
        if {"feature", "importance"}.issubset(feature_importance.columns):
            feature_table = feature_importance.sort_values(
                "importance", ascending=False
            ).reset_index(drop=True)
            feature_path = _write_csv(
                feature_table.head(20),
                resolved_paths.report_tables_dir / FEATURE_IMPORTANCE_TABLE_FILENAME,
            )
            result.written.append(feature_path)
        else:
            result.note_skip(
                f"Skipped {FEATURE_IMPORTANCE_TABLE_FILENAME}: feature importance file "
                "does not contain 'feature' and 'importance' columns."
            )
    else:
        result.note_skip(
            f"Skipped {FEATURE_IMPORTANCE_TABLE_FILENAME}: no walk-forward XGBoost "
            "feature-importance artifact is present."
        )

    return result


__all__ = [
    "ALPHA_BETA_TABLE_FILENAME",
    "COST_FIGURE_FILENAME",
    "COST_TABLE_FILENAME",
    "CLASSICAL_BASELINE_DIFFERENCE_TABLE_FILENAME",
    "CLASSICAL_BASELINE_TABLE_FILENAME",
    "CUMULATIVE_FIGURE_FILENAME",
    "DATA_SUMMARY_TABLE_FILENAME",
    "DECILE_FIGURE_FILENAME",
    "DECILE_TABLE_FILENAME",
    "FEATURE_IMPORTANCE_TABLE_FILENAME",
    "GenerationResult",
    "MAIN_TABLE_FILENAME",
    "FinalReportPaths",
    "REGIME_FIGURE_FILENAME",
    "REGIME_TABLE_FILENAME",
    "build_data_summary_table",
    "build_cost_breakeven_table",
    "build_main_walkforward_table",
    "build_prediction_decile_table",
    "build_final_report_paths",
    "generate_final_report_outputs",
    "validate_report_input_path",
]
