"""Drawdown curves for the per-model long-short strategy.

Reads ``uncertainty_sort_summary.parquet``, aggregates ``ls_ret`` across seeds
(mean per date), concatenates folds, and plots cumulative-wealth drawdowns as
one line per model on a single axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from mlfinance.eval.figures.style import (
    EXCLUDED_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
)

__all__ = ["plot_drawdowns"]


_MODEL_ORDER = ["ridge", "lasso", "mlp", "ft_transformer", "xgb", "stacking"]


def plot_drawdowns(
    *,
    tables_dir: Path,
    figures_dir: Path,
    predictions_dir: Path | None = None,
) -> Path:
    """Drawdown curve per model. Returns the output PDF path."""
    apply_style()
    del predictions_dir  # accepted for caller-uniformity; not used here.

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    summary_path = tables_dir / "uncertainty_sort_summary.parquet"
    if not summary_path.is_file():
        raise FileNotFoundError(f"uncertainty_sort_summary.parquet not found at {summary_path!s}.")

    df = pd.read_parquet(summary_path)
    df = df.loc[:, ["date", "ls_ret", "model", "fold", "seed"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ls_ret"] = df["ls_ret"].astype(np.float64)
    # Drop models that don't have full-panel coverage (MLP folds 1-3 diverged).
    df = df.loc[~df["model"].isin(EXCLUDED_MODELS)]

    # Aggregate seeds: mean ls_ret per (model, fold, date). Concatenating folds
    # then naturally chains the cumulative wealth across the walk-forward
    # window. Folds are disjoint in date, so a simple sort-by-date is enough.
    per_date = (
        df.groupby(["model", "fold", "date"], sort=False)["ls_ret"]
        .mean()
        .reset_index()
        .sort_values(["model", "date"])
        .reset_index(drop=True)
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "drawdowns.pdf"

    # Use the canonical model ordering so colors stay consistent across figures.
    present = set(per_date["model"].unique())
    models = [m for m in _MODEL_ORDER if m in present] + sorted(present - set(_MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    # Faint shading of the negative region so the eye reads drawdowns as
    # losses, not as a generic time series that could cross zero.
    ax.axhspan(-100, 0, color="lightgray", alpha=0.08, zorder=0)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5, zorder=1)

    for model_name in models:
        sub = per_date.loc[per_date["model"] == model_name].sort_values("date")
        if sub.empty:
            continue
        dates = sub["date"].to_numpy()
        rets = sub["ls_ret"].to_numpy()
        wealth = np.cumprod(1.0 + rets)
        running_max = np.maximum.accumulate(wealth)
        drawdown_pct = (wealth / running_max - 1.0) * 100.0

        color = MODEL_COLORS.get(model_name)
        ax.plot(
            dates,
            drawdown_pct,
            color=color,
            linewidth=1.1,
            label=MODEL_LABELS.get(model_name, model_name),
            zorder=2,
        )

        trough_idx = int(np.argmin(drawdown_pct))
        ax.plot(
            dates[trough_idx],
            drawdown_pct[trough_idx],
            marker="v",
            markersize=4,
            color=color if color else "black",
            linestyle="None",
            zorder=3,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Long-short strategy drawdowns, 1987-2024")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.0f}%"))
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)

    return output_path
