"""36-month rolling long-short Sharpe figure for the FIN-407 final report.

Reads ``uncertainty_sort_summary.parquet`` (one row per (model, fold, seed,
date)), averages the per-month long-short return across seeds, concatenates
folds into a single 1987-2024 monthly series per model, computes the
annualised 36-month rolling Sharpe ratio (``mean / std * sqrt(12)``) with
``min_periods=24`` so each model's line begins ~two years after its first
observation, and writes a single PDF with one line per model and a y=0
reference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlfinance.eval.figures.style import (
    EXCLUDED_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
)

logger = logging.getLogger(__name__)

__all__ = ["plot_rolling_sharpe"]


_REQUIRED_COLS: tuple[str, ...] = (
    "date",
    "ls_ret",
    "model",
    "fold",
    "seed",
)

_MODEL_ORDER = ["ridge", "lasso", "mlp", "ft_transformer", "xgb", "stacking"]


def plot_rolling_sharpe(
    *,
    tables_dir: Path,
    figures_dir: Path,
    predictions_dir: Path | None = None,
    window: int = 36,
    min_periods: int = 24,
) -> Path:
    """Rolling 36m Sharpe per model. Returns the output PDF path."""
    apply_style()
    del predictions_dir  # unused; kept for signature parity with sibling figures.

    if window <= 0:
        raise ValueError(f"window must be positive, got {window}.")
    if min_periods <= 0 or min_periods > window:
        raise ValueError(
            f"min_periods must satisfy 0 < min_periods <= window={window}, " f"got {min_periods}."
        )

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    summary_path = tables_dir / "uncertainty_sort_summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"uncertainty_sort_summary.parquet not found at {summary_path}.")

    summary = pd.read_parquet(summary_path)
    missing = [c for c in _REQUIRED_COLS if c not in summary.columns]
    if missing:
        raise KeyError(f"uncertainty_sort_summary.parquet is missing columns {missing}.")

    summary = summary.loc[:, list(_REQUIRED_COLS)].copy()
    summary["date"] = pd.to_datetime(summary["date"])
    summary["ls_ret"] = summary["ls_ret"].astype(np.float64)
    # Drop models that don't have full-panel coverage (MLP folds 1-3 diverged).
    summary = summary.loc[~summary["model"].isin(EXCLUDED_MODELS)]
    # Drop models that don't have full-panel coverage (MLP folds 1-3 diverged).
    summary = summary.loc[~summary["model"].isin(EXCLUDED_MODELS)]

    # Aggregate across seeds: one ls_ret per (model, fold, date).
    fold_level = (
        summary.groupby(["model", "fold", "date"], sort=False, observed=True)["ls_ret"]
        .mean()
        .reset_index()
    )

    # Concatenate folds into a single monthly series per model. Fold-boundary
    # overlaps (a given (model, date) appearing in two consecutive folds, e.g.
    # the test month shared between rolling windows) are averaged out so each
    # date contributes exactly once to the rolling Sharpe.
    monthly = (
        fold_level.groupby(["model", "date"], sort=False, observed=True)["ls_ret"]
        .mean()
        .reset_index()
        .sort_values(["model", "date"])
        .reset_index(drop=True)
    )

    annualisation = float(np.sqrt(12.0))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    # Light shading of the [0, 1] Sharpe corridor so the reader has a
    # visual anchor for "above zero but sub-1" performance.
    ax.axhspan(0.0, 1.0, color="lightgray", alpha=0.10, zorder=0)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5, zorder=1)

    # Iterate in canonical model order so colors stay consistent across figures.
    present_models = set(monthly["model"].unique())
    ordered = [m for m in _MODEL_ORDER if m in present_models] + sorted(
        present_models - set(_MODEL_ORDER)
    )
    for model_name in ordered:
        grp = monthly.loc[monthly["model"] == model_name]
        s = grp.set_index("date")["ls_ret"].sort_index()
        roll_mean = s.rolling(window=window, min_periods=min_periods).mean()
        roll_std = s.rolling(window=window, min_periods=min_periods).std(ddof=1)
        sharpe = (roll_mean / roll_std) * annualisation
        sharpe = sharpe.dropna()
        if sharpe.empty:
            logger.warning(
                "plot_rolling_sharpe: model %s has no rolling-Sharpe points "
                "(need >= %d months); skipping.",
                model_name,
                min_periods,
            )
            continue
        ax.plot(
            sharpe.index,
            sharpe.to_numpy(),
            label=MODEL_LABELS.get(model_name, str(model_name)),
            color=MODEL_COLORS.get(model_name),
            linewidth=1.2,
        )

    ax.set_title("36-month rolling long-short Sharpe (annualized), 1989-2024")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe (annualized)")
    ax.legend(loc="best", frameon=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "rolling_sharpe.pdf"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
