"""Cumulative long-short wealth-path figure for the FIN-407 final report.

Aggregates per-month long-short returns across folds and seeds for each model,
then plots $1-invested cumulative wealth on a log y-axis from 1987 to 2024.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mlfinance.eval.figures.style import (
    EXCLUDED_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
)

_MODEL_ORDER = ["ridge", "lasso", "mlp", "ft_transformer", "xgb", "stacking"]


def plot_wealth_path(
    *,
    tables_dir: Path,
    figures_dir: Path,
    predictions_dir: Path | None = None,  # unused, kept for common signature
) -> Path:
    """Cumulative wealth path per model, 1987-2024. Returns the output PDF path."""
    apply_style()
    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    summary_path = tables_dir / "uncertainty_sort_summary.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Required input not found: {summary_path}. "
            "Run the uncertainty-sort evaluation step before plotting wealth paths."
        )

    df = pd.read_parquet(summary_path)
    df["date"] = pd.to_datetime(df["date"])
    # Drop models that don't have full-panel coverage (MLP folds 1-3 diverged).
    df = df.loc[~df["model"].isin(EXCLUDED_MODELS)]

    # Average over seeds within (model, fold, date), then average over folds.
    # Each (model, fold) window is disjoint in time, so the fold-average is
    # really a concatenation across the 5 fold windows.
    per_seed_avg = df.groupby(["model", "fold", "date"], as_index=False)["ls_ret"].mean()
    monthly = per_seed_avg.groupby(["model", "date"], as_index=False)["ls_ret"].mean()

    # Build a wide (date x model) frame on a unified monthly index.
    wide = monthly.pivot(index="date", columns="model", values="ls_ret").sort_index()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "wealth_path.pdf"

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    plotted_any = False
    for model in _MODEL_ORDER:
        if model not in wide.columns:
            continue
        series = wide[model].dropna()
        if series.empty:
            continue

        # Cumulative wealth with $1 seed, reinvested at the first available date.
        wealth = (1.0 + series).cumprod()
        ax.plot(
            wealth.index,
            wealth.values,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model),
            linewidth=1.2,
        )
        plotted_any = True

    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (\\$1 invested)")
    ax.set_title("Cumulative long-short wealth path, 1987-2024")
    if plotted_any:
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

    return out_path


__all__ = ["plot_wealth_path"]
