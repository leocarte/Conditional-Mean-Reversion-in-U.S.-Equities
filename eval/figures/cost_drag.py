"""Sharpe-degradation grouped bar chart under three transaction-cost regimes.

Reads ``backtest_summary.parquet``, averages the gross, IBKR-tiered-net and
flat-net Sharpe ratios across folds and seeds within each model, and plots
them as a grouped bar chart with one cluster per model.

Cost regimes are encoded by luminance (light to dark grey-tinted shades of
the model's colour) so the eye groups by model cluster first, cost regime
second. This was a deliberate change away from three competing primary
colours that fought the model palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from mlfinance.eval.figures.style import (
    EXCLUDED_MODELS,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
)

__all__ = ["plot_cost_drag"]


_MODEL_ORDER = ["ridge", "lasso", "mlp", "ft_transformer", "xgb", "stacking"]

# (column, display label). Order = visual order in each cluster, left to right.
_REGIMES: list[tuple[str, str]] = [
    ("sharpe_gross", "Gross"),
    ("sharpe_net_ibkr", "Net (IBKR Pro tiered)"),
    ("sharpe_net_flat", "Net (flat 10 bp + impact)"),
]


def _lighten(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Linearly mix ``hex_color`` toward white by ``amount`` in [0, 1].

    amount=0 returns the base colour; amount=1 returns white. Cheap
    approximation of an HLS lightness shift that keeps hue/saturation
    perception roughly intact within a single hue family.
    """
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def plot_cost_drag(
    *,
    tables_dir: Path,
    figures_dir: Path,
    predictions_dir: Path | None = None,
) -> Path:
    """Sharpe by cost regime per model, grouped bar chart. Returns output path."""
    apply_style()
    del predictions_dir  # accepted for caller-uniformity; not used here.

    tables_dir = Path(tables_dir)
    figures_dir = Path(figures_dir)

    summary_path = tables_dir / "backtest_summary.parquet"
    if not summary_path.is_file():
        raise FileNotFoundError(f"backtest_summary.parquet not found at {summary_path!s}.")

    df = pd.read_parquet(summary_path)
    df = df.loc[~df["model"].isin(EXCLUDED_MODELS)]

    per_model = df.groupby("model", as_index=True)[
        ["sharpe_gross", "sharpe_net_ibkr", "sharpe_net_flat"]
    ].mean()

    models = [m for m in _MODEL_ORDER if m in per_model.index]

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "cost_drag.pdf"

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    n_groups = len(models)
    n_regimes = len(_REGIMES)
    bar_width = 0.24
    group_centers = np.arange(n_groups, dtype=float)
    offsets = (np.arange(n_regimes) - (n_regimes - 1) / 2.0) * bar_width

    # Three lightness levels per model: full saturation (gross), 30% lighter
    # (IBKR net), 60% lighter (flat net). Within each cluster the eye reads
    # darker = larger Sharpe (matching the typical gross >= net pattern).
    lightness_steps = [0.0, 0.30, 0.55]

    # Plot one legend handle per regime using a neutral grey, so the legend
    # explains the SHADE convention rather than naming five colours.
    legend_handles = []
    legend_labels = []

    for r_idx, (col, label) in enumerate(_REGIMES):
        values = np.array([per_model.loc[m, col] for m in models], dtype=float)
        positions = group_centers + offsets[r_idx]

        for m_idx, m in enumerate(models):
            base = MODEL_COLORS.get(m, "#444444")
            shaded = _lighten(base, lightness_steps[r_idx])
            ax.bar(
                positions[m_idx],
                values[m_idx],
                width=bar_width,
                color=shaded,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )

        # Only the gross bar gets value annotation, to reduce chart-junk.
        if r_idx == 0:
            for m_idx, value in enumerate(values):
                if not np.isfinite(value):
                    continue
                ax.annotate(
                    f"{value:.2f}",
                    xy=(positions[m_idx], value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#222222",
                )

        # Build a neutral-grey legend swatch for this regime.
        legend_handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=_lighten("#555555", lightness_steps[r_idx]),
                edgecolor="white",
                linewidth=0.5,
            )
        )
        legend_labels.append(label)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("Sharpe (annualized)")

    # Tighten y-limits to data with 15% headroom for the top annotations.
    all_values = per_model.values.flatten()
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size:
        ymax = float(all_values.max())
        ymin = min(0.0, float(all_values.min()))
        ax.set_ylim(ymin, ymax * 1.18)

    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5, zorder=1)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.4)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize=8,
        frameon=False,
        handlelength=1.0,
        handleheight=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)

    return out_path
