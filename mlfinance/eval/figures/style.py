"""Shared matplotlib style for FIN-407 final report figures.

Defaults tuned to read well in a LaTeX report compiled with Computer
Modern, without requiring a LaTeX installation in the container.

Public API
----------
apply_style()
    Push the report rcParams into ``matplotlib.rcParams``. Idempotent.
report_style()
    Context manager that applies the style for its body only.
MODEL_COLORS : dict[str, str]
    Okabe-Ito / Wong-2011 colorblind-safe palette, one fixed hex per model.
MODEL_LABELS : dict[str, str]
    Display labels for legends and axis ticks.
EXCLUDED_MODELS : set[str]
    Models filtered out of every figure (MLP, due to fold 1-3 divergence).

Design choices
--------------
Palette: Okabe-Ito 5-color subset (Wong 2011, Nat. Methods 8:441). Each
hex has a distinct L* luminance so the figure remains discriminable
under deuteranopia / protanopia / tritanopia and survives greyscale
printing. Same palette as Seaborn's "colorblind" cycle.

Typography: Computer Modern via matplotlib's bundled ``cmr10`` glyphs
(``mathtext.fontset='cm'``). No LaTeX install needed. Latin Modern Roman
and DejaVu Serif sit as fallbacks.

Axes: minimal-ink Tufte/Cleveland defaults. Top and right spines hidden,
axes.edgecolor #444 instead of pure black, axes.linewidth 0.6, grid
alpha 0.25, outward ticks 3pt. Data lines at 1.5pt sit cleanly above
the 0.6pt axes when the figure is shrunk to \\linewidth.

Output: 150 dpi screen / 300 dpi save. ``pdf.fonttype=42`` ships
TrueType so the figure passes Elsevier / Springer Nature submission
checks. ``savefig.bbox='tight'`` with a 0.02 inch pad avoids cropping
descenders at the bottom of the figure.

Sources
-------
- Wong, B. (2011). "Color blindness." Nature Methods 8:441.
- Okabe, M. and Ito, K. (2008). Color universal design (Okabe-Ito palette).
- Tufte, E. (1983). "The Visual Display of Quantitative Information."
- Cleveland, W. (1985). "The Elements of Graphing Data."
- SciencePlots (Garrett 2021). github.com/garrettj403/SciencePlots.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "apply_style",
    "report_style",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "EXCLUDED_MODELS",
]


# ---------------------------------------------------------------------------
# Palette and labels (Okabe-Ito / Wong 2011, colorblind-safe)
# ---------------------------------------------------------------------------
# Hex codes assigned with monotonically separated L* values so the palette
# stays discriminable in greyscale print.
#   #0072B2 Blue          L* ~ 45
#   #D55E00 Vermillion    L* ~ 51
#   #009E73 Bluish green  L* ~ 58
#   #CC79A7 Reddish purple L*~ 60
#   #E69F00 Orange        L* ~ 70   (reserved for MLP / 5th model)
#   #000000 Black         L* ~ 0    (stacking ensemble; reads as headline)
MODEL_COLORS: dict[str, str] = {
    "ridge": "#0072B2",  # Wong blue
    "lasso": "#D55E00",  # Wong vermillion
    "ft_transformer": "#009E73",  # Wong bluish-green
    "xgb": "#CC79A7",  # Wong reddish-purple
    "mlp": "#E69F00",  # Wong orange (kept for completeness)
    "stacking": "#000000",  # Okabe-Ito black (headline ensemble)
}

# Display labels for legends, table headers, and axis annotations.
MODEL_LABELS: dict[str, str] = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "mlp": "MLP (NN3)",
    "ft_transformer": "FT-Transformer",
    "xgb": "XGBoost",
    "stacking": "Stacking",
}

# Models excluded from every figure. MLP is dropped because folds 1-3
# diverged (optimization failure on the smaller training sets); the
# remaining folds 4-5 aggregate is not strictly comparable to the
# all-fold aggregates of the other models. Tables still include MLP per
# (fold, seed).
EXCLUDED_MODELS: set[str] = {"mlp"}


# ---------------------------------------------------------------------------
# rcParams - report defaults (no LaTeX dependency)
# ---------------------------------------------------------------------------
_REPORT_clusterARAMS: dict[str, object] = {
    # --- Output resolution ---------------------------------------------
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # TrueType, required by most journals
    "ps.fonttype": 42,
    # --- Typography (Computer Modern via mathtext, no LaTeX needed) ----
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    # Use ASCII minus instead of U+2212 (the substitute glyph is missing
    # in CMU Serif / Computer Modern Roman); harmless visual change.
    "axes.unicode_minus": False,
    # --- Font sizes (tuned for two-column \linewidth ~3.3 in) ----------
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # --- Axes geometry (Tufte + Cleveland minimal-ink) -----------------
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.6,
    "axes.labelpad": 3.0,
    "axes.titlepad": 6.0,
    # --- Grid (faint, behind data) -------------------------------------
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#B0B0B0",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    # --- Ticks (outward, short, Tufte) ---------------------------------
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    # --- Lines / markers -----------------------------------------------
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.6,
    # --- Legend (frameless, Tufte) -------------------------------------
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderpad": 0.3,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.3,
    # --- Color cycle (Okabe-Ito 5-class subset) ------------------------
    "axes.prop_cycle": mpl.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]),
}


def apply_style() -> None:
    """Push the report style into matplotlib.rcParams. Safe to call repeatedly."""
    mpl.rcParams.update(_REPORT_clusterARAMS)


@contextmanager
def report_style() -> Iterator[None]:
    """Context manager that applies the report style for its body only.

    Uses ``plt.rc_context`` so the rcParams are restored on exit.
    """
    with plt.rc_context(_REPORT_clusterARAMS):
        yield
