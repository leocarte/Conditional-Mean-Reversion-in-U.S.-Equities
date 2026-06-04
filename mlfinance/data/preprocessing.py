"""Cross-sectional preprocessing transforms (strictly within-month)."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cross_sectional_rank(
    df: pd.DataFrame,
    col: str,
    date_col: str = "date",
) -> pd.Series:
    """Rank ``col`` to [-1, 1] per date (Gu-Kelly-Xiu 2020 convention).

    NaNs in ``col`` propagate to NaN ranks so the missingness pattern is preserved.
    """
    if col not in df.columns:
        raise KeyError(f"cross_sectional_rank: column '{col}' not found.")
    if date_col not in df.columns:
        raise KeyError(f"cross_sectional_rank: column '{date_col}' not found.")

    pct = df.groupby(date_col, observed=True)[col].rank(pct=True, method="average")
    return (pct * 2.0 - 1.0).astype("float64")


def impute_missing_with_indicator(
    df: pd.DataFrame,
    cols: Iterable[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Median-impute each column per date and append a ``{col}_is_missing`` flag.

    Imputation value depends only on the cross-section at that date, never on
    other dates. Where a whole cross-section is NaN the median falls back to 0
    (Gu-Kelly-Xiu 2020 p. 2245).
    """
    if date_col not in df.columns:
        raise KeyError(f"impute_missing_with_indicator: column '{date_col}' not found.")

    cols_list = list(cols)
    missing = {c for c in cols_list if c not in df.columns}
    if missing:
        raise KeyError(
            f"impute_missing_with_indicator: column(s) {sorted(missing)} not in df."
        )

    out = df.copy()

    for c in cols_list:
        out[f"{c}_is_missing"] = out[c].isna().astype("int8")

    medians = out.groupby(date_col, observed=True)[cols_list].transform("median")

    medians = medians.fillna(0.0)
    out[cols_list] = out[cols_list].fillna(medians)

    return out


def winsorize_returns(
    df: pd.DataFrame,
    col: str = "ret_exc",
    lower: float = 0.005,
    upper: float = 0.995,
    date_col: str = "date",
) -> pd.DataFrame:
    """Clip ``col`` to per-month [lower, upper] quantiles (regulation: cap, don't drop)."""
    if col not in df.columns:
        raise KeyError(f"winsorize_returns: column '{col}' not found.")
    if date_col not in df.columns:
        raise KeyError(f"winsorize_returns: column '{date_col}' not found.")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"winsorize_returns: invalid quantiles lower={lower}, upper={upper}.")

    out = df.copy()
    grp = out.groupby(date_col, observed=True)[col]
    lo = grp.transform(lambda s: s.quantile(lower))
    hi = grp.transform(lambda s: s.quantile(upper))
    out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def dedupe_jkp_cz(
    jkp: pd.DataFrame,
    cz: pd.DataFrame,
    threshold: float = 0.95,
    date_col: str = "date",
) -> pd.DataFrame:
    """Outer-join JKP and Chen-Zimmerman wide panels; drop near-collinear factors.

    Builds connected components from the pairwise |corr| > threshold graph
    (union-find), then keeps the column with the longest non-null history
    per component (ties broken by name for determinism).
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"dedupe_jkp_cz: threshold must be in (0, 1); got {threshold}.")
    if date_col not in jkp.columns or date_col not in cz.columns:
        raise KeyError(f"dedupe_jkp_cz: both inputs must contain '{date_col}'.")

    merged = jkp.merge(cz, on=date_col, how="outer", suffixes=("_jkp", "_cz"))
    merged = merged.sort_values(date_col).reset_index(drop=True)

    factor_cols = [c for c in merged.columns if c != date_col]
    if len(factor_cols) <= 1:
        return merged

    corr = merged[factor_cols].corr().abs()

    np.fill_diagonal(corr.values, 0.0)
    adj = corr.values >= threshold

    n = len(factor_cols)
    parent = np.arange(n)

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    # O(p^2) over the correlation matrix only - no Python loop on the long panel.
    ii, jj = np.triu_indices(n, k=1)
    pairs = np.column_stack([ii[adj[ii, jj]], jj[adj[ii, jj]]])
    for i, j in pairs:
        _union(int(i), int(j))

    clusters: dict[int, list[str]] = {}
    for idx, name in enumerate(factor_cols):
        clusters.setdefault(_find(idx), []).append(name)

    non_null_counts = merged[factor_cols].notna().sum()
    keep: list[str] = []
    dropped_log: list[tuple[str, list[str]]] = []
    for root, members in clusters.items():
        if len(members) == 1:
            keep.append(members[0])
            continue
        ranked = sorted(members, key=lambda m: (-int(non_null_counts[m]), m))
        winner = ranked[0]
        keep.append(winner)
        dropped_log.append((winner, ranked[1:]))

    if dropped_log:
        for winner, losers in dropped_log:
            logger.info(
                "dedupe_jkp_cz: kept '%s'; dropped %d near-collinear factor(s): %s",
                winner,
                len(losers),
                losers,
            )

    keep = sorted(keep)
    out = merged[[date_col, *keep]].copy()
    return out
