"""Identifier linkers for the active CRSP/Compustat project surface."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def link_compustat_to_crsp_by_cusip(
    compustat: pd.DataFrame,
    crsp_monthly: pd.DataFrame,
) -> pd.DataFrame:
    """Build a deduplicated ``(gvkey, permno)`` mapping via CUSIP-8 overlap.

    A handful of firms re-CUSIP over time so a single gvkey can map to several
    permnos; we retain all pairs and let the downstream ``merge_asof`` pick the
    freshest at each date. Returns columns ``gvkey, permno, cusip8``.
    """
    required_left = {"gvkey", "cusip8"}
    if not required_left.issubset(compustat.columns):
        raise KeyError(
            f"link_compustat_to_crsp_by_cusip: compustat needs {sorted(required_left)}."
        )
    required_right = {"permno", "cusip8"}
    if not required_right.issubset(crsp_monthly.columns):
        raise KeyError(
            f"link_compustat_to_crsp_by_cusip: crsp_monthly needs {sorted(required_right)}."
        )

    left = (
        compustat[["gvkey", "cusip8"]]
        .dropna(subset=["gvkey", "cusip8"])
        .drop_duplicates()
        .copy()
    )
    right = (
        crsp_monthly[["permno", "cusip8"]]
        .dropna(subset=["permno", "cusip8"])
        .drop_duplicates()
        .copy()
    )
    if left.empty or right.empty:
        logger.warning(
            "link_compustat_to_crsp_by_cusip: empty input on at least one side - returning empty mapping."
        )
        return pd.DataFrame({
            "gvkey": pd.Series(dtype="string"),
            "permno": pd.Series(dtype="int64"),
            "cusip8": pd.Series(dtype="string"),
        })

    merged = left.merge(right, on="cusip8", how="inner")

    # groupby rather than drop_duplicates leaves room to add a frequency /
    # counter column later.
    out = (
        merged.groupby(["gvkey", "permno"], as_index=False, observed=True)["cusip8"]
        .first()
    )
    out["permno"] = out["permno"].astype("int64")
    out["gvkey"] = out["gvkey"].astype("string")
    out["cusip8"] = out["cusip8"].astype("string")
    out = out.sort_values(["gvkey", "permno"]).reset_index(drop=True)
    logger.info(
        "link_compustat_to_crsp_by_cusip: %d (gvkey, permno) pairs (%d unique gvkeys, %d unique permnos).",
        len(out),
        out["gvkey"].nunique(),
        out["permno"].nunique(),
    )
    return out
