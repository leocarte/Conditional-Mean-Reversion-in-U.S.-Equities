"""CRSP universe filters for the cost/universe robustness audit (benchmark bridge).

Each filter is applied only when its required column(s) are present in the input
frame; otherwise it is skipped and recorded in the returned report, never faked.
The provided Monthly CRSP parquet does not contain ``prc``, ``shrout``,
``shrcd``, or ``exchcd`` (see ``mlfinance/data/loaders.py``), so under the
project's provided-data constraint only the SIC-sector exclusion and the
optional per-month minimum-count guard are feasible today. The price,
share-code, exchange, and market-cap filters remain availability-gated and are
reported as skipped when those columns are absent.

All filters read time-``t`` (formation-date) columns only, so applying them to a
predictions frame keyed by the formation ``date`` introduces no look-ahead.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Optional filter -> the panel column(s) it requires. Drives availability gating
# and the schema-coverage audit.
FILTER_FIELD_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "sic_exclude": ("siccd",),
    "price_min": ("prc",),
    "shrcd_keep": ("shrcd",),
    "exchcd_keep": ("exchcd",),
    "market_cap_min": ("prc", "shrout"),
}


@dataclass(frozen=True)
class FilterReport:
    """Outcome of one filter: whether it ran, why, and its row-count effect."""

    name: str
    applied: bool
    reason: str
    rows_before: int
    rows_after: int


def field_availability(columns) -> dict[str, bool]:
    """For each optional filter, whether all its required columns are present."""
    cols = {str(c).lower() for c in columns}
    return {
        name: all(req in cols for req in reqs) for name, reqs in FILTER_FIELD_REQUIREMENTS.items()
    }


def _ranges_mask(values: pd.Series, ranges) -> pd.Series:
    """Boolean mask: numeric ``values`` within any inclusive ``[lo, hi]`` range."""
    numeric = pd.to_numeric(values, errors="coerce")
    mask = pd.Series(False, index=values.index)
    for lo, hi in ranges:
        mask |= numeric.between(float(lo), float(hi))
    return mask


def _apply_masked(
    out: pd.DataFrame,
    cols: set[str],
    reports: list[FilterReport],
    name: str,
    keep_mask: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    """Apply ``keep_mask`` if ``name``'s required columns exist; else skip + report."""
    missing = [c for c in FILTER_FIELD_REQUIREMENTS.get(name, ()) if c not in cols]
    before = len(out)
    if missing:
        reports.append(
            FilterReport(
                name, False, f"required column(s) {missing} absent; skipped", before, before
            )
        )
        return out
    kept = out.loc[keep_mask(out)].copy()
    reports.append(FilterReport(name, True, "applied", before, len(kept)))
    return kept


def apply_universe(
    frame: pd.DataFrame,
    spec: dict,
    *,
    date_col: str = "date",
    permno_col: str = "permno",
) -> tuple[pd.DataFrame, list[FilterReport]]:
    """Apply availability-gated universe filters; return ``(filtered, reports)``.

    ``spec`` keys (all optional):
      * ``sic_exclude`` -- inclusive ``[lo, hi]`` SIC ranges to DROP (needs ``siccd``)
      * ``price_min`` -- keep ``abs(prc) >= price_min`` (needs ``prc``)
      * ``shrcd_keep`` -- keep ``shrcd in {...}`` (needs ``shrcd``)
      * ``exchcd_keep`` -- keep ``exchcd in {...}`` (needs ``exchcd``)
      * ``market_cap_min`` -- keep ``abs(prc) * shrout >= market_cap_min`` (needs ``prc``, ``shrout``)
      * ``min_stocks_per_month`` -- drop dates with fewer than N rows (applied last)

    A filter whose required column is missing is skipped and reported, never faked,
    so this is safe on the current CRSP panel (only ``siccd`` present).
    """
    out = frame.reset_index(drop=True)
    cols = {str(c).lower() for c in out.columns}
    reports: list[FilterReport] = []

    if spec.get("sic_exclude"):
        sic_ranges = spec["sic_exclude"]
        out = _apply_masked(
            out, cols, reports, "sic_exclude", lambda d: ~_ranges_mask(d["siccd"], sic_ranges)
        )
    if spec.get("price_min") is not None:
        price_thr = float(spec["price_min"])
        out = _apply_masked(
            out,
            cols,
            reports,
            "price_min",
            lambda d: pd.to_numeric(d["prc"], errors="coerce").abs() >= price_thr,
        )
    if spec.get("shrcd_keep"):
        shrcd_set = {int(x) for x in spec["shrcd_keep"]}
        out = _apply_masked(
            out,
            cols,
            reports,
            "shrcd_keep",
            lambda d: pd.to_numeric(d["shrcd"], errors="coerce").isin(shrcd_set),
        )
    if spec.get("exchcd_keep"):
        exchcd_set = {int(x) for x in spec["exchcd_keep"]}
        out = _apply_masked(
            out,
            cols,
            reports,
            "exchcd_keep",
            lambda d: pd.to_numeric(d["exchcd"], errors="coerce").isin(exchcd_set),
        )
    if spec.get("market_cap_min") is not None:
        mktcap_thr = float(spec["market_cap_min"])
        out = _apply_masked(
            out,
            cols,
            reports,
            "market_cap_min",
            lambda d: (
                pd.to_numeric(d["prc"], errors="coerce").abs()
                * pd.to_numeric(d["shrout"], errors="coerce")
            )
            >= mktcap_thr,
        )

    min_n = spec.get("min_stocks_per_month")
    if min_n:
        before = len(out)
        counts = out.groupby(date_col)[permno_col].transform("count")
        out = out.loc[counts >= int(min_n)].copy()
        reports.append(FilterReport("min_stocks_per_month", True, "applied", before, len(out)))

    return out.reset_index(drop=True), reports
