"""Decile-sort portfolio construction (EW only; VW unavailable without mktcap)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def decile_sort(predictions: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """Assign each (date, permno) row to a decile by ``y_hat`` within date.

    Decile 1 is the lowest predicted return, ``n_deciles`` the highest. Ties
    are broken by ``rank(method='first')``. NaN ``y_hat`` rows are dropped.
    """
    if predictions.empty:
        return predictions.assign(decile=pd.Series([], dtype="int64"))

    df = predictions.dropna(subset=["y_hat"]).copy()
    rank = df.groupby("date")["y_hat"].rank(method="first", ascending=True)
    count = df.groupby("date")["y_hat"].transform("count")
    decile = np.ceil(rank.to_numpy() / count.to_numpy() * n_deciles).astype(np.int64)
    decile = np.clip(decile, 1, n_deciles)
    df["decile"] = decile
    return df


def _ew_weights_for_leg(panel: pd.DataFrame, sign: float) -> pd.Series:
    counts = panel.groupby("date")["permno"].transform("count")
    return sign / counts.astype(float)


def _vw_weights_for_leg(panel: pd.DataFrame, sign: float) -> pd.Series:
    """Value-weight by mktcap; unreachable today (Drive CRSP lacks mktcap)."""
    totals = panel.groupby("date")["mktcap"].transform("sum")
    return sign * panel["mktcap"].to_numpy() / totals.to_numpy()


def long_short_weights(
    decile_panel: pd.DataFrame,
    scheme: str = "EW",
    top_decile: int = 10,
    bottom_decile: int = 1,
) -> pd.DataFrame:
    """Build a wide (date, permno) long-short weight panel. Only ``scheme='EW'``.

    VW variants raise ``NotImplementedError`` because the provided CRSP file
    ships without PRC and SHROUT, so market cap cannot be reconstructed.
    """
    if decile_panel.empty:
        return pd.DataFrame()

    if scheme in ("VW", "VW_ex_micro20"):
        raise NotImplementedError(
            "VW schemes require market cap, which is not in the provided CRSP "
            "file. Use scheme='EW'."
        )

    valid = {"EW"}
    if scheme not in valid:
        raise ValueError(f"Unknown scheme {scheme!r}. Must be one of {valid}.")

    df = decile_panel.copy()

    long_panel = df[df["decile"] == top_decile].copy()
    short_panel = df[df["decile"] == bottom_decile].copy()

    # Warn rather than crash on degenerate sorts: if every row on a date
    # got bucketed into the same decile, the top or bottom leg will be empty
    # for that date. We warn (one line per affected date) so users can grep,
    # then proceed; the pivot below leaves missing cells as 0 weight, which
    # the downstream PnL treats as a zero return for that leg on that date.
    if not df.empty:
        all_dates = df["date"].drop_duplicates()
        long_dates = set(long_panel["date"].unique())
        short_dates = set(short_panel["date"].unique())
        for d in sorted(set(all_dates) - long_dates):
            logger.warning(
                "Empty long decile (top=%d) on %s; using 0 return for that date.",
                top_decile,
                pd.Timestamp(d).date().isoformat(),
            )
        for d in sorted(set(all_dates) - short_dates):
            logger.warning(
                "Empty short decile (bottom=%d) on %s; using 0 return for that date.",
                bottom_decile,
                pd.Timestamp(d).date().isoformat(),
            )

    long_panel["weight"] = _ew_weights_for_leg(long_panel, +1.0)
    short_panel["weight"] = _ew_weights_for_leg(short_panel, -1.0)

    weights_long = pd.concat([long_panel, short_panel], ignore_index=True)
    wide = (
        weights_long.pivot_table(index="date", columns="permno", values="weight", aggfunc="sum")
        .fillna(0.0)
        .sort_index()
    )
    return wide


def compute_realized_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Realised portfolio return ``r_{p,t} = sum_i w_{i,t} * r_{i,t}``.

    DATE ALIGNMENT CONVENTION: callers must pass weights and returns on the
    SAME REBALANCE-date axis t, where ``weights[t, i]`` is the position formed
    at rebalance event t (from a prediction made with information up to t),
    and ``returns[t, i]`` is the realized return of the period attributed to
    that rebalance (= R_{t+1} when the upstream pipeline stores realized_ret
    as the next-month return). No internal weights.shift(1) is applied here;
    re-introducing it together with the upstream pre-shift in
    ``_build_panels_from_predictions`` produces a silent off-by-one return.
    """
    if weights.empty or returns.empty:
        return pd.Series(dtype=float)

    cols = weights.columns.intersection(returns.columns)
    aligned_w = weights[cols].reindex(returns.index).fillna(0.0)
    aligned_r = returns[cols].fillna(0.0)
    return (aligned_w * aligned_r).sum(axis=1)
