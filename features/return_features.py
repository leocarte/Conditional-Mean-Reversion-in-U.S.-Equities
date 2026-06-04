"""CRSP return-based features for conditional mean reversion.

The core convention is: row (permno, date=t) contains features known at t
and labels for t+1. No function here should use future information in a
feature column.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _cumret(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    return float(np.prod(1.0 + x.to_numpy()) - 1.0)


def _rolling_log_cumret(s: pd.Series, *, window: int, min_count: int) -> pd.Series:
    """Vectorized ``prod(1 + r) - 1`` over a rolling window.

    Computes the cumulative return per window via ``log1p + rolling.sum + expm1``
    so each window is a C-vectorized sum rather than a Python callback per
    window. To preserve the previous ``rolling.apply(_cumret)`` semantics, a
    returned value requires both enough observations and no NaNs inside the
    actual rolling window. This matters after monthly-calendar expansion, where
    inserted missing months should make momentum unavailable.

    Numerically equivalent to ``_cumret`` to within floating-point precision
    (typically ~1e-15 relative error); ``log1p(-1) = -inf`` and ``expm1(-inf) =
    -1.0`` keep total-wipeout returns identical.

    The ``np.errstate(divide='ignore')`` wrap silences the expected
    ``RuntimeWarning: divide by zero encountered in log1p`` that numpy emits on
    every ``MthRet == -1.0`` (bankruptcy / total-loss CRSP rows). The -inf
    propagates correctly through the rolling sum and ``expm1`` to produce the
    mathematically correct cumret of -1.0, so the warning is pure noise.
    """
    with np.errstate(divide="ignore"):
        log_filled = np.log1p(s).fillna(0.0)
    valid = s.notna().astype("int64")
    present = pd.Series(1, index=s.index, dtype="int64")

    sum_log = log_filled.rolling(window, min_periods=1).sum()
    cnt_valid = valid.rolling(window, min_periods=1).sum()
    window_len = present.rolling(window, min_periods=1).sum()

    valid_window = (cnt_valid >= min_count) & (cnt_valid == window_len)
    return pd.Series(
        np.where(valid_window, np.expm1(sum_log), np.nan),
        index=s.index,
    )


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window, min_periods=max(6, window // 2)).cov(x)
    var = x.rolling(window, min_periods=max(6, window // 2)).var()
    return cov / var.replace(0.0, np.nan)


def _collapse_baseline_duplicates(
    df: pd.DataFrame,
    *,
    permno_col: str,
    date_col: str,
    ret_col: str,
    market_col: str,
) -> pd.DataFrame:
    """Enforce one row per ``(permno, date)`` before calendar expansion.

    Duplicate groups are only checked on the baseline economic fields:
    stock return and market return. Identifier or classification differences
    are tolerated; the first row after stable sorting is retained.
    """
    key_cols = [permno_col, date_col]
    duplicate_mask = df.duplicated(subset=key_cols, keep=False)
    if not duplicate_mask.any():
        return df

    compare_cols = [ret_col]
    if market_col in df.columns:
        compare_cols.append(market_col)

    duplicate_rows = df.loc[duplicate_mask, key_cols + compare_cols].copy()
    nunique = duplicate_rows.groupby(key_cols, sort=True)[compare_cols].nunique(dropna=False)
    conflicting_groups = nunique.gt(1).any(axis=1)
    if conflicting_groups.any():
        offending = nunique.loc[conflicting_groups]
        sample_pairs = [
            (int(permno), pd.Timestamp(date).date().isoformat())
            for permno, date in offending.index[:5]
        ]
        raise ValueError(
            "Conflicting duplicate (permno, date) groups found in the baseline CRSP panel: "
            f"{int(conflicting_groups.sum())} groups. "
            f"Sample offending pairs: {sample_pairs}"
        )

    collapsed = (
        df.sort_values(key_cols, kind="mergesort")
        .drop_duplicates(subset=key_cols, keep="first")
        .reset_index(drop=True)
    )
    n_duplicate_groups = int(duplicate_rows.groupby(key_cols, sort=False).ngroups)
    logger.info(
        "Collapsed %d duplicate (permno, date) groups in the baseline CRSP panel.",
        n_duplicate_groups,
    )
    return collapsed


def _expand_to_complete_monthly_calendar(
    df: pd.DataFrame,
    *,
    permno_col: str,
    date_col: str,
    market_col: str,
) -> pd.DataFrame:
    """Expand each permno to a full month-end calendar before time-series features.

    This prevents lag and rolling features from treating non-contiguous observed
    CRSP rows as consecutive months. Inserted rows are marked and dropped again
    after all time-series features and targets are computed.

    Vectorized via a (permno, date) skeleton + single left merge -- avoids the
    per-permno DataFrame reindex/reset loop that the previous implementation
    paid (34k+ iterations on the full CRSP universe).
    """
    market_by_date = (
        df[[date_col, market_col]]
        .sort_values(date_col)
        .drop_duplicates(date_col)
        .set_index(date_col)[market_col]
    )

    ranges = df.groupby(permno_col, sort=False)[date_col].agg(["min", "max"])
    if ranges.empty:
        # Same shape as a normal output: original columns + _is_observed.
        empty = df.iloc[0:0].copy()
        empty["_is_observed"] = pd.Series([], dtype=bool)
        return empty

    parts = [
        pd.DataFrame(
            {
                permno_col: permno,
                date_col: pd.date_range(row["min"], row["max"], freq="ME"),
            }
        )
        for permno, row in ranges.iterrows()
    ]
    skeleton = pd.concat(parts, ignore_index=True)

    # Left-merge df onto the (permno, date) skeleton; ``indicator`` distinguishes
    # observed rows from calendar-inserted ones. ``market_col`` is dropped from
    # the right side and re-mapped from ``market_by_date`` so inserted rows get
    # the market return for their date too (matching the original behavior).
    expanded = skeleton.merge(
        df.drop(columns=[market_col], errors="ignore"),
        on=[permno_col, date_col],
        how="left",
        indicator="_obs_marker",
    )
    expanded["_is_observed"] = expanded["_obs_marker"] == "both"
    expanded = expanded.drop(columns=["_obs_marker"])
    expanded[market_col] = expanded[date_col].map(market_by_date)
    return expanded


def add_return_features(
    df: pd.DataFrame,
    *,
    permno_col: str = "permno",
    date_col: str = "date",
    ret_col: str = "ret",
    market_col: str = "sprtrn",
) -> pd.DataFrame:
    """Add lagged return, reversal, momentum, volatility, and target columns.

    Parameters
    ----------
    df:
        Monthly CRSP-like panel with one row per security-month.
    permno_col, date_col, ret_col, market_col:
        Column names for security id, month-end date, stock return, and market return.

    Returns
    -------
    pandas.DataFrame
        Input rows plus feature columns and labels:
        ``target_ret_fwd_1m`` and ``target_excess_ret_fwd_1m``.
    """
    _require_columns(df, [permno_col, date_col, ret_col, market_col])
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]) + pd.offsets.MonthEnd(0)
    out = out.sort_values([permno_col, date_col]).reset_index(drop=True)
    out = _collapse_baseline_duplicates(
        out,
        permno_col=permno_col,
        date_col=date_col,
        ret_col=ret_col,
        market_col=market_col,
    )
    out = _expand_to_complete_monthly_calendar(
        out,
        permno_col=permno_col,
        date_col=date_col,
        market_col=market_col,
    )
    out = out.sort_values([permno_col, date_col]).reset_index(drop=True)

    g = out.groupby(permno_col, group_keys=False)
    out["ret_1m"] = g[ret_col].shift(0)
    out["ret_2m"] = g[ret_col].shift(1)
    out["ret_3m"] = g[ret_col].shift(2)

    # Momentum features follow the project convention on row t:
    # - mom_2_12 uses returns t-12..t-2, so it excludes the two most recent months.
    # - mom_1_12 uses returns t-12..t, so it includes the current month.
    # Vectorized via log1p+rolling.sum+expm1 inside ``_rolling_log_cumret`` --
    # the previous ``rolling.apply(_cumret)`` paid one Python callback per
    # window (~40M+ on the full CRSP universe).
    out["mom_2_12"] = g[ret_col].transform(
        lambda s: _rolling_log_cumret(s.shift(2), window=11, min_count=8)
    )
    out["mom_1_12"] = g[ret_col].transform(lambda s: _rolling_log_cumret(s, window=13, min_count=8))

    for w in (3, 6, 12):
        out[f"vol_{w}m"] = g[ret_col].transform(
            lambda s, w=w: s.rolling(w, min_periods=max(3, w // 2)).std()
        )

    def trailing_drawdown(s: pd.Series) -> pd.Series:
        # ``raw=True`` passes the rolling window as a 1-D numpy array instead of
        # wrapping it in a fresh Series per call -- 5-10x faster, identical
        # output, since the body uses only numpy ops.
        def window_drawdown(window: np.ndarray) -> float:
            clean = window[~np.isnan(window)]
            if clean.size < 6:
                return np.nan
            wealth = np.cumprod(1.0 + clean)
            return float((wealth / np.maximum.accumulate(wealth) - 1.0).min())

        return s.rolling(12, min_periods=6).apply(window_drawdown, raw=True)

    out["drawdown_12m"] = g[ret_col].transform(trailing_drawdown)

    # Align market return per date and estimate rolling beta/idiosyncratic volatility per stock.
    market_by_date = (
        out[[date_col, market_col]].drop_duplicates(date_col).set_index(date_col)[market_col]
    )
    out["_mkt_for_beta"] = out[date_col].map(market_by_date)

    betas = []
    idio_vols = []
    for _, sub in out[[permno_col, date_col, ret_col, "_mkt_for_beta"]].groupby(
        permno_col, sort=False
    ):
        beta = _rolling_beta(sub[ret_col], sub["_mkt_for_beta"], 12)
        resid = sub[ret_col] - beta * sub["_mkt_for_beta"]
        idio = resid.rolling(12, min_periods=6).std()
        betas.append(beta)
        idio_vols.append(idio)
    out["beta_12m"] = pd.concat(betas).sort_index()
    out["idio_vol_12m"] = pd.concat(idio_vols).sort_index()
    out = out.drop(columns=["_mkt_for_beta"])

    # Cross-sectional rank features. Use pct ranks in [0,1], then scale to [-1,1].
    for c in ["ret_1m", "mom_2_12", "vol_12m", "beta_12m"]:
        rank = out.groupby(date_col)[c].rank(pct=True)
        out[f"{c}_rank"] = 2.0 * rank - 1.0

    out["target_ret_fwd_1m"] = g[ret_col].shift(-1)
    next_mkt = (out[date_col] + pd.offsets.MonthEnd(1)).map(market_by_date)
    out["target_excess_ret_fwd_1m"] = out["target_ret_fwd_1m"] - next_mkt
    out = out.loc[out["_is_observed"]].drop(columns=["_is_observed"]).reset_index(drop=True)
    return out


def add_return_bins(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    target_col: str = "target_ret_fwd_1m",
    label_col: str = "target_bin",
) -> pd.DataFrame:
    """Add three cross-sectional target bins: bottom 30%, middle 40%, top 30%."""
    _require_columns(df, [date_col, target_col])
    out = df.copy()
    pct = out.groupby(date_col)[target_col].rank(pct=True)
    out[label_col] = np.select([pct <= 0.30, pct >= 0.70], [0, 2], default=1)
    out.loc[pct.isna(), label_col] = np.nan
    return out
