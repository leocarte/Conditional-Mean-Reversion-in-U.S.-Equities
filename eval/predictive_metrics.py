"""Predictive metrics for cross-sectional return forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlfinance.models.base import r_squared_oos


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(y_true - y_pred)))


def information_coefficient(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    y_col: str = "target_ret_fwd_1m",
    pred_col: str = "y_hat",
    method: str = "spearman",
) -> pd.Series:
    """Monthly cross-sectional IC time series."""
    grouped = df.groupby(date_col)[[y_col, pred_col]]
    return grouped.apply(
        lambda g: g.corr(method=method).iloc[0, 1],
        include_groups=False,
    )


def decile_spread(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    y_col: str = "target_ret_fwd_1m",
    pred_col: str = "y_hat",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Average realized return by prediction decile and top-minus-bottom spread."""
    x = df.dropna(subset=[y_col, pred_col]).copy()
    rank = x.groupby(date_col)[pred_col].rank(method="first", ascending=True)
    count = x.groupby(date_col)[pred_col].transform("count")
    x["decile"] = np.ceil(rank / count * n_deciles).clip(1, n_deciles).astype(int)
    by_decile = x.groupby("decile")[y_col].mean().rename("mean_realized_return").reset_index()
    spread_ts = x.groupby([date_col, "decile"])[y_col].mean().unstack("decile")
    if 1 in spread_ts.columns and n_deciles in spread_ts.columns:
        spread = spread_ts[n_deciles] - spread_ts[1]
    else:
        spread = pd.Series(dtype=float)
    by_decile.attrs["top_minus_bottom_mean"] = float(spread.mean()) if not spread.empty else np.nan
    return by_decile


def summarize_predictions(
    df: pd.DataFrame, y_col: str, pred_col: str, date_col: str = "date"
) -> dict[str, float]:
    clean = df.dropna(subset=[y_col, pred_col])
    ic = information_coefficient(
        clean, date_col=date_col, y_col=y_col, pred_col=pred_col, method="spearman"
    )
    return {
        "rmse": rmse(clean[y_col], clean[pred_col]),
        "mae": mae(clean[y_col], clean[pred_col]),
        "oos_r2": float(r_squared_oos(clean[y_col].to_numpy(), clean[pred_col].to_numpy())),
        "rank_ic_mean": float(ic.mean()),
        "rank_ic_tstat": (
            float(ic.mean() / (ic.std(ddof=1) / np.sqrt(ic.count())))
            if ic.count() > 1 and ic.std(ddof=1) > 0
            else np.nan
        ),
    }
