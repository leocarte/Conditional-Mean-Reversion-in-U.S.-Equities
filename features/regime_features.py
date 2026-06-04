"""Small lagged market-regime feature block.

Keep this deliberately small. The purpose is to test conditional mean reversion,
not to create a large factor-mining exercise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cumret(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    return float(np.prod(1.0 + x.to_numpy()) - 1.0)


def add_regime_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ret_col: str = "ret",
    market_col: str = "sprtrn",
) -> pd.DataFrame:
    """Attach lagged market and cross-sectional dispersion features by month."""
    if date_col not in df.columns or market_col not in df.columns or ret_col not in df.columns:
        raise KeyError(f"Need {date_col!r}, {ret_col!r}, and {market_col!r} columns.")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]) + pd.offsets.MonthEnd(0)

    monthly = (
        out.groupby(date_col)
        .agg(
            mkt_ret=(market_col, "first"),
            cross_sectional_dispersion_1m=(ret_col, "std"),
        )
        .sort_index()
    )
    monthly["mkt_ret_12m"] = monthly["mkt_ret"].rolling(12, min_periods=8).apply(_cumret, raw=False)
    monthly["mkt_vol_12m"] = monthly["mkt_ret"].rolling(12, min_periods=8).std()

    wealth = (1.0 + monthly["mkt_ret"].fillna(0.0)).cumprod()
    monthly["mkt_drawdown_12m"] = wealth / wealth.rolling(12, min_periods=8).max() - 1.0

    # Rolling median uses only current/past values; row at t is known after month t closes.
    med_vol = monthly["mkt_vol_12m"].rolling(120, min_periods=24).median()
    monthly["high_vol_regime"] = (monthly["mkt_vol_12m"] > med_vol).astype(float)
    monthly["bear_market_regime"] = (monthly["mkt_ret_12m"] < 0.0).astype(float)

    regime_cols = [
        "mkt_ret_12m",
        "mkt_vol_12m",
        "mkt_drawdown_12m",
        "cross_sectional_dispersion_1m",
        "high_vol_regime",
        "bear_market_regime",
    ]
    return out.merge(monthly[regime_cols].reset_index(), on=date_col, how="left")
