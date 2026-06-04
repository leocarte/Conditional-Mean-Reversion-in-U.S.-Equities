"""Portfolio performance metrics for monthly long-short returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min())


def summarize_monthly_returns(
    returns: pd.Series, *, periods_per_year: int = 12
) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        return {
            k: np.nan
            for k in [
                "ann_return",
                "ann_vol",
                "sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
                "hit_rate",
            ]
        }
    ann_return = float(r.mean() * periods_per_year)
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))
    downside = r[r < 0].std(ddof=1) * np.sqrt(periods_per_year)
    mdd = max_drawdown(r)
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol > 0 else np.nan,
        "sortino": ann_return / float(downside) if downside and downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": ann_return / abs(mdd) if mdd < 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
    }


def summarize_backtest(df: pd.DataFrame, return_col: str = "net_return") -> dict[str, float]:
    out = summarize_monthly_returns(df[return_col])
    if "turnover" in df.columns:
        out["avg_turnover"] = float(df["turnover"].mean())
    if "gross_return" in df.columns and return_col != "gross_return":
        gross = summarize_monthly_returns(df["gross_return"])
        out["gross_sharpe"] = gross["sharpe"]
    return out
