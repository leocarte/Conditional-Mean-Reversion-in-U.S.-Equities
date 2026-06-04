"""Performance metrics for the long-short backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, freq: int = 12) -> float:
    """Annualised Sharpe of an excess-return series (sqrt(freq) scaling)."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    std = r.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(np.sqrt(freq) * r.mean() / std)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown on the cumulative wealth path (returns a non-positive decimal)."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max
    return float(dd.min())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Per-period sum of absolute weight changes; first row uses |w_0| (inception trade)."""
    if weights.empty:
        return pd.Series(dtype=float)
    dw = weights.diff()
    dw.iloc[0] = weights.iloc[0]
    return dw.abs().sum(axis=1)


def cumulative_return(returns: pd.Series) -> pd.Series:
    """Cumulative compounded return path (level above $1 start, minus 1)."""
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def newey_west_tstat(returns: pd.Series, lags: int | None = None) -> float:
    """Newey-West HAC t-stat on the mean. Default lag = floor(4*(T/100)^(2/9))."""
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, int(lags))

    # Bartlett-kernel HAC variance of the mean; manual to avoid statsmodels dep.
    rbar = r.mean()
    u = r - rbar
    gamma0 = np.dot(u, u) / n
    gamma_sum = 0.0
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1.0)
        gamma_k = np.dot(u[k:], u[:-k]) / n
        gamma_sum += 2.0 * weight * gamma_k
    var_mean = (gamma0 + gamma_sum) / n
    if var_mean <= 0 or np.isnan(var_mean):
        return float("nan")
    return float(rbar / np.sqrt(var_mean))


def summary_table(
    gross_returns: pd.Series,
    net_returns_dict: dict[str, pd.Series],
    turnover_series: pd.Series,
    weights: pd.DataFrame,
    freq: int = 12,
) -> pd.DataFrame:
    """One-row summary across gross and net-of-cost specifications."""
    r = gross_returns.dropna()
    cols: dict[str, float] = {
        "ann_return": float(r.mean() * freq) if not r.empty else float("nan"),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(freq)) if len(r) > 1 else float("nan"),
        "sharpe_gross": sharpe(r, freq=freq),
    }
    cols["sharpe_net_ibkr"] = sharpe(
        net_returns_dict.get("ibkr", pd.Series(dtype=float)), freq=freq
    )
    cols["sharpe_net_flat"] = sharpe(
        net_returns_dict.get("flat", pd.Series(dtype=float)), freq=freq
    )
    for key, series in net_returns_dict.items():
        if key in {"ibkr", "flat"}:
            continue
        cols[f"sharpe_net_{key}"] = sharpe(series, freq=freq)

    cols["max_dd"] = max_drawdown(r)
    cols["turnover_mean"] = (
        float(turnover_series.dropna().mean()) if not turnover_series.empty else float("nan")
    )
    cols["gross_notional_mean"] = (
        float(weights.abs().sum(axis=1).mean()) if not weights.empty else float("nan")
    )
    cols["nw_tstat"] = newey_west_tstat(r)
    return pd.DataFrame([cols])
