"""HAC standard errors under three lag conventions (sqrt(T), NW1994, Andrews 1991).

Uses the Bartlett kernel ``w_j = 1 - j / (L + 1)`` matching statsmodels.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "newey_west_se",
    "newey_west_lag_sqrt_t",
    "newey_west_lag_nw1994",
    "andrews_1991_plugin_lag",
    "all_three_lags",
    "report_t_stats_3lags",
]


def newey_west_se(returns: np.ndarray, lag: int) -> tuple[float, float, float]:
    """Newey-West HAC standard error of the sample mean.

    Returns (mean, se, t_stat). t_stat is NaN when se == 0.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    T = r.size
    if T < 2:
        raise ValueError("Need at least 2 observations for HAC SE.")
    if lag < 0 or lag >= T:
        raise ValueError(f"lag must be in [0, T-1=({T - 1})], got {lag}.")

    mean = float(np.mean(r))
    e = r - mean

    gamma0 = float(np.dot(e, e) / T)
    var = gamma0
    for j in range(1, lag + 1):
        gamma_j = float(np.dot(e[j:], e[:-j]) / T)
        weight = 1.0 - j / (lag + 1)
        var += 2.0 * weight * gamma_j

    # Bartlett kernel guarantees var >= 0 in expectation but rounding can dip below.
    var = max(var, 0.0)
    se = float(np.sqrt(var / T))
    t_stat = mean / se if se > 0.0 else float("nan")
    return mean, se, t_stat


def newey_west_lag_sqrt_t(T: int) -> int:
    """Textbook ``floor(sqrt(T))`` lag."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    return int(np.sqrt(T))


def newey_west_lag_nw1994(T: int) -> int:
    """Newey-West (1994) automatic lag: ``floor(4 * (T/100)^(2/9))``."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}.")
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def andrews_1991_plugin_lag(returns: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in lag (Bartlett kernel).

    L = round(1.1447 * (4*rho^2*T / (1+rho^2)^2)^(1/3)), capped at T-1.
    Returns 0 when rho <= 0.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    T = r.size
    if T < 3:
        return 0
    e = r - r.mean()
    numer = float(np.dot(e[1:], e[:-1]))
    denom = float(np.dot(e[:-1], e[:-1]))
    if denom <= 0.0:
        return 0
    rho = numer / denom
    if rho <= 0.0:
        return 0
    rho = min(rho, 0.97)  # cap to avoid blow-up for near-unit-root series

    lag = 1.1447 * (4.0 * rho**2 * T / (1.0 + rho**2) ** 2) ** (1.0 / 3.0)
    return int(min(max(round(lag), 0), T - 1))


def all_three_lags(returns: np.ndarray) -> dict[str, int]:
    """Return ``{sqrt_t, nw1994, andrews1991}`` lag values for the series."""
    T = int(np.asarray(returns).size)
    return {
        "sqrt_t": newey_west_lag_sqrt_t(T),
        "nw1994": newey_west_lag_nw1994(T),
        "andrews1991": andrews_1991_plugin_lag(returns),
    }


def report_t_stats_3lags(returns: np.ndarray) -> dict:
    """Compute mean, NW SE, and t-stat under all three lag conventions."""
    r = np.asarray(returns, dtype=np.float64).ravel()
    lags = all_three_lags(r)
    by_lag: dict[str, dict[str, float]] = {}
    mean = float(np.mean(r))
    for name, lag in lags.items():
        _, se, t = newey_west_se(r, lag)
        by_lag[name] = {"lag": int(lag), "se": float(se), "t_stat": float(t)}
    return {"mean": mean, "n": int(r.size), "by_lag": by_lag}
