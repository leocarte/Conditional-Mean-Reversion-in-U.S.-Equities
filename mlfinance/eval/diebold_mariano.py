"""Diebold-Mariano test of equal predictive accuracy (squared-error loss)."""

from __future__ import annotations

import numpy as np
from scipy import stats

from mlfinance.eval.newey_west import newey_west_lag_nw1994, newey_west_se

__all__ = ["diebold_mariano"]


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    """DM test on squared-error loss differentials.

    Returns (dm_stat, p_value). Positive ``dm_stat`` means model 1 has larger
    squared errors (model 2 is more accurate). Identical forecasts return
    ``(0.0, 1.0)``.
    """
    e1 = np.asarray(e1, dtype=np.float64).ravel()
    e2 = np.asarray(e2, dtype=np.float64).ravel()
    if e1.shape != e2.shape:
        raise ValueError(f"e1 {e1.shape} and e2 {e2.shape} must match.")
    if e1.size < 2:
        raise ValueError("Need at least 2 observations.")

    d = e1**2 - e2**2

    if np.all(d == 0.0):
        return 0.0, 1.0

    if lag is None:
        lag = newey_west_lag_nw1994(int(d.size))
    lag = int(min(max(lag, 0), d.size - 1))

    mean, se, _ = newey_west_se(d, lag)
    if not np.isfinite(se) or se == 0.0:
        return float("nan"), float("nan")

    dm_stat = mean / se
    p_value = 2.0 * float(stats.norm.sf(abs(dm_stat)))
    return float(dm_stat), float(p_value)
