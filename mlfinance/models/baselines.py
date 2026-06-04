"""Simple forecast baselines for the conditional mean-reversion project."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConstantForecast:
    """BaseModel-compatible constant forecast."""

    name = "zero"

    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def fit(self, X_train, y_train, X_val=None, y_val=None):  # noqa: D401, ANN001
        return self

    def predict(self, X):  # noqa: ANN001
        return np.full(X.shape[0], self.value, dtype=float)


def baseline_predictions_from_panel(
    panel: pd.DataFrame,
    *,
    model: str,
    ret_1m_col: str = "ret_1m",
    mom_col: str = "mom_2_12",
    vol_col: str = "vol_12m",
    date_col: str = "date",
) -> pd.Series:
    """Return baseline signals from feature columns.

    Supported models: ``zero``, ``reversal``, ``zscore_reversal``, ``momentum``,
    ``reversal_plus_momentum``.
    """
    if model == "zero":
        return pd.Series(0.0, index=panel.index, name="y_hat")
    if model == "reversal":
        return -panel[ret_1m_col].astype(float)
    if model == "zscore_reversal":
        x = panel[ret_1m_col].astype(float)
        mu = x.groupby(panel[date_col]).transform("mean")
        sd = x.groupby(panel[date_col]).transform("std").replace(0.0, np.nan)
        return -((x - mu) / sd)
    if model == "vol_scaled_reversal":
        return -(
            panel[ret_1m_col].astype(float) / panel[vol_col].replace(0.0, np.nan).astype(float)
        )
    if model == "momentum":
        return panel[mom_col].astype(float)
    if model == "reversal_plus_momentum":
        rev = baseline_predictions_from_panel(panel, model="zscore_reversal", date_col=date_col)
        mom = panel[mom_col].astype(float)
        mom_z = (mom - mom.groupby(panel[date_col]).transform("mean")) / mom.groupby(
            panel[date_col]
        ).transform("std").replace(0.0, np.nan)
        return rev + mom_z
    raise ValueError(f"Unknown baseline model: {model}")
