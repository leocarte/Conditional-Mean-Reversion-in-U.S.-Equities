"""Statistical tests used in the final project."""

from __future__ import annotations

import numpy as np
import pandas as pd

# statsmodels is imported lazily inside the functions below: importing this module
# (and the whole mlfinance.eval package) must not require statsmodels at load time,
# so stages that never compute HAC stats can run on environments where statsmodels
# is unavailable. Only callers that actually fit the OLS pull it in.


def newey_west_mean_tstat(returns: pd.Series, lags: int = 11) -> dict[str, float]:
    """HAC t-stat for the mean of a monthly return series."""
    import statsmodels.api as sm

    r = returns.dropna().astype(float)
    if r.empty:
        return {"mean": np.nan, "tstat": np.nan, "n": 0}
    X = np.ones((len(r), 1))
    fit = sm.OLS(r.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return {"mean": float(fit.params[0]), "tstat": float(fit.tvalues[0]), "n": int(len(r))}


def market_alpha_beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    *,
    lags: int = 11,
) -> dict[str, float]:
    """HAC alpha/beta regression against the market benchmark."""
    import statsmodels.api as sm

    df = pd.concat([portfolio_returns.rename("rp"), market_returns.rename("rm")], axis=1).dropna()
    if df.empty:
        return {
            "alpha": np.nan,
            "alpha_tstat": np.nan,
            "beta": np.nan,
            "beta_tstat": np.nan,
            "n": 0,
        }
    X = sm.add_constant(df["rm"])
    fit = sm.OLS(df["rp"], X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
    return {
        "alpha": float(fit.params["const"]),
        "alpha_tstat": float(fit.tvalues["const"]),
        "beta": float(fit.params["rm"]),
        "beta_tstat": float(fit.tvalues["rm"]),
        "n": int(len(df)),
    }
