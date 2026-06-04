"""Alpha regression on Fama-French 5 + UMD with Newey-West HAC SEs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["alpha_regression", "bootstrap_alpha_ci", "holm_bonferroni"]


_DEFAULT_FACTORS: tuple[str, ...] = ("mkt_rf", "smb", "hml", "rmw", "cma", "umd")


def alpha_regression(
    portfolio_returns: pd.Series,
    ff5_umd: pd.DataFrame,
    nw_lag: int = 11,
    factors: tuple[str, ...] = _DEFAULT_FACTORS,
) -> dict:
    """Regress portfolio returns on FF5 + UMD with NW HAC SEs.

    Default ``nw_lag=11`` (~sqrt(132), the project's 1987-2024 OOS sample).
    Pass ``factors=("mkt_rf",)`` for plain CAPM or the first three for FF3.
    """
    y = pd.Series(portfolio_returns).copy()
    y.name = y.name or "portfolio"
    X_df = ff5_umd.copy()
    missing = [c for c in factors if c not in X_df.columns]
    if missing:
        raise KeyError(f"Missing factor columns: {missing}.")

    aligned = pd.concat([y, X_df[list(factors)]], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < len(factors) + 2:
        raise ValueError(
            f"Need > {len(factors) + 1} aligned non-NaN observations, " f"got {len(aligned)}."
        )
    y_vec = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
    X_mat = aligned[list(factors)].to_numpy(dtype=np.float64)
    n = X_mat.shape[0]

    X = np.column_stack([np.ones(n), X_mat])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y_vec
    resid = y_vec - X @ beta

    cov = _newey_west_cov(X, resid, lag=int(min(max(nw_lag, 0), n - 1)))
    se = np.sqrt(np.diag(cov))

    t_stats = beta / np.where(se > 0, se, np.nan)
    p_values = 2.0 * stats.norm.sf(np.abs(t_stats))

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    factor_names = list(factors)
    betas = pd.Series(beta[1:], index=factor_names, name="beta")
    betas_se = pd.Series(se[1:], index=factor_names, name="se")
    betas_t = pd.Series(t_stats[1:], index=factor_names, name="t_stat")
    betas_p = pd.Series(p_values[1:], index=factor_names, name="p_value")

    return {
        "alpha": float(beta[0]),
        "alpha_se": float(se[0]),
        "alpha_t": float(t_stats[0]),
        "alpha_p": float(p_values[0]),
        "alpha_annualized": float(beta[0] * 12.0),
        "betas": betas,
        "betas_se": betas_se,
        "betas_t": betas_t,
        "betas_p": betas_p,
        "r_squared": float(r2),
        "n_obs": int(n),
        "nw_lag": int(min(max(nw_lag, 0), n - 1)),
    }


def _newey_west_cov(X: np.ndarray, resid: np.ndarray, lag: int) -> np.ndarray:
    """Standard Newey-West sandwich estimator for OLS coefficients."""
    n, _ = X.shape
    u = resid.reshape(-1, 1)
    Xu = X * u
    S = Xu.T @ Xu / n
    for j in range(1, lag + 1):
        cross = Xu[j:].T @ Xu[:-j] / n
        weight = 1.0 - j / (lag + 1)
        S += weight * (cross + cross.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    return XtX_inv @ S @ XtX_inv / n


# ---------------------------------------------------------------------------
# Robustness checks (block bootstrap, multiple-testing corrections)
# ---------------------------------------------------------------------------


def bootstrap_alpha_ci(
    portfolio_returns: pd.Series,
    ff5_umd: pd.DataFrame,
    factors: tuple[str, ...] = _DEFAULT_FACTORS,
    reps: int = 5000,
    block_length: float | None = None,
    alpha_level: float = 0.05,
    seed: int = 20260519,
) -> dict:
    """Stationary-bootstrap CI for the FF5+UMD alpha (intercept).

    Wraps ``arch.bootstrap.StationaryBootstrap`` (Politis-Romano 1994) with
    automatic block-length selection (Politis-White 2004, PPW 2009
    correction) so the parametric Newey-West kernel assumption is checked
    by a nonparametric resampler that preserves the serial dependence of
    the residuals.

    Falls back to a simple percentile bootstrap with a heuristic block
    length if ``arch`` is unavailable.
    """
    df = pd.concat(
        [portfolio_returns.rename("__y__"), ff5_umd.loc[:, list(factors)]], axis=1, join="inner"
    ).dropna()
    if len(df) < max(60, len(factors) * 5):
        raise ValueError(f"bootstrap_alpha_ci needs >=60 aligned obs, got {len(df)}")

    y = df["__y__"].to_numpy(dtype=np.float64)
    X_raw = df[list(factors)].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(y)), X_raw])

    # Point estimate (for reference).
    alpha_hat = float(np.linalg.lstsq(X, y, rcond=None)[0][0])

    # Build the bootstrap statistic as a closure on (y, X) so the
    # resampler hands us paired observations.
    def _alpha_from(yb: np.ndarray, Xb: np.ndarray) -> float:
        return float(np.linalg.lstsq(Xb, yb, rcond=None)[0][0])

    try:
        from arch.bootstrap import StationaryBootstrap, optimal_block_length

        if block_length is None:
            # Run optimal block length on the OLS residuals.
            beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta_ols
            try:
                pw = optimal_block_length(resid)
                block_length = float(pw["stationary"].iloc[0])
            except Exception:  # noqa: BLE001
                block_length = max(2.0, (len(y) ** (1 / 3)))
            block_length = float(np.clip(block_length, 2.0, len(y) / 10.0))

        rng = np.random.default_rng(seed)
        bs = StationaryBootstrap(block_length, y, X, seed=int(rng.integers(0, 2**31 - 1)))
        ci = bs.conf_int(
            _alpha_from,
            reps=reps,
            method="bca",
            size=1 - alpha_level,
            tail="two",
        )
        ci_low, ci_high = float(ci[0, 0]), float(ci[1, 0])
    except ImportError:
        # Pure-numpy fallback: stationary bootstrap by hand.
        if block_length is None:
            block_length = max(2.0, (len(y) ** (1 / 3)))
        rng = np.random.default_rng(seed)
        n = len(y)
        p = 1.0 / block_length
        alphas = np.empty(reps)
        for r in range(reps):
            idx = np.empty(n, dtype=np.int64)
            i = int(rng.integers(0, n))
            for t in range(n):
                i = int(rng.integers(0, n)) if rng.random() < p else (i + 1) % n
                idx[t] = i
            alphas[r] = _alpha_from(y[idx], X[idx])
        ci_low, ci_high = (
            float(np.quantile(alphas, alpha_level / 2)),
            float(np.quantile(alphas, 1 - alpha_level / 2)),
        )

    return {
        "alpha_hat": alpha_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "block_length": float(block_length),
        "reps": int(reps),
        "alpha_level": float(alpha_level),
        "excludes_zero": (ci_low > 0) or (ci_high < 0),
    }


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down FWER adjustment.

    Given a dict of {label: raw_p}, returns a dict of {label:
    adjusted_p} where rejection at level alpha controls family-wise
    error rate at alpha. Strictly more powerful than plain Bonferroni
    (same FWER guarantee, uniformly more rejections). See Wikipedia /
    Holm 1979.
    """
    labels = list(p_values.keys())
    raw = np.array([p_values[k] for k in labels], dtype=np.float64)
    m = len(raw)
    order = np.argsort(raw)  # smallest p first
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        # Step-down factor: (m - rank). Holm's inequality.
        candidate = min(1.0, (m - rank) * raw[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return {labels[i]: float(adjusted[i]) for i in range(m)}
