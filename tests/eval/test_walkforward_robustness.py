"""Unit tests for the walk-forward post-processing robustness helpers (pure functions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfinance.run.walkforward_robustness import (
    _alpha_beta,
    _nw_mean_tstat,
    _regime_labels,
    _strategy_books,
    breakeven_bps,
    cost_grid_metrics,
)


def test_breakeven_formula_known_value() -> None:
    # 10000 * 0.01 / 2.0 = 50 bps
    assert breakeven_bps(0.01, 2.0) == pytest.approx(50.0)
    assert breakeven_bps(0.02, 2.5) == pytest.approx(80.0)


def test_breakeven_zero_or_missing_turnover_is_nan_not_inf() -> None:
    assert np.isnan(breakeven_bps(0.01, 0.0))
    assert np.isnan(breakeven_bps(0.01, -1.0))
    assert np.isnan(breakeven_bps(0.01, float("nan")))
    # finite gross with valid turnover stays finite
    assert np.isfinite(breakeven_bps(0.01, 2.0))


def test_cost_grid_extraction_preserves_fractional_bps() -> None:
    ms = pd.DataFrame(
        {
            "model_name": ["flow_compustat__mlp"] * 3,
            "feature_set": ["flow_compustat"] * 3,
            "model": ["mlp"] * 3,
            "cutoff": [0.10, 0.10, 0.10],
            "cost_bps": [5.0, 7.5, 12.5],
            "gross_sharpe": [2.0, 2.0, 2.0],
            "net_sharpe": [1.9, 1.8, 1.7],
            "net_ann_return": [0.2, 0.19, 0.18],
            "avg_turnover": [2.1, 2.1, 2.1],
            "breakeven_bps": [90.0, 90.0, 90.0],
            "rank_ic_mean": [0.04, 0.04, 0.04],
            "rank_ic_hac_tstat": [6.0, 6.0, 6.0],
            "oos_r2": [-0.004, -0.004, -0.004],
        }
    )
    out = cost_grid_metrics(ms)
    assert (out["cost_bps"] == 7.5).any()
    assert (out["cost_bps"] == 12.5).any()
    # avg_turnover was harmonized to 'turnover'
    assert "turnover" in out.columns
    assert {"feature_set", "model", "cutoff", "cost_bps", "net_sharpe"}.issubset(out.columns)


def test_nw_mean_tstat_runs_on_synthetic_returns() -> None:
    rng = np.random.default_rng(0)
    r = pd.Series(0.01 + 0.02 * rng.standard_normal(96))  # positive mean
    res = _nw_mean_tstat(r, hac_lag=11)
    assert res["n"] == 96
    assert np.isfinite(res["tstat"])
    assert res["tstat"] > 0  # positive-mean series -> positive t-stat


def test_nw_mean_tstat_small_n_is_nan_not_crash() -> None:
    res = _nw_mean_tstat(pd.Series([0.01, 0.02]), hac_lag=11)
    assert np.isnan(res["tstat"])


def test_alpha_beta_recovers_known_synthetic_beta() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-31", periods=72, freq="ME")
    rm = pd.Series(0.04 * rng.standard_normal(72), index=idx)
    true_alpha, true_beta = 0.002, 1.3
    rp = true_alpha + true_beta * rm + pd.Series(0.006 * rng.standard_normal(72), index=idx)
    res = _alpha_beta(rp, rm, hac_lag=6)
    assert res["beta"] == pytest.approx(true_beta, abs=0.15)
    assert res["alpha"] == pytest.approx(true_alpha, abs=0.004)
    assert np.isfinite(res["beta_tstat"])


def test_alpha_beta_too_few_months_is_nan() -> None:
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    res = _alpha_beta(
        pd.Series([0.01] * 4, index=idx), pd.Series([0.02] * 4, index=idx), hac_lag=11
    )
    assert np.isnan(res["beta"])
    assert res["n"] == 4


def test_regime_labels_period_split_counts() -> None:
    idx = pd.date_range("2018-01-31", "2021-12-31", freq="ME")  # 48 months
    market = pd.Series(0.01 + 0.005 * np.sin(np.arange(len(idx))), index=idx)
    dispersion = pd.Series(np.linspace(0.05, 0.15, len(idx)), index=idx)
    labels = _regime_labels(market, dispersion)
    assert set(labels) == {"vol", "trend", "dispersion", "period"}
    period = labels["period"]
    assert int((period == "2017-2019").sum()) == 24  # 2018 + 2019
    assert int((period == "2020-2024").sum()) == 24  # 2020 + 2021
    # the data-driven schemes split into at most two non-null labels
    for scheme in ("vol", "trend", "dispersion"):
        assert set(labels[scheme].dropna().unique()).issubset(
            {"high_vol", "low_vol", "bull", "bear", "high_dispersion", "low_dispersion"}
        )


def _toy_predictions(april_onwards_pred_shift: float) -> pd.DataFrame:
    """6 monthly cross-sections (Jan-Jun 2017), 4 permnos.

    Predictions for Apr-Jun are shifted by ``april_onwards_pred_shift`` so two
    versions differ ONLY from April; Jan-Mar are identical.
    """
    dates = pd.date_range("2017-01-31", periods=6, freq="ME")
    permnos = [101, 102, 103, 104]
    rows = []
    for di, d in enumerate(dates):
        for pi, p in enumerate(permnos):
            base_pred = float(pi - 1.5)  # deterministic cross-sectional ordering
            shift = april_onwards_pred_shift if di >= 3 else 0.0
            # from April, the shifted version reverses the predicted ordering
            pred = base_pred if shift == 0.0 else -base_pred
            rows.append(
                {
                    "date": d,
                    "permno": p,
                    "prediction": pred,
                    # monotonic in permno so the long-short leg is non-degenerate
                    # (a reversed ordering in April flips the sign of the gross return)
                    "target_ret_fwd_1m": 0.005 * pi + 0.001 * di,
                }
            )
    return pd.DataFrame(rows)


def test_quarterly_rebalance_does_not_use_future_predictions() -> None:
    base = _toy_predictions(0.0)
    altered = _toy_predictions(1.0)  # different Apr-Jun predictions, identical Jan-Mar

    kw = dict(
        date_col="date",
        permno_col="permno",
        target_col="target_ret_fwd_1m",
        n_buckets=2,
        cost_bps=10.0,
    )
    books_base = _strategy_books(base, **kw)
    books_alt = _strategy_books(altered, **kw)

    q_base = books_base["quarterly"].set_index("date").sort_index()
    q_alt = books_alt["quarterly"].set_index("date").sort_index()

    hold_months = pd.date_range("2017-01-31", periods=3, freq="ME")  # Jan(rebal), Feb, Mar (hold)
    for col in ("gross_return", "turnover", "net_return"):
        np.testing.assert_allclose(
            q_base.loc[hold_months, col].to_numpy(),
            q_alt.loc[hold_months, col].to_numpy(),
            err_msg=f"quarterly {col} in Jan-Mar must not depend on Apr+ predictions",
        )
    # sanity: the books DO diverge once the April rebalance ingests new predictions
    assert not np.allclose(
        q_base.loc[pd.Timestamp("2017-04-30"), "gross_return"],
        q_alt.loc[pd.Timestamp("2017-04-30"), "gross_return"],
    )
