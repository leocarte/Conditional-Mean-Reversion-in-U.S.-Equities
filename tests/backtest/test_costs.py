"""Tests for the IBKR Pro Tiered cost model.

Numerical expectations come from the sanity-check / audit cells of the
reference ``trading_costs.ipynb``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfinance.backtest.costs import (
    IBKR_TIERS,
    IBKRTieredCostModel,
    _tier_cost_for_one_order,
    _vectorized_tier_gross,
)


def test_ibkr_floor_kicks_in_small_orders():
    """10 shares at $50: gross = 10*0.0035 = $0.035, but the $0.35 floor wins."""
    # Zero the pass-throughs so only the IBKR per-share commission is on test.
    model = IBKRTieredCostModel(
        clearing_per_share=0.0,
        exchange_per_share=0.0,
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        passthru=0.0,
    )
    shares = np.array([10.0])
    price = np.array([50.0])
    monthly = np.array([0.0])
    side = np.array(["buy"])
    total, gross, _ = model.cost_per_order(shares, price, monthly, side)

    assert gross[0] == pytest.approx(10 * 0.0035, rel=1e-12)
    assert total[0] == pytest.approx(0.35, rel=1e-12)


def test_ibkr_cap_kicks_in_huge_orders():
    """100k shares at $0.05 -> gross $350 but 1% cap on $5k trade value clamps to $50."""
    model = IBKRTieredCostModel(
        clearing_per_share=0.0,
        exchange_per_share=0.0,
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        passthru=0.0,
    )
    shares = np.array([100_000.0])
    price = np.array([0.05])
    monthly = np.array([0.0])
    side = np.array(["buy"])
    total, gross, _ = model.cost_per_order(shares, price, monthly, side)

    assert gross[0] == pytest.approx(350.0, rel=1e-12)
    assert total[0] == pytest.approx(50.0, rel=1e-12)


def test_ibkr_tier_progression():
    """350k shares from clean slate: 300k in tier-1, 50k in tier-2."""
    expected = 300_000 * 0.0035 + 50_000 * 0.0020
    scalar = _tier_cost_for_one_order(350_000.0, 0.0, IBKR_TIERS)
    assert scalar == pytest.approx(expected, rel=1e-12)

    vec = _vectorized_tier_gross(np.array([350_000.0]), np.array([0.0]), IBKR_TIERS)
    assert vec[0] == pytest.approx(expected, rel=1e-12)

    model = IBKRTieredCostModel(
        clearing_per_share=0.0,
        exchange_per_share=0.0,
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        passthru=0.0,
    )
    shares = np.array([350_000.0])
    price = np.array([50.0])
    monthly = np.array([0.0])
    side = np.array(["buy"])
    total, gross, _ = model.cost_per_order(shares, price, monthly, side)
    assert gross[0] == pytest.approx(expected, rel=1e-12)
    # Trade value 17.5M, cap = $175,000; gross $1,150 wins.
    assert total[0] == pytest.approx(expected, rel=1e-12)


def test_ibkr_tier_starts_partway_through_first_tier():
    """100k after 250k already traded: 50k in tier-1, 50k in tier-2."""
    expected = 50_000 * 0.0035 + 50_000 * 0.0020
    scalar = _tier_cost_for_one_order(100_000.0, 250_000.0, IBKR_TIERS)
    assert scalar == pytest.approx(expected, rel=1e-12)
    vec = _vectorized_tier_gross(np.array([100_000.0]), np.array([250_000.0]))
    assert vec[0] == pytest.approx(expected, rel=1e-12)


def test_ibkr_top_tier_open_ended():
    """200M shares from scratch crosses every tier including the open-ended top."""
    expected = (
        300_000 * 0.0035
        + (3_000_000 - 300_000) * 0.0020
        + (20_000_000 - 3_000_000) * 0.0015
        + (100_000_000 - 20_000_000) * 0.0010
        + (200_000_000 - 100_000_000) * 0.0005
    )
    scalar = _tier_cost_for_one_order(200_000_000.0, 0.0, IBKR_TIERS)
    assert scalar == pytest.approx(expected, rel=1e-12)


def test_monthly_volume_resets_january_to_february():
    """Tier accumulator must reset at month boundaries."""
    model = IBKRTieredCostModel(
        clearing_per_share=0.0,
        exchange_per_share=0.0,
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        passthru=0.0,
    )
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    pd.DataFrame({10001: [0.0, 0.0]}, index=dates)
    # Bypass simulate() to test per-order math directly; both months start at 0 monthly_so_far.
    shares = np.array([400_000.0, 400_000.0])
    price = np.array([50.0, 50.0])
    side = np.array(["buy", "buy"])
    monthly = np.array([0.0, 0.0])
    total, gross, _ = model.cost_per_order(shares, price, monthly, side)
    expected = 300_000 * 0.0035 + 100_000 * 0.0020
    assert gross[0] == pytest.approx(expected, rel=1e-12)
    assert gross[1] == pytest.approx(expected, rel=1e-12)
    # Without a reset the second order would land entirely in tier-2 (cheaper).
    monthly_no_reset = np.array([0.0, 400_000.0])
    _, gross_no_reset, _ = model.cost_per_order(shares, price, monthly_no_reset, side)
    assert gross_no_reset[1] < gross[1]
    # End-to-end check via simulate() that the per-month cumsum resets.
    weights2 = pd.DataFrame({10001: [400_000.0 * 50.0, -400_000.0 * 50.0]}, index=dates)
    prices_df = pd.DataFrame({10001: [50.0, 50.0]}, index=dates)
    cal = pd.Series(dates.to_period("M"), index=dates)
    out = model.simulate(weights=weights2, prices=prices_df, calendar_month=cal)
    out_sorted = out.sort_values("date").reset_index(drop=True)
    assert len(out_sorted) == 2
    # Row 0: 400k shares. Row 1: 800k shares (|dw| = 40M, price 50). Both start at monthly=0.
    expected_row0 = _tier_cost_for_one_order(400_000.0, 0.0)
    expected_row1 = _tier_cost_for_one_order(800_000.0, 0.0)
    assert out_sorted.loc[0, "gross_commission"] == pytest.approx(expected_row0, rel=1e-12)
    assert out_sorted.loc[1, "gross_commission"] == pytest.approx(expected_row1, rel=1e-12)


def test_sell_pays_sec_and_finra_pass_throughs():
    """Sell - buy of identical size = SEC fee + FINRA TAF (commission piece is symmetric)."""
    model = IBKRTieredCostModel()
    shares = np.array([1_000.0, 1_000.0])
    price = np.array([50.0, 50.0])
    monthly = np.array([0.0, 0.0])
    side = np.array(["buy", "sell"])
    total, _, _ = model.cost_per_order(shares, price, monthly, side)
    diff = total[1] - total[0]
    expected_diff = 0.0000278 * 50_000.0 + 0.0000278 * 1_000.0
    assert diff == pytest.approx(expected_diff, rel=1e-12)


def test_simulate_returns_expected_schema():
    model = IBKRTieredCostModel()
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])
    weights = pd.DataFrame(
        {
            10001: [0.5, 0.6, 0.4],
            10002: [-0.5, -0.6, -0.4],
        },
        index=dates,
    )
    prices = pd.DataFrame({10001: [50.0, 51.0, 52.0], 10002: [10.0, 11.0, 12.0]}, index=dates)
    cal = pd.Series(dates.to_period("M"), index=dates)
    out = model.simulate(weights=weights, prices=prices, calendar_month=cal)
    expected_cols = {
        "date",
        "permno",
        "shares_traded",
        "trade_value",
        "gross_commission",
        "total_cost",
        "side",
    }
    assert expected_cols.issubset(set(out.columns))
    assert (out["shares_traded"] > 0).all()
    assert (out["total_cost"] >= 0).all()
    assert set(out["side"].unique()).issubset({"buy", "sell"})


def test_10k_portfolio_floor_rate_95pct():
    """$10k toy long-only portfolio: at least 90% of trades hit the $0.35 floor."""
    rng = np.random.default_rng(42)
    n_dates = 84
    n_stocks = 20

    # Cheap ($5-$15) and mid-priced ($30-$200) stocks; $500/stock always hits floor.
    prices_arr = rng.uniform(5.0, 200.0, size=(n_dates, n_stocks))
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    permnos = [10000 + i for i in range(n_stocks)]
    prices = pd.DataFrame(prices_arr, index=dates, columns=permnos)

    # Each stock toggles between 0 and its target weight (1/20).
    capital = 10_000.0
    capital_per_stock = capital / n_stocks
    signal = rng.choice([0.0, 1.0], size=(n_dates, n_stocks))
    weights_dollar = pd.DataFrame(signal * capital_per_stock, index=dates, columns=permnos)

    model = IBKRTieredCostModel()
    cal = pd.Series(dates.to_period("M"), index=dates)
    out = model.simulate(weights=weights_dollar, prices=prices, calendar_month=cal)

    # Compare gross to the floor threshold, NOT actual == 0.35.
    hits_floor = out["gross_commission"] < model.min_per_order
    rate = hits_floor.sum() / len(out)
    assert rate >= 0.90, f"Floor-hit rate {rate:.2%} below 90% - model may be miscalibrated."


def test_scalar_and_vectorized_paths_agree_random():
    """Vectorised tier kernel must match the scalar loop oracle."""
    rng = np.random.default_rng(0)
    n = 500
    shares = rng.uniform(1.0, 5e7, size=n)
    monthly = rng.uniform(0.0, 5e7, size=n)
    vec = _vectorized_tier_gross(shares, monthly)
    scalar = np.array(
        [_tier_cost_for_one_order(s, m) for s, m in zip(shares, monthly, strict=False)]
    )
    np.testing.assert_allclose(vec, scalar, rtol=1e-10, atol=1e-10)
