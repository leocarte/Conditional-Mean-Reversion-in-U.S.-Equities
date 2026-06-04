"""Unit tests for CRSP universe filters (pure pandas; no GPU/torch/xgboost)."""

from __future__ import annotations

import pandas as pd

from mlfinance.data.universe import apply_universe, field_availability


def _frame() -> pd.DataFrame:
    """Two month-ends x five stocks; ``siccd`` present, prc/shrcd/exchcd absent."""
    sics = {1: 1040, 2: 6020, 3: 4911, 4: 3570, 5: 6199}  # 2,5 financial; 3 utility
    rows = [
        {"permno": permno, "date": date, "siccd": sic, "prediction": 0.01}
        for date in pd.to_datetime(["2020-01-31", "2020-02-29"])
        for permno, sic in sics.items()
    ]
    return pd.DataFrame(rows)


def test_field_availability_flags_present_and_absent() -> None:
    avail = field_availability(["permno", "date", "siccd", "prediction"])
    assert avail["sic_exclude"] is True
    assert avail["price_min"] is False
    assert avail["market_cap_min"] is False


def test_sic_exclude_drops_financials() -> None:
    out, reports = apply_universe(_frame(), {"sic_exclude": [[6000, 6999]]})
    assert set(out["siccd"].unique()) == {1040, 4911, 3570}  # 6020, 6199 dropped
    rep = {r.name: r for r in reports}
    assert rep["sic_exclude"].applied is True
    assert rep["sic_exclude"].rows_after < rep["sic_exclude"].rows_before


def test_ex_fin_utils_drops_financials_and_utilities() -> None:
    out, _ = apply_universe(_frame(), {"sic_exclude": [[6000, 6999], [4900, 4949]]})
    assert set(out["siccd"].unique()) == {1040, 3570}


def test_missing_column_skips_filter_with_reason() -> None:
    df = _frame()  # no prc column
    out, reports = apply_universe(df, {"price_min": 5.0})
    rep = {r.name: r for r in reports}
    assert rep["price_min"].applied is False
    assert "prc" in rep["price_min"].reason
    assert len(out) == len(df)  # nothing dropped when the field is unavailable


def test_price_filter_applies_when_prc_present() -> None:
    df = _frame()
    df["prc"] = [3.0, 10.0, 7.0, 2.0, 50.0] * 2  # permno 1 ($3) and 4 ($2) below $5
    out, reports = apply_universe(df, {"price_min": 5.0})
    assert (pd.to_numeric(out["prc"]).abs() >= 5.0).all()
    assert {r.name: r for r in reports}["price_min"].applied is True


def test_filter_never_increases_row_count() -> None:
    df = _frame()
    for spec in ({}, {"sic_exclude": [[6000, 6999]]}, {"price_min": 5.0}):
        out, _ = apply_universe(df, spec)
        assert len(out) <= len(df)


def test_empty_spec_is_identity() -> None:
    df = _frame()
    out, reports = apply_universe(df, {})
    assert len(out) == len(df)
    assert reports == []


def test_min_stocks_per_month_drops_thin_dates() -> None:
    df = _frame()  # 5 stocks per date
    thin = df[(df["date"] == pd.Timestamp("2020-02-29")) & (df["permno"] != 1)].index
    df = df.drop(thin)  # February now has a single stock
    out, reports = apply_universe(df, {"min_stocks_per_month": 3})
    assert set(out["date"].dt.month.unique()) == {1}  # February dropped, January kept
    assert {r.name: r for r in reports}["min_stocks_per_month"].applied is True
