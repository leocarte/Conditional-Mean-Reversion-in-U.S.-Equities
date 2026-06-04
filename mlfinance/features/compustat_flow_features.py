"""Flow-only Compustat feature helpers for the audited CompFirmCharac schema.

This module is intentionally narrow:

- use only reviewed quarterly YTD flow fields from the cluster schema audit,
- enforce a conservative point-in-time lag before any later merge,
- never fabricate balance-sheet-style features that the file cannot support,
- expose only the audited flow block reused by the default-off Compustat integration path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlfinance.data.compustat_schema import (
    audit_compustat_point_in_time_merge,
    merge_compustat_point_in_time,
    prepare_compustat_for_point_in_time_merge,
)

SUPPORTED_COMPUSTAT_MERGE_KEYS = frozenset({"gvkey", "cusip8"})

FLOW_ONLY_COMPUSTAT_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "sales": ("saley", "revty"),
    "cogs": ("cogsy",),
    "earnings": ("niy", "iby"),
    "operating_income": ("oiadpy", "oibdpy"),
    "operating_cash_flow": ("oancfy",),
    "capex": ("capxy",),
    "rd": ("xrdy",),
    "sga": ("xsgay",),
    "interest_expense": ("xinty",),
    "dividends": ("dvy",),
    "depreciation": ("dpy",),
    "pretax_income": ("piy",),
    "taxes": ("txty",),
}

COMPUSTAT_FLOW_FEATURE_COLUMNS: tuple[str, ...] = (
    "compustat_sales_log1p",
    "compustat_cogs_log1p",
    "compustat_capex_log1p",
    "compustat_rd_log1p",
    "compustat_sga_log1p",
    "compustat_dividends_log1p",
    "compustat_depreciation_log1p",
    "compustat_sales_yoy_growth",
    "compustat_earnings_yoy_growth",
    "compustat_operating_cash_flow_yoy_growth",
    "compustat_capex_yoy_growth",
    "compustat_gross_margin",
    "compustat_operating_margin",
    "compustat_net_margin",
    "compustat_operating_cash_flow_margin",
    "compustat_capex_intensity",
    "compustat_rd_intensity",
    "compustat_sga_intensity",
    "compustat_interest_intensity",
    "compustat_dividends_intensity",
    "compustat_depreciation_intensity",
    "compustat_accrual_proxy",
    "compustat_tax_burden",
)

_FLOW_METADATA_COLUMNS: tuple[str, ...] = (
    "gvkey",
    "cusip",
    "cusip8",
    "datadate",
    "availability_date",
    "fyearq",
    "fqtr",
)


def normalize_cusip8(series: pd.Series) -> pd.Series:
    """Return strict 8-character CUSIP keys, dropping blank or malformed values."""
    cleaned = series.astype("string").str.strip().str.upper()
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA).mask(cleaned.eq("<NA>"), pd.NA)
    cleaned = cleaned.str[:8]
    return cleaned.where(cleaned.str.len() == 8)


def compustat_flow_feature_columns(
    *,
    add_missingness_flags: bool = True,
) -> tuple[str, ...]:
    """Return the reviewed flow-only feature columns exposed to the panel."""
    columns = list(COMPUSTAT_FLOW_FEATURE_COLUMNS)
    if add_missingness_flags:
        columns.extend(f"{column}_missing" for column in COMPUSTAT_FLOW_FEATURE_COLUMNS)
    return tuple(columns)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def _normalise_gvkey(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA).mask(cleaned.eq("<NA>"), pd.NA)
    return cleaned.str.zfill(6)


def _panel_merge_key_values(panel: pd.DataFrame, merge_key: str) -> pd.Series:
    if merge_key == "gvkey":
        if "gvkey" not in panel.columns:
            raise KeyError(
                "add_flow_compustat_features_to_panel: panel is missing required merge key "
                "'gvkey'."
            )
        return _normalise_gvkey(panel["gvkey"])

    if merge_key == "cusip8":
        for source_col in ("cusip8", "cusip", "hdrcusip"):
            if source_col in panel.columns:
                return normalize_cusip8(panel[source_col])
        raise KeyError(
            "add_flow_compustat_features_to_panel: panel is missing CUSIP information "
            "required for strict 'cusip8' linking."
        )

    raise ValueError(
        "add_flow_compustat_features_to_panel supports only 'gvkey' or 'cusip8' merge keys."
    )


def _coerce_numeric(series: pd.Series, index: pd.Index) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").reindex(index).astype("float64")


def _empty_float(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _first_available_numeric(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    out = _empty_float(df.index)
    found_any = False
    for column in candidates:
        if column not in df.columns:
            continue
        found_any = True
        numeric = _coerce_numeric(df[column], df.index)
        out = out.combine_first(numeric)
    return out if found_any else _empty_float(df.index)


def _positive_log1p(series: pd.Series) -> pd.Series:
    return np.log1p(series.where(series.ge(0)))


def _safe_abs_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.abs().where(denominator.notna() & denominator.ne(0))
    return numerator / safe_denominator


def _yoy_growth(df: pd.DataFrame, value: pd.Series, firm_key: str) -> pd.Series:
    growth = _empty_float(df.index)
    if "fqtr" not in df.columns:
        return growth

    valid_mask = df[firm_key].notna() & df["fqtr"].notna()
    if not valid_mask.any():
        return growth

    order_cols = [firm_key, "fqtr"]
    if "fyearq" in df.columns:
        order_cols.append("fyearq")
    order_cols.append("datadate")

    ordered = df.loc[valid_mask].copy()
    ordered["_value"] = value.loc[valid_mask]
    ordered = ordered.sort_values(order_cols)

    group_keys = [firm_key, "fqtr"]
    prior_value = ordered.groupby(group_keys, dropna=False)["_value"].shift(1)

    if "fyearq" in ordered.columns:
        prior_year = ordered.groupby(group_keys, dropna=False)["fyearq"].shift(1)
        valid_lag = ordered["fyearq"].sub(prior_year).eq(1)
    else:
        prior_date = ordered.groupby(group_keys, dropna=False)["datadate"].shift(1)
        valid_lag = (
            prior_date.notna()
            & ordered["datadate"].notna()
            & ordered["datadate"].dt.year.sub(prior_date.dt.year).eq(1)
        )

    denominator = prior_value.abs().where(prior_value.notna() & prior_value.ne(0))
    yoy = ((ordered["_value"] - prior_value) / denominator).where(valid_lag)
    growth.loc[ordered.index] = yoy
    return growth


def prepare_compustat_flow_frame(
    compustat: pd.DataFrame,
    *,
    merge_key: str = "gvkey",
    lag_months: int = 6,
) -> pd.DataFrame:
    """Return a point-in-time-safe flow-only Compustat frame with canonical inputs."""
    if merge_key not in SUPPORTED_COMPUSTAT_MERGE_KEYS:
        raise ValueError(
            "prepare_compustat_flow_frame supports only 'gvkey' or 'cusip8' merge keys."
        )

    base = _normalise_columns(compustat)
    if "datadate" not in base.columns:
        raise KeyError("prepare_compustat_flow_frame: missing required column 'datadate'.")

    if "gvkey" in base.columns:
        base["gvkey"] = _normalise_gvkey(base["gvkey"])
    if "cusip" in base.columns:
        base["cusip"] = base["cusip"].astype("string").str.strip().str.upper()
        base["cusip"] = base["cusip"].mask(base["cusip"].eq(""), pd.NA)
    if "cusip8" in base.columns:
        base["cusip8"] = normalize_cusip8(base["cusip8"])
    elif "cusip" in base.columns:
        base["cusip8"] = normalize_cusip8(base["cusip"])
    if "fyearq" in base.columns:
        base["fyearq"] = pd.to_numeric(base["fyearq"], errors="coerce").astype("Int64")
    if "fqtr" in base.columns:
        base["fqtr"] = pd.to_numeric(base["fqtr"], errors="coerce").astype("Int64")

    prepared = prepare_compustat_for_point_in_time_merge(
        base,
        merge_key=merge_key,
        lag_months=lag_months,
    )

    id_cols = [col for col in _FLOW_METADATA_COLUMNS if col in prepared.columns]
    out = prepared[id_cols].copy()
    for canonical, candidates in FLOW_ONLY_COMPUSTAT_FIELD_CANDIDATES.items():
        out[canonical] = _first_available_numeric(prepared, candidates)
    return out


def build_compustat_flow_features(
    compustat: pd.DataFrame,
    *,
    merge_key: str = "gvkey",
    lag_months: int = 6,
    add_missingness_flags: bool = True,
) -> pd.DataFrame:
    """Build reviewed flow-only accounting features without activating the pipeline."""
    out = prepare_compustat_flow_frame(compustat, merge_key=merge_key, lag_months=lag_months)

    sales = out["sales"]
    cogs = out["cogs"]
    earnings = out["earnings"]
    operating_income = out["operating_income"]
    operating_cash_flow = out["operating_cash_flow"]
    capex = out["capex"]
    rd = out["rd"]
    sga = out["sga"]
    interest_expense = out["interest_expense"]
    dividends = out["dividends"]
    depreciation = out["depreciation"]
    pretax_income = out["pretax_income"]
    taxes = out["taxes"]

    out["compustat_sales_log1p"] = _positive_log1p(sales)
    out["compustat_cogs_log1p"] = _positive_log1p(cogs)
    out["compustat_capex_log1p"] = _positive_log1p(capex)
    out["compustat_rd_log1p"] = _positive_log1p(rd)
    out["compustat_sga_log1p"] = _positive_log1p(sga)
    out["compustat_dividends_log1p"] = _positive_log1p(dividends)
    out["compustat_depreciation_log1p"] = _positive_log1p(depreciation)

    out["compustat_sales_yoy_growth"] = _yoy_growth(out, sales, merge_key)
    out["compustat_earnings_yoy_growth"] = _yoy_growth(out, earnings, merge_key)
    out["compustat_operating_cash_flow_yoy_growth"] = _yoy_growth(
        out, operating_cash_flow, merge_key
    )
    out["compustat_capex_yoy_growth"] = _yoy_growth(out, capex, merge_key)

    out["compustat_gross_margin"] = _safe_abs_ratio(sales - cogs, sales)
    out["compustat_operating_margin"] = _safe_abs_ratio(operating_income, sales)
    out["compustat_net_margin"] = _safe_abs_ratio(earnings, sales)
    out["compustat_operating_cash_flow_margin"] = _safe_abs_ratio(operating_cash_flow, sales)
    out["compustat_capex_intensity"] = _safe_abs_ratio(capex, sales)
    out["compustat_rd_intensity"] = _safe_abs_ratio(rd, sales)
    out["compustat_sga_intensity"] = _safe_abs_ratio(sga, sales)
    out["compustat_interest_intensity"] = _safe_abs_ratio(interest_expense, sales)
    out["compustat_dividends_intensity"] = _safe_abs_ratio(dividends, sales)
    out["compustat_depreciation_intensity"] = _safe_abs_ratio(depreciation, sales)
    out["compustat_accrual_proxy"] = _safe_abs_ratio(earnings - operating_cash_flow, sales)
    out["compustat_tax_burden"] = _safe_abs_ratio(taxes, pretax_income)

    if add_missingness_flags:
        for column in COMPUSTAT_FLOW_FEATURE_COLUMNS:
            out[f"{column}_missing"] = out[column].isna().astype("int8")

    return out


def add_flow_compustat_features_to_panel(
    panel: pd.DataFrame,
    compustat: pd.DataFrame,
    *,
    merge_key: str = "cusip8",
    date_col: str = "date",
    lag_months: int = 6,
    add_missingness_flags: bool = True,
) -> pd.DataFrame:
    """Build flow-only Compustat features and merge them onto formation rows."""
    if date_col not in panel.columns:
        raise KeyError(f"add_flow_compustat_features_to_panel: panel is missing '{date_col}'.")

    left = panel.copy()
    left[date_col] = pd.to_datetime(left[date_col]) + pd.offsets.MonthEnd(0)
    left[merge_key] = _panel_merge_key_values(left, merge_key)

    features = build_compustat_flow_features(
        compustat,
        merge_key=merge_key,
        lag_months=lag_months,
        add_missingness_flags=add_missingness_flags,
    )

    feature_cols = list(compustat_flow_feature_columns(add_missingness_flags=add_missingness_flags))
    merge_feature_cols = [column.removeprefix("compustat_") for column in feature_cols]
    feature_block = features[[merge_key, "datadate", "availability_date", *feature_cols]].rename(
        columns={column: column.removeprefix("compustat_") for column in feature_cols}
    )

    merged = merge_compustat_point_in_time(
        left,
        feature_block,
        merge_key=merge_key,
        formation_date_col=date_col,
        lag_months=lag_months,
        compustat_cols=merge_feature_cols,
        prefix="compustat_",
    )

    # For rows with no available Compustat match, leave accounting values as NaN
    # but make the reviewed missingness indicators usable binary flags.
    for column in compustat_flow_feature_columns(add_missingness_flags=True):
        if column.endswith("_missing") and column in merged.columns:
            merged[column] = merged[column].fillna(1).astype("int8")

    audit_compustat_point_in_time_merge(merged, formation_date_col=date_col, raise_on_fail=True)
    return merged


__all__ = [
    "COMPUSTAT_FLOW_FEATURE_COLUMNS",
    "FLOW_ONLY_COMPUSTAT_FIELD_CANDIDATES",
    "SUPPORTED_COMPUSTAT_MERGE_KEYS",
    "add_flow_compustat_features_to_panel",
    "build_compustat_flow_features",
    "compustat_flow_feature_columns",
    "normalize_cusip8",
    "prepare_compustat_flow_frame",
]
