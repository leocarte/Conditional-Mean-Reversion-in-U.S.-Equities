"""feature-audit Compustat schema audit and conservative point-in-time merge helpers.

These helpers are intentionally lightweight. They inspect a Compustat-like file
or DataFrame, report schema feasibility for the planned feature-audit accounting
block, and enforce conservative lagging before any point-in-time join to CRSP
formation months. They do not build a full feature set or alter model logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mlfinance.data.loaders import _coerce_datetime, _read_table

COMPUSTAT_REQUIRED_FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gvkey", ("gvkey",)),
    ("datadate", ("datadate",)),
    ("fyear", ("fyear",)),
    ("at", ("at",)),
    ("common_equity", ("ceq", "seq")),
    ("txditc", ("txditc",)),
    ("preferred_stock", ("pstkrv", "pstkl", "pstk")),
    ("earnings", ("ni", "ib")),
    ("sale", ("sale",)),
    ("cogs", ("cogs",)),
    ("dltt", ("dltt",)),
    ("dlc", ("dlc",)),
    ("oancf", ("oancf",)),
    ("capx", ("capx",)),
)

COMPUSTAT_CANDIDATE_MERGE_KEYS: tuple[str, ...] = ("gvkey", "cusip", "cusip8", "cik")

_DATE_COL = "datadate"
_DEFAULT_PIT_PREFIX = "compustat_"


@dataclass(frozen=True)
class CompustatSchemaAuditReport:
    """Structured schema-feasibility report for a Compustat-like extract."""

    source: str | None
    row_count: int
    column_count: int
    field_coverage: pd.DataFrame
    required_group_status: pd.DataFrame
    merge_key_coverage: pd.DataFrame
    parsable_datadate_count: int
    min_datadate: pd.Timestamp | None
    max_datadate: pd.Timestamp | None

    @property
    def missing_candidate_fields(self) -> list[str]:
        """Candidate fields absent from the extract."""
        missing = self.field_coverage.loc[~self.field_coverage["present"], "field"]
        return [str(field) for field in missing.tolist()]

    @property
    def unsatisfied_required_groups(self) -> list[str]:
        """Required groups with no usable alternative column."""
        missing = self.required_group_status.loc[
            ~self.required_group_status["satisfied"], "requirement"
        ]
        return [str(field) for field in missing.tolist()]

    def to_dict(self) -> dict[str, Any]:
        """Compact dict summary for logs or downstream formatting."""
        return {
            "source": self.source,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "missing_candidate_fields": self.missing_candidate_fields,
            "unsatisfied_required_groups": self.unsatisfied_required_groups,
            "parsable_datadate_count": self.parsable_datadate_count,
            "min_datadate": self.min_datadate,
            "max_datadate": self.max_datadate,
        }


def _load_frame(table_or_path: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if isinstance(table_or_path, pd.DataFrame):
        return table_or_path.copy(), None

    path = Path(table_or_path)
    return _read_table(path), str(path)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def _clean_string_series(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.eq(""), pd.NA).mask(out.eq("<NA>"), pd.NA)


def _field_non_null_count(df: pd.DataFrame, field: str) -> int:
    if field not in df.columns:
        return 0

    series = df[field]
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        series = _clean_string_series(series)
    return int(series.notna().sum())


def _candidate_field_order(
    required_groups: Sequence[tuple[str, Sequence[str]]],
    merge_keys: Sequence[str],
) -> list[str]:
    ordered: list[str] = []
    for _, alternatives in required_groups:
        for field in alternatives:
            if field not in ordered:
                ordered.append(field)
    for field in merge_keys:
        if field not in ordered:
            ordered.append(field)
    return ordered


def _normalise_merge_key(series: pd.Series, merge_key: str) -> pd.Series:
    cleaned = _clean_string_series(series)
    if merge_key == "gvkey":
        cleaned = cleaned.str.replace(r"\.0$", "", regex=True).str.zfill(6)
    elif merge_key == "cik":
        cleaned = cleaned.str.replace(r"\.0$", "", regex=True).str.zfill(10)
        cleaned = cleaned.mask(cleaned.eq("0" * 10), pd.NA)
    elif merge_key == "cusip8":
        cleaned = cleaned.str[:8]
    return cleaned


def _ensure_merge_key_column(df: pd.DataFrame, merge_key: str) -> pd.DataFrame:
    out = df.copy()
    if merge_key == "cusip8" and "cusip8" not in out.columns and "cusip" in out.columns:
        out["cusip8"] = _clean_string_series(out["cusip"]).str[:8]
    return out


def audit_compustat_schema(
    table_or_path: str | Path | pd.DataFrame,
    *,
    required_groups: Sequence[tuple[str, Sequence[str]]] = COMPUSTAT_REQUIRED_FIELD_GROUPS,
    merge_keys: Sequence[str] = COMPUSTAT_CANDIDATE_MERGE_KEYS,
) -> CompustatSchemaAuditReport:
    """Inspect a Compustat-like schema and return a structured feasibility report."""
    raw, source = _load_frame(table_or_path)
    df = _normalise_columns(raw)
    row_count = len(df)

    candidate_fields = _candidate_field_order(required_groups, merge_keys)
    field_rows: list[dict[str, Any]] = []
    field_non_null: dict[str, int] = {}
    for field in candidate_fields:
        present = field in df.columns
        non_null_count = _field_non_null_count(df, field)
        field_non_null[field] = non_null_count
        field_rows.append(
            {
                "field": field,
                "present": present,
                "non_null_count": non_null_count,
                "non_null_coverage": (float(non_null_count / row_count) if row_count else 0.0),
                "dtype": str(df[field].dtype) if present else None,
            }
        )

    required_rows: list[dict[str, Any]] = []
    for requirement, alternatives in required_groups:
        present_alternatives = tuple(field for field in alternatives if field in df.columns)
        usable_alternatives = tuple(
            field for field in present_alternatives if field_non_null.get(field, 0) > 0
        )
        missing_alternatives = tuple(field for field in alternatives if field not in df.columns)
        required_rows.append(
            {
                "requirement": requirement,
                "alternatives": tuple(alternatives),
                "satisfied": bool(usable_alternatives),
                "present_alternatives": present_alternatives,
                "usable_alternatives": usable_alternatives,
                "missing_alternatives": missing_alternatives,
            }
        )

    parsable_datadate_count = 0
    min_datadate: pd.Timestamp | None = None
    max_datadate: pd.Timestamp | None = None
    if _DATE_COL in df.columns:
        parsed_dates = _coerce_datetime(df[_DATE_COL])
        parsable_mask = parsed_dates.notna()
        parsable_datadate_count = int(parsable_mask.sum())
        if parsable_datadate_count:
            min_datadate = pd.Timestamp(parsed_dates[parsable_mask].min())
            max_datadate = pd.Timestamp(parsed_dates[parsable_mask].max())

    merge_rows: list[dict[str, Any]] = []
    for merge_key in merge_keys:
        present = merge_key in df.columns
        non_null_count = _field_non_null_count(df, merge_key)
        unique_non_null_count = (
            int(_normalise_merge_key(df[merge_key], merge_key).dropna().nunique()) if present else 0
        )
        if present and _DATE_COL in df.columns:
            merge_ready_mask = (_normalise_merge_key(df[merge_key], merge_key).notna()) & (
                _coerce_datetime(df[_DATE_COL]).notna()
            )
            merge_ready_count = int(merge_ready_mask.sum())
        else:
            merge_ready_count = 0
        merge_rows.append(
            {
                "field": merge_key,
                "present": present,
                "non_null_count": non_null_count,
                "non_null_coverage": (float(non_null_count / row_count) if row_count else 0.0),
                "unique_non_null_count": unique_non_null_count,
                "merge_ready_count": merge_ready_count,
                "merge_ready_coverage": (
                    float(merge_ready_count / row_count) if row_count else 0.0
                ),
            }
        )

    return CompustatSchemaAuditReport(
        source=source,
        row_count=row_count,
        column_count=len(df.columns),
        field_coverage=pd.DataFrame(field_rows),
        required_group_status=pd.DataFrame(required_rows),
        merge_key_coverage=pd.DataFrame(merge_rows),
        parsable_datadate_count=parsable_datadate_count,
        min_datadate=min_datadate,
        max_datadate=max_datadate,
    )


def validate_compustat_schema_or_raise(
    table_or_path: str | Path | pd.DataFrame,
    *,
    required_groups: Sequence[tuple[str, Sequence[str]]] = COMPUSTAT_REQUIRED_FIELD_GROUPS,
) -> CompustatSchemaAuditReport:
    """Return the schema report or raise with a clear missing-fields summary."""
    report = audit_compustat_schema(table_or_path, required_groups=required_groups)
    if not report.unsatisfied_required_groups:
        return report

    group_map = {name: tuple(alternatives) for name, alternatives in required_groups}
    missing_bits = [
        f"{group} ({'|'.join(group_map[group])})" for group in report.unsatisfied_required_groups
    ]
    raise ValueError(
        "Compustat schema audit failed; unsatisfied required groups: " f"{', '.join(missing_bits)}."
    )


def prepare_compustat_for_point_in_time_merge(
    table_or_path: str | Path | pd.DataFrame,
    *,
    merge_key: str = "gvkey",
    datadate_col: str = _DATE_COL,
    availability_date_col: str = "availability_date",
    lag_months: int = 6,
) -> pd.DataFrame:
    """Normalise Compustat timing so joins can enforce a conservative lag rule."""
    raw, _ = _load_frame(table_or_path)
    df = _ensure_merge_key_column(_normalise_columns(raw), merge_key)

    missing = [col for col in (merge_key, datadate_col) if col not in df.columns]
    if missing:
        raise KeyError(
            "prepare_compustat_for_point_in_time_merge: missing required column(s) " f"{missing}."
        )

    df[datadate_col] = _coerce_datetime(df[datadate_col])
    bad_datadates = int(df[datadate_col].isna().sum())
    if bad_datadates:
        raise ValueError(
            "prepare_compustat_for_point_in_time_merge: "
            f"{bad_datadates} row(s) have null or unparseable datadate."
        )

    conservative_availability = df[datadate_col] + pd.DateOffset(months=lag_months)
    if availability_date_col in df.columns:
        explicit_availability = _coerce_datetime(df[availability_date_col])
        bad_availability = int(explicit_availability.isna().sum())
        if bad_availability:
            raise ValueError(
                "prepare_compustat_for_point_in_time_merge: "
                f"{bad_availability} row(s) have null or unparseable availability_date."
            )
        availability = pd.concat(
            [
                conservative_availability.rename("conservative"),
                explicit_availability.rename("explicit"),
            ],
            axis=1,
        ).max(axis=1)
    else:
        availability = conservative_availability

    df[merge_key] = _normalise_merge_key(df[merge_key], merge_key)
    df[availability_date_col] = availability

    return df.sort_values([merge_key, availability_date_col, datadate_col]).reset_index(drop=True)


def audit_compustat_point_in_time_merge(
    merged: pd.DataFrame,
    *,
    formation_date_col: str = "date",
    compustat_datadate_col: str = f"{_DEFAULT_PIT_PREFIX}datadate",
    compustat_availability_date_col: str = f"{_DEFAULT_PIT_PREFIX}availability_date",
    raise_on_fail: bool = False,
) -> pd.DataFrame:
    """Check a merged panel for future Compustat dates relative to formation date."""
    required = [
        formation_date_col,
        compustat_datadate_col,
        compustat_availability_date_col,
    ]
    missing = [col for col in required if col not in merged.columns]
    if missing:
        raise KeyError(
            f"audit_compustat_point_in_time_merge: missing required column(s) {missing}."
        )

    formation_date = _coerce_datetime(merged[formation_date_col])
    compustat_datadate = _coerce_datetime(merged[compustat_datadate_col])
    compustat_availability_date = _coerce_datetime(merged[compustat_availability_date_col])

    violations: list[dict[str, Any]] = []
    matched_mask = compustat_datadate.notna()
    for row_index in merged.index[matched_mask & (compustat_datadate > formation_date)]:
        violations.append(
            {
                "row_index": int(row_index),
                "violation_type": "future_datadate",
                "formation_date": formation_date.loc[row_index],
                "compustat_datadate": compustat_datadate.loc[row_index],
                "compustat_availability_date": compustat_availability_date.loc[row_index],
            }
        )
    for row_index in merged.index[matched_mask & (compustat_availability_date > formation_date)]:
        violations.append(
            {
                "row_index": int(row_index),
                "violation_type": "future_availability_date",
                "formation_date": formation_date.loc[row_index],
                "compustat_datadate": compustat_datadate.loc[row_index],
                "compustat_availability_date": compustat_availability_date.loc[row_index],
            }
        )

    out = pd.DataFrame.from_records(
        violations,
        columns=[
            "row_index",
            "violation_type",
            "formation_date",
            "compustat_datadate",
            "compustat_availability_date",
        ],
    )
    if raise_on_fail and not out.empty:
        raise ValueError(
            "audit_compustat_point_in_time_merge found "
            f"{len(out)} violation(s): {sorted(out['violation_type'].unique())}."
        )
    return out


def merge_compustat_point_in_time(
    formation_panel: pd.DataFrame,
    compustat: str | Path | pd.DataFrame,
    *,
    merge_key: str = "gvkey",
    formation_date_col: str = "date",
    datadate_col: str = _DATE_COL,
    availability_date_col: str = "availability_date",
    lag_months: int = 6,
    compustat_cols: Sequence[str] | None = None,
    prefix: str = _DEFAULT_PIT_PREFIX,
) -> pd.DataFrame:
    """As-of merge Compustat rows onto formation months with a conservative lag."""
    if formation_date_col not in formation_panel.columns:
        raise KeyError(
            f"merge_compustat_point_in_time: formation panel is missing '{formation_date_col}'."
        )

    left = _ensure_merge_key_column(_normalise_columns(formation_panel), merge_key)
    if merge_key not in left.columns:
        raise KeyError(
            f"merge_compustat_point_in_time: formation panel is missing merge key '{merge_key}'."
        )

    left[formation_date_col] = _coerce_datetime(left[formation_date_col])
    bad_formation_dates = int(left[formation_date_col].isna().sum())
    if bad_formation_dates:
        raise ValueError(
            "merge_compustat_point_in_time: "
            f"{bad_formation_dates} row(s) have null or unparseable formation dates."
        )

    left[merge_key] = _normalise_merge_key(left[merge_key], merge_key)
    left = left.reset_index(drop=True).copy()
    left["_left_order"] = left.index

    right = prepare_compustat_for_point_in_time_merge(
        compustat,
        merge_key=merge_key,
        datadate_col=datadate_col,
        availability_date_col=availability_date_col,
        lag_months=lag_months,
    )

    keep_cols = (
        [merge_key, datadate_col, availability_date_col]
        if compustat_cols is None
        else [merge_key, datadate_col, availability_date_col, *compustat_cols]
    )
    keep_cols = list(dict.fromkeys(keep_cols))
    missing_comp_cols = [col for col in keep_cols if col not in right.columns]
    if missing_comp_cols:
        raise KeyError(
            "merge_compustat_point_in_time: requested Compustat column(s) missing from the "
            f"right frame: {missing_comp_cols}."
        )

    right = right[keep_cols].copy()
    rename_map = {col: f"{prefix}{col}" for col in right.columns if col != merge_key}
    right = right.rename(columns=rename_map)

    eligible_left = left[left[merge_key].notna()].copy()
    ineligible_left = left[left[merge_key].isna()].copy()

    if not eligible_left.empty and not right.empty:
        eligible_left = eligible_left.sort_values([formation_date_col, merge_key]).reset_index(
            drop=True
        )
        right = right.sort_values(
            [f"{prefix}{availability_date_col}", merge_key, f"{prefix}{datadate_col}"]
        ).reset_index(drop=True)
        merged_eligible = pd.merge_asof(
            eligible_left,
            right,
            by=merge_key,
            left_on=formation_date_col,
            right_on=f"{prefix}{availability_date_col}",
            direction="backward",
        )
    else:
        merged_eligible = eligible_left.copy()
        for col in rename_map.values():
            merged_eligible[col] = pd.Series([pd.NA] * len(merged_eligible))

    for col in rename_map.values():
        if col not in ineligible_left.columns:
            dtype = merged_eligible[col].dtype if col in merged_eligible.columns else "object"
            ineligible_left[col] = pd.Series(index=ineligible_left.index, dtype=dtype)

    if merged_eligible.empty:
        merged = ineligible_left.copy()
    elif ineligible_left.empty:
        merged = merged_eligible.copy()
    else:
        merged = pd.concat([merged_eligible, ineligible_left], ignore_index=True, sort=False)
    merged = merged.sort_values("_left_order").drop(columns="_left_order").reset_index(drop=True)

    audit_compustat_point_in_time_merge(
        merged,
        formation_date_col=formation_date_col,
        compustat_datadate_col=f"{prefix}{datadate_col}",
        compustat_availability_date_col=f"{prefix}{availability_date_col}",
        raise_on_fail=True,
    )
    return merged


__all__ = [
    "COMPUSTAT_CANDIDATE_MERGE_KEYS",
    "COMPUSTAT_REQUIRED_FIELD_GROUPS",
    "CompustatSchemaAuditReport",
    "audit_compustat_point_in_time_merge",
    "audit_compustat_schema",
    "merge_compustat_point_in_time",
    "prepare_compustat_for_point_in_time_merge",
    "validate_compustat_schema_or_raise",
]
