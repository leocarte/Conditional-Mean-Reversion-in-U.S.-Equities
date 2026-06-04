"""Active data helpers for the conditional mean-reversion project."""

from __future__ import annotations

from mlfinance.data.compustat_schema import (
    COMPUSTAT_CANDIDATE_MERGE_KEYS,
    COMPUSTAT_REQUIRED_FIELD_GROUPS,
    CompustatSchemaAuditReport,
    audit_compustat_point_in_time_merge,
    audit_compustat_schema,
    merge_compustat_point_in_time,
    prepare_compustat_for_point_in_time_merge,
    validate_compustat_schema_or_raise,
)
from mlfinance.data.linktable import link_compustat_to_crsp_by_cusip
from mlfinance.data.loaders import (
    JKP_TO_FF5_MAP,
    load_chen_zimmerman,
    load_compustat_quarterly,
    load_crsp_monthly,
    load_ff5_umd_from_jkp,
    load_jkp,
)
from mlfinance.data.preprocessing import (
    cross_sectional_rank,
    dedupe_jkp_cz,
    impute_missing_with_indicator,
    winsorize_returns,
)

__all__ = [
    "COMPUSTAT_CANDIDATE_MERGE_KEYS",
    "COMPUSTAT_REQUIRED_FIELD_GROUPS",
    "CompustatSchemaAuditReport",
    "JKP_TO_FF5_MAP",
    "audit_compustat_point_in_time_merge",
    "audit_compustat_schema",
    "cross_sectional_rank",
    "dedupe_jkp_cz",
    "impute_missing_with_indicator",
    "link_compustat_to_crsp_by_cusip",
    "load_chen_zimmerman",
    "load_compustat_quarterly",
    "load_crsp_monthly",
    "load_ff5_umd_from_jkp",
    "load_jkp",
    "merge_compustat_point_in_time",
    "prepare_compustat_for_point_in_time_merge",
    "validate_compustat_schema_or_raise",
    "winsorize_returns",
]
