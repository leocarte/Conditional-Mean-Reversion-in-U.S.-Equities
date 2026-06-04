"""feature-audit cluster data-schema audit.

Runs the feature-audit schema-feasibility audit on the REAL Compustat + CRSP files
(paths from ``configs/main.yaml``), reusing
``mlfinance.data.compustat_schema.audit_compustat_schema`` with a feature-audit
candidate list tuned to the actual ``CompFirmCharac`` schema (a Compustat
*quarterly year-to-date* flow extract, ``*y`` suffix). The report therefore
separates:

  * income-statement / cash-flow groups (present here -> features feasible), from
  * balance-sheet groups (absent here -> assets/equity-scaled ratios remain
    unavailable in this reviewed extract; reported as missing, never faked).

It also reports CRSP tradability-field availability under the provided Monthly
CRSP parquet and the cusip8 Compustat<->CRSP overlap (the de-facto key; no CCM
linktable is shipped).
No features, no model training, no target/split/backtest changes. Outputs are
written to git-ignored ``data/outputs/metrics/``.

Performance: Compustat is read once (the schema audit, the cusip8 set, and the
all-column coverage all come from one DataFrame); CRSP is read cusip-column-only
(projected), so its heavy return/string columns are never decoded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig

from mlfinance.data.compustat_schema import audit_compustat_schema
from mlfinance.data.loaders import _read_table
from mlfinance.utils.io import read_parquet

logger = logging.getLogger(__name__)

# feature-audit accounting groups tuned to the real CompFirmCharac (Compustat quarterly
# YTD flow extract). Flow groups list the actual ``*y`` names first; balance-sheet
# groups list what feature-audit ratios WANT (absent here -> reported missing, not faked).
COMPUSTAT_REQUIRED_FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fiscal_period", ("fyearq", "fyear")),
    ("sales", ("saley", "revty", "sale")),
    ("cogs", ("cogsy", "cogs")),
    ("earnings", ("niy", "iby", "ni", "ib")),
    ("operating_income", ("oiadpy", "oibdpy")),
    ("op_cash_flow", ("oancfy", "oancf")),
    ("capex", ("capxy", "capx")),
    ("rd", ("xrdy", "xrd")),
    ("sga", ("xsgay", "xsga")),
    ("dividends", ("dvy", "dvt")),
    # Balance-sheet stocks: needed for ROA / leverage / book-to-market / asset growth.
    ("total_assets", ("atq", "at")),
    ("common_equity", ("ceqq", "ceq", "seqq", "seq")),
    ("debt_long", ("dlttq", "dltt")),
    ("debt_current", ("dlcq", "dlc")),
    ("cash", ("cheq", "che")),
    ("deferred_tax", ("txditcq", "txditc")),
    ("preferred_stock", ("pstkq", "pstkrv", "pstkl", "pstk")),
)
COMPUSTAT_FLOW_GROUPS: tuple[str, ...] = (
    "sales",
    "cogs",
    "earnings",
    "operating_income",
    "op_cash_flow",
    "capex",
)
COMPUSTAT_BALANCE_SHEET_GROUPS: tuple[str, ...] = (
    "total_assets",
    "common_equity",
    "debt_long",
    "debt_current",
    "cash",
    "deferred_tax",
    "preferred_stock",
)

# CRSP fields required for price/share-code/exchange/microcap screens.
CRSP_TRADABILITY_FIELDS: dict[str, str] = {
    "prc": "price filter (abs(prc) >= 5)",
    "shrout": "market-cap / microcap (with prc)",
    "shrcd": "common-shares (shrcd in {10, 11})",
    "exchcd": "exchange (exchcd in {1, 2, 3})",
    "vol": "volume / liquidity proxy",
}


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _schema_names(path: str) -> list[str]:
    """Parquet column names from the footer (no data pages read)."""
    return [str(name) for name in pq.read_schema(str(path)).names]


def _cusip8_from_series(cusip: pd.Series) -> set[str]:
    """Return strict non-empty 8-character CUSIP keys for overlap audits."""
    cleaned = cusip.astype("string").str.strip().str.upper().str[:8]
    cleaned = cleaned.mask(cleaned.eq(""), pd.NA).mask(cleaned.eq("<NA>"), pd.NA)
    cleaned = cleaned[cleaned.str.len() == 8]
    return set(cleaned.dropna().unique())


def _crsp_cusip8(crsp_path: str) -> set[str]:
    """Unique 8-char CUSIPs from CRSP, reading only the cusip column (projected)."""
    col = next((n for n in _schema_names(crsp_path) if n.lower() in ("hdrcusip", "cusip")), None)
    if col is None:
        return set()
    return _cusip8_from_series(read_parquet(crsp_path, columns=[col])[col])


def _compustat_cusip8(comp_df: pd.DataFrame) -> set[str]:
    lower = {str(c).lower(): c for c in comp_df.columns}
    col = lower.get("cusip8") or lower.get("cusip")
    return _cusip8_from_series(comp_df[col]) if col is not None else set()


def _all_columns_coverage(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Non-null coverage + dtype for EVERY column (feature-audit feature discovery)."""
    n = len(comp_df)
    rows = [
        {
            "field": str(col),
            "dtype": str(comp_df[col].dtype),
            "non_null_coverage": round(float(comp_df[col].notna().mean()), 4) if n else 0.0,
        }
        for col in comp_df.columns
    ]
    return pd.DataFrame(rows).sort_values("field").reset_index(drop=True)


def run_compustat_schema_audit(cfg: DictConfig) -> dict[str, Path]:
    """Audit real Compustat + CRSP schemas/coverage; write reports. No modeling."""
    metrics_dir = Path(cfg.paths.metrics)
    comp_path = str(cfg.data.paths.compustat_quarterly)
    crsp_path = str(cfg.data.paths.crsp_monthly)

    # --- Compustat schema feasibility (reuse the audit utility, feature-audit groups) ---
    comp_df = _read_table(comp_path)
    report = audit_compustat_schema(comp_df, required_groups=COMPUSTAT_REQUIRED_FIELD_GROUPS)
    fc_path = _write_csv(
        report.field_coverage, metrics_dir / "compustat_compustat_field_coverage.csv"
    )
    rg_path = _write_csv(
        report.required_group_status, metrics_dir / "compustat_compustat_required_groups.csv"
    )
    mk_path = _write_csv(
        report.merge_key_coverage, metrics_dir / "compustat_compustat_merge_keys.csv"
    )
    cov_path = _write_csv(
        _all_columns_coverage(comp_df), metrics_dir / "compustat_compustat_all_columns_coverage.csv"
    )

    status = dict(
        zip(
            report.required_group_status["requirement"],
            report.required_group_status["satisfied"],
            strict=True,
        )
    )
    flow_satisfied = sorted(g for g in COMPUSTAT_FLOW_GROUPS if status.get(g))
    bs_satisfied = sorted(g for g in COMPUSTAT_BALANCE_SHEET_GROUPS if status.get(g))
    income_cashflow_feasible = all(
        status.get(g, False) for g in ("sales", "earnings", "op_cash_flow")
    )
    assets_scaled_features_feasible = bool(status.get("total_assets", False))
    full_balance_sheet_features_feasible = all(
        status.get(g, False) for g in COMPUSTAT_BALANCE_SHEET_GROUPS
    )
    logger.info(
        "Compustat: %d rows, %d cols; flow groups=%s; balance-sheet groups=%s",
        report.row_count,
        report.column_count,
        flow_satisfied or "(none)",
        bs_satisfied or "(none)",
    )

    # --- CRSP tradability-field availability on the provided raw schema ---
    crsp_names = {n.lower() for n in _schema_names(crsp_path)}
    crsp_rows = [
        {"field": field, "present_in_raw_crsp": field in crsp_names, "enables": enables}
        for field, enables in CRSP_TRADABILITY_FIELDS.items()
    ]
    crsp_fields_path = _write_csv(
        pd.DataFrame(crsp_rows), metrics_dir / "compustat_crsp_tradability_fields.csv"
    )

    # --- cusip8 link overlap (the de-facto Compustat<->CRSP key; no CCM table) ---
    comp_keys = _compustat_cusip8(comp_df)
    crsp_keys = _crsp_cusip8(crsp_path)
    overlap = comp_keys & crsp_keys
    link_rows = [
        {
            "compustat_unique_cusip8": len(comp_keys),
            "crsp_unique_cusip8": len(crsp_keys),
            "overlap_cusip8": len(overlap),
            "compustat_covered_frac": round(len(overlap) / len(comp_keys), 4) if comp_keys else 0.0,
            "crsp_covered_frac": round(len(overlap) / len(crsp_keys), 4) if crsp_keys else 0.0,
        }
    ]
    link_path = _write_csv(pd.DataFrame(link_rows), metrics_dir / "compustat_link_overlap.csv")

    # --- Headline feasibility summary ---
    present_tradability = sorted(f for f in CRSP_TRADABILITY_FIELDS if f in crsp_names)
    summary_rows = [
        {
            "compustat_rows": report.row_count,
            "compustat_total_columns": report.column_count,
            "compustat_datadate_min": report.min_datadate,
            "compustat_datadate_max": report.max_datadate,
            "income_cashflow_features_feasible": income_cashflow_feasible,
            "assets_scaled_features_feasible": assets_scaled_features_feasible,
            "full_balance_sheet_features_feasible": full_balance_sheet_features_feasible,
            "flow_groups_satisfied": "|".join(flow_satisfied) or "(none)",
            "balance_sheet_groups_satisfied": "|".join(bs_satisfied) or "(none)",
            "crsp_tradability_fields_present": "|".join(present_tradability) or "(none)",
            "crsp_scope_limited": len(present_tradability) < len(CRSP_TRADABILITY_FIELDS),
            "cusip8_link_overlap": len(overlap),
            "cusip8_link_feasible": len(overlap) > 0,
        }
    ]
    summary_path = _write_csv(
        pd.DataFrame(summary_rows), metrics_dir / "compustat_data_schema_summary.csv"
    )

    return {
        "compustat_field_coverage": fc_path,
        "compustat_required_groups": rg_path,
        "compustat_merge_keys": mk_path,
        "compustat_all_columns_coverage": cov_path,
        "crsp_tradability_fields": crsp_fields_path,
        "link_overlap": link_path,
        "summary": summary_path,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    outputs = run_compustat_schema_audit(cfg)
    for name, path in outputs.items():
        logger.info("Wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
