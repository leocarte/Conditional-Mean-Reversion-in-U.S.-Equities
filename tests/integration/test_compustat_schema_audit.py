"""Synthetic smoke test for the feature-audit cluster data-schema audit runner.

Pure pandas (no torch/xgboost): builds a tiny Compustat (quarterly YTD flow
shape, no balance sheet) + CRSP parquet, runs the audit, and checks the reports +
the feasibility split (income/cash-flow feasible, balance-sheet not; CRSP
tradability scope limited; cusip8 link overlap counted).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from mlfinance.run.compustat_schema_audit import run_compustat_schema_audit


def _build(tmp_path: Path) -> OmegaConf:
    # Compustat quarterly YTD flow shape: income/cash-flow present, NO balance sheet.
    comp = pd.DataFrame(
        {
            "gvkey": ["001045", "001078", "001234"],
            "datadate": ["2020-03-31", "2020-06-30", "2020-09-30"],
            "cusip": ["12345678", "23456789", "34567890"],
            "fyearq": [2020, 2020, 2020],
            "saley": [10.0, 20.0, 30.0],
            "cogsy": [6.0, 12.0, 18.0],
            "niy": [1.0, 2.0, 3.0],
            "oancfy": [2.0, 4.0, 6.0],
            "capxy": [1.0, 1.0, 1.0],
            # no at/ceq/dltt/dlc/che/txditc/pstk -> balance-sheet groups must be missing
        }
    )
    comp_path = tmp_path / "CompFirmCharac.parquet"
    comp.to_parquet(comp_path, index=False)

    # CRSP without prc/shrout/shrcd/exchcd; CUSIP present (uppercase like the raw file).
    crsp = pd.DataFrame(
        {
            "PERMNO": [10001, 10002, 10003],
            "MthCalDt": ["2020-03-31", "2020-03-31", "2020-03-31"],
            "MthRet": [0.01, 0.02, 0.03],
            "CUSIP": ["12345678", "23456789", "99999999"],
        }
    )
    crsp_path = tmp_path / "monthly_crsp.parquet"
    crsp.to_parquet(crsp_path, index=False)

    return OmegaConf.create(
        {
            "data": {
                "paths": {
                    "compustat_quarterly": str(comp_path),
                    "crsp_monthly": str(crsp_path),
                }
            },
            "paths": {"metrics": str(tmp_path / "metrics")},
        }
    )


def test_compustat_audit_writes_all_reports(tmp_path: Path) -> None:
    out = run_compustat_schema_audit(_build(tmp_path))
    for key in (
        "compustat_field_coverage",
        "compustat_required_groups",
        "compustat_merge_keys",
        "compustat_all_columns_coverage",
        "crsp_tradability_fields",
        "link_overlap",
        "summary",
    ):
        assert out[key].exists(), key

    coverage = pd.read_csv(out["compustat_all_columns_coverage"])
    assert len(coverage) == 9  # every column listed


def test_compustat_summary_splits_flow_vs_balance_sheet(tmp_path: Path) -> None:
    out = run_compustat_schema_audit(_build(tmp_path))
    summary = pd.read_csv(out["summary"]).iloc[0]

    assert int(summary["compustat_total_columns"]) == 9
    # Income/cash-flow groups present -> feasible; balance-sheet absent -> not feasible.
    assert bool(summary["income_cashflow_features_feasible"]) is True
    assert bool(summary["assets_scaled_features_feasible"]) is False
    assert bool(summary["full_balance_sheet_features_feasible"]) is False
    # No prc/shrout/shrcd/exchcd in CRSP -> robustness scope is limited.
    assert bool(summary["crsp_scope_limited"]) is True
    # cusip8 overlap: {12345678, 23456789} are common -> 2.
    assert int(summary["cusip8_link_overlap"]) == 2


def test_compustat_balance_sheet_groups_reported_missing(tmp_path: Path) -> None:
    out = run_compustat_schema_audit(_build(tmp_path))
    groups = pd.read_csv(out["compustat_required_groups"]).set_index("requirement")
    for bs_group in ("total_assets", "common_equity", "debt_long", "cash"):
        assert bool(groups.loc[bs_group, "satisfied"]) is False
    for flow_group in ("sales", "earnings", "op_cash_flow", "capex"):
        assert bool(groups.loc[flow_group, "satisfied"]) is True


def test_compustat_cusip8_overlap_ignores_blank_and_malformed_keys(tmp_path: Path) -> None:
    comp = pd.DataFrame(
        {
            "gvkey": ["001045", "001078", "001234", "001999", "002000"],
            "datadate": ["2020-03-31"] * 5,
            "cusip": ["12345678", "", "   ", None, "123"],
            "fyearq": [2020] * 5,
            "saley": [10.0] * 5,
            "cogsy": [6.0] * 5,
            "niy": [1.0] * 5,
            "oancfy": [2.0] * 5,
            "capxy": [1.0] * 5,
        }
    )
    comp_path = tmp_path / "CompFirmCharac.parquet"
    comp.to_parquet(comp_path, index=False)

    crsp = pd.DataFrame(
        {
            "PERMNO": [10001, 10002, 10003, 10004, 10005],
            "MthCalDt": ["2020-03-31"] * 5,
            "MthRet": [0.01, 0.02, 0.03, 0.04, 0.05],
            "CUSIP": ["12345678", "", "   ", None, "123"],
        }
    )
    crsp_path = tmp_path / "monthly_crsp.parquet"
    crsp.to_parquet(crsp_path, index=False)

    cfg = OmegaConf.create(
        {
            "data": {
                "paths": {
                    "compustat_quarterly": str(comp_path),
                    "crsp_monthly": str(crsp_path),
                }
            },
            "paths": {"metrics": str(tmp_path / "metrics")},
        }
    )

    out = run_compustat_schema_audit(cfg)
    link = pd.read_csv(out["link_overlap"]).iloc[0]

    assert int(link["compustat_unique_cusip8"]) == 1
    assert int(link["crsp_unique_cusip8"]) == 1
    assert int(link["overlap_cusip8"]) == 1
