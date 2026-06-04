"""Synthetic smoke tests for the lightweight feature-audit flow-Compustat audit runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from mlfinance.run.compustat_integration_smoke import (
    FORBIDDEN_FEATURE_TOKENS,
    KEY_FLOW_FEATURES,
    run_compustat_integration_smoke,
)


def _write_synthetic_crsp(path: Path) -> None:
    dates = pd.date_range("2020-01-31", "2020-11-30", freq="ME")
    rows: list[dict[str, object]] = []
    for permno, cusip_root, base_ret in (
        (10001, "11111111", 0.010),
        (10002, "22222222", 0.020),
        (10003, "33333333", 0.030),
        (10004, "", 0.040),
    ):
        for step, date in enumerate(dates):
            rows.append(
                {
                    "PERMNO": permno,
                    "PERMCO": permno,
                    "HdrCUSIP": f"{cusip_root}9" if cusip_root else "",
                    "Ticker": f"T{permno}",
                    "SICCD": 3571,
                    "NAICS": 334111,
                    "MthCalDt": date,
                    "MthRet": base_ret + 0.001 * step,
                    "sprtrn": 0.001 + 0.0001 * step,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_synthetic_compustat(path: Path) -> None:
    pd.DataFrame(
        {
            "gvkey": ["001000", "001000", "002000", "002000"],
            "cusip": ["111111119", "111111119", "222222229", "222222229"],
            "datadate": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
            "fyearq": [2020, 2020, 2020, 2020],
            "fqtr": [1, 2, 1, 2],
            "saley": [100.0, 999.0, 200.0, 888.0],
            "cogsy": [60.0, 555.0, 120.0, 444.0],
            "niy": [10.0, 111.0, 20.0, 222.0],
            "oancfy": [8.0, 88.0, 16.0, 176.0],
            "capxy": [3.0, 33.0, 4.0, 44.0],
            "xrdy": [2.0, 22.0, 5.0, 55.0],
            "xsgay": [12.0, 144.0, 18.0, 216.0],
        }
    ).to_parquet(path, index=False)


def _build(tmp_path: Path) -> OmegaConf:
    crsp_path = tmp_path / "monthly_crsp.parquet"
    compustat_path = tmp_path / "CompFirmCharac.parquet"
    _write_synthetic_crsp(crsp_path)
    _write_synthetic_compustat(compustat_path)
    return OmegaConf.create(
        {
            "data": {
                "panel_path": str(tmp_path / "panel.parquet"),
                "target_col": "target_ret_fwd_1m",
                "robustness_target_col": "target_excess_ret_fwd_1m",
                "date_col": "date",
                "permno_col": "permno",
                "ret_col": "ret",
                "market_col": "sprtrn",
                "use_compustat": False,
                "use_regime_features": False,
                "paths": {
                    "crsp_monthly": str(crsp_path),
                    "compustat_quarterly": str(compustat_path),
                },
            },
            "splits": {
                "locked": {
                    "train_start": "2020-03-31",
                    "train_end": "2020-06-30",
                    "val_start": "2020-07-31",
                    "val_end": "2020-08-31",
                    "test_start": "2020-09-30",
                    "test_end": "2020-10-31",
                }
            },
            "features": {
                "return_features": {"enabled": True, "min_history_months": 1},
                "compustat_features": {"enabled": False, "conservative_lag_months": 6},
            },
            "paths": {"metrics": str(tmp_path / "metrics")},
        }
    )


def test_smoke_writes_diagnostics_and_passes_guardrails(tmp_path: Path) -> None:
    out = run_compustat_integration_smoke(_build(tmp_path))
    for key in ("summary", "feature_coverage", "examples"):
        assert out[key].exists(), key

    summary = pd.read_csv(out["summary"]).iloc[0]
    assert int(summary["enabled_rows"]) == int(summary["return_only_rows"])
    assert bool(summary["targets_identical"]) is True
    assert bool(summary["split_masks_identical"]) is True
    assert int(summary["pit_violations"]) == 0
    assert int(summary["rows_with_any_compustat_feature"]) > 0
    assert int(summary["rows_missing_panel_cusip8"]) > 0
    assert int(summary["rows_without_compustat_key_match"]) > 0
    assert int(summary["rows_with_compustat_key_but_no_pit_record"]) > 0


def test_smoke_summary_distinguishes_missing_key_no_link_and_no_pit(tmp_path: Path) -> None:
    out = run_compustat_integration_smoke(_build(tmp_path))
    summary = pd.read_csv(out["summary"]).iloc[0]

    assert int(summary["rows_without_pit_available_compustat_record"]) == (
        int(summary["rows_missing_panel_cusip8"])
        + int(summary["rows_without_compustat_key_match"])
        + int(summary["rows_with_compustat_key_but_no_pit_record"])
    )
    assert int(summary["rows_without_pit_available_compustat_record"]) < int(
        summary["enabled_rows"]
    )
    assert float(summary["share_with_any_compustat_feature"]) > 0.0


def test_smoke_coverage_lists_key_features_and_no_forbidden(tmp_path: Path) -> None:
    out = run_compustat_integration_smoke(_build(tmp_path))

    coverage = pd.read_csv(out["feature_coverage"])
    key_in_report = set(coverage.loc[coverage["is_key_feature"], "feature"])
    assert set(KEY_FLOW_FEATURES).issubset(key_in_report)
    for feature in coverage["feature"]:
        assert not any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS)

    summary = pd.read_csv(out["summary"]).iloc[0]
    assert summary["forbidden_feature_columns"] == "(none)"


def test_smoke_examples_have_matched_and_unmatched_rows(tmp_path: Path) -> None:
    out = run_compustat_integration_smoke(_build(tmp_path))
    examples = pd.read_csv(out["examples"])

    assert set(examples["match_status"]).issubset({"matched", "unmatched"})
    assert (examples["match_status"] == "matched").any()
    assert (examples["match_status"] == "unmatched").any()
    assert {"matched", "no_compustat_key_match", "linked_but_no_pit_record"}.issubset(
        set(examples["diagnostic_status"])
    )
    assert "missing_panel_cusip8" in set(examples["diagnostic_status"])
