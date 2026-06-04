"""Synthetic smoke test for the cost/universe robustness audit runner.

Pure pandas (no torch/xgboost): builds a tiny panel + saved prediction parquets,
runs the audit end-to-end, and checks the three summary CSVs + the no-leak,
no-retrain, availability-gated behaviour the benchmark transition requires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlfinance.run.audit_costs_universe import run_audit_costs_universe
from mlfinance.run.linear_benchmarks import _split_id

_FINANCIAL_SIC = 6020
_NONFIN_SICS = [1040, 2080, 3570, 3674, 7372]


def _build(tmp_path: Path) -> OmegaConf:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-31", "2021-06-30", freq="ME")
    permnos = list(range(10001, 10013))  # 12 stocks
    sic = {
        p: (_FINANCIAL_SIC if i % 3 == 0 else _NONFIN_SICS[i % len(_NONFIN_SICS)])
        for i, p in enumerate(permnos)
    }

    panel_rows: list[dict] = []
    pred_rows: list[dict] = []
    for date in dates:
        for p in permnos:
            realized = float(rng.normal(0.0, 0.05))
            panel_rows.append(
                {"permno": p, "date": date, "siccd": sic[p], "target_ret_fwd_1m": realized}
            )
            for model in ("ridge", "xgboost"):
                pred_rows.append(
                    {
                        "permno": p,
                        "date": date,
                        "target_ret_fwd_1m": realized,
                        "prediction": float(rng.normal()),
                        "model_name": model,
                        "split_id": "placeholder",
                    }
                )

    panel_path = tmp_path / "model_panel.parquet"
    pd.DataFrame(panel_rows).to_parquet(panel_path, index=False)

    cfg = OmegaConf.create(
        {
            "data": {
                "panel_path": str(panel_path),
                "target_col": "target_ret_fwd_1m",
                "date_col": "date",
                "permno_col": "permno",
            },
            "splits": {"locked": {"test_start": "2021-01-31", "test_end": "2021-12-31"}},
            "portfolio": {
                "n_buckets": 4,
                "long_bucket": 4,
                "short_bucket": 1,
                "weighting": "EW",
            },
            "costs": {"main_bps": 10, "bps_grid": [0, 10, 25]},
            "universe": {
                "variants": {
                    "all": {},
                    "ex_financials": {"sic_exclude": [[6000, 6999]]},
                    "tradable": {
                        "price_min": 5.0,
                        "shrcd_keep": [10, 11],
                        "exchcd_keep": [1, 2, 3],
                    },
                },
                "min_stocks_per_month": 0,
            },
            "paths": {
                "predictions": str(tmp_path / "preds"),
                "backtests": str(tmp_path / "bt"),
                "metrics": str(tmp_path / "metrics"),
            },
        }
    )

    split_id = _split_id(cfg)
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.DataFrame(pred_rows)
    preds["split_id"] = split_id
    for model, frame in preds.groupby("model_name"):
        frame.to_parquet(preds_dir / f"{model}__{split_id}.parquet", index=False)

    return cfg


def test_audit_runner_writes_three_summaries(tmp_path: Path) -> None:
    cfg = _build(tmp_path)
    out = run_audit_costs_universe(cfg)
    for key in (
        "universe_audit_summary",
        "prediction_manifest",
        "cost_robustness_summary",
        "universe_robustness_summary",
        "universe_filter_report",
    ):
        assert out[key].exists(), key

    audit = pd.read_csv(out["universe_audit_summary"])
    avail = dict(zip(audit["field"], audit["present_in_panel"], strict=True))
    assert bool(avail["siccd"]) is True
    assert bool(avail["prc"]) is False  # absent -> infeasible, documented


def test_cost_grid_keeps_gross_and_weakly_lowers_net(tmp_path: Path) -> None:
    cfg = _build(tmp_path)
    out = run_audit_costs_universe(cfg)
    cost = pd.read_csv(out["cost_robustness_summary"])
    assert set(cost["cost_bps"]) == {0, 10, 25}
    for _, g in cost.groupby("model_name"):
        g = g.sort_values("cost_bps")
        assert g["gross_sharpe"].nunique() == 1  # cost never changes gross
        assert (g["net_ann_return"].diff().dropna() <= 1e-9).all()  # ++cost -> --net


def test_universe_variants_filter_and_tradable_noops(tmp_path: Path) -> None:
    cfg = _build(tmp_path)
    out = run_audit_costs_universe(cfg)
    uni = pd.read_csv(out["universe_robustness_summary"])
    assert set(uni["universe"]) == {"all", "ex_financials", "tradable"}
    assert {
        "filters_requested",
        "filters_applied",
        "filters_skipped",
        "is_noop_universe",
    }.issubset(uni.columns)

    n = {v: uni.loc[uni["universe"] == v, "n_rows"].iloc[0] for v in uni["universe"].unique()}
    assert n["ex_financials"] < n["all"]  # financials dropped via SICCD
    assert n["tradable"] == n["all"]  # price/shrcd screens skip (fields absent)

    all_rows = uni.loc[uni["universe"] == "all"]
    assert set(all_rows["filters_requested"]) == {0}
    assert set(all_rows["filters_applied"]) == {0}
    assert set(all_rows["filters_skipped"]) == {0}
    assert set(all_rows["is_noop_universe"]) == {False}

    tradable_rows = uni.loc[uni["universe"] == "tradable"]
    assert set(tradable_rows["filters_requested"]) == {3}
    assert set(tradable_rows["filters_applied"]) == {0}
    assert set(tradable_rows["filters_skipped"]) == {3}
    assert set(tradable_rows["is_noop_universe"]) == {True}

    report = pd.read_csv(out["universe_filter_report"])
    trad = report[report["variant"] == "tradable"]
    assert len(trad) >= 1 and (~trad["applied"]).all()  # screen attempted, all skipped


def test_prediction_manifest_lists_loaded_files(tmp_path: Path) -> None:
    cfg = _build(tmp_path)
    out = run_audit_costs_universe(cfg)
    manifest = pd.read_csv(out["prediction_manifest"])

    split_id = _split_id(cfg)
    expected = pd.DataFrame(
        [
            {
                "filename": f"{model}__{split_id}.parquet",
                "model_name": model,
                "row_count": 72,
                "min_date": "2021-01-31",
                "max_date": "2021-06-30",
                "split_id": split_id,
            }
            for model in ("ridge", "xgboost")
        ]
    ).sort_values("model_name", ignore_index=True)

    pd.testing.assert_frame_equal(manifest, expected)


def test_duplicate_panel_keys_raise_value_error(tmp_path: Path) -> None:
    cfg = _build(tmp_path)
    panel_path = Path(cfg.data.panel_path)
    panel = pd.read_parquet(panel_path)
    duplicate = panel.iloc[[0]].copy()
    pd.concat([panel, duplicate], ignore_index=True).to_parquet(panel_path, index=False)

    with pytest.raises(ValueError, match=r"duplicate key\(s\).*permno.*date"):
        run_audit_costs_universe(cfg)
