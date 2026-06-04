"""Synthetic smoke test for same-protocol walk-forward classical baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlfinance.run.classical_baselines import (
    BASELINE_SOURCE,
    DIFF_COMPARISON_PAIRS,
    OUTPUT_PREFIX,
    run_classical_baselines,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path: Path):
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "main.yaml")
    overrides = OmegaConf.create(
        {
            "data": {
                "panel_path": str(tmp_path / "model_panel.parquet"),
            },
            "paths": {
                "predictions": str(tmp_path / "outputs" / "predictions"),
                "backtests": str(tmp_path / "outputs" / "backtests"),
                "metrics": str(tmp_path / "outputs" / "metrics"),
            },
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _write_model_panel(path: Path) -> None:
    dates = pd.date_range("2016-01-31", "2025-12-31", freq="ME")
    permnos = list(range(10001, 10026))
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for d_idx, date in enumerate(dates):
        date_wave = 0.002 * np.sin(d_idx / 4.0)
        for p_idx, permno in enumerate(permnos):
            centered = p_idx - (len(permnos) - 1) / 2.0
            ret_1m = 0.01 * centered + date_wave
            mom_2_12 = -0.006 * centered + 0.001 * np.cos(d_idx / 6.0)
            rows.append(
                {
                    "date": date,
                    "permno": permno,
                    "target_ret_fwd_1m": -0.35 * ret_1m + 0.20 * mom_2_12 + 0.0005 * d_idx,
                    "target_excess_ret_fwd_1m": -0.25 * ret_1m + 0.15 * mom_2_12,
                    "ret_1m": ret_1m,
                    "mom_2_12": mom_2_12,
                    "vol_12m": 0.15 + 0.001 * p_idx,
                    "prediction": 999.0,
                    "realized_date": date + pd.offsets.MonthEnd(1),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_walkforward_main_backtests(path: Path) -> None:
    dates = pd.date_range("2017-01-31", "2024-12-31", freq="ME")
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for idx, date in enumerate(dates):
        rows.extend(
            [
                {
                    "date": date,
                    "model_name": "flow_compustat__mlp",
                    "cutoff": 0.10,
                    "cost_bps": 10.0,
                    "net_return": 0.020 + 0.0002 * idx,
                },
                {
                    "date": date,
                    "model_name": "flow_compustat__xgboost",
                    "cutoff": 0.10,
                    "cost_bps": 10.0,
                    "net_return": 0.015 + 0.00015 * idx,
                },
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_classical_baselines_runner_writes_isolated_outputs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_model_panel(Path(cfg.data.panel_path))
    _write_walkforward_main_backtests(Path(cfg.paths.backtests) / "walkforward_backtests.parquet")

    outputs = run_classical_baselines(cfg)

    for key in ("predictions", "backtests", "summary", "difference_tests", "preview"):
        assert outputs[key].exists(), key

    assert outputs["predictions"].name == f"{OUTPUT_PREFIX}_predictions.parquet"
    assert outputs["backtests"].name == f"{OUTPUT_PREFIX}_backtests.parquet"
    assert outputs["summary"].name == f"{OUTPUT_PREFIX}_summary.csv"
    assert outputs["difference_tests"].name == f"{OUTPUT_PREFIX}_difference_tests.csv"
    assert not (Path(cfg.paths.predictions) / "walkforward_predictions.parquet").exists()
    assert not (Path(cfg.paths.metrics) / "walkforward_model_summary.csv").exists()

    preds = pd.read_parquet(outputs["predictions"])
    assert set(preds["feature_set"].unique()) == {"classical_baseline"}
    assert set(preds["model_name"].unique()) == {
        "reversal",
        "zscore_reversal",
        "momentum",
        "reversal_plus_momentum",
    }
    years = sorted(pd.to_datetime(preds["date"]).dt.year.unique().tolist())
    assert years == list(range(2017, 2025))
    assert set(preds["target_col"].unique()) == {"target_ret_fwd_1m"}

    summary = pd.read_csv(outputs["summary"])
    assert set(np.round(summary["cutoff"].unique(), 2)) == {0.05, 0.10, 0.20}
    assert set(summary["feature_set"].unique()) == {"classical_baseline"}

    diff = pd.read_csv(outputs["difference_tests"])
    assert set(DIFF_COMPARISON_PAIRS).issubset(
        set(diff[["strategy_a", "strategy_b"]].itertuples(index=False, name=None))
    )
    assert set(diff["baseline_source"].unique()) == {BASELINE_SOURCE}

    preview = outputs["preview"].read_text(encoding="utf-8")
    assert "Projected model-panel columns" in preview
    assert "2017-01-31" in preview and "2024-12-31" in preview
