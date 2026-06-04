"""End-to-end synthetic smoke test for the walk-forward post-processing robustness runner.

Synthesises the corrected-audited version walk-forward OUTPUTS (predictions + backtests +
model_summary via the REAL ``_backtest_oos_grid``, plus a reversal baseline and a
loader-compatible CRSP file for ``sprtrn``), then runs ``run_postprocess`` and
asserts every table is produced with the required schema. Pure pandas + statsmodels
(no torch / xgboost), so it runs in the main suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlfinance.run.walkforward import _backtest_oos_grid
from mlfinance.run.walkforward_robustness import run_postprocess

REPO_ROOT = Path(__file__).resolve().parents[2]

FEATURE_SETS = ("return_only", "flow_compustat")
MODELS = ("mlp", "xgboost")
FOLDS = (2017, 2018)
N_PERMNOS = 20


def _write_synthetic_crsp(path: Path) -> None:
    """Monthly CRSP 2015-12..2018-12 with a varying sprtrn (loader-compatible)."""
    dates = pd.date_range("2015-12-31", "2018-12-31", freq="ME")
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for step, date in enumerate(dates):
        # sprtrn swings sign across the window so vol/trend regimes both populate.
        sprtrn = 0.02 * np.sin(step / 3.0) + 0.004 * rng.standard_normal()
        for permno in range(10001, 10001 + N_PERMNOS):
            rows.append(
                {
                    "PERMNO": int(permno),
                    "HdrCUSIP": f"{permno:08d}",
                    "SICCD": 3571 + (permno % 5),
                    "MthCalDt": date,
                    "MthRet": float(0.01 * rng.standard_normal()),
                    "sprtrn": float(sprtrn),
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _synthetic_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    permnos = list(range(10001, 10001 + N_PERMNOS))
    rows: list[dict[str, object]] = []
    for fs in FEATURE_SETS:
        for model in MODELS:
            for year in FOLDS:
                dates = pd.date_range(f"{year}-01-31", f"{year}-12-31", freq="ME")
                for date in dates:
                    target = 0.03 * rng.standard_normal(len(permnos))
                    pred = 0.6 * target + 0.02 * rng.standard_normal(len(permnos))
                    for permno, t, p in zip(permnos, target, pred, strict=True):
                        rows.append(
                            {
                                "feature_set": fs,
                                "model_name": model,
                                "fold_id": int(year),
                                "permno": int(permno),
                                "date": date,
                                "prediction": float(p),
                                "target_ret_fwd_1m": float(t),
                                "train_start": "1985-01-31",
                                "train_end": f"{year - 6}-12-31",
                                "val_start": f"{year - 5}-01-31",
                                "val_end": f"{year - 1}-12-31",
                                "test_start": f"{year}-01-31",
                                "test_end": f"{year}-12-31",
                                "selected_hparams_json": '{"seed": 1}',
                            }
                        )
    return pd.DataFrame(rows)


def _write_reversal_baseline(path: Path) -> None:
    """baseline_backtests.parquet with a reversal book over the OOS months."""
    dates = pd.date_range("2017-01-31", "2018-12-31", freq="ME")
    rng = np.random.default_rng(9)
    rows: list[dict[str, object]] = []
    gross = 0.004 + 0.02 * rng.standard_normal(len(dates))
    turn = np.full(len(dates), 1.8)
    for cost_bps in (0, 5, 10, 25, 50):
        cost = turn * (cost_bps / 10_000.0)
        for d, g, t, c in zip(dates, gross, turn, cost, strict=True):
            rows.append(
                {
                    "date": d,
                    "month": d,
                    "model_name": "reversal",
                    "split_id": "baseline_locked_test",
                    "cost_bps": int(cost_bps),
                    "gross_return": float(g),
                    "turnover": float(t),
                    "cost": float(c),
                    "net_return": float(g - c),
                    "long_return": float(g) / 2.0,
                    "short_return": float(g) / 2.0,
                    "n_long": 2,
                    "n_short": 2,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _cfg(tmp_path: Path) -> OmegaConf:
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "main.yaml")
    crsp = tmp_path / "monthly_crsp.parquet"
    _write_synthetic_crsp(crsp)
    overrides = OmegaConf.create(
        {
            "data": {"raw_dir": str(tmp_path), "paths": {"crsp_monthly": str(crsp)}},
            "paths": {
                "predictions": str(tmp_path / "outputs" / "predictions"),
                "backtests": str(tmp_path / "outputs" / "backtests"),
                "metrics": str(tmp_path / "outputs" / "metrics"),
            },
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _materialize_walkforward_outputs(cfg: OmegaConf) -> None:
    """Write predictions / backtests / model_summary / selected_hparams / reversal."""
    preds = _synthetic_predictions()
    pred_dir = Path(cfg.paths.predictions)
    bt_dir = Path(cfg.paths.backtests)
    mdir = Path(cfg.paths.metrics)
    for d in (pred_dir, bt_dir, mdir):
        d.mkdir(parents=True, exist_ok=True)

    preds.to_parquet(pred_dir / "walkforward_predictions.parquet", index=False)

    backtests, model_summary = _backtest_oos_grid(preds, cfg)
    backtests.to_parquet(bt_dir / "walkforward_backtests.parquet", index=False)
    model_summary.to_csv(mdir / "walkforward_model_summary.csv", index=False)

    sel = (
        preds.groupby(["feature_set", "model_name", "fold_id"], as_index=False)
        .agg(train_start=("train_start", "first"), val_end=("val_end", "first"))
        .assign(selected_hparams_json='{"seed": 1}', val_net_sharpe=1.0)
    )
    sel.to_csv(mdir / "walkforward_selected_hparams.csv", index=False)

    _write_reversal_baseline(bt_dir / "baseline_backtests.parquet")


def _run(tmp_path: Path) -> tuple[OmegaConf, dict[str, Path]]:
    cfg = _cfg(tmp_path)
    _materialize_walkforward_outputs(cfg)
    return cfg, run_postprocess(cfg)


def test_all_outputs_written(tmp_path: Path) -> None:
    _, out = _run(tmp_path)
    for key in (
        "cost_grid",
        "breakeven",
        "rebalance",
        "regime",
        "alpha_beta",
        "model_difference",
        "preview",
    ):
        assert out[key].exists(), key
    assert out["preview"].name == "walkforward_output_preview.md"


def test_cost_grid_preserves_fractional_bps(tmp_path: Path) -> None:
    _, out = _run(tmp_path)
    cg = pd.read_csv(out["cost_grid"])
    assert {"feature_set", "model", "cutoff", "cost_bps", "net_sharpe", "breakeven_bps"}.issubset(
        cg.columns
    )
    assert (cg["cost_bps"] == 7.5).any()
    assert (cg["cost_bps"] == 12.5).any()


def test_breakeven_table(tmp_path: Path) -> None:
    _, out = _run(tmp_path)
    be = pd.read_csv(out["breakeven"])
    assert {"feature_set", "model", "cutoff", "breakeven_bps", "mean_turnover"}.issubset(be.columns)
    # 4 strategies present
    assert be.groupby(["feature_set", "model"]).ngroups == 4


def test_rebalance_has_monthly_and_quarterly(tmp_path: Path) -> None:
    _, out = _run(tmp_path)
    rb = pd.read_csv(out["rebalance"])
    assert set(rb["rebalance"].unique()) == {"monthly", "quarterly"}
    # 4 strategies x 2 frequencies
    assert len(rb) == 8
    assert {"net_sharpe", "avg_turnover", "breakeven_bps", "hit_rate"}.issubset(rb.columns)


def test_regime_schemes_present(tmp_path: Path) -> None:
    _, out = _run(tmp_path)
    rr = pd.read_csv(out["regime"])
    assert {"vol", "trend", "dispersion", "period"}.issubset(set(rr["regime_scheme"].unique()))
    assert {
        "n_months",
        "mean_net_return",
        "net_sharpe",
        "hit_rate",
        "newey_west_tstat_mean_net",
        "alpha",
        "beta",
        "alpha_tstat",
    }.issubset(rr.columns)


def test_alpha_beta_has_gross_and_net(tmp_path: Path) -> None:
    _, out = _run(tmp_path)
    ab = pd.read_csv(out["alpha_beta"])
    assert set(ab["return_type"].unique()) == {"gross", "net"}
    assert {"alpha", "alpha_tstat", "beta", "beta_tstat", "nw_tstat_mean"}.issubset(ab.columns)


def test_model_difference_includes_within_and_reversal(tmp_path: Path) -> None:
    _, out = _run(tmp_path)
    md = pd.read_csv(out["model_difference"])
    pairs = set(md["pair"])
    assert "flow_compustat__mlp - return_only__mlp" in pairs
    assert "flow_compustat__xgboost - return_only__xgboost" in pairs
    assert "flow_compustat__mlp - flow_compustat__xgboost" in pairs
    # reversal baseline was synthesised -> reversal pairs present, sourced from baseline
    rev = md[md["strategy_b"] == "reversal"]
    assert not rev.empty
    assert rev["baseline_source"].str.contains("baseline").all()
    assert "nw_tstat_diff" in md.columns


def test_validation_rejects_pre1985_inputs(tmp_path: Path) -> None:
    import pytest

    cfg = _cfg(tmp_path)
    _materialize_walkforward_outputs(cfg)
    # Contaminate the predictions with a pre-1985 train_start and re-run.
    pred_path = Path(cfg.paths.predictions) / "walkforward_predictions.parquet"
    preds = pd.read_parquet(pred_path)
    preds["train_start"] = "1984-01-31"
    preds.to_parquet(pred_path, index=False)
    with pytest.raises(ValueError, match="contaminated"):
        run_postprocess(cfg)


def test_one_failing_analysis_does_not_block_others(tmp_path: Path, monkeypatch) -> None:
    """A failure in one table (here: rebalance) is isolated -- the cost/regime/stat
    outputs are still produced, the failed one is stubbed, and the preview records it."""
    import mlfinance.run.walkforward_robustness as m

    cfg = _cfg(tmp_path)
    _materialize_walkforward_outputs(cfg)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated rebalance failure")

    monkeypatch.setattr(m, "rebalance_robustness", _boom)
    out = m.run_postprocess(cfg)

    # every artefact still exists
    for key in (
        "cost_grid",
        "breakeven",
        "rebalance",
        "regime",
        "alpha_beta",
        "model_difference",
        "preview",
    ):
        assert out[key].exists(), key

    # the failed table is a stub; the others are real and intact
    rb = pd.read_csv(out["rebalance"])
    assert "status" in rb.columns and (rb["status"] == "SKIPPED").all()
    assert "cost_bps" in pd.read_csv(out["cost_grid"]).columns
    assert "regime_scheme" in pd.read_csv(out["regime"]).columns
    assert set(pd.read_csv(out["alpha_beta"])["return_type"].unique()) == {"gross", "net"}

    # the preview surfaces the isolated failure
    assert "SKIPPED" in out["preview"].read_text(encoding="utf-8")
