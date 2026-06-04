"""Synthetic smoke test for the walk-forward expanding-window walk-forward runner.

Pure pandas + sklearn (no torch/xgboost), so it runs in the main suite without
the macOS OpenMP guard. Builds a tiny synthetic CRSP panel, runs ONE fold (test
year 2017) on the linear families with smoke-capped candidate grids, and asserts
the walk-forward contract:

* train / validation / test windows do not overlap;
* only test-year rows are emitted as OOS predictions;
* validation selection scored candidates on the validation split (grid populated,
  selected config traces back to a graded candidate);
* the required prediction output schema is present;
* all five output artefacts are written.

The runner SUPPORTS ridge / elastic_net / xgboost / mlp; xgboost + mlp are
exercised on cluster (the contract logic here is model-agnostic, so the linear
families fully cover it cheaply and deterministically).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlfinance.run.walkforward import (
    REGIME_FEATURE_COLS,
    _assert_no_forbidden,
    _build_panel,
    aggregate_walkforward,
    run_walkforward,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PREDICTION_COLUMNS = {
    "permno",
    "date",
    "target_ret_fwd_1m",
    "prediction",
    "model_name",
    "feature_set",
    "target_col",
    "fold_id",
    "train_start",
    "train_end",
    "val_start",
    "val_end",
    "test_start",
    "test_end",
    "selected_hparams_json",
}


def _write_synthetic_crsp(path: Path) -> None:
    """Monthly CRSP 2009-2018, 12 permnos, with a faint reversal signal in returns."""
    dates = pd.date_range("2009-01-31", "2018-12-31", freq="ME")
    rng = np.random.default_rng(7)
    permnos = list(range(10001, 10013))
    rows: list[dict[str, object]] = []
    for permno in permnos:
        prev = 0.0
        for step, date in enumerate(dates):
            # mild mean reversion: next return tilts against the previous one.
            ret = -0.15 * prev + 0.03 * rng.standard_normal()
            rows.append(
                {
                    "PERMNO": int(permno),
                    "HdrCUSIP": f"{permno:08d}",
                    "SICCD": 3571 + (permno % 5),
                    "MthCalDt": date,
                    "MthRet": float(ret),
                    "sprtrn": 0.004 + 0.0001 * step,
                }
            )
            prev = ret
    pd.DataFrame(rows).to_parquet(path, index=False)


def _cfg(tmp_path: Path) -> OmegaConf:
    """Real configs/main.yaml with all data/output paths redirected to tmp."""
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "main.yaml")
    crsp = tmp_path / "monthly_crsp.parquet"
    _write_synthetic_crsp(crsp)
    overrides = OmegaConf.create(
        {
            "data": {
                "panel_path": str(tmp_path / "model_panel.parquet"),
                "raw_dir": str(tmp_path),
                "processed_dir": str(tmp_path),
                "paths": {"crsp_monthly": str(crsp)},
            },
            "paths": {
                "predictions": str(tmp_path / "outputs" / "predictions"),
                "backtests": str(tmp_path / "outputs" / "backtests"),
                "metrics": str(tmp_path / "outputs" / "metrics"),
            },
            "runtime": {"rebuild_panel": True},
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _run(tmp_path: Path) -> tuple[dict[str, Path], pd.DataFrame]:
    outputs = run_walkforward(
        _cfg(tmp_path),
        feature_sets=("return_only",),
        models=("ridge", "elastic_net"),
        test_years=(2017,),
        smoke=True,
    )
    preds = pd.read_parquet(outputs["predictions"])
    return outputs, preds


def test_walkforward_writes_all_outputs(tmp_path: Path) -> None:
    outputs, _ = _run(tmp_path)
    for key in (
        "predictions",
        "validation_grid",
        "selected_hparams",
        "backtests",
        "model_summary",
    ):
        assert outputs[key].exists(), key


def test_only_test_year_rows_are_emitted(tmp_path: Path) -> None:
    _, preds = _run(tmp_path)
    assert not preds.empty
    years = pd.to_datetime(preds["date"]).dt.year.unique().tolist()
    assert years == [2017], f"OOS predictions must be test-year 2017 only, got {years}"
    assert set(preds["fold_id"].unique()) == {2017}


def test_train_val_test_windows_do_not_overlap(tmp_path: Path) -> None:
    _, preds = _run(tmp_path)
    row = preds.iloc[0]
    train_end = pd.to_datetime(row["train_end"])
    val_start = pd.to_datetime(row["val_start"])
    val_end = pd.to_datetime(row["val_end"])
    test_start = pd.to_datetime(row["test_start"])
    # strict ordering: train < val < test, no overlap
    assert train_end < val_start, (train_end, val_start)
    assert val_end < test_start, (val_end, test_start)
    # the fold schedule: test 2017 -> train <= 2011, val 2012-2016, test 2017
    assert train_end.year <= 2011
    assert val_start.year == 2012 and val_end.year == 2016
    assert test_start.year == 2017
    # train window is not before the configured locked train_start.
    cfg_train_start = pd.to_datetime(
        OmegaConf.load(REPO_ROOT / "configs" / "main.yaml").splits.locked.train_start
    )
    assert pd.to_datetime(row["train_start"]) >= cfg_train_start


def test_required_prediction_schema(tmp_path: Path) -> None:
    _, preds = _run(tmp_path)
    missing = REQUIRED_PREDICTION_COLUMNS - set(preds.columns)
    assert not missing, f"prediction output missing columns: {missing}"
    # selected_hparams_json is valid JSON per row
    import json

    for raw in preds["selected_hparams_json"].unique():
        json.loads(raw)


def test_validation_selection_used_validation_only(tmp_path: Path) -> None:
    outputs, preds = _run(tmp_path)
    grid = pd.read_csv(outputs["validation_grid"])
    selected = pd.read_csv(outputs["selected_hparams"])

    # The grid graded candidates (>=2 per model from the smoke cap) and carries a
    # validation metric + the selection cost level -- i.e. selection ran on the
    # validation split, never on the test year.
    assert not grid.empty
    assert {"val_net_sharpe", "val_rank_ic", "selection_cost_bps"}.issubset(grid.columns)
    assert (grid["selection_cost_bps"] == int(_cfg(tmp_path).costs.main_bps)).all()
    for model in ("ridge", "elastic_net"):
        assert (grid["model_name"] == model).sum() >= 2

    # Each selected config is one of the graded validation candidates.
    for _, sel in selected.iterrows():
        same = grid[(grid["model_name"] == sel["model_name"]) & (grid["fold_id"] == sel["fold_id"])]
        assert sel["selected_hparams_json"] in set(same["params_json"])


def test_one_oos_block_per_model_and_fold(tmp_path: Path) -> None:
    _, preds = _run(tmp_path)
    # two linear models x one fold x one feature_set -> two strategy blocks,
    # each covering the 2017 test months for every permno.
    blocks = preds.groupby(["feature_set", "model_name", "fold_id"]).ngroups
    assert blocks == 2
    assert set(preds["model_name"].unique()) == {"ridge", "elastic_net"}


def test_default_run_keeps_main_target_and_no_seed_column(tmp_path: Path) -> None:
    # Byte-identity invariant: the accepted final audited version default path is unchanged --
    # target_ret_fwd_1m and NO `seed` column (the seed column is added only under a
    # non-canonical output_prefix, so default re-runs stay byte-identical).
    _, preds = _run(tmp_path)
    assert set(preds["target_col"].unique()) == {"target_ret_fwd_1m"}
    assert "seed" not in preds.columns


def test_seed_is_recorded_under_isolated_prefix(tmp_path: Path) -> None:
    # A robustness rerun (non-canonical output_prefix) records the seed in its
    # predictions + selected-hparams so a multi-seed aggregate can tell seeds apart.
    out = run_walkforward(
        _cfg(tmp_path),
        feature_sets=("return_only",),
        models=("ridge",),
        test_years=(2017,),
        seed=2,
        output_prefix="walkforward_seed_probe",
        smoke=True,
    )
    preds = pd.read_parquet(out["predictions"])
    assert "seed" in preds.columns and set(preds["seed"].unique()) == {2}
    sel = pd.read_csv(out["selected_hparams"])
    assert set(sel["seed"].unique()) == {2}


def test_target_override_isolates_outputs_and_uses_excess(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    out = run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("ridge",),
        test_years=(2017,),
        target_col="target_excess_ret_fwd_1m",
        output_prefix="walkforward_excess",
        smoke=True,
    )
    # outputs land under the isolated prefix, trained/evaluated on the excess label
    assert out["predictions"].name == "walkforward_excess_predictions.parquet"
    assert out["model_summary"].name == "walkforward_excess_model_summary.csv"
    preds = pd.read_parquet(out["predictions"])
    assert set(preds["target_col"].unique()) == {"target_excess_ret_fwd_1m"}
    assert "target_excess_ret_fwd_1m" in preds.columns
    # the accepted main-target audited version canonical file is NOT created/overwritten
    assert not (Path(cfg.paths.predictions) / "walkforward_predictions.parquet").exists()


def test_excess_target_shard_then_aggregate(tmp_path: Path) -> None:
    # Excess-target shards (single seed) -> aggregate must adopt the excess label
    # the shards persisted (else the backtest KeyErrors on the default target column).
    cfg = _cfg(tmp_path)
    run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("ridge",),
        test_years=(2017,),
        target_col="target_excess_ret_fwd_1m",
        output_prefix="walkforward_excess",
        shard_tag="excess-ro-ridge-2017",
        smoke=True,
    )
    agg = aggregate_walkforward(cfg, output_prefix="walkforward_excess")
    assert agg["predictions"].name == "walkforward_excess_predictions.parquet"
    assert agg["model_summary"].name == "walkforward_excess_model_summary.csv"
    summary = pd.read_csv(agg["model_summary"])
    assert not summary.empty
    assert {"feature_set", "model", "cutoff", "cost_bps", "net_sharpe"}.issubset(summary.columns)
    preds = pd.read_parquet(agg["predictions"])
    assert set(preds["target_col"].unique()) == {"target_excess_ret_fwd_1m"}
    # accepted main-target audited version canonical untouched
    assert not (Path(cfg.paths.predictions) / "walkforward_predictions.parquet").exists()


def test_multiseed_shards_aggregate_records_each_seed(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    for s in (1, 2, 3):
        run_walkforward(
            cfg,
            feature_sets=("return_only",),
            models=("ridge",),
            test_years=(2017,),
            seed=s,
            output_prefix="walkforward_mlp_multiseed",
            shard_tag=f"seed{s}-2017",
            smoke=True,
        )
    agg = aggregate_walkforward(cfg, output_prefix="walkforward_mlp_multiseed")
    # multiseed branch -> per-seed stability summary, NOT the pooled cost grid
    assert "summary" in agg and "backtests" not in agg
    preds = pd.read_parquet(agg["predictions"])
    assert set(preds["seed"].unique()) == {1, 2, 3}
    summary = pd.read_csv(agg["summary"])
    assert {"feature_set", "model", "seed", "net_sharpe"}.issubset(summary.columns)
    assert summary["seed"].astype(str).str.contains("ALL").any()  # cross-seed row
    assert (summary["seed"].astype(str) == "1").any()  # per-seed rows
    # accepted main audited version canonical untouched
    assert not (Path(cfg.paths.predictions) / "walkforward_predictions.parquet").exists()


def test_shard_run_defers_backtest_and_suffixes_outputs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    out = run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("ridge",),
        test_years=(2017,),
        shard_tag="ridge-2017",
        smoke=True,
    )
    # A shard writes only its slice; the global backtest/summary is deferred to
    # aggregate so concurrent shards never collide on the summary artefacts.
    assert "backtests" not in out and "model_summary" not in out
    assert out["predictions"].name.endswith("__ridge-2017.parquet")
    assert out["validation_grid"].name.endswith("__ridge-2017.csv")
    assert out["selected_hparams"].name.endswith("__ridge-2017.csv")


def test_aggregate_stitches_shards_into_cost_grid(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # Two independent shards (one model each), as the fan-out would submit them.
    run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("ridge",),
        test_years=(2017,),
        shard_tag="ridge-2017",
        smoke=True,
    )
    run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("elastic_net",),
        test_years=(2017,),
        shard_tag="en-2017",
        smoke=True,
    )

    agg = aggregate_walkforward(cfg)
    for key in ("predictions", "backtests", "model_summary"):
        assert agg[key].exists(), key

    # Both shards stitched into the canonical (un-suffixed) prediction file.
    merged = pd.read_parquet(agg["predictions"])
    assert agg["predictions"].name == "walkforward_predictions.parquet"
    assert set(merged["model_name"].unique()) == {"ridge", "elastic_net"}

    summary = pd.read_csv(agg["model_summary"])
    # Post-processing grid: cutoffs x fine cost grid + breakeven, per strategy.
    assert {"feature_set", "model", "cutoff", "cost_bps", "net_sharpe", "breakeven_bps"}.issubset(
        summary.columns
    )
    # Fractional bps survive (int-truncation in the shared summariser is undone).
    assert (summary["cost_bps"] == 7.5).any()
    # Thin synthetic panel (12 names/month) supports the 10%/20% cuts only.
    assert set(summary["cutoff"].unique()) == {0.10, 0.20}
    assert set(summary["model"].unique()) == {"ridge", "elastic_net"}


def test_train_window_respects_configured_train_start(tmp_path: Path) -> None:
    # Floor the train window inside the synthetic range and verify the runner honours
    # cfg.splits.locked.train_start (not the panel's earlier lookback-buffer start).
    cfg = OmegaConf.merge(
        _cfg(tmp_path),
        OmegaConf.create({"splits": {"locked": {"train_start": "2010-12-31"}}}),
    )
    outputs = run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("ridge",),
        test_years=(2017,),
        smoke=True,
    )
    preds = pd.read_parquet(outputs["predictions"])
    train_start = pd.to_datetime(preds["train_start"])
    assert (train_start >= pd.Timestamp("2010-12-31")).all()
    # the rest of the fold schedule is unchanged: val 2012-2016, test 2017.
    row = preds.iloc[0]
    assert pd.to_datetime(row["val_start"]).year == 2012
    assert pd.to_datetime(row["val_end"]).year == 2016
    assert pd.to_datetime(row["test_start"]).year == 2017


def test_aggregate_ignores_stale_canonical(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    # Two valid shards, exactly as the fan-out would write them.
    run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("ridge",),
        test_years=(2017,),
        shard_tag="ridge-2017",
        smoke=True,
    )
    run_walkforward(
        cfg,
        feature_sets=("return_only",),
        models=("elastic_net",),
        test_years=(2017,),
        shard_tag="en-2017",
        smoke=True,
    )
    pred_dir = Path(cfg.paths.predictions)
    canonical = pred_dir / "walkforward_predictions.parquet"
    # Plant a STALE canonical file with a fake model/fold no shard produced.
    stale = pd.read_parquet(pred_dir / "walkforward_predictions__ridge-2017.parquet").copy()
    stale["model_name"] = "STALE_MODEL"
    stale["fold_id"] = 1999
    stale.to_parquet(canonical, index=False)

    # Aggregate merges shard files ONLY -- the canonical is a derived artifact, never
    # an input, so the stale rows must be excluded.
    agg = aggregate_walkforward(cfg)
    merged = pd.read_parquet(agg["predictions"])
    assert "STALE_MODEL" not in set(merged["model_name"].unique())
    assert 1999 not in set(merged["fold_id"].unique())
    assert {"ridge", "elastic_net"}.issubset(set(merged["model_name"].unique()))

    # Re-running aggregate after the same stale plant is idempotent: the freshly
    # overwritten canonical was the shard-only merge, and even if a stale file is
    # planted again, the new aggregate ignores it.
    stale.to_parquet(canonical, index=False)
    agg2 = aggregate_walkforward(cfg)
    merged2 = pd.read_parquet(agg2["predictions"])
    assert "STALE_MODEL" not in set(merged2["model_name"].unique())


# ---------------------------------------------------------------------------
# walk-forward regime-feature ablation (optional; additive feature sets, isolated
# output prefix, never touches the accepted final audited version canonical files).
# ---------------------------------------------------------------------------


def test_return_regime_panel_adds_regime_columns(tmp_path: Path) -> None:
    panel, feats = _build_panel(_cfg(tmp_path), "return_regime")
    # all 6 regime columns are present in the panel AND in the model feature list
    for col in REGIME_FEATURE_COLS:
        assert col in panel.columns, col
        assert col in feats, col
    # the return-only base features are still there (regime EXTENDS, not replaces)
    assert "ret_1m" in feats and "mom_2_12" in feats


def test_default_return_only_feature_list_excludes_regime(tmp_path: Path) -> None:
    _, feats = _build_panel(_cfg(tmp_path), "return_only")
    assert not (
        set(REGIME_FEATURE_COLS) & set(feats)
    ), "default return_only must carry NO regime cols"


def test_feature_list_guard_rejects_future_columns() -> None:
    # the guard must reject any label/forward/prediction column
    for bad in ("target_ret_fwd_1m", "target_excess_ret_fwd_1m", "prediction", "y_pred"):
        with pytest.raises(ValueError, match="future/label"):
            _assert_no_forbidden(["ret_1m", bad])
    # the regime block itself is clean
    assert _assert_no_forbidden(list(REGIME_FEATURE_COLS)) == list(REGIME_FEATURE_COLS)


def test_regime_run_is_isolated_and_does_not_touch_canonical(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    out = run_walkforward(
        cfg,
        feature_sets=("return_regime",),
        models=("ridge",),
        test_years=(2017,),
        output_prefix="regime_features",
        smoke=True,
    )
    assert out["predictions"].name == "regime_features_predictions.parquet"
    preds = pd.read_parquet(out["predictions"])
    assert set(preds["feature_set"].unique()) == {"return_regime"}
    assert set(preds["target_col"].unique()) == {"target_ret_fwd_1m"}
    # accepted final audited version canonical is never created by a regime rerun
    assert not (Path(cfg.paths.predictions) / "walkforward_predictions.parquet").exists()


def test_regime_aggregate_uses_only_regime_shards_and_writes_comparison(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    metrics_dir = Path(cfg.paths.metrics)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Plant a minimal accepted-main summary so the ablation comparison can populate
    # (return_regime's baseline is return_only).
    pd.DataFrame(
        [
            {
                "model_name": "return_only__ridge",
                "feature_set": "return_only",
                "model": "ridge",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "net_sharpe": 0.50,
                "gross_sharpe": 0.70,
                "net_ann_return": 0.06,
                "breakeven_bps": 30.0,
                "rank_ic_mean": 0.01,
                "rank_ic_hac_tstat": 1.2,
            }
        ]
    ).to_csv(metrics_dir / "walkforward_model_summary.csv", index=False)

    # One regime shard, then aggregate under the regime prefix.
    run_walkforward(
        cfg,
        feature_sets=("return_regime",),
        models=("ridge",),
        test_years=(2017,),
        output_prefix="regime_features",
        shard_tag="regime-return_regime-ridge-2017",
        smoke=True,
    )
    agg = aggregate_walkforward(cfg, output_prefix="regime_features")
    assert agg["predictions"].name == "regime_features_predictions.parquet"
    assert agg["model_summary"].name == "regime_features_model_summary.csv"
    assert "ablation_comparison" in agg

    comp = pd.read_csv(agg["ablation_comparison"])
    assert {
        "feature_set",
        "model",
        "cutoff",
        "cost_bps",
        "net_sharpe",
        "breakeven_bps",
        "baseline_feature_set",
        "baseline_net_sharpe",
        "delta_net_sharpe_vs_baseline",
    }.issubset(comp.columns)
    row = comp[(comp["feature_set"] == "return_regime") & (comp["model"] == "ridge")].iloc[0]
    assert row["baseline_feature_set"] == "return_only"
    assert row["baseline_net_sharpe"] == pytest.approx(0.50)
    assert row["delta_net_sharpe_vs_baseline"] == pytest.approx(row["net_sharpe"] - 0.50)
    # the planted accepted-main summary was read-only, not overwritten with regime rows
    main = pd.read_csv(metrics_dir / "walkforward_model_summary.csv")
    assert set(main["feature_set"].unique()) == {"return_only"}


def test_regime_aggregate_skips_comparison_when_main_summary_absent(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    run_walkforward(
        cfg,
        feature_sets=("return_regime",),
        models=("ridge",),
        test_years=(2017,),
        output_prefix="regime_features",
        shard_tag="regime-return_regime-ridge-2017",
        smoke=True,
    )
    # no accepted-main summary planted -> comparison skipped, aggregate still succeeds
    agg = aggregate_walkforward(cfg, output_prefix="regime_features")
    assert agg["model_summary"].exists()
    assert "ablation_comparison" not in agg
