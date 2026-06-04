from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlfinance.eval.final_report_outputs import (
    CLASSICAL_BASELINE_DIFFERENCE_TABLE_FILENAME,
    CLASSICAL_BASELINE_TABLE_FILENAME,
    COST_FIGURE_FILENAME,
    COST_TABLE_FILENAME,
    DATA_SUMMARY_TABLE_FILENAME,
    DECILE_FIGURE_FILENAME,
    DECILE_TABLE_FILENAME,
    FINAL_MODEL_HIERARCHY_TABLE_FILENAME,
    MAIN_TABLE_FILENAME,
    SUPPLEMENTARY_ROBUSTNESS_TABLE_FILENAME,
    build_final_report_paths,
    generate_final_report_outputs,
    validate_report_input_path,
)


def _paths(tmp_path: Path):
    root = tmp_path
    for rel in (
        "data/outputs/metrics",
        "data/outputs/backtests",
        "data/outputs/predictions",
        "data/processed",
        "report/tables",
        "report/figures",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return build_final_report_paths(repo_root=root)


def _write_main_summary(paths) -> None:
    rows = [
        {
            "model_name": "flow_compustat__mlp",
            "feature_set": "flow_compustat",
            "model": "mlp",
            "cutoff": 0.10,
            "cost_bps": 0.0,
            "gross_sharpe": 2.30,
            "net_sharpe": 2.20,
            "net_ann_return": 0.22,
            "avg_turnover": 2.10,
            "breakeven_bps": 92.0,
            "rank_ic_mean": 0.041,
            "rank_ic_hac_tstat": 6.2,
        },
        {
            "model_name": "flow_compustat__mlp",
            "feature_set": "flow_compustat",
            "model": "mlp",
            "cutoff": 0.10,
            "cost_bps": 7.5,
            "gross_sharpe": 2.30,
            "net_sharpe": 2.08,
            "net_ann_return": 0.215,
            "avg_turnover": 2.10,
            "breakeven_bps": 92.0,
            "rank_ic_mean": 0.041,
            "rank_ic_hac_tstat": 6.2,
        },
        {
            "model_name": "flow_compustat__mlp",
            "feature_set": "flow_compustat",
            "model": "mlp",
            "cutoff": 0.10,
            "cost_bps": 10.0,
            "gross_sharpe": 2.30,
            "net_sharpe": 2.00,
            "net_ann_return": 0.210,
            "avg_turnover": 2.10,
            "breakeven_bps": 92.0,
            "rank_ic_mean": 0.041,
            "rank_ic_hac_tstat": 6.2,
        },
        {
            "model_name": "return_only__xgboost",
            "feature_set": "return_only",
            "model": "xgboost",
            "cutoff": 0.10,
            "cost_bps": 0.0,
            "gross_sharpe": 1.80,
            "net_sharpe": 1.60,
            "net_ann_return": 0.18,
            "avg_turnover": 2.50,
            "breakeven_bps": 70.0,
            "rank_ic_mean": 0.022,
            "rank_ic_hac_tstat": 2.1,
        },
        {
            "model_name": "return_only__xgboost",
            "feature_set": "return_only",
            "model": "xgboost",
            "cutoff": 0.10,
            "cost_bps": 7.5,
            "gross_sharpe": 1.80,
            "net_sharpe": 1.45,
            "net_ann_return": 0.175,
            "avg_turnover": 2.50,
            "breakeven_bps": 70.0,
            "rank_ic_mean": 0.022,
            "rank_ic_hac_tstat": 2.1,
        },
        {
            "model_name": "return_only__xgboost",
            "feature_set": "return_only",
            "model": "xgboost",
            "cutoff": 0.10,
            "cost_bps": 10.0,
            "gross_sharpe": 1.80,
            "net_sharpe": 1.31,
            "net_ann_return": 0.170,
            "avg_turnover": 2.50,
            "breakeven_bps": 70.0,
            "rank_ic_mean": 0.022,
            "rank_ic_hac_tstat": 2.1,
        },
        {
            "model_name": "flow_compustat__ridge",
            "feature_set": "flow_compustat",
            "model": "ridge",
            "cutoff": 0.10,
            "cost_bps": 10.0,
            "gross_sharpe": 0.30,
            "net_sharpe": -0.05,
            "net_ann_return": -0.001,
            "avg_turnover": 2.00,
            "breakeven_bps": 10.0,
            "rank_ic_mean": 0.013,
            "rank_ic_hac_tstat": 1.3,
        },
    ]
    pd.DataFrame(rows).to_csv(
        paths.metrics_dir / "walkforward_model_summary.csv",
        index=False,
    )


def _write_main_summary_report_style(paths) -> None:
    rows = [
        {
            "model_name": "flow_compustat__mlp",
            "feature_set": "flow_compustat",
            "model": "mlp",
            "cutoff": 0.10,
            "cost_bps": 10.0,
            "gross_SR": 2.30,
            "net_SR": 2.00,
            "net_ann": 0.210,
            "turnover": 2.10,
            "breakeven_bps": 92.0,
            "rank_IC": 0.041,
            "IC_t_HAC": 6.2,
        },
        {
            "model_name": "return_only__xgboost",
            "feature_set": "return_only",
            "model": "xgboost",
            "cutoff": 0.10,
            "cost_bps": 10.0,
            "gross_SR": 1.80,
            "net_SR": 1.31,
            "net_ann": 0.170,
            "turnover": 2.50,
            "breakeven_bps": 70.0,
            "rank_IC": 0.022,
            "IC_t_HAC": 2.1,
        },
    ]
    pd.DataFrame(rows).to_csv(
        paths.metrics_dir / "walkforward_model_summary.csv",
        index=False,
    )


def _write_predictions(paths) -> None:
    rows = []
    for month in pd.to_datetime(["2017-01-31", "2017-02-28"]):
        for permno in range(10001, 10011):
            rank = permno - 10000
            rows.append(
                {
                    "date": month,
                    "permno": permno,
                    "model_name": "flow_compustat__mlp",
                    "feature_set": "flow_compustat",
                    "prediction": float(rank),
                    "target_ret_fwd_1m": float(rank / 1000.0),
                }
            )
    pd.DataFrame(rows).to_parquet(
        paths.predictions_dir / "walkforward_predictions.parquet",
        index=False,
    )


def _write_model_panel(paths) -> None:
    rows = []
    for month in pd.to_datetime(["2017-01-31", "2017-02-28"]):
        for permno in range(10001, 10006):
            rows.append(
                {
                    "date": month,
                    "permno": permno,
                    "ret": 0.01,
                    "sprtrn": 0.005,
                    "target_ret_fwd_1m": 0.012,
                    "ret_1m": 0.01,
                    "mom_2_12": 0.02,
                }
            )
    pd.DataFrame(rows).to_parquet(paths.processed_dir / "model_panel.parquet", index=False)


def _write_same_protocol_baseline_metrics(paths) -> None:
    pd.DataFrame(
        [
            {
                "model_name": "classical_baseline__reversal",
                "feature_set": "classical_baseline",
                "model": "reversal",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "gross_sharpe": 1.10,
                "net_sharpe": 0.95,
                "net_ann_return": 0.09,
                "avg_turnover": 1.80,
                "breakeven_bps": 40.0,
                "rank_ic_mean": 0.011,
                "rank_ic_hac_tstat": 1.7,
            },
            {
                "model_name": "classical_baseline__zscore_reversal",
                "feature_set": "classical_baseline",
                "model": "zscore_reversal",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "gross_sharpe": 1.00,
                "net_sharpe": 0.85,
                "net_ann_return": 0.08,
                "avg_turnover": 1.70,
                "breakeven_bps": 38.0,
                "rank_ic_mean": 0.010,
                "rank_ic_hac_tstat": 1.5,
            },
            {
                "model_name": "classical_baseline__momentum",
                "feature_set": "classical_baseline",
                "model": "momentum",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "gross_sharpe": 0.80,
                "net_sharpe": 0.60,
                "net_ann_return": 0.06,
                "avg_turnover": 1.20,
                "breakeven_bps": 55.0,
                "rank_ic_mean": 0.009,
                "rank_ic_hac_tstat": 1.2,
            },
            {
                "model_name": "classical_baseline__reversal_plus_momentum",
                "feature_set": "classical_baseline",
                "model": "reversal_plus_momentum",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "gross_sharpe": 0.90,
                "net_sharpe": 0.70,
                "net_ann_return": 0.07,
                "avg_turnover": 1.60,
                "breakeven_bps": 50.0,
                "rank_ic_mean": 0.010,
                "rank_ic_hac_tstat": 1.3,
            },
        ]
    ).to_csv(paths.metrics_dir / "classical_baseline_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "flow_compustat__mlp - reversal",
                "strategy_a": "flow_compustat__mlp",
                "strategy_b": "reversal",
                "n_months": 95,
                "mean_diff_monthly": 0.01,
                "mean_diff_annualized": 0.12,
                "nw_tstat_diff": 2.4,
                "baseline_source": "classical_baseline",
            }
        ]
    ).to_csv(paths.metrics_dir / "classical_baseline_difference_tests.csv", index=False)


def _write_supplementary_metrics(paths) -> None:
    pd.DataFrame(
        [
            {
                "model_name": "flow_compustat__mlp",
                "feature_set": "flow_compustat",
                "model": "mlp",
                "target": "target_excess_ret_fwd_1m",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "gross_sharpe": 3.2323,
                "net_sharpe": 2.9699,
                "net_ann_return": 0.2861,
                "breakeven_bps": 123.97,
                "rank_ic_mean": 0.0476,
                "rank_ic_hac_tstat": 4.5148,
            }
        ]
    ).to_csv(paths.metrics_dir / "walkforward_excess_model_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "feature_set": "flow_compustat",
                "model": "mlp",
                "seed": 1,
                "net_sharpe": 1.7174,
            },
            {
                "feature_set": "flow_compustat",
                "model": "mlp",
                "seed": 2,
                "net_sharpe": 2.0286,
            },
            {
                "feature_set": "flow_compustat",
                "model": "mlp",
                "seed": 3,
                "net_sharpe": 2.1589,
            },
            {
                "feature_set": "flow_compustat",
                "model": "mlp",
                "seed": "ALL(mean+/-std)",
                "n_seeds": 3,
                "net_sharpe_mean": 1.9683,
                "net_sharpe_std": 0.2268,
                "net_sharpe_min": 1.7174,
                "net_sharpe_max": 2.1589,
            },
        ]
    ).to_csv(paths.metrics_dir / "walkforward_mlp_multiseed_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "feature_set": "flow_compustat_regime",
                "model": "mlp",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "net_sharpe": 1.520,
                "baseline_feature_set": "flow_compustat",
                "baseline_net_sharpe": 2.028,
                "delta_net_sharpe_vs_baseline": -0.508,
            },
            {
                "feature_set": "flow_compustat_regime",
                "model": "xgboost",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "net_sharpe": 1.439,
                "baseline_feature_set": "flow_compustat",
                "baseline_net_sharpe": 1.770,
                "delta_net_sharpe_vs_baseline": -0.332,
            },
            {
                "feature_set": "return_regime",
                "model": "mlp",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "net_sharpe": 1.398,
                "baseline_feature_set": "return_only",
                "baseline_net_sharpe": 1.192,
                "delta_net_sharpe_vs_baseline": 0.207,
            },
            {
                "feature_set": "return_regime",
                "model": "xgboost",
                "cutoff": 0.10,
                "cost_bps": 10.0,
                "net_sharpe": 1.267,
                "baseline_feature_set": "return_only",
                "baseline_net_sharpe": 1.313,
                "delta_net_sharpe_vs_baseline": -0.046,
            },
        ]
    ).to_csv(paths.metrics_dir / "walkforward_regime_feature_ablation_comparison.csv", index=False)


def test_missing_optional_inputs_are_skipped_cleanly(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_main_summary(paths)

    result = generate_final_report_outputs(paths)

    assert paths.report_tables_dir.joinpath(MAIN_TABLE_FILENAME).exists()
    assert paths.report_tables_dir.joinpath(COST_TABLE_FILENAME).exists()
    assert paths.report_figures_dir.joinpath(COST_FIGURE_FILENAME).exists()
    assert not paths.report_tables_dir.joinpath(SUPPLEMENTARY_ROBUSTNESS_TABLE_FILENAME).exists()
    assert not paths.report_tables_dir.joinpath(FINAL_MODEL_HIERARCHY_TABLE_FILENAME).exists()
    assert any("walkforward_regime_robustness.csv" in message for message in result.skipped)
    assert any("walkforward_predictions.parquet" in message for message in result.skipped)
    assert any("model_panel.parquet" in message for message in result.skipped)
    assert any("classical_baseline_summary.csv" in message for message in result.skipped)
    assert any("classical_baseline_difference_tests.csv" in message for message in result.skipped)
    assert any("walkforward_excess_model_summary.csv" in message for message in result.skipped)
    assert any("walkforward_mlp_multiseed_summary.csv" in message for message in result.skipped)
    assert any(
        "walkforward_regime_feature_ablation_comparison.csv" in message
        for message in result.skipped
    )


def test_same_protocol_baseline_tables_are_written_when_present(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_main_summary(paths)
    _write_same_protocol_baseline_metrics(paths)

    result = generate_final_report_outputs(paths)

    assert paths.report_tables_dir.joinpath(CLASSICAL_BASELINE_TABLE_FILENAME).exists()
    assert paths.report_tables_dir.joinpath(CLASSICAL_BASELINE_DIFFERENCE_TABLE_FILENAME).exists()
    assert paths.report_tables_dir.joinpath(FINAL_MODEL_HIERARCHY_TABLE_FILENAME).exists()
    assert any(path.name == CLASSICAL_BASELINE_TABLE_FILENAME for path in result.written)
    assert any(path.name == CLASSICAL_BASELINE_DIFFERENCE_TABLE_FILENAME for path in result.written)
    assert any(path.name == FINAL_MODEL_HIERARCHY_TABLE_FILENAME for path in result.written)


def test_main_summary_table_is_sorted_by_main_net_sharpe(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_main_summary(paths)

    generate_final_report_outputs(paths)
    table = pd.read_csv(paths.report_tables_dir / MAIN_TABLE_FILENAME)

    assert list(table["model"]) == ["mlp", "xgboost", "ridge"]
    assert list(table["net_SR"]) == [2.0, 1.31, -0.05]


def test_cost_outputs_preserve_fractional_bps(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_main_summary(paths)

    generate_final_report_outputs(paths)
    cost_table = pd.read_csv(paths.report_tables_dir / COST_TABLE_FILENAME)

    assert "net_SR_7p5bps" in cost_table.columns
    assert paths.report_figures_dir.joinpath(COST_FIGURE_FILENAME).exists()


def test_supplementary_robustness_table_is_written_when_optional_inputs_exist(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_main_summary(paths)
    _write_same_protocol_baseline_metrics(paths)
    _write_supplementary_metrics(paths)

    result = generate_final_report_outputs(paths)

    assert any(path.name == SUPPLEMENTARY_ROBUSTNESS_TABLE_FILENAME for path in result.written)
    table = pd.read_csv(paths.report_tables_dir / SUPPLEMENTARY_ROBUSTNESS_TABLE_FILENAME)

    assert set(table["experiment"]) == {
        "excess_target",
        "mlp_multiseed",
        "regime_feature_ablation",
    }

    excess = table.loc[table["experiment"] == "excess_target"].iloc[0]
    assert excess["target"] == "target_excess_ret_fwd_1m"
    assert excess["net_sharpe"] == pytest.approx(2.9699)
    assert excess["breakeven_bps"] == pytest.approx(123.97)

    multiseed = table.loc[table["experiment"] == "mlp_multiseed"].iloc[0]
    assert multiseed["seed_count"] == 3
    assert multiseed["mean_net_sharpe"] == pytest.approx(1.9683)
    assert multiseed["std_net_sharpe"] == pytest.approx(0.2268)
    assert multiseed["min_net_sharpe"] == pytest.approx(1.7174)
    assert multiseed["max_net_sharpe"] == pytest.approx(2.1589)

    regime = table.loc[
        (table["experiment"] == "regime_feature_ablation")
        & (table["feature_set"] == "flow_compustat_regime")
        & (table["model"] == "mlp")
    ].iloc[0]
    assert regime["reference_feature_set"] == "flow_compustat"
    assert regime["reference_model"] == "mlp"
    assert regime["reference_net_sharpe"] == pytest.approx(2.028)
    assert regime["delta_net_sharpe"] == pytest.approx(-0.508)


def test_final_model_hierarchy_combines_classical_and_ml_rows(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_main_summary(paths)
    _write_same_protocol_baseline_metrics(paths)
    main = pd.read_csv(paths.metrics_dir / "walkforward_model_summary.csv")
    main = pd.concat(
        [
            main,
            pd.DataFrame(
                [
                    {
                        "model_name": "return_only__mlp",
                        "feature_set": "return_only",
                        "model": "mlp",
                        "cutoff": 0.10,
                        "cost_bps": 10.0,
                        "gross_sharpe": 1.50,
                        "net_sharpe": 1.19,
                        "net_ann_return": 0.123,
                        "avg_turnover": 2.76,
                        "breakeven_bps": 47.2,
                        "rank_ic_mean": 0.024,
                        "rank_ic_hac_tstat": 4.04,
                    },
                    {
                        "model_name": "flow_compustat__xgboost",
                        "feature_set": "flow_compustat",
                        "model": "xgboost",
                        "cutoff": 0.10,
                        "cost_bps": 10.0,
                        "gross_sharpe": 1.99,
                        "net_sharpe": 1.77,
                        "net_ann_return": 0.213,
                        "avg_turnover": 2.18,
                        "breakeven_bps": 91.4,
                        "rank_ic_mean": 0.026,
                        "rank_ic_hac_tstat": 3.54,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    main.to_csv(paths.metrics_dir / "walkforward_model_summary.csv", index=False)

    generate_final_report_outputs(paths)
    table = pd.read_csv(paths.report_tables_dir / FINAL_MODEL_HIERARCHY_TABLE_FILENAME)

    assert list(table["strategy"]) == [
        "reversal",
        "zscore_reversal",
        "momentum",
        "reversal_plus_momentum",
        "return_only x xgboost",
        "return_only x mlp",
        "flow_compustat x xgboost",
        "flow_compustat x mlp",
    ]
    assert set(table["role"]) == {
        "classical_baseline",
        "return_only_ml",
        "flow_compustat_ml",
    }


def test_report_style_main_summary_aliases_are_normalized(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_main_summary_report_style(paths)

    generate_final_report_outputs(paths)
    table = pd.read_csv(paths.report_tables_dir / MAIN_TABLE_FILENAME)

    assert list(table.columns) == [
        "feature_set",
        "model",
        "gross_SR",
        "net_SR",
        "net_ann",
        "turnover",
        "breakeven_bps",
        "rank_IC",
        "IC_t_HAC",
    ]
    assert list(table["model"]) == ["mlp", "xgboost"]
    assert list(table["net_SR"]) == [2.0, 1.31]


def test_prediction_deciles_and_data_summary_are_written_when_inputs_exist(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_main_summary(paths)
    _write_predictions(paths)
    _write_model_panel(paths)

    generate_final_report_outputs(paths)

    deciles = pd.read_csv(paths.report_tables_dir / DECILE_TABLE_FILENAME)
    summary = pd.read_csv(paths.report_tables_dir / DATA_SUMMARY_TABLE_FILENAME)

    assert set(deciles["decile"]) == set(range(1, 11))
    assert "top_minus_bottom_mean" in deciles.columns
    assert summary.loc[0, "stock_months"] == 10
    assert summary.loc[0, "n_features"] == 2
    assert paths.report_figures_dir.joinpath(DECILE_FIGURE_FILENAME).exists()


def test_archive_like_input_paths_are_rejected(tmp_path: Path) -> None:
    archive_root = tmp_path / "data" / "outputs_archive" / "metrics"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / "walkforward_model_summary.csv"

    try:
        validate_report_input_path(archive_path, allowed_root=archive_root)
    except ValueError as exc:
        assert "archive-like path" in str(exc)
    else:
        raise AssertionError("archive-like paths must be rejected")
