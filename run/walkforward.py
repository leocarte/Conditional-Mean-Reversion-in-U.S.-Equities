"""Expanding-window walk-forward runner for the final audited evaluation.

Tests whether the benchmark and Compustat-augmented model rankings are stable across time by
re-fitting and re-selecting hyperparameters PER TEST YEAR on an expanding window,
instead of the single locked split. This is pure time-robustness on the PROVIDED
panel:

* no external CRSP / WRDS / re-export;
* no price / share-code / exchange / microcap / liquidity screens (that data does
  not exist in the provided panel);
* target construction, global split/backtest/cost logic, and ``configs/main.yaml``
  defaults are unchanged;
* no test-year tuning -- per fold, candidates are fit on TRAIN only, selected on
  VALIDATION only (net Sharpe at ``cfg.costs.main_bps`` using the top/bottom-decile
  long-short), refit on TRAIN+VALIDATION, and ONLY the test-year rows are emitted
  as out-of-sample predictions.

Fold schedule -- for each test year ``Y`` in 2017..2024::

    train = panel rows with year <= Y - 6     (>= panel start, ~1985)
    val   = panel rows with Y-5 <= year <= Y-1
    test  = panel rows with year == Y

  e.g. test 2017 -> train 1985-2011, val 2012-2016, test 2017;
       test 2024 -> train 1985-2018, val 2019-2023, test 2024.

Reuses (does NOT duplicate): ``build_model_panel`` + ``_with_compustat_overrides``
(panels), ``build_return_only_feature_list`` / ``build_flow_compustat_feature_list``
(features), ``LinearFeaturePreprocessor`` (winsorize+impute+zscore+missing-flags),
``RidgeEig`` / sklearn ``ElasticNet`` / ``XGBoostModel`` / ``MLPRegressor`` (models),
and the shared backtest stack (``_prediction_frame`` -> ``run_prediction_backtests``
-> ``summarize_backtests_by_cost``) for the validation net-Sharpe selection.

Outputs (all git-ignored under ``data/outputs/``)::

    predictions/walkforward_predictions.parquet
    metrics/walkforward_validation_grid.csv
    metrics/walkforward_selected_hparams.csv
    backtests/walkforward_backtests.parquet
    metrics/walkforward_model_summary.csv
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import ElasticNet

from mlfinance.features.regime_features import add_regime_features
from mlfinance.models.linear_preprocessing import LinearFeaturePreprocessor
from mlfinance.models.mlp_tabular import MLPRegressor
from mlfinance.models.ridge import RidgeEig
from mlfinance.models.xgb import XGBoostModel
from mlfinance.run.baseline_pipeline import build_model_panel, summarize_backtests_by_cost
from mlfinance.run.flow_compustat_linear import (
    _with_compustat_overrides,
    build_flow_compustat_feature_list,
)
from mlfinance.run.linear_benchmarks import (
    _prediction_frame,
    _write_csv,
    build_return_only_feature_list,
    run_prediction_backtests,
    summarize_prediction_outputs,
)
from mlfinance.utils.io import write_parquet
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

OUTPUT_PREFIX = "walkforward"
DEFAULT_SEED = 1
DEFAULT_TEST_YEARS: tuple[int, ...] = tuple(range(2017, 2025))
FEATURE_SETS: tuple[str, ...] = ("return_only", "flow_compustat")
MODELS: tuple[str, ...] = ("ridge", "elastic_net", "xgboost", "mlp")

# Optional regime-feature ablation (walk-forward). These feature sets are NOT in the
# default FEATURE_SETS -- they are reached only via an explicit override and write
# under their own output prefix so the accepted final audited version artefacts are never touched.
REGIME_PREFIX = "regime_features"
REGIME_FEATURE_COLS: tuple[str, ...] = (
    "mkt_ret_12m",
    "mkt_vol_12m",
    "mkt_drawdown_12m",
    "cross_sectional_dispersion_1m",
    "high_vol_regime",
    "bear_market_regime",
)
# Each regime feature set extends an accepted base feature set with the block above.
REGIME_BASELINE_MAP: dict[str, str] = {
    "return_regime": "return_only",
    "flow_compustat_regime": "flow_compustat",
}
# Defensive: never let a label / future / realized / prediction column into the
# model feature matrix (the accepted builders already exclude these; the regime
# block is clean, but we assert it).
_FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "target",
    "fwd",
    "prediction",
    "realized",
    "y_true",
    "y_pred",
)

# Per-fold random-search spaces (overnight = single seed; multi-seed MLP later).
XGB_SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [300, 800, 1500],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "min_child_weight": [10, 25, 50],
    "reg_lambda": [1, 10, 50],
    "reg_alpha": [0, 0.1, 1],
}
MLP_SEARCH_SPACE: dict[str, list[Any]] = {
    "hidden_dims": [(128, 64), (256, 128), (256, 128, 64)],
    "dropout": [0.0, 0.05, 0.1, 0.2],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3],
    "batch_size": [4096, 8192, 16384],
    "loss": ["mse", "huber"],
    "patience": [10, 20],
}
N_XGB_CANDIDATES = 16  # in [12, 20]
N_MLP_CANDIDATES = 12  # in [8, 15]
MLP_MAX_EPOCHS = 60  # early stopping (patience) caps effective epochs

# Smoke caps so the synthetic-panel test stays fast/deterministic.
SMOKE_N_NONLINEAR = 2
SMOKE_XGB_N_ESTIMATORS = 30
SMOKE_MLP_EPOCHS = 3

# Post-processing (aggregate) grids. Applied to the stitched OOS predictions ONLY
# -- they never affect hyperparameter selection (which uses cfg.costs.main_bps on
# the validation split). Cutoffs are the top/bottom fraction -> n_buckets 20/10/5.
FINE_COST_GRID: tuple[float, ...] = (0, 1, 2, 5, 7.5, 10, 12.5, 15, 20, 25, 35, 50, 75, 100)
CUTOFFS: tuple[float, ...] = (0.05, 0.10, 0.20)


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------
def yearly_expanding_folds(
    panel: pd.DataFrame,
    *,
    date_col: str,
    test_years: tuple[int, ...],
    train_start: str | None = None,
) -> list[dict[str, Any]]:
    """Expanding-window folds: for each test year Y, train<=Y-6, val Y-5..Y-1, test==Y.

    When ``train_start`` is provided (``cfg.splits.locked.train_start``) the train
    window is floored at that date, so the walk-forward honours the configured
    project start (e.g. 1985-01-31) instead of the earlier lookback-buffer rows the
    panel builder loads (``build_model_panel`` starts at ``train_start`` minus the
    feature-history lookback, so the raw panel reaches ~12 months before 1985).

    Returns one dict per fold with boolean masks (aligned to ``panel`` row order)
    and the ISO month-end date boundaries actually present in each window.
    """
    years = pd.to_datetime(panel[date_col]).dt.year
    dates = pd.to_datetime(panel[date_col])
    train_floor = pd.to_datetime(train_start) if train_start is not None else None

    def _bounds(mask: pd.Series) -> tuple[str | None, str | None]:
        sel = dates.loc[mask]
        if sel.empty:
            return None, None
        return sel.min().date().isoformat(), sel.max().date().isoformat()

    folds: list[dict[str, Any]] = []
    for y in test_years:
        train_mask = years <= (y - 6)
        if train_floor is not None:
            train_mask = train_mask & (dates >= train_floor)
        val_mask = (years >= (y - 5)) & (years <= (y - 1))
        test_mask = years == y
        train_start, train_end = _bounds(train_mask)
        val_start, val_end = _bounds(val_mask)
        test_start, test_end = _bounds(test_mask)
        folds.append(
            {
                "fold_id": int(y),
                "train_mask": train_mask.to_numpy(),
                "val_mask": val_mask.to_numpy(),
                "test_mask": test_mask.to_numpy(),
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return folds


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def _sample_search(space: dict[str, list[Any]], n: int, rng: np.random.Generator) -> list[dict]:
    """Sample ``n`` configs from ``space`` (one value per key), de-duplicated."""
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    # Cap attempts so a small space cannot loop forever chasing n uniques.
    for _ in range(n * 20):
        if len(out) >= n:
            break
        cfg_d = {k: choices[int(rng.integers(len(choices)))] for k, choices in space.items()}
        key = tuple(sorted((k, tuple(v) if isinstance(v, tuple) else v) for k, v in cfg_d.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cfg_d)
    return out


def _candidates(
    model_name: str, cfg: DictConfig, rng: np.random.Generator, smoke: bool
) -> list[dict]:
    """Candidate hyperparameter dicts for a model family (linear=config grid, nonlinear=random)."""
    if model_name == "ridge":
        grid = [float(z) for z in cfg.models.ridge.shrinkage_grid]
        if smoke:
            grid = grid[:SMOKE_N_NONLINEAR] or [1.0]
        return [{"shrinkage": z} for z in grid]
    if model_name == "elastic_net":
        alphas = [float(a) for a in cfg.models.elastic_net.alpha_grid]
        l1s = [float(x) for x in cfg.models.elastic_net.l1_ratio_grid]
        cands = [{"alpha": a, "l1_ratio": x} for a in alphas for x in l1s]
        return cands[:SMOKE_N_NONLINEAR] if smoke else cands
    if model_name == "xgboost":
        n = SMOKE_N_NONLINEAR if smoke else N_XGB_CANDIDATES
        cands = _sample_search(XGB_SEARCH_SPACE, n, rng)
        if smoke:
            for c in cands:
                c["n_estimators"] = SMOKE_XGB_N_ESTIMATORS
        return cands
    if model_name == "mlp":
        n = SMOKE_N_NONLINEAR if smoke else N_MLP_CANDIDATES
        return _sample_search(MLP_SEARCH_SPACE, n, rng)
    raise ValueError(f"Unknown model {model_name!r}.")


# ---------------------------------------------------------------------------
# Model fit / predict dispatch (uniform over the four families)
# ---------------------------------------------------------------------------
def _build_model(
    model_name: str,
    params: dict[str, Any],
    cfg: DictConfig,
    *,
    epochs: int,
    seed: int = DEFAULT_SEED,
) -> Any:
    if model_name == "ridge":
        return RidgeEig(shrinkage_grid=[float(params["shrinkage"])])
    if model_name == "elastic_net":
        return ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=int(cfg.models.elastic_net.max_iter),
            random_state=int(seed),
            fit_intercept=True,
        )
    if model_name == "xgboost":
        return XGBoostModel(seed=int(seed), **params)
    if model_name == "mlp":
        p = dict(params)
        hidden_dims = list(p.pop("hidden_dims"))
        return MLPRegressor(
            hidden_dims=hidden_dims,
            epochs=int(epochs),
            n_seeds=1,
            seed=int(seed),
            device="auto",
            **p,
        )
    raise ValueError(f"Unknown model {model_name!r}.")


def _fit_candidate(
    model_name: str,
    params: dict[str, Any],
    cfg: DictConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    mlp_epochs: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, int | None]:
    """Fit one candidate on TRAIN only, return (validation predictions, mlp best_epoch)."""
    model = _build_model(model_name, params, cfg, epochs=mlp_epochs, seed=seed)
    if model_name == "mlp":
        model.fit(x_train, y_train, x_val, y_val)  # early-stop on validation MSE
        return model.predict(x_val), getattr(model, "best_epoch_", None)
    model.fit(x_train, y_train)
    return model.predict(x_val), None


def _en_fit_predict(
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    max_iter: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, int | None]:
    """Fit one ElasticNet candidate on TRAIN, predict VAL.

    Module-level so joblib's loky backend can pickle it; the fit is deterministic
    (``random_state``), so the parallel result is identical to the sequential fit.
    """
    model = ElasticNet(
        alpha=float(params["alpha"]),
        l1_ratio=float(params["l1_ratio"]),
        max_iter=int(max_iter),
        random_state=int(seed),
        fit_intercept=True,
    )
    model.fit(x_train, y_train)
    return model.predict(x_val), None


def _fit_candidates(
    model_name: str,
    candidates: list[dict[str, Any]],
    cfg: DictConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    mlp_epochs: int,
    en_n_jobs: int,
    seed: int = DEFAULT_SEED,
) -> list[tuple[np.ndarray, int | None]]:
    """Fit every candidate on TRAIN, predict VAL.

    The Elastic Net sweep (the CPU bottleneck -- many independent coordinate-descent
    fits) runs in parallel with joblib; ``inner_max_num_threads=1`` pins each
    worker's BLAS so ``en_n_jobs`` workers do not oversubscribe the pod BLAS pool.
    Ridge is cheap and XGBoost / MLP are GPU-bound (a single GPU), so those stay
    sequential. Output is identical to the sequential sweep -- each fit is
    deterministic and joblib preserves input order.
    """
    if model_name == "elastic_net" and en_n_jobs > 1 and len(candidates) > 1:
        max_iter = int(cfg.models.elastic_net.max_iter)
        return list(
            Parallel(n_jobs=en_n_jobs, backend="loky", inner_max_num_threads=1)(
                delayed(_en_fit_predict)(p, x_train, y_train, x_val, max_iter, seed)
                for p in candidates
            )
        )
    return [
        _fit_candidate(
            model_name, p, cfg, x_train, y_train, x_val, y_val, mlp_epochs=mlp_epochs, seed=seed
        )
        for p in candidates
    ]


def _refit_predict(
    model_name: str,
    params: dict[str, Any],
    cfg: DictConfig,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_test: np.ndarray,
    *,
    best_epoch: int | None,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Refit the selected candidate on TRAIN+VALIDATION, predict TEST rows."""
    epochs = max(int(best_epoch), 1) if best_epoch is not None else MLP_MAX_EPOCHS
    model = _build_model(model_name, params, cfg, epochs=epochs, seed=seed)
    model.fit(x_fit, y_fit)  # no validation frame (refit uses the validated budget)
    return model.predict(x_test)


# ---------------------------------------------------------------------------
# Validation scoring (net Sharpe at main_bps + rank-IC tie-breaker)
# ---------------------------------------------------------------------------
def _score_validation(
    val_frame: pd.DataFrame, y_val_pred: np.ndarray, cfg: DictConfig, model_tag: str
) -> tuple[float, float]:
    """Validation net Sharpe at ``cfg.costs.main_bps`` and rank-IC mean for one candidate."""
    preds = _prediction_frame(
        val_frame,
        y_val_pred,
        cfg=cfg,
        model_name=model_tag,
        split="validation",
        split_id=f"{OUTPUT_PREFIX}_val",
    )
    backtests = run_prediction_backtests(preds, cfg, split_id=f"{OUTPUT_PREFIX}_val")
    cost = summarize_backtests_by_cost(backtests)
    main_bps = int(cfg.costs.main_bps)
    row = cost.loc[cost["cost_bps"] == main_bps]
    net_sharpe = float(row["net_sharpe"].iloc[0]) if not row.empty else float("nan")
    pred_summary = summarize_prediction_outputs(preds, cfg)
    rank_ic = (
        float(pred_summary["rank_ic_mean"].iloc[0])
        if not pred_summary.empty and "rank_ic_mean" in pred_summary
        else float("nan")
    )
    return net_sharpe, rank_ic


def _better(net_a: float, ic_a: float, net_b: float, ic_b: float) -> bool:
    """True if (net_a, ic_a) beats (net_b, ic_b): net Sharpe primary, rank-IC tie-breaker."""

    def _nan_low(x: float) -> float:
        return -np.inf if (x is None or np.isnan(x)) else x

    if _nan_low(net_a) != _nan_low(net_b):
        return _nan_low(net_a) > _nan_low(net_b)
    return _nan_low(ic_a) > _nan_low(ic_b)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _assert_no_forbidden(feature_cols: list[str]) -> list[str]:
    """Guard: no label/future/realized/prediction column may enter the feature matrix."""
    bad = [
        c for c in feature_cols if any(tok in str(c).lower() for tok in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if bad:
        raise ValueError(f"Refusing to use future/label columns as features: {bad}")
    return feature_cols


def _add_regime_block(
    panel: pd.DataFrame, base_feature_cols: list[str], cfg: DictConfig
) -> tuple[pd.DataFrame, list[str]]:
    """Attach the lagged market-regime block and extend the feature list.

    Regime columns are all month-``t``-end information (trailing market windows and
    the std of CONTEMPORANEOUS month-``t`` cross-sectional returns -- the label is the
    SEPARATE ``shift(-1)`` column), so they are known before the ``t+1`` portfolio is
    formed -> no look-ahead, no one-month lag needed (same convention as the accepted
    return features like ``ret_1m``).
    """
    panel = add_regime_features(
        panel,
        date_col=cfg.data.date_col,
        ret_col=cfg.data.ret_col,
        market_col=cfg.data.market_col,
    )
    regime_cols = [c for c in REGIME_FEATURE_COLS if c in panel.columns]
    return panel, _assert_no_forbidden(list(base_feature_cols) + regime_cols)


def _build_panel(cfg: DictConfig, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Build the panel + model feature list for a feature set, reusing existing builders."""
    if feature_set == "return_only":
        panel = build_model_panel(cfg)
        return panel, build_return_only_feature_list(panel)
    if feature_set == "flow_compustat":
        panel = build_model_panel(_with_compustat_overrides(cfg))
        # include_missingness_flags=False mirrors the feature-audit dedup fix (the
        # preprocessor regenerates the *_missing indicators itself).
        return panel, build_flow_compustat_feature_list(panel, include_missingness_flags=False)
    if feature_set == "return_regime":
        panel = build_model_panel(cfg)
        return _add_regime_block(panel, build_return_only_feature_list(panel), cfg)
    if feature_set == "flow_compustat_regime":
        panel = build_model_panel(_with_compustat_overrides(cfg))
        base = build_flow_compustat_feature_list(panel, include_missingness_flags=False)
        return _add_regime_block(panel, base, cfg)
    raise ValueError(f"Unknown feature_set {feature_set!r}.")


def run_walkforward(
    cfg: DictConfig,
    *,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
    models: tuple[str, ...] = MODELS,
    test_years: tuple[int, ...] = DEFAULT_TEST_YEARS,
    shard_tag: str | None = None,
    target_col: str | None = None,
    output_prefix: str = OUTPUT_PREFIX,
    seed: int = DEFAULT_SEED,
    smoke: bool = False,
) -> dict[str, Path]:
    """Run the expanding-window walk-forward; write OOS predictions + diagnostics.

    Optional overrides (additive; defaults reproduce the accepted final audited version run byte-for-byte):
    ``target_col`` selects a different EXISTING label column in-memory (e.g.
    ``target_excess_ret_fwd_1m``) for both training and the validation/OOS backtest --
    it does NOT change target construction or ``configs/main.yaml`` defaults; ``seed``
    re-seeds candidate sampling + every model fit (multi-seed stability); ``output_prefix``
    isolates the output filenames so a robustness rerun never overwrites the accepted
    main-target audited version artefacts.
    """
    if target_col is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"target_col": str(target_col)}}))
    date_col = cfg.data.date_col
    permno_col = cfg.data.permno_col
    target_col = cfg.data.target_col
    robust_col = cfg.data.robustness_target_col
    metrics_dir = Path(cfg.paths.metrics)
    rng = np.random.default_rng(int(seed))
    mlp_epochs = SMOKE_MLP_EPOCHS if smoke else MLP_MAX_EPOCHS
    # Parallelise the Elastic Net candidate sweep (the CPU bottleneck). Read the
    # shared `models.elastic_net.n_jobs` knob if present, else default to 6; smoke
    # uses 1 to avoid loky spawn overhead on the tiny synthetic panel.
    en_n_jobs = 1 if smoke else int(OmegaConf.select(cfg, "models.elastic_net.n_jobs", default=6))

    oos_rows: list[pd.DataFrame] = []
    grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for feature_set in feature_sets:
        panel, feature_cols = _build_panel(cfg, feature_set)
        panel = panel.sort_values([date_col, permno_col]).reset_index(drop=True)
        folds = yearly_expanding_folds(
            panel,
            date_col=date_col,
            test_years=test_years,
            train_start=OmegaConf.select(cfg, "splits.locked.train_start", default=None),
        )

        for fold in folds:
            train_frame = panel.loc[fold["train_mask"]].reset_index(drop=True)
            val_frame = panel.loc[fold["val_mask"]].reset_index(drop=True)
            test_frame = panel.loc[fold["test_mask"]].reset_index(drop=True)
            if train_frame.empty or val_frame.empty or test_frame.empty:
                logger.warning(
                    "walk-forward WF: skipping fold %s/%s (empty split: train=%d val=%d test=%d).",
                    feature_set,
                    fold["fold_id"],
                    len(train_frame),
                    len(val_frame),
                    len(test_frame),
                )
                continue

            # Train-only preprocessor for the validation sweep; a separate
            # train+validation preprocessor for the refit -- neither ever sees test.
            pre = LinearFeaturePreprocessor(feature_cols, date_col=date_col)
            x_train = pre.fit_transform(train_frame).to_numpy()
            x_val = pre.transform(val_frame).to_numpy()
            y_train = train_frame[target_col].to_numpy(dtype="float64")
            y_val = val_frame[target_col].to_numpy(dtype="float64")

            fit_frame = pd.concat([train_frame, val_frame], ignore_index=True)
            pre_fit = LinearFeaturePreprocessor(feature_cols, date_col=date_col)
            x_fit = pre_fit.fit_transform(fit_frame).to_numpy()
            x_test = pre_fit.transform(test_frame).to_numpy()
            y_fit = fit_frame[target_col].to_numpy(dtype="float64")

            for model_name in models:
                candidates = _candidates(model_name, cfg, rng, smoke)
                cand_results = _fit_candidates(
                    model_name,
                    candidates,
                    cfg,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    mlp_epochs=mlp_epochs,
                    en_n_jobs=en_n_jobs,
                    seed=seed,
                )
                best: dict[str, Any] | None = None
                for rank, (params, (y_val_pred, best_epoch)) in enumerate(
                    zip(candidates, cand_results, strict=True)
                ):
                    net_sharpe, rank_ic = _score_validation(
                        val_frame, y_val_pred, cfg, f"{feature_set}__{model_name}"
                    )
                    grid_rows.append(
                        {
                            "feature_set": feature_set,
                            "model_name": model_name,
                            "fold_id": fold["fold_id"],
                            "candidate_rank": rank,
                            "params_json": json.dumps(_jsonable(params), sort_keys=True),
                            "val_net_sharpe": net_sharpe,
                            "val_rank_ic": rank_ic,
                            "best_epoch": best_epoch,
                            "selection_cost_bps": int(cfg.costs.main_bps),
                        }
                    )
                    if best is None or _better(
                        net_sharpe, rank_ic, best["net_sharpe"], best["rank_ic"]
                    ):
                        best = {
                            "params": params,
                            "net_sharpe": net_sharpe,
                            "rank_ic": rank_ic,
                            "best_epoch": best_epoch,
                        }

                # Refit the selected candidate on train+validation; predict test only.
                y_test_pred = _refit_predict(
                    model_name,
                    best["params"],
                    cfg,
                    x_fit,
                    y_fit,
                    x_test,
                    best_epoch=best["best_epoch"],
                    seed=seed,
                )
                hparams_json = json.dumps(_jsonable(best["params"]), sort_keys=True)
                out = pd.DataFrame(
                    {
                        permno_col: test_frame[permno_col].to_numpy(),
                        date_col: test_frame[date_col].to_numpy(),
                        target_col: test_frame[target_col].to_numpy(),
                        "prediction": np.asarray(y_test_pred, dtype="float64"),
                        "model_name": model_name,
                        "feature_set": feature_set,
                        "target_col": target_col,
                        "fold_id": fold["fold_id"],
                        "train_start": fold["train_start"],
                        "train_end": fold["train_end"],
                        "val_start": fold["val_start"],
                        "val_end": fold["val_end"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                        "selected_hparams_json": hparams_json,
                    }
                )
                if robust_col in test_frame.columns:
                    out[robust_col] = test_frame[robust_col].to_numpy()
                oos_rows.append(out)
                selected_rows.append(
                    {
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "fold_id": fold["fold_id"],
                        "selected_hparams_json": hparams_json,
                        "val_net_sharpe": best["net_sharpe"],
                        "val_rank_ic": best["rank_ic"],
                        "best_epoch": best["best_epoch"],
                        "train_start": fold["train_start"],
                        "train_end": fold["train_end"],
                        "val_start": fold["val_start"],
                        "val_end": fold["val_end"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                    }
                )
                logger.info(
                    "walk-forward WF: %s/%s fold %s selected %s (val net Sharpe=%.3f).",
                    feature_set,
                    model_name,
                    fold["fold_id"],
                    hparams_json,
                    best["net_sharpe"],
                )

    if not oos_rows:
        raise ValueError(
            "walk-forward walk-forward produced no OOS predictions (all folds empty?)."
        )

    predictions = pd.concat(oos_rows, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    # Robustness reruns (any non-canonical output_prefix) carry a `seed` column so a
    # multi-seed aggregate can tell seeds apart. The canonical default run is left
    # BYTE-IDENTICAL to the accepted final audited version outputs (no extra column, no value change).
    if output_prefix != OUTPUT_PREFIX:
        predictions["seed"] = int(seed)
        selected_df["seed"] = int(seed)
    suffix = f"__{shard_tag}" if shard_tag else ""
    predictions_path = Path(cfg.paths.predictions) / f"{output_prefix}_predictions{suffix}.parquet"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)

    grid_path = _write_csv(
        pd.DataFrame(grid_rows), metrics_dir / f"{output_prefix}_validation_grid{suffix}.csv"
    )
    selected_path = _write_csv(
        selected_df, metrics_dir / f"{output_prefix}_selected_hparams{suffix}.csv"
    )
    outputs: dict[str, Path] = {
        "predictions": predictions_path,
        "validation_grid": grid_path,
        "selected_hparams": selected_path,
    }

    if shard_tag:
        # Fan-out shard: write only this shard's slice and defer the global
        # backtest / cost-grid / breakeven to the aggregate step, so concurrent
        # shards never collide on the summary artefacts.
        logger.info(
            "walk-forward WF shard '%s' done: %d OOS rows (backtest deferred to aggregate).",
            shard_tag,
            len(predictions),
        )
        return outputs

    # Single (non-sharded) run: backtest the stitched OOS predictions over the
    # fine cost grid x cutoffs + breakeven. Reuses the shared backtest stack.
    backtests, model_summary = _backtest_oos_grid(predictions, cfg)
    backtests_path = Path(cfg.paths.backtests) / f"{output_prefix}_backtests.parquet"
    backtests_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(backtests, backtests_path)
    summary_path = _write_csv(model_summary, metrics_dir / f"{output_prefix}_model_summary.csv")
    outputs["backtests"] = backtests_path
    outputs["model_summary"] = summary_path

    logger.info(
        "walk-forward walk-forward done: %d OOS rows across %d strategies, %d folds.",
        len(predictions),
        predictions.groupby(["feature_set", "model_name"]).ngroups,
        len(test_years),
    )
    return outputs


def _build_oos_backtest_input(predictions: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Stitched per-strategy prediction frame for the shared backtest stack.

    Each ``feature_set x model_name`` pair becomes one ``model_name`` tag
    (``f"{feature_set}__{model_name}"``) so the canonical backtester scores every
    walk-forward strategy as its own book.
    """
    bt_frames: list[pd.DataFrame] = []
    for (feature_set, model_name), group in predictions.groupby(
        ["feature_set", "model_name"], sort=True
    ):
        strategy = f"{feature_set}__{model_name}"
        frame = _prediction_frame(
            group.reset_index(drop=True),
            group["prediction"].to_numpy(dtype="float64"),
            cfg=cfg,
            model_name=strategy,
            split="test",
            split_id=f"{OUTPUT_PREFIX}_oos",
        )
        bt_frames.append(frame)
    return pd.concat(bt_frames, ignore_index=True)


def _cutoff_cfg(cfg: DictConfig, n_buckets: int) -> DictConfig:
    """Copy of ``cfg`` cut at ``n_buckets`` with a degenerate ``bps_grid=[0]``.

    The fine cost grid is applied analytically in :func:`_expand_costs`, so the
    backtester is invoked once per cutoff at zero cost. Never mutates the caller's
    config; ``configs/main.yaml`` defaults are untouched.
    """
    return OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "portfolio": {
                    "n_buckets": int(n_buckets),
                    "long_bucket": int(n_buckets),
                    "short_bucket": 1,
                },
                "costs": {"bps_grid": [0]},
            }
        ),
    )


def _expand_costs(base_monthly: pd.DataFrame, costs_bps: tuple[float, ...]) -> pd.DataFrame:
    """Replicate the monthly book at every cost in ``costs_bps`` (float-exact).

    ``gross_return`` and ``turnover`` are cost-independent, so net returns for the
    whole fine grid are a deterministic transform of the single ``bps=0`` book.
    Mirrors the cost arithmetic in ``run_prediction_backtests`` but keeps fractional
    bps (e.g. 7.5, 12.5) instead of truncating to int.
    """
    keep = [
        "date",
        "month",
        "model_name",
        "split_id",
        "gross_return",
        "turnover",
        "long_return",
        "short_return",
        "n_long",
        "n_short",
    ]
    base = base_monthly[[c for c in keep if c in base_monthly.columns]].copy()
    out: list[pd.DataFrame] = []
    for c in costs_bps:
        frame = base.copy()
        frame["cost_bps"] = float(c)
        frame["cost"] = frame["turnover"] * (float(c) / 10_000.0)
        frame["net_return"] = frame["gross_return"] - frame["cost"]
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def _breakeven_by_strategy(base_monthly: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy breakeven cost ``10000 * mean(gross_return) / mean(turnover)``.

    Computed on the gross (cost-independent) book. Zero / missing / non-finite
    average turnover yields NaN instead of dividing by zero.
    """
    rows: list[dict[str, Any]] = []
    for model_name, frame in base_monthly.groupby("model_name", sort=True):
        mean_gross = float(frame["gross_return"].mean())
        mean_turn = float(frame["turnover"].mean())
        breakeven = (
            10_000.0 * mean_gross / mean_turn
            if np.isfinite(mean_turn) and mean_turn > 0.0
            else float("nan")
        )
        rows.append(
            {
                "model_name": model_name,
                "mean_gross_return": mean_gross,
                "mean_turnover": mean_turn,
                "breakeven_bps": breakeven,
            }
        )
    return pd.DataFrame(rows)


def _backtest_oos_grid(
    predictions: pd.DataFrame, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest the stitched OOS predictions over ``CUTOFFS`` x ``FINE_COST_GRID``.

    Post-processing only -- it never affects hyperparameter selection (which ran on
    the validation split at ``cfg.costs.main_bps``). Returns ``(monthly, summary)``:

    * ``monthly``  -- per (strategy, cutoff, cost_bps, month) gross/net book;
    * ``summary``  -- per (strategy, cutoff, cost_bps) gross/net Sharpe, net ann
      return, avg turnover, breakeven bps, plus the cut-independent rank-IC / OOS-R2.

    A cutoff is skipped when the thinnest month has fewer tradable names than the
    cut needs (both legs could not populate) -- on the real panel every cutoff
    qualifies; on a tiny synthetic panel the finest cut is dropped rather than faked.
    """
    bt_input = _build_oos_backtest_input(predictions, cfg)
    date_col = cfg.data.date_col
    permno_col = cfg.data.permno_col

    # Cut-independent predictive metrics (rank-IC, OOS-R2, errors): computed once.
    pred_summary = summarize_prediction_outputs(bt_input, cfg)
    pred_metrics = pred_summary[
        [
            c
            for c in (
                "model_name",
                "rmse",
                "mae",
                "oos_r2",
                "rank_ic_mean",
                "rank_ic_hac_tstat",
                "n_predictions",
                "n_months",
            )
            if c in pred_summary.columns
        ]
    ]

    min_month_count = int(bt_input.groupby([date_col, "model_name"])[permno_col].count().min())
    eligible = [c for c in CUTOFFS if int(round(1.0 / float(c))) <= min_month_count]
    if not eligible:
        eligible = [max(CUTOFFS)]
        logger.warning(
            "walk-forward WF: thinnest month has %d names; backtesting coarsest cut %.2f only.",
            min_month_count,
            eligible[0],
        )

    fine_costs = tuple(float(c) for c in FINE_COST_GRID)
    int_to_exact = {int(c): c for c in FINE_COST_GRID}  # recover fractional labels

    monthly_blocks: list[pd.DataFrame] = []
    summary_blocks: list[pd.DataFrame] = []
    for cutoff in eligible:
        n_buckets = int(round(1.0 / float(cutoff)))
        cfg_cut = _cutoff_cfg(cfg, n_buckets)
        base_monthly = run_prediction_backtests(bt_input, cfg_cut, split_id=f"{OUTPUT_PREFIX}_oos")
        monthly_fine = _expand_costs(base_monthly, fine_costs)
        monthly_fine.insert(0, "cutoff", float(cutoff))
        monthly_fine["n_buckets"] = n_buckets
        monthly_blocks.append(monthly_fine)

        cost_summary = summarize_backtests_by_cost(monthly_fine)
        cost_summary["cost_bps"] = cost_summary["cost_bps"].map(
            lambda i: int_to_exact.get(int(i), float(i))
        )
        cost_summary.insert(1, "cutoff", float(cutoff))
        cost_summary["n_buckets"] = n_buckets
        cost_summary = cost_summary.merge(
            _breakeven_by_strategy(base_monthly),
            on="model_name",
            how="left",
            validate="many_to_one",
        )
        summary_blocks.append(cost_summary)

    monthly = pd.concat(monthly_blocks, ignore_index=True)
    model_summary = pd.concat(summary_blocks, ignore_index=True).merge(
        pred_metrics, on="model_name", how="left", validate="many_to_one"
    )
    parts = model_summary["model_name"].str.split("__", n=1, expand=True)
    model_summary.insert(1, "feature_set", parts[0])
    model_summary.insert(2, "model", parts[1])
    model_summary = model_summary.sort_values(
        ["feature_set", "model", "cutoff", "cost_bps"]
    ).reset_index(drop=True)
    return monthly, model_summary


def _merge_shard_csvs(
    metrics_dir: Path, name: str, output_prefix: str = OUTPUT_PREFIX
) -> Path | None:
    """Concatenate ``{prefix}_{name}__*.csv`` shards into the canonical CSV (overwrites it).

    The pre-existing canonical CSV is NEVER read back in -- the canonical is a derived
    artifact, only the ``__<tag>`` shards are inputs. This prevents a stale canonical
    file from contaminating a fresh aggregate. To re-aggregate with old data, keep the
    old shard files in place; aggregate will pick them up.
    """
    default = metrics_dir / f"{output_prefix}_{name}.csv"
    shards = sorted(metrics_dir.glob(f"{output_prefix}_{name}__*.csv"))
    if not shards:
        return None
    merged = (
        pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return _write_csv(merged, default)


def summarize_multiseed(predictions: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Per-seed OOS net Sharpe (@ default cut / main_bps) + cross-seed mean/std.

    Multi-seed stability view for a strategy run under several seeds. Each
    (feature_set, model, seed) is scored as its own book via the shared backtest
    stack; an ``ALL(mean+/-std)`` row reports the cross-seed dispersion.
    """
    main_bps = int(cfg.costs.main_bps)
    rows: list[dict[str, Any]] = []
    for (fs, model, seed), g in predictions.groupby(
        ["feature_set", "model_name", "seed"], sort=True
    ):
        frame = _prediction_frame(
            g.reset_index(drop=True),
            g["prediction"].to_numpy(dtype="float64"),
            cfg=cfg,
            model_name=f"{fs}__{model}__seed{seed}",
            split="test",
            split_id=f"{OUTPUT_PREFIX}_multiseed",
        )
        monthly = run_prediction_backtests(frame, cfg, split_id=f"{OUTPUT_PREFIX}_multiseed")
        cost = summarize_backtests_by_cost(monthly)
        row = cost.loc[cost["cost_bps"] == main_bps]
        pred_summary = summarize_prediction_outputs(frame, cfg)
        rows.append(
            {
                "feature_set": fs,
                "model": model,
                "seed": int(seed),
                "n_months": int(pd.to_datetime(g[cfg.data.date_col]).dt.to_period("M").nunique()),
                "gross_sharpe": (
                    float(row["gross_sharpe"].iloc[0]) if not row.empty else float("nan")
                ),
                "net_sharpe": float(row["net_sharpe"].iloc[0]) if not row.empty else float("nan"),
                "net_ann_return": (
                    float(row["net_ann_return"].iloc[0]) if not row.empty else float("nan")
                ),
                "rank_ic_mean": (
                    float(pred_summary["rank_ic_mean"].iloc[0])
                    if not pred_summary.empty and "rank_ic_mean" in pred_summary
                    else float("nan")
                ),
            }
        )
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed
    agg = (
        per_seed.groupby(["feature_set", "model"])
        .agg(
            n_seeds=("seed", "nunique"),
            net_sharpe_mean=("net_sharpe", "mean"),
            net_sharpe_std=("net_sharpe", "std"),
            net_sharpe_min=("net_sharpe", "min"),
            net_sharpe_max=("net_sharpe", "max"),
            rank_ic_mean_mean=("rank_ic_mean", "mean"),
        )
        .reset_index()
    )
    agg["seed"] = "ALL(mean+/-std)"
    return pd.concat([per_seed, agg], ignore_index=True)


def _regime_ablation_comparison(
    summary: pd.DataFrame, cfg: DictConfig, *, cutoff: float = 0.10
) -> pd.DataFrame | None:
    """Compare regime-feature strategies to their accepted-main baselines @ cutoff / main_bps.

    Returns None (and logs) when the summary has no regime feature sets, or when the
    accepted main-target summary (``walkforward_model_summary.csv``) is absent --
    the regime model summary is still written; only the comparison is skipped, never
    failing the aggregate.
    """
    if "feature_set" not in summary.columns or "model" not in summary.columns:
        return None
    regime_fs = [fs for fs in summary["feature_set"].unique() if fs in REGIME_BASELINE_MAP]
    if not regime_fs:
        return None
    main_bps = float(int(cfg.costs.main_bps))
    main_path = Path(cfg.paths.metrics) / f"{OUTPUT_PREFIX}_model_summary.csv"
    if not main_path.exists():
        logger.warning(
            "Regime ablation: accepted main summary %s not found -> skipping comparison "
            "(the regime model summary is still written).",
            main_path,
        )
        return None
    main = pd.read_csv(main_path)

    def _at(df: pd.DataFrame, fs: str, model: str) -> pd.Series | None:
        sel = df[
            (df["feature_set"] == fs)
            & (df["model"] == model)
            & (df["cutoff"].round(3) == round(float(cutoff), 3))
            & (df["cost_bps"].round(3) == round(main_bps, 3))
        ]
        return sel.iloc[0] if not sel.empty else None

    rows: list[dict[str, Any]] = []
    for fs in sorted(regime_fs):
        base_fs = REGIME_BASELINE_MAP[fs]
        for model in sorted(summary.loc[summary["feature_set"] == fs, "model"].unique()):
            r = _at(summary, fs, model)
            if r is None:
                continue
            b = _at(main, base_fs, model)
            base_ns = float(b["net_sharpe"]) if b is not None else float("nan")
            rows.append(
                {
                    "feature_set": fs,
                    "model": model,
                    "cutoff": float(cutoff),
                    "cost_bps": main_bps,
                    "net_sharpe": float(r["net_sharpe"]),
                    "gross_sharpe": float(r["gross_sharpe"]),
                    "net_ann_return": float(r["net_ann_return"]),
                    "breakeven_bps": float(r["breakeven_bps"]),
                    "rank_ic_mean": float(r.get("rank_ic_mean", float("nan"))),
                    "rank_ic_hac_tstat": float(r.get("rank_ic_hac_tstat", float("nan"))),
                    "baseline_feature_set": base_fs,
                    "baseline_net_sharpe": base_ns,
                    "delta_net_sharpe_vs_baseline": float(r["net_sharpe"]) - base_ns,
                }
            )
    return pd.DataFrame(rows) if rows else None


def aggregate_walkforward(
    cfg: DictConfig, *, output_prefix: str = OUTPUT_PREFIX
) -> dict[str, Path]:
    """Stitch fan-out shards and (re)build the summaries, no refit.

    Merges ONLY the ``{output_prefix}_predictions__*.parquet`` shard files (and the
    matching ``__<tag>`` CSV shards). The pre-existing canonical file is NEVER read
    back in -- it is a derived artefact, so a stale canonical cannot contaminate a
    fresh aggregate. De-dups on (feature_set, model_name, fold_id, permno, date) plus
    ``seed`` when present.

    If a ``seed`` column with more than one value is present (a multi-seed run), the
    pooled cost-grid backtest is skipped (it would see duplicate permno-date rows) and
    a per-seed stability summary (:func:`summarize_multiseed`) is written instead.
    """
    pred_dir = Path(cfg.paths.predictions)
    metrics_dir = Path(cfg.paths.metrics)
    permno_col = cfg.data.permno_col
    date_col = cfg.data.date_col

    default_path = pred_dir / f"{output_prefix}_predictions.parquet"
    shard_paths = sorted(pred_dir.glob(f"{output_prefix}_predictions__*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No {output_prefix}_predictions__*.parquet shards under {pred_dir}; "
            "run the fan-out shards first. (Aggregate never reads the canonical file "
            "-- only shards are inputs.)"
        )
    merged = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
    subset = ["feature_set", "model_name", "fold_id", permno_col, date_col]
    if "seed" in merged.columns:
        subset.append("seed")
    predictions = merged.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)

    # Adopt the label the shards were trained/emitted on (the shards persist the
    # realized-return column NAMED by their target plus a "target_col" metadata
    # column). For an excess-target rerun this is "target_excess_ret_fwd_1m", so the
    # backtest must score against THAT column -- otherwise _prediction_frame looks for
    # the default cfg.data.target_col and KeyErrors. In-memory only; no YAML change.
    if "target_col" in predictions.columns:
        tcols = [str(t) for t in predictions["target_col"].dropna().unique()]
        if len(tcols) == 1 and tcols[0] != str(cfg.data.target_col):
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"target_col": tcols[0]}}))

    default_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(default_path, index=False)
    _merge_shard_csvs(metrics_dir, "validation_grid", output_prefix)
    _merge_shard_csvs(metrics_dir, "selected_hparams", output_prefix)

    multiseed = "seed" in predictions.columns and predictions["seed"].nunique() > 1
    if multiseed:
        summary = summarize_multiseed(predictions, cfg)
        summary_path = _write_csv(summary, metrics_dir / f"{output_prefix}_summary.csv")
        logger.info(
            "walk-forward WF multi-seed aggregate: %d shard file(s), %d seeds -> %d OOS rows.",
            len(shard_paths),
            int(predictions["seed"].nunique()),
            len(predictions),
        )
        return {"predictions": default_path, "summary": summary_path}

    backtests, model_summary = _backtest_oos_grid(predictions, cfg)
    backtests_path = Path(cfg.paths.backtests) / f"{output_prefix}_backtests.parquet"
    backtests_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(backtests, backtests_path)
    summary_path = _write_csv(model_summary, metrics_dir / f"{output_prefix}_model_summary.csv")
    logger.info(
        "walk-forward WF aggregate: merged %d shard file(s) -> %d OOS rows.",
        len(shard_paths),
        len(predictions),
    )
    outputs = {
        "predictions": default_path,
        "backtests": backtests_path,
        "model_summary": summary_path,
    }
    # Regime-feature ablation: if these are regime strategies, also emit the
    # vs-accepted-main comparison (skipped gracefully if the main summary is absent).
    comparison = _regime_ablation_comparison(model_summary, cfg)
    if comparison is not None:
        outputs["ablation_comparison"] = _write_csv(
            comparison, metrics_dir / "walkforward_regime_feature_ablation_comparison.csv"
        )
    return outputs


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy / tuple values to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _sanitize_tag(tag: str) -> str:
    """Filename-safe shard tag (keep alnum . _ -, collapse the rest to '-')."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in tag.strip())
    return cleaned.strip("-") or "shard"


def _parse_overrides(cfg: DictConfig) -> dict[str, Any]:
    """Optional run overrides from a transient ``cfg.walkforward`` block and/or
    ``WALKFORWARD_*`` environment variables (env wins). No ``configs/main.yaml`` default.

    Env contract (comma-separated where plural):
      ``WALKFORWARD_FEATURE_SETS``, ``WALKFORWARD_MODELS``, ``WALKFORWARD_TEST_YEARS``,
      ``WALKFORWARD_SHARD_TAG``, ``WALKFORWARD_SMOKE``, and the robustness knobs
      ``WALKFORWARD_TARGET_COL`` (e.g. target_excess_ret_fwd_1m),
      ``WALKFORWARD_OUTPUT_PREFIX`` (isolate outputs), ``WALKFORWARD_SEED``.
    """
    out: dict[str, Any] = {}
    wf = OmegaConf.select(cfg, "walkforward", default=None)
    if wf is not None:
        if OmegaConf.select(wf, "feature_sets", default=None) is not None:
            out["feature_sets"] = tuple(str(x) for x in wf.feature_sets)
        if OmegaConf.select(wf, "models", default=None) is not None:
            out["models"] = tuple(str(x) for x in wf.models)
        if OmegaConf.select(wf, "test_years", default=None) is not None:
            out["test_years"] = tuple(int(x) for x in wf.test_years)
        if OmegaConf.select(wf, "shard_tag", default=None) is not None:
            out["shard_tag"] = _sanitize_tag(str(wf.shard_tag))
        if OmegaConf.select(wf, "smoke", default=None) is not None:
            out["smoke"] = bool(wf.smoke)
        if OmegaConf.select(wf, "target_col", default=None) is not None:
            out["target_col"] = str(wf.target_col)
        if OmegaConf.select(wf, "output_prefix", default=None) is not None:
            out["output_prefix"] = _sanitize_tag(str(wf.output_prefix))
        if OmegaConf.select(wf, "seed", default=None) is not None:
            out["seed"] = int(wf.seed)

    env_fs = os.environ.get("WALKFORWARD_FEATURE_SETS")
    if env_fs:
        out["feature_sets"] = tuple(s.strip() for s in env_fs.split(",") if s.strip())
    env_models = os.environ.get("WALKFORWARD_MODELS")
    if env_models:
        out["models"] = tuple(s.strip() for s in env_models.split(",") if s.strip())
    env_years = os.environ.get("WALKFORWARD_TEST_YEARS")
    if env_years:
        out["test_years"] = tuple(int(s) for s in env_years.split(",") if s.strip())
    env_shard = os.environ.get("WALKFORWARD_SHARD_TAG")
    if env_shard and env_shard.strip():
        out["shard_tag"] = _sanitize_tag(env_shard)
    env_smoke = os.environ.get("WALKFORWARD_SMOKE")
    if env_smoke is not None and env_smoke.strip() != "":
        out["smoke"] = env_smoke.strip().lower() in ("1", "true", "yes", "on")
    env_target = os.environ.get("WALKFORWARD_TARGET_COL")
    if env_target and env_target.strip():
        out["target_col"] = env_target.strip()
    env_prefix = os.environ.get("WALKFORWARD_OUTPUT_PREFIX")
    if env_prefix and env_prefix.strip():
        out["output_prefix"] = _sanitize_tag(env_prefix)
    env_seed = os.environ.get("WALKFORWARD_SEED")
    if env_seed and env_seed.strip():
        out["seed"] = int(env_seed)
    return out


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    configure_logging()
    mode = os.environ.get("WALKFORWARD_MODE", "").strip() or str(
        OmegaConf.select(cfg, "walkforward.mode", default="run")
    )
    overrides = _parse_overrides(cfg)
    if mode == "aggregate":
        outputs = aggregate_walkforward(
            cfg, output_prefix=overrides.get("output_prefix", OUTPUT_PREFIX)
        )
    else:
        outputs = run_walkforward(cfg, **overrides)
    for name, path in outputs.items():
        logger.info("Wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
