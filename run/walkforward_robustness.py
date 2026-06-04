"""walk-forward post-processing robustness package (NO retraining).

Consumes the corrected **audited version** walk-forward OUTPUTS already produced on cluster and
emits robustness / statistical tables for the final report. This runner never
retrains a model, never changes target / split / backtest / training logic, and
uses provided data only (the market benchmark ``sprtrn`` is read from the
canonical Monthly CRSP file -- no external CRSP / WRDS / re-export).

Inputs (read-only, under git-ignored ``data/outputs/``)::

    predictions/walkforward_predictions.parquet
    backtests/walkforward_backtests.parquet
    metrics/walkforward_model_summary.csv
    metrics/walkforward_selected_hparams.csv

Outputs (all git-ignored under ``data/outputs/metrics/``)::

    walkforward_cost_grid_metrics.csv        # standalone fine cost-grid table
    walkforward_breakeven_costs.csv          # breakeven bps per strategy x cutoff
    walkforward_rebalance_robustness.csv     # monthly vs quarterly rebalance
    walkforward_regime_robustness.csv        # vol / trend / dispersion / period regimes
    walkforward_alpha_beta_stats.csv         # HAC alpha/beta vs sprtrn + NW mean t-stat
    walkforward_model_difference_tests.csv   # paired NW mean-difference tests
    walkforward_output_preview.md      # human-readable digest

Reuses (does NOT duplicate): ``mlfinance.eval.statistical_tests``
(``newey_west_mean_tstat``, ``market_alpha_beta``), ``mlfinance.eval.portfolio_metrics``
(``summarize_monthly_returns`` -> Sharpe / hit-rate), the canonical portfolio
primitives (``decile_sort`` / ``long_short_weights`` / ``compute_realized_returns``
/ ``turnover``) for the quarterly book, and the accepted grids ``FINE_COST_GRID`` /
``CUTOFFS`` from the walk-forward runner.

Anti-contamination: :func:`validate_walkforward_inputs` refuses to run if any prediction or
selected-hparam row has ``train_start`` before ``cfg.splits.locked.train_start``
(1985-01-31) or shows OOS leakage -- so archived pre-1985 outputs cannot be
post-processed by accident.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlfinance.backtest.metrics import turnover
from mlfinance.backtest.portfolio import (
    compute_realized_returns,
    decile_sort,
    long_short_weights,
)
from mlfinance.data.loaders import load_crsp_monthly
from mlfinance.eval.portfolio_metrics import summarize_monthly_returns
from mlfinance.eval.statistical_tests import market_alpha_beta, newey_west_mean_tstat
from mlfinance.run.walkforward import FINE_COST_GRID, OUTPUT_PREFIX
from mlfinance.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

HYDRA_CONFIG_PATH = "../../configs"
HYDRA_CONFIG_NAME = "main"

# Focus cutoff/cost for the single-number robustness views (rebalance / regime /
# alpha-beta / diff tests). The full cost grid x all cutoffs lives in the cost-grid
# and breakeven tables.
MAIN_CUTOFF = 0.10
# Calendar-quarter rebalance months. These align with the walk-forward fold
# boundaries (each test year starts in January), so a quarterly hold never spans
# two folds / two retrained models.
QUARTERLY_REBALANCE_MONTHS = (1, 4, 7, 10)
PERIOD_SPLIT_YEAR = 2019  # <=2019 vs >=2020


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _split_strategy(model_name: str) -> tuple[str, str]:
    """``"flow_compustat__xgboost"`` -> ``("flow_compustat", "xgboost")``."""
    fs, _, m = str(model_name).partition("__")
    return fs, m


def _safe_lag(n: int, hac_lag: int) -> int:
    """HAC lag clamped to ``[0, n-1]`` so statsmodels never gets maxlags >= nobs."""
    return max(0, min(int(hac_lag), int(n) - 1))


def breakeven_bps(mean_gross_return: float, mean_turnover: float) -> float:
    """``10000 * mean(gross_return) / mean(turnover)``; NaN on zero/missing turnover."""
    mt = float(mean_turnover)
    if not np.isfinite(mt) or mt <= 0.0:
        return float("nan")
    return 10_000.0 * float(mean_gross_return) / mt


def _nw_mean_tstat(returns: pd.Series, hac_lag: int) -> dict[str, float]:
    """Newey-West HAC t-stat of the mean, with a defensive lag clamp + small-n guard."""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 3:
        return {
            "mean": float(r.mean()) if len(r) else float("nan"),
            "tstat": float("nan"),
            "n": int(len(r)),
        }
    return newey_west_mean_tstat(r, lags=_safe_lag(len(r), hac_lag))


def _alpha_beta(rp: pd.Series, rm: pd.Series, hac_lag: int) -> dict[str, float]:
    """HAC alpha/beta vs the market; NaN when fewer than 6 aligned months."""
    df = pd.concat([pd.Series(rp).rename("rp"), pd.Series(rm).rename("rm")], axis=1).dropna()
    if len(df) < 6:
        return {
            "alpha": np.nan,
            "alpha_tstat": np.nan,
            "beta": np.nan,
            "beta_tstat": np.nan,
            "n": int(len(df)),
        }
    return market_alpha_beta(df["rp"], df["rm"], lags=_safe_lag(len(df), hac_lag))


# ---------------------------------------------------------------------------
# Input loading + anti-contamination validation
# ---------------------------------------------------------------------------
def _input_paths(cfg: DictConfig) -> dict[str, Path]:
    pred_dir = Path(cfg.paths.predictions)
    bt_dir = Path(cfg.paths.backtests)
    metrics_dir = Path(cfg.paths.metrics)
    return {
        "predictions": pred_dir / f"{OUTPUT_PREFIX}_predictions.parquet",
        "backtests": bt_dir / f"{OUTPUT_PREFIX}_backtests.parquet",
        "model_summary": metrics_dir / f"{OUTPUT_PREFIX}_model_summary.csv",
        "selected_hparams": metrics_dir / f"{OUTPUT_PREFIX}_selected_hparams.csv",
        "metrics_dir": metrics_dir,
        "backtests_dir": bt_dir,
    }


def validate_walkforward_inputs(
    pred: pd.DataFrame, sel: pd.DataFrame, cfg: DictConfig
) -> dict[str, Any]:
    """Confirm inputs are the final audited outputs; raise on pre-1985 train or leakage.

    Guards against post-processing archived contaminated (pre-1985 train-buffer)
    outputs. Returns a diagnostics dict that also feeds the output preview.
    """
    date_col = cfg.data.date_col
    floor = pd.to_datetime(cfg.splits.locked.train_start)

    ts_pred = pd.to_datetime(pred["train_start"])
    ts_sel = pd.to_datetime(sel["train_start"])
    d = pd.to_datetime(pred[date_col])
    info: dict[str, Any] = {
        "pred_rows": int(len(pred)),
        "strategy_folds": int(pred.groupby(["feature_set", "model_name", "fold_id"]).ngroups),
        "pred_train_start_min": str(ts_pred.min().date()),
        "pred_train_start_max": str(ts_pred.max().date()),
        "sel_train_start_min": str(ts_sel.min().date()),
        "sel_train_start_max": str(ts_sel.max().date()),
        "train_start_floor": str(floor.date()),
        "pred_pre_floor": int((ts_pred < floor).sum()),
        "sel_pre_floor": int((ts_sel < floor).sum()),
        "year_ne_fold": int((d.dt.year != pred["fold_id"]).sum()),
        "date_le_val_end": int((d <= pd.to_datetime(pred["val_end"])).sum()),
        "null_selected_hparams_json": int(pred["selected_hparams_json"].isna().sum()),
    }
    bad = (
        info["pred_pre_floor"]
        or info["sel_pre_floor"]
        or info["year_ne_fold"]
        or info["date_le_val_end"]
        or info["null_selected_hparams_json"]
    )
    if bad:
        raise ValueError(
            "walk-forward post-process: inputs look contaminated (pre-1985 train or OOS "
            f"leakage) -- refusing to run. Diagnostics: {info}"
        )
    logger.info("walk-forward post-process: inputs validated as final audited (%s).", info)
    return info


# ---------------------------------------------------------------------------
# 1) standalone cost-grid table  (extract/clean from model_summary)
# ---------------------------------------------------------------------------
def cost_grid_metrics(model_summary: pd.DataFrame) -> pd.DataFrame:
    """Project the accepted fine cost grid out of the aggregate model summary."""
    df = model_summary.copy()
    if "turnover" not in df.columns and "avg_turnover" in df.columns:
        df = df.rename(columns={"avg_turnover": "turnover"})
    if "feature_set" not in df.columns or "model" not in df.columns:
        parts = df["model_name"].map(_split_strategy)
        df["feature_set"] = [p[0] for p in parts]
        df["model"] = [p[1] for p in parts]
    cols = [
        "feature_set",
        "model",
        "cutoff",
        "cost_bps",
        "gross_sharpe",
        "net_sharpe",
        "net_ann_return",
        "turnover",
        "breakeven_bps",
        "rank_ic_mean",
        "rank_ic_hac_tstat",
        "oos_r2",
    ]
    have = [c for c in cols if c in df.columns]
    grid = {round(float(c), 3) for c in FINE_COST_GRID}
    out = df[df["cost_bps"].round(3).isin(grid)][have].copy()
    return out.sort_values(["feature_set", "model", "cutoff", "cost_bps"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2) standalone breakeven table  (recompute from the gross monthly book)
# ---------------------------------------------------------------------------
def breakeven_costs(backtests: pd.DataFrame) -> pd.DataFrame:
    """Per (strategy, cutoff) breakeven bps from the cost-independent (bps=0) book."""
    base = backtests[backtests["cost_bps"].round(3) == 0.0]
    rows: list[dict[str, Any]] = []
    for (model_name, cutoff), g in base.groupby(["model_name", "cutoff"], sort=True):
        fs, m = _split_strategy(model_name)
        mg = float(g["gross_return"].mean())
        mt = float(g["turnover"].mean())
        rows.append(
            {
                "feature_set": fs,
                "model": m,
                "cutoff": float(cutoff),
                "n_months": int(pd.to_datetime(g["date"]).nunique()),
                "mean_gross_return": mg,
                "mean_turnover": mt,
                "breakeven_bps": breakeven_bps(mg, mt),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_set", "model", "cutoff"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) rebalancing robustness  (monthly vs quarterly, no look-ahead)
# ---------------------------------------------------------------------------
def _strategy_books(
    pred_g: pd.DataFrame,
    *,
    date_col: str,
    permno_col: str,
    target_col: str,
    n_buckets: int,
    cost_bps: float,
) -> dict[str, pd.DataFrame]:
    """Monthly and quarterly long-short books for ONE strategy's predictions.

    Quarterly forms weights only on calendar-quarter months and HOLDS them across
    the two intervening months (forward-filled weights -> no future information).
    """
    pp = pred_g.rename(columns={"prediction": "y_hat"})[[permno_col, date_col, "y_hat"]].copy()
    deciles = decile_sort(
        pp.rename(columns={permno_col: "permno", date_col: "date"}), n_deciles=n_buckets
    )
    weights = long_short_weights(deciles, scheme="EW", top_decile=n_buckets, bottom_decile=1)
    returns_panel = pred_g.pivot_table(
        index=date_col, columns=permno_col, values=target_col, aggfunc="first"
    ).sort_index()
    returns_panel.index = pd.to_datetime(returns_panel.index)
    weights.index = pd.to_datetime(weights.index)
    weights = weights.reindex(returns_panel.index)

    def _book(w: pd.DataFrame) -> pd.DataFrame:
        gross = compute_realized_returns(w, returns_panel)
        turn = turnover(w).reindex(gross.index).fillna(0.0)
        cost = turn * (float(cost_bps) / 10_000.0)
        return pd.DataFrame(
            {
                "date": gross.index,
                "gross_return": gross.to_numpy(),
                "turnover": turn.to_numpy(),
                "net_return": (gross - cost).to_numpy(),
            }
        )

    # Monthly is the canonical book and always computed. Quarterly is the novel
    # piece -> best-effort: if anything about the hold logic raises on real data,
    # we still return the monthly book rather than losing both.
    books: dict[str, pd.DataFrame] = {"monthly": _book(weights.fillna(0.0))}
    try:
        wq = weights.copy()
        is_reb = wq.index.month.isin(QUARTERLY_REBALANCE_MONTHS)
        wq.loc[~is_reb] = np.nan  # drop non-rebalance rows ...
        wq = wq.ffill().fillna(0.0)  # ... then HOLD the last rebalance weights forward
        books["quarterly"] = _book(wq)
    except Exception as exc:  # noqa: BLE001 - quarterly is optional, never fatal
        logger.warning("walk-forward post-process: quarterly book skipped (%s).", exc)
    return books


def rebalance_robustness(
    predictions: pd.DataFrame,
    cfg: DictConfig,
    *,
    cutoff: float = MAIN_CUTOFF,
    cost_bps: float | None = None,
) -> pd.DataFrame:
    """Monthly vs quarterly rebalance for every strategy at the focus cutoff/cost."""
    date_col = cfg.data.date_col
    permno_col = cfg.data.permno_col
    target_col = cfg.data.target_col
    cost_bps = int(cfg.costs.main_bps) if cost_bps is None else cost_bps
    n_buckets = int(round(1.0 / float(cutoff)))

    rows: list[dict[str, Any]] = []
    for (fs, model), g in predictions.groupby(["feature_set", "model_name"], sort=True):
        gg = (
            g[[date_col, permno_col, target_col, "prediction"]]
            .dropna(subset=["prediction", target_col])
            .copy()
        )
        if int(gg.groupby(date_col)[permno_col].count().min()) < n_buckets:
            logger.warning(
                "rebalance: %s/%s thinnest month < %d names; skipping cutoff %.2f.",
                fs,
                model,
                n_buckets,
                cutoff,
            )
            continue
        books = _strategy_books(
            gg,
            date_col=date_col,
            permno_col=permno_col,
            target_col=target_col,
            n_buckets=n_buckets,
            cost_bps=cost_bps,
        )
        for freq, book in books.items():
            net = summarize_monthly_returns(book["net_return"])
            gross = summarize_monthly_returns(book["gross_return"])
            mg = float(book["gross_return"].mean())
            mt = float(book["turnover"].mean())
            rows.append(
                {
                    "feature_set": fs,
                    "model": model,
                    "rebalance": freq,
                    "cutoff": float(cutoff),
                    "cost_bps": float(cost_bps),
                    "n_months": int(len(book)),
                    "gross_sharpe": gross["sharpe"],
                    "net_sharpe": net["sharpe"],
                    "net_ann_return": net["ann_return"],
                    "avg_turnover": mt,
                    "hit_rate": net["hit_rate"],
                    "breakeven_bps": breakeven_bps(mg, mt),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["feature_set", "model", "rebalance"]).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# market series + regimes
# ---------------------------------------------------------------------------
def _market_monthly(cfg: DictConfig, oos_dates: pd.DatetimeIndex) -> pd.Series:
    """Per-month market return (``sprtrn``) from the canonical Monthly CRSP file.

    Loads ~13 months before the first OOS month so trailing-window regime labels are
    well-defined on every OOS month. Provided-data only; read-only.
    """
    start = (oos_dates.min() - pd.DateOffset(months=13)).date().isoformat()
    crsp = load_crsp_monthly(cfg.data.paths.crsp_monthly, start_date=start)
    mkt = crsp.groupby("date")["sprtrn"].first().astype("float64")
    mkt.index = pd.to_datetime(mkt.index)
    return mkt.sort_index()


def _regime_labels(market: pd.Series, dispersion: pd.Series) -> dict[str, pd.Series]:
    """Month -> regime label for four schemes. Trailing windows use the full market
    series (incl. pre-OOS months) so OOS labels are never NaN-defaulted.

    * vol:        trailing-12m std of sprtrn, split at its median (high_vol/low_vol)
    * trend:      trailing-12m sum of sprtrn >= 0 -> bull, else bear
    * dispersion: cross-sectional std of realized returns, split at median
    * period:     2017-2019 vs 2020-2024 (calendar)
    """
    idx = market.index
    vol = market.rolling(12, min_periods=6).std()
    trail = market.rolling(12, min_periods=6).sum()
    out: dict[str, pd.Series] = {}
    out["vol"] = pd.Series(np.where(vol >= vol.median(), "high_vol", "low_vol"), index=idx).where(
        vol.notna()
    )
    out["trend"] = pd.Series(np.where(trail >= 0.0, "bull", "bear"), index=idx).where(trail.notna())
    disp = dispersion.reindex(idx)
    out["dispersion"] = pd.Series(
        np.where(disp >= disp.median(), "high_dispersion", "low_dispersion"), index=idx
    ).where(disp.notna())
    out["period"] = pd.Series(
        np.where(idx.year <= PERIOD_SPLIT_YEAR, "2017-2019", "2020-2024"), index=idx
    )
    return out


def regime_robustness(
    backtests: pd.DataFrame,
    predictions: pd.DataFrame,
    market: pd.Series,
    cfg: DictConfig,
    *,
    cutoff: float = MAIN_CUTOFF,
    cost_bps: float | None = None,
    hac_lag: int = 11,
) -> pd.DataFrame:
    """Per (strategy, regime) net-return stats + HAC mean t-stat + alpha/beta."""
    date_col = cfg.data.date_col
    permno_col = cfg.data.permno_col
    target_col = cfg.data.target_col
    cost_bps = int(cfg.costs.main_bps) if cost_bps is None else cost_bps

    sub = backtests[
        (backtests["cutoff"].round(3) == round(float(cutoff), 3))
        & (backtests["cost_bps"].round(3) == round(float(cost_bps), 3))
    ].copy()
    sub["date"] = pd.to_datetime(sub["date"])

    uniq = predictions.drop_duplicates([date_col, permno_col]).copy()
    uniq["date"] = pd.to_datetime(uniq[date_col])
    dispersion = uniq.groupby("date")[target_col].std()
    labels = _regime_labels(market, dispersion)

    rows: list[dict[str, Any]] = []
    for model_name, g in sub.groupby("model_name", sort=True):
        fs, m = _split_strategy(model_name)
        g = g.set_index("date").sort_index()
        mkt_aligned = market.reindex(g.index)
        for scheme, lab in labels.items():
            lab_oos = lab.reindex(g.index)
            for regime in sorted(set(lab_oos.dropna().unique())):
                mask = (lab_oos == regime).to_numpy()
                gr = g.loc[mask]
                if len(gr) < 2:
                    continue
                net = gr["net_return"]
                nw = _nw_mean_tstat(net, hac_lag)
                ab = _alpha_beta(net, mkt_aligned.loc[gr.index], hac_lag)
                summ = summarize_monthly_returns(net)
                rows.append(
                    {
                        "feature_set": fs,
                        "model": m,
                        "regime_scheme": scheme,
                        "regime": regime,
                        "n_months": int(len(gr)),
                        "mean_gross_return": float(gr["gross_return"].mean()),
                        "mean_net_return": float(net.mean()),
                        "net_sharpe": summ["sharpe"],
                        "turnover": float(gr["turnover"].mean()),
                        "hit_rate": summ["hit_rate"],
                        "newey_west_tstat_mean_net": nw["tstat"],
                        "alpha": ab["alpha"],
                        "beta": ab["beta"],
                        "alpha_tstat": ab["alpha_tstat"],
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["feature_set", "model", "regime_scheme", "regime"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# alpha/beta + NW mean (full sample)
# ---------------------------------------------------------------------------
def alpha_beta_stats(
    backtests: pd.DataFrame,
    market: pd.Series,
    cfg: DictConfig,
    *,
    cutoff: float = MAIN_CUTOFF,
    cost_bps: float | None = None,
    hac_lag: int = 11,
) -> pd.DataFrame:
    """Full-sample HAC alpha/beta vs sprtrn + NW mean t-stat, gross and net."""
    cost_bps = int(cfg.costs.main_bps) if cost_bps is None else cost_bps
    sub = backtests[
        (backtests["cutoff"].round(3) == round(float(cutoff), 3))
        & (backtests["cost_bps"].round(3) == round(float(cost_bps), 3))
    ].copy()
    sub["date"] = pd.to_datetime(sub["date"])

    rows: list[dict[str, Any]] = []
    for model_name, g in sub.groupby("model_name", sort=True):
        fs, m = _split_strategy(model_name)
        g = g.set_index("date").sort_index()
        mkt_aligned = market.reindex(g.index)
        for return_type in ("gross", "net"):
            r = g[f"{return_type}_return"]
            nw = _nw_mean_tstat(r, hac_lag)
            ab = _alpha_beta(r, mkt_aligned, hac_lag)
            rows.append(
                {
                    "feature_set": fs,
                    "model": m,
                    "return_type": return_type,
                    "cutoff": float(cutoff),
                    "cost_bps": float(cost_bps),
                    "n_months": int(len(r.dropna())),
                    "mean_monthly_return": float(r.mean()),
                    "nw_tstat_mean": nw["tstat"],
                    "alpha": ab["alpha"],
                    "alpha_tstat": ab["alpha_tstat"],
                    "beta": ab["beta"],
                    "beta_tstat": ab["beta_tstat"],
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["feature_set", "model", "return_type"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# model-difference tests
# ---------------------------------------------------------------------------
def _load_reversal_monthly(cfg: DictConfig, *, cost_bps: float) -> pd.Series | None:
    """Reversal baseline monthly net return from baseline outputs, or None.

    Reads ``baseline_backtests.parquet`` (locked-split, 10% decile cut)
    and extracts the ``reversal`` book at the matching cost. Returns None if the file
    or the reversal rows are absent -- the difference test then reports the pair as
    UNAVAILABLE rather than fabricating it.
    """
    path = Path(cfg.paths.backtests) / "baseline_backtests.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "model_name" not in df.columns:
        return None
    rev = df[df["model_name"].astype(str).str.fullmatch("reversal", case=False)]
    if rev.empty:
        return None
    if "cost_bps" in rev.columns:
        rev = rev[rev["cost_bps"].round(3) == round(float(cost_bps), 3)]
    col = "net_return" if "net_return" in rev.columns else None
    if col is None or rev.empty:
        return None
    rev = rev.copy()
    rev["date"] = pd.to_datetime(rev["date"])
    return rev.groupby("date")[col].mean().sort_index()


def model_difference_tests(
    backtests: pd.DataFrame,
    cfg: DictConfig,
    *,
    cutoff: float = MAIN_CUTOFF,
    cost_bps: float | None = None,
    hac_lag: int = 11,
    reversal: pd.Series | None = None,
) -> pd.DataFrame:
    """Paired Newey-West mean-difference tests on OOS monthly net returns."""
    cost_bps = int(cfg.costs.main_bps) if cost_bps is None else cost_bps
    sub = backtests[
        (backtests["cutoff"].round(3) == round(float(cutoff), 3))
        & (backtests["cost_bps"].round(3) == round(float(cost_bps), 3))
    ].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    nets = {
        mn: g.set_index("date")["net_return"].sort_index() for mn, g in sub.groupby("model_name")
    }

    rows: list[dict[str, Any]] = []

    def _add(a_name: str, b_name: str, a: pd.Series, b: pd.Series, source: str) -> None:
        d = (a - b).dropna()
        if len(d) < 3:
            return
        nw = _nw_mean_tstat(d, hac_lag)
        rows.append(
            {
                "pair": f"{a_name} - {b_name}",
                "strategy_a": a_name,
                "strategy_b": b_name,
                "n_months": int(len(d)),
                "mean_diff_monthly": float(d.mean()),
                "mean_diff_annualized": float(d.mean() * 12.0),
                "nw_tstat_diff": nw["tstat"],
                "baseline_source": source,
            }
        )

    within = [
        ("flow_compustat__mlp", "return_only__mlp"),
        ("flow_compustat__xgboost", "return_only__xgboost"),
        ("flow_compustat__mlp", "flow_compustat__xgboost"),
    ]
    for a_name, b_name in within:
        if a_name in nets and b_name in nets:
            _add(a_name, b_name, nets[a_name], nets[b_name], "walk_forward")

    if reversal is not None and not reversal.empty:
        for a_name in ("flow_compustat__mlp", "flow_compustat__xgboost"):
            if a_name in nets:
                _add(
                    a_name,
                    "reversal",
                    nets[a_name],
                    reversal,
                    "baseline_locked_split_10pct (different protocol: fixed-rule, single split)",
                )
    else:
        rows.append(
            {
                "pair": "<walk-forward> - reversal",
                "strategy_a": "",
                "strategy_b": "reversal",
                "n_months": 0,
                "mean_diff_monthly": float("nan"),
                "mean_diff_annualized": float("nan"),
                "nw_tstat_diff": float("nan"),
                "baseline_source": "UNAVAILABLE - reversal monthly returns not found in outputs; not fabricated",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------
def _write_preview(
    path: Path,
    *,
    val_info: dict[str, Any],
    results: dict[str, pd.DataFrame],
    cfg: DictConfig,
    reversal_available: bool,
    skipped: dict[str, str],
) -> Path:
    main_bps = int(cfg.costs.main_bps)

    def _md_table(df: pd.DataFrame | None, n: int = 12) -> str:
        if df is None or df.empty:
            return "_(empty)_\n"
        with pd.option_context("display.width", 200, "display.max_columns", 60):
            body = df.head(n).round(4).to_string(index=False)
        return "```\n" + body + "\n```\n"

    def _section(name: str, title: str, transform=None, n: int = 12) -> list[str]:
        if name in skipped:
            return [f"## {title}", f"_SKIPPED: {skipped[name]}_", ""]
        df = results.get(name)
        if transform is not None and df is not None and not df.empty:
            # preview rendering must never raise (e.g. a stub frame lacks expected cols)
            with contextlib.suppress(Exception):
                df = transform(df)
        return [f"## {title}", _md_table(df, n=n), ""]

    def _cg_main(df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df["cutoff"].round(3) == round(MAIN_CUTOFF, 3))
            & (df["cost_bps"].round(3) == float(main_bps))
        ].sort_values("net_sharpe", ascending=False)

    def _ab_net(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["return_type"] == "net"].sort_values("alpha", ascending=False)

    # Did quarterly actually populate? (best-effort piece)
    rb = results.get("rebalance")
    quarterly_ok = (
        rb is not None and "rebalance" in rb.columns and (rb["rebalance"] == "quarterly").any()
    )

    lines = [
        "# walk-forward post-processing robustness - output preview",
        "",
        "Generated by `mlfinance.run.walkforward_robustness` (no retraining).",
        "",
        "## Input validation (final audited, anti-contamination)",
        "```",
        f"pred train_start: {val_info['pred_train_start_min']} .. {val_info['pred_train_start_max']}",
        f"sel  train_start: {val_info['sel_train_start_min']} .. {val_info['sel_train_start_max']}",
        f"train_start floor: {val_info['train_start_floor']}",
        f"pred rows < floor: {val_info['pred_pre_floor']}   sel rows < floor: {val_info['sel_pre_floor']}",
        f"year != fold_id: {val_info['year_ne_fold']}   date <= val_end: {val_info['date_le_val_end']}",
        f"null selected_hparams_json: {val_info['null_selected_hparams_json']}",
        f"rows: {val_info['pred_rows']}   strategy-folds: {val_info['strategy_folds']}",
        "```",
        "",
    ]
    lines += _section(
        "cost_grid",
        f"Cost-grid @ {int(MAIN_CUTOFF * 100)}% cut, {main_bps} bps (ranked by net Sharpe)",
        _cg_main,
    )
    lines += _section("breakeven", "Breakeven bps by strategy x cutoff")
    lines += _section(
        "rebalance",
        f"Rebalance robustness (monthly vs quarterly, {int(MAIN_CUTOFF * 100)}% cut, {main_bps} bps)",
        n=20,
    )
    lines += _section("alpha_beta", "Alpha/beta vs sprtrn (net, ranked by alpha)", _ab_net)
    lines += _section(
        "model_difference", "Model-difference tests (NW mean-difference on monthly net)", n=20
    )
    lines += _section("regime", "Regime robustness (head)", n=16)
    lines += [
        "## Notes",
        f"- Sections skipped (isolated failures, did not block others): {sorted(skipped) or 'none'}.",
        f"- Quarterly rebalance populated: {quarterly_ok} "
        "(monthly is always computed; quarterly is best-effort and never blocks the rest).",
        f"- Reversal baseline available for difference tests: {reversal_available}.",
        "- 'Flat-cost robust under the provided-data universe' -- NOT a complete tradability claim "
        "(no microcap / liquidity / common-share / exchange screens; provided Monthly CRSP lacks "
        "prc / shrout / shrcd / exchcd).",
        "- Regime labels are diagnostic only and were never used for model selection.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_postprocess(cfg: DictConfig) -> dict[str, Path]:
    """Run the full post-processing robustness package; write all tables."""
    paths = _input_paths(cfg)
    for key in ("predictions", "backtests", "model_summary", "selected_hparams"):
        if not paths[key].exists():
            raise FileNotFoundError(
                f"walk-forward post-process input missing: {paths[key]}. "
                "Run the final audited walk-forward + aggregate first."
            )

    predictions = pd.read_parquet(paths["predictions"])
    backtests = pd.read_parquet(paths["backtests"])
    model_summary = pd.read_csv(paths["model_summary"])
    selected = pd.read_csv(paths["selected_hparams"])

    # Validation is the anti-contamination guard and is INTENTIONALLY fatal: we do
    # not post-process pre-1985 / leaky inputs under any circumstance.
    val_info = validate_walkforward_inputs(predictions, selected, cfg)
    main_bps = int(cfg.costs.main_bps)
    hac_lag = int(cfg.eval.newey_west_lag_primary)
    oos_dates = pd.DatetimeIndex(pd.to_datetime(predictions[cfg.data.date_col]).unique())
    mdir = paths["metrics_dir"]

    outputs: dict[str, Path] = {}
    results: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}

    def _try(name: str, filename: str, fn) -> None:
        """Run one analysis in isolation: a failure is recorded + stubbed, never
        propagated -- so no single table (e.g. quarterly rebalance) can block the rest.
        """
        try:
            df = fn()
        except Exception as exc:  # noqa: BLE001 - per-table isolation is intentional
            logger.exception(
                "walk-forward post-process: '%s' failed; writing stub + continuing.", name
            )
            skipped[name] = f"{type(exc).__name__}: {exc}"
            df = pd.DataFrame([{"status": "SKIPPED", "reason": skipped[name]}])
        results[name] = df
        outputs[name] = _write_csv(df, mdir / filename)

    # The market benchmark (sprtrn) feeds regime + alpha/beta ONLY. If it cannot be
    # loaded, those two degrade gracefully; cost-grid / breakeven / rebalance / diff
    # are unaffected.
    market: pd.Series | None
    try:
        market = _market_monthly(cfg, oos_dates)
    except Exception as exc:  # noqa: BLE001
        logger.exception("walk-forward post-process: market series (sprtrn) unavailable.")
        market = None
        skipped["_market"] = f"{type(exc).__name__}: {exc}"

    def _need_market() -> pd.Series:
        if market is None:
            raise RuntimeError(f"market series (sprtrn) unavailable: {skipped.get('_market')}")
        return market

    # Always-safe extraction / aggregation tables.
    _try("cost_grid", "walkforward_cost_grid_metrics.csv", lambda: cost_grid_metrics(model_summary))
    _try("breakeven", "walkforward_breakeven_costs.csv", lambda: breakeven_costs(backtests))
    # Rebalance: monthly is robust; quarterly is best-effort inside _strategy_books.
    # The whole step is additionally isolated here so it can never block what follows.
    _try(
        "rebalance",
        "walkforward_rebalance_robustness.csv",
        lambda: rebalance_robustness(predictions, cfg, cutoff=MAIN_CUTOFF, cost_bps=main_bps),
    )
    _try(
        "regime",
        "walkforward_regime_robustness.csv",
        lambda: regime_robustness(
            backtests,
            predictions,
            _need_market(),
            cfg,
            cutoff=MAIN_CUTOFF,
            cost_bps=main_bps,
            hac_lag=hac_lag,
        ),
    )
    _try(
        "alpha_beta",
        "walkforward_alpha_beta_stats.csv",
        lambda: alpha_beta_stats(
            backtests, _need_market(), cfg, cutoff=MAIN_CUTOFF, cost_bps=main_bps, hac_lag=hac_lag
        ),
    )
    reversal: pd.Series | None = None
    try:
        reversal = _load_reversal_monthly(cfg, cost_bps=main_bps)
    except Exception as exc:  # noqa: BLE001
        logger.warning("walk-forward post-process: reversal baseline load failed (%s).", exc)
    _try(
        "model_difference",
        "walkforward_model_difference_tests.csv",
        lambda: model_difference_tests(
            backtests,
            cfg,
            cutoff=MAIN_CUTOFF,
            cost_bps=main_bps,
            hac_lag=hac_lag,
            reversal=reversal,
        ),
    )

    outputs["preview"] = _write_preview(
        mdir / "walkforward_output_preview.md",
        val_info=val_info,
        results=results,
        cfg=cfg,
        reversal_available=reversal is not None and not reversal.empty,
        skipped=skipped,
    )
    if skipped:
        logger.warning("walk-forward post-process completed WITH skipped sections: %s", skipped)
    else:
        logger.info("walk-forward post-process done: all tables written, nothing skipped.")
    return outputs


@hydra.main(version_base=None, config_path=HYDRA_CONFIG_PATH, config_name=HYDRA_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    configure_logging()
    outputs = run_postprocess(cfg)
    for name, path in outputs.items():
        logger.info("Wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
