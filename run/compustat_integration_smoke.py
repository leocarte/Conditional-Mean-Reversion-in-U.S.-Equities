"""feature-audit flow-only Compustat smoke / audit runner (diagnostics only).

Builds a lightweight target-only return panel via ``load_crsp_monthly`` plus
the production duplicate-collapse / calendar-expansion / target-shift helpers,
without touching the core rolling-feature implementation. It then attaches the
flow-only Compustat block from a single prepared feature frame and writes
diagnostics comparing the return-only and Compustat-enabled panels. It does NOT
train models, change configs, or claim performance.

Skipping the rolling feature stack avoids the multi-minute Python-callback
``rolling.apply`` work (momentum, drawdown, beta, idiosyncratic vol) that the
runner's diagnostics never read, while preserving the production row and target
construction path used for the smoke's invariance checks.

Guardrails (raise on violation): no point-in-time leakage
(``compustat_availability_date <= date``); no forbidden balance-sheet / scaled
feature names; identical targets and split masks between the two panels. Outputs
go to git-ignored ``data/outputs/metrics/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlfinance.data.compustat_schema import (
    audit_compustat_point_in_time_merge,
    merge_compustat_point_in_time,
)
from mlfinance.data.loaders import load_compustat_quarterly, load_crsp_monthly
from mlfinance.features.compustat_flow_features import (
    COMPUSTAT_FLOW_FEATURE_COLUMNS,
    build_compustat_flow_features,
    compustat_flow_feature_columns,
    normalize_cusip8,
)
from mlfinance.features.return_features import (
    _collapse_baseline_duplicates,
    _expand_to_complete_monthly_calendar,
)
from mlfinance.run.linear_benchmarks import locked_split_masks

logger = logging.getLogger(__name__)

KEY_FLOW_FEATURES: tuple[str, ...] = (
    "compustat_sales_log1p",
    "compustat_gross_margin",
    "compustat_net_margin",
    "compustat_operating_cash_flow_margin",
    "compustat_capex_intensity",
    "compustat_accrual_proxy",
)

FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "roa",
    "leverage",
    "book_to_market",
    "asset_growth",
    "debt_to_assets",
    "assets_scaled",
    "equity_scaled",
)

_COMPUSTAT_DATADATE = "compustat_datadate"
_COMPUSTAT_AVAILABILITY = "compustat_availability_date"


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _panel_start_date(cfg: DictConfig) -> str:
    """Mirror ``baseline._panel_start_date``: ``train_start`` minus the min-history lookback."""
    train_start = pd.to_datetime(cfg.splits.locked.train_start)
    min_history = int(
        OmegaConf.select(cfg, "features.return_features.min_history_months", default=12)
    )
    return str(
        (train_start - pd.DateOffset(months=min_history)).to_period("M").to_timestamp("M").date()
    )


def _date_min_max(frame: pd.DataFrame, col: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if col not in frame.columns:
        return None, None
    dates = pd.to_datetime(frame[col], errors="coerce").dropna()
    return (None, None) if dates.empty else (dates.min(), dates.max())


def _build_return_targets_only_panel(
    df: pd.DataFrame,
    *,
    permno_col: str,
    date_col: str,
    ret_col: str,
    market_col: str,
    target_col: str,
    robustness_target_col: str | None,
) -> pd.DataFrame:
    """Reuse the production duplicate/calendar/target path without rolling features."""
    _require_columns(df, [permno_col, date_col, ret_col, market_col])
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]) + pd.offsets.MonthEnd(0)
    out = out.sort_values([permno_col, date_col]).reset_index(drop=True)
    out = _collapse_baseline_duplicates(
        out,
        permno_col=permno_col,
        date_col=date_col,
        ret_col=ret_col,
        market_col=market_col,
    )
    out = _expand_to_complete_monthly_calendar(
        out,
        permno_col=permno_col,
        date_col=date_col,
        market_col=market_col,
    )
    out = out.sort_values([permno_col, date_col]).reset_index(drop=True)

    market_by_date = (
        out[[date_col, market_col]].drop_duplicates(date_col).set_index(date_col)[market_col]
    )
    out[target_col] = out.groupby(permno_col, group_keys=False)[ret_col].shift(-1)
    if robustness_target_col:
        next_mkt = (out[date_col] + pd.offsets.MonthEnd(1)).map(market_by_date)
        out[robustness_target_col] = out[target_col] - next_mkt

    out = out.loc[out["_is_observed"]].drop(columns=["_is_observed"]).reset_index(drop=True)
    return out


def _normalized_panel_cusip8(panel: pd.DataFrame) -> pd.Series:
    for source_col in ("cusip8", "cusip", "hdrcusip"):
        if source_col in panel.columns:
            return normalize_cusip8(panel[source_col])
    raise KeyError("Smoke runner could not derive a panel 'cusip8' key from CRSP inputs.")


def run_compustat_integration_smoke(cfg: DictConfig) -> dict[str, Path]:
    """Diagnose the flow-only Compustat-enabled panel build vs the return-only build."""
    metrics_dir = Path(cfg.paths.metrics)
    permno_col = cfg.data.permno_col
    date_col = cfg.data.date_col
    target_col = cfg.data.target_col
    robustness_target_col = OmegaConf.select(cfg, "data.robustness_target_col", default=None)
    lag_months = int(
        OmegaConf.select(cfg, "features.compustat_features.conservative_lag_months", default=6)
    )

    # --- Build a lightweight target-only return panel, then attach Compustat ---
    # We do NOT go through ``build_model_panel`` for either panel: the smoke
    # never reads the rolling feature stack (momentum, drawdown, beta, idio
    # vol), so we skip the Python-callback ``rolling.apply`` work. The target
    # columns and the ``(permno, date)`` universe come from the same calendar-
    # expansion + target-shift path used in production, so the guardrails
    # (rows / targets / split masks identical between the two panels) remain
    # meaningful. The enabled panel is the return-only panel with the flow-only
    # Compustat block left-merged on top -- no row or target changes.
    crsp = load_crsp_monthly(cfg.data.paths.crsp_monthly, start_date=_panel_start_date(cfg))
    compustat = load_compustat_quarterly(cfg.data.paths.compustat_quarterly)
    panel_return_only = (
        _build_return_targets_only_panel(
            crsp,
            permno_col=permno_col,
            date_col=date_col,
            ret_col=cfg.data.ret_col,
            market_col=cfg.data.market_col,
            target_col=target_col,
            robustness_target_col=robustness_target_col,
        )
        .dropna(subset=[target_col])
        .sort_values([permno_col, date_col])
        .reset_index(drop=True)
    )
    panel_return_only["cusip8"] = _normalized_panel_cusip8(panel_return_only)

    compustat_flow = build_compustat_flow_features(
        compustat,
        merge_key="cusip8",
        lag_months=lag_months,
        add_missingness_flags=True,
    )
    feature_cols = list(compustat_flow_feature_columns(add_missingness_flags=True))
    merge_feature_cols = [column.removeprefix("compustat_") for column in feature_cols]
    feature_block = compustat_flow[
        ["cusip8", "datadate", "availability_date", *feature_cols]
    ].rename(columns={column: column.removeprefix("compustat_") for column in feature_cols})
    compustat_key_set = set(compustat_flow["cusip8"].dropna().unique().tolist())

    panel_enabled = (
        merge_compustat_point_in_time(
            panel_return_only,
            feature_block,
            merge_key="cusip8",
            formation_date_col=date_col,
            lag_months=lag_months,
            compustat_cols=merge_feature_cols,
            prefix="compustat_",
        )
        .sort_values([permno_col, date_col])
        .reset_index(drop=True)
    )
    for column in compustat_flow_feature_columns(add_missingness_flags=True):
        if column.endswith("_missing") and column in panel_enabled.columns:
            panel_enabled[column] = panel_enabled[column].fillna(1).astype("int8")

    # --- Guardrail 1: identical (permno, date) and target columns ---
    if not panel_return_only[[permno_col, date_col]].equals(panel_enabled[[permno_col, date_col]]):
        raise ValueError("Enabled and return-only panels differ on (permno, date).")
    for col in [c for c in (target_col, robustness_target_col) if c]:
        if col in panel_return_only.columns and not np.allclose(
            panel_return_only[col].to_numpy(dtype="float64"),
            panel_enabled[col].to_numpy(dtype="float64"),
            equal_nan=True,
        ):
            raise ValueError(f"Target column '{col}' differs between panels.")

    # --- Guardrail 2: identical locked split masks ---
    masks_ro = locked_split_masks(panel_return_only, cfg)
    masks_en = locked_split_masks(panel_enabled, cfg)
    for split in masks_ro:
        if (
            not masks_ro[split]
            .reset_index(drop=True)
            .equals(masks_en[split].reset_index(drop=True))
        ):
            raise ValueError(f"Split mask '{split}' differs between panels.")

    # --- Guardrail 3: no forbidden (balance-sheet / scaled) feature names ---
    forbidden = sorted(
        c
        for c in panel_enabled.columns
        if any(token in c.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Forbidden feature columns present in enabled panel: {forbidden}")

    # --- Guardrail 4: point-in-time (no future availability_date) ---
    pit_violations = audit_compustat_point_in_time_merge(
        panel_enabled, formation_date_col=date_col, raise_on_fail=False
    )
    if not pit_violations.empty:
        raise ValueError(
            f"PIT audit found {len(pit_violations)} violation(s): "
            f"{sorted(pit_violations['violation_type'].unique())}."
        )

    # --- Feature coverage ---
    feature_cols = [c for c in COMPUSTAT_FLOW_FEATURE_COLUMNS if c in panel_enabled.columns]
    n = len(panel_enabled)
    coverage = pd.DataFrame(
        [
            {
                "feature": c,
                "non_null_count": int(panel_enabled[c].notna().sum()),
                "non_null_coverage": round(float(panel_enabled[c].notna().mean()), 4) if n else 0.0,
                "is_key_feature": c in KEY_FLOW_FEATURES,
            }
            for c in feature_cols
        ]
    )
    cov_path = _write_csv(coverage, metrics_dir / "compustat_flow_feature_coverage.csv")

    # --- Link / missingness diagnostics ---
    if feature_cols:
        any_feature = panel_enabled[feature_cols].notna().any(axis=1)
    else:
        any_feature = pd.Series(False, index=panel_enabled.index)
    valid_panel_key = panel_enabled["cusip8"].notna()
    known_compustat_key = panel_enabled["cusip8"].isin(compustat_key_set)
    has_pit_record = (
        panel_enabled[_COMPUSTAT_DATADATE].notna()
        if _COMPUSTAT_DATADATE in panel_enabled.columns
        else pd.Series(False, index=panel_enabled.index)
    )
    rows_missing_panel_cusip8 = int((~valid_panel_key).sum())
    rows_without_compustat_key_match = int((valid_panel_key & ~known_compustat_key).sum())
    rows_with_compustat_key_but_no_pit_record = int((known_compustat_key & ~has_pit_record).sum())
    rows_without_pit_available_compustat_record = int((~has_pit_record).sum())

    dd_min, dd_max = _date_min_max(panel_enabled, _COMPUSTAT_DATADATE)
    av_min, av_max = _date_min_max(panel_enabled, _COMPUSTAT_AVAILABILITY)
    ro_min, ro_max = _date_min_max(panel_return_only, date_col)
    en_min, en_max = _date_min_max(panel_enabled, date_col)

    summary = pd.DataFrame(
        [
            {
                "return_only_rows": len(panel_return_only),
                "enabled_rows": len(panel_enabled),
                "return_only_date_min": ro_min,
                "return_only_date_max": ro_max,
                "enabled_date_min": en_min,
                "enabled_date_max": en_max,
                "n_flow_features": len(feature_cols),
                "rows_with_any_compustat_feature": int(any_feature.sum()),
                "share_with_any_compustat_feature": (
                    round(float(any_feature.mean()), 4) if n else 0.0
                ),
                "rows_missing_panel_cusip8": rows_missing_panel_cusip8,
                "share_missing_panel_cusip8": (
                    round(rows_missing_panel_cusip8 / n, 4) if n else 0.0
                ),
                "rows_without_compustat_key_match": rows_without_compustat_key_match,
                "share_without_compustat_key_match": (
                    round(rows_without_compustat_key_match / n, 4) if n else 0.0
                ),
                "rows_with_compustat_key_but_no_pit_record": (
                    rows_with_compustat_key_but_no_pit_record
                ),
                "share_with_compustat_key_but_no_pit_record": (
                    round(rows_with_compustat_key_but_no_pit_record / n, 4) if n else 0.0
                ),
                "rows_without_pit_available_compustat_record": (
                    rows_without_pit_available_compustat_record
                ),
                "share_without_pit_available_compustat_record": (
                    round(rows_without_pit_available_compustat_record / n, 4) if n else 0.0
                ),
                "pit_violations": int(len(pit_violations)),
                "compustat_datadate_min": dd_min,
                "compustat_datadate_max": dd_max,
                "compustat_availability_date_min": av_min,
                "compustat_availability_date_max": av_max,
                "targets_identical": True,
                "split_masks_identical": True,
                "forbidden_feature_columns": "(none)",
            }
        ]
    )
    summary_path = _write_csv(summary, metrics_dir / "compustat_flow_smoke_summary.csv")

    # --- Matched / unmatched example rows ---
    example_cols = [
        c
        for c in (
            permno_col,
            date_col,
            "cusip8",
            _COMPUSTAT_DATADATE,
            _COMPUSTAT_AVAILABILITY,
            *KEY_FLOW_FEATURES,
        )
        if c in panel_enabled.columns
    ]
    diagnostic_status = pd.Series(
        "linked_but_no_pit_record",
        index=panel_enabled.index,
        dtype="string",
    )
    diagnostic_status.loc[valid_panel_key & ~known_compustat_key] = "no_compustat_key_match"
    diagnostic_status.loc[~valid_panel_key] = "missing_panel_cusip8"
    diagnostic_status.loc[has_pit_record] = "matched"

    example_frames = []
    for status in (
        "matched",
        "linked_but_no_pit_record",
        "no_compustat_key_match",
        "missing_panel_cusip8",
    ):
        mask = diagnostic_status.eq(status)
        if not mask.any():
            continue
        example_frames.append(
            panel_enabled.loc[mask, example_cols]
            .head(5)
            .assign(
                match_status="matched" if status == "matched" else "unmatched",
                diagnostic_status=status,
            )
        )
    examples = pd.concat(example_frames, ignore_index=True) if example_frames else pd.DataFrame()
    examples_path = _write_csv(examples, metrics_dir / "compustat_flow_smoke_examples.csv")

    logger.info(
        "Flow-Compustat smoke: %d rows; %.1f%% with >=1 feature; missing panel key=%d; "
        "no key match=%d; linked/no PIT=%d; PIT=%d",
        n,
        100.0 * float(any_feature.mean()) if n else 0.0,
        rows_missing_panel_cusip8,
        rows_without_compustat_key_match,
        rows_with_compustat_key_but_no_pit_record,
        len(pit_violations),
    )

    return {
        "summary": summary_path,
        "feature_coverage": cov_path,
        "examples": examples_path,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    outputs = run_compustat_integration_smoke(cfg)
    for name, path in outputs.items():
        logger.info("Wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
