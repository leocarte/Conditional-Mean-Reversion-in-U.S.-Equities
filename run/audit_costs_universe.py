"""Cost and universe robustness audit (benchmark bridge).

Re-evaluates the existing return-only model predictions under (a) the flat-bps
cost grid and (b) feasible CRSP universe restrictions, WITHOUT retraining any
model. It loads the saved per-model prediction parquets, re-runs the existing
``run_prediction_backtests`` (unchanged) on the full and the universe-filtered
predictions, and summarises with ``summarize_backtests_by_cost``.

The provided Monthly CRSP parquet ships without ``prc``/``shrout``/``shrcd``/
``exchcd`` (see ``mlfinance/data/loaders.py``), so only SIC-sector restrictions
are feasible today; the price / exchange / share-code / market-cap screens are
gated on column availability and skipped + reported, never faked (see
``mlfinance/data/universe.py`` and ``docs/costs_universe_robustness_note.md``).

Performance: no model is retrained -- only the cheap, vectorised backtest is
re-run on saved predictions; the panel is read column-projected to
``[permno, date, <universe fields>]`` to skip decoding the heavy CRSP strings.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

from mlfinance.data.universe import FILTER_FIELD_REQUIREMENTS, apply_universe
from mlfinance.run.baseline_pipeline import summarize_backtests_by_cost
from mlfinance.run.linear_benchmarks import _split_id, run_prediction_backtests
from mlfinance.utils.io import read_parquet

logger = logging.getLogger(__name__)

# Optional CRSP field -> the universe filter it would enable if present.
CANDIDATE_UNIVERSE_FIELDS: dict[str, str] = {
    "siccd": "sic_exclude (sector)",
    "prc": "price_min",
    "shrout": "market_cap_min (needs prc)",
    "shrcd": "shrcd_keep (common shares)",
    "exchcd": "exchcd_keep (exchange)",
}

_REPORT_COLUMNS = ["variant", "name", "applied", "reason", "rows_before", "rows_after"]


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _panel_field_names(panel_path: str) -> set[str]:
    """Lower-cased column names from the parquet footer (no data pages read)."""
    return {str(n).lower() for n in pq.read_schema(str(panel_path)).names}


def _as_date_string(value: pd.Timestamp | str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def build_universe_audit(panel_path: str) -> pd.DataFrame:
    """Availability + non-null coverage of optional CRSP universe fields."""
    names = _panel_field_names(panel_path)
    present = [f for f in CANDIDATE_UNIVERSE_FIELDS if f in names]
    coverage: dict[str, float] = {}
    if present:
        sub = read_parquet(panel_path, columns=present)
        coverage = {f: float(sub[f].notna().mean()) for f in present}
    rows = [
        {
            "field": field,
            "enables_filter": enables,
            "present_in_panel": field in names,
            "nonnull_coverage": round(coverage[field], 4) if field in coverage else float("nan"),
            "filter_feasible": field in names,
        }
        for field, enables in CANDIDATE_UNIVERSE_FIELDS.items()
    ]
    return pd.DataFrame(rows)


def load_saved_predictions(cfg: DictConfig, split_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate locked-test prediction parquets and emit a per-file manifest."""
    pred_dir = Path(cfg.paths.predictions)
    date_col = cfg.data.date_col
    files = sorted(pred_dir.glob(f"*__{split_id}.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No prediction parquets matching '*__{split_id}.parquet' in {pred_dir}. "
            "Produce them first via baseline + linear-benchmarks + nonlinear-benchmarks "
            "(locally or on cluster)."
        )
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for path in files:
        frame = read_parquet(path)
        if date_col not in frame.columns:
            raise KeyError(f"Prediction file {path.name} is missing required '{date_col}' column.")

        model_names = sorted(
            {str(x) for x in frame.get("model_name", pd.Series(dtype="object")).dropna()}
        )
        if len(model_names) > 1:
            raise ValueError(
                f"Prediction file {path.name} contains multiple model_name values: {model_names}"
            )
        split_ids = sorted(
            {str(x) for x in frame.get("split_id", pd.Series(dtype="object")).dropna()}
        )
        if len(split_ids) > 1:
            raise ValueError(
                f"Prediction file {path.name} contains multiple split_id values: {split_ids}"
            )

        dates = pd.to_datetime(frame[date_col], errors="coerce")
        frames.append(frame)
        manifest_rows.append(
            {
                "filename": path.name,
                "model_name": model_names[0] if model_names else path.stem.rsplit("__", 1)[0],
                "row_count": int(len(frame)),
                "min_date": _as_date_string(dates.min()),
                "max_date": _as_date_string(dates.max()),
                "split_id": split_ids[0] if split_ids else split_id,
            }
        )

    predictions = pd.concat(frames, ignore_index=True)
    manifest = (
        pd.DataFrame(manifest_rows).sort_values(["model_name", "filename"]).reset_index(drop=True)
    )
    logger.info("Loaded %d prediction rows from %d model files.", len(predictions), len(files))
    return predictions, manifest


def merge_universe_fields(predictions: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Left-merge AVAILABLE CRSP universe fields onto predictions by [permno, date] (time-t)."""
    permno_col = cfg.data.permno_col
    date_col = cfg.data.date_col
    names = _panel_field_names(cfg.data.panel_path)
    fields = [f for f in CANDIDATE_UNIVERSE_FIELDS if f in names]
    if not fields:
        return predictions
    panel = read_parquet(cfg.data.panel_path, columns=[permno_col, date_col, *fields])
    panel[date_col] = pd.to_datetime(panel[date_col]) + pd.offsets.MonthEnd(0)
    dup_mask = panel.duplicated(subset=[permno_col, date_col], keep=False)
    if dup_mask.any():
        dup_keys = (
            panel.loc[dup_mask, [permno_col, date_col]]
            .drop_duplicates()
            .sort_values([date_col, permno_col])
            .reset_index(drop=True)
        )
        sample = dup_keys.head(5).assign(**{date_col: lambda d: d[date_col].dt.date.astype(str)})
        raise ValueError(
            "Panel universe fields must be unique on "
            f"[{permno_col}, {date_col}] before merging onto predictions. "
            f"Found {len(dup_keys)} duplicate key(s) in {cfg.data.panel_path}; sample="
            f"{sample.to_dict(orient='records')}"
        )
    return predictions.merge(
        panel,
        on=[permno_col, date_col],
        how="left",
        validate="many_to_one",
    )


def _requested_universe_filter_names(spec: dict[str, Any]) -> list[str]:
    requested: list[str] = []
    for name in FILTER_FIELD_REQUIREMENTS:
        value = spec.get(name)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            continue
        requested.append(name)
    return requested


def _universe_summary_flags(
    variant: str,
    raw_spec: dict[str, Any],
    reports: list[Any],
) -> dict[str, Any]:
    requested_names = _requested_universe_filter_names(raw_spec)
    report_map = {report.name: report for report in reports}
    filters_requested = len(requested_names)
    filters_applied = sum(
        bool(report_map[name].applied) for name in requested_names if name in report_map
    )
    filters_skipped = sum(
        not bool(report_map[name].applied) for name in requested_names if name in report_map
    )
    return {
        "universe": variant,
        "filters_requested": filters_requested,
        "filters_applied": filters_applied,
        "filters_skipped": filters_skipped,
        "is_noop_universe": bool(
            filters_requested > 0 and filters_applied == 0 and filters_skipped == filters_requested
        ),
    }


def run_audit_costs_universe(cfg: DictConfig) -> dict[str, Path]:
    """Cost + universe robustness audit over saved predictions (no retraining)."""
    split_id = _split_id(cfg)
    metrics_dir = Path(cfg.paths.metrics)
    permno_col = cfg.data.permno_col
    date_col = cfg.data.date_col

    audit = build_universe_audit(cfg.data.panel_path)
    audit_path = _write_csv(audit, metrics_dir / "universe_audit_summary.csv")
    logger.info("Universe field availability:\n%s", audit.to_string(index=False))

    predictions, prediction_manifest = load_saved_predictions(cfg, split_id)
    manifest_path = _write_csv(prediction_manifest, metrics_dir / "prediction_manifest.csv")
    predictions = merge_universe_fields(predictions, cfg)

    # Cost robustness on the unrestricted (all-CRSP) universe.
    bt_all = run_prediction_backtests(predictions, cfg, split_id=split_id)
    cost_summary = summarize_backtests_by_cost(bt_all)
    cost_summary.insert(0, "universe", "all")
    cost_path = _write_csv(cost_summary, metrics_dir / "cost_robustness_summary.csv")

    # Universe robustness across the configured (availability-gated) variants.
    min_n = int(OmegaConf.select(cfg, "universe.min_stocks_per_month", default=0) or 0)
    variants = OmegaConf.to_container(cfg.universe.variants, resolve=True) or {}
    summaries: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []
    variant_flags: list[dict[str, Any]] = []
    for variant, raw_spec in variants.items():
        spec = dict(raw_spec or {})
        if min_n > 0:
            spec.setdefault("min_stocks_per_month", min_n)
        filtered, reports = apply_universe(
            predictions, spec, date_col=date_col, permno_col=permno_col
        )
        report_rows.extend({"variant": variant, **asdict(r)} for r in reports)
        variant_flags.append(_universe_summary_flags(variant, dict(raw_spec or {}), reports))
        bt = run_prediction_backtests(filtered, cfg, split_id=split_id)
        summary = summarize_backtests_by_cost(bt)
        summary.insert(0, "universe", variant)
        summary.insert(1, "n_rows", len(filtered))
        summaries.append(summary)

    universe_summary = pd.concat(summaries, ignore_index=True)
    universe_summary = universe_summary.merge(
        pd.DataFrame(variant_flags),
        on="universe",
        how="left",
        validate="many_to_one",
    )
    universe_path = _write_csv(universe_summary, metrics_dir / "universe_robustness_summary.csv")
    report_path = _write_csv(
        pd.DataFrame(report_rows, columns=_REPORT_COLUMNS),
        metrics_dir / "universe_filter_report.csv",
    )

    return {
        "universe_audit_summary": audit_path,
        "prediction_manifest": manifest_path,
        "cost_robustness_summary": cost_path,
        "universe_robustness_summary": universe_path,
        "universe_filter_report": report_path,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="main")
def main(cfg: DictConfig) -> None:
    outputs = run_audit_costs_universe(cfg)
    for name, path in outputs.items():
        logger.info("Wrote %s -> %s", name, path)


if __name__ == "__main__":
    main()
