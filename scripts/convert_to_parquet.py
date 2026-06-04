"""Convert raw-data CSVs to snappy-compressed parquet for faster loading.

Reads the three large CSVs in ``data/raw`` (CRSP monthly, Compustat
quarterly, JKP factors) and writes parquets next to them. Parquet is
~5-10x faster to load and ~3-5x smaller than CSV.

The conversion is non-destructive by default: source CSVs stay in place
unless ``--delete-csv`` is passed explicitly.

Usage:
    python scripts/convert_to_parquet.py               # convert all, keep CSVs
    python scripts/convert_to_parquet.py --force       # regenerate even if exists
    python scripts/convert_to_parquet.py --delete-csv  # remove CSV after conversion
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass
class CsvSpec:
    """One CSV -> parquet conversion job."""

    filename: str
    label: str
    reader: Callable[[Path], pd.DataFrame]


def read_crsp(path: Path) -> pd.DataFrame:
    """Read ``monthly_crsp.csv`` with parsed dates and nullable Int64 PERMNO."""
    df = pd.read_csv(
        path,
        parse_dates=["MthCalDt"],
        engine="c",
    )
    df["PERMNO"] = df["PERMNO"].astype("Int64")
    return df


def read_compustat(path: Path) -> pd.DataFrame:
    """Read ``CompFirmCharac.csv``. Column 10 is mixed-type hence low_memory=False.

    gvkey / cik stored as strings to preserve leading zeros across the pipeline.
    """
    df = pd.read_csv(
        path,
        low_memory=False,
        parse_dates=["datadate"],
    )
    df["gvkey"] = df["gvkey"].astype("Int64").astype("string").str.zfill(6)
    # cik is float-coded in source (NaNs); Int64 -> 10-char string keeps real NaNs.
    cik_int = df["cik"].astype("Int64")
    df["cik"] = cik_int.astype("string").str.zfill(10)
    df.loc[cik_int.isna(), "cik"] = pd.NA
    return df


def read_jkp(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["date"],
        engine="c",
    )
    return df


def _fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024 / 1024:,.1f} MB"


def _fmt_secs(s: float) -> str:
    return f"{s:6.2f} s"


def convert_one(spec: CsvSpec, *, force: bool, delete_csv: bool) -> dict | None:
    """Convert one CSV; returns benchmark stats or None if skipped."""
    csv_path = RAW_DIR / spec.filename
    parquet_path = csv_path.with_suffix(".parquet")

    print(f"\n--- {spec.label} ---")
    print(f"CSV     : {csv_path}")
    print(f"Parquet : {parquet_path}")

    if not csv_path.exists():
        if parquet_path.exists():
            print("  -> CSV absent and parquet exists, nothing to do.")
            return None
        print(f"  ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return None

    if parquet_path.exists() and not force:
        print("  -> Parquet exists; skipping (use --force to regenerate).")
        return None

    csv_bytes = csv_path.stat().st_size

    t0 = time.perf_counter()
    df = spec.reader(csv_path)
    csv_read_secs = time.perf_counter() - t0
    print(
        f"  Read CSV     : {_fmt_secs(csv_read_secs)} "
        f"({df.shape[0]:,} rows x {df.shape[1]} cols, {_fmt_mb(csv_bytes)})"
    )

    t0 = time.perf_counter()
    df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    write_secs = time.perf_counter() - t0
    parquet_bytes = parquet_path.stat().st_size
    print(f"  Write parquet: {_fmt_secs(write_secs)} " f"({_fmt_mb(parquet_bytes)})")

    # Free in-memory frame so the parquet read benchmark isn't aided by page-cache warm-up.
    del df
    t0 = time.perf_counter()
    df2 = pd.read_parquet(parquet_path, engine="pyarrow")
    parquet_read_secs = time.perf_counter() - t0
    print(
        f"  Read parquet : {_fmt_secs(parquet_read_secs)} "
        f"({df2.shape[0]:,} rows x {df2.shape[1]} cols)"
    )

    speedup = csv_read_secs / parquet_read_secs if parquet_read_secs > 0 else float("inf")
    size_ratio = csv_bytes / parquet_bytes if parquet_bytes > 0 else float("inf")
    print(
        f"  Speedup      : {speedup:5.2f}x faster | " f"size reduction: {size_ratio:4.2f}x smaller"
    )

    stats = {
        "label": spec.label,
        "filename": spec.filename,
        "csv_bytes": csv_bytes,
        "parquet_bytes": parquet_bytes,
        "csv_read_secs": csv_read_secs,
        "parquet_read_secs": parquet_read_secs,
        "write_secs": write_secs,
        "rows": int(df2.shape[0]),
        "cols": int(df2.shape[1]),
        "speedup": speedup,
        "size_ratio": size_ratio,
    }

    if delete_csv:
        csv_path.unlink()
        print(f"  Deleted CSV  : {csv_path.name} (--delete-csv)")
    else:
        print(f"  Kept CSV     : {csv_path.name} (default)")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate parquet even if it already exists.",
    )
    parser.add_argument(
        "--delete-csv",
        action="store_true",
        help="Delete the source CSV after a successful conversion " "(default: keep it in place).",
    )
    args = parser.parse_args()

    specs: list[CsvSpec] = [
        CsvSpec("monthly_crsp.csv", "CRSP monthly returns", read_crsp),
        CsvSpec("CompFirmCharac.csv", "Compustat fundamentals", read_compustat),
        CsvSpec(
            r"[usa]_[all_factors]_[monthly]_[vw_cap].csv",
            "JKP factor returns",
            read_jkp,
        ),
    ]

    print(f"Raw-data directory: {RAW_DIR}")
    if not RAW_DIR.exists():
        print(f"ERROR: directory not found: {RAW_DIR}", file=sys.stderr)
        return 1

    overall_t0 = time.perf_counter()
    all_stats: list[dict] = []
    for spec in specs:
        stats = convert_one(spec, force=args.force, delete_csv=args.delete_csv)
        if stats is not None:
            all_stats.append(stats)
    total_secs = time.perf_counter() - overall_t0

    print("\n" + "=" * 76)
    print("Summary")
    print("=" * 76)
    if not all_stats:
        print("No files were converted.")
    else:
        header = f"{'File':<40} {'CSV':>10} {'Parquet':>10} {'CSV t':>8} {'Pq t':>8} {'Speedup':>8}"
        print(header)
        print("-" * len(header))
        tot_csv = 0
        tot_pq = 0
        tot_csv_t = 0.0
        tot_pq_t = 0.0
        for s in all_stats:
            print(
                f"{s['filename']:<40} "
                f"{_fmt_mb(s['csv_bytes']):>10} "
                f"{_fmt_mb(s['parquet_bytes']):>10} "
                f"{s['csv_read_secs']:>7.2f}s "
                f"{s['parquet_read_secs']:>7.2f}s "
                f"{s['speedup']:>7.2f}x"
            )
            tot_csv += s["csv_bytes"]
            tot_pq += s["parquet_bytes"]
            tot_csv_t += s["csv_read_secs"]
            tot_pq_t += s["parquet_read_secs"]
        print("-" * len(header))
        total_speedup = tot_csv_t / tot_pq_t if tot_pq_t > 0 else float("inf")
        print(
            f"{'TOTAL':<40} "
            f"{_fmt_mb(tot_csv):>10} "
            f"{_fmt_mb(tot_pq):>10} "
            f"{tot_csv_t:>7.2f}s "
            f"{tot_pq_t:>7.2f}s "
            f"{total_speedup:>7.2f}x"
        )
        print(f"\nTotal time saved per full panel-build run: {tot_csv_t - tot_pq_t:.2f} s")
    print(f"Total wall time      : {total_secs:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
