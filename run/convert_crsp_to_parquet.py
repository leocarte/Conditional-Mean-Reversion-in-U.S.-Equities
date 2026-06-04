"""Safely convert the raw Monthly CRSP CSV to parquet for baseline.

This helper never deletes or mutates the source CSV. It exists so the active
baseline path can benefit from parquet load speed without relying on the
older broad conversion script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlfinance.utils.io import write_parquet


def convert_crsp_csv_to_parquet(
    csv_path: str | Path,
    parquet_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Convert one Monthly CRSP CSV to parquet without touching the CSV."""
    src = Path(csv_path)
    dst = src.with_suffix(".parquet") if parquet_path is None else Path(parquet_path)

    if not src.exists():
        raise FileNotFoundError(f"Monthly CRSP CSV not found: {src}")
    if dst.exists() and not force:
        return dst

    df = pd.read_csv(src, low_memory=False)
    return write_parquet(df, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/monthly_crsp.csv",
        help="Source Monthly CRSP CSV path.",
    )
    parser.add_argument(
        "--output",
        dest="parquet_path",
        default=None,
        help="Destination parquet path. Defaults to the CSV path with .parquet suffix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the parquet if it already exists.",
    )
    args = parser.parse_args()

    out = convert_crsp_csv_to_parquet(
        args.csv_path,
        parquet_path=args.parquet_path,
        force=args.force,
    )
    print(out)


if __name__ == "__main__":  # pragma: no cover
    main()
