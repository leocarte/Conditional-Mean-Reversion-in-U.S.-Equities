"""Parquet I/O helpers with schema-validation hooks.

Used by every CLI in ``mlfinance.run`` so misnamed columns surface at the
I/O boundary, not deep inside a model class.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PathLike = str | Path


__all__ = [
    "read_parquet",
    "read_parquet_schema_names",
    "write_parquet",
    "ensure_parent",
    "validate_columns",
]


def ensure_parent(path: PathLike) -> Path:
    """Create the parent of ``path`` if missing; return ``Path(path)``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    where: str = "<dataframe>",
) -> None:
    """Raise ``ValueError`` if any of ``required_columns`` is missing."""
    required = list(required_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{where}: missing required columns {missing}. "
            f"Available columns: {list(df.columns)[:25]}" + ("..." if len(df.columns) > 25 else "")
        )


def read_parquet_schema_names(path: PathLike) -> list[str]:
    """Read only parquet schema metadata and return the column names."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")

    names = list(pq.read_schema(p).names)
    logger.debug("Read parquet schema %s - cols=%d", p, len(names))
    return names


def read_parquet(
    path: PathLike,
    *,
    columns: Iterable[str] | None = None,
    required_columns: Iterable[str] | None = None,
    engine: str = "pyarrow",
) -> pd.DataFrame:
    """Read a parquet file and optionally validate the schema (after projection)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")

    cols = list(columns) if columns is not None else None
    df = pd.read_parquet(p, columns=cols, engine=engine)

    logger.debug(
        "Read parquet %s - rows=%d, cols=%d, mem=%.1f MB",
        p,
        len(df),
        len(df.columns),
        df.memory_usage(deep=True).sum() / 1024 / 1024,
    )

    if required_columns is not None:
        validate_columns(df, required_columns, where=str(p))

    return df


def write_parquet(
    df: pd.DataFrame,
    path: PathLike,
    *,
    required_columns: Iterable[str] | None = None,
    compression: str = "snappy",
    engine: str = "pyarrow",
    index: bool = False,
) -> Path:
    """Write a parquet file, optionally validating the schema first.

    Writes atomically via a sibling ``<path>.tmp`` file then ``os.replace``.
    A pod killed mid-write therefore never leaves a partially-written
    parquet at the target path; readers either see the previous version
    (if any) or the new one in full. The backtester reads every parquet
    in ``data/processed/predictions/`` and would crash on a corrupt file,
    losing a 1-hour train pod's output and stalling the whole pipeline,
    so this guard is worth the extra rename.
    """
    p = ensure_parent(path)

    if required_columns is not None:
        validate_columns(df, required_columns, where=str(p))

    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp, compression=compression, engine=engine, index=index)
    tmp.replace(p)
    logger.debug(
        "Wrote parquet %s (atomic via .tmp) - rows=%d, cols=%d, compression=%s",
        p,
        len(df),
        len(df.columns),
        compression,
    )
    return p
