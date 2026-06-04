"""Shared pytest fixtures: small synthetic panels mirroring the real CRSP/JKP shape."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_STOCKS = 50
N_MONTHS = 3
N_JKP_FEATURES = 12
START_MONTH = "2010-01-31"
SEED = 0


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def month_index() -> pd.DatetimeIndex:
    return pd.date_range(START_MONTH, periods=N_MONTHS, freq="ME")


@pytest.fixture(scope="session")
def permno_list() -> list[int]:
    return list(range(10001, 10001 + N_STOCKS))


@pytest.fixture(scope="session")
def synthetic_panel(
    rng: np.random.Generator,
    month_index: pd.DatetimeIndex,
    permno_list: list[int],
) -> pd.DataFrame:
    """3-month x 50-stock synthetic panel with JKP-style features and a ret_exc target."""
    records = []
    for permno in permno_list:
        for date in month_index:
            row = {
                "permno": permno,
                "date": date,
                "yyyymm": int(date.strftime("%Y%m")),
                "mktcap": float(rng.lognormal(mean=8.0, sigma=1.5)),
                "gvkey": permno * 10,
                "cik": permno * 100,
                "ret_exc": float(rng.normal(loc=0.005, scale=0.08)),
            }
            for i in range(N_JKP_FEATURES):
                row[f"jkp_{i:02d}"] = float(rng.standard_normal())
            records.append(row)

    df = pd.DataFrame.from_records(records)

    df["ret_exc_rank"] = (
        df.groupby("date")["ret_exc"].rank(method="average", pct=True).astype("float64")
    )

    id_cols = ["permno", "date", "yyyymm", "gvkey", "cik", "mktcap"]
    target_cols = ["ret_exc", "ret_exc_rank"]
    feature_cols = [f"jkp_{i:02d}" for i in range(N_JKP_FEATURES)]
    return df[id_cols + target_cols + feature_cols].reset_index(drop=True)


@pytest.fixture
def tmp_parquet_dir(tmp_path: Path) -> Path:
    d = tmp_path / "parquet"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def feature_columns() -> list[str]:
    return [f"jkp_{i:02d}" for i in range(N_JKP_FEATURES)]
