"""Focused tests for the linear benchmark linear-model pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from mlfinance.models.linear_preprocessing import LinearFeaturePreprocessor
from mlfinance.run.linear_benchmarks import (
    RETURN_ONLY_FEATURE_COLUMNS,
    build_return_only_feature_list,
    load_model_panel,
    locked_split_masks,
)
from mlfinance.utils.io import read_parquet_schema_names


def _main_like_cfg():
    return OmegaConf.create(
        {
            "data": {
                "date_col": "date",
            },
            "splits": {
                "locked": {
                    "train_start": "1985-01-31",
                    "train_end": "2009-12-31",
                    "val_start": "2010-01-31",
                    "val_end": "2016-12-31",
                    "test_start": "2017-01-31",
                    "test_end": "2024-12-31",
                }
            },
        }
    )


def test_locked_split_boundaries_match_main_config() -> None:
    cfg = _main_like_cfg()
    panel = pd.DataFrame({"date": pd.date_range("1984-12-31", "2025-01-31", freq="ME")})

    masks = locked_split_masks(panel, cfg)
    train_dates = panel.loc[masks["train"], "date"]
    validation_dates = panel.loc[masks["validation"], "date"]
    test_dates = panel.loc[masks["test"], "date"]

    assert train_dates.min() == pd.Timestamp("1985-01-31")
    assert train_dates.max() == pd.Timestamp("2009-12-31")
    assert validation_dates.min() == pd.Timestamp("2010-01-31")
    assert validation_dates.max() == pd.Timestamp("2016-12-31")
    assert test_dates.min() == pd.Timestamp("2017-01-31")
    assert test_dates.max() == pd.Timestamp("2024-12-31")

    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["train"] & masks["test"]).any()
    assert not (masks["validation"] & masks["test"]).any()


def test_return_only_feature_list_is_explicit_and_excludes_forbidden_columns() -> None:
    panel = pd.DataFrame(
        columns=[
            *RETURN_ONLY_FEATURE_COLUMNS,
            "target_ret_fwd_1m",
            "target_excess_ret_fwd_1m",
            "future_label",
            "prediction",
            "realized_return",
        ]
    )

    feature_cols = build_return_only_feature_list(panel)

    assert feature_cols == RETURN_ONLY_FEATURE_COLUMNS
    assert "target_ret_fwd_1m" not in feature_cols
    assert "target_excess_ret_fwd_1m" not in feature_cols
    assert "future_label" not in feature_cols
    assert "prediction" not in feature_cols
    assert "realized_return" not in feature_cols


def test_projected_model_panel_loader_preserves_rows_and_skips_unused_strings(
    tmp_path: Path,
) -> None:
    panel_path = tmp_path / "model_panel.parquet"
    rows: list[dict[str, object]] = []
    for row_idx, (permno, date, target) in enumerate(
        [
            (20002, "2021-02-10", 0.02),
            (20001, "2021-01-03", 0.01),
            (20001, "2021-03-15", None),
            (None, "2021-01-15", 0.03),
        ]
    ):
        row = {
            "permno": permno,
            "date": date,
            "target_ret_fwd_1m": target,
            "cusip": f"CUSIP{row_idx}",
            "cusip8": f"CUSIP8{row_idx}",
            "ticker": f"T{row_idx}",
        }
        for feature_idx, feature in enumerate(RETURN_ONLY_FEATURE_COLUMNS, start=1):
            row[feature] = float(feature_idx + row_idx)
        rows.append(row)

    rows[1]["ret_2m"] = None  # feature NaN should survive panel loading
    raw = pd.DataFrame(rows)
    raw.to_parquet(panel_path, index=False)

    schema_names = read_parquet_schema_names(panel_path)
    feature_cols = build_return_only_feature_list(schema_names)
    cfg = OmegaConf.create(
        {
            "data": {
                "panel_path": str(panel_path),
                "permno_col": "permno",
                "date_col": "date",
                "target_col": "target_ret_fwd_1m",
            }
        }
    )

    loaded = load_model_panel(cfg, feature_cols)
    expected = raw[
        [
            "permno",
            "date",
            "target_ret_fwd_1m",
            *RETURN_ONLY_FEATURE_COLUMNS,
        ]
    ].copy()
    expected["date"] = pd.to_datetime(expected["date"]) + pd.offsets.MonthEnd(0)
    expected = expected.dropna(subset=["permno", "date", "target_ret_fwd_1m"])
    expected = expected.sort_values(["date", "permno"]).reset_index(drop=True)

    assert feature_cols == RETURN_ONLY_FEATURE_COLUMNS
    assert loaded.columns.tolist() == expected.columns.tolist()
    assert "cusip" not in loaded.columns
    assert "cusip8" not in loaded.columns
    assert "ticker" not in loaded.columns
    pd.testing.assert_frame_equal(loaded, expected)


def test_linear_preprocessor_fit_uses_only_estimation_sample() -> None:
    train_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-31", "2018-02-28", "2018-03-31"]),
            "ret_1m": [1.0, 2.0, 3.0],
            "ret_2m": [0.0, 1.0, 2.0],
        }
    )
    preprocessor = LinearFeaturePreprocessor(["ret_1m", "ret_2m"], date_col="date")
    preprocessor.fit(train_df)

    transformed = preprocessor.transform(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2018-04-30"]),
                "ret_1m": [1000.0],
                "ret_2m": [None],
            }
        )
    )

    metadata = preprocessor.metadata()
    assert metadata["n_rows"] == 3
    assert metadata["fit_start"] == "2018-01-31"
    assert metadata["fit_end"] == "2018-03-31"
    assert float(preprocessor.upper_bounds_["ret_1m"]) < 1000.0
    assert transformed.columns.tolist() == [
        "ret_1m",
        "ret_2m",
        "ret_1m_missing",
        "ret_2m_missing",
    ]
    assert transformed.loc[0, "ret_2m_missing"] == 1.0
