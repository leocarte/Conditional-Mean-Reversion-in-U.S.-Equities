from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from mlfinance.run.classical_baselines import (
    FEATURE_SET,
    build_classical_baseline_predictions,
    required_model_panel_columns,
)


def _cfg():
    return OmegaConf.create(
        {
            "data": {
                "date_col": "date",
                "permno_col": "permno",
                "target_col": "target_ret_fwd_1m",
            }
        }
    )


def test_required_model_panel_columns_are_minimal() -> None:
    cols = required_model_panel_columns(
        models=("reversal", "momentum"),
        date_col="date",
        permno_col="permno",
        target_col="target_ret_fwd_1m",
        available_columns=[
            "date",
            "permno",
            "target_ret_fwd_1m",
            "ret_1m",
            "mom_2_12",
            "prediction",
            "realized_date",
            "target_excess_ret_fwd_1m",
        ],
    )
    assert cols == ["date", "permno", "target_ret_fwd_1m", "ret_1m", "mom_2_12"]
    assert "prediction" not in cols
    assert "realized_date" not in cols
    assert "target_excess_ret_fwd_1m" not in cols


def test_build_predictions_are_deterministic_and_ignore_future_like_columns() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2017-01-31",
                    "2017-01-31",
                    "2017-01-31",
                    "2017-02-28",
                    "2017-02-28",
                    "2017-02-28",
                ]
            ),
            "permno": [101, 102, 103, 101, 102, 103],
            "target_ret_fwd_1m": [0.02, -0.01, 0.01, 0.01, 0.0, -0.02],
            "ret_1m": [0.10, 0.00, -0.10, 0.06, 0.02, -0.04],
            "mom_2_12": [-0.20, 0.00, 0.20, -0.10, 0.05, 0.15],
            "vol_12m": [0.3, 0.3, 0.3, 0.2, 0.2, 0.2],
            "prediction": [999.0] * 6,
            "realized_date": pd.to_datetime(["2030-01-31"] * 6),
            "future_target_noise": [1000.0, -1000.0, 500.0, 400.0, -400.0, 250.0],
        }
    )

    preds = build_classical_baseline_predictions(
        panel,
        _cfg(),
        models=("reversal", "momentum", "zscore_reversal"),
    )

    assert {"prediction", "model_name", "feature_set", "target_col"}.issubset(preds.columns)
    assert set(preds["feature_set"].unique()) == {FEATURE_SET}
    assert set(preds["target_col"].unique()) == {"target_ret_fwd_1m"}

    reversal = preds.loc[preds["model_name"] == "reversal", "prediction"].to_numpy()
    np.testing.assert_allclose(reversal, -panel["ret_1m"].to_numpy())

    momentum = preds.loc[preds["model_name"] == "momentum", "prediction"].to_numpy()
    np.testing.assert_allclose(momentum, panel["mom_2_12"].to_numpy())

    jan = preds[
        (preds["model_name"] == "zscore_reversal")
        & (pd.to_datetime(preds["date"]) == pd.Timestamp("2017-01-31"))
    ].sort_values("permno")
    expected_jan = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(jan["prediction"].to_numpy(), expected_jan)
