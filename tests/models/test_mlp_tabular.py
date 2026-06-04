"""Unit tests for the compact tabular MLP estimator (CPU, no GPU required)."""

from __future__ import annotations

import numpy as np
import pytest

from mlfinance.models.mlp_tabular import MLPRegressor

_CPU_KW = {"device": "cpu", "use_bf16": False, "batch_size": 64}


def _data(n: int = 400, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype("float32")
    beta = rng.standard_normal(d)
    y = (x @ beta) * 0.01 + rng.standard_normal(n) * 0.02
    return x, y


def test_fit_predict_shape_and_finite() -> None:
    x, y = _data()
    model = MLPRegressor(hidden_dims=[16], epochs=3, **_CPU_KW)
    model.fit(x[:300], y[:300], x[300:350], y[300:350])
    preds = model.predict(x[350:])
    assert preds.shape == (50,)
    assert preds.dtype == np.float64
    assert np.all(np.isfinite(preds))


def test_cpu_runs_are_deterministic() -> None:
    x, y = _data()
    kw = {"hidden_dims": [16], "epochs": 3, "seed": 11, **_CPU_KW}
    a = MLPRegressor(**kw).fit(x[:300], y[:300], x[300:350], y[300:350]).predict(x[350:])
    b = MLPRegressor(**kw).fit(x[:300], y[:300], x[300:350], y[300:350]).predict(x[350:])
    assert np.allclose(a, b, atol=1e-6)


def test_no_validation_trains_full_epoch_budget() -> None:
    x, y = _data()
    model = MLPRegressor(hidden_dims=[8], epochs=4, **_CPU_KW)
    model.fit(x, y)
    assert model.best_epoch_ == 4


def test_early_stopping_keeps_best_epoch_within_budget() -> None:
    x, y = _data()
    model = MLPRegressor(hidden_dims=[8], epochs=10, patience=2, **_CPU_KW)
    model.fit(x[:300], y[:300], x[300:350], y[300:350])
    assert model.best_epoch_ is not None
    assert 1 <= model.best_epoch_ <= 10


def test_multi_seed_averaging_trains_each_seed() -> None:
    x, y = _data()
    model = MLPRegressor(hidden_dims=[8], epochs=2, n_seeds=3, **_CPU_KW)
    model.fit(x[:300], y[:300])
    assert len(model._models) == 3
    assert model.predict(x[300:]).shape == (100,)


def test_mismatched_validation_arguments_raise() -> None:
    x, y = _data()
    model = MLPRegressor(**_CPU_KW)
    with pytest.raises(ValueError):
        model.fit(x, y, x, None)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        MLPRegressor().predict(np.zeros((2, 3), dtype="float32"))


def test_empty_predict_returns_empty_array() -> None:
    x, y = _data()
    model = MLPRegressor(hidden_dims=[8], epochs=1, **_CPU_KW).fit(x, y)
    assert model.predict(np.zeros((0, 8), dtype="float32")).shape == (0,)


def test_default_loss_is_mse() -> None:
    # Existing callers never pass `loss`; default must stay MSE (unchanged behaviour).
    assert MLPRegressor().loss == "mse"


def test_huber_loss_accepted_and_normalised() -> None:
    assert MLPRegressor(loss="huber").loss == "huber"
    assert MLPRegressor(loss="Huber").loss == "huber"  # case-insensitive


def test_invalid_loss_raises() -> None:
    with pytest.raises(ValueError):
        MLPRegressor(loss="mae")


def test_huber_fit_predict_smoke() -> None:
    x, y = _data()
    model = MLPRegressor(hidden_dims=[8], epochs=3, loss="huber", **_CPU_KW)
    model.fit(x[:300], y[:300], x[300:350], y[300:350])
    preds = model.predict(x[350:])
    assert preds.shape == (50,)
    assert preds.dtype == np.float64
    assert np.all(np.isfinite(preds))
