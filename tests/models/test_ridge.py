"""Tests for mlfinance.models.ridge."""

from __future__ import annotations

import numpy as np
import pytest

from mlfinance.models.ridge import RidgeEig, RidgelessModel, ridge_eig, ridgeless_min_norm


def _make_underparam(seed: int = 0, T: int = 200, P: int = 20):
    rng = np.random.RandomState(seed)
    X = rng.randn(T, P)
    beta_true = rng.randn(P)
    y = X @ beta_true + 0.1 * rng.randn(T)
    return X, y, beta_true


def _make_overparam(seed: int = 0, T: int = 20, P: int = 200):
    rng = np.random.RandomState(seed)
    X = rng.randn(T, P)
    beta_true = rng.randn(P) * 0.1
    y = X @ beta_true + 0.05 * rng.randn(T)
    return X, y, beta_true


def test_ridge_eig_matches_closed_form() -> None:
    """ridge_eig matches the textbook closed form (P < T branch)."""
    X, y, _ = _make_underparam(seed=1, T=200, P=20)
    T, P = X.shape
    zs = [0.01, 0.1, 1.0]

    betas, _ = ridge_eig(X, y, X[:1], zs)

    eye = np.eye(P)
    for k, z in enumerate(zs):
        A = X.T @ X / T + z * eye
        b = X.T @ y / T
        beta_cf = np.linalg.solve(A, b)
        np.testing.assert_allclose(betas[:, k], beta_cf, atol=1e-10, rtol=1e-10)


def test_ridge_eig_matches_closed_form_overparam() -> None:
    """ridge_eig also matches the closed form when P > T."""
    X, y, _ = _make_overparam(seed=2, T=20, P=200)
    T, P = X.shape
    zs = [0.05, 0.5]

    betas, _ = ridge_eig(X, y, X[:1], zs)

    eye_p = np.eye(P)
    for k, z in enumerate(zs):
        A = X.T @ X / T + z * eye_p
        b = X.T @ y / T
        beta_cf = np.linalg.solve(A, b)
        np.testing.assert_allclose(betas[:, k], beta_cf, atol=1e-8, rtol=1e-8)


def test_ridge_pgtm_branch() -> None:
    """Over-parameterised branch agrees with the explicit P×P closed form."""
    rng = np.random.RandomState(7)
    T, P = 15, 50
    X = rng.randn(T, P)
    y = rng.randn(T)
    zs = [0.5]

    betas, preds = ridge_eig(X, y, X, zs)

    eye_p = np.eye(P)
    A = X.T @ X / T + zs[0] * eye_p
    b = X.T @ y / T
    beta_cf = np.linalg.solve(A, b)
    np.testing.assert_allclose(betas[:, 0], beta_cf, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(preds[:, 0], X @ beta_cf, atol=1e-8, rtol=1e-8)


def test_ridge_eig_wrapper_picks_best_shrinkage() -> None:
    """RidgeEig wrapper picks the shrinkage that maximises val R²."""
    X, y, _ = _make_underparam(seed=3, T=400, P=20)
    X_train, X_val = X[:300], X[300:]
    y_train, y_val = y[:300], y[300:]

    model = RidgeEig(shrinkage_grid=[1e-6, 1e-2, 1.0, 100.0])
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Smallest shrinkage should win on this near-noiseless under-param problem.
    assert model.best_shrinkage <= 1e-2

    preds = model.predict(X_val)
    assert preds.shape == (X_val.shape[0],)


def test_ridge_eig_requires_paired_val_inputs() -> None:
    X, y, _ = _make_underparam(seed=4)
    model = RidgeEig(shrinkage_grid=[1e-3])
    with pytest.raises(ValueError):
        model.fit(X, y, X_val=X, y_val=None)


def test_ridgeless_min_norm_interpolates_in_overparam() -> None:
    """P >> T: min-norm interpolator fits train near-exactly."""
    X, y, _ = _make_overparam(seed=5, T=10, P=200)
    beta, preds = ridgeless_min_norm(X, y, X, eps=1e-12)
    np.testing.assert_allclose(preds, y, atol=1e-6, rtol=1e-6)
    assert beta.shape == (X.shape[1],)


def test_ridgeless_model_wrapper_runs() -> None:
    X, y, _ = _make_overparam(seed=6, T=20, P=80)
    model = RidgelessModel(eps=1e-10).fit(X, y)
    preds = model.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-4, rtol=1e-4)
