"""Random Features model: sample (theta, b) once, then Ridge on activation((X theta + b)/h)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from mlfinance.models.base import r_squared_oos
from mlfinance.models.ridge import ridge_eig

Activation = Literal["relu", "tanh", "cos"]
_ALLOWED_ACTIVATIONS: tuple[str, ...] = ("relu", "tanh", "cos")


def generate_random_weights(d: int, P: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw a random projection theta of shape (d, P) and bias (1, P) from N(0, 1).

    Uses the legacy ``RandomState`` for byte-for-byte parity with hw2/hw3 refs.
    """
    rng = np.random.RandomState(int(seed))
    theta = rng.randn(d, P)
    bias = rng.randn(1, P)
    return theta, bias


def compute_random_features(
    X: np.ndarray,
    theta: np.ndarray,
    bias: np.ndarray,
    h: float,
    activation: Activation = "relu",
) -> np.ndarray:
    """Apply the random projection and the elementwise activation."""
    if activation not in _ALLOWED_ACTIVATIONS:
        raise ValueError(f"activation must be one of {_ALLOWED_ACTIVATIONS!r}, got {activation!r}.")
    if h <= 0:
        raise ValueError(f"bandwidth h must be strictly positive, got {h}.")

    Z = X @ theta + bias
    s = Z / float(h)
    if activation == "relu":
        return s * (s > 0)
    if activation == "tanh":
        return np.tanh(s)
    return np.cos(s)


class RandomFeaturesModel:
    """Random-features Ridge with grid search over (h, z, activation).

    The projection is held fixed across grid points so the comparison is
    fair, and we pick the triple that maximises OOS R² on the val set.
    """

    name: str = "random_features"

    def __init__(
        self,
        P: int,
        h_grid: list[float],
        z_grid: list[float],
        activation_grid: list[str] | None = None,
        seed: int = 0,
    ):
        if P <= 0:
            raise ValueError(f"P must be positive, got {P}.")
        if len(h_grid) == 0 or len(z_grid) == 0:
            raise ValueError("h_grid and z_grid must be non-empty.")
        self.P: int = int(P)
        self.h_grid: list[float] = list(h_grid)
        self.z_grid: list[float] = list(z_grid)
        self.activation_grid: list[str] = (
            list(activation_grid) if activation_grid is not None else list(_ALLOWED_ACTIVATIONS)
        )
        for a in self.activation_grid:
            if a not in _ALLOWED_ACTIVATIONS:
                raise ValueError(f"activation {a!r} not in {_ALLOWED_ACTIVATIONS!r}.")
        self.seed: int = int(seed)

        self._theta: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._best_h: float | None = None
        self._best_z: float | None = None
        self._best_activation: str | None = None
        self._best_beta: np.ndarray | None = None
        self._val_r2: float | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> RandomFeaturesModel:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        d = X_train.shape[1]
        theta, bias = generate_random_weights(d, self.P, self.seed)
        self._theta, self._bias = theta, bias

        best_r2 = -np.inf
        best_h: float | None = None
        best_z: float | None = None
        best_activation: str | None = None
        best_beta: np.ndarray | None = None

        for activation in self.activation_grid:
            for h in self.h_grid:
                S_train = compute_random_features(X_train, theta, bias, h, activation)
                if X_val is not None:
                    S_val = compute_random_features(X_val, theta, bias, h, activation)
                    betas, val_preds = ridge_eig(S_train, y_train, S_val, self.z_grid)
                    for k, z in enumerate(self.z_grid):
                        r2 = r_squared_oos(y_val, val_preds[:, k])
                        if r2 > best_r2:
                            best_r2 = r2
                            best_h = h
                            best_z = z
                            best_activation = activation
                            best_beta = betas[:, k].copy()
                else:
                    # No val set: take the smallest z + first activation/h. Caller should pass val.
                    betas, _ = ridge_eig(S_train, y_train, S_train[:1], self.z_grid)
                    if best_beta is None:
                        best_h = h
                        best_z = self.z_grid[0]
                        best_activation = activation
                        best_beta = betas[:, 0].copy()
                        best_r2 = float("nan")

        if best_beta is None:
            raise RuntimeError("RandomFeaturesModel.fit failed to populate a best beta.")

        self._best_h = best_h
        self._best_z = best_z
        self._best_activation = best_activation
        self._best_beta = best_beta
        self._val_r2 = best_r2 if best_r2 != -np.inf else None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if (
            self._theta is None
            or self._bias is None
            or self._best_beta is None
            or self._best_h is None
            or self._best_activation is None
        ):
            raise RuntimeError("RandomFeaturesModel.predict called before fit.")
        S = compute_random_features(X, self._theta, self._bias, self._best_h, self._best_activation)
        return S @ self._best_beta

    @property
    def best_h(self) -> float:
        if self._best_h is None:
            raise RuntimeError("best_h queried before fit.")
        return self._best_h

    @property
    def best_z(self) -> float:
        if self._best_z is None:
            raise RuntimeError("best_z queried before fit.")
        return self._best_z

    @property
    def best_activation(self) -> str:
        if self._best_activation is None:
            raise RuntimeError("best_activation queried before fit.")
        return self._best_activation
