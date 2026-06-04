"""Ridge regression via eigendecomposition (handles P < T and P >= T)."""

from __future__ import annotations

import numpy as np

from mlfinance.models.base import r_squared_oos


def ridge_eig(
    S_train: np.ndarray,
    y_train: np.ndarray,
    S_test: np.ndarray,
    shrinkage_list: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge for a list of shrinkages via one eigendecomposition.

    Solves ``beta(z) = argmin (1/T) ||S beta - y||^2 + z ||beta||^2`` for
    every ``z``. Picks the P×P decomposition when P < T (under-parameterised)
    or the T×T decomposition when P >= T (over-parameterised; matters for
    random-features and the large final-project models).
    """
    # Preserve input dtype: float32 input stays float32, which halves memory
    # and uses LAPACK's SSYRK/SSYEVD (~2x faster than DSYRK/DSYEVD on Xeon).
    # For our P<=2173 gram with z >= 1e-4 the cond number stays well below
    # float32 precision limits; eigh residual ~1e-5 is below the noise floor
    # of monthly cross-sectional R^2. Int/bool inputs get promoted to float64.
    dtype = S_train.dtype if S_train.dtype in (np.float32, np.float64) else np.float64
    S_train = np.asarray(S_train, dtype=dtype)
    y_train = np.asarray(y_train, dtype=dtype).reshape(-1)
    S_test = np.asarray(S_test, dtype=dtype)

    t_, p_ = S_train.shape
    # Cast Python-float shrinkages to the input dtype. Otherwise
    # `f32_array + python_float` would silently upcast the entire path
    # solve to f64 under NumPy 1.x promotion rules, doubling memory and
    # falling back to DSYRK/DGEMM (~2x slower than SSYRK/SGEMM).
    one = np.asarray(1.0, dtype=dtype)
    zs = [np.asarray(z, dtype=dtype) for z in shrinkage_list]
    if p_ < t_:
        eigenvalues, eigenvectors = np.linalg.eigh(S_train.T @ S_train / t_)
        means = S_train.T @ y_train.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate(
            [(one / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in zs],
            axis=1,
        )
        betas = eigenvectors @ intermed
    else:
        # Identity (S.T S / T + zI)^-1 S.T = S.T (S S.T / T + zI)^-1
        # so beta(z) = S.T @ U @ (q / (T (ev + z))) where ev, U = eigh(S S.T / T) and q = U.T y.
        eigenvalues, eigenvectors = np.linalg.eigh(S_train @ S_train.T / t_)
        means = y_train.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate(
            [(one / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in zs],
            axis=1,
        )
        tmp = eigenvectors.T @ S_train
        betas = tmp.T @ intermed

    predictions = S_test @ betas
    return betas, predictions


class RidgeEig:
    """Ridge with path-fit and val-R²-based shrinkage selection.

    Fits the full ridge path in one decomposition; picks the shrinkage that
    maximises OOS R² on the validation set. Without a val set, the smallest
    shrinkage in the grid is used.
    """

    name: str = "ridge"

    def __init__(self, shrinkage_grid: list[float]):
        if len(shrinkage_grid) == 0:
            raise ValueError("shrinkage_grid must be non-empty.")
        if any(z <= 0 for z in shrinkage_grid):
            raise ValueError("All shrinkages must be strictly positive.")
        self.shrinkage_grid: list[float] = list(shrinkage_grid)
        self._betas: np.ndarray | None = None
        self._best_shrinkage: float | None = None
        self._best_idx: int | None = None
        self._val_r2s: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> RidgeEig:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        # Fit the full ridge path once. Pass a tiny dummy as S_test (we ignore
        # the returned preds and matmul with X_val directly). Avoids
        # materialising a concat([X_train, X_val]) buffer that previously blew
        # up to ~36 GB float64 on real folds and caused 4x variance between
        # pods racing for the same memory.
        betas, _ = ridge_eig(X_train, y_train, X_train[:1], self.shrinkage_grid)
        self._betas = betas

        if X_val is not None and y_val is not None:
            val_preds = X_val @ betas
            r2s = np.array(
                [r_squared_oos(y_val, val_preds[:, k]) for k in range(len(self.shrinkage_grid))]
            )
            self._val_r2s = r2s
            best_idx = int(np.argmax(r2s))
        else:
            self._val_r2s = None
            best_idx = 0

        self._best_idx = best_idx
        self._best_shrinkage = self.shrinkage_grid[best_idx]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._betas is None or self._best_idx is None:
            raise RuntimeError("RidgeEig.predict called before fit.")
        return X @ self._betas[:, self._best_idx]

    @property
    def best_shrinkage(self) -> float:
        if self._best_shrinkage is None:
            raise RuntimeError("best_shrinkage queried before fit.")
        return self._best_shrinkage

    @property
    def beta(self) -> np.ndarray:
        if self._betas is None or self._best_idx is None:
            raise RuntimeError("beta queried before fit.")
        return self._betas[:, self._best_idx]


def ridgeless_min_norm(
    S_train: np.ndarray,
    y_train: np.ndarray,
    S_test: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Min-norm interpolator (z -> 0 limit of ridge), dispatched via tiny shrinkage."""
    betas, preds = ridge_eig(S_train, y_train, S_test, [float(eps)])
    return betas[:, 0], preds[:, 0]


class RidgelessModel:
    """BaseModel-compatible ridgeless / min-norm interpolator."""

    name: str = "ridgeless"

    def __init__(self, eps: float = 1e-10):
        if eps <= 0:
            raise ValueError("eps must be strictly positive.")
        self.eps: float = float(eps)
        self._beta: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,  # noqa: ARG002 - kept for protocol consistency
        y_val: np.ndarray | None = None,  # noqa: ARG002
    ) -> RidgelessModel:
        beta, _ = ridgeless_min_norm(X_train, y_train, X_train[:1], eps=self.eps)
        self._beta = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._beta is None:
            raise RuntimeError("RidgelessModel.predict called before fit.")
        return X @ self._beta
