"""Lasso wrapper using sklearn.lasso_path for warm-started alpha-grid sweep."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

from mlfinance.models.base import r_squared_oos


class LassoModel:
    """Lasso with built-in alpha selection via `lasso_path` (warm-started path).

    Standardises features internally (StandardScaler fit on X_train only) so
    coordinate descent converges in well under ``max_iter``. ``fit_intercept``
    defaults to False because the standardised features have mean 0 and the
    rank target has mean 0.

    Internally uses ``sklearn.linear_model.lasso_path``, which solves the
    entire regularization path in one call with automatic warm-start across
    descending alphas. Compared to the previous per-alpha `Lasso(...).fit(...)`
    loop this gives ~10-30x speedup on wide panels (the per-alpha cold-start
    pays full coordinate-descent convergence cost for each alpha).

    Other speed knobs:
      * ``tol=1e-4`` (sklearn default; was 1e-6 = 100x stricter, no economic gain)
      * ``selection='random'`` instead of cyclic (~2x faster on wide panels)
      * ``precompute=True`` (1290x1290 Gram is 13 MB, accelerates every CD pass)
      * ``copy_X=False`` (we already own the scaled copy)
    """

    name: str = "lasso"

    def __init__(
        self,
        alpha_grid: list[float],
        max_iter: int = 10_000,
        tol: float = 1e-4,
        fit_intercept: bool = False,
    ):
        if len(alpha_grid) == 0:
            raise ValueError("alpha_grid must be non-empty.")
        if any(a <= 0 for a in alpha_grid):
            raise ValueError("All alphas must be strictly positive.")
        self.alpha_grid: list[float] = list(alpha_grid)
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)
        self.fit_intercept: bool = bool(fit_intercept)
        self._best_alpha: float | None = None
        self._best_coef: np.ndarray | None = None
        self._val_r2s: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> LassoModel:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        # Standardize on training data only. Keep float32: sklearn's
        # lasso_path accepts float32 natively (>= 1.4) and will only upcast
        # internally if its C-level CD kernel requires float64, allocating
        # the f64 copy at most ONCE inside lasso_path. The legacy
        # `X_train.astype(np.float64, copy=False)` above created a 34 GB
        # float64 buffer here (NOT a no-op because X_train is float32),
        # which combined with StandardScaler's own copy and the lasso_path
        # internal upcast pushed peak RSS to ~85 GB on the 80 GB pod.
        scaler = StandardScaler(copy=False)
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32, copy=False)
        self._scaler = scaler

        # lasso_path solves the full path with internal warm-start. Sort alphas
        # descending so the warm-start (large alpha -> small alpha) is correct;
        # large alpha gives sparse beta near zero, then each successive solve
        # only needs to add a few non-zeros.
        alphas_sorted = sorted(self.alpha_grid, reverse=True)
        y_train_f32 = y_train.astype(np.float32, copy=False)
        # selection='random' converges faster but is non-deterministic w.r.t.
        # the global numpy RNG state (which the outer set_seed() leaves in an
        # unpredictable state). Pass random_state=0 explicitly so the path is
        # bit-reproducible regardless of the outer seed. This is REQUIRED
        # for the train.py "skip seeds for deterministic models" optimization
        # to produce bit-identical preds across seeds.
        _, coefs_path, _ = lasso_path(
            X_train_scaled,
            y_train_f32,
            alphas=alphas_sorted,
            max_iter=self.max_iter,
            tol=self.tol,
            selection="random",
            random_state=0,
            precompute=True,
            copy_X=False,
        )
        # coefs_path shape: (n_features, n_alphas), columns in alphas_sorted order

        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(np.asarray(X_val, dtype=np.float32)).astype(
                np.float32, copy=False
            )
            val_preds = X_val_scaled @ coefs_path  # (n_val, n_alphas)
            r2s = np.array(
                [r_squared_oos(y_val, val_preds[:, k]) for k in range(len(alphas_sorted))],
                dtype=np.float64,
            )
            best_idx = int(np.argmax(r2s))
            self._val_r2s = r2s
        else:
            best_idx = len(alphas_sorted) - 1  # smallest alpha (last in descending sort)
            self._val_r2s = None

        self._best_alpha = alphas_sorted[best_idx]
        self._best_coef = coefs_path[:, best_idx].copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._best_coef is None or self._scaler is None:
            raise RuntimeError("LassoModel.predict called before fit.")
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float32)).astype(
            np.float32, copy=False
        )
        return X_scaled @ self._best_coef

    @property
    def best_alpha(self) -> float:
        if self._best_alpha is None:
            raise RuntimeError("best_alpha queried before fit.")
        return self._best_alpha

    @property
    def coef_(self) -> np.ndarray:
        if self._best_coef is None:
            raise RuntimeError("coef_ queried before fit.")
        return self._best_coef.copy()
