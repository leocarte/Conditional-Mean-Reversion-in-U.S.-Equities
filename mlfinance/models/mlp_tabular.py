"""Compact tabular MLP regressor for the nonlinear benchmark return-only deep model.

A dependency-light, ``BaseModel``-compatible estimator: dense layers with
ReLU/GELU, optional LayerNorm, dropout, AdamW (weight decay), gradient
clipping, and validation-MSE early stopping. Optional multi-seed averaging.

For speed on GPU the train/validation tensors are moved to the device **once**
and minibatches are formed by indexing a seeded permutation, so there is no
per-batch host->device copy and no DataLoader worker overhead (the model is
tiny and was otherwise data-loading bound). The shuffle permutation is drawn
from a CPU generator seeded per fit, so runs stay reproducible regardless of
device.

The estimator expects feature matrices that are *already* preprocessed
(winsorized, imputed, standardized, missing-flagged) by the shared
``LinearFeaturePreprocessor`` used across the benchmark path, so it performs
no internal scaling. The target is the raw next-month return, so predictions
are returned unclipped (only NaN/Inf are sanitized).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _build_network(
    input_dim: int,
    hidden_dims: list[int],
    *,
    activation: str,
    dropout: float,
    use_layer_norm: bool,
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must have at least one entry.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    if activation == "relu":
        act_layer: type[nn.Module] = nn.ReLU
    elif activation == "gelu":
        act_layer = nn.GELU
    else:
        raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}.")

    dims = [int(input_dim), *[int(h) for h in hidden_dims]]
    layers: list[nn.Module] = []
    for i in range(len(hidden_dims)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        layers.append(act_layer())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


class MLPRegressor:
    """Compact tabular MLP with AdamW, early stopping, and optional seed averaging.

    Parameters mirror the ``models.mlp`` block in ``configs/main.yaml``. When
    ``X_val``/``y_val`` are supplied to :meth:`fit`, training early-stops on
    validation MSE and the best-epoch weights are restored; otherwise the model
    trains for the full ``epochs`` budget (used for the train+validation refit,
    where ``epochs`` is set to the validated best-epoch count).
    """

    name: str = "mlp"

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        *,
        activation: str = "relu",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 4096,
        epochs: int = 30,
        patience: int = 5,
        grad_clip: float = 1.0,
        n_seeds: int = 1,
        seed: int = 1,
        device: str = "auto",
        use_bf16: bool = True,
        loss: str = "mse",
    ) -> None:
        self.hidden_dims: list[int] = (
            list(hidden_dims) if hidden_dims is not None else [256, 128, 64]
        )
        self.activation: str = str(activation)
        self.dropout: float = float(dropout)
        self.use_layer_norm: bool = bool(use_layer_norm)
        self.lr: float = float(lr)
        self.weight_decay: float = float(weight_decay)
        self.batch_size: int = int(batch_size)
        self.epochs: int = int(epochs)
        self.patience: int = int(patience)
        self.grad_clip: float = float(grad_clip)
        self.n_seeds: int = max(int(n_seeds), 1)
        self.seed: int = int(seed)
        self.device_str: str = str(device)
        self.use_bf16: bool = bool(use_bf16)
        # Training loss. Default "mse" preserves the nonlinear-benchmark / feature-audit
        # behaviour exactly (existing callers never pass `loss`). "huber" is an
        # additive option used by the walk-forward walk-forward search; early-stopping
        # selection still uses validation MSE for consistency across candidates.
        loss = str(loss).lower()
        if loss not in ("mse", "huber"):
            raise ValueError(f"loss must be 'mse' or 'huber', got {loss!r}.")
        self.loss: str = loss

        self._models: list[nn.Module] = []
        self.best_epoch_: int | None = None
        self.history_: list[dict[str, Any]] = []

    def _resolve_device(self) -> torch.device:
        if self.device_str in ("auto", "cuda"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_str)

    def _autocast_ctx(self, device: torch.device):
        if self.use_bf16 and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return _NullCtx()

    def _evaluate_mse(
        self, model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, device: torch.device
    ) -> float:
        """Mean squared error over ``x_val`` (already on ``device``), batched, NaN-safe."""
        model.eval()
        ctx = self._autocast_ctx(device)
        n = int(x_val.shape[0])
        sq_error = torch.zeros((), dtype=torch.float64, device=device)
        with torch.inference_mode():
            for start in range(0, n, self.batch_size):
                with ctx:
                    out = model(x_val[start : start + self.batch_size])
                out = torch.nan_to_num(out.float(), nan=0.0, posinf=0.0, neginf=0.0)
                target = y_val[start : start + self.batch_size].float()
                sq_error += ((out - target) ** 2).sum().double()
        return float(sq_error.item()) / max(n, 1)

    def _fit_single(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        val: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[nn.Module, dict[str, Any]]:
        torch.manual_seed(seed)
        model = _build_network(
            int(x_train.shape[1]),
            self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            use_layer_norm=self.use_layer_norm,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion: nn.Module = nn.HuberLoss() if self.loss == "huber" else nn.MSELoss()
        ctx = self._autocast_ctx(device)
        # CPU generator for the per-epoch shuffle keeps runs reproducible on any device.
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(seed)
        n = int(x_train.shape[0])

        best_val = math.inf
        best_epoch = -1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_since_improve = 0
        epochs_run = 0

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n, generator=shuffle_gen).to(device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                with ctx:
                    loss = criterion(model(x_train[idx]), y_train[idx])
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
            epochs_run = epoch + 1

            if val is not None:
                val_mse = self._evaluate_mse(model, val[0], val[1], device)
                if val_mse < best_val:
                    best_val = val_mse
                    best_epoch = epoch
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.patience:
                        break

        if val is not None:
            model.load_state_dict(best_state)
            n_epochs_for_refit = best_epoch + 1
        else:
            n_epochs_for_refit = epochs_run

        info = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_mse": (None if math.isinf(best_val) else best_val),
            "n_epochs_for_refit": max(int(n_epochs_for_refit), 1),
            "epochs_run": epochs_run,
        }
        return model, info

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> MLPRegressor:
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        device = self._resolve_device()
        # Move the whole training (and validation) matrices to the device once.
        x_train = torch.as_tensor(np.asarray(X_train, dtype=np.float32)).to(device)
        y_train_t = torch.as_tensor(np.asarray(y_train, dtype=np.float32)).reshape(-1, 1).to(device)
        val: tuple[torch.Tensor, torch.Tensor] | None = None
        if X_val is not None and y_val is not None:
            val = (
                torch.as_tensor(np.asarray(X_val, dtype=np.float32)).to(device),
                torch.as_tensor(np.asarray(y_val, dtype=np.float32)).reshape(-1, 1).to(device),
            )

        self._models = []
        self.history_ = []
        refit_epochs: list[int] = []
        for s in range(self.n_seeds):
            model, info = self._fit_single(
                x_train, y_train_t, val, seed=self.seed + s, device=device
            )
            self._models.append(model)
            self.history_.append(info)
            refit_epochs.append(int(info["n_epochs_for_refit"]))

        self.best_epoch_ = max(refit_epochs) if refit_epochs else self.epochs
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("MLPRegressor.predict called before fit.")
        device = self._resolve_device()
        x = torch.as_tensor(np.asarray(X, dtype=np.float32))
        if x.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        x = x.to(device)

        ctx = self._autocast_ctx(device)
        seed_means = np.zeros((x.shape[0],), dtype=np.float64)
        for model in self._models:
            model.to(device)
            model.eval()
            chunks: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, x.shape[0], self.batch_size):
                    with ctx:
                        out = model(x[start : start + self.batch_size])
                    chunks.append(out.detach().float().cpu().numpy().reshape(-1))
            seed_means += np.concatenate(chunks).astype(np.float64)
        preds = seed_means / float(len(self._models))
        return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


class _NullCtx:
    """No-op context manager so the train/eval loops stay branch-free."""

    def __enter__(self) -> _NullCtx:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
