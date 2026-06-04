"""Seed management for full pipeline determinism.

Pins ``random``, ``numpy.random``, ``torch.manual_seed`` (CPU + every CUDA
device), cuDNN flags, and ``PYTHONHASHSEED``. The CRSP panel is large enough
that a single non-deterministic kernel can move Sharpe by several bps.
"""

from __future__ import annotations

import os
import random

import numpy as np

# Torch is imported lazily so NumPy-only tests don't pay the cost.
try:
    import torch  # type: ignore[import-not-found]

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


__all__ = ["set_seed", "get_rng"]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Pin every known source of randomness to ``seed``.

    ``deterministic=True`` (default) enables PyTorch deterministic algorithms
    and disables cuDNN benchmarking. ``warn_only=True`` keeps the pipeline
    running on the handful of ops without a deterministic CUDA kernel.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy seed needed for sklearn/xgboost reproducibility

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (RuntimeError, TypeError):  # pragma: no cover - old torch
                torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Required by torch for fully deterministic matmul on Ampere+ GPUs.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_rng(seed: int) -> np.random.Generator:
    """Return a fresh, seeded ``numpy.random.Generator``. Prefer this over ``np.random.*``."""
    return np.random.default_rng(seed)


def get_torch_generator(seed: int, device: str | None = None):
    """Return a seeded ``torch.Generator`` on ``device`` (CPU if None)."""
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("PyTorch is not installed; cannot construct a torch.Generator.")
    g = torch.Generator(device=device or "cpu")
    g.manual_seed(seed)
    return g
