"""Logging and MLflow initialisation (idempotent)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["configure_logging", "init_mlflow"]


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _resolve_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return logging.INFO


def configure_logging(
    cfg: Any | None = None,
    *,
    level: Any | None = None,
    log_file: os.PathLike | None = None,
) -> None:
    """Configure the root logger from a Hydra config or explicit args."""
    if level is None and cfg is not None:
        level = getattr(getattr(cfg, "logging", None), "log_level", "INFO")
    level_int = _resolve_level(level if level is not None else "INFO")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, mode="a", encoding="utf-8"))

    # force=True so repeated calls reconfigure cleanly (Hydra re-runs main in tests).
    logging.basicConfig(
        level=level_int,
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
        handlers=handlers,
        force=True,
    )

    for noisy in ("urllib3", "matplotlib", "PIL", "transformers.tokenization_utils_base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def init_mlflow(cfg: Any) -> Any | None:
    """Initialise MLflow; returns the module or None if mlflow isn't installed."""
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        logger.warning("mlflow is not installed; experiment tracking disabled.")
        return None

    uri = getattr(getattr(cfg, "logging", None), "mlflow_uri", "file:./mlruns")
    experiment = getattr(getattr(cfg, "logging", None), "experiment_name", "mlfinance")

    mlflow.set_tracking_uri(uri)
    # 10-pod parallel training races on first-time experiment creation
    # against the shared `file:./mlruns` backend on the home PVC:
    # multiple `mkdir mlruns/<id>` + `meta.yaml` writes can collide.
    # After the first creation set_experiment is read-only and safe.
    # Wrap in a small retry/ignore loop so the train pod doesn't fail
    # just because another pod won the create race.
    try:
        mlflow.set_experiment(experiment)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MLflow set_experiment raced (%s); retrying once after a short delay.",
            exc,
        )
        import time

        time.sleep(1.0)
        try:
            mlflow.set_experiment(experiment)
        except Exception as exc2:  # noqa: BLE001
            logger.warning(
                "MLflow set_experiment failed twice (%s); proceeding without experiment tracking.",
                exc2,
            )
            return mlflow
    logger.info("MLflow tracking URI=%s, experiment=%s", uri, experiment)
    return mlflow
