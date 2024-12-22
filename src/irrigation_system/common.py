"""Shared utilities for the Intelligent Irrigation System project."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .evaluation import regression_metrics

__all__ = [
    "RunArtifacts",
    "ensure_dir",
    "regression_metrics",
    "save_json",
    "set_all_seeds",
    "timestamp_utc",
]


def set_all_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and TensorFlow reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def timestamp_utc() -> str:
    """Return a UTC timestamp string suitable for run IDs."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist, return Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write *payload* as pretty-printed JSON."""
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@dataclass
class RunArtifacts:
    """Paths produced by a single experiment run."""

    run_id: str
    model_path: Path
    predictions_path: Path
    metrics_path: Path
    final_log_path: Path
