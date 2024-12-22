from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    experiment_name: str
    task: str
    seed: int
    dataset_path: str
    model_output_dir: str
    artifact_dir: str
    data: dict[str, Any]
    training: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig(**payload)
