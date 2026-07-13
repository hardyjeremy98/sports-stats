from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

from pitchlab_train.config import ExperimentConfig


class Experiment(ABC):
    """One training or evaluation job. Subclasses validate config.params with
    their own pydantic model and implement run(). Results land in
    <output_dir>/<name>-<timestamp>/ with a result.json summary."""

    task_name: str

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def workdir(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        d = Path(self.config.output_dir) / f"{self.config.name}-{stamp}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_result(self, workdir: Path, result: dict) -> None:
        result = {
            "experiment": self.config.name,
            "task": self.task_name,
            "config": self.config.model_dump(mode="json"),
            **result,
        }
        (workdir / "result.json").write_text(json.dumps(result, indent=2))

    @abstractmethod
    def run(self) -> dict: ...
