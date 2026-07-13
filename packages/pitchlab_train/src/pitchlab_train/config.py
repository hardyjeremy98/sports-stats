from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Where training data comes from.

    kind:
      roboflow   — a Roboflow Universe project (the roboflow/sports datasets);
      qa-labels  — labeled examples distilled from human QA decisions (the
                   proprietary in-domain stream, locked decision #2);
      local      — a directory already on disk (COCO/YOLO format).
    """

    kind: str = "roboflow"
    # roboflow: e.g. "roboflow-jvuqo/football-players-detection-3zvbc/12"
    reference: str = ""
    format: str = "coco"
    local_path: str | None = None


class ExperimentConfig(BaseModel):
    """One training/eval experiment. `task` selects the experiment class from
    the registry; `params` are validated by that class."""

    name: str
    task: str  # detector-rfdetr | eval-pipelines | (future: reid, jersey-ocr, spotting)
    description: str = ""
    dataset: DatasetConfig = DatasetConfig()
    params: dict = Field(default_factory=dict)
    output_dir: str = "data/experiments"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))
