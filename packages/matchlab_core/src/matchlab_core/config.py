from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from matchlab_core.schemas.run import StageKind


class StageConfig(BaseModel):
    """Selects an implementation for a stage slot and parameterizes it.

    `impl` is a registry key (see matchlab_core.registry). `params` are passed
    to the implementation's constructor — each implementation validates them
    against its own pydantic Params model.
    """

    impl: str
    params: dict = Field(default_factory=dict)
    enabled: bool = True


class VideoConfig(BaseModel):
    # Process every Nth frame. Detection quality allows subsampling; artifacts
    # always record source frame indices so overlays stay aligned.
    sample_stride: int = 1
    max_frames: int | None = None  # cap for quick Lab experiments
    resize_width: int | None = None


class PipelineConfig(BaseModel):
    """A named, versioned recipe for processing a clip. Stored as YAML in
    configs/; the Lab can create runs from any config and diff two of them."""

    name: str
    description: str = ""
    sport: str = "soccer"  # single hardcoded sport in v1; the seam for later
    video: VideoConfig = VideoConfig()
    stages: dict[StageKind, StageConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


def list_configs(config_dir: str | Path) -> list[PipelineConfig]:
    configs = []
    for path in sorted(Path(config_dir).glob("pipeline.*.yaml")):
        configs.append(PipelineConfig.from_yaml(path))
    return configs
