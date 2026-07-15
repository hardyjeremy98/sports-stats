"""API response/request models — the contract the web UI is built against."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class VideoOut(BaseModel):
    id: int
    filename: str
    size_bytes: int
    duration_s: float
    fps: float
    width: int
    height: int
    has_ground_truth: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class ConfigOut(BaseModel):
    name: str
    description: str
    sport: str
    yaml: str
    stages: dict  # stage kind -> {impl, params, enabled}


class RegistryOut(BaseModel):
    # stage kind -> available implementation names
    stages: dict[str, list[str]]


class RunCreate(BaseModel):
    video_id: int
    # Either a named config from configs/ ...
    config_name: str | None = None
    # ... or ad-hoc YAML from the Lab's config editor.
    config_yaml: str | None = None
    label: str | None = None


class RunOut(BaseModel):
    id: str
    video_id: int
    config_name: str
    label: str | None
    status: str
    error: str | None
    progress_stage: str | None
    progress_frac: float
    progress_msg: str | None
    metrics: dict | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class RunDetailOut(RunOut):
    config_yaml: str
    manifest: dict | None = None  # RunManifest when the run dir has one
    video: VideoOut | None = None


class RunDiffOut(BaseModel):
    """Everything the Lab's diff view needs in one payload."""

    run_a: RunDetailOut
    run_b: RunDetailOut
    config_changes: list[dict]  # [{path, a, b}]
    metric_deltas: dict[str, dict]  # key -> {a, b, delta}
    stats_a: dict | None
    stats_b: dict | None
    timeline_a: list | None
    timeline_b: list | None
    eval_a: dict | None = None
    eval_b: dict | None = None
    switch_diff: dict | None = None  # fixed/introduced/persisted ID-switch instances


class QAOut(BaseModel):
    id: int
    run_id: str
    qa_id: int
    payload: dict
    status: str
    corrected_player_id: int | None
    corrected_event_type: str | None
    note: str | None
    decided_at: datetime | None

    model_config = {"from_attributes": True}


class QACorrect(BaseModel):
    player_id: int | None = None
    event_type: str | None = None
    note: str | None = None
