from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel

from matchlab_core.schemas.run import ArtifactName

# Canonical file name per artifact. JSONL for per-frame streams (cheap to
# range-request later), JSON for whole-clip results.
ARTIFACT_FILES: dict[ArtifactName, str] = {
    ArtifactName.DETECTIONS: "detections.jsonl",
    ArtifactName.BALL: "ball.jsonl",
    ArtifactName.TRACKLETS: "tracklets.json",
    ArtifactName.TEAMS: "teams.json",
    ArtifactName.CALIBRATION: "calibration.jsonl",
    ArtifactName.CALIBRATION_RAW: "calibration_raw.jsonl",
    ArtifactName.CAMERA_MOTION: "camera_motion.jsonl",
    ArtifactName.PLAYERS: "players.json",
    ArtifactName.MINIMAP: "minimap.jsonl",
    ArtifactName.POSSESSION: "possession.json",
    ArtifactName.EVENTS: "events.json",
    ArtifactName.SPOTTING: "spotting.json",
    ArtifactName.STATS: "stats.json",
    ArtifactName.QA_ITEMS: "qa_items.json",
    ArtifactName.TIMELINE: "timeline.json",
    ArtifactName.ANNOTATED_VIDEO: "annotated.mp4",
    ArtifactName.CROPS: "crops",
    ArtifactName.EVAL: "eval.json",
    ArtifactName.ASSOCIATION: "association.json",
    ArtifactName.REID_EMBEDDINGS: "reid_embeddings.npz",
    ArtifactName.FRAME_FEATURES: "frame_features.npz",
    ArtifactName.NAMING: "naming.json",
    ArtifactName.REID_DETAIL: "reid_detail.json",
}


class ArtifactStore:
    """Reads/writes a run directory. Every artifact is a plain file so runs are
    portable, diffable, and directly servable to the Lab UI."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def path(self, name: ArtifactName) -> Path:
        return self.run_dir / ARTIFACT_FILES[name]

    def relpath(self, name: ArtifactName) -> str:
        return ARTIFACT_FILES[name]

    def exists(self, name: ArtifactName) -> bool:
        return self.path(name).exists()

    def write_jsonl(self, name: ArtifactName, rows: Iterable[BaseModel]) -> str:
        path = self.path(name)
        with open(path, "w") as f:
            for row in rows:
                f.write(row.model_dump_json() + "\n")
        return self.relpath(name)

    def read_jsonl(self, name: ArtifactName, model: type[BaseModel]) -> Iterator[BaseModel]:
        with open(self.path(name)) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield model.model_validate_json(line)

    def write_json(self, name: ArtifactName, data: BaseModel | list[BaseModel]) -> str:
        path = self.path(name)
        if isinstance(data, BaseModel):
            path.write_text(data.model_dump_json(indent=None))
        else:
            path.write_text(json.dumps([d.model_dump(mode="json") for d in data]))
        return self.relpath(name)

    def read_json_list(self, name: ArtifactName, model: type[BaseModel]) -> list[BaseModel]:
        raw = json.loads(self.path(name).read_text())
        return [model.model_validate(item) for item in raw]

    def read_json(self, name: ArtifactName, model: type[BaseModel]) -> BaseModel:
        return model.model_validate_json(self.path(name).read_text())

    def crops_dir(self) -> Path:
        d = self.path(ArtifactName.CROPS)
        d.mkdir(exist_ok=True)
        return d
