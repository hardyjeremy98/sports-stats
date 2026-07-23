from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box


class TrackletFrame(BaseModel):
    frame_idx: int
    box: Box
    confidence: float
    # Where this box came from (SPO-15). Both shipped trackers (`botsort`,
    # `iou`) only ever emit matched detections today, so they set this to
    # "observed" for every frame explicitly -- the schema just makes that
    # fact visible instead of implicit. "predicted"/"interpolated" are
    # vocabulary for future smoothing/gap-filling stages (not implemented
    # yet -- see docs/prds/tracklet-modernization.md). Defaulted so
    # artifacts written by pre-SPO-15 code (no `source` key) still parse.
    source: Literal["observed", "predicted", "interpolated"] = "observed"


class Tracklet(BaseModel):
    """A short-term track from the tracker — the unit of identity in MatchLab.

    Identity, team, and association decisions are made per tracklet (aggregated
    over its frames), never per frame. Cross-tracklet association groups
    tracklets into PlayerEntity records in a later stage.
    """

    tracklet_id: int
    cls: DetectionClass = DetectionClass.PLAYER
    frames: list[TrackletFrame]

    @property
    def start_frame(self) -> int:
        return self.frames[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.frames[-1].frame_idx

    @property
    def mean_confidence(self) -> float:
        if not self.frames:
            return 0.0
        return sum(f.confidence for f in self.frames) / len(self.frames)
