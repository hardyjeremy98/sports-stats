from __future__ import annotations

from pydantic import BaseModel

from pitchlab_core.schemas.detections import DetectionClass
from pitchlab_core.schemas.geometry import Box


class TrackletFrame(BaseModel):
    frame_idx: int
    box: Box
    confidence: float


class Tracklet(BaseModel):
    """A short-term track from the tracker — the unit of identity in PitchLab.

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
