from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from matchlab_core.schemas.geometry import Box, Point


class DetectionClass(StrEnum):
    # Class set of the roboflow/sports player-detection model.
    BALL = "ball"
    GOALKEEPER = "goalkeeper"
    PLAYER = "player"
    REFEREE = "referee"


class Detection(BaseModel):
    box: Box
    confidence: float
    cls: DetectionClass


class FrameDetections(BaseModel):
    """All detections for one processed frame. One JSONL row in detections.jsonl."""

    frame_idx: int
    t: float  # seconds into the video
    detections: list[Detection]


class BallObservation(BaseModel):
    """The single resolved ball position for a frame (2D-only in v1).

    One JSONL row in ball.jsonl. `interpolated` marks gap-filled frames where the
    ball was not detected; those carry reduced confidence downstream.
    """

    frame_idx: int
    t: float
    xy: Point
    confidence: float
    interpolated: bool = False
