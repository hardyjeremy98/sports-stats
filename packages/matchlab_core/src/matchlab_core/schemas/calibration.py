from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.schemas.geometry import Point


class FrameCalibration(BaseModel):
    """Image→pitch homography for one frame. One JSONL row in calibration.jsonl.

    `homography` maps image pixels to pitch centimeters (row-major 3x3).
    The keypoint model is known to be fragile on non-broadcast footage, so
    calibrators report per-frame confidence and whether the homography was
    carried/smoothed from neighboring frames rather than estimated fresh.
    """

    frame_idx: int
    t: float
    homography: list[list[float]] | None = None
    n_keypoints: int = 0
    keypoints_image: list[Point] = []
    keypoint_confidences: list[float] = []
    confidence: float = 0.0
    smoothed: bool = False


class ExternalHomography(BaseModel):
    """One record of the calibration exchange contract
    (docs/reference/external-calibrators-setup.md): a fresh, per-frame
    image→pitch-cm homography from an external calibrator subprocess, keyed by
    source-video `frame_idx`. `homography` is null when the external model
    could not calibrate that frame; the pipeline-side stage applies its own
    EMA/carry smoothing over these fresh estimates, so this schema carries no
    smoothing state itself.
    """

    frame_idx: int
    homography: list[list[float]] | None = None
    confidence: float = 0.0
    n_points: int = 0
