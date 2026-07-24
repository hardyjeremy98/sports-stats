from __future__ import annotations

from pydantic import BaseModel, field_validator

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
    (external calibrator subprocess): a fresh, per-frame image→pitch-cm
    homography keyed by source-video `frame_idx`. `homography` is null when the
    external model could not calibrate that frame; the pipeline-side stage
    applies its own temporal smoothing over these fresh estimates, so this
    schema carries no smoothing state itself.

    The validator enforces only the *structural* contract: a present homography
    is exactly 3×3. Geometric usability (invertibility) and agreement with the
    manifest's frame set are checked by the bridge, not here — see
    matchlab_core.calib.bridge.
    """

    frame_idx: int
    homography: list[list[float]] | None = None
    confidence: float = 0.0
    n_points: int = 0

    @field_validator("homography")
    @classmethod
    def _homography_is_3x3(
        cls, value: list[list[float]] | None
    ) -> list[list[float]] | None:
        if value is None:
            return value
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("homography must be a 3×3 matrix when not null")
        return value
