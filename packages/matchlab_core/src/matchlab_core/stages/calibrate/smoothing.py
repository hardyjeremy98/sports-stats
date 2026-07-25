"""Shared temporal smoothing for pitch calibrators.

Every keypoint-based calibrator produces a *fresh* per-frame image→pitch
homography that visibly flickers (the underlying models jitter, and weak frames
— pans past featureless midfield, heavy occlusion — fail outright). This module
turns a sequence of fresh estimates into the stabilized `FrameCalibration`
stream downstream consumes: EMA-blend each fresh homography with the previous
smoothed one, and on a failed frame carry the last good homography for up to
`max_carry_frames` frames with decaying confidence before giving up (null).

`yolo-pitch-local` is this module's stage consumer (`roboflow-keypoints` has its
own inline EMA blend instead). `pnlcalib` does not use this online smoother — it
applies the offline whole-clip smoother in `matchlab_core.calib.smoother`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from matchlab_core.schemas import FrameCalibration
from matchlab_core.schemas.geometry import Point


@dataclass
class FreshCalibration:
    """One frame's fresh (pre-smoothing) calibration estimate. `homography` is
    None when the model could not calibrate this frame. The keypoint fields are
    carried through verbatim to the emitted FrameCalibration (for the Lab's
    failure browser) regardless of whether the fresh homography succeeded."""

    frame_idx: int
    t: float
    homography: np.ndarray | None
    confidence: float
    n_keypoints: int = 0
    keypoints_image: list[Point] = field(default_factory=list)
    keypoint_confidences: list[float] = field(default_factory=list)


def _blend(prev: np.ndarray, fresh: np.ndarray, alpha: float) -> np.ndarray:
    a = prev / prev[2, 2]
    b = fresh / fresh[2, 2]
    return alpha * a + (1 - alpha) * b


def smooth_calibrations(
    records: Iterable[FreshCalibration],
    *,
    ema_alpha: float,
    max_carry_frames: int,
    carry_decay: float,
) -> list[FrameCalibration]:
    """Apply EMA blending + last-good carry over fresh per-frame estimates."""
    out: list[FrameCalibration] = []
    smoothed_h: np.ndarray | None = None
    carry = 0
    carried_conf = 0.0

    for rec in records:
        if rec.homography is not None:
            smoothed_h = (
                rec.homography
                if smoothed_h is None
                else _blend(smoothed_h, rec.homography, ema_alpha)
            )
            carry = 0
            carried_conf = rec.confidence
            out.append(
                FrameCalibration(
                    frame_idx=rec.frame_idx,
                    t=rec.t,
                    homography=smoothed_h.tolist(),
                    n_keypoints=rec.n_keypoints,
                    keypoints_image=rec.keypoints_image,
                    keypoint_confidences=rec.keypoint_confidences,
                    confidence=round(rec.confidence, 4),
                    smoothed=False,
                )
            )
        elif smoothed_h is not None and carry < max_carry_frames:
            carry += 1
            carried_conf *= carry_decay
            out.append(
                FrameCalibration(
                    frame_idx=rec.frame_idx,
                    t=rec.t,
                    homography=smoothed_h.tolist(),
                    n_keypoints=rec.n_keypoints,
                    keypoints_image=rec.keypoints_image,
                    keypoint_confidences=rec.keypoint_confidences,
                    confidence=round(carried_conf, 4),
                    smoothed=True,
                )
            )
        else:
            out.append(
                FrameCalibration(
                    frame_idx=rec.frame_idx,
                    t=rec.t,
                    homography=None,
                    n_keypoints=rec.n_keypoints,
                    keypoints_image=rec.keypoints_image,
                    keypoint_confidences=rec.keypoint_confidences,
                    confidence=0.0,
                    smoothed=False,
                )
            )
    return out
