"""Re-scoring must produce artifacts a consumer can actually use (SPO-84).

Regression: the first cut of `gate2-resmooth` rebuilt FrameCalibration rows from
scratch with only the smoother's own outputs, silently dropping the reprojected
pitch keypoints the stage fills in. `calibration.jsonl` still validated and every
metric still computed, so nothing failed — but the Lab's "Pitch keypoints" overlay
went blank across all twelve re-scored Gate 2 runs, and with it the
fresh/smoothed/interpolated colour coding that rides on those dots.
"""

from __future__ import annotations

import numpy as np
from matchlab_core.pitch import get_pitch
from matchlab_core.schemas.calibration import FrameCalibration
from matchlab_train.experiments.gate2_resmooth import _calibration_rows


def _homography_for(width: int, height: int) -> list[list[float]]:
    """A plausible broadcast image->pitch-cm homography for the FIFA spec."""
    import cv2

    pitch = get_pitch("fifa")
    image_quad = np.array(
        [(0.28 * width, 0.30 * height), (0.72 * width, 0.30 * height),
         (0.97 * width, 0.94 * height), (0.03 * width, 0.94 * height)],
        dtype=np.float64,
    )
    pitch_quad = np.array(
        [(0.0, 0.0), (pitch.length, 0.0), (pitch.length, pitch.width), (0.0, pitch.width)],
        dtype=np.float64,
    )
    H, _ = cv2.findHomography(image_quad, pitch_quad, 0)
    return H.tolist()


def test_resmoothed_rows_carry_reprojected_pitch_keypoints() -> None:
    width, height = 1920, 1080
    H = _homography_for(width, height)
    raws = [{"frame_idx": i, "t": i / 25.0, "homography": H, "confidence": 0.9} for i in range(40)]

    rows = _calibration_rows(raws, frame_size=(width, height), pitch=get_pitch("fifa"))

    assert len(rows) == len(raws)
    # Every row must validate as the artifact schema the Lab reads.
    parsed = [FrameCalibration.model_validate(r) for r in rows]

    with_h = [c for c in parsed if c.homography is not None]
    assert with_h, "expected usable homographies"

    # The overlay draws keypoints_image; empty means a blank layer.
    assert all(c.n_keypoints > 0 for c in with_h)
    assert all(len(c.keypoints_image) == c.n_keypoints for c in with_h)
    assert all(len(c.keypoint_confidences) == c.n_keypoints for c in with_h)

    # Keypoints land in (or near) the frame — a meaningless cluster at the
    # vanishing line would still be "non-empty" but useless.
    margin = 0.05 * max(width, height)
    for c in with_h:
        for kp in c.keypoints_image:
            assert -margin <= kp.x <= width + margin
            assert -margin <= kp.y <= height + margin


def test_resmoothed_rows_without_homography_have_no_keypoints() -> None:
    width, height = 1920, 1080
    raws = [{"frame_idx": i, "t": i / 25.0, "homography": None, "confidence": 0.0} for i in range(20)]

    rows = _calibration_rows(raws, frame_size=(width, height), pitch=get_pitch("fifa"))
    parsed = [FrameCalibration.model_validate(r) for r in rows]

    assert all(c.homography is None for c in parsed)
    assert all(c.n_keypoints == 0 and c.keypoints_image == [] for c in parsed)
