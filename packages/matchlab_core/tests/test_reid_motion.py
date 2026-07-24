"""Camera-motion compensation for the re-ID engine's motion gate (SPO-55):
homography composition across frames, and sparse-optical-flow estimation
recovering a known synthetic pan."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.motion import CameraMotion, estimate_camera_motion
from matchlab_core.video import Frame


def _translation(dx: float, dy: float = 0.0) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def test_map_point_composes_homographies_forward_and_backward():
    # Sampled frames 10, 15, 20; each step shifts content left by 30 px.
    cm = CameraMotion(
        frame_idxs=[10, 15, 20],
        step_homographies=[_translation(-30.0), _translation(-30.0)],
    )
    np.testing.assert_allclose(cm.map_point((400.0, 100.0), 10, 20), (340.0, 100.0))
    np.testing.assert_allclose(cm.map_point((400.0, 100.0), 10, 15), (370.0, 100.0))
    # Backward mapping inverts the chain.
    np.testing.assert_allclose(cm.map_point((340.0, 100.0), 20, 10), (400.0, 100.0))
    # Same frame -> identity.
    np.testing.assert_allclose(cm.map_point((7.0, 8.0), 15, 15), (7.0, 8.0))


def test_map_point_without_motion_data_is_identity():
    cm = CameraMotion(frame_idxs=[], step_homographies=[])
    np.testing.assert_allclose(cm.map_point((5.0, 6.0), 0, 100), (5.0, 6.0))


def test_estimate_recovers_synthetic_pan():
    # A textured world panned right by 4 px/frame: scene content shifts LEFT,
    # so the recovered per-step homography is ~T(-4, 0).
    rng = np.random.default_rng(0)
    world = (rng.random((200, 400)) * 255).astype(np.uint8)
    import cv2

    world = cv2.GaussianBlur(world, (5, 5), 0)
    frames = []
    for i in range(4):
        crop = world[50:150, 40 + 4 * i : 240 + 4 * i]
        img = np.repeat(crop[:, :, None], 3, axis=2)
        frames.append(Frame(frame_idx=i * 2, t=i * 0.1, image=img))

    cm = estimate_camera_motion(frames, downscale=1)
    mapped = cm.map_point((100.0, 50.0), 0, 6)  # 3 steps of -4 px
    assert mapped[0] == pytest.approx(100.0 - 12.0, abs=1.0)
    assert mapped[1] == pytest.approx(50.0, abs=1.0)
