"""Camera-motion compensation for the motion-feasibility gate.

The engine gates merges on how far a player would have had to travel between
one tracklet's end and another's start — but on panning footage, raw pixel
distance conflates player motion with camera motion. `estimate_camera_motion`
recovers per-step homographies from sparse optical flow (cheap, the same class
of GMC BoT-SORT uses); `CameraMotion` composes them so any image point can be
mapped from one sampled frame's camera to another's. Estimation failures
degrade to identity steps — compensation is opportunistic, never a dependency.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from matchlab_core.video import Frame


class CameraMotion:
    """Composable inter-frame homographies over the run's sampled frames.

    `step_homographies[i]` maps image coordinates of `frame_idxs[i]` to
    `frame_idxs[i+1]`. Points map across arbitrary frame spans by composing
    the chain (inverted for backward spans); frames outside the recorded
    range clamp to the nearest endpoint, and an empty CameraMotion is the
    identity — uncompensated behavior, never a crash.
    """

    def __init__(self, frame_idxs: list[int], step_homographies: list[np.ndarray]):
        if frame_idxs and len(step_homographies) != len(frame_idxs) - 1:
            raise ValueError(
                f"{len(frame_idxs)} frames need {len(frame_idxs) - 1} step "
                f"homographies, got {len(step_homographies)}"
            )
        self.frame_idxs = list(frame_idxs)
        self.steps = [np.asarray(h, dtype=np.float64) for h in step_homographies]

    def _pos(self, frame_idx: int) -> int:
        """Index of the nearest recorded frame at or below `frame_idx`
        (clamped into range)."""
        i = int(np.searchsorted(self.frame_idxs, frame_idx, side="right")) - 1
        return max(0, min(i, len(self.frame_idxs) - 1))

    def homography(self, from_frame: int, to_frame: int) -> np.ndarray:
        """Composed homography mapping `from_frame` image coords to
        `to_frame` image coords."""
        h = np.eye(3)
        if not self.frame_idxs:
            return h
        a, b = self._pos(from_frame), self._pos(to_frame)
        if a <= b:
            for i in range(a, b):
                h = self.steps[i] @ h
        else:
            for i in range(b, a):
                h = self.steps[i] @ h
            h = np.linalg.inv(h)
        return h

    def map_point(
        self, pt: tuple[float, float], from_frame: int, to_frame: int
    ) -> tuple[float, float]:
        h = self.homography(from_frame, to_frame)
        v = h @ np.array([pt[0], pt[1], 1.0])
        return (float(v[0] / v[2]), float(v[1] / v[2]))


def estimate_camera_motion(
    frames: Iterable[Frame],
    *,
    downscale: int = 2,
    max_corners: int = 400,
) -> CameraMotion:
    """Sparse-optical-flow GMC over consecutive sampled frames: Shi–Tomasi
    corners on the previous frame, pyramidal LK flow to the current one,
    RANSAC homography on the matches. Any step where estimation fails (too
    few corners/inliers) becomes an identity step."""
    frame_idxs: list[int] = []
    steps: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None
    scale = max(1, int(downscale))

    for fr in frames:
        gray = cv2.cvtColor(fr.image, cv2.COLOR_BGR2GRAY)
        if scale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale))
        if prev_gray is not None:
            h = _step_homography(prev_gray, gray, max_corners)
            if h is not None and scale > 1:
                s = np.diag([scale, scale, 1.0])
                h = s @ h @ np.linalg.inv(s)
            steps.append(np.eye(3) if h is None else h)
        frame_idxs.append(fr.frame_idx)
        prev_gray = gray
    return CameraMotion(frame_idxs=frame_idxs, step_homographies=steps)


def _step_homography(
    prev_gray: np.ndarray, gray: np.ndarray, max_corners: int
) -> np.ndarray | None:
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=7
    )
    if pts is None or len(pts) < 8:
        return None
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    ok = status.reshape(-1).astype(bool)
    if ok.sum() < 8:
        return None
    h, _mask = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 3.0)
    return h
