"""Offline global smoother for per-frame image->pitch homography trajectories.

A pure function (no I/O, no video decoding) that turns a sequence of raw
per-frame homography estimates — with gaps and gross outliers — into a smoothed,
gap-interpolated trajectory carrying per-frame provenance.

Design: **smooth in the point-correspondence parameterization, never blend 3x3
matrices elementwise.** Each frame's homography H maps image pixels -> pitch
centimetres. We represent H by the *image-space* positions of a small set of
fixed pitch anchor points, obtained by projecting those anchors through H^-1.
Those per-anchor image trajectories are what we smooth / interpolate over time;
each output homography is then refit (DLT via ``cv2.findHomography``) from the
smoothed image<->pitch correspondences. This keeps every intermediate a valid
projective transform and avoids the artefacts of averaging matrix entries.

Units and index space
----------------------
The function operates in the **index space of the provided sequence** (position
0..N-1), and the smoothing / outlier windows are measured in *sequence
positions* (samples). **Gaps, however, are measured in FRAME units**: the span
between two bracketing anchors is ``frame_idx[right] - frame_idx[left]`` and is
compared against ``max_gap_frames``. This is what makes strided ``frame_idx``
values (0, 2, 4, ...) behave correctly — a small number of skipped samples that
nonetheless spans many source frames is still a large gap.

Provenance
----------
Every input position produces exactly one :class:`SmoothedFrame`:

* ``FRESH`` — a usable raw estimate was present and accepted; the output is the
  smoothed refit anchored on it.
* ``SMOOTHED`` — a raw estimate was present but rejected as an outlier; the
  output is reconstructed from neighbouring anchors.
* ``INTERPOLATED`` — no raw estimate; the frame sits inside a gap <=
  ``max_gap_frames`` and is filled linearly from both endpoints.
* ``ABSENT`` — no output homography (gap > cap, or only one/zero surrounding
  anchor — leading/trailing gaps are NOT extrapolated).

Interpolation always requires anchors on *both* sides; single-sided
extrapolation is deliberately not performed. A singular (non-invertible) raw
homography is treated exactly as a missing estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

__all__ = [
    "RawEstimate",
    "SmoothStatus",
    "SmoothedFrame",
    "smooth_homography_trajectory",
]


@dataclass(frozen=True)
class RawEstimate:
    """A raw per-frame estimate. ``homography`` maps image px -> pitch cm
    (row-major 3x3 nested lists), or ``None`` when the frame could not be
    calibrated."""

    frame_idx: int
    homography: list[list[float]] | None = None
    confidence: float = 0.0


class SmoothStatus(str, Enum):
    """Per-frame provenance of a smoothed output homography."""

    FRESH = "fresh"
    SMOOTHED = "smoothed"
    INTERPOLATED = "interpolated"
    ABSENT = "absent"


@dataclass(frozen=True)
class SmoothedFrame:
    """One output frame: the smoothed homography (or ``None`` if ABSENT), its
    provenance status, and a confidence carried/interpolated from the inputs."""

    frame_idx: int
    homography: list[list[float]] | None
    status: SmoothStatus
    confidence: float


# Below this |det| the raw homography is treated as singular (== missing).
_MIN_ABS_DET = 1e-9
# Below this the projective denominator is degenerate (point at/behind horizon).
_MIN_ABS_W = 1e-9


def _anchor_image_points(
    homography: list[list[float]] | None, pitch: np.ndarray
) -> np.ndarray | None:
    """Project pitch anchors through H^-1 into image space, or ``None`` if the
    matrix is missing / singular / sends an anchor to the horizon."""
    if homography is None:
        return None
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return None
    det = float(np.linalg.det(H))
    if not np.isfinite(det) or abs(det) < _MIN_ABS_DET:
        return None
    Hinv = np.linalg.inv(H)
    homog = np.hstack([pitch, np.ones((pitch.shape[0], 1))])  # (N, 3)
    proj = homog @ Hinv.T  # (N, 3)
    w = proj[:, 2]
    if not np.all(np.isfinite(proj)) or np.any(np.abs(w) < _MIN_ABS_W):
        return None
    return proj[:, :2] / w[:, None]


def _refit_homography(
    image_points: np.ndarray, pitch: np.ndarray
) -> list[list[float]] | None:
    """Refit an image->pitch homography from smoothed correspondences (DLT)."""
    if not np.all(np.isfinite(image_points)):
        return None
    H, _ = cv2.findHomography(
        image_points.astype(np.float64), pitch.astype(np.float64), 0
    )
    if H is None or not np.all(np.isfinite(H)):
        return None
    return H.tolist()


def smooth_homography_trajectory(
    estimates: Sequence[RawEstimate],
    *,
    pitch_points: Sequence[tuple[float, float]],
    max_gap_frames: int = 150,
    outlier_threshold_px: float = 50.0,
    smoothing_window: int = 9,
) -> list[SmoothedFrame]:
    """Smooth a raw homography trajectory. See the module docstring for the full
    contract (parameterization, gap/outlier semantics, units)."""
    n = len(estimates)
    if n == 0:
        return []

    pitch = np.asarray(pitch_points, dtype=np.float64)
    half = max(smoothing_window // 2, 0)

    frame_ids = [e.frame_idx for e in estimates]
    confidences = [float(e.confidence) for e in estimates]

    # 1. Project each usable raw H into image-space anchor points.
    raw_points: list[np.ndarray | None] = [
        _anchor_image_points(e.homography, pitch) for e in estimates
    ]
    has_raw = [p is not None for p in raw_points]

    # 2. Outlier rejection: deviation from a robust (median) local window.
    outlier = [False] * n
    for i in range(n):
        if not has_raw[i]:
            continue
        window = [
            raw_points[j]
            for j in range(max(0, i - half), min(n, i + half + 1))
            if has_raw[j]
        ]
        median = np.median(np.stack(window, axis=0), axis=0)  # (N, 2)
        deviation = float(np.max(np.linalg.norm(raw_points[i] - median, axis=1)))
        if deviation > outlier_threshold_px:
            outlier[i] = True

    accepted = [has_raw[i] and not outlier[i] for i in range(n)]

    # 3. Smooth accepted anchor trajectories with a centred window measured in
    #    sequence positions (so large gaps do not bleed across the boundary).
    smoothed_points: list[np.ndarray | None] = [None] * n
    for i in range(n):
        if not accepted[i]:
            continue
        window = [
            raw_points[j]
            for j in range(max(0, i - half), min(n, i + half + 1))
            if accepted[j]
        ]
        smoothed_points[i] = np.mean(np.stack(window, axis=0), axis=0)

    accepted_indices = [i for i in range(n) if accepted[i]]

    # 4. Emit one output frame per input position.
    out: list[SmoothedFrame] = []
    for i in range(n):
        if accepted[i]:
            H = _refit_homography(smoothed_points[i], pitch)
            if H is not None:
                out.append(SmoothedFrame(frame_ids[i], H, SmoothStatus.FRESH, confidences[i]))
                continue
            # Degenerate refit: fall through to gap-fill / absent handling.

        left = _nearest(accepted_indices, i, before=True)
        right = _nearest(accepted_indices, i, before=False)
        status = SmoothStatus.SMOOTHED if has_raw[i] else SmoothStatus.INTERPOLATED

        if left is None or right is None:
            out.append(SmoothedFrame(frame_ids[i], None, SmoothStatus.ABSENT, 0.0))
            continue

        span = frame_ids[right] - frame_ids[left]
        if span > max_gap_frames or span <= 0:
            out.append(SmoothedFrame(frame_ids[i], None, SmoothStatus.ABSENT, 0.0))
            continue

        alpha = (frame_ids[i] - frame_ids[left]) / span
        points = (1.0 - alpha) * smoothed_points[left] + alpha * smoothed_points[right]
        conf = (1.0 - alpha) * confidences[left] + alpha * confidences[right]
        H = _refit_homography(points, pitch)
        if H is None:
            out.append(SmoothedFrame(frame_ids[i], None, SmoothStatus.ABSENT, 0.0))
        else:
            out.append(SmoothedFrame(frame_ids[i], H, status, conf))

    return out


def _nearest(sorted_indices: list[int], i: int, *, before: bool) -> int | None:
    """Nearest accepted sequence position strictly before/after ``i``."""
    if before:
        candidates = [j for j in sorted_indices if j < i]
        return candidates[-1] if candidates else None
    candidates = [j for j in sorted_indices if j > i]
    return candidates[0] if candidates else None
