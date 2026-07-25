"""Offline global smoother for per-frame image->pitch homography trajectories.

A pure function (no I/O, no video decoding) that turns a sequence of raw
per-frame homography estimates — with gaps and gross outliers — into a smoothed,
gap-interpolated trajectory carrying per-frame provenance.

Parameterization: visible-pitch grid (v2)
------------------------------------------
We smooth in a **point-correspondence** parameterization, never by blending 3x3
matrices elementwise. Each frame's homography ``H`` maps image pixels -> pitch
centimetres. We represent ``H`` by projecting a small, fixed **image-space grid**
(a 3x3 lattice inset 10% from the edges of the caller-provided ``frame_size``)
*forward* through ``H`` into pitch cm. Those per-grid-point pitch trajectories
are what we smooth / interpolate over time; each output homography is refit (DLT
via ``cv2.findHomography``) from the fixed image grid <-> smoothed pitch grid.

Why the grid is in image space (this is the v1 fix). v1 did the opposite: it
projected fixed *pitch* anchors (field corners / centre circle) *backward* through
``H^-1`` into image space and smoothed those. On zoomed broadcast views those pitch
anchors lie far OUTSIDE the visible frame, so ``H^-1`` extrapolates them to wild,
noise-amplified image positions (measured ±200-400 px on SNMOT-123 while genuine
pans move by 55-90 px/frame). A static positional-median rejection with a pixel
threshold then discarded 44-93% of *correct* frames and the sparse survivors
lagged/snapped — the visually-bad minimap. Grid points are always inside the frame,
so their forward projection samples only *actually-visible* pitch and never
extrapolates behind the horizon.

Outlier rejection: motion-compensated (this is the other half of the v1 fix)
---------------------------------------------------------------------------
A real pan moves every grid point coherently frame-to-frame; that is **signal**,
not an outlier, and must be accepted. So a frame is judged not against a static
window median (which treats any pan as deviation) but against a **local robust
constant-velocity motion model**: the per-grid-point velocity is the *median* of
one-step velocities in the window, and the predicted position is the *median* over
window neighbours of ``P[j] + v*(t-j)``. Taking medians makes both the velocity and
the anchor robust to a single flipped/garbage frame inside the window. A frame is
rejected only when the median (over grid points) residual from that model exceeds
``outlier_threshold_cm``. On SNMOT-123 the genuine fast-pan frames peak at ~1700 cm
residual while the lone gross homography flip (frame 71) sits at ~40000 cm; the
default 2500 cm threshold lives in that wide empty gap — well above real pan/camera
noise (~25 m of visible pitch, a quarter of the field) yet far below a flip.

Window aggregation: per-grid-point median (this is the v3 fix)
--------------------------------------------------------------
Rejection is per *frame* and cannot be perfect: a homography whose projective row
is slightly wrong sends its horizon-ward grid points thousands of centimetres away
while barely moving the near-field ones, so the median-over-grid residual stays
small and the frame is accepted. Aggregating such a window with an arithmetic
**mean** lets that one survivor drag every grid point — measured at 18.9 m for a
player-height projection on SNMOT-122 frame 440 — and the DLT refit then spreads
the error across the whole frame. Taking the **median** per grid point outvotes it.

This is not free. The median selects independently per coordinate, so it can mix
source frames and produce a grid that is not the projection of any single
homography (DLT self-residual ~65 cm against the mean's ~2 cm), which costs
variance on clean pans: ~1.8 m of smoothing error on a synthetic fast pan against
the mean's ~1.1 m. That is the deliberate trade — see the v3 design spec. On the
twelve real Gate 2 sequences it is a net win on every clip, clean ones included.

Units and index space
---------------------
The function operates in the **index space of the provided sequence** (position
0..N-1); the smoothing and motion-model windows are measured in *sequence
positions* (samples). **Gaps, however, are measured in FRAME units**: the span
between two bracketing anchors is ``frame_idx[right] - frame_idx[left]`` and is
compared against ``max_gap_frames``. This is what makes strided ``frame_idx``
values (0, 2, 4, ...) behave correctly — a small number of skipped samples that
nonetheless spans many source frames is still a large gap.

The default ``smoothing_window`` is 15 samples. It was 9 through v2; widening it
was measured, not assumed — on the twelve Gate 2 sequences it takes the worst
clip's windowed implausible-speed rate from 3.47% to 2.49% with coverage
unchanged at 1.000.

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
homography, or one that sends a grid point to / behind the horizon, is treated
exactly as a missing estimate.
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
# Below this the projective denominator is degenerate (grid point at/behind horizon).
_MIN_ABS_W = 1e-9
# Fraction the image grid is inset from each frame edge (keeps grid points off the
# extreme margins where a strong perspective view may cross the horizon).
_GRID_INSET = 0.1


def _image_grid(frame_size: tuple[int, int]) -> np.ndarray:
    """A fixed 3x3 image-space lattice inset ``_GRID_INSET`` from each edge of the
    frame, returned as ``(9, 2)`` float pixel coordinates. Well-spread across the
    frame so the DLT refit of each smoothed homography is well-conditioned."""
    w, h = float(frame_size[0]), float(frame_size[1])
    xs = [_GRID_INSET * w, 0.5 * w, (1.0 - _GRID_INSET) * w]
    ys = [_GRID_INSET * h, 0.5 * h, (1.0 - _GRID_INSET) * h]
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def _grid_pitch_points(
    homography: list[list[float]] | None, grid: np.ndarray
) -> np.ndarray | None:
    """Project the fixed image grid *forward* through ``H`` into pitch cm, or
    ``None`` if the matrix is missing / non-finite / sends a grid point to the
    horizon (degenerate projective denominator)."""
    if homography is None:
        return None
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return None
    det = float(np.linalg.det(H))
    if not np.isfinite(det) or abs(det) < _MIN_ABS_DET:
        return None
    homog = np.hstack([grid, np.ones((grid.shape[0], 1))])  # (N, 3)
    proj = homog @ H.T  # (N, 3)
    w = proj[:, 2]
    if not np.all(np.isfinite(proj)) or np.any(np.abs(w) < _MIN_ABS_W):
        return None
    return proj[:, :2] / w[:, None]


def _refit_homography(
    grid: np.ndarray, pitch_points: np.ndarray
) -> list[list[float]] | None:
    """Refit an image->pitch homography from the fixed image grid and the
    (smoothed) pitch-space grid points (DLT)."""
    if not np.all(np.isfinite(pitch_points)):
        return None
    H, _ = cv2.findHomography(
        grid.astype(np.float64), pitch_points.astype(np.float64), 0
    )
    if H is None or not np.all(np.isfinite(H)):
        return None
    return H.tolist()


def _motion_model_residual(
    points: list[np.ndarray | None], has_raw: list[bool], i: int, half: int
) -> float | None:
    """Median (over grid points) residual of frame ``i``'s pitch grid from a local
    robust constant-velocity model built from its window neighbours. ``None`` when
    there are too few neighbours to form a model (the frame is then accepted)."""
    n = len(points)
    lo, hi = max(0, i - half), min(n, i + half + 1)
    neighbours = [j for j in range(lo, hi) if j != i and has_raw[j]]
    if len(neighbours) < 2:
        return None
    # Robust per-grid-point velocity: median of one-step velocities in the window,
    # skipping any step that touches frame i itself.
    steps = [
        points[j] - points[j - 1]
        for j in range(lo + 1, hi)
        if j != i and j - 1 != i and has_raw[j] and has_raw[j - 1]
    ]
    velocity = (
        np.median(np.stack(steps, axis=0), axis=0)
        if steps
        else np.zeros_like(points[i])
    )
    # Robust anchor: median over neighbours of their motion-compensated prediction
    # of frame i. One flipped neighbour cannot drag the median.
    predictions = np.stack(
        [points[j] + velocity * (i - j) for j in neighbours], axis=0
    )
    predicted = np.median(predictions, axis=0)
    return float(np.median(np.linalg.norm(points[i] - predicted, axis=1)))


def smooth_homography_trajectory(
    estimates: Sequence[RawEstimate],
    *,
    frame_size: tuple[int, int],
    max_gap_frames: int = 150,
    outlier_threshold_cm: float = 2500.0,
    smoothing_window: int = 15,
) -> list[SmoothedFrame]:
    """Smooth a raw homography trajectory. See the module docstring for the full
    contract (visible-pitch grid parameterization, motion-compensated outlier
    rejection, gap/status semantics, units)."""
    n = len(estimates)
    if n == 0:
        return []

    grid = _image_grid(frame_size)
    half = max(smoothing_window // 2, 0)

    frame_ids = [e.frame_idx for e in estimates]
    confidences = [float(e.confidence) for e in estimates]

    # 1. Project each usable raw H forward into pitch-space grid points.
    raw_points: list[np.ndarray | None] = [
        _grid_pitch_points(e.homography, grid) for e in estimates
    ]
    has_raw = [p is not None for p in raw_points]

    # 2. Motion-compensated outlier rejection: reject a frame only when its grid
    #    departs from the local robust constant-velocity model (pans are signal).
    outlier = [False] * n
    for i in range(n):
        if not has_raw[i]:
            continue
        residual = _motion_model_residual(raw_points, has_raw, i, half)
        if residual is not None and residual > outlier_threshold_cm:
            outlier[i] = True

    accepted = [has_raw[i] and not outlier[i] for i in range(n)]

    # 3. Smooth accepted pitch-grid trajectories with a centred window measured in
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
        # MEDIAN, not mean: outlier rejection is per-frame and imperfect, so a
        # contaminated frame that passes it must not be able to drag the window.
        # One survivor in a 9-frame mean moved a player 18.9 m on SNMOT-122/440.
        smoothed_points[i] = np.median(np.stack(window, axis=0), axis=0)

    accepted_indices = [i for i in range(n) if accepted[i]]

    # 4. Emit one output frame per input position.
    out: list[SmoothedFrame] = []
    for i in range(n):
        if accepted[i]:
            H = _refit_homography(grid, smoothed_points[i])
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
        H = _refit_homography(grid, points)
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
