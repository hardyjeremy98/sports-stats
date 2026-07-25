"""Reproject pitch-template vertices into the image through a calibration.

A line/heatmap calibrator (PnLCalib) has no per-frame detected-keypoint set of its
own, but the Lab's "Pitch keypoints" overlay draws `keypoints_image`. Placing the
canonical pitch vertices on the field via the calibration is exactly what that
overlay is for, and lets a viewer judge calibration quality directly.

This lives here — a pure function, no stage context — because more than one
producer writes `calibration.jsonl`: the `pnlcalib` stage during a pipeline run,
and offline re-scoring tools that re-smooth persisted raw estimates. Both must
fill these fields identically, or an artifact rewritten offline loses the overlay
(SPO-84: exactly that happened, and nothing failed — the rows still validated and
every metric still computed, the Lab just went blank).
"""

from __future__ import annotations

import numpy as np

from matchlab_core.schemas.geometry import Point

__all__ = ["reproject_pitch_vertices", "MARGIN_FRAC"]

# Keypoints are kept this far outside the frame (fraction of the larger frame
# dimension) before being dropped.
MARGIN_FRAC = 0.05


def reproject_pitch_vertices(
    homography: list[list[float]] | None,
    vertices: np.ndarray | list[tuple[float, float]],
    frame_size: tuple[int, int],
) -> list[Point]:
    """Pitch-spec vertices (cm) reprojected into image pixels via ``homography``
    (image px -> pitch cm), culled to what is actually visible.

    Vertices on the far side of the image horizon reproject into a meaningless
    cluster near the vanishing line, so keep only those on the foreground side of
    the horizon (the same side as the image bottom-centre, which is always
    near-field), and drop anything well outside the frame.

    Returns an empty list when the homography is missing or non-invertible.
    """
    if homography is None:
        return []
    hm = np.asarray(homography, dtype=np.float64)
    if hm.shape != (3, 3) or not np.all(np.isfinite(hm)):
        return []
    try:
        h_inv = np.linalg.inv(hm)
    except np.linalg.LinAlgError:
        return []

    verts = np.asarray(vertices, dtype=np.float64)
    w, h = float(frame_size[0]), float(frame_size[1])
    margin = MARGIN_FRAC * max(w, h)

    # The image horizon is where the image->cm map sends points to infinity:
    # hz . [x, y, 1] = 0.
    hz = hm[2]
    fg = np.sign(hz[0] * (w / 2.0) + hz[1] * h + hz[2])

    pts: list[Point] = []
    for vx, vy in verts:
        pt = h_inv @ np.array([vx, vy, 1.0])
        if abs(pt[2]) < 1e-9:  # vertex on the horizon / at infinity
            continue
        x, y = pt[0] / pt[2], pt[1] / pt[2]
        if np.sign(hz[0] * x + hz[1] * y + hz[2]) != fg:  # behind the horizon
            continue
        if -margin <= x <= w + margin and -margin <= y <= h + margin:
            pts.append(Point(x=float(x), y=float(y)))
    return pts
