"""TDD for the offline global homography smoother (SPO-63).

Synthetic homography trajectories only. Every assertion is on *where known
pitch points project* under the output homographies (image-space anchor / probe
positions), never on raw matrix entries — the point-correspondence
parameterization is an internal implementation detail.

Homography convention (matches FrameCalibration.homography): image pixels ->
pitch centimeters, row-major 3x3.
"""

from __future__ import annotations

import cv2
import numpy as np
from matchlab_core.calib.smoother import (
    RawEstimate,
    SmoothStatus,
    smooth_homography_trajectory,
)

# Four well-spread pitch anchors (cm): the pitch corners of a FIFA-ish field.
PITCH_ANCHORS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (10500.0, 0.0),
    (10500.0, 6800.0),
    (0.0, 6800.0),
]
# A convex image quad (px) that the corners map to under a plausible wide
# broadcast camera in a 1280x720 frame (well-conditioned vertical parallax).
BASE_IMAGE_QUAD = np.array(
    [(350.0, 150.0), (930.0, 150.0), (1250.0, 680.0), (30.0, 680.0)],
    dtype=np.float64,
)
# A probe pitch point we track through the homographies (centre-ish, not an anchor).
PROBE_PITCH = (5250.0, 3400.0)


def _image_to_pitch_H(image_quad: np.ndarray) -> list[list[float]]:
    """Homography mapping image px -> pitch cm from a 4-point correspondence."""
    H, _ = cv2.findHomography(
        image_quad.astype(np.float64),
        np.array(PITCH_ANCHORS, dtype=np.float64),
        0,
    )
    return H.tolist()


def _probe_image_location(homography: list[list[float]]) -> np.ndarray:
    """Where PROBE_PITCH lands in the image under an image->pitch homography."""
    H = np.array(homography, dtype=np.float64)
    Hinv = np.linalg.inv(H)
    v = Hinv @ np.array([PROBE_PITCH[0], PROBE_PITCH[1], 1.0])
    return v[:2] / v[2]


def _mean_consecutive_jitter(locations: list[np.ndarray]) -> float:
    steps = [
        float(np.linalg.norm(locations[i] - locations[i - 1]))
        for i in range(1, len(locations))
    ]
    return float(np.mean(steps)) if steps else 0.0


def test_constant_camera_noise_is_smoothed() -> None:
    """Constant camera + per-frame gaussian noise -> output jitter << input jitter."""
    rng = np.random.default_rng(0)
    n = 60
    raw_locs: list[np.ndarray] = []
    estimates: list[RawEstimate] = []
    for i in range(n):
        image_quad = BASE_IMAGE_QUAD + rng.normal(0.0, 8.0, BASE_IMAGE_QUAD.shape)
        H = _image_to_pitch_H(image_quad)
        estimates.append(RawEstimate(frame_idx=i, homography=H, confidence=1.0))
        raw_locs.append(_probe_image_location(H))

    out = smooth_homography_trajectory(estimates, pitch_points=PITCH_ANCHORS, smoothing_window=9)

    assert len(out) == n
    assert all(f.status is SmoothStatus.FRESH for f in out)
    assert all(f.homography is not None for f in out)

    out_locs = [_probe_image_location(f.homography) for f in out]
    input_jitter = _mean_consecutive_jitter(raw_locs)
    output_jitter = _mean_consecutive_jitter(out_locs)
    assert output_jitter < input_jitter * 0.5


def _panned_quad(t: float) -> np.ndarray:
    """Base quad translated horizontally by a per-frame pan rate."""
    return BASE_IMAGE_QUAD + np.array([2.0 * t, 0.0])


def test_slow_pan_is_tracked_without_lag() -> None:
    """A smooth linear pan + noise: output tracks the true pan (no lag at
    interior frames) with reduced jitter."""
    rng = np.random.default_rng(1)
    n = 60
    true_locs: list[np.ndarray] = []
    raw_locs: list[np.ndarray] = []
    estimates: list[RawEstimate] = []
    for i in range(n):
        true_H = _image_to_pitch_H(_panned_quad(i))
        true_locs.append(_probe_image_location(true_H))
        noisy = _panned_quad(i) + rng.normal(0.0, 4.0, BASE_IMAGE_QUAD.shape)
        H = _image_to_pitch_H(noisy)
        estimates.append(RawEstimate(frame_idx=i, homography=H, confidence=1.0))
        raw_locs.append(_probe_image_location(H))

    out = smooth_homography_trajectory(estimates, pitch_points=PITCH_ANCHORS, smoothing_window=9)
    out_locs = [_probe_image_location(f.homography) for f in out]

    # Interior frames (full symmetric window) track the true pan closely. Per-frame
    # probe motion is ~2px, so a window-scale lag would be ~8px; a centred smoother
    # has zero lag on a linear trend, leaving only sub-6px noise residual.
    interior = range(10, n - 10)
    tracking_err = max(float(np.linalg.norm(out_locs[i] - true_locs[i])) for i in interior)
    assert tracking_err < 6.0

    # And jitter about the trend is reduced vs the raw estimates.
    raw_jitter = _mean_consecutive_jitter([raw_locs[i] for i in interior])
    out_jitter = _mean_consecutive_jitter([out_locs[i] for i in interior])
    assert out_jitter < raw_jitter * 0.6


def test_gross_outlier_is_rejected_and_reconstructed() -> None:
    """A single gross outlier mid-pan: its output lands near the interpolated
    neighbour position, not near the outlier, and is flagged SMOOTHED."""
    n = 41
    estimates: list[RawEstimate] = []
    for i in range(n):
        quad = _panned_quad(i)
        if i == 20:
            quad = quad + np.array([400.0, 300.0])  # gross shift, well past threshold
        estimates.append(
            RawEstimate(frame_idx=i, homography=_image_to_pitch_H(quad), confidence=1.0)
        )

    out = smooth_homography_trajectory(estimates, pitch_points=PITCH_ANCHORS, smoothing_window=9)

    assert out[20].status is SmoothStatus.SMOOTHED
    assert out[20].homography is not None

    outlier_loc = _probe_image_location(estimates[20].homography)
    reconstructed = _probe_image_location(out[20].homography)
    neighbour_avg = 0.5 * (
        _probe_image_location(out[19].homography)
        + _probe_image_location(out[21].homography)
    )
    assert float(np.linalg.norm(reconstructed - neighbour_avg)) < 5.0
    assert float(np.linalg.norm(reconstructed - outlier_loc)) > 100.0


def test_short_gap_is_interpolated_monotonically() -> None:
    """A gap k < max_gap_frames: filled INTERPOLATED, probe point moves
    monotonically between the endpoints with no jump at either boundary."""
    n = 60
    gap = range(25, 35)  # 10 missing frames, well under the cap
    estimates: list[RawEstimate] = []
    for i in range(n):
        if i in gap:
            estimates.append(RawEstimate(frame_idx=i, homography=None))
        else:
            estimates.append(
                RawEstimate(frame_idx=i, homography=_image_to_pitch_H(_panned_quad(i)))
            )

    out = smooth_homography_trajectory(
        estimates, pitch_points=PITCH_ANCHORS, max_gap_frames=150, smoothing_window=9
    )

    assert all(out[i].status is SmoothStatus.INTERPOLATED for i in gap)
    assert all(out[i].homography is not None for i in gap)

    xs = [float(_probe_image_location(out[i].homography)[0]) for i in range(24, 36)]
    diffs = np.diff(xs)
    assert np.all(diffs > 0)  # strictly monotonic across the whole span incl. boundaries
    # No jump at the boundaries: steps entering/leaving the gap are comparable to
    # steps inside it (not a discontinuity).
    assert max(diffs) < 3.0 * min(diffs)


def test_long_gap_is_absent() -> None:
    """A gap k > max_gap_frames: those frames are ABSENT with homography None."""
    n = 60
    gap = range(10, 50)  # 40 missing frames
    estimates: list[RawEstimate] = []
    for i in range(n):
        if i in gap:
            estimates.append(RawEstimate(frame_idx=i, homography=None))
        else:
            estimates.append(
                RawEstimate(frame_idx=i, homography=_image_to_pitch_H(_panned_quad(i)))
            )

    out = smooth_homography_trajectory(
        estimates, pitch_points=PITCH_ANCHORS, max_gap_frames=20, smoothing_window=9
    )

    for i in gap:
        assert out[i].status is SmoothStatus.ABSENT
        assert out[i].homography is None
        assert out[i].confidence == 0.0


def test_leading_gap_is_absent_not_extrapolated() -> None:
    """A leading gap (no anchor on the left) is ABSENT, never extrapolated."""
    n = 30
    estimates: list[RawEstimate] = []
    for i in range(n):
        if i < 5:
            estimates.append(RawEstimate(frame_idx=i, homography=None))
        else:
            estimates.append(
                RawEstimate(frame_idx=i, homography=_image_to_pitch_H(_panned_quad(i)))
            )

    out = smooth_homography_trajectory(estimates, pitch_points=PITCH_ANCHORS)

    for i in range(5):
        assert out[i].status is SmoothStatus.ABSENT
        assert out[i].homography is None
    assert out[5].status is SmoothStatus.FRESH


def test_degenerate_inputs() -> None:
    """Empty -> empty; all-None -> all ABSENT; single fresh frame -> FRESH."""
    assert smooth_homography_trajectory([], pitch_points=PITCH_ANCHORS) == []

    all_none = [RawEstimate(frame_idx=i, homography=None) for i in range(5)]
    out = smooth_homography_trajectory(all_none, pitch_points=PITCH_ANCHORS)
    assert len(out) == 5
    assert all(f.status is SmoothStatus.ABSENT and f.homography is None for f in out)

    H = _image_to_pitch_H(BASE_IMAGE_QUAD)
    single = smooth_homography_trajectory(
        [RawEstimate(frame_idx=7, homography=H, confidence=0.9)],
        pitch_points=PITCH_ANCHORS,
    )
    assert len(single) == 1
    assert single[0].status is SmoothStatus.FRESH
    assert single[0].frame_idx == 7
    assert single[0].homography is not None
    # A single frame is smoothed over a window of itself -> reproduces its H.
    loc_in = _probe_image_location(H)
    loc_out = _probe_image_location(single[0].homography)
    assert float(np.linalg.norm(loc_in - loc_out)) < 1e-6


def test_singular_raw_homography_treated_as_missing() -> None:
    """A singular candidate matrix does not crash and is treated as missing."""
    singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0]]
    estimates = [
        RawEstimate(frame_idx=0, homography=_image_to_pitch_H(_panned_quad(0))),
        RawEstimate(frame_idx=1, homography=singular),
        RawEstimate(frame_idx=2, homography=_image_to_pitch_H(_panned_quad(2))),
    ]
    out = smooth_homography_trajectory(estimates, pitch_points=PITCH_ANCHORS)
    # No usable raw at frame 1 -> filled from both sides as INTERPOLATED.
    assert out[1].status is SmoothStatus.INTERPOLATED
    assert out[1].homography is not None


def test_determinism() -> None:
    """Same input twice -> byte-identical output."""
    rng = np.random.default_rng(2)
    estimates = [
        RawEstimate(
            frame_idx=i,
            homography=_image_to_pitch_H(
                _panned_quad(i) + rng.normal(0.0, 5.0, BASE_IMAGE_QUAD.shape)
            ),
            confidence=0.8,
        )
        for i in range(30)
    ]
    a = smooth_homography_trajectory(estimates, pitch_points=PITCH_ANCHORS)
    b = smooth_homography_trajectory(estimates, pitch_points=PITCH_ANCHORS)
    assert a == b


def test_gap_measured_in_frame_units_with_stride() -> None:
    """frame_idx is strided: the gap cap is applied in FRAME units, not sample
    count. Two adjacent samples spanning > cap frames is a long gap (ABSENT)."""
    stride = 100
    # Samples at frame_idx 0 and 100 with 99..1 missing frames between them,
    # represented as a single missing sample at frame_idx 100? No — model as
    # strided samples where one sample is missing.
    estimates = [
        RawEstimate(frame_idx=0, homography=_image_to_pitch_H(_panned_quad(0))),
        RawEstimate(frame_idx=stride, homography=None),  # missing sample
        RawEstimate(frame_idx=2 * stride, homography=_image_to_pitch_H(_panned_quad(2))),
    ]
    # Bracketing anchors span 2*stride = 200 frames > cap 150 -> ABSENT.
    out = smooth_homography_trajectory(
        estimates, pitch_points=PITCH_ANCHORS, max_gap_frames=150
    )
    assert out[1].status is SmoothStatus.ABSENT
    # Widen the cap past the frame span -> now interpolated.
    out2 = smooth_homography_trajectory(
        estimates, pitch_points=PITCH_ANCHORS, max_gap_frames=250
    )
    assert out2[1].status is SmoothStatus.INTERPOLATED
    assert out2[1].homography is not None
