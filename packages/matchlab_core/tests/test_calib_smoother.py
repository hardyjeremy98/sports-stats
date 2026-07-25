"""TDD for the offline global homography smoother (SPO-63 / SPO-70).

Every assertion is on *where known points project* under the output homographies
(image or pitch positions), never on raw matrix entries — the visible-pitch grid
parameterization is an internal implementation detail.

Homography convention (matches FrameCalibration.homography): image pixels ->
pitch centimeters, row-major 3x3.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from matchlab_core.calib.smoother import (
    RawEstimate,
    SmoothStatus,
    smooth_homography_trajectory,
)

# The frame the synthetic image quads / grid live in.
FRAME_SIZE = (1280, 720)

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


def _probe_pitch_position(homography: list[list[float]], x: float, y: float) -> np.ndarray:
    """Where an image point lands in pitch cm under an image->pitch homography.

    The physical quantity the product cares about. `_probe_image_location` inverts
    the homography to place a fixed *pitch* point in the image, which has high gain
    on tilted views — a sub-metre pitch-space error reads as hundreds of pixels there.
    """
    H = np.array(homography, dtype=np.float64)
    v = H @ np.array([x, y, 1.0])
    return v[:2] / v[2]


# A player-height image point: low in the frame, where players actually stand.
PLAYER_PROBE_PX = (640.0, 650.0)


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

    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE, smoothing_window=9)

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

    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE, smoothing_window=9)
    out_locs = [_probe_image_location(f.homography) for f in out]

    # Interior frames (full symmetric window) track the true pan closely. A
    # centred smoother has zero lag on a linear trend, leaving only noise residual.
    interior = range(10, n - 10)
    tracking_err = max(float(np.linalg.norm(out_locs[i] - true_locs[i])) for i in interior)
    assert tracking_err < 6.0

    # And jitter about the trend is reduced vs the raw estimates.
    raw_jitter = _mean_consecutive_jitter([raw_locs[i] for i in interior])
    out_jitter = _mean_consecutive_jitter([out_locs[i] for i in interior])
    assert out_jitter < raw_jitter * 0.6


def test_fast_pan_is_signal_not_outlier() -> None:
    """A *fast* linear pan (grid moves far faster than in a slow pan) plus noise:
    the motion-compensated rejector must treat the coherent camera motion as
    signal, keeping (almost) every frame FRESH with no lag. This is the exact
    case v1's static window-median rejection got wrong."""
    rng = np.random.default_rng(7)
    n = 60

    def fast_quad(t: float) -> np.ndarray:
        # ~30 px/frame at the quad -> the probe/grid moves many px/frame, the
        # anchor-speed regime that v1 misclassified as outliers.
        return BASE_IMAGE_QUAD + np.array([30.0 * t, 0.0])

    estimates: list[RawEstimate] = []
    for i in range(n):
        noisy = fast_quad(i) + rng.normal(0.0, 4.0, BASE_IMAGE_QUAD.shape)
        estimates.append(
            RawEstimate(frame_idx=i, homography=_image_to_pitch_H(noisy), confidence=1.0)
        )

    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE, smoothing_window=9)

    fresh = sum(1 for f in out if f.status is SmoothStatus.FRESH)
    assert fresh / n >= 0.90

    # No lag / no flatten, asserted in PITCH METRES at player height — the unit the
    # product cares about. The previous assertion used `_probe_image_location`, whose
    # inverse projection has high gain: v2 measures 1.10 m there but 6.7 px, while
    # robust aggregation measures 1.81 m and 144.7 px. The pixel number tracked probe
    # gain, not physical harm (see the v3 design spec, "The cost, and why it is worth
    # paying"). 2.5 m still fails on genuine lag or a flattened pan.
    true_pitch = [
        _probe_pitch_position(_image_to_pitch_H(fast_quad(i)), *PLAYER_PROBE_PX)
        for i in range(n)
    ]
    out_pitch = [_probe_pitch_position(f.homography, *PLAYER_PROBE_PX) for f in out]
    interior = range(10, n - 10)
    tracking_err_m = max(
        float(np.linalg.norm(out_pitch[i] - true_pitch[i])) / 100.0 for i in interior
    )
    assert tracking_err_m < 2.5

    # The pan is tracked, not smoothed away: the output covers a comparable distance
    # to the truth over the interior. This is what actually catches lag/flatten, and
    # it is independent of the error bound above.
    def _path_len(traj: list[np.ndarray]) -> float:
        return float(sum(np.linalg.norm(traj[i] - traj[i - 1]) for i in range(11, n - 10)))

    assert abs(_path_len(out_pitch) - _path_len(true_pitch)) <= 0.25 * _path_len(true_pitch)


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

    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE, smoothing_window=9)

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


def test_single_flip_mid_pan_is_rejected_and_continuous() -> None:
    """A single frame whose homography is grossly wrong (a mirror flip) mid-pan is
    rejected (SMOOTHED, not FRESH); the surrounding pan stays FRESH and the probe
    trajectory is continuous across the flip (no spike)."""
    n = 41
    flip_at = 20
    estimates: list[RawEstimate] = []
    for i in range(n):
        H = np.array(_image_to_pitch_H(_panned_quad(i)), dtype=np.float64)
        if i == flip_at:
            # Mirror the pitch mapping about the field's long axis — a valid,
            # invertible, high-"confidence" matrix that is geometrically a flip.
            mirror = np.array([[-1.0, 0.0, 10500.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            H = mirror @ H
        estimates.append(
            RawEstimate(frame_idx=i, homography=H.tolist(), confidence=1.0)
        )

    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE, smoothing_window=9)

    assert out[flip_at].status is SmoothStatus.SMOOTHED
    assert out[flip_at].homography is not None
    assert out[flip_at - 1].status is SmoothStatus.FRESH
    assert out[flip_at + 1].status is SmoothStatus.FRESH

    # The output probe trajectory has no spike at the flip: the step across it is
    # comparable to the ordinary pan step, not the huge raw flip excursion.
    locs = [_probe_image_location(f.homography) for f in out]
    steps = [float(np.linalg.norm(locs[i] - locs[i - 1])) for i in range(1, n)]
    ordinary = float(np.median(steps))
    assert steps[flip_at] < 5.0 * ordinary
    assert steps[flip_at + 1] < 5.0 * ordinary


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
        estimates, frame_size=FRAME_SIZE, max_gap_frames=150, smoothing_window=9
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
        estimates, frame_size=FRAME_SIZE, max_gap_frames=20, smoothing_window=9
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

    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE)

    for i in range(5):
        assert out[i].status is SmoothStatus.ABSENT
        assert out[i].homography is None
    assert out[5].status is SmoothStatus.FRESH


def test_degenerate_inputs() -> None:
    """Empty -> empty; all-None -> all ABSENT; single fresh frame -> FRESH."""
    assert smooth_homography_trajectory([], frame_size=FRAME_SIZE) == []

    all_none = [RawEstimate(frame_idx=i, homography=None) for i in range(5)]
    out = smooth_homography_trajectory(all_none, frame_size=FRAME_SIZE)
    assert len(out) == 5
    assert all(f.status is SmoothStatus.ABSENT and f.homography is None for f in out)

    H = _image_to_pitch_H(BASE_IMAGE_QUAD)
    single = smooth_homography_trajectory(
        [RawEstimate(frame_idx=7, homography=H, confidence=0.9)],
        frame_size=FRAME_SIZE,
    )
    assert len(single) == 1
    assert single[0].status is SmoothStatus.FRESH
    assert single[0].frame_idx == 7
    assert single[0].homography is not None
    # A single frame is smoothed over a window of itself -> reproduces its H.
    loc_in = _probe_image_location(H)
    loc_out = _probe_image_location(single[0].homography)
    assert float(np.linalg.norm(loc_in - loc_out)) < 1e-3


def test_singular_raw_homography_treated_as_missing() -> None:
    """A singular candidate matrix does not crash and is treated as missing."""
    singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0]]
    estimates = [
        RawEstimate(frame_idx=0, homography=_image_to_pitch_H(_panned_quad(0))),
        RawEstimate(frame_idx=1, homography=singular),
        RawEstimate(frame_idx=2, homography=_image_to_pitch_H(_panned_quad(2))),
    ]
    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE)
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
    a = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE)
    b = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE)
    assert a == b


def test_gap_measured_in_frame_units_with_stride() -> None:
    """frame_idx is strided: the gap cap is applied in FRAME units, not sample
    count. Two adjacent samples spanning > cap frames is a long gap (ABSENT)."""
    stride = 100
    estimates = [
        RawEstimate(frame_idx=0, homography=_image_to_pitch_H(_panned_quad(0))),
        RawEstimate(frame_idx=stride, homography=None),  # missing sample
        RawEstimate(frame_idx=2 * stride, homography=_image_to_pitch_H(_panned_quad(2))),
    ]
    # Bracketing anchors span 2*stride = 200 frames > cap 150 -> ABSENT.
    out = smooth_homography_trajectory(
        estimates, frame_size=FRAME_SIZE, max_gap_frames=150
    )
    assert out[1].status is SmoothStatus.ABSENT
    # Widen the cap past the frame span -> now interpolated.
    out2 = smooth_homography_trajectory(
        estimates, frame_size=FRAME_SIZE, max_gap_frames=250
    )
    assert out2[1].status is SmoothStatus.INTERPOLATED
    assert out2[1].homography is not None


# --- REAL-DATA regression: the point of Task 10 (SPO-70) ---------------------

_FIXTURE = (
    Path(__file__).parent / "fixtures" / "snmot123_pan_raw_homographies.json"
)


def _median_filter_2d(traj: np.ndarray, k: int) -> np.ndarray:
    """Per-column centred median filter (window k), edge-clamped."""
    n = len(traj)
    half = k // 2
    out = np.empty_like(traj)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(traj[lo:hi], axis=0)
    return out


def _local_linear_jitter(traj: np.ndarray) -> float:
    """Mean residual of each point from a 5-frame centred local linear fit."""
    n = len(traj)
    res = []
    for t in range(n):
        lo, hi = max(0, t - 2), min(n, t + 3)
        idx = np.arange(lo, hi)
        A = np.vstack([idx, np.ones(len(idx))]).T
        pred = np.array(
            [np.linalg.lstsq(A, traj[idx, c], rcond=None)[0] @ [t, 1.0] for c in range(2)]
        )
        res.append(float(np.linalg.norm(traj[t] - pred)))
    return float(np.mean(res))


def test_real_snmot123_pan_is_accepted_and_tracked() -> None:
    """100 REAL image->FIFA-cm homographies from SNMOT-123 including a sustained
    fast pan (the fixture that broke v1, which rejected 90%+ of these frames).

    v2 must (a) ACCEPT >=80% as FRESH, (b) TRACK the pan without lag/flatten, and
    (c) reduce jitter vs the raw trajectory."""
    data = json.loads(_FIXTURE.read_text())
    w, h = data["image_size"]
    records = data["records"]
    estimates = [
        RawEstimate(
            frame_idx=r["frame_idx"],
            homography=r["homography"],
            confidence=r.get("confidence", 0.0),
        )
        for r in records
    ]

    out = smooth_homography_trajectory(estimates, frame_size=(w, h), smoothing_window=9)
    assert len(out) == len(records)

    # (a) Acceptance: at least 80% of frames survive as FRESH (v1 managed ~12%).
    fresh = sum(1 for f in out if f.status is SmoothStatus.FRESH)
    assert fresh / len(out) >= 0.80

    # Probe: the image-frame centre projected into pitch cm.
    centre = np.array([w / 2.0, h / 2.0, 1.0])

    def pitch_of(H: list[list[float]]) -> np.ndarray:
        p = np.array(H, dtype=np.float64) @ centre
        return p[:2] / p[2]

    raw_probe = np.array([pitch_of(r["homography"]) for r in records])
    # Every frame gets an output homography here (only the lone flip is rejected,
    # and it is reconstructed, not dropped) -> a dense smoothed trajectory.
    assert all(f.homography is not None for f in out)
    out_probe = np.array([pitch_of(f.homography) for f in out])

    # (b1) Smoothed motion is physically plausible: each frame-to-frame probe step
    # is <= 2 m for >=99% of steps (the raw trajectory spikes to kilometres at the
    # flip; the smoother must not).
    steps = np.linalg.norm(np.diff(out_probe, axis=0), axis=1)
    assert np.mean(steps <= 200.0) >= 0.99

    # (b2) No lag/flatten: over the pan segment the smoothed total path length is
    # within 20% of the raw robust (median-filtered) total — the pan is preserved,
    # not smoothed away.
    seg = slice(19, 46)
    raw_robust_total = float(
        np.linalg.norm(np.diff(_median_filter_2d(raw_probe[seg], 5), axis=0), axis=1).sum()
    )
    out_seg_total = float(np.linalg.norm(np.diff(out_probe[seg], axis=0), axis=1).sum())
    assert abs(out_seg_total - raw_robust_total) <= 0.20 * raw_robust_total

    # (c) Output jitter (residual vs a 5-frame local linear fit) is strictly lower
    # than the raw trajectory's.
    assert _local_linear_jitter(out_probe) < _local_linear_jitter(raw_probe)


def test_contaminated_neighbour_does_not_drag_the_window() -> None:
    """A frame whose PROJECTIVE ROW is perturbed sends its horizon-ward grid points
    far away while barely moving the near-field ones. Only 3 of 9 grid points blow
    up, so the median-over-grid rejection accepts the frame (this is v2's real blind
    spot, measured on SNMOT-122). A mean over the window then drags every grid point
    — including the near-field ones where players stand — and the DLT refit spreads
    that error across the whole frame.

    Nothing here is rejected: the assertion is that a clean frame's own output is not
    corrupted by an accepted-but-contaminated neighbour.
    """
    n, centre, corrupt = 21, 10, 12
    clean = [_image_to_pitch_H(BASE_IMAGE_QUAD + np.array([8.0 * i, 0.0])) for i in range(n)]

    # Perturb only the projective row of one neighbour.
    bad = np.array(clean[corrupt], dtype=np.float64)
    eps = 3e-3
    bad[2, 1] += eps * bad[2, 2]
    bad[2, 0] -= 0.3 * eps * bad[2, 2]

    homographies = list(clean)
    homographies[corrupt] = bad.tolist()
    estimates = [
        RawEstimate(frame_idx=i, homography=homographies[i], confidence=1.0) for i in range(n)
    ]

    out = smooth_homography_trajectory(estimates, frame_size=FRAME_SIZE, smoothing_window=9)

    # Premise: the contaminated frame is ACCEPTED, not rejected. If this ever fails,
    # the blind spot has moved and the test below is no longer testing what it claims.
    assert out[corrupt].status is SmoothStatus.FRESH

    # The clean centre frame's player-height projection must stay put. v2 (mean)
    # measures ~4.4 m of error here; robust aggregation measures ~0.1 m.
    truth = _probe_pitch_position(clean[centre], *PLAYER_PROBE_PX)
    got = _probe_pitch_position(out[centre].homography, *PLAYER_PROBE_PX)
    assert float(np.linalg.norm(got - truth)) / 100.0 < 1.0
