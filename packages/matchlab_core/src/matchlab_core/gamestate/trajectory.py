"""Offline per-player pitch-space trajectory smoother.

The calibration smoother (`matchlab_core.calib.smoother`) makes the *camera*
trajectory coherent. This makes the *player* trajectory coherent, and it is a
separate problem: even a perfect homography still yields a shimmering dot,
because the detector's box bottom-centre jitters by several pixels every frame
and depth multiplies those pixels into centimetres.

Measured on the Gate 2 panel before this existed: the median frame-to-frame
*acceleration* of a rendered dot was ~11 cm — roughly 690 m/s², which no human
produces. That is pure noise, and it is what makes the 2D replay look alive with
the wrong kind of movement.

Three jobs, in this order:

1. **Reject the physically impossible.** A step implying more than
   ``max_speed_mps`` cannot be a player, whatever produced it — a bad homography
   frame, a detector flicker, an identity swap. Rejected observations are
   reconstructed from neighbours rather than rendered. This catches calibration
   teleports the camera-side smoother could not, because it reasons about human
   dynamics instead of camera geometry.
2. **Smooth what remains**, with a centred window, so real motion is preserved
   with no lag on a linear trend.
3. **Bridge short gaps.** A dot that winks out for a few frames and returns is
   the most distracting artifact in the replay, and short dropouts are the
   common case (77% of gaps on the Gate 2 panel are ≤25 frames).

Defaults were chosen by measurement on that panel, not by taste. ``window=21``
(0.84 s at 25 fps) puts median rendered acceleration at 1.5 cm/frame² — below the
~1.6 cm a human can produce — while a hard 10 m/s² cut is reproduced to within
6 cm. Widening further keeps buying smoothness but starts flattening real cuts
(11 cm error at 25, 21 cm at 31), which is the point where the filter would begin
inventing a calmer match than the one that was played.

What it deliberately does NOT do: bridge long gaps. Those are ~23% of gaps but
the overwhelming majority of missing frame-time (up to 482 frames on the panel),
and filling them would be inventing positions for a player nobody tracked.
Absence is reported, never fabricated — the same discipline as the calibration
smoother's ``absent`` status and the identity program's abstention rule.

Note the contrast with the calibration smoother, which cannot use a plain local
fit: a homography has 8 degrees of freedom, so aggregating its projected points
independently can produce a grid no homography realizes. A 2D player position
carries no such constraint, which is why a straightforward local polynomial fit
is both safe and the right tool here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "TrackPoint",
    "PointStatus",
    "SmoothedPoint",
    "smooth_track",
]


@dataclass(frozen=True)
class TrackPoint:
    """One observed pitch position (cm) at a source-video frame."""

    frame_idx: int
    x: float
    y: float


class PointStatus(str, Enum):
    OBSERVED = "observed"
    SMOOTHED = "smoothed"
    INTERPOLATED = "interpolated"
    ABSENT = "absent"


@dataclass(frozen=True)
class SmoothedPoint:
    frame_idx: int
    x: float
    y: float
    status: PointStatus


def _reject_implausible(
    frames: np.ndarray,
    xs: np.ndarray,
    fps: float,
    max_speed_mps: float,
    reject_threshold_cm: float = 200.0,
    window: int = 9,
) -> np.ndarray:
    """Boolean mask of observations to keep.

    Each point is compared against a *motion-compensated robust prediction* built
    from its window neighbours: velocity is the median of one-step differences
    (excluding steps touching the point itself), and the prediction is the median
    over neighbours of where each would put it. Both medians mean the estimate
    survives a burst of consecutive bad frames, and motion compensation means a
    genuine sprint is signal rather than an outlier.

    Comparing against immediate neighbours instead — "unreachable from both
    sides" — cannot see a burst at all: two consecutive bad frames vouch for each
    other, since each is reachable from the other. That is the same failure mode
    the calibration smoother's per-frame rejection had, and it left players
    rendered tens of metres off the pitch.
    """
    n = len(frames)
    keep = np.ones(n, dtype=bool)
    if n < 3:
        return keep
    half = max(window // 2, 1)
    cap_cm_per_frame = (max_speed_mps * 100.0) / fps
    t = frames.astype(np.float64)

    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neighbours = [j for j in range(lo, hi) if j != i]
        if len(neighbours) < 2:
            continue
        steps = [
            (xs[j] - xs[j - 1]) / max(t[j] - t[j - 1], 1.0)
            for j in range(lo + 1, hi)
            if j != i and j - 1 != i
        ]
        velocity = (
            np.median(np.stack(steps, axis=0), axis=0) if steps else np.zeros(2, dtype=np.float64)
        )
        predicted = np.median(
            np.stack([xs[j] + velocity * (t[i] - t[j]) for j in neighbours], axis=0), axis=0
        )
        residual = float(np.linalg.norm(xs[i] - predicted))
        # Tolerate genuine motion the local model cannot express (a turn, an
        # acceleration) up to one frame of full-speed travel.
        if residual > reject_threshold_cm + cap_cm_per_frame:
            keep[i] = False
    return keep


def _local_quadratic(frames: np.ndarray, xs: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay-equivalent: fit a degree-2 polynomial in time over a centred
    window and evaluate it at the centre.

    Chosen over a moving average or a median because it reproduces constant
    velocity AND constant acceleration exactly — so a real sprint or turn comes
    through untouched while uncorrelated per-frame noise is averaged down. The
    fit is in true frame units, so irregular sample spacing (a bridged dropout)
    is handled correctly rather than being treated as uniform.
    """
    n = len(frames)
    if n < 3:
        return xs.copy()
    half = max(window // 2, 1)
    out = np.empty_like(xs)
    t_all = frames.astype(np.float64)
    # The window is measured in FRAMES, not in samples. Two consequences, both
    # necessary: a long absence cannot bleed across the gap into the fit (samples
    # on the far side are simply out of range), and a track fragmented by
    # rejections is still smoothed against every nearby sample instead of being
    # cut into stubs too short for a fit to do anything. Sample-space windows got
    # both of these wrong.
    lo_idx = np.searchsorted(t_all, t_all - half, side="left")
    hi_idx = np.searchsorted(t_all, t_all + half, side="right")
    for i in range(n):
        lo, hi = int(lo_idx[i]), int(hi_idx[i])
        if hi - lo < 3:
            out[i] = xs[i]
            continue
        t = t_all[lo:hi] - t_all[i]
        # Evaluating the fit at t = 0 is just the constant term.
        out[i] = [np.polyfit(t, xs[lo:hi, c], 2)[-1] for c in range(2)]
    return out


def smooth_track(
    observations: Sequence[TrackPoint],
    *,
    fps: float = 25.0,
    max_gap_frames: int = 25,
    max_speed_mps: float = 12.0,
    window: int = 21,
    reject_threshold_cm: float = 120.0,
) -> list[SmoothedPoint]:
    """Smooth one player's pitch-space trajectory. See the module docstring.

    Emits one point per source frame from the first to the last observation.
    Frames inside a gap longer than ``max_gap_frames`` are omitted entirely
    (nothing is drawn); leading and trailing frames are never extrapolated.
    """
    obs = sorted(observations, key=lambda p: p.frame_idx)
    if not obs:
        return []
    if len(obs) == 1:
        return [SmoothedPoint(obs[0].frame_idx, obs[0].x, obs[0].y, PointStatus.OBSERVED)]

    frames = np.array([p.frame_idx for p in obs], dtype=np.int64)
    xs = np.array([[p.x, p.y] for p in obs], dtype=np.float64)

    keep = _reject_implausible(frames, xs, fps, max_speed_mps, reject_threshold_cm)
    kept_frames, kept_xs = frames[keep], xs[keep]
    if len(kept_frames) == 0:
        return []

    # The fit windows by FRAME distance, so a long absence excludes itself and no
    # explicit segmentation is needed here; bridgeability is decided per gap below.
    smoothed = _local_quadratic(kept_frames, kept_xs, window)

    observed = {int(f) for f in frames}
    kept_set = {int(f) for f in kept_frames}

    out: list[SmoothedPoint] = []
    for a in range(len(kept_frames)):
        f = int(kept_frames[a])
        out.append(
            SmoothedPoint(f, float(smoothed[a][0]), float(smoothed[a][1]), PointStatus.OBSERVED)
        )
        if a + 1 >= len(kept_frames):
            break
        nxt = int(kept_frames[a + 1])
        span = nxt - f
        if span <= 1:
            continue

        # Anything filled into this interval is anchored on BOTH bracketing
        # samples, so it may only be filled when the span between them is
        # bridgeable — short enough, and a journey a player could actually make.
        # The same rule segmented the track above, evaluated on the same raw
        # samples so the two can never disagree.
        #
        # This bound also protects the rejected-observation case below. A rejected
        # point sitting in a NON-bridgeable span brackets two disjoint segments,
        # so reconstructing it would place the player midway across a jump that
        # never happened — reintroducing the teleport rejection just removed.
        if span - 1 > max_gap_frames:
            continue
        reach_m = float(np.linalg.norm(kept_xs[a + 1] - kept_xs[a])) / 100.0
        if reach_m > max_speed_mps * (span / fps):
            continue

        for g in range(f + 1, nxt):
            alpha = (g - f) / span
            x = (1 - alpha) * smoothed[a][0] + alpha * smoothed[a + 1][0]
            y = (1 - alpha) * smoothed[a][1] + alpha * smoothed[a + 1][1]
            # Observed-but-rejected keeps its dot: the player WAS seen there, only
            # the position was wrong, so it is corrected rather than deleted —
            # deleting it would turn a calibration glitch into a visual hole.
            status = (
                PointStatus.SMOOTHED
                if (g in observed and g not in kept_set)
                else PointStatus.INTERPOLATED
            )
            out.append(SmoothedPoint(g, float(x), float(y), status))

    out.sort(key=lambda p: p.frame_idx)
    return out
