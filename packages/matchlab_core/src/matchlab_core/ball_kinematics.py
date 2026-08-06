"""Ball-trajectory touch detection — the heuristic action-spotting baseline (B3).

A touch is a significant change in the ball's own motion: it turns, or its speed
changes sharply. This signal is INDEPENDENT of the nearest-player possession
estimator (`stages/possession/heuristic_image.py`): it reads the ball's
kinematics and knows nothing about which players are near it. Two signals with
different failure modes can corroborate each other without either being ground
truth — see `matchlab_core.event_crossval`.

Physics reference: Link & Hoernig (PLoS ONE 2017) detect kicks at ball
acceleration >= 4 m/s^2 on clean metric trajectory data. We have 2-D pixel
positions and no calibration, so thresholds here are on scale-free scores
(normalised turn angle, relative speed change) rather than metric acceleration.

HONEST LIMITATIONS, both consequences of having no pitch geometry:

  * **Camera motion** is conflated with ball motion in pixel space. Compensated
    (optionally, default on) by subtracting the median frame-to-frame
    displacement of tracked player boxes: players move in many directions, the
    camera moves them all the same way, so the median is a robust pan estimate.
  * **Depth change is not compensated at all.** A ball moving toward the camera
    changes apparent speed without changing its real speed, and can register as
    a touch. Only pitch calibration fixes this. This is the same limitation that
    invalidated the depth-discordance proxy in the possessor-label audit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel

from matchlab_core.schemas import BallObservation, Tracklet

_EPS = 1e-9


class Params(BaseModel):
    smooth_radius: int = 2            # velocity smoothing half-window, frames
    min_speed_px: float = 1.5         # below this the ball is "not moving"; no touch
    touch_threshold: float = 0.35     # score above which a peak counts as a touch
    min_separation_frames: int = 6    # refractory window; one strike -> one touch
    compensate_camera: bool = True    # subtract median player-box displacement
    interpolated_weight: float = 0.5  # score multiplier when the ball was gap-filled


@dataclass(frozen=True)
class BallTouch:
    frame_idx: int
    t: float
    score: float
    turn: float
    speed_change: float
    interpolated: bool


def _camera_shift(tracklets: list[Tracklet]) -> dict[int, tuple[float, float]]:
    """frame_idx -> median (dx, dy) of tracked boxes from the previous frame.

    A robust pan estimate: real player motion is multi-directional and cancels in
    the median, while camera motion displaces every box the same way.
    """
    centres: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for tr in tracklets:
        for fr in tr.frames:
            centres[tr.tracklet_id][fr.frame_idx] = (
                (fr.box.x1 + fr.box.x2) / 2.0,
                (fr.box.y1 + fr.box.y2) / 2.0,
            )

    deltas: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for by_frame in centres.values():
        for f, (cx, cy) in by_frame.items():
            prev = by_frame.get(f - 1)
            if prev is not None:
                deltas[f].append((cx - prev[0], cy - prev[1]))

    out: dict[int, tuple[float, float]] = {}
    for f, ds in deltas.items():
        xs = sorted(d[0] for d in ds)
        ys = sorted(d[1] for d in ds)
        mid = len(xs) // 2
        out[f] = (xs[mid], ys[mid])
    return out


def _runs(ball: list[BallObservation]) -> list[list[BallObservation]]:
    """Split the track on frame gaps. Motion across a gap is unobserved, so no
    touch may be claimed at the seam."""
    runs: list[list[BallObservation]] = []
    for obs in sorted(ball, key=lambda b: b.frame_idx):
        if runs and runs[-1][-1].frame_idx + 1 == obs.frame_idx:
            runs[-1].append(obs)
        else:
            runs.append([obs])
    return runs


def _velocities(
    run: list[BallObservation], shift: dict[int, tuple[float, float]], compensate: bool
) -> list[tuple[float, float]]:
    """Per-step camera-compensated displacement; index i is the step i -> i+1."""
    vels: list[tuple[float, float]] = []
    for a, b in zip(run, run[1:]):
        dx = b.xy.x - a.xy.x
        dy = b.xy.y - a.xy.y
        if compensate:
            sx, sy = shift.get(b.frame_idx, (0.0, 0.0))
            dx -= sx
            dy -= sy
        vels.append((dx, dy))
    return vels


def _smooth(vels: list[tuple[float, float]], radius: int) -> list[tuple[float, float]]:
    if radius <= 0:
        return vels
    out: list[tuple[float, float]] = []
    for i in range(len(vels)):
        window = vels[max(0, i - radius) : min(len(vels), i + radius + 1)]
        out.append(
            (sum(v[0] for v in window) / len(window), sum(v[1] for v in window) / len(window))
        )
    return out


def _turn_score(v_in: tuple[float, float], v_out: tuple[float, float]) -> float:
    """Angle between successive velocities, normalised: 0 straight, 1 reversal."""
    n_in = math.hypot(*v_in)
    n_out = math.hypot(*v_out)
    if n_in < _EPS or n_out < _EPS:
        return 0.0
    cos = (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (n_in * n_out)
    return (1.0 - max(-1.0, min(1.0, cos))) / 2.0


def _speed_change_score(v_in: tuple[float, float], v_out: tuple[float, float]) -> float:
    s_in, s_out = math.hypot(*v_in), math.hypot(*v_out)
    return abs(s_out - s_in) / (s_in + s_out + _EPS)


def detect_touches(
    ball: list[BallObservation],
    tracklets: list[Tracklet],
    params: Params | None = None,
) -> list[BallTouch]:
    """Detect ball touches from trajectory kinematics.

    `tracklets` are used only for camera-motion compensation; pass [] to skip it
    (equivalent to `compensate_camera=False`).
    """
    p = params or Params()
    shift = _camera_shift(tracklets) if p.compensate_camera else {}

    scored: list[BallTouch] = []
    for run in _runs(ball):
        if len(run) < 3:
            continue
        vels = _smooth(_velocities(run, shift, p.compensate_camera), p.smooth_radius)
        # Compare velocities that STRADDLE the smoothing window rather than
        # adjacent ones: smoothing deliberately blends motion within +/-radius, so
        # an adjacent comparison measures change inside the window and flattens
        # the very turn it should preserve. A 90-degree turn scores ~0.03 that way
        # and ~0.5 straddled.
        lag = p.smooth_radius + 1
        # Step i is run[i] -> run[i+1]; the change straddling step i pivots at run[i].
        for i in range(lag, len(vels) - lag):
            v_in, v_out = vels[i - lag], vels[i + lag]
            if math.hypot(*v_in) < p.min_speed_px and math.hypot(*v_out) < p.min_speed_px:
                continue
            turn = _turn_score(v_in, v_out)
            speed_change = _speed_change_score(v_in, v_out)
            obs = run[i]
            score = max(turn, speed_change)
            if obs.interpolated:
                score *= p.interpolated_weight
            scored.append(
                BallTouch(
                    frame_idx=obs.frame_idx,
                    t=obs.t,
                    score=round(score, 4),
                    turn=round(turn, 4),
                    speed_change=round(speed_change, 4),
                    interpolated=obs.interpolated,
                )
            )

    return _pick_peaks(scored, p.touch_threshold, p.min_separation_frames)


def _pick_peaks(
    scored: list[BallTouch], threshold: float, min_separation: int
) -> list[BallTouch]:
    """Greedy non-maximum suppression: strongest first, then suppress neighbours
    within the refractory window so one strike yields one touch."""
    candidates = [s for s in scored if s.score >= threshold]
    kept: list[BallTouch] = []
    for cand in sorted(candidates, key=lambda s: (-s.score, s.frame_idx)):
        if all(abs(cand.frame_idx - k.frame_idx) >= min_separation for k in kept):
            kept.append(cand)
    return sorted(kept, key=lambda s: s.frame_idx)
