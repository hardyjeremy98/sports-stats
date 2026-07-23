"""Ball resolution: raw ball detections -> one BallObservation per frame.

The ball detector fires false positives (heads, socks, far-side balls) and
misses fast/occluded balls. Strategy, following roboflow/sports' BallTracker:
keep a short buffer of accepted positions and pick the candidate closest to the
buffer centroid; then linearly interpolate small gaps (2D-only in v1).
"""

from __future__ import annotations

from collections import deque

from matchlab_core.schemas import BallObservation, Detection, DetectionClass, FrameDetections
from matchlab_core.schemas.geometry import Point


def resolve_ball_track(
    frames: list[FrameDetections],
    fps: float,
    buffer_size: int = 10,
    max_gap_frames: int = 30,
) -> list[BallObservation]:
    buffer: deque[tuple[float, float]] = deque(maxlen=buffer_size)
    observed: list[BallObservation] = []

    for fd in frames:
        candidates = [d for d in fd.detections if d.cls == DetectionClass.BALL]
        if not candidates:
            continue
        best = _closest_to_centroid(candidates, buffer)
        c = best.box.center
        buffer.append((c.x, c.y))
        observed.append(
            BallObservation(
                frame_idx=fd.frame_idx,
                t=fd.t,
                xy=Point(x=c.x, y=c.y),
                confidence=best.confidence,
            )
        )

    return _interpolate_gaps(observed, fps, max_gap_frames)


def _closest_to_centroid(
    candidates: list[Detection], buffer: deque[tuple[float, float]]
) -> Detection:
    if not buffer or len(candidates) == 1:
        return max(candidates, key=lambda d: d.confidence)
    cx = sum(p[0] for p in buffer) / len(buffer)
    cy = sum(p[1] for p in buffer) / len(buffer)
    return min(
        candidates,
        key=lambda d: (d.box.center.x - cx) ** 2 + (d.box.center.y - cy) ** 2,
    )


def _interpolate_gaps(
    observed: list[BallObservation], fps: float, max_gap_frames: int
) -> list[BallObservation]:
    if not observed:
        return []
    out: list[BallObservation] = [observed[0]]
    for prev, nxt in zip(observed, observed[1:]):
        gap = nxt.frame_idx - prev.frame_idx
        if 1 < gap <= max_gap_frames:
            for k in range(1, gap):
                a = k / gap
                fi = prev.frame_idx + k
                out.append(
                    BallObservation(
                        frame_idx=fi,
                        t=fi / fps,
                        xy=Point(
                            x=prev.xy.x + a * (nxt.xy.x - prev.xy.x),
                            y=prev.xy.y + a * (nxt.xy.y - prev.xy.y),
                        ),
                        confidence=min(prev.confidence, nxt.confidence) * 0.5,
                        interpolated=True,
                    )
                )
        out.append(nxt)
    return out
