"""Shared ball->player candidate geometry (SPO-83).

Extracted from `heuristic_image.py` so the possessor estimator, the label-risk
profiler (`matchlab_core.possession_profile`) and the denoiser
(`matchlab_core.possession_denoise`) rank candidates with one implementation.
Two copies of this geometry would let the profiler report on a ranking the
estimator never used.

Lives at package top level, NOT under `stages/`, on purpose: it is pure geometry
with no registry or StageContext dependency, and importing anything inside
`stages/` executes `stages/__init__.py`, which registers every implementation.
That made a cycle the moment a stage (`possession/viterbi.py`) depended on a
top-level module that needed this geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from matchlab_core.schemas import BallObservation, Box, DetectionClass, Point, Tracklet

POSSESSOR_CLASSES = frozenset({DetectionClass.PLAYER, DetectionClass.GOALKEEPER})


@dataclass(frozen=True)
class PossessorCandidate:
    tracklet_id: int
    box: Box
    box_confidence: float
    distance: float


def dist_point_box(pt: Point, box: Box) -> float:
    """Euclidean distance from a point to a box; 0 when the point is inside."""
    dx = max(box.x1 - pt.x, 0.0, pt.x - box.x2)
    dy = max(box.y1 - pt.y, 0.0, pt.y - box.y2)
    return math.hypot(dx, dy)


def index_possessor_boxes(
    tracklets: list[Tracklet],
) -> dict[int, list[tuple[int, Box, float]]]:
    """frame_idx -> [(tracklet_id, box, box_confidence)] for possessor classes."""
    out: dict[int, list[tuple[int, Box, float]]] = {}
    for tr in tracklets:
        if tr.cls not in POSSESSOR_CLASSES:
            continue
        for fr in tr.frames:
            out.setdefault(fr.frame_idx, []).append((tr.tracklet_id, fr.box, fr.confidence))
    return out


def rank_candidates(
    ball: BallObservation, boxes: list[tuple[int, Box, float]]
) -> list[PossessorCandidate]:
    """Candidates sorted nearest-first by distance from the ball, ties by id."""
    candidates = [
        PossessorCandidate(
            tracklet_id=tid,
            box=box,
            box_confidence=conf,
            distance=dist_point_box(ball.xy, box),
        )
        for tid, box, conf in boxes
    ]
    candidates.sort(key=lambda c: (c.distance, c.tracklet_id))
    return candidates
