"""Phase 1 possessor estimator (SPO-79): image-space nearest-player-to-ball.

Calibration-optional and minimap-free — it reads only player boxes and the ball
position in pixel space. For each frame with a ball observation it finds the
nearest player box; if that player is within `possession_radius_px` and clearly
nearer than the runner-up (`min_margin_px`), they are the possessor, else the
frame abstains (loose ball / contested). A small windowed-majority pass smooths
single-frame flips — the categorical analog of the paper's Gaussian smoothing.

Fallback / honest limitation: this heuristic *requires* the ball. Frames with no
ball observation produce no possessor row (abstention); a clip with no ball at
all yields an empty timeline. The learned Peral estimator (SPO-83 gate) is the
one that drops the ball dependency — the heuristic baseline does not.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel

from matchlab_core.interfaces import PossessionEstimator, StageContext
from matchlab_core.possession_ranking import index_possessor_boxes, rank_candidates
from matchlab_core.registry import register
from matchlab_core.schemas import (
    BallObservation,
    PossessorFrame,
    Team,
    TeamAssignment,
    Tracklet,
)
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    possession_radius_px: float = 60.0   # ball must be within this of a player box
    min_margin_px: float = 0.0           # abstain if nearest & runner-up within this
    smooth_radius: int = 2               # windowed-majority smoothing radius (0 = off)
    interpolated_ball_weight: float = 0.75  # confidence multiplier for gap-filled ball


def _smooth(labels: list[int | None], radius: int) -> list[int | None]:
    """Centered windowed-majority vote; ties keep the original label."""
    if radius <= 0:
        return labels
    out: list[int | None] = []
    n = len(labels)
    for i in range(n):
        window = labels[max(0, i - radius) : min(n, i + radius + 1)]
        counts = Counter(window)
        top = counts.most_common()
        best_count = top[0][1]
        winners = {lbl for lbl, c in top if c == best_count}
        out.append(labels[i] if labels[i] in winners else top[0][0])
    return out


@register(StageKind.POSSESSION, "possession-heuristic-image")
class HeuristicImagePossession(PossessionEstimator):
    def __init__(self, **params):
        self.params = Params(**params)

    def estimate(
        self,
        ctx: StageContext,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
        ball: list[BallObservation],
    ) -> list[PossessorFrame]:
        p = self.params
        team_by_tid = {t.tracklet_id: t.team for t in teams}

        boxes_by_frame = index_possessor_boxes(tracklets)

        # Raw per-frame nearest-player possessor (with abstention), pre-smoothing.
        raw: list[tuple[int, float, int | None, float, float]] = []  # frame,t,tid,conf,margin
        for obs in sorted(ball, key=lambda b: b.frame_idx):
            ranked = rank_candidates(obs, boxes_by_frame.get(obs.frame_idx, []))
            if not ranked:
                raw.append((obs.frame_idx, obs.t, None, 0.0, 0.0))
                continue
            d0 = ranked[0].distance
            d1 = ranked[1].distance if len(ranked) > 1 else d0 + p.possession_radius_px
            margin = d1 - d0
            if d0 > p.possession_radius_px or margin < p.min_margin_px:
                raw.append((obs.frame_idx, obs.t, None, 0.0, round(margin, 3)))
                continue
            weight = p.interpolated_ball_weight if obs.interpolated else 1.0
            conf = min(1.0, obs.confidence * ranked[0].box_confidence * weight)
            raw.append(
                (obs.frame_idx, obs.t, ranked[0].tracklet_id, round(conf, 4), round(margin, 3))
            )

        smoothed = _smooth([r[2] for r in raw], p.smooth_radius)

        out: list[PossessorFrame] = []
        for (frame_idx, t, orig_tid, conf, margin), tid in zip(raw, smoothed):
            team = team_by_tid.get(tid, Team.UNKNOWN) if tid is not None else Team.UNKNOWN
            out.append(
                PossessorFrame(
                    frame_idx=frame_idx,
                    t=round(t, 3),
                    possessor_tracklet_id=tid,
                    team=team,
                    # conf/margin are the pre-smoothing nearest-player values;
                    # zeroed where smoothing turned the frame loose.
                    confidence=conf if tid == orig_tid else (0.0 if tid is None else conf),
                    margin=margin,
                )
            )
        return out
