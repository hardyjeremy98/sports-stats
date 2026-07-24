from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.schemas.team import Team


class PossessorFrame(BaseModel):
    """One frame of the possessor timeline: who (if anyone) holds the ball.

    This is the image-space possession signal (SPO-77/79) — keyed by
    `possessor_tracklet_id`, distinct from `PossessionSegment` (pitch-space,
    entity-keyed, written by the `events` engine). `None` means loose ball /
    no confident possessor (abstention). `margin` is the separation between the
    nearest and second-nearest candidate — the attribution certainty the
    transition->event rules (SPO-78) use to flag contested events.
    """

    frame_idx: int
    t: float
    possessor_tracklet_id: int | None = None
    team: Team = Team.UNKNOWN
    confidence: float = 0.0
    margin: float = 0.0
