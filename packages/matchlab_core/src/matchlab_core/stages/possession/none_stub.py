"""Disabled possession stub: emits no possessor timeline. Lets a pipeline
declare the slot (or omit it) and stay green with zero config."""

from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.interfaces import PossessionEstimator, StageContext
from matchlab_core.registry import register
from matchlab_core.schemas import BallObservation, PossessorFrame, TeamAssignment, Tracklet
from matchlab_core.schemas.run import StageKind


class Params(BaseModel):
    pass


@register(StageKind.POSSESSION, "none")
class NoPossession(PossessionEstimator):
    def __init__(self, **params):
        self.params = Params(**params)

    def estimate(
        self,
        ctx: StageContext,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
        ball: list[BallObservation],
    ) -> list[PossessorFrame]:
        return []
