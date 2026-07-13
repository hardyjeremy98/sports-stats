"""v2 seam: learned action spotting (shots, tackles, interceptions — the
"what happened, when" T-DEED-style model). Deferred by scope; the slot, the
Event types, and the QA path all exist today so the v2 model drops in without
schema or UI changes. This stub returns no events and every v1 config keeps
the stage disabled."""

from __future__ import annotations

from pydantic import BaseModel

from pitchlab_core.interfaces import EventSpotter, StageContext
from pitchlab_core.registry import register
from pitchlab_core.schemas import Event
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    checkpoint: str = ""


@register(StageKind.SPOTTING, "none")
class NoSpotting(EventSpotter):
    def __init__(self, **params):
        self.params = Params(**params)

    def spot(self, ctx: StageContext) -> list[Event]:
        return []
