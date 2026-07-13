"""Reserved slot for a learned/query-propagation tracker (MOTRv2/MOTIP-style).

Locked decision #4: the slot exists now so swapping the learned tracker in
later never touches calling code — it is registered, appears in the Lab's
config editor, and satisfies the Tracker interface. A learned implementation
would ignore `detections` and consume ctx.frames() directly (the interface
passes both by design).
"""

from __future__ import annotations

from pydantic import BaseModel

from pitchlab_core.interfaces import StageContext, Tracker
from pitchlab_core.registry import register
from pitchlab_core.schemas import FrameDetections, Tracklet
from pitchlab_core.schemas.run import StageKind


class Params(BaseModel):
    checkpoint: str = ""


@register(StageKind.TRACK, "learned-motr")
class LearnedTrackerStub(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)

    def prepare(self, ctx: StageContext) -> None:
        raise NotImplementedError(
            "The learned query-propagation tracker is a v2 slot. Train/port a "
            "MOTRv2- or MOTIP-style model, implement track() here, and every "
            "config/UI path already works with it."
        )

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        raise NotImplementedError
