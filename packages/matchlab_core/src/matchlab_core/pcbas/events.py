"""Player-centric action events -- the unit both the ingest and the metric speak.

Lives in `matchlab_core`, not `matchlab_train`, because `pcbas.eval` scores these and
core must never import train. `matchlab_train.datasets.footpass_pcbas` produces them;
`matchlab_core.pcbas.eval.score_events` consumes them.

A `PCBASEvent` is deliberately NOT a `matchlab_core.event_gt.GroundTruthEvent`: that
model has no player field at all, so it cannot express the one thing this task is
about. `PCBASEvents.to_event_ground_truth` downcasts to it for the existing MatchLab
GT plumbing, discarding identity -- see `docs/superpowers/specs/
2026-07-27-player-centric-action-spotting-design.md` on that schema gap.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from matchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
from matchlab_core.pcbas.schema import CLASS_NAMES, N_CLASSES, N_SLOTS


class PCBASEvent(BaseModel):
    """One action, attributed to one tactical role slot at one frame.

    `slot` is the scored identity. `shirt_number` rides along for reporting and for
    the ADR 008 export-time roster remap; it is NEVER what the metric matches on,
    because a slot-native model does not predict shirts.
    """

    frame_idx: int
    left_to_right: int = Field(ge=0, le=1)
    role_id: int = Field(ge=1, le=13)
    slot: int = Field(ge=0, lt=N_SLOTS)
    shirt_number: int
    class_id: int = Field(ge=0, lt=N_CLASSES)
    score: float = 1.0
    t: float | None = None

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


class PCBASEvents(BaseModel):
    """Every action in one half, plus enough provenance to say where it came from."""

    key: str
    game_id: str
    half: int
    fps: float
    events: list[PCBASEvent] = []

    def to_event_ground_truth(self, source: str = "footpass") -> EventGroundTruth:
        """Lossy downcast to MatchLab's existing event GT. Drops slot, role and
        shirt -- `GroundTruthEvent` has nowhere to put them."""
        return EventGroundTruth(
            source=source,
            sequence=self.key,
            fps=self.fps,
            events=[
                GroundTruthEvent(
                    class_=e.class_name,
                    frame_idx=e.frame_idx,
                    t=e.t if e.t is not None else e.frame_idx / self.fps,
                    half=self.half,
                )
                for e in self.events
            ],
        )
