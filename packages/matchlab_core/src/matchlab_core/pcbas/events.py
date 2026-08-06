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

from pydantic import BaseModel, Field, model_validator

from matchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
from matchlab_core.pcbas.schema import CLASS_NAMES, N_CLASSES, N_ROLES, N_SLOTS, slot_index


class PCBASEvent(BaseModel):
    """One action, attributed to one player at one frame.

    Two identities coexist and they are NOT interchangeable:

    * `slot` -- the tactical role slot (ADR 008). What our model predicts, and what
      `score_events(identity="slot")` matches on.
    * `shirt_number` -- the jersey. What the reference's on-disk playbyplay JSON
      exchange uses, reachable only via the export-time roster remap.

    Both are optional because neither format carries both: the reference JSON has no
    role, and a slot-native prediction has no shirt until it is remapped. `slot` is
    derived from `left_to_right` + `role_id` when omitted, and cross-checked when
    both are supplied -- a disagreement there is a silent-corruption bug.
    """

    frame_idx: int
    left_to_right: int = Field(ge=0, le=1)
    role_id: int | None = Field(default=None, ge=1, le=N_ROLES)
    slot: int | None = Field(default=None, ge=0, lt=N_SLOTS)
    shirt_number: int | None = None
    class_id: int = Field(ge=0, lt=N_CLASSES)
    score: float = 1.0
    t: float | None = None

    # Ground-truth-only observability flags, carried by the reference's playbyplay
    # GT rows. `has_bbox` is False for the 17.5% of VAL actions whose player is
    # off-screen; those are the events only the sequence stage can recover, and the
    # TAAD-vs-DST gap on them (33 vs 390 recovered) is the argument for the two-stage
    # split. `is_replay` marks frames with no visible players at all.
    has_bbox: bool | None = None
    is_replay: bool | None = None

    @model_validator(mode="after")
    def _reconcile_slot(self) -> PCBASEvent:
        if self.role_id is None:
            return self
        derived = slot_index(self.left_to_right, self.role_id)
        if self.slot is None:
            object.__setattr__(self, "slot", derived)
        elif self.slot != derived:
            raise ValueError(
                f"slot {self.slot} contradicts left_to_right={self.left_to_right} "
                f"role_id={self.role_id} (which packs to {derived})"
            )
        return self

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
