"""`pcbas_events.json` -- the per-event view of a PCBAS run, for the Lab.

This is an INSPECTION artifact, not an exchange format. Where `spotting.json`
(schemas/spotting.py) carries a spotter's class + time and nothing else, a
player-centric spotter's output is only meaningful with the player attached, and
its performance is only readable with the matcher's verdict attached. So every row
here is one matcher decision -- a true positive, a false positive, or a missed
ground-truth event -- rather than one prediction.

Both sides of the comparison live in ONE array on purpose. The Lab renders a
timeline where a miss and a false alarm sit at the same instant; splitting
predictions and ground truth into two artifacts would make that the UI's problem to
re-join, and the join is exactly the part that must not be re-derived (the greedy,
score-ordered matcher in `pcbas.eval` is the only thing entitled to decide which
prediction claims which GT event).

Predictions below the report's `conf_thresh` appear nowhere: the reference drops
them before matching, so they are not false positives and not events.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from matchlab_core.pcbas.eval import PCBASReport

Verdict = Literal["tp", "fp", "fn"]


class PCBASLabEvent(BaseModel):
    """One matcher decision, with everything the Lab needs to show it in place."""

    verdict: Verdict
    # The frame the Lab seeks to: the PREDICTION's frame for tp/fp, the ground
    # truth's for fn. `gt_frame_idx`/`frame_error` expose the localisation offset a
    # true positive absorbs, which no aggregate F1 can show.
    frame_idx: int
    t: float
    class_id: int
    class_name: str
    score: float | None = None  # None for fn -- ground truth has no confidence
    gt_frame_idx: int | None = None
    frame_error: int | None = None

    # Attribution. `left_to_right` is the pitch SIDE the player attacks toward, not
    # a club (FOOTPASS's TEAM column flips at half time), so it is only meaningful
    # within one half -- which is exactly this artifact's scope.
    shirt_number: int | None = None
    left_to_right: int | None = None
    role_id: int | None = None
    slot: int | None = None

    # Ground-truth observability, present on tp/fn only. `has_bbox=False` marks the
    # events whose player is off-screen -- unreachable for any purely visual model,
    # and the reason the sequence stage exists. Worth filtering on in the Lab.
    has_bbox: bool | None = None
    is_replay: bool | None = None
    # The acting player's box in 640x352 video pixels at `frame_idx`, when the
    # tactical data observes them there. Lets the overlay point at WHO acted.
    box: tuple[float, float, float, float] | None = None


class PCBASLabEvents(BaseModel):
    """One half's decisions, plus the report they were counted into.

    `report` is produced by the same matcher call that produced `events`, so the
    headline metrics the Lab shows can never disagree with the rows beneath them.
    """

    key: str
    game_id: str
    half: int
    fps: float
    identity: str
    delta: int
    conf_thresh: float
    report: PCBASReport
    events: list[PCBASLabEvent] = []
