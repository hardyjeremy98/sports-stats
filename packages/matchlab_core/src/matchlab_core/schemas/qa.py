from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from matchlab_core.schemas.events import Event


class QAReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    CONTESTED_ATTRIBUTION = "contested_attribution"
    UNKNOWN_IDENTITY = "unknown_identity"
    CALIBRATION_GAP = "calibration_gap"


class QAStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class QAItem(BaseModel):
    """A pipeline output that needs a human eye. Written by the events stage,
    persisted by the server, reviewed in the Lab UI."""

    qa_id: int
    event: Event
    reason: QAReason
    status: QAStatus = QAStatus.PENDING
    # Candidate players when attribution was ambiguous (ordered by likelihood).
    candidate_player_ids: list[int] = []
    note: str | None = None


class LabeledExample(BaseModel):
    """A training label distilled from a QA decision. Accept → positive label of
    the model's own output; correct → label with the human's fix. These are the
    seed data for the v2 learned attribution head."""

    source_run_id: str
    qa_id: int
    event_type: str
    frame_idx: int
    t: float
    # What the pipeline predicted and what the human decided.
    predicted_player_id: int | None
    labeled_player_id: int | None
    accepted: bool
