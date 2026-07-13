from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from pitchlab_core.schemas.team import Team


class EventType(StrEnum):
    # v1: produced by the heuristic possession/pass state machine.
    TOUCH = "touch"
    PASS = "pass"
    MISSED_PASS = "missed_pass"
    RESTART = "restart"
    POSSESSION_GAIN = "possession_gain"
    # v2 seam: types the learned spotting/attribution stages will emit.
    # Present in the schema now so QA labels collected on them are usable later.
    SHOT = "shot"
    TACKLE = "tackle"
    INTERCEPTION = "interception"


class PossessionSegment(BaseModel):
    """A span where one player is in control of the ball."""

    player_id: int
    team: Team
    start_frame: int
    end_frame: int
    start_t: float
    end_t: float
    confidence: float


class Event(BaseModel):
    event_id: int
    type: EventType
    frame_idx: int
    t: float
    player_id: int | None = None
    team: Team = Team.UNKNOWN
    confidence: float = 0.0
    # Contested events (crowded attribution, low confidence, v2 types) are
    # queued for human QA; QA decisions become training labels.
    contested: bool = False
    # Type-specific extras, e.g. {"receiver_player_id": 4, "distance_cm": 1200}
    attrs: dict[str, float | int | str | bool | None] = {}


class StatLine(BaseModel):
    player_id: int
    label: str  # identity label if known, else "Player {id}"
    team: Team
    passes: int = 0
    missed_passes: int = 0
    touches: int = 0
    restarts: int = 0
    possession_seconds: float = 0.0


class StatSheet(BaseModel):
    """The per-player stat sheet — the user-facing deliverable."""

    players: list[StatLine]
