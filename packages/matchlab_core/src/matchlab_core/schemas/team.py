from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class Team(StrEnum):
    HOME = "home"
    AWAY = "away"
    REFEREE = "referee"
    UNKNOWN = "unknown"


class TeamAssignment(BaseModel):
    """Kit-color team decision for one tracklet (aggregated over its crops)."""

    tracklet_id: int
    team: Team
    confidence: float
    # Mean kit color of the tracklet's crops, for UI swatches. RGB 0-255.
    kit_color: tuple[int, int, int] | None = None
