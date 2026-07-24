from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.schemas.team import Team


class MinimapPlayer(BaseModel):
    player_id: int
    # Pitch coordinates in centimeters; origin top-left, x along the length,
    # y along the width. Extent depends on the run's pitch spec (config `pitch:`)
    # -- 12000x7000 for roboflow, 10500x6800 for fifa. See matchlab_core.pitch.
    x: float
    y: float
    team: Team
    confidence: float = 1.0


class MinimapBall(BaseModel):
    x: float
    y: float
    confidence: float = 1.0
    interpolated: bool = False


class MinimapFrame(BaseModel):
    """Fused game state for one frame. One JSONL row in minimap.jsonl —
    this is what the 2D replay renders."""

    frame_idx: int
    t: float
    players: list[MinimapPlayer]
    ball: MinimapBall | None = None
    calibration_confidence: float = 0.0
