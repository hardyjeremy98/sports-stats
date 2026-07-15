from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.team import Team


class IdentityKind(StrEnum):
    FACE = "face"
    JERSEY = "jersey"
    MANUAL = "manual"  # set via QA correction
    NONE = "none"


class IdentityEvidence(BaseModel):
    """One piece of evidence behind an identity decision — a scored candidate
    frame (e.g. a high-resolution face crop or a legible jersey crop)."""

    tracklet_id: int
    frame_idx: int
    score: float
    crop_artifact: str | None = None  # relative path under the run dir, for the Lab UI
    upscaled: bool = False
    box: Box | None = None  # head-crop region in source-frame pixel coords
    raw_crop_artifact: str | None = None  # pre-upscale crop, present only when upscaling fired


class PlayerIdentity(BaseModel):
    kind: IdentityKind = IdentityKind.NONE
    # Display label: jersey number ("7"), roster name, or face-cluster tag ("P3").
    label: str | None = None
    confidence: float = 0.0
    evidence: list[IdentityEvidence] = []


class PlayerEntity(BaseModel):
    """A physical player over the whole clip: the output of the offline global
    cross-tracklet association pass, plus the identity decision for the group."""

    player_id: int
    tracklet_ids: list[int]
    team: Team = Team.UNKNOWN
    identity: PlayerIdentity = PlayerIdentity()
    association_confidence: float = 1.0
