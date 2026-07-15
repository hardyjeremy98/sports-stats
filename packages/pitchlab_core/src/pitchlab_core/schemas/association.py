from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class AssociationRejectReason(StrEnum):
    """Why a candidate pair of tracklets did not end up merged. Ordered as the
    associator evaluates them; the first one that fails is the recorded reason."""

    NO_FEATURES = "no_features"
    TEMPORAL_OVERLAP = "temporal_overlap"
    GAP_TOO_LONG = "gap_too_long"
    SPEED_IMPLAUSIBLE = "speed_implausible"
    COLOR_TOO_FAR = "color_too_far"
    EMBED_TOO_FAR = "embed_too_far"  # learned re-ID associator's gate
    SPAN_CONFLICT = "span_conflict"  # rejected at union-find time


class AssociationPair(BaseModel):
    """One pairwise decision the associator made. Recorded for every tracklet
    pair that passes the structural filters (same team, neither a referee) —
    referee/team-mismatch pairs are never recorded, to bound the O(n^2) payload.

    Exactly one of color_distance/embed_distance is populated, per the
    associator impl that produced this report."""

    a: int  # tracklet ids, a = earlier-starting
    b: int
    gap_s: float | None = None  # null past the first failing constraint
    dist_px: float | None = None
    color_distance: float | None = None
    embed_distance: float | None = None
    affinity: float | None = None
    decision: Literal["merged", "rejected"]
    reason: AssociationRejectReason | None = None  # null when merged


class AssociationEntitySummary(BaseModel):
    player_id: int
    tracklet_ids: list[int]
    merge_edges: list[tuple[int, int]]  # accepted union-find edges


class AssociationReport(BaseModel):
    """association.json: the full per-pair decision trail behind one run's
    cross-tracklet association pass, for the Lab's association inspector."""

    impl: str
    params: dict
    pairs: list[AssociationPair]
    entities: list[AssociationEntitySummary]
