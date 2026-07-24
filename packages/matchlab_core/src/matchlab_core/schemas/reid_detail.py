"""reid_detail.json — the re-ID engine's working, shown. Per tracklet: the
view prototypes (which source frames built them, which frame is the exemplar
crop, per-part visibility). Per scored pair: the winning prototype pair and
its per-part cosine evidence. Per tracklet: the ranked gate-passing
candidates the decision rule saw. Rendered by the Lab's merge inspector;
boxes for crops come from tracklets.json (frame-indexed client-side)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PrototypeDetail(BaseModel):
    exemplar_frame_idx: int  # highest-quality member — the crop to show
    member_frame_idxs: list[int]  # all source frames, quality-join order
    part_visibility: list[float]  # length P, mean visibility per part


class CandidateRank(BaseModel):
    tracklet_id: int
    affinity: float


class TrackletDetail(BaseModel):
    tracklet_id: int
    starved: bool = False
    prototypes: list[PrototypeDetail] = Field(default_factory=list)
    # Gate-passing similarity candidates, best first (the ranking the
    # mutual-best rule and margin test read).
    candidates: list[CandidateRank] = Field(default_factory=list)


class PairDetail(BaseModel):
    a: int  # tracklet ids, a < b
    b: int
    affinity: float
    a_proto: int  # winning prototype index on each side
    b_proto: int
    part_cosines: list[float | None]  # length P; None = part excluded
    part_weights: list[float | None]  # min(vis_a, vis_b); None = excluded


class ReidDetailReport(BaseModel):
    impl: str
    params: dict = Field(default_factory=dict)
    n_parts: int = 0
    tracklets: list[TrackletDetail] = Field(default_factory=list)
    pairs: list[PairDetail] = Field(default_factory=list)
