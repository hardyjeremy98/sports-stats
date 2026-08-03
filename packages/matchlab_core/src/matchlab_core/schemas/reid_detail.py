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


class ChannelScore(BaseModel):
    """One evidence channel's contribution to a merge decision, in nats.

    `llr=None` means the channel had no evidence and abstained -- distinct from
    an llr of 0.0, which is a real "this is neither for nor against". Keeping
    them apart is the difference between "position said nothing" and "position
    said the two are equally likely", and only the first points at a broken
    input.
    """

    name: str  # body | occupancy | gap | transition | jersey
    raw: float | None = None  # the measurement before calibration
    llr: float | None = None
    weight: float = 1.0
    contribution: float = 0.0  # weight * llr, what was actually summed


class PairChannels(BaseModel):
    """Why one pair scored what it did. Recorded for the decisive candidate of
    each tracklet -- merged OR rejected under threshold -- so a merge that did
    not happen can be explained, not just the ones that did."""

    a: int
    b: int
    decision: str  # merged | rejected
    total: float
    threshold: float
    # Not `pass` (a Python keyword) and not an alias for it: an alias only
    # applies on input, so the artifact silently serialised as `pass_` and the
    # UI read undefined. An explicit name avoids the whole trap.
    pass_no: int = 1
    channels: list[ChannelScore] = Field(default_factory=list)


class CandidateEvidence(BaseModel):
    """One thread this tracklet was scored against, with its channel working."""

    partner: int  # tracklet id representing the candidate thread
    total: float
    channels: list[ChannelScore] = Field(default_factory=list)


class MergeDecision(BaseModel):
    """One tracklet's merge decision and the alternatives behind it.

    `decision` is "merged" or "abstained". An abstention with a populated
    candidate list is "nothing scored high enough"; an abstention with an empty
    one is "nothing was even eligible" -- two very different situations that a
    bare count cannot tell apart.
    """

    tracklet_id: int
    # Which pass decided it: 1 = tracklet joins an accumulated thread,
    # 2 = whole thread against whole thread. Pass-2 merges were previously
    # unrecorded, so a run could merge and still report "no merged decisions".
    pass_no: int = 1
    decision: str
    chosen: int | None = None  # partner tracklet id when merged
    total: float | None = None  # best candidate's score
    runner_up: float | None = None
    threshold: float
    n_candidates_total: int = 0  # before truncation to `candidates`
    candidates: list[CandidateEvidence] = Field(default_factory=list)


class ReidDetailReport(BaseModel):
    impl: str
    params: dict = Field(default_factory=dict)
    n_parts: int = 0
    tracklets: list[TrackletDetail] = Field(default_factory=list)
    pairs: list[PairDetail] = Field(default_factory=list)
    # Per-channel working from the two-pass engine. Empty under the pairwise
    # engine, which scores a single similarity and has no channels to show.
    pair_channels: list[PairChannels] = Field(default_factory=list)
    # Per-tracklet decisions with their ranked candidates -- the merge
    # inspector's primary view. Empty under the pairwise engine.
    decisions: list[MergeDecision] = Field(default_factory=list)
    # Fit/serve coherence report (FusionModel.serving_diagnostics): per channel
    # the served raw-value distribution next to the fitted one its calibrator
    # encodes, with `flag: true` where the drift exceeds anything a legitimate
    # substrate change has produced. Empty under the pairwise engine.
    coherence: dict = Field(default_factory=dict)
