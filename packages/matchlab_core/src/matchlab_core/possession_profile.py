"""Label-risk profile for weak possessor labels (SPO-83, criterion 2).

WHAT THIS IS NOT: an accuracy measure. No per-frame possessor ground truth
exists on any MatchLab data tier, so nothing here can say how often a weak
label is *wrong*. Every number describes the structure of the label set itself
-- how much of it is asserted at all, how much rests on a near-tie between
candidates, how much is temporally implausible, and how often the winning
candidate looks like a projection artifact rather than a real possessor. A
hand-labelled held-out set is the only route to an accuracy figure and is a
separate, deferred piece of work.

The profiler is impl-agnostic: it consumes any `PossessionEstimator` output, so
it serves the learned `possession-peral` estimator unchanged if that is ever
built.

The depth-discordance indicator is a PROXY for the "ball in front of a distant
player" false-possession mode (Peral et al., VISAPP 2025). Bounding-box height
is the only depth cue available without calibration, so the proxy also fires on
two players at genuinely the same depth whose boxes differ in height -- one
crouching, one occluded, one truncated at the frame edge. Never quote the rate
without that caveat.
"""

from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.schemas import BallObservation, PossessorFrame, Tracklet
from matchlab_core.stages.possession.heuristic_image import Params
from matchlab_core.stages.possession.ranking import index_possessor_boxes, rank_candidates

DEFAULT_TAU_GRID_PX = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
DEFAULT_DEPTH_RATIO_GRID = (1.2, 1.5, 2.0)
DEFAULT_TE_FRAMES = 3


class AbstentionBreakdown(BaseModel):
    """Why frames carry no label. Sums to total_frames - asserted_frames."""

    no_ball_observation: int = 0  # no timeline row at all -- ball never observed
    outside_radius: int = 0       # nearest candidate beyond possession_radius_px
    contested_tie: int = 0        # nearest vs runner-up closer than min_margin_px


class CurvePoint(BaseModel):
    """One point of a threshold sweep. Swept rather than reported at a single
    threshold so no flattering cut-off can be chosen after seeing the data."""

    threshold: float
    count: int
    fraction: float


class SegmentStats(BaseModel):
    """Temporal structure of the possessor field. Labels that flicker faster
    than physical ball control are noise regardless of candidate margin."""

    count: int = 0
    total_segment_frames: int = 0
    mean_frames: float = 0.0
    below_te_count: int = 0
    below_te_fraction: float = 0.0
    changes: int = 0
    span_seconds: float = 0.0
    changes_per_second: float = 0.0


class PossessorLabelProfile(BaseModel):
    total_frames: int
    asserted_frames: int
    coverage: float
    abstention: AbstentionBreakdown
    contested_curve: list[CurvePoint] = []
    depth_evaluable_frames: int = 0
    depth_discordance: list[CurvePoint] = []
    segments: SegmentStats = SegmentStats()
    implausible_team_flips: int = 0


def _fraction(count: int, denom: int) -> float:
    return count / denom if denom else 0.0


def profile_possessor_labels(
    timeline: list[PossessorFrame],
    tracklets: list[Tracklet],
    ball: list[BallObservation],
    params: Params,
    *,
    total_frames: int,
    fps: float = 25.0,
    tau_grid_px: tuple[float, ...] = DEFAULT_TAU_GRID_PX,
    depth_ratio_grid: tuple[float, ...] = DEFAULT_DEPTH_RATIO_GRID,
    te_frames: int = DEFAULT_TE_FRAMES,
) -> PossessorLabelProfile:
    """Profile a possessor timeline against the inputs it was derived from.

    `total_frames` is caller-supplied (sequence length / manifest frame count):
    the timeline alone cannot distinguish "ball not observed" from "clip ended".
    """
    if len(timeline) > total_frames:
        raise ValueError(
            f"total_frames ({total_frames}) is below the timeline row count "
            f"({len(timeline)}) -- the caller passed the wrong frame count"
        )

    asserted = [r for r in timeline if r.possessor_tracklet_id is not None]
    abstained = [r for r in timeline if r.possessor_tracklet_id is None]
    abstention = AbstentionBreakdown(
        no_ball_observation=total_frames - len(timeline),
        outside_radius=sum(1 for r in abstained if r.margin >= params.min_margin_px),
        contested_tie=sum(1 for r in abstained if r.margin < params.min_margin_px),
    )

    contested_curve = [
        CurvePoint(
            threshold=tau,
            count=(c := sum(1 for r in asserted if r.margin < tau)),
            fraction=_fraction(c, len(asserted)),
        )
        for tau in tau_grid_px
    ]

    boxes_by_frame = index_possessor_boxes(tracklets)
    ball_by_frame = {b.frame_idx: b for b in ball}
    # Height ratio of the nearest *other* candidate to the possessor. Deliberately
    # "nearest other" rather than ranked[1]: smoothing can leave a possessor who
    # was not the raw nearest, and the runner-up must stay a genuine rival.
    depth_ratios: list[float] = []
    for r in asserted:
        obs = ball_by_frame.get(r.frame_idx)
        if obs is None:
            continue
        ranked = rank_candidates(obs, boxes_by_frame.get(r.frame_idx, []))
        possessor = next(
            (c for c in ranked if c.tracklet_id == r.possessor_tracklet_id), None
        )
        others = [c for c in ranked if c.tracklet_id != r.possessor_tracklet_id]
        if possessor is None or not others:
            continue
        possessor_h = possessor.box.y2 - possessor.box.y1
        if possessor_h <= 0:
            continue
        depth_ratios.append((others[0].box.y2 - others[0].box.y1) / possessor_h)

    depth_discordance = [
        CurvePoint(
            threshold=ratio,
            count=(c := sum(1 for d in depth_ratios if d > ratio)),
            fraction=_fraction(c, len(depth_ratios)),
        )
        for ratio in depth_ratio_grid
    ]

    return PossessorLabelProfile(
        total_frames=total_frames,
        asserted_frames=len(asserted),
        coverage=_fraction(len(asserted), total_frames),
        abstention=abstention,
        contested_curve=contested_curve,
        depth_evaluable_frames=len(depth_ratios),
        depth_discordance=depth_discordance,
    )
