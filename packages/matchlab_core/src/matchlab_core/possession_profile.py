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
"""

from __future__ import annotations

from pydantic import BaseModel

from matchlab_core.schemas import BallObservation, PossessorFrame, Tracklet
from matchlab_core.stages.possession.heuristic_image import Params

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

    return PossessorLabelProfile(
        total_frames=total_frames,
        asserted_frames=len(asserted),
        coverage=_fraction(len(asserted), total_frames),
        abstention=abstention,
    )
