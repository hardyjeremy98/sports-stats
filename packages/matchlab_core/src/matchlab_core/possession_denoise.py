"""Possession denoising by Viterbi decode (B3, Notion development-path step 2).

The SPO-79 heuristic decides each frame independently and then applies a
windowed-majority smoother. Both are wrong for a signal whose defining property
is persistence: a 3px geometry wobble mid-hold becomes a possession change, and
a label-only smoother cannot tell a noise flip from a pass because it never sees
the ball, the touch or the teams.

This module keeps the heuristic's evidence *exactly* -- same geometry, same
confidence -- and changes only the temporal model: a first-order HMM over
{LOOSE} + nearby candidates, decoded by Viterbi, whose transition costs carry
three physical priors (a touch must corroborate a switch; the ball must actually
travel between two different holders; team flips are rarer than same-team
passes).

Deliberately NOT modelled: pitch-space reachability and the 13-way tactical
role. Both need pitch coordinates, SNMOT carries no pitch keypoints, and an
unmeasurable prior is how a system comes to look better without being better.
See docs/superpowers/specs/2026-07-27-possession-denoise-design.md.

Like the heuristic, this REQUIRES the ball: the trellis has one column per ball
observation, so a clip with no ball yields an empty timeline. Dropping the ball
dependency is the learned estimator's job, not the denoiser's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from matchlab_core.ball_kinematics import Params as KinematicsParams
from matchlab_core.ball_kinematics import detect_touches
from matchlab_core.schemas import (
    BallObservation,
    PossessorFrame,
    Team,
    TeamAssignment,
    Tracklet,
)
from matchlab_core.stages.possession.ranking import index_possessor_boxes, rank_candidates

_EPS = 1e-9

LOOSE: int | None = None


class DenoiseParams(BaseModel):
    # --- emission ---
    possession_radius_px: float = 60.0
    max_candidates: int = 4
    distance_weight: float = 1.0
    confidence_weight: float = 0.3
    loose_cost: float = 1.2
    interpolated_ball_weight: float = 0.75
    # --- transition ---
    switch_cost: float = 2.0
    touch_bonus: float = 1.5
    no_touch_penalty: float = 0.75
    touch_tolerance_frames: int = 4
    min_travel_px: float = 8.0
    travel_window_frames: int = 3
    no_travel_penalty: float = 2.0
    team_flip_penalty: float = 1.0


@dataclass(frozen=True)
class FrameStates:
    """One trellis column. `states[0]` is always LOOSE (None)."""

    frame_idx: int
    t: float
    states: tuple[int | None, ...]
    costs: tuple[float, ...]
    confidences: tuple[float, ...]
    margin: float


@dataclass(frozen=True)
class TransitionContext:
    """Everything the transition costs need, precomputed once per clip."""

    touch_frames: frozenset[int]
    travel_px: Mapping[int, float]
    team_by_tid: Mapping[int, Team]


def build_trellis(
    ball: list[BallObservation],
    tracklets: list[Tracklet],
    params: DenoiseParams,
) -> list[FrameStates]:
    """One column per ball observation, ordered by frame_idx."""
    boxes_by_frame = index_possessor_boxes(tracklets)
    columns: list[FrameStates] = []
    for obs in sorted(ball, key=lambda b: b.frame_idx):
        all_ranked = rank_candidates(obs, boxes_by_frame.get(obs.frame_idx, []))
        ranked = [c for c in all_ranked if c.distance <= params.possession_radius_px][
            : params.max_candidates
        ]

        # Margin is defined exactly as heuristic_image.py defines it, so the two
        # impls' `margin` fields stay comparable: it describes the INPUT geometry,
        # not the decision.
        if all_ranked:
            d0 = all_ranked[0].distance
            d1 = (
                all_ranked[1].distance
                if len(all_ranked) > 1
                else d0 + params.possession_radius_px
            )
            margin = d1 - d0
        else:
            margin = 0.0

        ball_conf = obs.confidence * (
            params.interpolated_ball_weight if obs.interpolated else 1.0
        )
        states: list[int | None] = [LOOSE]
        costs: list[float] = [params.loose_cost]
        confs: list[float] = [0.0]
        for c in ranked:
            conf = min(1.0, ball_conf * c.box_confidence)
            states.append(c.tracklet_id)
            costs.append(
                params.distance_weight * (c.distance / max(_EPS, params.possession_radius_px))
                - params.confidence_weight * math.log(max(_EPS, conf))
            )
            confs.append(round(conf, 4))
        columns.append(
            FrameStates(
                frame_idx=obs.frame_idx,
                t=obs.t,
                states=tuple(states),
                costs=tuple(costs),
                confidences=tuple(confs),
                margin=round(margin, 3),
            )
        )
    return columns


def transition_cost(
    prev_state: int | None,
    cur_state: int | None,
    frame_idx: int,
    ctx: TransitionContext,
    params: DenoiseParams,
) -> float:
    """Cost of moving from `prev_state` to `cur_state` entering `frame_idx`."""
    if prev_state == cur_state:
        return 0.0
    cost = params.switch_cost
    near_touch = any(
        abs(frame_idx - f) <= params.touch_tolerance_frames for f in ctx.touch_frames
    )
    cost += -params.touch_bonus if near_touch else params.no_touch_penalty
    if prev_state is not None and cur_state is not None:
        # Unknown displacement (clip edges, or a caller that supplied none) is
        # neutral -- inf, never 0.0. Treating "we don't know" as "the ball didn't
        # move" would silently veto switches near every clip boundary.
        if ctx.travel_px.get(frame_idx, float("inf")) < params.min_travel_px:
            cost += params.no_travel_penalty
        prev_team = ctx.team_by_tid.get(prev_state, Team.UNKNOWN)
        cur_team = ctx.team_by_tid.get(cur_state, Team.UNKNOWN)
        if (
            prev_team is not Team.UNKNOWN
            and cur_team is not Team.UNKNOWN
            and prev_team is not cur_team
        ):
            cost += params.team_flip_penalty
    return max(0.0, cost)


def viterbi_decode(
    trellis: list[FrameStates],
    ctx: TransitionContext,
    params: DenoiseParams,
) -> list[int | None]:
    """Min-cost state path. Ties break toward the lower-indexed state, and
    `build_trellis` orders states by (distance, tracklet_id), so the tie-break
    matches `rank_candidates`."""
    if not trellis:
        return []

    best = list(trellis[0].costs)
    back: list[list[int]] = []
    for col_idx in range(1, len(trellis)):
        col = trellis[col_idx]
        prev_col = trellis[col_idx - 1]
        new_best: list[float] = []
        ptrs: list[int] = []
        for j, state in enumerate(col.states):
            reach = [
                best[i] + transition_cost(prev_state, state, col.frame_idx, ctx, params)
                for i, prev_state in enumerate(prev_col.states)
            ]
            arg = min(range(len(reach)), key=lambda i: (reach[i], i))
            new_best.append(reach[arg] + col.costs[j])
            ptrs.append(arg)
        best = new_best
        back.append(ptrs)

    last = min(range(len(best)), key=lambda i: (best[i], i))
    path = [last]
    for ptrs in reversed(back):
        last = ptrs[last]
        path.append(last)
    path.reverse()
    return [col.states[i] for col, i in zip(trellis, path)]


def _ball_travel(ball: list[BallObservation], window: int) -> dict[int, float]:
    """frame_idx -> straight-line ball displacement across +/- `window` frames."""
    obs = sorted(ball, key=lambda b: b.frame_idx)
    by_idx = {o.frame_idx: o for o in obs}
    travel: dict[int, float] = {}
    for o in obs:
        lo = by_idx.get(o.frame_idx - window)
        hi = by_idx.get(o.frame_idx + window)
        if lo is None or hi is None:
            travel[o.frame_idx] = float("inf")  # unknown -> never penalise
            continue
        travel[o.frame_idx] = math.hypot(hi.xy.x - lo.xy.x, hi.xy.y - lo.xy.y)
    return travel


def denoise_possession(
    tracklets: list[Tracklet],
    teams: list[TeamAssignment],
    ball: list[BallObservation],
    *,
    params: DenoiseParams | None = None,
    kinematics: KinematicsParams | None = None,
) -> list[PossessorFrame]:
    """Full pipeline: trellis -> priors -> decode -> PossessorFrame timeline."""
    p = params or DenoiseParams()
    trellis = build_trellis(ball, tracklets, p)
    if not trellis:
        return []

    touches = detect_touches(ball, tracklets, kinematics or KinematicsParams())
    ctx = TransitionContext(
        touch_frames=frozenset(t.frame_idx for t in touches),
        travel_px=_ball_travel(ball, p.travel_window_frames),
        team_by_tid={t.tracklet_id: t.team for t in teams},
    )
    labels = viterbi_decode(trellis, ctx, p)

    out: list[PossessorFrame] = []
    for col, tid in zip(trellis, labels):
        idx = col.states.index(tid)
        out.append(
            PossessorFrame(
                frame_idx=col.frame_idx,
                t=round(col.t, 3),
                possessor_tracklet_id=tid,
                team=ctx.team_by_tid.get(tid, Team.UNKNOWN) if tid is not None else Team.UNKNOWN,
                confidence=col.confidences[idx],
                margin=col.margin,
            )
        )
    return out
