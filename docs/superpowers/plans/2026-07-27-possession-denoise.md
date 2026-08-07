# Possession Denoising (Viterbi) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-frame argmin + majority smoothing in the possession signal
with a first-order HMM decoded by Viterbi, carrying touch, ball-travel and
team-flip priors, exposed as a second impl in the existing possession slot.

**Architecture:** A pure decoder module (`possession_denoise.py`) builds a
trellis from the same geometry the heuristic uses and decodes it; a thin stage
(`stages/possession/viterbi.py`) registers it under `possession-viterbi`.
Nothing downstream of `transition_to_events` changes, so the ablation against
`possession-heuristic-image` is a config swap.

**Tech Stack:** Python 3.12, pydantic v2, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-27-possession-denoise-design.md`

## Global Constraints

- Line length 100 (`uv run ruff check packages` must pass).
- `from __future__ import annotations` at the top of every new module.
- Shared ball→player geometry comes from `stages/possession/ranking.py`
  (`index_possessor_boxes`, `rank_candidates`) — never reimplemented.
- Confidence on an asserted frame is `min(1.0, ball_conf · box_conf · weight)`,
  identical to `heuristic_image.py`, so the two impls stay comparable.
- `Team.UNKNOWN` is neutral evidence — never penalised (ADR 003).
- Deterministic tie-breaking by tracklet id, matching `rank_candidates`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
  ```
- Work happens on `worktree-spo-action-spotting-prd`. Never on `main`.

## File Structure

| File | Responsibility |
|---|---|
| `packages/matchlab_core/src/matchlab_core/possession_denoise.py` | `DenoiseParams`, `FrameStates`, `TransitionContext`, trellis build, Viterbi decode, `denoise_possession` entry point |
| `packages/matchlab_core/tests/test_possession_denoise.py` | Unit tests for all of the above |
| `packages/matchlab_core/src/matchlab_core/stages/possession/viterbi.py` | `@register(StageKind.POSSESSION, "possession-viterbi")` |
| `packages/matchlab_core/tests/test_possession_viterbi_stage.py` | Registry + artifact-shape + impl-differ tests |
| `packages/matchlab_core/src/matchlab_core/stages/__init__.py` | Import the new module so it registers |
| `configs/pipeline.possession-viterbi-smoke.yaml` | Smoke config |
| `packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py` | `_possession_timeline` dispatch so crossval/localization can select the estimator |
| `packages/matchlab_train/src/matchlab_train/cli.py` | `--estimator` on `crossval-events`; `--signal possession-viterbi` on `spot-localization` |

---

### Task 1: Trellis and base Viterbi decode

Emissions and switch cost only — no priors. This task alone must already remove
single-frame flips, which is the core claim.

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/possession_denoise.py`
- Test: `packages/matchlab_core/tests/test_possession_denoise.py`

**Interfaces:**
- Consumes: `matchlab_core.stages.possession.ranking.{index_possessor_boxes, rank_candidates, PossessorCandidate}`; `matchlab_core.schemas.{BallObservation, PossessorFrame, Team, TeamAssignment, Tracklet}`
- Produces:
  - `DenoiseParams` (pydantic model, fields listed in Step 3)
  - `FrameStates` frozen dataclass: `frame_idx: int`, `t: float`, `states: tuple[int | None, ...]`, `costs: tuple[float, ...]`, `confidences: tuple[float, ...]`, `margin: float` — `states[0]` is always `None` (LOOSE)
  - `build_trellis(ball, boxes_by_frame, params) -> list[FrameStates]`
  - `viterbi_decode(trellis, ctx, params) -> list[int | None]`
  - `denoise_possession(tracklets, teams, ball, *, params=None, kinematics=None) -> list[PossessorFrame]`

- [ ] **Step 1: Write the failing tests**

Create `packages/matchlab_core/tests/test_possession_denoise.py`:

```python
from __future__ import annotations

from matchlab_core.possession_denoise import (
    DenoiseParams,
    TransitionContext,
    build_trellis,
    denoise_possession,
    viterbi_decode,
)
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)

# Geometry note: boxes are 10px wide, so a player at x sits in [x, x+10] and the
# ball at BALL_X=15 is OUTSIDE every box used below. Overlapping the ball would
# make several candidates distance 0, and rank_candidates' (distance, id)
# tie-break would decide the test instead of the model.
BALL_X = 15.0


def _tracklet(tid: int, xs: list[float], cls=DetectionClass.PLAYER) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(
                frame_idx=i,
                box=Box(x1=x, y1=0.0, x2=x + 10.0, y2=20.0),
                confidence=0.9,
            )
            for i, x in enumerate(xs)
        ],
    )


def _held(tid: int, x: float, n: int = 21, **kw) -> Tracklet:
    return _tracklet(tid, [x] * n, **kw)


def _ball(xs: list[float], y: float = 10.0) -> list[BallObservation]:
    return [
        BallObservation(frame_idx=i, t=i / 25.0, xy=Point(x=x, y=y), confidence=0.9)
        for i, x in enumerate(xs)
    ]


def _static_ball(n: int = 21, x: float = BALL_X) -> list[BallObservation]:
    return _ball([x] * n)


def _team(tid: int, team: Team) -> TeamAssignment:
    return TeamAssignment(tracklet_id=tid, team=team, confidence=0.9)


def _ctx(teams: list[TeamAssignment] | None = None) -> TransitionContext:
    return TransitionContext(
        touch_frames=frozenset(),
        travel_px={},
        team_by_tid={t.tracklet_id: t.team for t in (teams or [])},
    )


def _flip_inputs():
    """tid=1 holds throughout (dist 5px); tid=2 edges it on frame 10 only (2px)."""
    holder = _held(1, 0.0)
    rival_xs = [40.0] * 21
    rival_xs[10] = 17.0
    return [holder, _tracklet(2, rival_xs)], _static_ball()


def test_trellis_has_one_column_per_ball_observation():
    trellis = build_trellis(_static_ball(5), [_held(1, 0.0, n=5)], DenoiseParams())
    assert [c.frame_idx for c in trellis] == [0, 1, 2, 3, 4]


def test_loose_is_always_state_zero():
    trellis = build_trellis(_static_ball(3), [_held(1, 0.0, n=3)], DenoiseParams())
    assert all(col.states[0] is None for col in trellis)


def test_column_with_no_candidate_in_radius_is_loose_only():
    trellis = build_trellis(_static_ball(3), [_held(1, 5000.0, n=3)], DenoiseParams())
    assert all(col.states == (None,) for col in trellis)


def test_empty_ball_yields_empty_timeline():
    assert denoise_possession([_held(1, 0.0)], [], []) == []


def test_single_frame_flip_is_removed():
    tracklets, ball = _flip_inputs()
    params = DenoiseParams()
    labels = viterbi_decode(build_trellis(ball, tracklets, params), _ctx(), params)
    assert set(labels) == {1}


def test_zero_switch_cost_reproduces_per_frame_argmin():
    tracklets, ball = _flip_inputs()
    params = DenoiseParams(switch_cost=0.0, no_touch_penalty=0.0)
    labels = viterbi_decode(build_trellis(ball, tracklets, params), _ctx(), params)
    assert labels[10] == 2
    assert labels[0] == 1


def test_decode_is_deterministic_under_equal_costs():
    params = DenoiseParams()
    tracklets = [_held(1, 0.0, n=5), _held(2, 0.0, n=5)]
    labels = viterbi_decode(build_trellis(_static_ball(5), tracklets, params), _ctx(), params)
    assert set(labels) == {1}


def test_denoise_returns_one_possessor_frame_per_ball_observation():
    out = denoise_possession(
        [_held(1, 0.0, n=6)], [_team(1, Team.HOME)], _static_ball(6)
    )
    assert [f.frame_idx for f in out] == [0, 1, 2, 3, 4, 5]
    assert all(f.possessor_tracklet_id == 1 for f in out)
    assert all(f.team is Team.HOME for f in out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_denoise.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_core.possession_denoise'`

- [ ] **Step 3: Write the module**

Create `packages/matchlab_core/src/matchlab_core/possession_denoise.py`:

```python
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

Deliberately NOT modelled: pitch-space reachability and tactical role. Both need
pitch coordinates, SNMOT carries no pitch keypoints, and an unmeasurable prior
is how a system comes to look better without being better. See the spec.
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
            candidates = [
                best[i]
                + transition_cost(prev_state, state, col.frame_idx, ctx, params)
                for i, prev_state in enumerate(prev_col.states)
            ]
            arg = min(range(len(candidates)), key=lambda i: (candidates[i], i))
            new_best.append(candidates[arg] + col.costs[j])
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_denoise.py -q`
Expected: PASS (8 tests)

If `test_single_frame_flip_is_removed` fails, the switch cost is too low
relative to the emission gap — check that `no_touch_penalty` is being added
(the synthetic ball is static, so `detect_touches` finds nothing and
`viterbi_decode` should see an empty `touch_frames`).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/possession_denoise.py \
        packages/matchlab_core/tests/test_possession_denoise.py
git commit -m "feat(possession): Viterbi trellis + decode over the heuristic signal (B3)"
```

---

### Task 2: Prior behaviour and ablation switches

Task 1 wired the priors; this task proves each one does what it claims and that
zeroing it is a clean ablation.

**Files:**
- Modify: `packages/matchlab_core/tests/test_possession_denoise.py` (append)

**Interfaces:**
- Consumes: everything Task 1 produced.
- Produces: nothing new — behavioural tests only.

- [ ] **Step 1: Write the failing tests**

Append to `packages/matchlab_core/tests/test_possession_denoise.py`:

```python
def _switching_inputs():
    """tid=1 holds frames 0-9, tid=2 holds 10-20 — a real change of holder.

    The ball moves 40px between the two holders, so this is the case the travel
    prior must ALLOW, not the case it must veto.
    """
    tracklets = [_held(1, 0.0), _held(2, 40.0)]
    ball = _ball([5.0 if i < 10 else 45.0 for i in range(21)])
    return tracklets, ball


def test_corroborated_switch_survives():
    """The disconfirming test: a real pass must NOT be smoothed away."""
    tracklets, ball = _switching_inputs()
    params = DenoiseParams()
    ctx = TransitionContext(
        touch_frames=frozenset({10}),
        travel_px={i: 100.0 for i in range(21)},
        team_by_tid={},
    )
    labels = viterbi_decode(build_trellis(ball, tracklets, params), ctx, params)
    assert labels[0] == 1
    assert labels[-1] == 2


def test_uncorroborated_switch_costs_more_than_corroborated():
    params = DenoiseParams()
    ctx_touch = TransitionContext(frozenset({10}), {10: 100.0}, {})
    ctx_none = TransitionContext(frozenset(), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, ctx_touch, params) < transition_cost(
        1, 2, 10, ctx_none, params
    )


def test_switch_without_ball_travel_is_penalised():
    params = DenoiseParams()
    moved = TransitionContext(frozenset({10}), {10: 100.0}, {})
    still = TransitionContext(frozenset({10}), {10: 0.0}, {})
    assert transition_cost(1, 2, 10, still, params) > transition_cost(1, 2, 10, moved, params)


def test_team_flip_costs_more_than_same_team():
    params = DenoiseParams()
    same = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.HOME})
    flip = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.AWAY})
    assert transition_cost(1, 2, 10, flip, params) > transition_cost(1, 2, 10, same, params)


def test_unknown_team_is_neutral():
    """ADR 003: missing evidence is neutral, never penalised."""
    params = DenoiseParams()
    known = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.HOME})
    unknown = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME})
    assert transition_cost(1, 2, 10, unknown, params) == transition_cost(1, 2, 10, known, params)


def test_each_prior_at_zero_is_a_clean_ablation():
    params = DenoiseParams(touch_bonus=0.0, no_touch_penalty=0.0)
    ctx_touch = TransitionContext(frozenset({10}), {10: 100.0}, {})
    ctx_none = TransitionContext(frozenset(), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, ctx_touch, params) == transition_cost(
        1, 2, 10, ctx_none, params
    )

    params = DenoiseParams(no_travel_penalty=0.0)
    still = TransitionContext(frozenset({10}), {10: 0.0}, {})
    moved = TransitionContext(frozenset({10}), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, still, params) == transition_cost(1, 2, 10, moved, params)

    params = DenoiseParams(team_flip_penalty=0.0)
    flip = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.AWAY})
    same = TransitionContext(frozenset({10}), {10: 100.0}, {1: Team.HOME, 2: Team.HOME})
    assert transition_cost(1, 2, 10, flip, params) == transition_cost(1, 2, 10, same, params)


def test_travel_prior_never_penalises_unknown_displacement():
    """Clip edges have no +/-window neighbour; unknown must not be treated as still."""
    from matchlab_core.possession_denoise import _ball_travel

    travel = _ball_travel(_static_ball(5), window=3)
    assert travel[0] == float("inf")
    assert travel[4] == float("inf")


def test_transition_cost_is_never_negative():
    params = DenoiseParams(touch_bonus=99.0)
    ctx = TransitionContext(frozenset({10}), {10: 100.0}, {})
    assert transition_cost(1, 2, 10, ctx, params) == 0.0
```

Add `transition_cost` to the import at the top of the test file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_denoise.py -q`
Expected: FAIL — `ImportError: cannot import name 'transition_cost'` until the
import line is added; then all 9 new tests should pass against Task 1's module.
If any *behavioural* test fails, fix the module, not the test.

- [ ] **Step 3: Fix whatever the tests expose**

No new code is planned here. If `test_corroborated_switch_survives` fails, the
priors are too strong — that is the spec's stated risk, and the fix is to lower
`switch_cost`, never to weaken the test.

- [ ] **Step 4: Run the full core suite**

Run: `uv run pytest packages/matchlab_core -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/tests/test_possession_denoise.py
git commit -m "test(possession): prior behaviour + per-prior ablation switches (B3)"
```

---

### Task 3: The `possession-viterbi` stage

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/stages/possession/viterbi.py`
- Create: `packages/matchlab_core/tests/test_possession_viterbi_stage.py`
- Modify: `packages/matchlab_core/src/matchlab_core/stages/__init__.py`
- Create: `configs/pipeline.possession-viterbi-smoke.yaml`

**Interfaces:**
- Consumes: `denoise_possession`, `DenoiseParams` from Task 1.
- Produces: registry entry `("possession", "possession-viterbi")`.

- [ ] **Step 1: Write the failing tests**

Create `packages/matchlab_core/tests/test_possession_viterbi_stage.py`:

```python
from __future__ import annotations

import matchlab_core.stages  # noqa: F401  -- registers implementations
from matchlab_core.registry import get_impl
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.schemas.run import StageKind

BALL_X = 15.0


def _tracklet(tid: int, xs: list[float]) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(
                frame_idx=i,
                box=Box(x1=x, y1=0.0, x2=x + 10.0, y2=20.0),
                confidence=0.9,
            )
            for i, x in enumerate(xs)
        ],
    )


def _ball(n: int) -> list[BallObservation]:
    return [
        BallObservation(frame_idx=i, t=i / 25.0, xy=Point(x=BALL_X, y=10.0), confidence=0.9)
        for i in range(n)
    ]


def _flip_inputs():
    rival_xs = [40.0] * 21
    rival_xs[10] = 17.0  # tid=2 edges tid=1 on one frame only
    return [_tracklet(1, [0.0] * 21), _tracklet(2, rival_xs)], _ball(21)


def test_stage_is_registered():
    assert get_impl(StageKind.POSSESSION, "possession-viterbi") is not None


def test_stage_returns_a_possessor_frame_per_ball_observation():
    impl = get_impl(StageKind.POSSESSION, "possession-viterbi")()
    tracklets = [_tracklet(1, [0.0] * 6)]
    teams = [TeamAssignment(tracklet_id=1, team=Team.HOME, confidence=0.9)]
    out = impl.estimate(None, tracklets, teams, _ball(6))
    assert [f.frame_idx for f in out] == [0, 1, 2, 3, 4, 5]


def test_stage_accepts_params():
    impl = get_impl(StageKind.POSSESSION, "possession-viterbi")(switch_cost=0.0)
    assert impl.params.switch_cost == 0.0


def test_viterbi_and_heuristic_differ_on_a_single_frame_flip():
    """The whole point of the ablation: identical inputs, different timelines."""
    heuristic = get_impl(StageKind.POSSESSION, "possession-heuristic-image")(smooth_radius=0)
    viterbi = get_impl(StageKind.POSSESSION, "possession-viterbi")()
    tracklets, ball = _flip_inputs()
    teams = [TeamAssignment(tracklet_id=t, team=Team.HOME, confidence=0.9) for t in (1, 2)]
    h = heuristic.estimate(None, tracklets, teams, ball)
    v = viterbi.estimate(None, tracklets, teams, ball)
    assert h[10].possessor_tracklet_id == 2
    assert v[10].possessor_tracklet_id == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_viterbi_stage.py -q`
Expected: FAIL — the registry has no `possession-viterbi`.

- [ ] **Step 3: Write the stage**

Create `packages/matchlab_core/src/matchlab_core/stages/possession/viterbi.py`:

```python
"""Viterbi-decoded possessor estimator (B3, Notion development-path step 2).

Same slot and same evidence as `possession-heuristic-image` (SPO-79) -- the only
difference is the temporal model, which is the point: swapping the two impls in
a config is a controlled ablation of per-frame argmin + majority smoothing
against a first-order HMM with tactical priors. All the modelling lives in
`matchlab_core.possession_denoise`; this file is only the registry adapter.
"""

from __future__ import annotations

from matchlab_core.ball_kinematics import Params as KinematicsParams
from matchlab_core.interfaces import PossessionEstimator, StageContext
from matchlab_core.possession_denoise import DenoiseParams, denoise_possession
from matchlab_core.registry import register
from matchlab_core.schemas import (
    BallObservation,
    PossessorFrame,
    TeamAssignment,
    Tracklet,
)
from matchlab_core.schemas.run import StageKind


@register(StageKind.POSSESSION, "possession-viterbi")
class ViterbiPossession(PossessionEstimator):
    def __init__(self, **params):
        kinematics = params.pop("kinematics", None) or {}
        self.params = DenoiseParams(**params)
        self.kinematics = KinematicsParams(**kinematics)

    def estimate(
        self,
        ctx: StageContext,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
        ball: list[BallObservation],
    ) -> list[PossessorFrame]:
        return denoise_possession(
            tracklets,
            teams,
            ball,
            params=self.params,
            kinematics=self.kinematics,
        )
```

- [ ] **Step 4: Register it**

In `packages/matchlab_core/src/matchlab_core/stages/__init__.py`, change:

```python
from matchlab_core.stages.possession import (  # noqa: F401
    heuristic_image as possession_heuristic_image,
)
from matchlab_core.stages.possession import none_stub as possession_none  # noqa: F401
```

to:

```python
from matchlab_core.stages.possession import (  # noqa: F401
    heuristic_image as possession_heuristic_image,
)
from matchlab_core.stages.possession import none_stub as possession_none  # noqa: F401
from matchlab_core.stages.possession import viterbi as possession_viterbi  # noqa: F401
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_viterbi_stage.py -q`
Expected: PASS (4 tests)

- [ ] **Step 6: Add the smoke config**

Create `configs/pipeline.possession-viterbi-smoke.yaml` — identical to
`configs/pipeline.possession-heuristic-smoke.yaml` except for the header comment
and the `possession:` block:

```yaml
  possession:
    impl: possession-viterbi
    params:
      possession_radius_px: 60.0
      switch_cost: 2.0
      touch_bonus: 1.5
      no_travel_penalty: 2.0
      team_flip_penalty: 1.0
```

Set `name: possession-viterbi-smoke` and describe it as the denoised counterpart
of the heuristic smoke config.

- [ ] **Step 7: Full suite, lint, commit**

```bash
uv run pytest packages -q
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/stages/possession/viterbi.py \
        packages/matchlab_core/tests/test_possession_viterbi_stage.py \
        packages/matchlab_core/src/matchlab_core/stages/__init__.py \
        configs/pipeline.possession-viterbi-smoke.yaml
git commit -m "feat(possession): possession-viterbi stage + smoke config (B3)"
```

---

### Task 4: Measurement — ablation across both benchmarks

The two impls must be runnable through the *same* crossval and localisation
drivers, or the comparison is not controlled.

**Files:**
- Modify: `packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py`
- Modify: `packages/matchlab_train/src/matchlab_train/cli.py`
- Test: `packages/matchlab_train/tests/test_possessor_audit.py` (append)

**Interfaces:**
- Consumes: `denoise_possession` (Task 1), the `possession-viterbi` registry entry (Task 3).
- Produces: `_possession_timeline(estimator, tracklets, teams, ball, **params)` used by both `crossval_sequence` and `localize_soccernet_tracking`; `estimator` recorded on `CrossvalReport` and `SpottingLocalizationReport`.

- [ ] **Step 1: Write the failing tests**

Append to `packages/matchlab_train/tests/test_possessor_audit.py`:

```python
def test_possession_timeline_dispatches_to_both_estimators():
    from matchlab_train.datasets.possessor_audit import _possession_timeline

    tracklets, teams, ball = _simple_inputs()  # reuse the module's existing fixture helper
    h = _possession_timeline("possession-heuristic-image", tracklets, teams, ball)
    v = _possession_timeline("possession-viterbi", tracklets, teams, ball)
    assert len(h) == len(v)


def test_possession_timeline_rejects_unknown_estimator():
    import pytest

    from matchlab_train.datasets.possessor_audit import _possession_timeline

    tracklets, teams, ball = _simple_inputs()
    with pytest.raises(ValueError, match="unknown estimator"):
        _possession_timeline("nope", tracklets, teams, ball)
```

If `_simple_inputs` does not already exist in that test module, add it building
two tracklets and a static ball exactly as `test_possession_denoise.py` does.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/matchlab_train/tests/test_possessor_audit.py -q`
Expected: FAIL — `_possession_timeline` does not exist.

- [ ] **Step 3: Add the dispatch**

In `possessor_audit.py`, add near the other helpers:

```python
DEFAULT_ESTIMATOR = "possession-heuristic-image"


def _possession_timeline(
    estimator: str,
    tracklets: list[Tracklet],
    teams: list[TeamAssignment],
    ball: list[BallObservation],
    **params,
) -> list[PossessorFrame]:
    """Run one possession estimator on oracle inputs. Both impls share the slot
    interface, so the ablation swaps only this string."""
    if estimator in (DEFAULT_ESTIMATOR, "possession"):
        return HeuristicImagePossession(**params).estimate(None, tracklets, teams, ball)
    if estimator == "possession-viterbi":
        return denoise_possession(tracklets, teams, ball, params=DenoiseParams(**params))
    raise ValueError(f"unknown estimator: {estimator!r}")
```

Import `denoise_possession` and `DenoiseParams` from `matchlab_core.possession_denoise`.

- [ ] **Step 4: Route both drivers through it**

In `crossval_sequence`, replace

```python
    timeline = HeuristicImagePossession(**params).estimate(None, tracklets, teams, ball)
```

with

```python
    timeline = _possession_timeline(estimator, tracklets, teams, ball, **params)
```

and add `estimator: str = DEFAULT_ESTIMATOR` to the keyword-only parameters of
`crossval_sequence`, `crossval_sequences` and `crossval_soccernet_tracking`,
threading it through. Add an `estimator: str` field to `CrossvalReport` and set
it in `crossval_sequences`.

In `localize_soccernet_tracking`, replace

```python
        if signal == "possession":
            timeline = HeuristicImagePossession(**params).estimate(None, tracklets, teams, ball)
            frames = [e.frame_idx for e in transition_to_events(timeline)]
```

with

```python
        if signal in ("possession", "possession-viterbi"):
            estimator = (
                DEFAULT_ESTIMATOR if signal == "possession" else "possession-viterbi"
            )
            timeline = _possession_timeline(estimator, tracklets, teams, ball, **params)
            frames = [e.frame_idx for e in transition_to_events(timeline)]
```

and extend the docstring to name the third signal.

- [ ] **Step 5: Expose it on the CLI**

In `packages/matchlab_train/src/matchlab_train/cli.py`:
- add `--estimator` (default `possession-heuristic-image`,
  choices `possession-heuristic-image`/`possession-viterbi`) to the
  `crossval-events` parser and pass it through;
- extend the `spot-localization` `--signal` choices to include
  `possession-viterbi`.

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest packages -q
uv run ruff check packages
```
Expected: PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py \
        packages/matchlab_train/src/matchlab_train/cli.py \
        packages/matchlab_train/tests/test_possessor_audit.py
git commit -m "feat(b3): estimator dispatch so crossval/localisation can ablate the denoiser"
```

---

### Task 5: Run the ablation and write the report

**Files:**
- Create: `docs/reports/2026-07-27-b3-possession-denoise-ablation.md`
- Modify: `docs/implementation-status.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: the CLI surface from Task 4.
- Produces: measured numbers; no code.

- [ ] **Step 1: Run the crossval arm, both estimators**

```bash
uv run matchlab-train crossval-events --root data/soccernet/tracking/test \
  --estimator possession-heuristic-image --out /tmp/crossval-heuristic.json
uv run matchlab-train crossval-events --root data/soccernet/tracking/test \
  --estimator possession-viterbi --out /tmp/crossval-viterbi.json
```

Baseline to beat: **agreement 71.4% @ ±1 s**.

- [ ] **Step 2: Run the localisation guard, both signals**

```bash
uv run matchlab-train spot-localization --signal possession --out /tmp/loc-heuristic.json
uv run matchlab-train spot-localization --signal possession-viterbi --out /tmp/loc-viterbi.json
```

Baseline to hold: **3.0 frames median on ball-contact classes**. Recall that
`snmot_localization_error` scores the *nearest* prediction, so this number can
only rise or hold when predictions are removed — it is a guard against deleting
real events, never evidence of improvement.

- [ ] **Step 3: Run the per-prior ablation**

Four crossval runs with `possession-viterbi`, varying one parameter each. If the
CLI does not accept arbitrary denoise params, drive them from a short script
under the scratchpad rather than widening the CLI surface:

| Row | Change |
|---|---|
| all on | defaults |
| no touch | `touch_bonus=0, no_touch_penalty=0` |
| no travel | `no_travel_penalty=0` |
| no team flip | `team_flip_penalty=0` |

- [ ] **Step 4: Run the segment-statistics arm**

```bash
uv run matchlab-train audit-possessor-labels --root data/soccernet/tracking/test \
  --out /tmp/audit-heuristic.json
```

Compare against the same profile computed on the denoised timeline. Baselines:
**1,133 segments, mean 19.1 frames, 42 implausible team flips.**

- [ ] **Step 5: Write the report**

Create `docs/reports/2026-07-27-b3-possession-denoise-ablation.md` carrying:
the code revision, the tier and sequence count, the three measured arms, the
four-row prior ablation, and an explicit statement of what is **not** claimed
(possessor accuracy — no per-frame possessor GT exists on any tier).

If agreement rose while localisation degraded, report that as a **negative**
finding and scope it to the decision rule tested, per the standing lesson. Do
not tune the parameters until the metric agrees.

- [ ] **Step 6: Update the governed docs**

- `docs/implementation-status.md`: add `possession-viterbi` to the possession
  slot, with its measured numbers and the acquisition caveat.
- `CLAUDE.md`: add the ablation commands under the possession-transition block.

- [ ] **Step 7: Commit**

```bash
git add docs/
git commit -m "docs(b3): possession-denoise ablation report + status update"
```
