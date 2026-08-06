# Weak Possessor-Label Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close SPO-83 acceptance criterion 2 — assess weak possessor-label quality, including the false-possession contamination proxy — with real evidence from the local SoccerNet-tracking GT (boxes + teams + ball), replacing a deferral.

**Architecture:** Extract the ball→player candidate ranking out of `heuristic_image.py` into a shared module so the estimator and a new profiler share one geometry implementation. Add a pure, impl-agnostic label-risk profiler in `matchlab_core` that turns a possessor timeline plus its inputs into counts and threshold-swept curves. Add a GT-driven audit driver in `matchlab_train` that builds oracle tracklets/teams/ball from `GroundTruth`, runs the estimator, profiles each sequence, applies a declared exclusion rule, and aggregates.

**Tech Stack:** Python 3.12 (pinned via `.python-version`), pydantic v2, pytest, `uv` for env management. No new dependencies. No GPU, no model weights, no video decode.

**Spec:** [`docs/superpowers/specs/2026-07-25-possessor-label-audit-design.md`](../specs/2026-07-25-possessor-label-audit-design.md)

## Global Constraints

- Line length 100 (ruff, config in root `pyproject.toml`). Run `uv run ruff check packages` before every commit.
- Run tests with `uv run pytest packages -q`. The dev env is `uv sync --group cv --group eval --group dev`; never run `uv sync --group X` alone (it removes other groups).
- **No accuracy claims.** There is no per-frame possessor ground truth on any tier. Every number produced here describes the label set's own structure. Docstrings, JSON field docs, and the report must not state or imply possessor accuracy.
- **`PossessorFrame` and the artifact set must not change.** No new `ArtifactName`, no edits to `web/src/lib/types.ts`, no change to `possession_timeline.json`'s shape.
- **`packages/matchlab_core/tests/test_possession_heuristic.py` must never be edited.** It is the guard that Task 1's refactor preserved estimator behaviour.
- `matchlab_train` may import `matchlab_core`; `matchlab_core` must never import `matchlab_train`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
  ```
- Working directory for all commands: `/home/jeremy/code/MatchDay/lab/.claude/worktrees/spo-action-spotting-prd` (branch `worktree-spo-action-spotting-prd`).

## File Structure

| File | Responsibility |
|---|---|
| `packages/matchlab_core/src/matchlab_core/stages/possession/ranking.py` (create) | Ball→player candidate geometry: index boxes by frame, rank candidates by distance. Sole source of truth, used by both estimator and profiler. |
| `packages/matchlab_core/src/matchlab_core/stages/possession/heuristic_image.py` (modify) | Refactored onto `ranking.py`. Behaviour unchanged. |
| `packages/matchlab_core/src/matchlab_core/possession_profile.py` (create) | Pure label-risk profiler + schema + aggregation. No I/O in the core function. |
| `packages/matchlab_core/tests/test_possession_profile.py` (create) | Unit tests for every indicator on hand-built timelines. |
| `packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py` (create) | GT → oracle inputs adapter, per-sequence audit, exclusion rule, aggregation into a report. |
| `packages/matchlab_train/tests/test_possessor_audit.py` (create) | Adapter + exclusion-rule tests on a synthetic `GroundTruth`. |
| `packages/matchlab_train/src/matchlab_train/cli.py` (modify) | `audit-possessor-labels` subcommand. |
| `docs/reports/2026-07-25-spo83-possessor-label-audit.md` (create) | Hand-written narrative over the real-data run. |
| `docs/reference/possession-transition-gate.md` (modify) | Criterion 2 now has evidence; correct the "no oracle isolation possible" claim. |

---

### Task 1: Extract candidate ranking from the estimator

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/stages/possession/ranking.py`
- Modify: `packages/matchlab_core/src/matchlab_core/stages/possession/heuristic_image.py`
- Test: `packages/matchlab_core/tests/test_possession_heuristic.py` (existing, **run only, never edit**)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PossessorCandidate` — pydantic-free dataclass with fields `tracklet_id: int`, `box: Box`, `box_confidence: float`, `distance: float`
  - `index_possessor_boxes(tracklets: list[Tracklet]) -> dict[int, list[tuple[int, Box, float]]]`
  - `rank_candidates(ball: BallObservation, boxes: list[tuple[int, Box, float]]) -> list[PossessorCandidate]`
  - `dist_point_box(pt: Point, box: Box) -> float`
  - `POSSESSOR_CLASSES: frozenset[DetectionClass]`

- [ ] **Step 1: Write the failing test**

Create `packages/matchlab_core/tests/test_possession_ranking.py`:

```python
"""SPO-83: the shared ball->player candidate geometry, extracted from the
estimator so the profiler cannot drift from it."""

from __future__ import annotations

from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.stages.possession.ranking import (
    POSSESSOR_CLASSES,
    dist_point_box,
    index_possessor_boxes,
    rank_candidates,
)


def _player(tid, frame, xyxy, conf=0.9, cls=DetectionClass.PLAYER):
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(
                frame_idx=frame,
                box=Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                confidence=conf,
            )
        ],
    )


def test_distance_is_zero_inside_the_box():
    assert dist_point_box(Point(x=10, y=20), Box(x1=0, y1=0, x2=20, y2=40)) == 0.0


def test_distance_is_edge_distance_outside_the_box():
    assert dist_point_box(Point(x=30, y=20), Box(x1=0, y1=0, x2=20, y2=40)) == 10.0


def test_index_keeps_players_and_goalkeepers_only():
    tracklets = [
        _player(1, 0, (0, 0, 20, 40)),
        _player(2, 0, (30, 0, 50, 40), cls=DetectionClass.GOALKEEPER),
        _player(9, 0, (60, 0, 80, 40), cls=DetectionClass.REFEREE),
    ]
    boxes = index_possessor_boxes(tracklets)
    assert sorted(tid for tid, _, _ in boxes[0]) == [1, 2]
    assert DetectionClass.REFEREE not in POSSESSOR_CLASSES


def test_rank_orders_by_distance_and_carries_boxes():
    tracklets = [_player(1, 0, (0, 0, 20, 40)), _player(2, 0, (100, 0, 120, 40))]
    boxes = index_possessor_boxes(tracklets)[0]
    ball = BallObservation(frame_idx=0, t=0.0, xy=Point(x=10, y=20), confidence=1.0)
    ranked = rank_candidates(ball, boxes)
    assert [c.tracklet_id for c in ranked] == [1, 2]
    assert ranked[0].distance == 0.0
    assert ranked[0].box.y2 - ranked[0].box.y1 == 40.0
    assert ranked[0].box_confidence == 0.9


def test_rank_on_empty_candidates_is_empty():
    ball = BallObservation(frame_idx=0, t=0.0, xy=Point(x=0, y=0), confidence=1.0)
    assert rank_candidates(ball, []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_ranking.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_core.stages.possession.ranking'`

- [ ] **Step 3: Write the ranking module**

Create `packages/matchlab_core/src/matchlab_core/stages/possession/ranking.py`:

```python
"""Shared ball->player candidate geometry (SPO-83).

Extracted from `heuristic_image.py` so the possessor estimator and the
label-risk profiler (`matchlab_core.possession_profile`) rank candidates with
one implementation. Two copies of this geometry would let the profiler report
on a ranking the estimator never used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from matchlab_core.schemas import BallObservation, Box, DetectionClass, Point, Tracklet

POSSESSOR_CLASSES = frozenset({DetectionClass.PLAYER, DetectionClass.GOALKEEPER})


@dataclass(frozen=True)
class PossessorCandidate:
    tracklet_id: int
    box: Box
    box_confidence: float
    distance: float


def dist_point_box(pt: Point, box: Box) -> float:
    """Euclidean distance from a point to a box; 0 when the point is inside."""
    dx = max(box.x1 - pt.x, 0.0, pt.x - box.x2)
    dy = max(box.y1 - pt.y, 0.0, pt.y - box.y2)
    return math.hypot(dx, dy)


def index_possessor_boxes(
    tracklets: list[Tracklet],
) -> dict[int, list[tuple[int, Box, float]]]:
    """frame_idx -> [(tracklet_id, box, box_confidence)] for possessor classes."""
    out: dict[int, list[tuple[int, Box, float]]] = {}
    for tr in tracklets:
        if tr.cls not in POSSESSOR_CLASSES:
            continue
        for fr in tr.frames:
            out.setdefault(fr.frame_idx, []).append((tr.tracklet_id, fr.box, fr.confidence))
    return out


def rank_candidates(
    ball: BallObservation, boxes: list[tuple[int, Box, float]]
) -> list[PossessorCandidate]:
    """Candidates sorted nearest-first by distance from the ball, ties by id."""
    candidates = [
        PossessorCandidate(
            tracklet_id=tid,
            box=box,
            box_confidence=conf,
            distance=dist_point_box(ball.xy, box),
        )
        for tid, box, conf in boxes
    ]
    candidates.sort(key=lambda c: (c.distance, c.tracklet_id))
    return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_ranking.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Refactor the estimator onto the shared module**

In `packages/matchlab_core/src/matchlab_core/stages/possession/heuristic_image.py`, delete the module-level `_POSSESSOR_CLASSES` and `_dist_point_box`, drop the now-unused `math` import, and replace the box-indexing plus per-frame ranking inside `estimate` with calls to the shared helpers. The resulting `estimate` body:

```python
    def estimate(
        self,
        ctx: StageContext,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
        ball: list[BallObservation],
    ) -> list[PossessorFrame]:
        p = self.params
        team_by_tid = {t.tracklet_id: t.team for t in teams}
        boxes_by_frame = index_possessor_boxes(tracklets)

        # Raw per-frame nearest-player possessor (with abstention), pre-smoothing.
        raw: list[tuple[int, float, int | None, float, float]] = []  # frame,t,tid,conf,margin
        for obs in sorted(ball, key=lambda b: b.frame_idx):
            ranked = rank_candidates(obs, boxes_by_frame.get(obs.frame_idx, []))
            if not ranked:
                raw.append((obs.frame_idx, obs.t, None, 0.0, 0.0))
                continue
            d0 = ranked[0].distance
            d1 = ranked[1].distance if len(ranked) > 1 else d0 + p.possession_radius_px
            margin = d1 - d0
            if d0 > p.possession_radius_px or margin < p.min_margin_px:
                raw.append((obs.frame_idx, obs.t, None, 0.0, round(margin, 3)))
                continue
            weight = p.interpolated_ball_weight if obs.interpolated else 1.0
            conf = min(1.0, obs.confidence * ranked[0].box_confidence * weight)
            raw.append((obs.frame_idx, obs.t, ranked[0].tracklet_id, round(conf, 4), round(margin, 3)))

        smoothed = _smooth([r[2] for r in raw], p.smooth_radius)
        ...  # remainder of the method is unchanged
```

Update the import block to add:

```python
from matchlab_core.stages.possession.ranking import index_possessor_boxes, rank_candidates
```

and remove `Box` and `Point` from the `matchlab_core.schemas` import if they become unused (check with ruff).

> Note the original sorted a tuple `(distance, tid, conf)`, which tie-broke by
> tracklet id. `rank_candidates` preserves that tie-break deliberately — do not
> change it, or `test_contested_margin_abstains` semantics shift.

- [ ] **Step 6: Run the estimator's existing tests unmodified**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_heuristic.py -q`
Expected: PASS, all 11 tests, **with no edits to that file**. If any test fails, the refactor changed behaviour — revert and redo Step 5. Do not adjust the test.

- [ ] **Step 7: Run the full suite and lint**

Run: `uv run pytest packages -q && uv run ruff check packages`
Expected: PASS, no lint findings.

- [ ] **Step 8: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/stages/possession/ranking.py \
        packages/matchlab_core/src/matchlab_core/stages/possession/heuristic_image.py \
        packages/matchlab_core/tests/test_possession_ranking.py
git commit -m "$(cat <<'EOF'
refactor(possession): extract shared ball->player candidate ranking (SPO-83)

The label-risk profiler needs the same candidate geometry the estimator uses.
Extracting it keeps one source of truth; test_possession_heuristic.py runs
unmodified as the behaviour guard.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 2: Profile schema + coverage and abstention breakdown

**Files:**
- Create: `packages/matchlab_core/src/matchlab_core/possession_profile.py`
- Test: `packages/matchlab_core/tests/test_possession_profile.py`

**Interfaces:**
- Consumes: `matchlab_core.stages.possession.heuristic_image.Params` (for `min_margin_px`).
- Produces:
  - `AbstentionBreakdown(no_ball_observation: int, outside_radius: int, contested_tie: int)`
  - `CurvePoint(threshold: float, count: int, fraction: float)`
  - `SegmentStats(count, total_segment_frames, mean_frames, below_te_count, below_te_fraction, changes, span_seconds, changes_per_second)`
  - `PossessorLabelProfile(total_frames, asserted_frames, coverage, abstention, contested_curve, depth_evaluable_frames, depth_discordance, segments, implausible_team_flips)`
  - `profile_possessor_labels(timeline, tracklets, ball, params, *, total_frames, fps=25.0, tau_grid_px=DEFAULT_TAU_GRID_PX, depth_ratio_grid=DEFAULT_DEPTH_RATIO_GRID, te_frames=DEFAULT_TE_FRAMES) -> PossessorLabelProfile`
  - `DEFAULT_TAU_GRID_PX = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)`
  - `DEFAULT_DEPTH_RATIO_GRID = (1.2, 1.5, 2.0)`
  - `DEFAULT_TE_FRAMES = 3`

  Tasks 3–5 fill `contested_curve`, `depth_discordance`/`depth_evaluable_frames`, and `segments`/`implausible_team_flips`. This task returns them empty/zero.

- [ ] **Step 1: Write the failing test**

Create `packages/matchlab_core/tests/test_possession_profile.py`:

```python
"""SPO-83: label-risk profiler for weak possessor labels.

NOT an accuracy measure -- no per-frame possessor ground truth exists on any
tier. Every assertion here is about the label set's own structure. Timelines are
hand-built so each indicator's expected value is known by construction.
"""

from __future__ import annotations

import pytest

from matchlab_core.possession_profile import profile_possessor_labels
from matchlab_core.schemas import PossessorFrame, Team
from matchlab_core.stages.possession.heuristic_image import Params


def _row(frame_idx, tid, margin=50.0, team=Team.HOME, conf=0.9):
    """One timeline row. tid=None means the estimator abstained."""
    return PossessorFrame(
        frame_idx=frame_idx,
        t=frame_idx / 25.0,
        possessor_tracklet_id=tid,
        team=team if tid is not None else Team.UNKNOWN,
        confidence=conf if tid is not None else 0.0,
        margin=margin,
    )


def test_coverage_counts_asserted_rows():
    timeline = [_row(0, 1), _row(1, 1), _row(2, None), _row(3, None)]
    p = profile_possessor_labels(
        timeline, [], [], Params(min_margin_px=10.0), total_frames=10
    )
    assert p.total_frames == 10
    assert p.asserted_frames == 2
    assert p.coverage == pytest.approx(0.2)


def test_abstention_causes_sum_to_all_non_asserted_frames():
    timeline = [
        _row(0, 1),                 # asserted
        _row(1, None, margin=50.0),  # abstained, margin above min -> outside radius
        _row(2, None, margin=1.0),   # abstained, margin below min -> contested tie
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(min_margin_px=10.0), total_frames=7
    )
    a = p.abstention
    assert a.no_ball_observation == 4      # 7 total - 3 rows
    assert a.outside_radius == 1
    assert a.contested_tie == 1
    assert a.no_ball_observation + a.outside_radius + a.contested_tie == 6
    assert p.asserted_frames + 6 == p.total_frames


def test_empty_timeline_is_all_no_ball():
    p = profile_possessor_labels([], [], [], Params(), total_frames=5)
    assert p.asserted_frames == 0
    assert p.coverage == 0.0
    assert p.abstention.no_ball_observation == 5


def test_total_frames_below_row_count_is_a_programming_error():
    with pytest.raises(ValueError, match="total_frames"):
        profile_possessor_labels([_row(0, 1), _row(1, 1)], [], [], Params(), total_frames=1)


def test_zero_total_frames_does_not_divide_by_zero():
    p = profile_possessor_labels([], [], [], Params(), total_frames=0)
    assert p.coverage == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_core.possession_profile'`

- [ ] **Step 3: Write the module**

Create `packages/matchlab_core/src/matchlab_core/possession_profile.py`:

```python
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
```

> The `outside_radius` / `contested_tie` split is recovered from the recorded
> `margin` rather than a new field, so `PossessorFrame` and the artifact stay
> unchanged. One consequence to state in the report: with the shipped default
> `min_margin_px=0.0` no abstention can be classified as a tie, and the
> "no candidates in frame" case (estimator records `margin=0.0`) lands in
> `outside_radius`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/possession_profile.py \
        packages/matchlab_core/tests/test_possession_profile.py
git commit -m "$(cat <<'EOF'
feat(possession): label-risk profile schema + coverage/abstention (SPO-83)

Structure of the weak label set, not its accuracy -- no possessor GT exists.
Abstention cause is recovered from the recorded margin so PossessorFrame and
possession_timeline.json are untouched.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 3: Contested-margin curve

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/possession_profile.py`
- Test: `packages/matchlab_core/tests/test_possession_profile.py`

**Interfaces:**
- Consumes: `CurvePoint`, `_fraction`, `profile_possessor_labels` from Task 2.
- Produces: `profile_possessor_labels(...).contested_curve` populated — one `CurvePoint` per τ in `tau_grid_px`, `count` = asserted rows with `margin < τ`, `fraction` = count / asserted_frames.

- [ ] **Step 1: Write the failing test**

Append to `packages/matchlab_core/tests/test_possession_profile.py`:

```python
def test_contested_curve_counts_asserted_rows_below_each_tau():
    timeline = [
        _row(0, 1, margin=1.0),
        _row(1, 1, margin=6.0),
        _row(2, 1, margin=30.0),
        _row(3, None, margin=0.5),  # abstained -- must not enter the curve
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=4, tau_grid_px=(0.0, 2.0, 10.0, 40.0)
    )
    assert [pt.threshold for pt in p.contested_curve] == [0.0, 2.0, 10.0, 40.0]
    assert [pt.count for pt in p.contested_curve] == [0, 1, 2, 3]
    assert p.contested_curve[-1].fraction == pytest.approx(1.0)


def test_contested_curve_is_monotone_non_decreasing():
    timeline = [_row(i, 1, margin=float(i)) for i in range(20)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=20)
    counts = [pt.count for pt in p.contested_curve]
    assert counts == sorted(counts)


def test_contested_curve_with_no_asserted_rows_is_all_zero():
    timeline = [_row(0, None, margin=0.0)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=1)
    assert all(pt.count == 0 and pt.fraction == 0.0 for pt in p.contested_curve)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q -k contested_curve`
Expected: FAIL — `assert [] == [0.0, 2.0, 10.0, 40.0]` (curve is empty)

- [ ] **Step 3: Implement**

In `profile_possessor_labels`, before the `return`, add:

```python
    contested_curve = [
        CurvePoint(
            threshold=tau,
            count=(c := sum(1 for r in asserted if r.margin < tau)),
            fraction=_fraction(c, len(asserted)),
        )
        for tau in tau_grid_px
    ]
```

and pass `contested_curve=contested_curve` in the `PossessorLabelProfile(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/possession_profile.py \
        packages/matchlab_core/tests/test_possession_profile.py
git commit -m "$(cat <<'EOF'
feat(possession): contested-margin curve over a tau sweep (SPO-83)

Fraction of asserted labels resting on a near-tie, swept over tau so no
flattering threshold can be picked after the fact.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 4: Depth-discordance sweep (the false-possession proxy)

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/possession_profile.py`
- Test: `packages/matchlab_core/tests/test_possession_profile.py`

**Interfaces:**
- Consumes: `index_possessor_boxes`, `rank_candidates` (Task 1); `CurvePoint`, `_fraction` (Task 2).
- Produces: `.depth_discordance` (one `CurvePoint` per ratio in `depth_ratio_grid`) and `.depth_evaluable_frames`.

**Definition.** For each asserted row, locate the possessor's box in that frame
and the *nearest other* candidate (the runner-up by distance from the ball —
deliberately not `ranked[1]`, since smoothing can make the possessor differ from
the raw nearest). A row is discordant at ratio `r` when
`runner_up_height / possessor_height > r`: a nearer player (larger box) sits
comparably close in pixels while a further player wins on 2D distance. Rows with
fewer than two candidates, or where the possessor has no box that frame, are not
evaluable and are excluded from the denominator.

- [ ] **Step 1: Write the failing test**

Append to `packages/matchlab_core/tests/test_possession_profile.py`. **Merge the
import line into the file's existing top-of-file import block** — a mid-file
module-level import trips ruff `E402`; the same applies to every later task that
appends imports.

```python
# merge into the existing `from matchlab_core.schemas import ...` at the top:
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    PossessorFrame,
    Team,
    Tracklet,
    TrackletFrame,
)


def _tracklet(tid, frame_boxes, conf=0.9, cls=DetectionClass.PLAYER):
    """frame_boxes: dict frame_idx -> (x1, y1, x2, y2)."""
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(
                frame_idx=f,
                box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                confidence=conf,
            )
            for f, b in sorted(frame_boxes.items())
        ],
    )


def _ball_obs(frame, x, y):
    return BallObservation(
        frame_idx=frame, t=frame / 25.0, xy=Point(x=x, y=y), confidence=1.0
    )


def test_depth_discordance_flags_a_much_taller_runner_up():
    # Possessor (tid 1) is 20px tall; the runner-up (tid 2) is 80px tall --
    # a much nearer player sitting comparably close in pixels.
    tracklets = [
        _tracklet(1, {0: (0, 0, 10, 20)}),
        _tracklet(2, {0: (14, 0, 40, 80)}),
    ]
    ball = [_ball_obs(0, 5, 10)]
    p = profile_possessor_labels(
        [_row(0, 1)], tracklets, ball, Params(), total_frames=1,
        depth_ratio_grid=(1.2, 2.0, 8.0),
    )
    assert p.depth_evaluable_frames == 1
    assert [pt.count for pt in p.depth_discordance] == [1, 1, 0]
    assert p.depth_discordance[0].fraction == pytest.approx(1.0)


def test_depth_concordance_when_candidates_are_similar_height():
    tracklets = [
        _tracklet(1, {0: (0, 0, 10, 40)}),
        _tracklet(2, {0: (14, 0, 24, 42)}),
    ]
    ball = [_ball_obs(0, 5, 20)]
    p = profile_possessor_labels(
        [_row(0, 1)], tracklets, ball, Params(), total_frames=1,
        depth_ratio_grid=(1.2, 2.0),
    )
    assert p.depth_evaluable_frames == 1
    assert [pt.count for pt in p.depth_discordance] == [0, 0]


def test_single_candidate_frames_are_not_depth_evaluable():
    tracklets = [_tracklet(1, {0: (0, 0, 10, 20)})]
    ball = [_ball_obs(0, 5, 10)]
    p = profile_possessor_labels(
        [_row(0, 1)], tracklets, ball, Params(), total_frames=1
    )
    assert p.depth_evaluable_frames == 0
    assert all(pt.count == 0 for pt in p.depth_discordance)


def test_runner_up_is_nearest_other_not_rank_one_after_smoothing():
    # Smoothing made tid 2 the possessor even though tid 1 is nearest the ball.
    # The runner-up must then be tid 1 (nearest *other*), not tid 2 itself.
    tracklets = [
        _tracklet(1, {0: (0, 0, 10, 100)}),   # nearest, tall
        _tracklet(2, {0: (30, 0, 40, 20)}),   # possessor after smoothing, short
    ]
    ball = [_ball_obs(0, 5, 50)]
    p = profile_possessor_labels(
        [_row(0, 2)], tracklets, ball, Params(), total_frames=1,
        depth_ratio_grid=(2.0,),
    )
    assert p.depth_evaluable_frames == 1
    assert p.depth_discordance[0].count == 1  # 100/20 = 5.0 > 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q -k depth`
Expected: FAIL — `assert 0 == 1` on `depth_evaluable_frames`

- [ ] **Step 3: Implement**

Add the import at the top of `possession_profile.py`:

```python
from matchlab_core.stages.possession.ranking import index_possessor_boxes, rank_candidates
```

and inside `profile_possessor_labels`, after the contested curve:

```python
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
```

Pass `depth_evaluable_frames=len(depth_ratios)` and `depth_discordance=depth_discordance` in the `PossessorLabelProfile(...)` call.

Extend the module docstring with the proxy's honest failure mode:

```python
# ... appended to the module docstring:
# The depth-discordance indicator is a PROXY for the "ball in front of a distant
# player" false-possession mode (Peral et al., VISAPP 2025). Bounding-box height
# is the only depth cue available without calibration, so the proxy fires on two
# players at genuinely the same depth whose boxes differ in height -- one
# crouching, one occluded, one truncated at the frame edge. Never quote the rate
# without that caveat.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/possession_profile.py \
        packages/matchlab_core/tests/test_possession_profile.py
git commit -m "$(cat <<'EOF'
feat(possession): depth-discordance sweep as false-possession proxy (SPO-83)

Runner-up/possessor bbox-height ratio, swept -- the measurable stand-in for
Peral's "ball in front of a distant player" mode. Proxy, not a contamination
measurement; its failure mode is documented at the definition.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 5: Temporal instability, team-flip implausibility, and aggregation

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/possession_profile.py`
- Test: `packages/matchlab_core/tests/test_possession_profile.py`

**Interfaces:**
- Consumes: `SegmentStats`, `CurvePoint`, `_fraction` (Task 2).
- Produces:
  - `.segments` and `.implausible_team_flips` populated
  - `aggregate_profiles(profiles: list[PossessorLabelProfile]) -> PossessorLabelProfile` — sums counts across sequences and recomputes fractions. Raises `ValueError` on mismatched curve grids. Used by Task 8.

**Definitions.** A *segment* is a maximal run of rows with the same non-`None`
possessor **and contiguous `frame_idx`** — a frame gap breaks the run, since the
ball was unobserved in between. `changes` counts adjacent row pairs whose
possessor differs (including to/from `None`); `span_seconds` is
`(last.frame_idx - first.frame_idx + 1) / fps`. A team flip is implausible when
two consecutive segments have different, non-`UNKNOWN` teams and the earlier
segment is shorter than `te_frames`.

- [ ] **Step 1: Write the failing test**

Append to `packages/matchlab_core/tests/test_possession_profile.py`, merging the
import into the top-of-file block (ruff `E402`):

```python
# merge into the existing `from matchlab_core.possession_profile import ...`:
from matchlab_core.possession_profile import aggregate_profiles


def test_segments_split_on_possessor_change():
    timeline = [_row(0, 1), _row(1, 1), _row(2, 2), _row(3, 2), _row(4, 2)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=5)
    assert p.segments.count == 2
    assert p.segments.total_segment_frames == 5
    assert p.segments.mean_frames == pytest.approx(2.5)


def test_segments_split_on_a_frame_gap_even_with_the_same_possessor():
    timeline = [_row(0, 1), _row(1, 1), _row(5, 1)]  # frames 2-4 unobserved
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=6)
    assert p.segments.count == 2


def test_below_te_counts_short_segments():
    # Segment lengths 1, 2, 3, 4 with te=3 -> two segments below threshold.
    timeline = [
        _row(0, 1),
        _row(1, 2), _row(2, 2),
        _row(3, 3), _row(4, 3), _row(5, 3),
        _row(6, 4), _row(7, 4), _row(8, 4), _row(9, 4),
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=10, te_frames=3
    )
    assert p.segments.count == 4
    assert p.segments.below_te_count == 2
    assert p.segments.below_te_fraction == pytest.approx(0.5)


def test_changes_per_second_uses_fps_and_span():
    # 25 rows spanning 1 second at 25 fps, alternating every 5 frames -> 4 changes.
    timeline = [_row(i, 1 + i // 5) for i in range(25)]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=25, fps=25.0
    )
    assert p.segments.changes == 4
    assert p.segments.span_seconds == pytest.approx(1.0)
    assert p.segments.changes_per_second == pytest.approx(4.0)


def test_abstention_rows_count_as_a_change():
    timeline = [_row(0, 1), _row(1, None), _row(2, 1)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=3)
    assert p.segments.changes == 2


def test_short_segment_switching_team_is_implausible():
    timeline = [
        _row(0, 1, team=Team.HOME),                       # 1-frame segment
        _row(1, 2, team=Team.AWAY), _row(2, 2, team=Team.AWAY),
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=3, te_frames=3
    )
    assert p.implausible_team_flips == 1


def test_long_segment_switching_team_is_plausible():
    timeline = [
        _row(0, 1, team=Team.HOME), _row(1, 1, team=Team.HOME),
        _row(2, 1, team=Team.HOME), _row(3, 1, team=Team.HOME),
        _row(4, 2, team=Team.AWAY),
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=5, te_frames=3
    )
    assert p.implausible_team_flips == 0


def test_unknown_team_never_counts_as_a_flip():
    timeline = [_row(0, 1, team=Team.UNKNOWN), _row(1, 2, team=Team.AWAY)]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=2, te_frames=3
    )
    assert p.implausible_team_flips == 0


def test_aggregate_sums_counts_and_recomputes_fractions():
    a = profile_possessor_labels(
        [_row(0, 1, margin=1.0), _row(1, 1, margin=100.0)], [], [], Params(),
        total_frames=4, tau_grid_px=(2.0,),
    )
    b = profile_possessor_labels(
        [_row(0, 1, margin=1.0)], [], [], Params(), total_frames=6, tau_grid_px=(2.0,),
    )
    agg = aggregate_profiles([a, b])
    assert agg.total_frames == 10
    assert agg.asserted_frames == 3
    assert agg.coverage == pytest.approx(0.3)
    assert agg.contested_curve[0].count == 2
    assert agg.contested_curve[0].fraction == pytest.approx(2 / 3)


def test_aggregate_rejects_mismatched_curve_grids():
    a = profile_possessor_labels([], [], [], Params(), total_frames=1, tau_grid_px=(2.0,))
    b = profile_possessor_labels([], [], [], Params(), total_frames=1, tau_grid_px=(5.0,))
    with pytest.raises(ValueError, match="grid"):
        aggregate_profiles([a, b])


def test_aggregate_of_nothing_is_empty_not_a_crash():
    agg = aggregate_profiles([])
    assert agg.total_frames == 0
    assert agg.coverage == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q -k "segments or team or aggregate or changes or below_te"`
Expected: FAIL — `ImportError: cannot import name 'aggregate_profiles'`

- [ ] **Step 3: Implement**

Add to `possession_profile.py`, above `profile_possessor_labels`:

```python
def _segments(timeline: list[PossessorFrame]) -> list[list[PossessorFrame]]:
    """Maximal runs of the same non-None possessor over contiguous frames.

    A frame gap breaks a run even when the possessor id matches: the ball was
    unobserved in between, so the two runs are not one continuous possession.
    """
    runs: list[list[PossessorFrame]] = []
    for row in timeline:
        if row.possessor_tracklet_id is None:
            continue
        if (
            runs
            and runs[-1][-1].possessor_tracklet_id == row.possessor_tracklet_id
            and runs[-1][-1].frame_idx + 1 == row.frame_idx
        ):
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs
```

and inside `profile_possessor_labels`, after the depth block:

```python
    ordered = sorted(timeline, key=lambda r: r.frame_idx)
    runs = _segments(ordered)
    changes = sum(
        1
        for prev, nxt in zip(ordered, ordered[1:])
        if prev.possessor_tracklet_id != nxt.possessor_tracklet_id
    )
    span_seconds = (
        (ordered[-1].frame_idx - ordered[0].frame_idx + 1) / fps if ordered and fps else 0.0
    )
    below_te = sum(1 for run in runs if len(run) < te_frames)
    total_segment_frames = sum(len(run) for run in runs)
    segments = SegmentStats(
        count=len(runs),
        total_segment_frames=total_segment_frames,
        mean_frames=_fraction(total_segment_frames, len(runs)),
        below_te_count=below_te,
        below_te_fraction=_fraction(below_te, len(runs)),
        changes=changes,
        span_seconds=span_seconds,
        changes_per_second=changes / span_seconds if span_seconds else 0.0,
    )

    implausible_team_flips = sum(
        1
        for prev, nxt in zip(runs, runs[1:])
        if len(prev) < te_frames
        and prev[-1].team != nxt[0].team
        and Team.UNKNOWN not in (prev[-1].team, nxt[0].team)
    )
```

Pass `segments=segments` and `implausible_team_flips=implausible_team_flips` in the
`PossessorLabelProfile(...)` call, and add `Team` to the schemas import:

```python
from matchlab_core.schemas import BallObservation, PossessorFrame, Team, Tracklet
```

Then append the aggregator at the end of the module:

```python
def _sum_curves(profiles: list[PossessorLabelProfile], attr: str, denom: int) -> list[CurvePoint]:
    grids = {tuple(pt.threshold for pt in getattr(p, attr)) for p in profiles}
    if len(grids) > 1:
        raise ValueError(
            f"cannot aggregate {attr}: profiles were computed on different "
            f"threshold grids {sorted(grids)}"
        )
    grid = next(iter(grids))
    totals = [
        sum(getattr(p, attr)[i].count for p in profiles) for i in range(len(grid))
    ]
    return [
        CurvePoint(threshold=th, count=c, fraction=_fraction(c, denom))
        for th, c in zip(grid, totals)
    ]


def aggregate_profiles(profiles: list[PossessorLabelProfile]) -> PossessorLabelProfile:
    """Pool per-sequence profiles: sum every count, recompute every fraction.

    Fractions are never averaged -- a 20-frame sequence would otherwise weigh as
    much as a 750-frame one.
    """
    if not profiles:
        return PossessorLabelProfile(
            total_frames=0, asserted_frames=0, coverage=0.0, abstention=AbstentionBreakdown()
        )

    total_frames = sum(p.total_frames for p in profiles)
    asserted = sum(p.asserted_frames for p in profiles)
    depth_evaluable = sum(p.depth_evaluable_frames for p in profiles)
    seg_count = sum(p.segments.count for p in profiles)
    seg_frames = sum(p.segments.total_segment_frames for p in profiles)
    below_te = sum(p.segments.below_te_count for p in profiles)
    changes = sum(p.segments.changes for p in profiles)
    span_seconds = sum(p.segments.span_seconds for p in profiles)

    return PossessorLabelProfile(
        total_frames=total_frames,
        asserted_frames=asserted,
        coverage=_fraction(asserted, total_frames),
        abstention=AbstentionBreakdown(
            no_ball_observation=sum(p.abstention.no_ball_observation for p in profiles),
            outside_radius=sum(p.abstention.outside_radius for p in profiles),
            contested_tie=sum(p.abstention.contested_tie for p in profiles),
        ),
        contested_curve=_sum_curves(profiles, "contested_curve", asserted),
        depth_evaluable_frames=depth_evaluable,
        depth_discordance=_sum_curves(profiles, "depth_discordance", depth_evaluable),
        segments=SegmentStats(
            count=seg_count,
            total_segment_frames=seg_frames,
            mean_frames=_fraction(seg_frames, seg_count),
            below_te_count=below_te,
            below_te_fraction=_fraction(below_te, seg_count),
            changes=changes,
            span_seconds=span_seconds,
            changes_per_second=changes / span_seconds if span_seconds else 0.0,
        ),
        implausible_team_flips=sum(p.implausible_team_flips for p in profiles),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q`
Expected: PASS (23 passed)

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest packages -q && uv run ruff check packages`
Expected: PASS, no lint findings.

- [ ] **Step 6: Commit**

```bash
git add packages/matchlab_core/src/matchlab_core/possession_profile.py \
        packages/matchlab_core/tests/test_possession_profile.py
git commit -m "$(cat <<'EOF'
feat(possession): temporal instability, team-flip check, profile aggregation (SPO-83)

Segments break on possessor change AND frame gap; aggregation sums counts and
recomputes fractions so a short sequence cannot outweigh a long one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 6: `profile_run_dir` adapter

**Files:**
- Modify: `packages/matchlab_core/src/matchlab_core/possession_profile.py`
- Test: `packages/matchlab_core/tests/test_possession_profile.py`

**Interfaces:**
- Consumes: `profile_possessor_labels` (Tasks 2–5); `ArtifactStore`, `ArtifactName` from `matchlab_core.artifacts` / `matchlab_core.schemas.run`.
- Produces: `profile_run_dir(run_dir: str | Path, **kwargs) -> PossessorLabelProfile` — reads `possession_timeline.json`, `tracklets.json`, `ball.jsonl` and the manifest's frame count. `**kwargs` forward to `profile_possessor_labels`.

This is what lets the same profiler serve the real SoccerNet-ball eval once that data exists, with no new code.

- [ ] **Step 1: Write the failing test**

Append to `packages/matchlab_core/tests/test_possession_profile.py`, merging the
imports into the top-of-file block (ruff `E402`):

```python
# merge into the top-of-file imports:
import json

from matchlab_core.possession_profile import profile_run_dir


def test_profile_run_dir_reads_a_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "possession_timeline.json").write_text(
        json.dumps([_row(0, 1).model_dump(mode="json"), _row(1, 1).model_dump(mode="json")])
    )
    (run / "tracklets.json").write_text(
        json.dumps([_tracklet(1, {0: (0, 0, 10, 20), 1: (0, 0, 10, 20)}).model_dump(mode="json")])
    )
    (run / "ball.jsonl").write_text(
        "\n".join(json.dumps(_ball_obs(f, 5, 10).model_dump(mode="json")) for f in (0, 1))
    )
    (run / "manifest.json").write_text(json.dumps({"run_id": "r", "frame_count": 10, "fps": 25.0}))

    p = profile_run_dir(run)
    assert p.total_frames == 10
    assert p.asserted_frames == 2
    assert p.abstention.no_ball_observation == 8


def test_profile_run_dir_without_a_timeline_raises(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"run_id": "r", "frame_count": 5}))
    with pytest.raises(FileNotFoundError, match="possession_timeline"):
        profile_run_dir(run)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q -k run_dir`
Expected: FAIL — `ImportError: cannot import name 'profile_run_dir'`

- [ ] **Step 3: Implement**

Add to the imports of `possession_profile.py`:

```python
import json
from pathlib import Path

from matchlab_core.artifacts import ArtifactStore
from matchlab_core.schemas.run import ArtifactName
```

and append:

```python
def profile_run_dir(run_dir: str | Path, **kwargs) -> PossessorLabelProfile:
    """Profile a completed run's possessor timeline from its artifacts.

    Frame count and fps come from the manifest -- the timeline alone cannot
    distinguish "ball not observed" from "clip ended".
    """
    store = ArtifactStore(run_dir)
    if not store.exists(ArtifactName.POSSESSION_TIMELINE):
        raise FileNotFoundError(f"no possession_timeline.json in {run_dir}")

    timeline = store.read_json_list(ArtifactName.POSSESSION_TIMELINE, PossessorFrame)
    tracklets = (
        store.read_json_list(ArtifactName.TRACKLETS, Tracklet)
        if store.exists(ArtifactName.TRACKLETS)
        else []
    )
    ball = (
        list(store.read_jsonl(ArtifactName.BALL, BallObservation))
        if store.exists(ArtifactName.BALL)
        else []
    )
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    kwargs.setdefault("total_frames", int(manifest.get("frame_count") or len(timeline)))
    kwargs.setdefault("fps", float(manifest.get("fps") or 25.0))
    return profile_possessor_labels(timeline, tracklets, ball, Params(), **kwargs)
```

> If `ArtifactName.POSSESSION_TIMELINE` is not the enum member name, read
> `packages/matchlab_core/src/matchlab_core/artifacts.py::ARTIFACT_FILES` and use
> the member mapped to `possession_timeline.json`. Do not add a new member.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_core/tests/test_possession_profile.py -q`
Expected: PASS (25 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_core/src/matchlab_core/possession_profile.py \
        packages/matchlab_core/tests/test_possession_profile.py
git commit -m "$(cat <<'EOF'
feat(possession): profile_run_dir adapter over ArtifactStore (SPO-83)

Lets the same profiler serve the real SoccerNet-ball eval once that data
exists, with no new code.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 7: GT → oracle possession inputs adapter

**Files:**
- Create: `packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py`
- Test: `packages/matchlab_train/tests/test_possessor_audit.py`

**Interfaces:**
- Consumes: `matchlab_core.gt.GroundTruth`, `GroundTruthTrack`, `GroundTruthFrame`.
- Produces: `gt_to_possession_inputs(gt: GroundTruth) -> tuple[list[Tracklet], list[TeamAssignment], list[BallObservation]]`

**Mapping.** Roles `player`/`goalkeeper` → `Tracklet` with the GT `track_id`
preserved, `confidence=1.0`, `source="observed"` (perfect tracking by
construction). Role `referee` → a `Tracklet` with `cls=REFEREE` so the
estimator's class filter is genuinely exercised. Teams: `left`→`HOME`,
`right`→`AWAY`, role `referee`→`REFEREE`, anything else →`UNKNOWN`, matching
`stages/team/oracle.py`. The single `role="ball"` track → one `BallObservation`
per annotated frame at the box centre, `confidence=1.0`, `interpolated=False`;
frames with no ball annotation get **no row** — that is genuine absence, not a
gap to fill.

- [ ] **Step 1: Write the failing test**

Create `packages/matchlab_train/tests/test_possessor_audit.py`:

```python
"""SPO-83: GT -> oracle possession inputs, on a hand-built GroundTruth so every
surrounding input is pinned and a failure localizes to the adapter."""

from __future__ import annotations

from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.schemas import Box, DetectionClass, Team
from matchlab_train.datasets.possessor_audit import gt_to_possession_inputs


def _track(track_id, role, team, frames):
    """frames: dict frame_idx -> (x1, y1, x2, y2)."""
    return GroundTruthTrack(
        track_id=track_id,
        role=role,
        team=team,
        frames=[
            GroundTruthFrame(
                frame_idx=f, box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3])
            )
            for f, b in sorted(frames.items())
        ],
    )


def _gt(tracks, seq_length=4, fps=25.0):
    return GroundTruth(
        source="soccernet-tracking", sequence="TEST-1", fps=fps,
        width=1920, height=1080, seq_length=seq_length, tracks=tracks,
    )


def test_players_and_goalkeepers_become_possessor_class_tracklets():
    gt = _gt([
        _track(1, "player", "left", {0: (0, 0, 10, 40)}),
        _track(2, "goalkeeper", "right", {0: (50, 0, 60, 40)}),
    ])
    tracklets, _, _ = gt_to_possession_inputs(gt)
    by_id = {t.tracklet_id: t for t in tracklets}
    assert by_id[1].cls == DetectionClass.PLAYER
    assert by_id[2].cls == DetectionClass.GOALKEEPER
    assert by_id[1].frames[0].confidence == 1.0
    assert by_id[1].frames[0].source == "observed"


def test_referees_are_kept_as_referee_class_tracklets():
    gt = _gt([_track(9, "referee", None, {0: (0, 0, 10, 40)})])
    tracklets, teams, _ = gt_to_possession_inputs(gt)
    assert tracklets[0].cls == DetectionClass.REFEREE
    assert teams[0].team == Team.REFEREE


def test_team_mapping_matches_the_oracle_team_stage():
    gt = _gt([
        _track(1, "player", "left", {0: (0, 0, 10, 40)}),
        _track(2, "player", "right", {0: (50, 0, 60, 40)}),
        _track(3, "player", None, {0: (80, 0, 90, 40)}),
    ])
    _, teams, _ = gt_to_possession_inputs(gt)
    by_id = {t.tracklet_id: t.team for t in teams}
    assert by_id == {1: Team.HOME, 2: Team.AWAY, 3: Team.UNKNOWN}


def test_ball_track_becomes_observations_at_box_centres():
    gt = _gt([_track(99, "ball", None, {0: (10, 20, 14, 24), 1: (30, 40, 34, 44)})])
    _, _, ball = gt_to_possession_inputs(gt)
    assert [b.frame_idx for b in ball] == [0, 1]
    assert (ball[0].xy.x, ball[0].xy.y) == (12.0, 22.0)
    assert ball[0].t == 0.0
    assert ball[1].t == 0.04  # 1 / 25 fps
    assert all(b.confidence == 1.0 and not b.interpolated for b in ball)


def test_unannotated_ball_frames_produce_no_observation():
    # Ball annotated on frames 0 and 3 only; 1 and 2 are genuine absence.
    gt = _gt([_track(99, "ball", None, {0: (10, 20, 14, 24), 3: (30, 40, 34, 44)})])
    _, _, ball = gt_to_possession_inputs(gt)
    assert [b.frame_idx for b in ball] == [0, 3]


def test_ball_is_not_a_possessor_candidate_tracklet():
    gt = _gt([
        _track(1, "player", "left", {0: (0, 0, 10, 40)}),
        _track(99, "ball", None, {0: (10, 20, 14, 24)}),
    ])
    tracklets, _, _ = gt_to_possession_inputs(gt)
    assert [t.tracklet_id for t in tracklets] == [1]


def test_no_ball_track_yields_no_observations():
    gt = _gt([_track(1, "player", "left", {0: (0, 0, 10, 40)})])
    _, _, ball = gt_to_possession_inputs(gt)
    assert ball == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_train/tests/test_possessor_audit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'matchlab_train.datasets.possessor_audit'`

- [ ] **Step 3: Implement**

Create `packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py`:

```python
"""Weak possessor-label audit on GT inputs (SPO-83, criterion 2).

Runs the Phase 1 image-space possessor estimator on ORACLE inputs -- GT boxes,
GT teams, GT ball, straight from a SoccerNet-tracking `GroundTruth` -- and
profiles the resulting weak labels. Isolating the possession layer this way
removes detector/tracker/ball-detection error from the picture, so what the
profile describes is the estimator's own behaviour.

This produces NO accuracy number. There is no per-frame possessor ground truth
on any tier; see `matchlab_core.possession_profile` for what the indicators do
and do not mean.
"""

from __future__ import annotations

from matchlab_core.gt import GroundTruth
from matchlab_core.schemas import (
    BallObservation,
    DetectionClass,
    Point,
    Team,
    TeamAssignment,
    Tracklet,
    TrackletFrame,
)

# GT role -> detector class. "left"/"right" are camera-relative, not real
# home/away -- mirrors stages/team/oracle.py so audit and stage never disagree.
_ROLE_TO_CLASS = {
    "player": DetectionClass.PLAYER,
    "goalkeeper": DetectionClass.GOALKEEPER,
    "referee": DetectionClass.REFEREE,
}
_TEAM_FROM_SIDE = {"left": Team.HOME, "right": Team.AWAY}


def gt_to_possession_inputs(
    gt: GroundTruth,
) -> tuple[list[Tracklet], list[TeamAssignment], list[BallObservation]]:
    """GT -> (tracklets, team assignments, ball observations) for the estimator."""
    tracklets: list[Tracklet] = []
    teams: list[TeamAssignment] = []
    ball: list[BallObservation] = []
    fps = gt.fps or 25.0

    for track in gt.tracks:
        if track.role == "ball":
            for fr in track.frames:
                ball.append(
                    BallObservation(
                        frame_idx=fr.frame_idx,
                        t=fr.frame_idx / fps,
                        xy=Point(
                            x=(fr.box.x1 + fr.box.x2) / 2.0,
                            y=(fr.box.y1 + fr.box.y2) / 2.0,
                        ),
                        confidence=1.0,
                        interpolated=False,
                    )
                )
            continue

        cls = _ROLE_TO_CLASS.get(track.role)
        if cls is None:
            continue

        tracklets.append(
            Tracklet(
                tracklet_id=track.track_id,
                cls=cls,
                frames=[
                    TrackletFrame(
                        frame_idx=fr.frame_idx, box=fr.box, confidence=1.0, source="observed"
                    )
                    for fr in track.frames
                ],
            )
        )
        team = (
            Team.REFEREE
            if track.role == "referee"
            else _TEAM_FROM_SIDE.get(track.team or "", Team.UNKNOWN)
        )
        teams.append(TeamAssignment(tracklet_id=track.track_id, team=team, confidence=1.0))

    ball.sort(key=lambda b: b.frame_idx)
    return tracklets, teams, ball
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_train/tests/test_possessor_audit.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check packages
git add packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py \
        packages/matchlab_train/tests/test_possessor_audit.py
git commit -m "$(cat <<'EOF'
feat(train): GT -> oracle possession inputs adapter (SPO-83)

SoccerNet-tracking GT carries a ball track (gameinfo 'ball;1'), so GT boxes +
teams + ball give the possession layer the oracle isolation the eval config
declared impossible. Team mapping mirrors stages/team/oracle.py.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 8: Audit driver, exclusion rule, and CLI

**Files:**
- Modify: `packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py`
- Modify: `packages/matchlab_train/src/matchlab_train/cli.py`
- Test: `packages/matchlab_train/tests/test_possessor_audit.py`

**Interfaces:**
- Consumes: `gt_to_possession_inputs` (Task 7); `profile_possessor_labels`, `aggregate_profiles`, `PossessorLabelProfile` (Tasks 2–5); `HeuristicImagePossession`.
- Produces:
  - `MIN_BALL_COVERAGE = 0.5`
  - `SequenceAudit(sequence, total_frames, ball_gt_frames, ball_coverage, excluded, profile)`
  - `AuditReport(tier, min_ball_coverage, caveat, sequences, aggregate)`
  - `audit_sequence(gt, **params) -> SequenceAudit`
  - `audit_sequences(seq_dirs, *, min_ball_coverage=MIN_BALL_COVERAGE, **params) -> AuditReport`
  - CLI: `matchlab-train audit-possessor-labels --root <dir> [--limit N] --out report.json`

**Exclusion rule.** A sequence whose GT ball coverage (`ball frames / seq_length`)
is below `min_ball_coverage` is **excluded from the aggregate but still listed**,
with its coverage, in `report.sequences`. Averaging the near-zero-ball sequences
in would make the report read "the heuristic abstains constantly" when the cause
is missing ball *annotation*, not estimator behaviour.

- [ ] **Step 1: Write the failing test**

Append to `packages/matchlab_train/tests/test_possessor_audit.py`, merging the
imports into the top-of-file block (ruff `E402`):

```python
# merge into the top-of-file imports:
import pytest

from matchlab_train.datasets.possessor_audit import audit_sequence, audit_sequences


def _scene(seq_length, ball_frames, name="TEST-1"):
    """Two players 100px apart; the ball sits on player 1 for `ball_frames`."""
    players = [
        _track(1, "player", "left", {f: (0, 0, 20, 40) for f in range(seq_length)}),
        _track(2, "player", "right", {f: (100, 0, 120, 40) for f in range(seq_length)}),
    ]
    ball = _track(99, "ball", None, {f: (8, 18, 12, 22) for f in range(ball_frames)})
    gt = _gt([*players, ball], seq_length=seq_length)
    gt.sequence = name
    return gt


def test_audit_sequence_reports_ball_coverage_and_profiles():
    audit = audit_sequence(_scene(10, 10), smooth_radius=0)
    assert audit.sequence == "TEST-1"
    assert audit.total_frames == 10
    assert audit.ball_gt_frames == 10
    assert audit.ball_coverage == pytest.approx(1.0)
    assert audit.excluded is False
    assert audit.profile.asserted_frames == 10
    assert audit.profile.coverage == pytest.approx(1.0)


def test_low_ball_coverage_sequence_is_excluded_but_still_listed():
    report = audit_sequences(
        [_scene(10, 10, "GOOD-1"), _scene(10, 1, "SPARSE-1")], smooth_radius=0
    )
    by_name = {s.sequence: s for s in report.sequences}
    assert by_name["GOOD-1"].excluded is False
    assert by_name["SPARSE-1"].excluded is True
    assert by_name["SPARSE-1"].ball_coverage == pytest.approx(0.1)
    assert len(report.sequences) == 2  # excluded sequences stay visible


def test_aggregate_covers_retained_sequences_only():
    report = audit_sequences(
        [_scene(10, 10, "GOOD-1"), _scene(10, 1, "SPARSE-1")], smooth_radius=0
    )
    assert report.aggregate.total_frames == 10  # SPARSE-1's 10 frames excluded
    assert report.aggregate.asserted_frames == 10


def test_exclusion_threshold_is_configurable_and_recorded():
    report = audit_sequences([_scene(10, 6, "MID-1")], min_ball_coverage=0.8)
    assert report.min_ball_coverage == 0.8
    assert report.sequences[0].excluded is True
    assert report.aggregate.total_frames == 0


def test_report_carries_a_no_accuracy_caveat():
    report = audit_sequences([_scene(10, 10)])
    assert "accuracy" in report.caveat.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/matchlab_train/tests/test_possessor_audit.py -q -k "audit_sequence or coverage or aggregate or threshold or caveat"`
Expected: FAIL — `ImportError: cannot import name 'audit_sequence'`

- [ ] **Step 3: Implement the driver**

Append to `possessor_audit.py`:

```python
from pathlib import Path

from matchlab_core.gt import load_soccernet_sequence
from matchlab_core.possession_profile import (
    PossessorLabelProfile,
    aggregate_profiles,
    profile_possessor_labels,
)
from matchlab_core.stages.possession.heuristic_image import HeuristicImagePossession
from pydantic import BaseModel

# Sequences with sparse ball annotation are excluded from the aggregate: their
# low label coverage reflects missing GT, not estimator behaviour. They stay in
# `sequences` with their coverage so the choice is auditable.
MIN_BALL_COVERAGE = 0.5

CAVEAT = (
    "Label-risk profile, NOT an accuracy measure. No per-frame possessor ground "
    "truth exists on any MatchLab tier, so nothing here says how often a weak "
    "label is wrong. Depth discordance is a PROXY for the 'ball in front of a "
    "distant player' mode (Peral et al. VISAPP 2025) using bbox height as the "
    "only available depth cue -- it also fires on same-depth players whose boxes "
    "differ in height. A hand-labelled held-out set is the only route to an "
    "accuracy figure and remains deferred."
)


class SequenceAudit(BaseModel):
    sequence: str
    total_frames: int
    ball_gt_frames: int
    ball_coverage: float
    excluded: bool = False
    profile: PossessorLabelProfile


class AuditReport(BaseModel):
    tier: str
    min_ball_coverage: float
    caveat: str
    estimator: str
    params: dict
    sequences: list[SequenceAudit]
    aggregate: PossessorLabelProfile


def audit_sequence(gt: GroundTruth, **params) -> SequenceAudit:
    """Run the heuristic estimator on GT inputs and profile its weak labels."""
    tracklets, teams, ball = gt_to_possession_inputs(gt)
    estimator = HeuristicImagePossession(**params)
    timeline = estimator.estimate(None, tracklets, teams, ball)
    profile = profile_possessor_labels(
        timeline, tracklets, ball, estimator.params,
        total_frames=gt.seq_length, fps=gt.fps or 25.0,
    )
    return SequenceAudit(
        sequence=gt.sequence or "unknown",
        total_frames=gt.seq_length,
        ball_gt_frames=len(ball),
        ball_coverage=(len(ball) / gt.seq_length if gt.seq_length else 0.0),
        excluded=False,
        profile=profile,
    )


def audit_sequences(
    sequences: list[GroundTruth],
    *,
    tier: str = "soccernet-tracking",
    min_ball_coverage: float = MIN_BALL_COVERAGE,
    **params,
) -> AuditReport:
    """Audit many sequences; aggregate over those with enough ball annotation."""
    audits = []
    for gt in sequences:
        audit = audit_sequence(gt, **params)
        audit.excluded = audit.ball_coverage < min_ball_coverage
        audits.append(audit)

    retained = [a.profile for a in audits if not a.excluded]
    return AuditReport(
        tier=tier,
        min_ball_coverage=min_ball_coverage,
        caveat=CAVEAT,
        estimator="possession-heuristic-image",
        params=HeuristicImagePossession(**params).params.model_dump(mode="json"),
        sequences=audits,
        aggregate=aggregate_profiles(retained),
    )


def audit_soccernet_tracking(
    root: str | Path, *, limit: int | None = None, **kwargs
) -> AuditReport:
    """Load SNMOT sequence dirs under `root` and audit them."""
    seq_dirs = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "gameinfo.ini").exists())
    if limit is not None:
        seq_dirs = seq_dirs[:limit]
    return audit_sequences([load_soccernet_sequence(d) for d in seq_dirs], **kwargs)
```

> Move the `from pathlib import Path` / `from pydantic import BaseModel` imports
> up into the module's existing import block rather than leaving them mid-file;
> ruff will flag `E402` otherwise.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/matchlab_train/tests/test_possessor_audit.py -q`
Expected: PASS (12 passed)

- [ ] **Step 5: Add the CLI subcommand**

In `packages/matchlab_train/src/matchlab_train/cli.py`, after the
`derive-possessor-labels` parser block (around line 110), add:

```python
    ap_p = sub.add_parser(
        "audit-possessor-labels",
        help="Profile WEAK possessor labels on GT/oracle inputs (SPO-83; "
        "label-risk profile, not an accuracy measure)",
    )
    ap_p.add_argument(
        "--root", default="data/soccernet/tracking/test",
        help="Dir of SNMOT sequence dirs (each with gameinfo.ini)",
    )
    ap_p.add_argument("--limit", type=int, default=None, help="Audit at most N sequences")
    ap_p.add_argument("--out", required=True, help="Output report JSON path")
    ap_p.add_argument(
        "--min-ball-coverage", type=float, default=0.5,
        help="Exclude sequences with GT ball coverage below this from the aggregate",
    )
```

and, after the `derive-possessor-labels` dispatch block (around line 226), add:

```python
    if args.command == "audit-possessor-labels":
        from pathlib import Path

        from matchlab_train.datasets.possessor_audit import audit_soccernet_tracking

        report = audit_soccernet_tracking(
            args.root, limit=args.limit, min_ball_coverage=args.min_ball_coverage
        )
        Path(args.out).write_text(report.model_dump_json(indent=2))
        excluded = [s.sequence for s in report.sequences if s.excluded]
        agg = report.aggregate
        print(f"wrote audit of {len(report.sequences)} sequences to {args.out}")
        print(f"excluded (ball coverage < {report.min_ball_coverage}): {excluded or 'none'}")
        print(f"aggregate label coverage: {agg.coverage:.3f} over {agg.total_frames} frames")
        print(f"NOT AN ACCURACY MEASURE -- {report.caveat}")
        return 0
```

- [ ] **Step 6: Verify the CLI runs end to end on two real sequences**

Run:
```bash
uv run matchlab-train audit-possessor-labels \
  --root /home/jeremy/code/MatchDay/lab/data/soccernet/tracking/test \
  --limit 2 --out /tmp/claude-1000/-home-jeremy-code-MatchDay-lab/e529e701-fa7f-48bb-b879-be5196dee3e4/scratchpad/audit-smoke.json
```
Expected: exits 0, prints a coverage line and the caveat, and writes a JSON file whose `sequences` array has 2 entries.

- [ ] **Step 7: Run the full suite and lint**

Run: `uv run pytest packages -q && uv run ruff check packages`
Expected: PASS, no lint findings.

- [ ] **Step 8: Commit**

```bash
git add packages/matchlab_train/src/matchlab_train/datasets/possessor_audit.py \
        packages/matchlab_train/tests/test_possessor_audit.py \
        packages/matchlab_train/src/matchlab_train/cli.py
git commit -m "$(cat <<'EOF'
feat(train): audit-possessor-labels driver + CLI (SPO-83)

Per-sequence audit over SNMOT GT, aggregate over sequences with >=50% GT ball
coverage; sparse-ball sequences stay listed with their coverage so the
exclusion is auditable rather than baked in.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

---

### Task 9: Real-data run, extreme-case hand check, and report

**Files:**
- Create: `docs/reports/2026-07-25-spo83-possessor-label-audit.md`
- Modify: `docs/reference/possession-transition-gate.md`
- Modify: `docs/implementation-status.md`

**Interfaces:**
- Consumes: the CLI from Task 8.
- Produces: no code. A written report, and a corrected gate doc.

**This task is gated on the hand check.** Synthetic timelines prove the
arithmetic, not that the profiler says anything true about real footage — a
proxy metric reports a confident number on garbage. Do not write the report
before Step 3 passes.

- [ ] **Step 1: Run the audit over the full tier**

Run:
```bash
uv run matchlab-train audit-possessor-labels \
  --root /home/jeremy/code/MatchDay/lab/data/soccernet/tracking/test \
  --out /tmp/claude-1000/-home-jeremy-code-MatchDay-lab/e529e701-fa7f-48bb-b879-be5196dee3e4/scratchpad/spo83-audit.json
```
Expected: 49 sequences audited; SNMOT-139, SNMOT-149 and SNMOT-193 reported as excluded (their GT ball coverage is 1%, 1%, 0%); the aggregate covers the remaining 46.

If a different set is excluded, do not adjust the threshold to match this
expectation — record what the data actually says and note the discrepancy in
the report.

- [ ] **Step 2: Record the code revision for provenance**

Run: `git rev-parse --short HEAD`
Keep the value — every measured claim in the report must carry dataset, split,
sequence set, estimator params, and this revision, per `docs/README.md`
governance.

- [ ] **Step 3: Hand-check the extremes (gates the write-up)**

Write a throwaway script in the scratchpad that, for the two sequences with the
highest depth-discordance fraction, prints the 10 asserted frames with the
largest runner-up/possessor height ratio: frame index, possessor tracklet id,
runner-up id, both box heights, both distances to the ball, and the margin.

Then open the corresponding frames from
`data/soccernet/tracking/test/<seq>/img1/` and look at them. For each of the 10,
decide: is the flagged frame the situation the metric claims to detect (the ball
near a *further* player while a nearer one is close in pixels), or is it a
same-depth pair whose boxes differ for another reason (occlusion, crouching, a
box truncated at the frame edge)?

Record the tally in the report. **If most flagged frames are plainly clean
possession, the indicator is measuring box-height noise — say exactly that in
the report and do not quote the rate as a contamination proxy.** Repeat the same
inspection for the longest contested run (lowest-margin stretch) in those
sequences.

- [ ] **Step 4: Write the report**

Create `docs/reports/2026-07-25-spo83-possessor-label-audit.md` with, in order:

1. **What this is not** — first section, before any number: no possessor GT
   exists; nothing here is an accuracy or a true contamination rate.
2. **Provenance** — dataset (SoccerNet-tracking), split (test), sequence set
   (49 audited / 46 retained, excluded ones named with their coverage), oracle
   inputs (GT boxes + GT teams + GT ball), estimator + params, code revision
   from Step 2.
3. **The exclusion rule** — the 50% threshold, in the header, with the dropped
   sequences named. Not a footnote.
4. **Results** — coverage and abstention breakdown; the contested curve; the
   depth-discordance sweep; segment stats and change rate; team flips.
5. **The hand check** — the Step 3 tally verbatim, including any finding that
   the proxy misfires.
6. **What this means for the SPO-83 gate** — whether weak labels look good
   enough to bootstrap Peral training, framed against the gate doc's GO/NO-GO
   conditions. State plainly that criterion 1 (pass AP@1) and criterion 3 (the
   decision) remain open; this closes criterion 2 only.
7. **Follow-up** — a hand-labelled held-out possessor set is the only route to a
   real accuracy/contamination number; scope it as a new issue if the gate goes GO.

- [ ] **Step 5: Correct the gate doc**

In `docs/reference/possession-transition-gate.md`:
- Under "Weak-label quality (SPO-82)", replace the "sample frames and estimate
  the false-possession rate" instruction with a link to the new report and its
  headline numbers.
- Correct the claim in the "Reference ceilings" section that "no oracle-tracklet
  isolation is possible" — that holds for the *event* number on the
  SoccerNet-ball tier, but the SoccerNet-tracking tier carries a GT ball track,
  which is what this audit uses. Keep the original claim visible as superseded
  rather than silently rewriting it, per docs governance.

Make the same correction to note 2 of
`configs/pipeline.possession-heuristic-eval.yaml`, which states the claim in its
original form.

- [ ] **Step 6: Update implementation status**

In `docs/implementation-status.md`, under the possession-transition capability,
add the audit tool and link the report. State the honest bound: label-risk
profile on oracle inputs, no accuracy measured, pass AP@1 still unmeasured.

- [ ] **Step 7: Commit**

```bash
git add docs/reports/2026-07-25-spo83-possessor-label-audit.md \
        docs/reference/possession-transition-gate.md \
        docs/implementation-status.md \
        configs/pipeline.possession-heuristic-eval.yaml
git commit -m "$(cat <<'EOF'
docs(possession): weak-label audit report on oracle SNMOT inputs (SPO-83)

Closes gate criterion 2 with measured evidence: label coverage, contested-margin
curve, depth-discordance proxy, temporal stability over 46 retained sequences.
Corrects the "no oracle isolation possible" claim in the gate doc and eval
config -- it holds for the event number, not for the tracking tier's GT ball.
Criteria 1 (pass AP@1) and 3 (the GO/NO-GO) remain open.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Q2atBQGwacpjWVuZJNyYK2
EOF
)"
```

- [ ] **Step 8: Report status to the human**

SPO-83 is **not** closeable after this task. Summarize for the user: criterion 2
is now evidenced, criteria 1 and 3 remain human-gated (SoccerNet-ball data +
weights + GPU for the pass number; their decision for the GO/NO-GO). Offer to
post the audit summary as a comment on SPO-83 — do not change the issue's status.
