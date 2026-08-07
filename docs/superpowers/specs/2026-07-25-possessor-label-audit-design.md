# Design — Weak Possessor-Label Audit (SPO-83, criterion 2)

**Status:** approved, not yet implemented
**Issue:** [SPO-83](https://linear.app/sports-statistics/issue/SPO-83) — Phase 2 go/no-go gate
**PRD:** [`docs/prds/action-spotting-possession-transition.md`](../../prds/action-spotting-possession-transition.md) (SPO-76)
**Gate doc:** [`docs/reference/possession-transition-gate.md`](../../reference/possession-transition-gate.md)

## Problem

SPO-83 is a human decision gate, not an implementation slice. Three acceptance
criteria are open; two cannot be closed without data and a human:

| Criterion | Blocked by |
|---|---|
| 1. Pass avg-mAP@1 write-up vs Peral's references | `data/soccernet/ball/` and `data/weights/` are both empty. No number is producible. |
| 3. GO/NO-GO decision + follow-on issues | The human's call. |

Criterion 2 — "weak-label quality assessed, **including the false-possession
contamination rate where measurable**" — is currently unaddressed:
`matchlab-train derive-possessor-labels` (SPO-82) emits weak labels and nothing
measures them. This design closes criterion 2.

### The substrate the gate doc missed

`configs/pipeline.possession-heuristic-eval.yaml` (note 2) and the gate doc both
state that oracle-tracklet isolation is impossible because SoccerNet-ball videos
carry event GT only. True for the *event* number — but it obscured that the
**SoccerNet-tracking tier carries a GT ball track**. `gameinfo.ini` declares
`trackletID_N = ball;1`, `gt.py::_parse_role` already maps it to `role="ball"`,
and `stages/detect/oracle.py` already emits GT ball observations via
`resolve_ball_track`.

Verified on the local data (`load_soccernet_sequence`, 2026-07-25):

- 49 SNMOT sequences, 750 frames each @ 25 fps, 1920×1080 — 36,750 frames
- per sequence ~21 players, 1 goalkeeper, 2 referees, 1 ball
- GT ball coverage 79.7% overall, unevenly: 26 sequences ≥ 94%, and
  SNMOT-139 (1%), SNMOT-149 (1%), SNMOT-193 (0%)

So ~29k frames of GT boxes + GT teams + GT ball are on disk, CPU-only, no
weights. This also delivers PRD user story 10 ("run the possession stage against
oracle/GT tracklets and teams"), which the eval config declares impossible.

## What this is not

There is **no per-frame possessor ground truth** on any tier. Nothing in this
design measures possessor *accuracy*. Every output is a property of the weak
label set's own structure — a **label-risk profile**, bounding how much of the
set is decided on a near-tie or is temporally implausible. A separate
hand-labelled held-out set is the only path to an accuracy number; it is scoped
as a follow-up, not folded in here.

## Indicators

Computed from a possessor timeline plus the inputs it was derived from.

1. **Coverage / abstention breakdown** — asserted vs. abstained frames split by
   cause: no ball observation, ball outside `possession_radius_px`, sub-
   `min_margin_px` tie. Bounds what fraction of frames can carry a training
   label at all.
2. **Contested rate as a curve over τ** — fraction of asserted frames with
   `margin < τ`, swept over τ rather than reported at one threshold, so no
   flattering threshold can be picked after the fact. These are the coin-flip
   labels.
3. **Depth-discordance rate** — the measurable proxy for Peral et al.'s "ball in
   front of a distant player" false-possession mode. Among asserted frames, the
   fraction where `runner_up_box.height / possessor_box.height` exceeds a ratio
   `r`: a *nearer* player (larger box) sits comparably close in pixels while a
   *further* player wins on 2D distance. Bbox height is the only depth cue
   available without calibration. Reported as a sweep over `r`
   (default grid 1.2 / 1.5 / 2.0), for the same reason indicator 2 is swept —
   a single ratio invites picking a flattering one.
4. **Temporal instability** — possessor segment-length distribution, fraction of
   segments shorter than `Te` (3 frames, the PRD's Peral-aligned default), and
   possessor flips per second (using the sequence fps, 25 for SNMOT). Labels
   flickering faster than physical ball control are noise regardless of margin.
5. **Team-flip implausibility** — possessor changes that also switch team where
   the preceding segment is shorter than `Te` frames.

Indicator 3 answers the "where measurable" clause. Its failure mode is real and
must be stated wherever the number appears: two players at genuinely the same
depth with different box heights (one crouching, one occluded, one truncated at
the frame edge) score as discordant.

## Components

### 1. `matchlab_core/stages/possession/ranking.py` (new — extraction)

`heuristic_image.py` inlines "index player boxes by frame, rank candidates by
distance from the ball". The profiler needs the same geometry for indicator 3.
Extracting it avoids two sources of truth:

```python
index_possessor_boxes(tracklets) -> dict[int, list[tuple[int, Box, float]]]
rank_candidates(ball_obs, boxes) -> list[Candidate]  # sorted by dist; tid, box, dist
```

`heuristic_image.py` is refactored onto these. Behaviour-preserving:
`test_possession_heuristic.py` must stay green **unmodified**. Needing to edit it
means the refactor changed behaviour — a defect, not a test update.

### 2. `matchlab_core/possession_profile.py` (new)

Pure profiler, sibling to `evaluation.py` / `action_spotting_eval.py`:

```python
profile_possessor_labels(timeline, tracklets, ball, params, *, total_frames)
    -> PossessorLabelProfile
profile_run_dir(run_dir) -> PossessorLabelProfile   # thin ArtifactStore adapter
```

No I/O in the first; no estimator dependency; works on any `PossessionEstimator`
output, so it serves the real SoccerNet-ball eval unchanged once that data
exists.

`total_frames` is passed by the caller — the audit driver supplies
`GroundTruth.seq_length`, and `profile_run_dir` reads the manifest's frame count
— because the timeline alone cannot distinguish "no ball row" from "clip ended".

Abstention cause is recovered from what the timeline already records: a row with
`possessor_tracklet_id=None` and `margin >= min_margin_px` is out-of-radius,
below it is a tie, and a frame with no row at all is no-ball. So
`PossessorFrame` is **not** widened — no artifact schema change, nothing to sync
in `web/src/lib/types.ts`.

### 3. `matchlab_train/datasets/possessor_audit.py` (new)

GT → oracle inputs → estimator → profiler, per sequence:

- `GroundTruthTrack` with role `player`/`goalkeeper` → `Tracklet`, GT `track_id`
  preserved, `confidence=1.0`, `source="observed"` — perfect tracking by
  construction.
- Team mapping `left`→`HOME`, `right`→`AWAY`, role `referee`→`REFEREE`, matching
  `stages/team/oracle.py` so audit and oracle stage never disagree.
- The single `role="ball"` track → `BallObservation` per annotated frame, box
  centre, `confidence=1.0`, `interpolated=False`. Frames with no ball annotation
  get **no row** — genuine absence, not a gap to fill.

### 4. CLI + report

`matchlab-train audit-possessor-labels --tier soccernet --limit N --out report.json`,
alongside `derive-possessor-labels`. Writes machine-readable JSON; the narrative
`docs/reports/2026-07-25-spo83-possessor-label-audit.md` is hand-written from it.

## Sequence-selection rule

The near-zero-ball sequences are the honesty hazard: averaged in, coverage
collapses and the report reads as "the heuristic abstains constantly" when the
cause is missing ball *annotation*, not estimator behaviour.

**Rule: exclude sequences with GT ball coverage below 50%; report them by name
with their coverage; profile over the retained set.** On current data that
retains 46 of 49 and drops SNMOT-139, -149, -193.

The threshold and the dropped list go in the report *header*, not a footnote.
The JSON carries both retained and excluded sets so the choice is auditable
rather than baked in.

## Testing

TDD throughout, split by what each layer can catch.

**Unit — `packages/matchlab_core/tests/test_possession_profile.py`.** Each
indicator gets a hand-built timeline whose answer is known by construction:

- coverage — known mix of asserted / out-of-radius / tie / absent-row frames
  returns exactly those four counts, summing to `total_frames`
- contested curve — margins placed either side of each τ; curve is monotone
  non-decreasing in τ and zero at τ=0
- depth discordance — two candidates with hand-set box heights, one discordant
  pair and one concordant; the discordant pair is counted, the other is not
- temporal instability — segments of length 1, 2, 3, 10 give a known sub-`Te`
  fraction and flip count
- team-flip — a team-switching possessor change inside `Te` is flagged; the same
  change spanning more than `Te` is not

**Adapter — `packages/matchlab_train/tests/test_possessor_audit.py`** on a tiny
synthetic `GroundTruth`: role filtering (referees excluded from possessor
candidates, retained as team assignments), the left/right mapping, ball frames
without annotation producing no `BallObservation`, and the <50% rule keeping and
dropping the right sequences.

**Refactor guard:** `test_possession_heuristic.py` runs unmodified.

**Real-data check, gating the write-up.** Synthetic timelines prove the
arithmetic, not that the profiler says anything true about real footage — a
proxy metric reports a confident number on garbage. Before writing the report,
run the audit over all 46 retained sequences and hand-check the extremes: pull
the highest-discordance frames and the longest contested runs on a couple of
sequences and confirm the flagged frames are the situations the metric claims to
detect. If depth discordance fires on plainly clean possession, it is measuring
box-height noise; say so in the report rather than publish the rate.

## Out of scope

- **Pass avg-mAP@1** (criterion 1) — needs `data/soccernet/ball/` + weights + GPU.
  Remains human-gated.
- **The GO/NO-GO decision** (criterion 3) — the human's, informed by this.
- **Hand-labelled possessor held-out set** — the only route to a true accuracy
  and contamination number. Scoped as a follow-up issue; this audit's output is
  what tells the human whether that labelling time is worth spending.
- **Building `possession-peral`** — build-gated per the PRD.
- **Any change to `PossessorFrame`, the artifact set, the web types, or the
  existing `events`-slot `possession-heuristic`.** This work is additive.
