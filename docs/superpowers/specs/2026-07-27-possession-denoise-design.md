# Possession denoising by Viterbi decode — design

**Track:** B3-B4 (possession-transition action spotting)
**Position in the plan:** Notion B3 development path **step 2**, partial —
"Peral-style possession-tube model on existing tracklets; **CRF/Viterbi
selection first**". This spec implements step 2's *selection layer* over the
existing heuristic signal. It does **not** implement the tube model; the visual
front end stays the SPO-79 nearest-player estimator.

## The problem

`possession-heuristic-image` decides each frame independently, then applies a
windowed-majority smoother. Both properties are wrong for the signal:

* **Independence.** Nothing in the estimator knows that possession is a
  *persistent* state. A player who is nearest for 40 frames and 3 px further
  than a neighbour on frame 21 loses the ball for one frame and gets it back.
* **The smoother is symmetric and content-free.** A windowed majority cannot
  distinguish "this flip is noise" from "this flip is a pass", because it sees
  only labels — not the ball, not the touch, not the teams.

The label audit measured the consequence: **1,133 possession segments across 25
SNMOT sequences, mean 19.1 frames**, with a large tail below the Te=3 filter,
and **42 team flips** whose implied ball travel is physically implausible. Those
are not passes. They are decision noise that `transition_to_events` converts
into spurious PASS/RECEPTION pairs.

The FOOTPASS ablation is the external evidence that this is the right layer to
attack: DST lifted precision from ~25% to ~68% on **unchanged** visual
predictions. The accuracy lives in the sequence model, not the frame model.

## Approach

Replace per-frame argmin + majority smoothing with a **first-order HMM decoded
by Viterbi** over the whole clip. States are the candidate possessors at each
frame plus an explicit `LOOSE` state; emissions come from the existing geometry;
transitions carry the tactical priors.

Rejected alternatives:

* **Widen the majority window.** Cheaper, but strictly worse: it has no way to
  spend its smoothing budget where a switch is *implausible* rather than merely
  brief. It also erases real quick passes at exactly the same rate it erases
  noise.
* **Learned Conv-TasNet smoother (Peral block 2).** Needs possessor labels we do
  not have; the label audit showed our weak labels carry ~8% sub-5px-margin
  assertions, so training on them would fit the noise. Viterbi first is what
  tells us whether sequence structure helps at all before paying for labels.
* **Semi-Markov / explicit duration model.** A first-order switch penalty *is* a
  geometric duration prior, and `transition_to_events` already applies Te. A
  second duration mechanism would be two knobs controlling one behaviour.

## Where it lives

A **second implementation in the existing possession slot**, not a new pass.

```
StageKind.POSSESSION
  ├── possession-heuristic-image   (SPO-79, unchanged — the ablation baseline)
  └── possession-viterbi           (this spec)
```

`PossessionEstimator.estimate(ctx, tracklets, teams, ball)` already receives
every input the trellis needs. Nothing downstream changes: the output is still
`list[PossessorFrame]` written to `possession_timeline.json`, and
`transition_to_events` consumes it unmodified. **The ablation is a config swap.**

### Components

| File | Responsibility |
|---|---|
| `matchlab_core/possession_denoise.py` | Pure trellis construction + Viterbi decode. No stage imports, no I/O. |
| `matchlab_core/stages/possession/viterbi.py` | Registers `possession-viterbi`; adapts stage inputs, calls the decoder. |
| `configs/pipeline.possession-viterbi-smoke.yaml` | Smoke config mirroring the heuristic smoke config. |

Shared geometry stays in `stages/possession/ranking.py` — both impls call
`index_possessor_boxes` / `rank_candidates`, so they cannot drift apart. Touch
corroboration reuses `ball_kinematics.detect_touches`, computed internally from
the same `ball` and `tracklets` the stage already holds.

## The model

### States

At frame *f* with a ball observation, the state set is

```
{ LOOSE } ∪ { c.tracklet_id : c ∈ rank_candidates(ball_f, boxes_f),
              c.distance ≤ possession_radius_px }
```

capped at `max_candidates` nearest (default 4) for tractability. Frames **with
no ball observation admit only `LOOSE`** — the heuristic's honest limitation is
preserved, not papered over. A clip with no ball yields an all-`LOOSE` timeline.

### Emissions (costs; lower is better)

```
cost(c) = distance_weight · (c.distance / possession_radius_px)
        − confidence_weight · log(max(ε, c.box_confidence · ball_conf))
cost(LOOSE) = loose_cost                       # constant, tunes the abstention rate
```

`ball_conf` is `obs.confidence`, multiplied by `interpolated_ball_weight` when
`obs.interpolated`. This is deliberately the same evidence the heuristic uses —
**the point of the experiment is that only the temporal model changes.**

### Transitions (costs)

Staying in the same state costs 0. `LOOSE`↔player and player→player switches
cost `switch_cost`, adjusted by four priors:

1. **Touch corroboration.** A possession change requires someone to have touched
   the ball. If a `BallTouch` lies within `touch_tolerance_frames` of the
   switch, subtract `touch_bonus`; otherwise add `no_touch_penalty`. This is the
   prior with the most independent evidence behind it — the touch detector was
   validated at 83.4% within 60 px of a player vs a 64.0% null.
2. **Ball-travel consistency.** A switch between two *different* players is only
   physical if the ball moved. If ball displacement across
   `±travel_window_frames` is below `min_travel_px`, add `no_travel_penalty`.
   Catches the audit's implausible flips directly.
3. **Team flip.** Turnovers are rarer than same-team passes. Switching to a
   different known team adds `team_flip_penalty`. `Team.UNKNOWN` on either side
   is **neutral** — no penalty, no bonus (ADR 003: missing evidence is neutral).
4. **Ball-proximity gate on the destination.** Switching *to* a player who is
   not within `possession_radius_px` is already impossible by construction (they
   are not in the state set), so no separate reachability term exists in v1.

**Deliberately deferred:** pitch-space reachability and the 13-way tactical
role. Both need pitch coordinates; SNMOT carries no pitch keypoints, so a prior
using them would be **unmeasurable on the only event GT we can reach**. Adding
an untestable prior is how you get a system that looks better and isn't. It goes
in when a calibrated tier exists to score it on.

### Decode

Standard Viterbi: forward pass accumulating min cost with backpointers, then
backtrace. Cost is O(F · K²) with K ≤ `max_candidates`+1 — trivially fast for a
30 s clip. Deterministic tie-breaking by tracklet id, matching `rank_candidates`.

### Output confidence

`PossessorFrame.confidence` is the decoded state's emission-based confidence on
the heuristic's own scale (`ball_conf · box_confidence · weight`), so the two
impls' confidences remain comparable and `transition_to_events`'
`low_confidence` contested-flag threshold keeps its meaning. `margin` stays the
raw nearest-vs-runner-up pixel separation — an input property, not a decision
property. `LOOSE` frames emit `confidence=0.0`, as today.

## Measurement

Three numbers, run as an ablation of `possession-heuristic-image` vs
`possession-viterbi` on the 25-sequence SNMOT tier:

1. **Cross-validation agreement with ball-trajectory touches must rise.**
   Baseline 71.4% @ ±1 s. This is the metric denoising *should* improve, and it
   is corroboration, not accuracy — two heuristics agreeing bounds nothing about
   absolute correctness.
2. **SNMOT localisation must not degrade.** Baseline: possession signal 3.0
   frames median on ball-contact classes. This is the **disconfirming guard**,
   and it is a hard one by construction: `snmot_localization_error` scores the
   *nearest* prediction, so removing predictions can only increase or hold the
   error. **A denoiser is structurally incapable of looking good on this
   metric.** It can only fail to look bad — which is exactly what makes it a
   valid guard against the denoiser deleting real events.
3. **Segment statistics move the predicted direction.** From
   `possession_profile`: segment count down, mean segment length up, below-Te
   fraction down, implausible team flips down. These are descriptive, not a
   pass/fail — they say *what the model did*, while (1) and (2) say whether it
   helped.

**Per-prior ablation:** each of the three priors independently disable-able by
setting its weight to 0, and the report must carry the four-row table (all on;
no touch; no travel; no team flip). A prior that does not move (1) or (3) is not
justified by the fact that it sounds physical.

**Explicitly not claimed:** possessor accuracy. There is still no per-frame
possessor GT on any tier — that is what FOOTPASS/PCBAS would buy and it remains
un-acquired. This spec cannot and does not measure whether the denoised
possessor is *correct*, only whether it is more self-consistent and no worse
localised.

## Testing

Unit tests against hand-built timelines with known-correct decodes:

* a single-frame flip inside a long hold is removed (the core case)
* a genuine pass corroborated by a touch **survives** — the disconfirming test,
  and the one that fails if the priors are too strong
* a team flip with no ball travel is removed
* a team flip *with* ball travel and a touch survives
* frames with no ball observation decode to `LOOSE`
* an empty ball list yields an all-`LOOSE` timeline of the right length
* each prior at weight 0 reproduces the un-prior'd decode
* decode is deterministic under equal costs (tracklet-id tie-break)

Stage tests mirror `test_possession_stage.py`: registry lookup, artifact shape,
and one end-to-end swap showing the two impls produce different timelines from
identical inputs.

## Risk

The honest one: **Viterbi with hand-set costs can be tuned to whatever the
crossval metric likes.** Both signals are heuristics on the same clip, and
agreement rises trivially if the denoiser is tuned toward the touch detector —
prior 1 makes that risk structural, not hypothetical. Mitigations: parameters
are set from the physical quantities they represent (frames, pixels) rather than
fitted; the ablation reports what each prior buys; and the localisation guard
must hold. If agreement rises while localisation degrades, the denoiser is
deleting real events and the result is a **negative** finding — scoped, per the
standing lesson, to the decision rule tested, not to sequence denoising in
general.
