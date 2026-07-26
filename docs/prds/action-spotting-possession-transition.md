# PRD — Action Spotting via Possession Transitions (B3)

> ## ⚠️ PHASE 2 SUPERSEDED — 2026-07-27
>
> **This PRD's Phase 2 (`possession-peral`, the learned Peral estimator) is superseded.**
> Peral's block 2 is a smoother over a possession likelihood; the FOOTPASS/PCBAS
> reference (`TAAD → DST`) is a tactical sequence-to-sequence translator. Same slot,
> materially different capability — and unlike Peral, DST has released code and an
> external benchmark. Successor design:
> [`../superpowers/specs/2026-07-27-player-centric-action-spotting-design.md`](../superpowers/specs/2026-07-27-player-centric-action-spotting-design.md);
> identity contract governed by [ADR 008](../decisions/008-role-slots-are-not-roster-slots.md).
>
> **Phase 1 is NOT superseded and remains live.** `possession-heuristic-image`,
> `possession-viterbi` (added 2026-07-27) and `transition_to_events` are the
> calibration-free, role-free path — the only path that runs on tiers with no pitch
> keypoints, which today is every tier except FOOTPASS.
>
> SPO-83's go/no-go gate is unaffected and remains the owner's.

> **Subsystem:** B3 — Action Spotting ("what happened, when"). Canonical direction:
> Notion "B3 — Action Spotting" (revised 2026-07-23) and its child "Ball-Action Spotting —
> Quick-Win Implementation Plan". This PRD implements the doc's **recommended lead track**:
> possession-transition spotting. The T-DEED reference spotter (SPO-45–50, project
> "Reference Action-Spotting (T-DEED)") is a **separate, deferred** track and is not touched
> here.

## Problem Statement

Analysts want to know **which touch events happened and when** in a match — primarily
**passes** (and, secondarily, receptions) — localized to sub-second precision, and attributed
to a player. MatchDay's only spotting capability today is the T-DEED reference stage: a
learned, GPL-isolated, raw-frames model that (a) has no in-domain weights for our footage,
(b) suffers a measured ~40–45 mAP-point collapse on amateur/non-broadcast footage, and (c)
emits class+time only, with **no player attribution**.

Separately, a **pitch-space** heuristic possession engine already exists
(`events` slot, impl `possession-heuristic`): it runs a nearest-player-to-ball state machine
over the fused minimap and emits pass/touch/restart events with player attribution. But it is
**not measured** against any event benchmark (its output goes to `events.json`, which the
avg-mAP evaluator does not read), and it **hard-depends on pitch calibration + minimap fusion
+ ball tracking** — coverage MatchDay does not yet have on most footage. So we have no honest
number for "how good is our possession-based pass detection," and no calibration-optional path.

Peral et al. (VISAPP 2025, *Temporally Accurate Events Detection Through Ball Possessor
Recognition*) describes an **image-space** approach — per-player video tubes → per-frame ball
possessor → events from possession transitions — that needs only player bounding boxes (no
ball tracking, no calibration), reports per-frame possessor accuracy 71.9% and pass F1
67.3@0.6s / 74.3@1s on clean tactical footage, and is **attribution-native**. It has **no
public code and no downloadable weights**, and its training data (per-frame possessor labels)
is private and unavailable in any public dataset. We want to adopt this approach, but we first
need a measured baseline and an honest read on whether the learned model is worth building.

## Solution

Introduce a new **`possession` pipeline stage** (a first-class `StageKind` slot) that consumes
game-state artifacts (tracklets, teams, ball, optionally pose) and produces (1) a per-frame /
per-segment **ball-possessor** artifact and (2) **timed touch events** (passes, receptions)
that are written both to the attributed `events.json` **and** to `spotting.json` so the
existing avg-mAP evaluator scores them against `EventGroundTruth`.

Two swappable implementations register under this slot:

1. **`possession-heuristic-image` (Phase 1 — built).** An image-space, calibration-optional
   nearest-player-in-pixels possessor estimator with temporal hysteresis and team labels,
   followed by a transition→event rule layer. This is the Notion plan's Step 1 baseline. It is
   benchmark-scored: **pass** avg-mAP against SoccerNet Ball Action Spotting GT (which has a
   `PASS` class), giving MatchDay its first honest possession-based pass number and a floor the
   learned model must beat.

2. **`possession-peral` (Phase 2 — specified, gated).** A faithful reimplementation of Peral
   et al.: per-player 7-frame tubes (ResNet50+TSM, additive-angular-margin loss) → per-tube
   possession likelihood → temporal possessor selection (Conv-TasNet + TDNN over the
   likelihood field) → the same transition→event rule layer. **Build is gated on Phase 1
   results**; this PRD fully specifies it (the paper hands us every hyperparameter) but does
   **not** commit to training it until the baseline shows the handheld gap and the quality of
   heuristic-derived possessor labels.

Both impls share one **transition→event rule module** and one **event→SpottedEvent bridge**,
so the expensive learned model is a drop-in swap for the possessor estimator only.

Honest measurement boundaries baked into the design:
- Only **passes** are benchmark-scorable in Phase 1 — SoccerNet-ball GT has **no reception
  class** and **no per-frame possessor labels**. Receptions are emitted but not scored until we
  invest in our own labels (deferred by decision).
- `SpottedEvent` carries no `player_id`; avg-mAP scores **class+time only**. Player attribution
  is emitted in `events.json` / possession segments but is a **B4** measurement concern, not
  scored here.

## User Stories

1. As a match analyst, I want passes localized to <1s with a confidence, so that I can jump
   straight to the moments that matter without scrubbing.
2. As a match analyst, I want each detected pass attributed to a passer (and, when available, a
   receiver), so that I can build per-player involvement without manual tagging.
3. As a match analyst, I want receptions surfaced alongside passes, so that I can follow who
   received the ball even where no benchmark score exists yet.
4. As a match analyst, I want low-confidence / contested events flagged for review rather than
   silently asserted, so that I can trust the un-flagged ones.
5. As a researcher, I want a per-frame ball-possessor artifact, so that I can inspect and debug
   the signal the event layer is built on.
6. As a researcher, I want the possession stage to run **without** pitch calibration or ball
   tracking, so that I can spot events on footage where calibration coverage is absent.
7. As a researcher, I want the heuristic possessor baseline scored with avg-mAP against
   SoccerNet Ball Action Spotting, so that I have an honest floor before investing in a learned
   model.
8. As a researcher, I want the baseline and the learned Peral model to be two swappable impls
   of one stage slot selected by config, so that I can A/B them on identical inputs with a
   one-line config change.
9. As a researcher, I want the possession estimator isolated behind a narrow interface from the
   transition→event rules, so that swapping heuristic↔learned changes nothing downstream.
10. As a researcher, I want to run the possession stage against **oracle/GT tracklets and
    teams**, so that I can measure the event layer in isolation from tracker/team errors.
11. As a researcher, I want a documented, reproducible way to derive weak per-frame possessor
    labels from tracklets, so that I can train the Peral model later without new annotation.
12. As a researcher, I want the Peral reimplementation fully specified with the paper's exact
    hyperparameters (tube length, margins, filter widths, event thresholds), so that a build
    can start without re-deriving them.
13. As a researcher, I want the learned model's build explicitly gated on Phase 1 measurements,
    so that we do not spend training effort before the baseline justifies it.
14. As a Lab user, I want spotted passes/receptions rendered on the timeline and video overlay,
    so that I can visually verify them frame-by-frame.
15. As a Lab user, I want possession segments visualized per player, so that I can see who the
    system thinks had the ball and when.
16. As a pipeline operator, I want the possession stage to declare its provenance (impl,
    params, inputs consumed), so that a run's spotting output is auditable.
17. As a pipeline operator, I want a `none`/stub possession impl, so that pipelines that don't
    want possession spotting stay green with zero config.
18. As a maintainer, I want the new stage to reuse the existing `spotting.json` +
    `SpottedEvent` + avg-mAP evaluator contract, so that no new metric or evaluator branch is
    introduced for passes.
19. As a maintainer, I want the existing pitch-space `possession-heuristic` (`events` slot)
    left untouched, so that this change is additive and does not regress the fused-game-state
    lineage.
20. As a maintainer, I want the possession stage to consume artifacts only through the
    `ArtifactStore`, so that the portability invariant (stages touch the filesystem only via the
    store) holds.
21. As a maintainer, I want the transition→event rules unit-tested against hand-built
    possession sequences, so that pass/reception logic is pinned independent of any model.
22. As a maintainer, I want the event→SpottedEvent bridge tested for taxonomy + frame-index
    correctness, so that scored output can never silently drift from emitted events.
23. As a data engineer, I want the weak-label derivation to reuse existing `tracklets.json` /
    `ball.jsonl` / `teams.json`, so that no new ingest or download is required to bootstrap
    Phase 2.
24. As a product owner, I want passes prioritized over receptions in scope, so that we ship the
    benchmark-measurable capability first.
25. As a product owner, I want the honest measurement boundaries (passes-only scoring, no
    possessor GT, attribution unscored) stated in the artifact and status docs, so that no one
    over-reads the number.

## Implementation Decisions

**New stage slot.** Add `POSSESSION` to `StageKind`, ordered **before** `EVENTS` (so a future
events engine could consume its output) and independent of `SPOTTING`. Rationale: the
possession-transition signal needs game-state inputs (tracklets/teams/ball/pose), which fits
neither the raw-frames `EventSpotter.spot(ctx)` signature nor the pitch-space
`EventEngine.detect_events(ctx, minimap, players)` signature. A dedicated slot with its own
interface is the honest seam. The existing `events`-slot `possession-heuristic` is a separate
lineage and is **not** modified, moved, or deprecated by this PRD.

**Stage interface (deep module).** A `PossessionEstimator` base with a single method returning
a per-frame/segment possessor structure from game-state inputs — the only thing that differs
between heuristic and learned. Interface takes tracklets, team assignments, ball observations
(optional), and (for the learned impl) frame features/pose, read via `ctx.store`. Two
registered impls: `possession-heuristic-image` and `possession-peral`, plus a `none` stub. The
narrow interface is the whole point: the learned model is a same-shape swap.

**Event derivation (shared deep module).** One `transition_to_events` pure function maps a
possessor timeline → touch events: reception when a player **gains** possession, pass when a
player **loses** it to a possessor change; the paper's filter parameters (`Ts`=skip
first-touch reception, `Te`=minimum possession frames) are applied here. This module is
impl-agnostic and is the primary unit-test target.

**Event → SpottedEvent bridge (shared).** One function writes derived events to **both**
`events.json` (attributed `Event` with `player_id`) and `spotting.json` (`SpottedEvent`,
class+time, for avg-mAP). Add a `RECEPTION` value to `EventType`; map derived pass/reception to
the `SpottedEvent` class strings that align with SoccerNet-ball GT (`PASS` scorable; reception
emitted with a MatchDay-native class that has no GT match yet — documented, not scored).

**New artifact.** Add `ArtifactName.POSSESSION_TIMELINE` → `possession_timeline.json`
(per-frame possessor: frame_idx, possessor tracklet_id|null, confidence/margin, team). Distinct
from the existing `possession.json` (pitch-space segments written by the `events` engine) to
avoid clobbering that lineage. Events continue to flow into the existing `events.json` and
`spotting.json`; no new evaluator.

**Heuristic impl (Phase 1).** Image-space nearest-player-to-ball (pixel distance to bbox,
ball optional; when ball absent, fall back to a configured proxy — documented limitation),
temporal hysteresis (min-hold frames, contested-margin between nearest and second-nearest),
team labels from `teams.json`, Gaussian smoothing of the possessor field. Params exposed via a
pydantic model with the paper-aligned defaults where applicable.

**Peral impl (Phase 2 — specified, build-gated).** Blocks exactly per the paper:
- *Tubes:* per-player crops, square+20% margin, resized 128×128, collar `Tf=3` (7-frame tube);
  ResNet50 + Temporal Shift Module; penultimate-layer features concatenated → FC → 256-d;
  binary possessor head trained with additive angular margin (`m=0.5`, `s=1`).
- *Possessor selection:* Gaussian-smoothed `(2·Tp+1)×N` likelihood window, `Tp=2`; Conv-TasNet
  (1-D temporal) + TDNN → avg-pool → FC → `N+1` one-hot (players + none); cross-entropy.
- *Events:* shared `transition_to_events` with `Ts=7`, `Te=3`.
Isolation posture mirrors the tracker/T-DEED pattern only if a heavy/independent training env
is warranted; the model itself is permissive to reimplement (no vendored GPL code).

**Weak-label derivation (Phase 2 enabler).** A documented, reproducible harness that turns
existing `tracklets.json` + `ball.jsonl` (+ `teams.json`) into per-frame possessor labels via
the Phase 1 heuristic, for training the learned model. Explicitly noted as **weak/noisy**
(the "ball in front of a distant player" false-possession mode the paper flags will
contaminate labels); an honest eval requires a small hand-labeled held-out set (deferred).

**Config contract.** Standard `stages.possession.{impl, params}` block, selectable per YAML
like every other slot. Add a smoke config wiring `possession-heuristic-image` on stub upstream
stages, and an eval config targeting SoccerNet-ball with oracle/GT tracklets+teams so the event
layer is measured in isolation.

**Measurement.** Phase 1 success = a reported **pass avg-mAP@1** against SoccerNet Ball Action
Spotting via the existing `action_spotting_eval` path (no new metric). The number is reported
with its provenance (dataset, split, tracklet source, code revision) per docs governance; no
number is claimed until a run produces it.

## Testing Decisions

**What makes a good test here:** exercise external behavior through the stage/module interface
on constructed inputs; never assert on internal state or model internals. Pin every surrounding
input (use oracle/GT tracklets + teams, frozen ball) so a test failure localizes to the module
under test — consistent with the repo's "isolate subsystem tests with GT inputs" practice.

**Modules to test (user-confirmed: rule + bridge modules; the learned model is not unit-tested
in this PRD):**
- `transition_to_events` — hand-built possessor timelines → expected pass/reception events,
  including `Ts`/`Te` edge cases (first-touch pass skips reception; sub-`Te` blips filtered;
  possessor→none→possessor). This is the correctness core and gets the densest coverage.
- Event→SpottedEvent bridge — taxonomy mapping and frame-index/time fidelity; a derived pass
  must appear in `spotting.json` as the GT-aligned `PASS` class at the right frame_idx, and in
  `events.json` with the right `player_id`.
- `possession-heuristic-image` estimator — a tiny synthetic scene (2–3 tracklets + a ball
  path) yields the expected possessor timeline; contested/hysteresis behavior at the margin.
- Stage wiring — the `possession` slot builds from config, reads only via `ArtifactStore`, and
  writes the new `possession_timeline.json` artifact; the `none` stub is a no-op.

**Prior art:** mirror the SPO-49 avg-mAP tests (`action_spotting_eval`) for the scoring path,
and the existing stage-registration/artifact round-trip tests for wiring. Reuse the
event-GT/SoccerNet-ball ingest fixtures from SPO-47.

## Out of Scope

- **Building/training the Peral learned model.** Phase 2 is fully specified but build-gated on
  Phase 1 results (user decision: "heuristic baseline only for now").
- **Reception benchmark scoring.** No SoccerNet-ball GT class exists; deferred until we invest
  in our own reception labels.
- **Per-frame possessor GT and player-attribution scoring.** `SpottedEvent` scores class+time
  only; attribution accuracy is a B4 measurement concern.
- **A proper hand-labeled possessor train/eval set.** Deferred; Phase 2 bootstraps on weak
  heuristic labels only.
- **Shots, tackles, interceptions, set-pieces, audio, camera-motion cues.** The Notion plan's
  wider taxonomy and propose-and-confirm shots are later phases (user scope: possessor +
  pass/reception only).
- **Changes to the existing pitch-space `possession-heuristic` (`events` slot) or the T-DEED
  reference spotter (SPO-45–50).** This PRD is additive.
- **Amateur/phone-footage validation.** Per repo scope, development runs on broadcast/benchmark
  and oracle/GT data; the handheld recovery curve is a documented later phase.

## Further Notes

- **Peral availability (research finding):** no downloadable weights; no public code (authors:
  IRI CSIC-UPC + Kognia Sports Intelligence, a commercial firm); training data (per-frame
  possessor + touch events on LaLiga/SerieA tactical footage) is private. Reimplementation is
  tractable — the paper specifies all hyperparameters — so the risk is **data/labels, not the
  model**. Ceilings are on clean tactical footage (per-frame possessor 71.9%); amateur/handheld
  will be lower and abstention-heavy by design.
- **Sequencing rationale:** the heuristic baseline doubles as (a) the floor the learned model
  must beat and (b) the weak-label generator that Phase 2 trains on — so Phase 1 is a hard
  prerequisite regardless of whether Phase 2 proceeds.
- **Watch item:** SoccerNet 2026 replaces Team-BAS with *Player-Centric Ball Action Spotting*
  (action + responsible player) — almost exactly B3+B4. If its baselines/labels land, they may
  supply the possessor GT this PRD currently lacks; revisit the Phase 2 label plan then.
- **Docs to update on landing:** `docs/implementation-status.md` (new possession capability +
  its measured pass number, honest boundaries), `CLAUDE.md` stage-slot list, and the Notion B3
  page's status.
