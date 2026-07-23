# PRD: Reference Action-Spotting Stage (T-DEED)

**Status:** Draft for decomposition (2026-07-19).
**Owner:** Jeremy
**Precedence:** Planning document. Sits below the accepted ADRs and
`player-identity-vision.md`; where this PRD and an ADR disagree, the ADR wins.
**Depends on:** nothing. This is a **standalone new capability axis** (events, not
tracklets) and runs in parallel with the shippable-tracklet program
(`docs/prds/shippable-multi-cue-tracklet-system.md`, SPO-36..44). It shares no code path
with that work beyond the common stage/registry/artifact contracts.
**Supersedes:** nothing. It fills the dormant `StageKind.SPOTTING` v2 seam with its first
real implementation.
**Related:** ADR 004 (semantic-identity metrics discipline — this PRD applies the same
"measure before you believe" bar to a new task), `docs/prds/shippable-multi-cue-tracklet-system.md`
(the same licensing wall and the reference→shippable playbook this PRD mirrors),
`docs/implementation-status.md`.

---

## Problem Statement

MatchLab can track and associate players, but it cannot answer **"what happened, and when"** —
it has no action/event spotting. The `StageKind.SPOTTING` slot has been a registered but
empty v2 seam since v1 (only a disabled `NoSpotting` stub), and there is no way to run a real
action spotter, no ground truth for timed events, and no metric to score one.

We want to close that gap **fast**, on our own terms, with two hard realities acknowledged
up front:

1. **The state of the art is not shippable — on every axis.** The current SOTA lineage for
   frame-precise sport action spotting is **T-DEED** (the basis of the 2024 SoccerNet Ball
   Action Spotting winner and every 2025 Team-Ball-Action podium entry). Its code is
   **GPL-3.0**; the strong alternatives (ASTRA GPL-3.0, COMEDIAN CeCILL-2.1) are copyleft too;
   the only permissive release (Spivak, Apache-2.0) is a 2022-era feature-based method, not
   SOTA. And **every released weight set is trained on SoccerNet broadcast footage**, which is
   research/non-commercial data. So there is no download — code *or* weights — that can enter
   the product. This is the identical wall the tracklet program hit.

2. **We have no timed-event evaluation at all.** Ground truth in this repo is MOT bounding
   boxes only (`gt.py`); `evaluation.py` computes IDF1/MOTA/HOTA/purity/detection-AP — all
   tracking metrics. There is no timestamped-event GT, no tolerance-window mAP, nothing to
   tell us whether a spotter is any good.

What we need first is not a shipped feature but a **capability we can run and measure**: a
SOTA spotter we can point at *any* footage we choose, wired into our pipeline as a modular
stage, with a real benchmark number attached — explicitly Lab/internal only, never shipped.

## Solution

Stand up **T-DEED as a runnable, modular reference action-spotting stage** — the direct
analog of how this repo already treats `ultralytics` (AGPL, lazily supplied per-invocation,
"local-eval only, never shipped"), `sportsmot`/`insightface` (non-commercial, gated, eval
tier only), and `external-trackers/` (SOTA reference code in isolated venvs, never in deps).

Concretely, when this PRD lands you can:

- Run the pipeline with `spotting.impl: tdeed` on **any video** — an ingested SoccerNet
  sequence, a clip, uploaded footage — and get a `spotting.json` artifact of timed
  ball-action events (T-DEED's native 12-class taxonomy) rendered in the Lab.
- Score a run whose video has ball-action ground truth and read a real **avg-mAP@1** number
  in `eval.json` / `runs.metrics` / the benchmark matrix, computed by a metric we own and
  unit-test.
- Do all of the above with **zero GPL code and zero non-commercial data in the shippable
  repo**: the T-DEED code and its SoccerNet-trained weights live in an isolated sibling
  environment (`external-spotters/`), reached only through a subprocess bridge.

**What this explicitly is not:** a shippable spotter. Both the code (GPL) and the weights
(SoccerNet/NC data) are non-commercial. The clean-room, permissively-trained,
T-DEED-*equivalent* spotter is a **documented follow-up PRD** — exactly the shape of the
SPO-32/35 reference-tracker work feeding the shippable-tracklet PRD. This PRD's job is to make
the capability real, runnable-anywhere, and measurable, so the shippable decision is made
against numbers instead of guesses.

**The scoping insight:** the pipeline seam already exists. `StageKind.SPOTTING`,
`EventSpotter.spot(ctx) -> list[Event]`, the registry, the runner ordering (SPOTTING between
EVENTS and ANNOTATE), and the reserved `SHOT`/`TACKLE`/`INTERCEPTION` event types are all in
place. So "modular plug in and out" is satisfied by construction — flipping the stage on/off
is a one-line config change, and the GPL/NC risk is confined to one subprocess boundary and
one sibling directory that never merges into deps.

## User Stories

1. As a researcher, I want to run the pipeline with `spotting.impl: tdeed` on an arbitrary
   video, so that I get timed action events without touching any external tool by hand.
2. As a researcher, I want spotted events written to a dedicated `spotting.json` artifact, so
   that they are cleanly separated from the heuristic EventEngine's `events.json` and easy to
   inspect and score.
3. As a researcher, I want T-DEED's native 12-class ball-action taxonomy preserved verbatim
   in `spotting.json`, so that no fine-grained class information is lost to a premature mapping.
4. As a researcher, I want a documented (not enforced) mapping table from T-DEED classes to
   our `EventType` enum, so that a later product stage can consume spotting output without
   re-deriving it.
5. As a researcher, I want to point the spotter at footage that has **no** ground truth, so
   that I can qualitatively review spotting on our own clips, not just benchmark sequences.
6. As a researcher, I want the spotter to run as a normal pipeline stage (via `ctx.frames()`),
   so that it honors `sample_stride`/`max_frames`/`resize_width` like every other stage.
7. As a Lab user, I want spotted events rendered on the video timeline/overlay, so that I can
   see what the model spotted and when.
8. As a maintainer, I want the GPL T-DEED code to live in an isolated sibling directory with
   its own venv, so that it never becomes a dependency of the shippable packages.
9. As a maintainer, I want the in-repo stage to reach T-DEED only through a subprocess bridge,
   so that no GPL module is ever imported into shipped code.
10. As a maintainer, I want the bridge to fail loudly and cleanly on a missing env, non-zero
    exit, or malformed output, so that a broken external setup never corrupts a run silently.
11. As an evaluator, I want a timestamped-event ground-truth representation, so that spotting
    quality can be scored the way tracking quality already is.
12. As an evaluator, I want an `ingest-soccernet-ball` path that pulls SoccerNet Ball Action
    Spotting labels into that GT representation, so that I have a benchmark tier to score
    against.
13. As an evaluator, I want a tolerance-window **avg-mAP@1** metric computed over predicted
    vs. ground-truth events, so that I get the field-standard number for ball action spotting.
14. As an evaluator, I want that metric folded into `eval.json`, `runs.metrics`, and the
    benchmark matrix, so that it appears alongside the tracking metrics with no bespoke tooling.
15. As an evaluator, I want the worker to auto-score a spotting run when its video has
    ball-action GT, so that scoring is automatic like MOT scoring.
16. As a maintainer, I want a `configs/datasets/soccernet-ball.json` tier manifest marking
    the data eval-only and non-commercial, so that its licensing posture is explicit and
    enforced by layout (like `sportsmot.json`).
17. As a maintainer, I want a `configs/pipeline.tdeed-spotting-eval.yaml` and a fast smoke
    config, so that a full benchmark run and a CI-sized sanity run are both one command.
18. As a maintainer, I want released SoccerNet ball-action weights used as-is (inference
    only, no training), so that we reach real numbers on the shortest path.
19. As a maintainer, I want the whole effort done on an isolated git worktree and merged back
    when green, so that in-flight tracklet work is never disturbed.
20. As a reader of the docs, I want `implementation-status.md` updated to record spotting as a
    **reference/non-shippable** capability with its measured avg-mAP, so that no one mistakes
    it for a shipped feature.
21. As a product planner, I want the follow-up "shippable clean-room spotter" path named and
    scoped as out-of-scope-here, so that the reference→shippable sequence is on record.

## Implementation Decisions

**Posture (load-bearing).** Reference/internal tier only. GPL code and SoccerNet-trained
weights are both non-commercial; this stage runs on arbitrary footage for internal analysis
and on SoccerNet for benchmarking, and is **never shipped**. SoccerNet is a benchmark tier,
never a training source. The shippable clean-room equivalent is a separate, later PRD.

**Model & task.** T-DEED, targeting **SoccerNet Ball Action Spotting** (frame-precise,
12-class, end-to-end from raw frames, metric mAP@1). Chosen over classic 17-class SNAS
because it is the current SOTA lineage *and* consumes raw frames directly — no licensed
feature-extractor (Baidu/ResNet) second licensing surface, and a clean fit for a
`ctx.frames()`-driven stage. Released ball-action weights used **as-is**; no training in this
PRD.

**Modules to build (in-repo, permissive):**

- **T-DEED subprocess bridge** — the single boundary to the isolated env. Takes frames/clip +
  weights + params, runs the external CLI, returns parsed `SpottedEvent`s. All GPL isolation
  and all failure handling (missing env, non-zero exit, malformed/empty output, timeout)
  collapse into this one seam. It imports nothing from T-DEED.
- **`tdeed` EventSpotter stage** — `@register(StageKind.SPOTTING, "tdeed")`, implements
  `spot(ctx)` by reading `ctx.frames()`, calling the bridge, returning events, and reporting
  `provenance()` (weights id, external commit, params). A pydantic `Params` model carries
  weights path, confidence threshold, NMS/merge window, device.
- **Spotting artifact + schema** — new `ArtifactName.SPOTTING` → `spotting.json`; a pydantic
  container of `SpottedEvent { class: str, frame_idx: int, t: float, confidence: float,
  half?: int }` indexed by **source-video frame_idx** (the repo-wide artifact convention).
  Mirrored by hand in `web/src/lib/types.ts` + `web/src/lib/artifacts.ts`, plus a Lab overlay
  drawing spotted events on the timeline.
- **Event ground truth** — a timestamped-event GT representation distinct from the existing
  MOT-box `GroundTruth` (events have a time and a class, not a per-frame box). Attached to a
  video like MOT GT is, so the worker can auto-score.
- **avg-mAP metric** — a pure function `average_map(preds, gt, tolerances)` implementing
  tolerance-window matching (predictions matched to GT events within ±tolerance, greedy by
  descending confidence, one GT per prediction), per-class AP, and the mean. Wired into
  `evaluation.py` → `eval.json` → `runs.metrics` → `GET /api/benchmark`.
- **SoccerNet Ball-Action ingest** — a `matchlab_train` adapter + `ingest-soccernet-ball` CLI
  that reads SoccerNet ball-action label files and produces the event-GT JSON + a registered
  `Video` with its `gt_path`. Plus `configs/datasets/soccernet-ball.json` tier manifest
  (eval-only, non-commercial).
- **Class-taxonomy mapping table** — a documented, pure map from T-DEED's 12 classes to
  `EventType`-or-null. Documentation/consumption aid; **not enforced** on `spotting.json`.

**Modules to build (isolated, GPL, never in deps):**

- **`external-spotters/`** — sibling to `external-trackers/`, holding the vendored T-DEED
  repo, its own venv, released weights, and a thin CLI entrypoint the bridge invokes
  (frames/clip in → events JSON out). Not referenced by any `pyproject` dependency group.

**Wiring / contracts.**

- Runner: no change required — `SPOTTING` already runs between `EVENTS` and `ANNOTATE`. The
  stage additionally writes its own `spotting.json` (the runner's existing merge of `spot()`
  output into `events.json` is left intact but is not the artifact of record for spotting).
- Stage signature stays `spot(ctx)` — T-DEED is frame-only, so no widening to
  tracklets/minimap is needed.
- Configs: `configs/pipeline.tdeed-spotting-eval.yaml` (full) + a smoke config; the stage is
  toggled purely by `spotting.impl`/`enabled` (modular plug in/out by construction).
- Adding the artifact touches exactly: `ArtifactName` enum + `ARTIFACT_FILES` (core), nothing
  in the server (the endpoint resolves by logical name), `types.ts` + `artifacts.ts` (web) —
  per the documented artifact-adding contract.

**Workflow.** Isolated git worktree off the current branch; merge back to the developed repo
when tests + a smoke run are green.

## Testing Decisions

**What makes a good test here:** assert external behavior — the number a caller gets — not
internal structure. The one module whose correctness is non-obvious and load-bearing is the
metric; everything else is either a thin adapter (covered by an end-to-end smoke run) or an
I/O boundary.

**Module under test:** the **avg-mAP metric**, as a pure function.

- Feed hand-built toy prediction/GT event sets with a hand-computed expected avg-mAP@1 and
  assert equality.
- Cover the edge cases that define the metric's contract: no predictions; no GT; confidence
  ties; a prediction just **inside** vs just **outside** the tolerance window; one GT event
  that must not be double-matched by two predictions; multi-class averaging.

**Explicitly not unit-tested (by decision):** the subprocess bridge, the event-GT loader/
ingest, and the taxonomy mapping table. The bridge and stage are exercised by an end-to-end
smoke run against the smoke config; the loader is exercised by that same run consuming an
ingested fixture sequence.

**Prior art:** the tracking-metric tests around `evaluation.py`/`hota.py` (toy sequences with
known IDF1/MOTA/HOTA) are the model to follow — same "toy input, hand-computed expected
metric, assert equality" shape.

## Out of Scope

- **A shippable spotter.** GPL code and NC-trained weights make this stage reference-only.
  The clean-room, permissively-trained T-DEED-equivalent (and the permissive event-labelled
  training data it needs) is a **separate follow-up PRD** — same reference→shippable sequence
  as SPO-32/35 → shippable-tracklet.
- **Any training or fine-tuning.** Released weights, inference only.
- **Classic 17-class SoccerNet Action Spotting** and its feature-based SOTA (ASTRA/COMEDIAN)
  and the Baidu/ResNet feature-extraction pipeline they require.
- **Team-attributed spotting** (the 2025 Team-Ball-Action task / Team-mAP@1).
- **Product-pipeline consumption** of spotted events (e.g. the heuristic EventEngine or a
  product surface reading `spotting.json`). The mapping table is provided for that future
  work but nothing consumes it here.
- **Team/roster/identity coupling.** Spotting answers "what happened, when" — not "who"; it
  stays independent of the association/identity tasks per the three-tasks-are-separate
  invariant.

## Further Notes

- **Two-benchmark caveat, recorded so it isn't relitigated:** "action spotting" is two tasks
  — coarse 17-class SNAS (feature-based; ASTRA/COMEDIAN) and frame-precise Ball Action
  Spotting (end-to-end; T-DEED). This PRD deliberately targets the latter for its frame-native
  fit and current-SOTA lineage. Cross-paper avg-mAP numbers mix test-vs-challenge splits and
  tight-vs-loose tolerances; the metric we build and the numbers we quote must be pinned to a
  named split to be comparable.
- **Licensing summary to carry into the follow-up PRD:** T-DEED GPL-3.0, ASTRA GPL-3.0,
  COMEDIAN CeCILL-2.1, Spivak Apache-2.0 code / CC-BY-4.0 weights; **all** released weights are
  SoccerNet-trained (non-commercial data). Reference reimplementation + retrain on permissive
  data is the only shippable path.
- **Reference URLs:** SoccerNet Ball Action Spotting task
  (https://www.soccer-net.org/tasks/ball-action-spotting); T-DEED paper
  (https://arxiv.org/html/2404.05392) and code (GPL-3.0, https://github.com/arturxe2/T-DEED);
  SoccerNet 2025 challenge results (https://arxiv.org/html/2508.19182v1).
- **Docs to update on landing:** `implementation-status.md` (spotting = reference/
  non-shippable capability + measured avg-mAP, pinned to split + weights + revision);
  `configs/datasets/README.md` (new ball-action tier); `CLAUDE.md` only if a new
  operator command (`ingest-soccernet-ball`, the spotting eval config) warrants a one-liner.
