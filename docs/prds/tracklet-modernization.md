# PRD: Benchmark-First Tracklet Modernization Program

**Status:** Draft for decomposition; amended 2026-07-17 — Phase 2 rescoped from a
matched-data detector fine-tuning ladder to adoption of frozen reference detectors,
consistent with the Phase 0 exit-gate decision (SPO-21, detection-first; see Phase 2
rescope note). Amended 2026-07-18 — CAMELTrack moved from Phase 4 / research-watch to a
Phase 3 runnable as-published reference row, per the Phase 3 pre-registration
([`docs/reports/2026-07-18-phase3-preregistration.md`](../reports/2026-07-18-phase3-preregistration.md),
SPO-29; issue SPO-35). See the Phase 3 candidate list and the Phase 4 / research-watch notes.
Amended 2026-07-19 — **Phases 4 and 5 are SUPERSEDED** by
[`shippable-multi-cue-tracklet-system.md`](shippable-multi-cue-tracklet-system.md), which
carries the forward build/ship program (core Phase 4 purity policies folded in; SAM2 parked;
phone-footage/domain acceptance deferred pending owned data). Phases 0–3 stand.
**Date:** 2026-07-16
**Owner:** Jeremy
**Precedence:** Planning document. Sits below the accepted ADRs and
`../../../docs/player-identity-vision.md`; where this PRD and an ADR disagree, the ADR wins.
**Related:** ADR 001–004, `docs/canvases/tracklet-SOTA-roadmap.canvas.tsx` (analytical
source, verified against code 2026-07-16), `docs/implementation-status.md`.

---

## Problem Statement

MatchLab's raw tracklet system is the foundation everything else stands on. Detection quality
and short-term identity continuity bound what team classification, body and face evidence,
offline association, roster identity, minimap trajectories, event attribution, and final
player statistics can achieve. Today that foundation is a credible baseline, not
state-of-the-art: a BoT-SORT-style tracker with no learned appearance, fed by a hosted
detector whose lineage is not pinned, evaluated with a metric suite that cannot answer the
questions that matter.

Concretely, as verified against the code:

- We cannot separate detector failure from tracker failure. There is no detection-only
  evaluation, no oracle-detection experiment, and no HOTA/DetA/AssA decomposition — only
  IDF1/MOTA-family metrics computed on already-filtered tracklets.
- We cannot measure the failure mode we care most about. A tracklet that silently switches
  between players is worse than a fragmented one, because fragmentation is repairable
  offline while contamination must first be detected and split. No purity or mixed-identity
  metric exists; the merge-quality scorer collapses each tracklet to its majority ground
  truth identity, silently discarding exactly the contamination we need to see.
- We cannot trust comparisons over time. Run manifests snapshot the config but record no
  weight hashes, package versions, git revision, or evaluation-set identity. The tracker
  wrapper silently falls back to a zero-argument constructor on package API drift, meaning
  configured parameters can vanish without any signal in the results.
- The current stack carries known quality risks: all person detections enter the tracker as
  a single class and roles are reconstructed by nearest-centre matching; the detection
  confidence threshold is applied before the tracker, so low-score boxes can never be
  recovered by low-score association; short tracks are dropped before evaluation; boxes have
  no observed/predicted provenance.
- We evaluate on one dataset family (SoccerNet broadcast sequences). The product domain is
  ground-level amateur phone footage, of which we currently possess none — labeled or
  otherwise. Public-benchmark gains cannot be assumed to transfer.

Meanwhile the tracking literature has moved: sports-specific trackers (Deep-EIoU, TDLP),
learned association (CAMELTrack), tracklet split-and-reconnect refinement (GTA), and
mask-based correction (SAM2 variants) report large gains on SportsMOT and SoccerNet. We have
no harness to evaluate any of them, and no evidence-based way to decide which are worth
adopting.

## Solution

A phased, benchmark-first program that builds the measurement system before touching models,
then progressively evaluates and adopts detector and tracklet-generation candidates along an
evidence ladder from abundant public labels to scarce owned phone footage:

1. **SportsMOT** — broad multi-sport algorithm selection.
2. **SoccerNet Tracking** — soccer-specific validation (ingest tooling exists; acquisition
   and ingest verification are a Phase 0 dependency, since local data presence is an
   environment fact, not a repository guarantee).
3. **SoccerTrack** — fixed-camera amateur/fisheye transfer.
4. **Handheld sports footage (AFMOT-class)** — opportunistic bridge tier, entered only if a
   usable dataset is found and licensed.
5. **Owned phone footage** — final adaptation and acceptance; capture starts immediately
   because no usable footage exists today.

The governing objective is **high-purity tracklets**: a short pure tracklet beats a longer
tracklet that silently switches players. Uncertain online matches should terminate rather
than force continuity. Fragmentation is the offline associator's job to repair.

Five evaluation layers are kept strictly separate and never conflated:

1. **Detection quality** — per-frame box evidence.
2. **Raw online tracklet quality** — the immutable tracker output; the program's baseline
   layer, never mutated by downstream refinement.
3. **Refined/split tracklet quality** — the output of offline tracklet splitting and
   reconnection (Phase 4). Refinement changes tracker output, so it is scored as its own
   layer; calling its results "raw tracklet metrics" would destroy the immutable baseline.
4. **Offline physical-player association (entities)** — the existing body-ReID
   associator's layer; this program consumes its reusable parts but does not change it.
5. **Semantic roster identity** — ADR 004's layer; out of scope here except as a
   downstream health check.

Candidates are compared under a **frozen-detections protocol** (every tracker consumes
identical exported detections), with as-published pipelines allowed only as clearly-labeled
reference rows. External research trackers integrate through an **import adapter**
(MOT-format output scored and inspected without repo integration); only benchmark winners
are promoted to registered pipeline stages. Every comparison run carries **immutable
provenance** (weights hash, package versions, git revision, evaluation-set hash), and every
stop/go gate uses a pre-registered minimum delta evaluated on held-out sequences across at
least two dataset tiers.

## Program Phases and Gates

Each phase is independently valuable, has explicit entrance and exit criteria, and ends in a
stop/go decision. No phase assumes a later phase will run.

### Phase 0 — Measurement foundation (the minimum instrumentation phase)

**Gate question:** Can we attribute every missed track and identity switch to detection,
online association, or offline association?

**Entrance:** none (starts immediately).

**Scope:**
- Dataset acquisition and ingest verification: SoccerNet Tracking sequences ingested (or
  re-ingested — the ingest command exists but local data presence must be verified, never
  assumed) and a SportsMOT validation subset acquired and ingested, each with a recorded
  split manifest. This is a Phase 0 dependency, not a background assumption.
- Tracklet purity evaluator: per-tracklet GT composition, purity, mixed-identity duration,
  tracklets per GT player, track-length distribution — applicable to both the raw and
  refined tracklet layers.
- HOTA adapter: HOTA/DetA/AssA/LocA at tracklet and entity level via a pinned or vendored
  TrackEval, validated against reference outputs; the existing IDF1/MOTA suite is retained.
- Detection evaluator: precision/recall/AP by player-height bin, consecutive miss-burst
  length per GT track, duplicate detections, temporal box jitter.
- Oracle detector: a registered detect implementation that emits ground-truth boxes as
  detections (with optional dropout/jitter knobs), enabling the detector-independent
  tracker ceiling experiment.
- Provenance recorder: weights SHA-256, model/package versions, git revision, resolved
  input transforms/confidence/NMS, and evaluation-set hash recorded as first-class run and
  benchmark metadata. Hosted-API detector responses are cached and hashed so hosted
  detections are frozen and replayable rather than assumed repeatable.
- Generic MOT ground-truth parsers and ingest commands: SportsMOT (MOT format) and
  SoccerTrack layouts mapped into the existing ground-truth schema (which already fits;
  only parsers are missing).
- Benchmark runner v1: an offline experiment that takes a dataset manifest and a candidate
  matrix, runs the pipeline (or imports external tracklets), scores all layers, and emits
  per-sequence rows plus aggregates with provenance.
- External tracklet exchange: detections export for feeding external trackers, and a
  MOT-format importer producing standard tracklet artifacts plus a provenance sidecar.
- Non-engineering: phone-footage capture kickoff and handheld-dataset scouting begin now,
  in parallel with everything.

**Exit criteria:** the current baseline is scored on ingested SoccerNet sequences plus a
SportsMOT validation subset with the full metric stack; the oracle-detection run bounds the
tracker's detector-independent ceiling; every switch in the failure browser carries an
evidence-based layer attribution (detection, online association, refinement, offline
association) **or an explicit ambiguous tag** — oracle experiments support categorization,
they do not promise full causal attribution; repeat runs of the same config over cached
detections agree within pre-registered metric tolerances, with provenance stamped.

**Stop/go:** if oracle-detection tracklets are already near-perfect, the program re-weights
toward detection; if they are poor, tracker replacement rises in priority. Either way the
phase pays for itself.

### Phase 1 — Harden the current baseline

**Gate question:** What can the existing stack achieve before any model is replaced?

**Entrance:** Phase 0 metric stack exists (code fixes may start in parallel with Phase 0).

**Scope:**
- Remove the silent zero-argument tracker-constructor fallback; fail loudly and pin the
  tracker package version.
- Carry person class through tracking via source detection index instead of flattening to a
  single class and reconstructing by nearest centre.
- Expose all tracker parameters in configs; add per-frame box provenance (observed vs
  predicted/interpolated) to the tracklet schema before any interpolation is introduced.
- Parameter sweeps using benchmark runner: sample stride, detector confidence (including
  lowering the pre-tracker threshold so low-score association has material to work with),
  lost-track buffer, activation threshold, minimum track length (measured pre- and
  post-filter), camera-motion compensation on/off.

**Exit criteria:** a reproducible, provenance-stamped hardened baseline configuration on
held-out SportsMOT and SoccerNet sequences, with documented sensitivity to each parameter.
This baseline is the comparator for every later phase.

**Stop/go:** if parameter hardening alone closes most of the gap to the oracle ceiling,
later phases shrink accordingly.

### Phase 2 — Frozen reference detectors (rescoped 2026-07-17)

**Gate question (rescoped):** What frozen, provenance-stamped detection source does each
dataset tier use for tracker selection?

**Rescope decision (2026-07-17).** The original matched-data detector fine-tuning ladder is
skipped in favour of importing an existing checkpoint. This aligns with the Phase 0 exit
gate decision recorded the same day
([`docs/reports/2026-07-17-phase0-exit-gate.md`](../reports/2026-07-17-phase0-exit-gate.md),
SPO-21): **GO, detection-first — Phase 2 imports a standard pretrained YOLOX as the frozen
comparator detector and is prioritised ahead of Phase 3.** Rationale:

- The Phase 0 gate measured detection as the current stack's dominant error source: 63–75%
  of baseline ID switches are detection-attributed, and the football-specialised incumbent
  produces near-zero detections on SportsMOT basketball/volleyball. The remedy is domain
  coverage, which the literature's shared sports-tuned checkpoint provides off the shelf —
  a fine-tuning program is not required to close this gap, only an import. (The Phase 1
  gate corroborates the sequencing: parameter hardening closed only ~4% of the gap to the
  oracle ceiling.)
- On the public leaderboards themselves, that shared YOLOX is near-saturated on detection
  and methods differ almost entirely in association — so adopting it both fixes our
  measured detection deficit and makes Phase 3 tracker results directly comparable to
  published SportsMOT numbers, which an in-house fine-tune would not be.
- Any detector fine-tuned on SportsMOT — including an in-house one — inherits the dataset's
  CC BY-NC 4.0 license on the training-data axis, so the ladder could never have produced a
  shipping detector; the shipping detector was always going to come from the Phase 5
  phone-footage fine-tune. (Gate caveat, restated: the comparator detector is not the
  shipped detector — the production amateur-footage detector question stays open until
  Phase 5.)

**Entrance:** Phase 0 external-exchange tooling (detections export) exists.

**Scope (what replaces the ladder):**
- Adopt existing downloadable checkpoints as the frozen reference detector per tier:
  - **SportsMOT tier:** MixSort's SportsMOT-fine-tuned YOLOX (MIT-licensed repository;
    weights trained on CC BY-NC SportsMOT, therefore recorded in provenance as
    **selection-only, non-shippable**). This is the shared public detector of the SportsMOT
    literature.
  - **Soccer tier (SoccerNet + soccer clips):** the hosted incumbent detector, frozen and
    replayable via the hosted-response cache. Local AGPL YOLO weights remain a
    non-shippable local reference.
  - **Excluded:** Deep-EIoU's YOLOX checkpoint — its repository and weights carry no
    license, and unlicensed code/weights grant no execution right.
- Export frozen detections once per tier over the evaluation sequences, hash them, and feed
  every Phase 3 tracker candidate identical input.
- Detector architecture selection is **deferred to Phase 5** as a two-candidate bake-off
  fine-tuned on owned phone footage: YOLOX (Apache 2.0 code and COCO weights) and RF-DETR
  (fine-tuning harness already exists). D-FINE and RT-DETRv2 drop to research watch.

**Exit criteria:** frozen, hashed, provenance-stamped reference detections exist for the
held-out SportsMOT and SoccerNet evaluation sequences, with license status (selection-only
vs shippable) recorded per checkpoint.

**Revisit trigger (replaces the stop/go):** the Phase 0 gate already established a large
detection-attributable gap for the *incumbent* detector; the import above is the chosen
remedy. The matched-data ladder below re-enters scope only if (a) after the imported YOLOX
detections are scored, a large detection-attributable gap **persists** on the sports tiers
(i.e. the import did not close what the gate measured), or (b) Phase 5 transfer fails for
detection-attributable reasons (small/distant players, motion blur, missed detections on
phone footage).

**Conditional annex — superseded matched-data ladder (retained only for the revisit
trigger):** fine-tuned RF-DETR Medium/Large, D-FINE, RT-DETRv2, and YOLOX, each fine-tuned
on the same dataset and split manifest with 2–3 training seeds, judged on the detection
evaluator suite plus downstream raw tracklet IDF1/HOTA/purity, crop yield, runtime, and
VRAM, with as-published checkpoints as clearly-labeled reference rows only, and promotion
requiring the pre-registered delta on held-out sequences in at least two tiers. If this
annex is ever activated, the fine-tuning dataset must be permissively licensed or owned for
any checkpoint intended to outlive benchmark selection.

### Phase 3 — Online tracker benchmark ladder

**Gate question:** Which online tracker produces the purest raw tracklets on frozen
detections, and does body appearance reduce within-team switches?

**Entrance:** frozen reference detections exported per tier (Phase 2 as rescoped); tracker
evaluation can begin as soon as they exist, since the frozen-detections protocol makes the
detector a swappable input.

**Candidates:**
- Hardened BoT-SORT-style baseline (comparator).
- Full BoT-SORT with quality-gated body ReID, reusing the existing embedder registry, crop
  quality gate, and calibration approach from the offline associator work.
- TDLP bbox-only and TDLP + body ReID (MIT, official SportsMOT weights) via the import
  adapter — the pair directly measures the marginal value of appearance in learned
  association.
- OC-SORT (lightweight motion-model ablation).
- Deep-EIoU as the sports-specific reference upper bound. Its official repository has no
  clear license, so it is a **paper-only reference** (published numbers, protocol caveats
  noted) unless licensing is clarified; if a runnable reference is genuinely needed, its
  ideas (ExpansionIoU, iterative matching) are clean-room reimplemented instead of
  executing the unlicensed code.
- CAMELTrack as a **runnable, clearly-labeled as-published reference row** (added
  2026-07-18, SPO-29 pre-registration / SPO-35). Apache-2.0 code (runnable via the import
  adapter, unlike Deep-EIoU); SportsMOT-trained weights → selection-only, non-shippable. Its
  SOTA checkpoint consumes pose keypoints, breaking frozen-detection input parity, so it is a
  reference row (SportsMOT native + SoccerNet transfer), **not** a promotion candidate under
  the primary-delta gate. Native multi-cue appearance — does not bear on the "marginal value
  of appearance" question, which stays scoped to the BoT-SORT+body-ReID pair.

**Body-ReID integration experiment (explicitly separate):** the offline `global-reid`
associator is frozen during this phase. Moving the same embedder into online tracking is a
controlled experiment whose success metric is raw tracklet IDF1/switches/purity — an
offline body-ReID change must affect entity-level metrics only, and any run where an
offline-layer change moves raw tracklet metrics indicates a harness bug. Appearance must be
quality-gated so low-resolution or occluded crops cannot force a match.

**Exit criteria:** a tracker winner (or confirmation of the baseline) under the promotion
objective hierarchy (purity/mixed-track duration primary; HOTA/AssA and IDF1 secondary;
detection recall, crop yield, runtime, and VRAM as guardrails) on one local GPU, consistent
across SportsMOT and SoccerNet held-out sequences.

**Stop/go:** import-adapter candidates that win are promoted to registered pipeline stages;
losers are documented and dropped. If appearance shows no within-team switch reduction, the
online-ReID line stops and the program leans on Phase 4 policies instead.

### Phase 4 — Tracklet purity as the control policy

> **SUPERSEDED (2026-07-19)** by
> [`shippable-multi-cue-tracklet-system.md`](shippable-multi-cue-tracklet-system.md). Its core
> policies (terminate-over-force, GTA-style split/reconnect) are folded into that PRD as a
> refined-tracklet layer; SAM2-class correction is parked to research-watch. Retained below for
> historical context.

**Gate question:** Can one policy detect ambiguity before it creates mixed tracklets,
across sports, without dataset-specific threshold overfitting?

**Entrance:** Phase 3 winner integrated; assignment-margin instrumentation added to it.

**Scope:**
- Log assignment margins and competing candidate scores as first-class tracklet metadata.
- Terminate-over-force policy: prefer ending a tracklet to accepting a near-tied match;
  measure the contamination-vs-fragmentation trade explicitly.
- GTA-style offline tracklet splitting and reconnection: split where appearance or motion
  changes abruptly, then conservatively reconnect. This operates between raw tracking and
  the existing offline associator and produces a distinct **refined-tracklet artifact**
  scored at its own layer: raw tracklet metrics remain the immutable tracker baseline
  (unchanged by construction), refined-tracklet metrics score the split/reconnect output,
  and entity metrics confirm downstream recovery.
- Constraint taxonomy applied to assignment decisions — hard impossibility constraints
  (one-to-one assignment, temporal non-overlap, entry/exit boundaries, physically plausible
  movement, pitch-space reachability when calibration confidence permits) versus soft
  evidence (appearance similarity, team/role consistency, formation and positional
  tendencies, manual same/different verdicts). Hard constraints may veto; soft evidence may
  only re-rank.
- Selective SAM2 correction enters here only against measured hard windows — ambiguous
  crowded intervals identified by margin logs — not as a whole-pipeline replacement,
  compute-gated on the single-GPU budget. *(Amended 2026-07-18: CAMELTrack, previously
  listed here as a selective hard-window corrector, was moved to Phase 3 as a runnable
  as-published reference row — SPO-29 pre-registration / SPO-35. A Phase 4 selective-window
  use remains possible but is no longer the primary way CAMELTrack enters the program.)*

**Exit criteria:** refined-tracklet mixed-track duration falls on at least two sports
without per-dataset threshold tuning, and entity-level metrics remain non-inferior
(fragmentation added by splitting must be recoverable by the offline associator). Raw
tracklet metrics are reported unchanged alongside, as the immutable comparator.

**Stop/go:** each policy/tool is retained only if the purity gain survives held-out
evaluation; SAM2-class correction additionally must justify its runtime per match.

### Phase 5 — Transfer to amateur, handheld, and phone footage

> **SUPERSEDED (2026-07-19)** by
> [`shippable-multi-cue-tracklet-system.md`](shippable-multi-cue-tracklet-system.md). Product-domain
> (phone-footage) validation is deferred there as an explicit accepted risk pending owned data;
> the detector question is reframed as adopt-and-measure. Retained below for historical context.

**Gate question:** Which gains survive the move from labelled broadcast benchmarks to the
product domain?

**Entrance:** tracker finalist(s) from Phase 3/4 plus the detector bake-off shortlist
(YOLOX-COCO, RF-DETR — per the Phase 2 rescope); SoccerTrack ingested; phone footage
captured (started in Phase 0) and a small phone evaluation set labeled; handheld tier
entered only if the scouting task found a usable licensed dataset, otherwise finalists jump
from SoccerTrack to phone footage with wider pre-registered acceptance margins.

**Scope:**
- Full SoccerTrack evaluation (scale, distortion, amateur players, fixed camera).
- Handheld tier evaluation if available (camera motion, blur, collisions, zoom).
- Phone evaluation set: a handful of 30–60 second clips, annotated with model-assisted
  pre-labels from the best stack plus manual correction, held out from all tuning, with
  condition slices (player scale, occlusion, camera motion).
- Final adaptation: the deferred detector bake-off — fine-tune both shortlist
  architectures (YOLOX from Apache COCO weights, RF-DETR via the existing harness) on a
  small match-separated phone set and select by the pre-registered delta;
  temporally-consistent pseudo-labels to expand training data; re-tune camera compensation
  and blur handling for phone motion.
- Cost accounting: runtime, VRAM, and cost per match for the finalist stack.

**Exit criteria (program acceptance):** the chosen stack beats the hardened baseline on
held-out phone clips by the pre-registered margin on purity and raw IDF1/HOTA, passes
licensing review for the shipping path, and fits the cost-per-match budget. Public-tier
gains that do not reproduce on phone footage are recorded as domain-limited findings, not
shipped.

### Dependencies and parallelization

- Phase 0 blocks all evaluation work; it is the program's critical path.
- Baseline code hardening (Phase 1 fixes) can proceed in parallel with Phase 0 metrics.
- Phone capture and handheld-dataset scouting run in parallel with all phases from day one.
- Phase 2 (as rescoped) is a short adoption-and-export task; Phase 3 tracker evaluation
  starts as soon as frozen reference detections are exported.
- Benchmark sweeps and the Phase 5 detector fine-tunes must be serially scheduled on the
  single local GPU; the PRD's phase gates assume wall-clock, not parallel compute.
- Phases 4 and 5 are strictly sequential after Phase 3.

### Candidate triage

**Immediately actionable:** oracle detector; hardened BoT-SORT baseline; MixSort's public
SportsMOT-fine-tuned YOLOX as the frozen SportsMOT-tier reference detector (MIT repository;
NC training data → selection-only, non-shippable); TDLP bbox-only and TDLP+ReID via import
adapter; OC-SORT; GTA-style split/reconnect; full BoT-SORT+ReID reusing the existing
embedder stack. Deferred to Phase 5: YOLOX-COCO (Apache) and RF-DETR (harness exists) as
the phone-footage fine-tuning shortlist.

**Benchmark references:** Deep-EIoU (no clear license — paper-only reference or clean-room
reproduction, never executed as-is; its YOLOX checkpoint is equally unlicensed and is not
executed either); local AGPL YOLO weights (AGPL permits local execution; non-shippable per
existing policy); hosted incumbent detector (soccer-tier reference; not a cross-sport
comparator, provenance-limited, frozen via the hosted-response cache).

**Research watch (re-evaluate when code/weights/licenses mature):** D-FINE and RT-DETRv2
(re-enter only via the Phase 2 revisit trigger); SAMIDARE, HyperSSM, NOOUGAT; SAM2-class
mask tracking as a whole-pipeline approach (selective correction on hard windows is the
only sanctioned entry point, in Phase 4). *(CAMELTrack was on research-watch here; promoted
2026-07-18 to a Phase 3 runnable as-published reference row — SPO-29 pre-registration /
SPO-35 — its Apache-2.0 code making it executable via the import adapter.)*

## User Stories

*(2026-07-17 rescope note: stories 19, 20, and 35 now apply only to the Phase 5 detector
bake-off and to the Phase 2 conditional annex if its revisit trigger fires.)*

1. As a pipeline developer, I want an oracle-detection mode that feeds ground-truth boxes
   into any tracker, so that I can separate detector limitations from association
   limitations before choosing what to replace.
2. As a pipeline developer, I want HOTA, DetA, AssA, and localization accuracy computed on
   every ground-truth-scored run, so that I can compare our stack against published sports
   tracking results on their own terms.
3. As a pipeline developer, I want per-tracklet purity and mixed-identity duration metrics,
   so that contaminated tracklets are measured directly instead of being silently collapsed
   into majority-identity scores.
4. As a pipeline developer, I want detection precision, recall, and AP reported by
   player-height bin, so that small distant players cannot hide behind good frame-average
   numbers.
5. As a pipeline developer, I want consecutive detection-miss burst lengths per ground-truth
   track, so that I can see the misses that actually fragment tracks rather than only a
   global miss rate.
6. As a pipeline developer, I want duplicate-detection and temporal box-jitter metrics, so
   that loose or unstable boxes that damage IoU matching and contaminate body crops are
   visible.
7. As a pipeline developer, I want tracklets-per-ground-truth-player and track-length
   distributions, so that I can distinguish repairable fragmentation from silent identity
   swaps.
8. As an evaluation engineer, I want every comparison run stamped with model architecture
   and revision, checkpoint hash, package versions, training lineage, input transforms,
   confidence/NMS settings, license status, git revision, and evaluation-set hash, so that
   any two results can be compared or reproduced months later without archaeology.
9. As an evaluation engineer, I want the tracker wrapper to fail loudly when its
   constructor signature drifts, so that configured parameters can never silently vanish
   from an experiment.
10. As an evaluation engineer, I want a benchmark runner that takes a dataset manifest and
    a candidate matrix and emits per-sequence rows plus aggregates, so that a full
    evaluation campaign is one reproducible command instead of hand-assembled runs.
11. As an evaluation engineer, I want results sliced by dataset tier, sport, camera type,
    player scale, occlusion, and camera motion, so that headline averages cannot hide
    domain-specific regressions.
12. As an evaluation engineer, I want stop/go gates with pre-registered minimum deltas on
    held-out sequences across at least two tiers, so that promotion decisions are made on
    evidence rather than enthusiasm.
13. As a researcher, I want SportsMOT and SoccerTrack sequences ingested into the same
    ground-truth format as SoccerNet, so that every dataset tier is scored by the same
    evaluator.
14. As a researcher, I want to export frozen detections per detector and sequence, so that
    every tracker candidate consumes identical input and association quality is isolated
    from detection quality.
15. As a researcher, I want to import MOT-format tracklets produced by an external research
    repository and score them with the full metric stack, so that evaluating a new tracker
    does not require integrating its dependencies into our environment.
16. As a researcher, I want imported external results tagged with their own provenance and
    license status, so that reference-only systems can never silently enter a shipping
    comparison.
17. As a researcher, I want as-published pipeline results recorded as clearly-labeled
    reference rows alongside frozen-detection results, so that literature reproduction and
    controlled comparison do not get conflated.
18. As a researcher, I want the tracker parameter space (stride, confidence, lost buffer,
    activation threshold, minimum length, camera compensation) sweepable through the
    benchmark runner, so that the current baseline is fully hardened before any model is
    replaced.
19. As a researcher, I want detector fine-tuning experiments to record their dataset split
    manifest, training commit, configuration, and seed, so that fine-tuned checkpoints are
    reproducible artifacts rather than loose files.
20. As a researcher, I want training variance measured once per fine-tuned architecture
    with a small number of seeds, so that promotion deltas can be judged against known
    noise without paying multi-seed cost on every gate.
21. As a researcher, I want assignment margins and competing candidate scores logged during
    tracking, so that ambiguous windows can be identified, measured, and targeted by
    selective correction instead of guessing where trackers struggle.
22. As a researcher, I want a terminate-over-force policy evaluated explicitly on the
    contamination-versus-fragmentation trade, so that the purity asymmetry is enforced by
    measurement rather than assertion.
23. As a researcher, I want GTA-style splitting and reconnection scored as a separate
    refined-tracklet layer with raw tracklet metrics reported unchanged alongside, so that
    purity gains are visible without losing the immutable tracker baseline.
24. As a researcher, I want the online body-ReID experiment to reuse the offline
    associator's embedder interface, quality-gated crops, and calibration method while the
    offline associator itself stays frozen, so that the marginal value of online appearance
    is measured without disturbing validated work.
25. As a researcher, I want quality-approved body and face crop yield per player reported
    for every detector and tracker candidate, so that a stack cannot win on tracking
    metrics while starving downstream identity evidence.
26. As a Lab user, I want a curated subset of benchmark sequences inspectable in the Lab
    with the existing overlay and failure-browser tooling, so that metric deltas can be
    traced to visible behavior.
27. As a Lab user, I want purity and HOTA-family numbers visible in the run evaluation
    view, so that the new decision metrics appear where existing metrics already live.
28. As a product owner, I want candidates classified as immediately actionable, benchmark
    reference, or research watch, so that engineering effort is spent only where
    implementation maturity and licensing permit.
29. As a product owner, I want runtime, VRAM, and cost per match reported for every
    finalist stack, so that acceptance reflects deployability on the compute we actually
    have.
30. As a product owner, I want licensing and commercial-use status carried through
    provenance and checked at the final gate, so that a non-shippable component cannot
    reach the shipping path unnoticed.
31. As a product owner, I want phone-footage capture and handheld-dataset scouting started
    in the program's first phase, so that the final acceptance tier exists by the time
    finalists are ready instead of blocking them.
32. As a future annotator, I want the phone evaluation set produced by model-assisted
    pre-labels plus manual correction under a documented protocol, so that scarce labeling
    effort yields a trustworthy held-out acceptance set.
33. As a downstream identity developer, I want detection, raw tracklet, refined tracklet,
    entity, and semantic identity metrics reported as separate layers on every scored run,
    so that an improvement in one layer can never masquerade as an improvement in another.
34. As an evaluation engineer, I want hosted-API detector responses cached and hashed and
    gate comparisons made within pre-registered metric tolerances, so that non-bitwise-
    deterministic GPU inference and hosted endpoints cannot silently invalidate a
    comparison.
35. As an evaluation engineer, I want detector candidates compared in matched-data and
    as-published classes that are never mixed in one table, so that architecture choices
    are not confounded by checkpoint lineage.

## Implementation Decisions

**Evaluation and metrics**
- The HOTA family (HOTA/DetA/AssA/LocA) comes from TrackEval, pinned or vendored behind the
  existing evaluation dependency extra, validated against reference outputs. The existing
  motmetrics-based IDF1/MOTA suite is retained unchanged; each backend is authoritative for
  its own metrics. TrackEval's numpy-2 compatibility is verified at adoption time (the
  motmetrics IoU workaround is precedent).
- New pure-function evaluators for detection quality (P/R/AP by height bin, miss bursts,
  duplicates, jitter) and tracklet purity (per-tracklet GT composition, mixed-identity
  duration, tracklets per GT player, length distributions) are added alongside the existing
  evaluator and folded into the same evaluation artifact, at both tracklet and entity level
  where applicable.
- Minimum-length filtering is measured pre- and post-filter so short valid tracks are
  visible as a policy cost rather than silently discarded.

**Harness and orchestration**
- The benchmark harness is offline-first: a config-driven experiment in the training
  package (following the existing ablation-experiment pattern), sweep-capable, requiring no
  server or database. The Lab is used for inspection of a curated subset of sequences, not
  as the benchmark executor. Benchmark matrices come from experiment results, not the run
  table; the existing benchmark API remains for Lab-run aggregation.
- External trackers integrate through an import adapter: exported detections feed the
  external system, and its MOT-format output is imported as standard tracklet artifacts
  plus a provenance sidecar. Only benchmark winners are promoted to registered track-stage
  implementations. Losers never enter the dependency tree.
- Frozen detections are the primary comparison protocol; as-published detector+tracker
  combinations are permitted only as labeled reference rows.
- The oracle detector is a registered detect-stage implementation reading the video's
  ground truth, with optional dropout and jitter parameters for sensitivity analysis.

**Provenance**
- A provenance block becomes first-class run metadata: weights SHA-256, model architecture
  and revision, package versions, pretraining/fine-tuning lineage and dataset split
  manifest, training commit/config/seed where applicable, input transforms and
  confidence/NMS/tiling/TTA settings, license and commercial-use status, git revision, and
  evaluation-set hash. The benchmark runner refuses to aggregate runs whose provenance is
  missing or inconsistent (the existing embedding-artifact provenance gate is precedent).

**Baseline hardening**
- The silent zero-argument tracker-constructor fallback is replaced with a loud failure and
  a pinned tracker package version.
- Person class is carried through tracking via source detection index; nearest-centre class
  reconstruction is removed.
- The tracklet schema gains per-frame box provenance (observed vs predicted/interpolated)
  before any smoothing or interpolation is introduced; current artifacts contain only
  matched detection boxes and must not imply otherwise.
- Detector confidence, activation threshold, and low-score association are tuned jointly,
  since the current pre-tracker threshold starves low-score recovery.

**Boundaries with existing work**
- The offline body-ReID associator (embedder registry, quality-gated crop sampler,
  embedding artifacts with provenance meta, threshold calibration harness) is complete,
  merged work. This program does not modify it. Its reusable parts are inputs to online
  tracker candidates; moving embeddings online is a separate controlled experiment measured
  on raw tracklet metrics with the offline associator frozen.
- An offline association change must never move raw tracklet metrics; the harness treats
  such movement as a defect.
- Constraint taxonomy: hard impossibility constraints (one-to-one assignment, temporal
  non-overlap, entry/exit boundaries, physically plausible movement, pitch-space
  reachability when calibration confidence permits) may veto assignments; soft evidence
  (appearance, team/role consistency, formation and positional tendencies, manual
  same/different verdicts, roster exclusivity and substitution timing where relevant) may
  only re-rank. Team/role consistency is treated as soft because role classification is
  itself a prediction.

**Licensing and compute**
- Code, weights, and training datasets are licensed separately; provenance records license
  status per axis, and every result row carries it. A permissive codebase does not make its
  checkpoints or their training data permissive, and vice versa — YOLOX's Apache code is
  shipping-eligible even where a specific checkpoint is not.
- Shipping-path components must be permissively licensed on every applicable axis.
- Local benchmark execution itself requires a license that permits it. AGPL code may run
  locally as a clearly-tagged non-shippable reference. Code with no license grants no
  execution right: unclear-license implementations (e.g. Deep-EIoU) are paper-only
  references or clean-room reproductions until their status is clarified.
- The program budget is one local consumer GPU. Sweeps and fine-tunes are serially
  scheduled; candidate scope (notably SAM2-class correction) is gated on measured runtime;
  tracker-training work is deferred in favor of published weights.

**Datasets and annotation**
- Dataset tiers: SportsMOT, SoccerNet Tracking (present), SoccerTrack, opportunistic
  handheld tier, owned phone footage. Each tier gets source-separated train/validation/test
  usage with recorded split manifests; held-out sequences are never used for tuning.
- Phone capture begins in Phase 0. The phone evaluation set is a small number of 30–60
  second clips labeled via model-assisted pre-labels plus manual correction, with condition
  metadata, held out from all adaptation.
- The handheld tier is entered only if scouting finds a usable licensed dataset; otherwise
  Phase 5 proceeds from SoccerTrack to phone footage with wider pre-registered margins.

**Evidence rules**
- Stop/go gates require: held-out sequences, consistent direction across at least two
  dataset tiers, and a minimum delta pre-registered before the run (for example, a stated
  IDF1 gain or mixed-track-duration reduction).
- Promotion objective hierarchy, applied at every gate: **primary** — mixed-track duration
  and tracklet purity; **secondary** — HOTA/AssA and IDF1 at the appropriate layer;
  **guardrails** that must not regress beyond stated bounds — detection recall,
  quality-approved crop yield per player, runtime, and VRAM. Added fragmentation is
  acceptable only when entity-level recovery remains non-inferior.
- Determinism is not assumed. GPU inference and hosted APIs need not be bitwise
  deterministic: comparisons run over cached, frozen detections (hosted responses cached
  and hashed); repeat-run stability is measured once per harness; and gates use
  pre-registered metric tolerances, not exact-match expectations. Fine-tuned detector
  architectures additionally get 2–3 training seeds once to establish variance bands
  (after the 2026-07-17 rescope: the Phase 5 bake-off, plus the Phase 2 annex only if its
  revisit trigger fires).
- Public benchmark gains are never assumed to transfer to phone footage; transfer is
  measured at Phase 5 and non-transferring gains are recorded as domain-limited findings.

## Testing Decisions

- Tests assert external behavior — metric values, artifact contents, refusal conditions —
  never internal implementation details. The core technique is handcrafted tiny sequences
  (a few frames, a few tracks) whose correct metric values are computed by hand, including
  known-purity cases: a tracklet spanning two ground-truth identities must report its
  contamination and mixed duration exactly.
- Modules under dedicated test: the ground-truth parsers (SportsMOT/SoccerTrack fixtures),
  the detection evaluator, the purity evaluator, the HOTA adapter (validated against
  TrackEval reference outputs on fixture sequences), the oracle detector, the external
  exchange (export/import round-trip preserving boxes, ids, and provenance; refusal on
  malformed or provenance-less input), and the provenance recorder (stable hashing,
  detection of mismatched evaluation sets).
- The benchmark runner gets a dedicated integration-test suite: it is the program's
  decision-making backbone, and indirect coverage is too weak for the component every gate
  depends on. It is exercised end-to-end with the synthetic detector and stub configs
  against golden per-sequence rows and aggregates, including refusal on missing or
  inconsistent provenance, tolerance-band comparison behavior, and the import path for
  external tracklets.
- Baseline-hardening changes are exercised through existing test patterns and do not get a
  dedicated suite initially.
- Prior art to follow: the existing ground-truth evaluation tests and the fake-embedder
  tests for the re-ID associator (deterministic fakes over real model weights; synthetic
  stage implementations for pipeline-shaped tests).

## Out of Scope

- Event spotting and event-attribution improvements, except where existing event artifacts
  serve as passive downstream health checks.
- Final roster identity product behavior, semantic identity resolvers, and roster
  enrollment.
- Changes to the offline body-ReID associator or its calibration harness (frozen input to
  this program, not a subject of it).
- General product UI redesign; Lab changes are limited to surfacing new metrics and
  inspecting curated benchmark sequences.
- Implementing every research model: SAMIDARE, HyperSSM, NOOUGAT, and whole-pipeline mask
  tracking remain watch items unless their gate conditions are met.
- Jersey OCR as an identity foundation (ADR 001); OCR remains optional reference evidence.
- Tracker training/fine-tuning beyond published weights (deferred unless a Phase 3 winner
  demonstrably needs domain adaptation, which would be scoped separately).
- Live/online-product tracking constraints; this remains an offline upload-and-process
  system (ADR 002).

## Further Notes

**Smallest first implementation slice.** The purity evaluator, the HOTA adapter, and the
oracle detector, wired into a minimal benchmark-runner pass over SoccerNet sequences,
scoring the current baseline. Verifying (or refreshing) the local SoccerNet ingest is part
of the slice, not an assumption — local dataset presence is an environment fact. No new
models, no UI. This single slice converts the program's central question — is the binding constraint
detection or association, and how much contamination do current tracklets carry? — from
opinion into measurement, and every subsequent phase consumes its output. SportsMOT
ingestion is the immediate follow-up, not part of the first slice.

**Why measurement precedes models.** The repository's own history is the argument: kit-colour
association was adopted plausibly and measured ineffective; the remaining switches were
found to be tracker-level only after entity-level evaluation existed. This program's
premise is that the same discipline, applied one layer down, is cheaper than any premature
model adoption.

**Canvas verification note.** The roadmap canvas (July 2026) was verified against code on
2026-07-16. Its current-stack claims all held (class flattening, silent constructor
fallback, pre-tracker confidence gating, pre-evaluation length filter, missing provenance,
missing HOTA/purity/detection metrics, no oracle path). Two of its framings were stale: the
body-ReID offline associator is merged, not in progress, and an RF-DETR fine-tuning harness
already exists. This PRD supersedes the canvas as the program's planning document; the
canvas remains an analytical artifact.

**Decomposition guidance.** Phases 0 and 1 decompose naturally into one issue per module
(each independently reviewable with its own tests); Phases 2–5 decompose into one issue per
candidate or per gate. Every issue inherits this PRD's evidence rules and provenance
requirements. Pre-registered deltas for each gate should be fixed in the corresponding
issue before its benchmark runs, not retrofitted.
