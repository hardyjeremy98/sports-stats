# PRD: Shippable Multi-Cue Tracklet System

**Status:** Draft for decomposition (2026-07-19).
**Owner:** Jeremy
**Precedence:** Planning document. Sits below the accepted ADRs and
`player-identity-vision.md`; where this PRD and an ADR disagree, the ADR wins.
**Supersedes:** **Phases 4 and 5** of
[`docs/prds/tracklet-modernization.md`](tracklet-modernization.md). That program's
Phases 0–3 stand (they built the measurement system and selected a tracker direction);
this PRD replaces its forward-looking build/ship phases. Core Phase 4 purity policies are
folded in here; SAM2-class correction is parked to research-watch (see Out of Scope).
**Depends on:** the SPO-34 exit gate's CAMEL-vs-TDLP decision (association head only; the
rest of the pipeline is architecture-independent — see Implementation Decisions).
**Related:** ADR 001–004, `docs/prds/tracklet-modernization.md`,
`docs/reports/2026-07-18-phase3-preregistration.md`,
`docs/reports/2026-07-18-spo3{2,5}-*-run.md`, `docs/implementation-status.md`.

---

## Problem Statement

Phase 3 of the tracklet-modernization program established, by measurement, both a direction
and a wall:

- **The direction:** multi-cue learned association (appearance + pose + motion) is the way
  to high-quality tracklets. CAMELTrack reached HOTA 0.89 / IDF1 0.93 on held-out SportsMOT
  while holding tracklet purity within a whisker of our hardened baseline — high association
  quality **without** the purity collapse that bbox-only learned association (TDLP-bbox)
  suffered. The SPO-34 gate will confirm the specific association head (CAMEL vs a full-TDLP
  head), but the multi-cue pipeline is the target.
- **The wall:** every state-of-the-art option we can download is **non-shippable**, and on
  more than one axis. The association weights are trained on SportsMOT (CC BY-NC 4.0). The
  appearance/ReID components (KPR, OSNet) are trained on research-only person-ReID datasets
  (Market-1501, Occluded-Duke). The common pose front-end (Ultralytics) is AGPL. And none of
  these systems is even *runnable on arbitrary footage* as delivered — we only produced
  tracklets from pre-computed benchmark states, never from a new video.

So we have a measured target and a benchmark result, but **no shippable, runnable tracker**.
We cannot adopt our way to a production tracklet system; we have to build a licensing-clean
equivalent that reproduces the reference's quality closely enough to be worth shipping.

Separately, we have **no owned footage in the product domain (amateur phone video) yet**.
That means product-domain validation is not currently possible, and this PRD does not pretend
otherwise: its goal is a shippable tracker that provably preserves the SOTA reference's
**benchmark** quality. Confirming that benchmark quality transfers to phone footage is a
distinct, later program, gated on data we do not have (see Out of Scope).

## Solution

Build an in-house multi-cue tracklet pipeline in which **every component is permissive on
every axis (code, weights, training data)**, that runs on arbitrary footage, and that
reproduces the non-shippable SOTA reference's quality on shared public benchmarks within a
pre-registered threshold.

The insight that scopes the work: **the pipeline is shared; only the association head is in
question.** A multi-cue tracker is detector → pose → ReID/appearance → feature assembly →
association → (offline associator + purity policies). CAMELTrack and full-TDLP differ only in
the last learned stage; everything upstream and downstream is common. So the SPO-34 choice
does not fork this PRD — it fills one slot.

Components, by how they are obtained:

1. **Detector — adopt and measure.** Start from a permissive base (YOLOX / RF-DETR, both
   Apache; RF-DETR already has an in-repo fine-tuning harness). Measure it on the benchmark
   tiers under the frozen-detections protocol. Fine-tune only if the recall tail (small,
   distant, occluded, motion-blurred players) misses the bar. No training assumed up front.
2. **Pose — adopt.** RTMPose (Apache-2.0, MMPose). Wire it in as the keypoint front-end; no
   training expected.
3. **Appearance embedder — adopt a permissive foundation embedder.** The research-only
   person-ReID checkpoints (OSNet, KPR) are *not* the only option: strong, **commercially
   licensed** general visual embedders exist — **DINOv2 (Apache-2.0)** and CLIP backbones —
   and serve as the appearance cue directly. Adopt one; fine-tune on synthetic
   (RandPerson/UnrealPerson) or owned data **only if** a measured domain gap justifies it.
   Expose it through the existing embedder-registry interface with the existing quality-gate +
   calibration. (This axis was over-scoped as "train from scratch" in the first draft — it is
   an adopt-and-maybe-adapt, like the detector and pose.)
4. **Association head — retrain (the one genuinely-trained model).** The TDLP link-prediction
   head (SPO-34) is a *lightweight matcher* trained with corrupt-and-recover augmentation, so
   it needs a modest amount of identity-labeled sequences, not a large corpus. Retrain it on a
   permissive source — **MOTSynth-class synthetic MOT (perfect GT identities, primary
   candidate)**, with pseudo-labels from a dev-only teacher over permissive video as a
   fallback. A retrain is required regardless because (a) the released TDLP weights are
   SportsMOT-trained (CC BY-NC) and (b) swapping the appearance model changes the features the
   head consumes. SportsMOT/DanceTrack stay **evaluation-only**.
5. **Assembly.** Compose the above into a registered pipeline that turns a video into raw
   tracklets, route its output through the **existing, frozen offline associator**, and add a
   **refined-tracklet layer** implementing the core Phase 4 purity policies
   (terminate-over-force; GTA-style split-and-reconnect). Raw tracklet metrics remain the
   immutable comparator; refined metrics are scored as their own layer.

**Acceptance for this PRD (two bars, one in scope now):**

- **Bar A — benchmark-parity vs the SOTA TDLP, IN SCOPE.** The north star: a **working,
  licensing-clean TDLP** (permissive detector + RTMPose + DINOv2/CLIP appearance + the
  retrained TDLP head) lands **within a pre-registered threshold of the non-shippable SOTA
  TDLP reference** on shared held-out public benchmarks (SportsMOT + SoccerNet). That
  reference is already measured (SportsMOT held-out: purity ≈ 0.95, HOTA ≈ 0.90,
  `docs/reports/2026-07-19-tdlp-full-spike.md`), so Bar A is a direct TDLP-vs-TDLP
  cost-of-shippability comparison across the full metric stack — **tracklet purity /
  mixed-track duration primary**, HOTA/AssA/IDF1 secondary, detection recall / crop yield /
  runtime / VRAM as guardrails — **and** every component passes a per-axis licensing review
  (code, weights, training data all permissive/owned).
- **Bar B — phone-footage domain acceptance, DEFERRED.** Beating the hardened baseline on
  owned phone clips by a pre-registered margin is the real product gate, but it is **out of
  scope here** because no owned footage exists yet. Recorded as the top accepted risk and the
  subject of a future PRD; nothing in this PRD is allowed to *claim* product-domain quality on
  the strength of benchmark parity alone.

Exact threshold values for both bars are **deferred to per-issue pre-registration** (the
SPO-29 precedent), fixed before each run so they cannot be retrofitted to results.

## User Stories

1. As a product owner, I want every component of the tracklet system to be permissively
   licensed on code, weights, and training-data axes, so that the whole tracker can ship
   commercially without a licensing exception.
2. As a product owner, I want a single per-axis licensing checklist carried through
   provenance and checked at acceptance, so that a non-shippable component (NC weights,
   research-only ReID data, AGPL front-end) can never reach the shipping path unnoticed.
3. As a pipeline developer, I want the in-house tracker to run end-to-end on an arbitrary
   video (not only on pre-computed benchmark states), so that it is an actual tracker rather
   than a benchmark-replay artifact.
4. As a pipeline developer, I want the association head to be a single swappable slot in an
   otherwise shared multi-cue pipeline, so that the SPO-34 CAMEL-vs-TDLP decision changes one
   component and nothing else.
5. As a researcher, I want to adopt a permissive detector base and measure it on the
   benchmark tiers before deciding to fine-tune, so that we only pay for detector training if
   the measured recall tail actually requires it.
6. As a researcher, I want RTMPose wired in as the pose front-end, so that the keypoint cue is
   shippable off-the-shelf with no training.
7. As a researcher, I want to adopt a permissively-licensed foundation embedder (DINOv2 /
   CLIP) as the appearance cue and fine-tune it only if a domain gap is measured, so that the
   appearance modality is shippable without training from scratch or depending on
   research-only person-ReID datasets.
8. As a researcher, I want the appearance embedder to reuse the existing quality-gated crop
   sampler and calibration harness, so that low-resolution or occluded crops cannot force a
   match and the embedder is calibrated the same way the offline associator's is.
9. As a researcher, I want the SPO-34-selected association head retrained on
   permissive/synthetic tracking data, so that the association model is shippable even though
   the reference was trained on non-commercial SportsMOT.
10. As a researcher, I want a pseudo-labeling pipeline that runs a dev-only reference tracker
    over a permissive video corpus to produce tracker-states for association training, so that
    we can train the association head without a licensed labelled tracking corpus.
11. As an evaluation engineer, I want the in-house stack scored against the non-shippable SOTA
    reference on the same held-out benchmark sequences under the frozen-detections protocol, so
    that Bar A measures the true cost of shippability rather than confounding it with detector
    differences.
12. As an evaluation engineer, I want the full metric stack (detection, raw tracklet, refined
    tracklet, entity) reported per layer with purity/mixed-track primary, so that a benchmark
    parity claim can never hide a purity regression behind a HOTA number.
13. As an evaluation engineer, I want Bar A and Bar B expressed as pre-registered deltas fixed
    before each run, so that acceptance is evidence-based and cannot be retrofitted.
14. As a researcher, I want the tracker's raw tracklets routed through the existing frozen
    offline associator unchanged, so that entity-level recovery is measured without modifying
    validated work.
15. As a researcher, I want a terminate-over-force policy on the online tracker, so that an
    uncertain match ends a tracklet rather than contaminating it, and the
    contamination-versus-fragmentation trade is measured explicitly.
16. As a researcher, I want a GTA-style offline split-and-reconnect stage producing a distinct
    refined-tracklet artifact scored at its own layer, so that purity gains are visible while
    the raw tracker baseline stays immutable.
17. As a researcher, I want assignment margins and competing-candidate scores logged as
    first-class tracklet metadata, so that the purity policies act on measured ambiguity rather
    than guesses.
18. As an evaluation engineer, I want every training run (ReID, association, any detector
    fine-tune) to record its dataset, split manifest, training commit/config/seed, and license
    status, so that shippable checkpoints are reproducible artifacts with clean provenance.
19. As a product owner, I want runtime, VRAM, and cost-per-match reported for the assembled
    stack on one local GPU, so that acceptance reflects deployability on the compute we have.
20. As a Lab user, I want the in-house stack's benchmark results and per-axis license status
    surfaced in the existing evaluation/benchmark views, so that the shippability decision
    lives where the other metrics already are.
21. As a downstream identity developer, I want the shippable tracker exposed as a registered
    track-stage with the same tracklet artifact contract as today, so that downstream systems
    consume it without changes.
22. As a product owner, I want product-domain (phone-footage) validation explicitly deferred
    and labelled as unproven, so that we never ship on benchmark parity mistaken for domain
    quality.

## Implementation Decisions

**Shared pipeline, one variable.** The build is a registered multi-cue track pipeline:
permissive detector → RTMPose → shippable ReID embedder → feature assembly → association head
(SPO-34 slot) → existing offline associator → refined-tracklet purity layer. Only the
association head depends on SPO-34; it is instantiated behind a stable interface so CAMEL and
a TDLP-style head are interchangeable.

**Reuse, don't rebuild.** This PRD consumes existing, validated machinery rather than
duplicating it: the offline body-ReID associator (embedder registry, quality-gated crop
sampler, embedding-artifact provenance, threshold-calibration harness) is a **frozen input**;
the benchmark runner, the frozen-detections export/import protocol, the external-tracker
import adapter, and the full metric stack (detection / raw tracklet / refined tracklet /
entity, purity + HOTA-family + IDF1/MOTA) are reused as-is.

**Detector: adopt-and-measure, not train-by-default.** A permissive base (YOLOX-Apache or
RF-DETR) is exported as frozen detections and scored on the benchmark tiers; a fine-tune is
triggered **only** if the measured recall tail (by player-height bin / miss-burst) misses the
pre-registered bar. Ultralytics-derived detectors are excluded from the shipping path (AGPL).

**Pose: RTMPose (Apache), adopted.** Keypoints feed the multi-cue feature assembly.

**ReID: shippable embedder trained on synthetic.** A generalizable-ReID embedder trained on
RandPerson / UnrealPerson (synthetic; no real-person privacy or NC-dataset dependency),
exposed through the existing embedder-registry interface and gated/calibrated with the
existing harness. Self-supervised pretraining on owned footage is a future enhancement, not a
dependency.

**Association: shippable head trained on permissive/synthetic tracking data.** The SPO-34 head
retrained via association-centric / link-prediction training over tracker-states drawn from
(a) synthetic MOT (MOTSynth-class) and/or (b) pseudo-labels from a dev-only reference tracker
run over a permissive video corpus. SportsMOT/DanceTrack are **evaluation-only**. The
association-training data source is the program's largest open risk and is scoped as its own
issue with a pre-registered fallback if no source reaches Bar A.

**Purity policies as a refined-tracklet layer.** Terminate-over-force is a tracker-level
policy (logged margins gate termination); GTA-style split-and-reconnect is an offline stage
producing a distinct refined-tracklet artifact scored at its own layer. Raw tracklet metrics
are reported unchanged alongside. SAM2-class correction is **not** implemented here.

**Licensing checklist as a first-class acceptance artifact.** Every component and checkpoint
records code / weights / training-data license per axis in provenance; the benchmark runner /
acceptance step refuses to certify a stack with any non-permissive axis on the shipping path
(the existing embedding-provenance gate is precedent).

**Frozen-detections protocol for Bar A.** The in-house stack and the SOTA reference are scored
on identical held-out benchmark detections so Bar A isolates the shippable-retrain cost from
detector differences; the reference remains a clearly-labelled non-shippable row.

## Testing Decisions

- Tests assert external behaviour — metric values, artifact contents, refusal conditions —
  never implementation details. The core technique remains handcrafted tiny sequences with
  hand-computed correct values, including known-purity contamination cases.
- **Purity policies** get dedicated tests: terminate-over-force on a handcrafted near-tied
  match must end the tracklet; GTA-style split on a handcrafted two-identity tracklet must
  split at the appearance/motion discontinuity and conservatively reconnect, with the
  refined-tracklet layer scored separately from the immutable raw layer.
- **ReID embedder** integration follows the existing fake-embedder pattern (deterministic
  fakes over real weights for pipeline-shaped tests); the quality-gate/calibration reuse is
  covered by the existing harness's tests, extended for the new embedder.
- **Licensing/provenance gate** gets refusal tests: a stack carrying any non-permissive axis
  on the shipping path must fail certification; a clean stack must pass; provenance fields must
  round-trip.
- **Benchmark-parity (Bar A)** is exercised through the existing benchmark-runner
  integration-test pattern (synthetic detector + stub configs, golden rows), extended to the
  reference-vs-in-house comparison and the tolerance-band check.
- Detector adopt-and-measure and RTMPose adoption are exercised through existing stage/test
  patterns and do not get dedicated suites initially.
- Which modules get tests is confirmed per decomposed issue; the purity policies, the ReID
  embedder interface, and the licensing gate are the priority targets.

## Out of Scope

- **Phone-footage capture and product-domain (Bar B) acceptance** — deferred to a future PRD;
  no owned footage exists yet. Benchmark parity (Bar A) must never be presented as
  product-domain quality.
- **SAM2-class mask correction** — parked to research-watch (compute-gated); only the core
  Phase 4 purity policies (terminate-over-force, GTA split/reconnect) are in scope.
- **Changes to the offline body-ReID associator or its calibration harness** — frozen input,
  reused, not modified.
- **The CAMEL-vs-TDLP decision itself** — that is the SPO-34 gate; this PRD fills the head
  slot with whatever it selects.
- **Jersey-OCR identity, roster/semantic identity, team classification, event attribution** —
  downstream of tracklets, unchanged here.
- **Live/online-product tracking** — remains an offline upload-and-process system (ADR 002).

## Further Notes

**Why Bar B is deferred rather than dropped.** The program's discipline is that public
benchmark gains are never assumed to transfer to the product domain. With no owned footage we
cannot run that test, so we scope to what is provable now (shippable + benchmark-parity) and
record product-domain validation as an explicit, named debt — not as an implicit claim. The
moment owned footage and a small labelled eval set exist, the deferred Bar B becomes its own
PRD and gate.

**Scope reality check (revised 2026-07-19).** Three of the four components are **adopt, not
train**: permissive detector (YOLOX/RT-DETR/RF-DETR, Apache), pose (RTMPose, Apache), and
appearance (**DINOv2 Apache / CLIP** — commercially licensed foundation embedders; the
research-only OSNet/KPR are not the only option). Good free commercial-use models exist for
all three — an earlier draft over-scoped appearance as "train from scratch," which was wrong.
**The one genuinely-trained model is the association head**, and even that is a *bounded*
task: a lightweight matcher, corrupt-and-recover training, needing only a modest set of
identity-labeled sequences. Its data-sourcing (SPO-39) is a **bounded spike** with
**MOTSynth (synthetic, perfect GT, permissive) as the primary candidate** and pseudo-labels
as fallback — not an open-ended "no data exists" wall. If MOTSynth's license/quality or the
fallback both fail to reach Bar A, escalate before further build investment.

**Continuity with measurement-precedes-models.** This PRD keeps the parent program's evidence
rules: held-out sequences, consistent direction across ≥2 tiers, pre-registered deltas,
immutable provenance, and per-layer metric separation. It is the build phase the measurement
system was created to serve.
