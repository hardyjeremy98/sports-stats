# Assembled licensing-clean TDLP tracker — build report

**Issues:** SPO-42 (assemble), SPO-40 (head vendor + preliminary retrain harness), SPO-38
(DINOv2 appearance), SPO-41 (licensing gate on the assembled stack), SPO-39 (association data
— blocker) · **PRD:** [`shippable-multi-cue-tracklet-system.md`](../prds/shippable-multi-cue-tracklet-system.md) ·
**Date:** 2026-07-19 · **Branch:** `spo-42-assemble-shippable-tdlp` (off main, unpushed, not merged)

**Status: RUNNABLE END-TO-END on arbitrary video. First Bar A number pending a trained head
(preliminary training in progress; shippable training BLOCKED on SPO-39 — see §5).**

## 1. What was built

A registered, licensing-clean multi-cue tracker that turns an arbitrary video into raw
tracklets, using the SPO-34-selected **TDLP link-prediction head**:

```
RF-DETR detector (Apache)  ->  tdlp-shippable track stage:
    RTMPose keypoints (Apache code/weights)  +  DINOv2 global appearance (Apache)
    -> feature assembly -> vendored TDLP head -> offline association loop
    -> Tracklet artifacts  ->  (existing offline associator, unchanged)
```

- **Vendored TDLP head** (`pitchlab_core/_vendor/tdlp/`, MIT, upstream @50344b9): the
  pure-torch link-prediction architecture only — no `motrack`/`mmdet`/`hydra`. Local
  extension: a `global_appearance` feature encoder so the head consumes a **single global
  appearance embedding** (DINOv2 CLS) instead of the upstream research-only KPR **6-part**
  features. *This means a retrain is required for architecture reasons, not only licensing.*
- **In-repo offline association loop** (`stages/track/tdlp/loop.py`): a faithful port of
  TDLP's `_convert_data`/`_association`/`_track` (causal frame-by-frame with a memory
  window), self-contained (local tracklet + SciPy Hungarian) so no third-party tracker
  dependency enters the tree — the SPO-31 vendor-and-extend pattern.
- **DINOv2 appearance embedder** (SPO-38, `embedders/dinov2.py`): timm
  `vit_small_patch14_dinov2.lvd142m`, dim 384, Apache code + **Apache weights (verified)**;
  training-data axis (LVD-142M) unspecified → flagged, not asserted clean.
- **Registered stage** `tdlp-shippable` (`StageKind.TRACK`): runs on arbitrary video, head
  behind a swappable interface (checkpoint or random-init), modalities toggleable.

## 2. Runnable end-to-end — verified

`configs/pipeline.tdlp-shippable-smoke.yaml` on `data/clips/08fd33_0.mp4` (40 frames, CUDA):
detect 12.7 s → **tdlp-shippable track 12.7 s → 24 tracklets, 859 tracklet-frames**, full
artifact set + per-stage licensing provenance written. Head was random-initialized (logged
loudly) — this is the **plumbing proof**, not a quality result. This is the SPO-42 acceptance
criterion (registered stage, arbitrary video, standard artifact, routes into the existing
associator unchanged) met.

## 3. Per-axis licensing certification (SPO-41) on the real stack

`certify_stack` over the assembled stack's actual `provenance()` correctly **refuses** it and
names the blockers (test: `test_licensing_tdlp_stack.py`):

| component | code | weights | training-data | verdict |
| --- | --- | --- | --- | --- |
| RF-DETR base | Apache | Apache (Roboflow grant) | COCO/O365 (CC BY) | **PASS** |
| RTMPose (stock body7) | Apache | Apache | **AI Challenger NC** | refuse (training_data) |
| DINOv2 ViT-S/14 | Apache | Apache | LVD-142M unspecified | refuse (training_data UNKNOWN) |
| TDLP head (untrained/NC-prelim) | MIT | none / NC-preliminary | n/a / NC | refuse (weights) |

So the shipping path is correctly **blocked today** on: RTMPose stock weights (needs a
COCO+synthetic head retrain — cheap, documented follow-up), the DINOv2 training-data axis
(product-owner sign-off — same class as the COCO/SportsMOT sign-offs), and the association
head (SPO-40, below). The gate surfaces every one rather than letting them slip.

## 4. First Bar A number — preliminary head (in progress)

The released TDLP weights are unusable (CC-BY-NC **and** KPR-6-part-shaped), so the head must
be trained on our shippable features. A **preliminary** head trainer
(`pitchlab_train/tdlp_head_train.py`, corrupt-and-recover link prediction over bbox + DINOv2
appearance) trains on the **NC eval-tier tuning splits** (SoccerNet SNMOT-116..123 + SportsMOT
tuning) — a **NON-SHIPPABLE** checkpoint whose only purpose is to de-risk the harness and give
a first Bar A data point. Eval configs (`benchmark-spo44-{soccernet,sportsmot}.yaml`) score it
over frozen reference detections (held-out) so the number isolates the head+appearance cost,
compared to the already-measured **SOTA TDLP reference (SportsMOT held-out purity ≈ 0.95–0.97,
HOTA ≈ 0.90)** and the SPO-30 comparator (SoccerNet 0.9257 / SportsMOT 0.9455 purity). Numbers
land in this report's addendum when the run completes.

## 5. BLOCKER — shippable association training data (SPO-39, HITL)

The SPO-39 spike already established (and this build confirms the premise): **MOTSynth is
disqualified** (CC-BY-NC + GTA-V EULA) — the brief's "train on MOTSynth" premise is void.
There is no off-the-shelf permissive sports-tracking training set. The permissive paths are
(a) **MEVA** (CC-BY, surveillance domain) pseudo-labelled, or (b) **PeopleSansPeople**
(Apache, Unity, needs sequence-rendering engineering) — both multi-day builds and each needs a
recorded product-owner licensing sign-off. **This gates the shippable SPO-40 retrain.** The
preliminary head (§4) unblocks everything else; the shippable head swaps in via the same
interface once a data source is chosen. **Recommendation:** stand up the MEVA pseudo-label
pipeline as primary (no new data collection, reuses the frozen-det infra), pre-register
PeopleSansPeople as the synthetic fallback — but this is a product-investment decision, so it
is flagged HITL, not taken unilaterally.

## 6. Decisions taken (autonomous)

1. **Vendor the head architecture + reimplement the loop** (not vendor `motrack`) — keeps the
   dependency tree clean and lets the SPO-43 purity policy hook the assignment margin.
2. **`global_appearance` encoder** replacing KPR 6-part — the shippable appearance shape.
3. **Motion (FoD) encoding disabled for v1** — the transformer track-encoder already models
   history; keeps the feature transform simple. Re-evaluate at the Bar A gate.
4. **Preliminary head trained on NC eval-tier tuning data**, clearly labelled non-shippable,
   to produce a first number now rather than block on the multi-day permissive-data pipeline.
5. **Pose omitted from the preliminary** (RTMPose runs on CPU here — onnxruntime-gpu absent —
   so it is the extraction bottleneck); appearance is the decisive cue per Phase-3 findings.

## 7. Tests

Vendored head forward/cost-matrix; global-appearance encoder + visibility gate; loop
assignment/gating/new-track/min-length; feature assembly; stage end-to-end (untrained,
deterministic-head correctness, multi-cue path); trainer clip-targets + loss-descent +
checkpoint-load; assembled-stack licensing refusal. Full shippable-tracker suite green
(229 passed, 2 slow-skipped).
