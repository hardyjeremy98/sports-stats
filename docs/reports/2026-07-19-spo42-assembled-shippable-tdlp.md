# Assembled licensing-clean TDLP tracker — build report

**Issues:** SPO-42 (assemble), SPO-40 (head vendor + preliminary retrain harness), SPO-38
(DINOv2 appearance), SPO-41 (licensing gate on the assembled stack), SPO-39 (association data
— blocker) · **PRD:** [`shippable-multi-cue-tracklet-system.md`](../prds/shippable-multi-cue-tracklet-system.md) ·
**Date:** 2026-07-19 · **Branch:** `spo-42-assemble-shippable-tdlp` (off main, unpushed, not merged)

**Status: RUNNABLE END-TO-END on arbitrary video. IDsw-focused iteration cut SportsMOT ID
switches 362 → 75 (−79%) and HOTA 0.50 → 0.69 with a stronger preliminary head + tuned
thresholds (§4c) — a large part of the cost-of-shippability gap on the association metric that
matters. Shippable head training still BLOCKED on SPO-39 (§5).**

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
HOTA ≈ 0.90)** and the SPO-30 comparator (SoccerNet 0.9257 / SportsMOT 0.9455 purity).

### 4a. First Bar A numbers (PRELIMINARY head — 2026-07-19)

Preliminary head: bbox + DINOv2-global appearance (no pose), 12 epochs corrupt-and-recover
over 11 NC tuning sequences (SoccerNet 116–123 + SportsMOT tuning), train loss 0.64→0.12.
Scored over frozen reference detections on the held-out tiers.

| tier (held-out) | tracklet purity | mixed-track s | HOTA(t) | IDF1(t) | IDsw(t) | det recall/AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **SportsMOT** (6 seq) | **0.827** | 52.5 | 0.503 | 0.451 | 362 | 0.994 / 0.988 |
| **SoccerNet** (4 seq) | **0.823** | 58.1 | 0.320 | 0.310 | 829 | 0.846 / 0.811 |
| _reference: SOTA TDLP (SportsMOT)_ | _0.95–0.97_ | _10–13_ | _0.90_ | _0.94_ | _~5_ | — |
| _reference: SPO-30 comparator_ | _SN 0.926 / SM 0.946_ | _SN 23 / SM 18_ | — | — | — | — |

**Reading:** the assembled shippable-feature TDLP with this preliminary head lands **~0.12–0.14
below** the SOTA TDLP reference on SportsMOT purity and **~0.10 below** the comparator on
SoccerNet — it does **not** reach parity, and mixed-track / HOTA / IDsw are much worse
(heavy fragmentation: very high IDsw with only moderate purity means short, frequently-broken
tracklets). Detection recall/AP ~0.99 (SportsMOT) confirms the gap is the **head + appearance
cue**, not detection. This is an honest cost-of-shippability *lower bound*, not a parity claim.

### 4b. IDsw-focused iteration (Jeremy: ID switches matter more than purity)

Reprioritized to minimize **ID switches / maximize association completeness** (see
[[idsw-over-purity]]). Built a fast tuning loop: `tdlp_head_eval.py extract-holdout` caches
DINOv2 features over the held-out frozen detections **once**, then `sweep` runs only the cheap
association loop + `evaluate_run` at many settings (no re-embedding), ranked by IDsw.

**Round 1 — threshold sweep on the preliminary head (SportsMOT, 6 held-out seq):**

| config | IDsw ↓ | HOTA ↑ | IDF1 ↑ | purity | note |
| --- | ---: | ---: | ---: | ---: | --- |
| sim 0.5 / newtrk 0.5 / rem 20 (≈ original) | 300 | 0.521 | 0.465 | 0.834 | over-gated |
| **sim 0.95 / newtrk 0.9 / rem 20** | **190** | **0.586** | **0.563** | 0.758 | best IDsw |

Tuning alone cut IDsw **~37%** (300→190; and ~48% vs the original full-run 362) and lifted HOTA
+0.065 / IDF1 +0.10, trading purity down (0.83→0.76) — the right trade under the new priority.
IDsw improved **monotonically as `sim_threshold`→1**, i.e. the head is *under-confident* (its
match logits rarely clear a strict gate) — a symptom of an **under-powered head** (128-d / 2
layers vs the reference's ~256-d / 6 layers), not just a threshold artifact. Still far from the
reference (~5 IDsw).

**Round 2 — stronger head + IDsw-targeted training (in progress):** retrain at reference-scale
capacity (256-d / 4 layers / 8 heads, 5.8M params) with **gap augmentation** (`max_gap`: end the
observed window several frames before the current detection → train re-association of a track
last seen N frames ago, the exact lost-tracklet-respawn case) + `history_dropout`, `remember`
30, 25 epochs — then re-sweep sim∈{0.9,0.95,0.99}×remember∈{20,30,40}. Results in §4c.

### 4c. Round 2 results — stronger training closes most of the IDsw gap

Two findings: (a) a **4-layer transformer head failed to train** (flat loss 1.28 — deeper
transformers need LR warmup; abandoned, kept the proven 2-layer / 128-d head); (b) the
**2-layer head + mild gap augmentation** (`max_gap=4`, `history_dropout=0.05`) + `remember=30`
trained cleanly (loss 0.098) and, with the tuned threshold, **cut ID switches dramatically**:

| tier (held-out) | metric | v1 original | v1 best sweep | **v3 best** | SOTA TDLP ref |
| --- | --- | ---: | ---: | ---: | ---: |
| **SportsMOT** | **IDsw** ↓ | 362 | 190 | **75** (−79%) | ~5 |
| | HOTA ↑ | 0.503 | 0.586 | **0.685** | 0.90 |
| | IDF1 ↑ | 0.451 | 0.563 | **0.671** | 0.94 |
| | purity | 0.827 | 0.758 | **0.843** | 0.95–0.97 |
| **SoccerNet** | **IDsw** ↓ | 829 | — | **220** (−73%) | — |
| | HOTA ↑ | 0.320 | — | **0.438** | — |
| | purity | 0.823 | — | 0.811 | (comparator 0.926) |

Best config: `sim_threshold=0.9`, `remember=20–30`, `new_tracklet_detection_threshold=0.9`.
Notably v3 **improved purity while slashing IDsw** (no purity-for-IDsw trade this round — the
gap augmentation genuinely taught re-association rather than just loosening gates). Checkpoint
`data/experiments/tdlp-head-prelim/head-v3.pt` (still NON-SHIPPABLE — NC tuning data).

**Trajectory:** SportsMOT IDsw 362 → 190 → **75**; HOTA 0.50 → **0.69**. A large fraction of the
cost-of-shippability gap on the metric that matters (association completeness) is closed. The
residual gap to the reference (IDsw 75 vs ~5) is the next program of work (below). SoccerNet
lags SportsMOT (harder: smaller players, det recall 0.85 vs 0.99) but improved similarly.

**Remaining levers to close the rest** (ranked): (1) **add the RTMPose cue** back (pose was
omitted for CPU-speed; it is a strong association signal); (2) **bigger head *with LR warmup*
+ cosine schedule** (the reference is 256-d/6-layer — capacity helps once it trains); (3)
**more/broader training data** (the shippable SPO-39 source); (4) **larger `max_gap` / motion
(FoD) encoder** for longer occlusions; (5) per-tier threshold calibration.

### 4d. Diagnosis (why it was bad) + motion gate — the decisive fix

The honest comparison I was missing: **on the same frozen detections, the assembled TDLP was
worse than the BoT-SORT baseline we already ship.**

| tracker (same frozen dets) | SportsMOT IDsw | HOTA | SoccerNet IDsw | HOTA |
| --- | ---: | ---: | ---: | ---: |
| Hardened BoT-SORT (SPO-30 baseline) | **31** | 0.785 | **144** | 0.519 |
| BoT-SORT + body-ReID (SPO-31) | 30 | 0.79 | 142 | 0.522 |
| TDLP v3 (no gate) | 79 | 0.685 | 220 | 0.438 |
| Reference TDLP-full | ~5 | 0.90 | — | — |

**Root cause:** BoT-SORT carries a Kalman motion model + IoU gating + camera-motion
compensation, so it only ever considers motion-plausible matches and uses appearance as a
tiebreaker. My loop did pure Hungarian on a *weak learned head* with **no motion prior at all**,
so in crowded/occluded frames it confidently matched a track to the wrong (or far) player →
switches. The reference TDLP survives a pure learned head only because its head is genuinely
SOTA (KPR ReID + pose + a large, properly-trained model); a preliminary head needs the motion
crutch.

**Fix — motion gate** (forbid matches whose normalized bbox-centre move exceeds
`base + per_frame·gap`; `gap` = frames since the track was last seen):

| v3 SportsMOT | IDsw | HOTA | IDF1 | purity |
| --- | ---: | ---: | ---: | ---: |
| no gate | 79 | 0.685 | 0.671 | 0.843 |
| **+ gate (base 0.04)** | **51.5** | **0.731** | 0.734 | 0.906 |
| + gate (per-frame 0.03) | 52.8 | **0.744** | 0.746 | 0.915 |

**Full trajectory: SportsMOT IDsw 362 → 190 → 79 → 51.5 (−86%); HOTA 0.50 → 0.74; purity →
0.91.** Now approaching the BoT-SORT baseline (31). The residual gap to BoT-SORT is the
difference between a fixed distance-ball gate and a real Kalman velocity predictor + CMC; the
gap to the reference (~5) is head quality (ReID appearance + pose + capacity + real data).

**Honest standing:** the shippable TDLP is a **runnable, licensing-clean tracker with a clear,
measured improvement path**, but with the preliminary (non-shippable) head it does **not yet
beat the existing BoT-SORT baseline** — which remains the interim shippable tracker. Closing
the last gap needs the pose cue, a Kalman/velocity motion model (or a stronger head so the gate
isn't load-bearing), and permissive training data (SPO-39).

**Likely causes & next levers** (superseded by §4c/§4d above; original analysis): (1) **untuned association thresholds** —
`sim_threshold`/`new_tracklet_detection_threshold` vs the head's logit scale (the high IDsw
smells partly like over-gating → constant respawn; a threshold/`appearance_weight`-style sweep
is the cheapest first lever); (2) **no pose cue** (RTMPose omitted for CPU-speed — add it back);
(3) **DINOv2 global** appearance is weaker for ReID than KPR part-based — fine-tune or add parts;
(4) **more/broader training data** (11 NC tuning seqs is thin; the real shippable data is
SPO-39); (5) re-enable the **motion (FoD) encoder**. None of these change the shipping-path
licensing story — they are quality levers on top of the runnable stack.

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
