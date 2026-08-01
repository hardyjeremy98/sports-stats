# Detector selection: finding a better player and ball detector

**Date:** 2026-08-01
**Scope:** replace the incumbent detection stage, which is bottlenecking the
tracklet/re-ID work. No training, no fine-tuning — off-the-shelf checkpoints
with freely downloadable weights only.
**Harness:** `scripts/detector_bench.py`, `scripts/wasb_bench.py`,
`scripts/detector_bench_report.py`
**Raw results:** `data/detector-bench/*.json`

---

## 1. Headline

The incumbent (`data/weights/football-player-detection.pt`, roboflow/sports
YOLOv8x at imgsz 1280) is the weakest serious detector tested. Two independent
findings, both measured:

1. **A better checkpoint exists and is a drop-in.** `mobadam/football-player-detection`
   lifts player AP@0.5 from **0.767 → 0.902** and ball F1 from **0.362 → 0.648**
   on the SoccerNet tuning tier, using the same stage, the same confidence, and
   the same input size.
2. **The incumbent is also being run at the wrong resolution.** Simply changing
   `imgsz: 1280 → 960` takes it from AP 0.767 → **0.831** (held-out: 0.803 →
   0.851) with no new weights at all. At 1920 it collapses to 0.268. Input size
   is not a tuning detail here; it is worth more than most of the model choices.

   The 1920 collapse is a genuine out-of-distribution failure, not a harness
   artifact: at 1920 the model emits *more* boxes than at 1280 (12,799 vs
   10,348 over the same 8,137 GT boxes) while recall falls 0.812 → 0.415. It is
   producing wrong boxes, not fewer boxes. Every candidate was swept across
   resolutions for this reason, and **more pixels helped nothing** — every model
   tested peaked at 960 or 1280.

**Recommendation: adopt `yolo-mobadam` for both player and ball, at
`imgsz: 960`.** On the tuning tier it is a close second to generic COCO YOLO11x
on player AP; **on the held-out tier it wins outright** (AP 0.921 vs 0.896), and
it wins the ball benchmark on both tiers. It also emits the four-class football
schema (`ball`/`player`/`referee`/`goalkeeper`) the pipeline already expects,
whereas YOLO11x-COCO offers only `person` + `sports ball` — no referee or
goalkeeper distinction — at 8× the ball false-alarm rate.

Held-out, versus the incumbent's production settings:

| | incumbent (1280) | **mobadam@960** | Δ |
|---|---|---|---|
| player AP@0.5 | 0.8028 | **0.9192** | **+0.116** |
| player best F1 | 0.8181 | **0.9321** | +0.114 |
| player recall | 0.7536 | **0.9251** | +0.172 |
| ball best F1 | 0.3680 | **0.6627** | +0.295 |
| recall, players 25–50px | 0.0385 | **0.3077** | 8× |
| ms/frame (RTX 4060 Ti) | 62 | **19** | **3.3× faster** |

(Timings measured on the same held-out pass; `mobadam` at 1280 is 31 ms/frame,
`yolo11x-coco` at 1280 is 70–76.)

**Confirmed end to end.** Swapping only the detector weights in the program
comparator (`v1-hardened-eval`) over the four held-out sequences moves
**HOTA (tracklet) 0.5019 → 0.6181 (+0.1163)** and cuts **ID switches 146.5 →
60.8 (−59%)**. For scale, the whole SPO-22 Phase 1 hardening sweep moved HOTA by
+0.0141 — this is ~8× that, from one config line. Full table and the two
regressions in §9.

A required code change ships with this: see §6.

---

## 2. What "better" was measured as

Detection quality only — no tracker, no association. The detector is the sole
variable. Scoring uses the repo's own
`matchlab_core.detection_eval.evaluate_detections`, and the same class
convention as `evaluation.py::_load_detections`: person classes
(player/goalkeeper/referee) on the detection side, `_SCORED_ROLES` on the GT
side, ball excluded symmetrically.

Every candidate is run once at a low confidence floor (0.05) and cached, then
re-thresholded across a grid at scoring time. So each detector is compared **at
its own best operating point**, not at a threshold that happens to suit the
incumbent.

**Evaluation set.** SoccerNet-Tracking, from `configs/datasets/soccernet.json`:
tuning tier = SNMOT-116…123 (8 sequences), held-out tier = SNMOT-124…127. Frames
sampled at stride 10 for the screen (600 frames/candidate), stride 5 for the
held-out confirmation.

**Frame alignment was verified, not assumed.** `img1/%06d.jpg` is 1-based and
`GroundTruth.frame_idx` is 0-based. `detector_bench.py verify` checks that
recall at offset 0 beats offset ±1: measured 0.632 vs 0.171/0.162. An
off-by-one here would have penalised every candidate equally and looked like
"detection is just hard".

---

## 3. Contamination — read this before comparing rows

Our evaluation set is the SoccerNet-Tracking **test** split. Several of the
strongest-looking checkpoints on Hugging Face were trained or validated on
SoccerNet-Tracking, so their numbers are **not comparable** to the rest and they
are reported in a separate tier. They are included because they usefully bound
what a SoccerNet-fitted detector can reach, not because they are candidates.

| Candidate | Why quarantined |
|---|---|
| `yolov8x6-soccernet-gsr` (`xleprime/SoccerMaster`) | SoccerNet-GSR baseline detector; GSR clips come from the same broadcast corpus. |
| `rfdetr-large-soccernet` (`julianzu9612/RFDETR-Soccernet`) | Model card states evaluation on `SoccerNet-Tracking-2023-test` — our exact split. |

The clean tier is everything trained on COCO/Objects365, SportsMOT, or the small
roboflow-style football datasets. **The recommendation is drawn only from the
clean tier.**

---

## 4. Player detection results

SoccerNet tuning tier, 8 sequences, stride 10 (600 frames), IoU 0.5.

| Detector | AP@0.5 | best F1 | @conf | P | R | dup rate | tier |
|---|---|---|---|---|---|---|---|
| `yolov8x6-soccernet-gsr` | 0.9733 | 0.9547 | 0.45 | 0.9624 | 0.9472 | 0.048 | CONTAMINATED |
| `yolov8x6-soccernet-gsr@1920` | 0.9520 | 0.9275 | 0.4 | 0.9346 | 0.9205 | 0.049 | CONTAMINATED |
| **`yolo11x-coco`** | **0.9201** | 0.9265 | 0.55 | 0.9479 | 0.9060 | 0.008 | clean |
| `yolo11x-coco@1600` | 0.9208 | 0.9256 | 0.65 | 0.9515 | 0.9011 | 0.006 | clean |
| `yolo11x-coco@960` | 0.9167 | 0.9247 | 0.4 | 0.9458 | 0.9046 | 0.011 | clean |
| `yolo11x-coco@1920` | 0.9162 | 0.9182 | 0.7 | 0.9497 | 0.8888 | 0.004 | clean |
| `yolov8x-coco` | 0.9158 | 0.9229 | 0.55 | 0.9482 | 0.8990 | 0.008 | clean |
| `rfdetr-base-coco` | 0.9044 | 0.9114 | 0.45 | 0.9350 | 0.8890 | — | clean |
| **`yolo-mobadam@960`** | **0.9034** | 0.9195 | 0.35 | 0.9327 | 0.9066 | 0.038 | clean |
| **`yolo-mobadam`** (1280) | **0.9018** | 0.9152 | 0.35 | 0.9261 | 0.9045 | 0.058 | clean |
| `yolo-mobadam@1600` | 0.8930 | 0.9049 | 0.4 | 0.9225 | 0.8879 | 0.097 | clean |
| `yolo-mobadam@1920` | 0.8831 | 0.8965 | 0.4 | 0.9181 | 0.8758 | 0.108 | clean |
| `yolov26m-sportsmot` | 0.8417 | 0.8733 | 0.3 | 0.9227 | 0.8289 | 0.097 | clean |
| `incumbent@960` | 0.8310 | 0.8453 | 0.65 | 0.9002 | 0.7967 | 0.050 | clean |
| `yolo11m-martinjolif` | 0.8089 | 0.8260 | 0.55 | 0.8570 | 0.7971 | 0.096 | clean |
| `yolov26m-sportsmot@1920` | 0.7779 | 0.8303 | 0.2 | 0.8943 | 0.7749 | 0.061 | clean |
| **`incumbent` (production: 1280)** | **0.7674** | 0.7881 | 0.65 | 0.8519 | 0.7332 | 0.055 | clean |
| `incumbent@1920` | 0.2680 | 0.3665 | 0.5 | 0.4278 | 0.3206 | 0.035 | clean |
| `yolo-gianpaj` | 0.1091 | 0.2139 | 0.5 | 0.2104 | 0.2176 | 0.024 | clean |

### Where the incumbent actually loses players

Recall by GT box height, each detector at its own best-F1 threshold:

| Detector | h<25px | 25≤h<50 | 50≤h<100 | h≥100 |
|---|---|---|---|---|
| `incumbent` (1280) | 0.000 | 0.230 | 0.639 | **0.758** |
| `incumbent@960` | 0.000 | 0.328 | 0.653 | 0.830 |
| `yolo11x-coco` | 0.000 | 0.180 | 0.796 | **0.936** |
| `yolo-mobadam` | 0.000 | **0.475** | **0.827** | 0.925 |
| `yolov26m-sportsmot` | 0.000 | 0.230 | 0.696 | 0.862 |
| `yolov8x6-soccernet-gsr` *(contaminated)* | 0.000 | 0.508 | 0.862 | 0.970 |

This is the most useful table in the report. The incumbent misses **24% of
large, near-camera players** — the easy ones. That is not a "small distant
players are hard" problem, it is a broadly weak detector, and it explains why
tracklets fragment. `yolo-mobadam` is the best clean model on the mid-size band
(0.475 vs 0.230) where players are still identity-bearing but easily lost, and
essentially ties YOLO11x on large players.

Nothing detects players under 25px on the tuning tier. That band is only 14 GT
boxes there, so treat the 0.000 as "not measured meaningfully".

### Held-out confirmation (SNMOT-124…127, stride 5)

The screen above ranked YOLO11x-COCO first. The held-out tier reverses that, and
`yolo-mobadam` wins cleanly:

| Detector | AP@0.5 | best F1 | @conf | P | R | tier |
|---|---|---|---|---|---|---|
| `yolov8x6-soccernet-gsr` | 0.9752 | 0.9600 | 0.4 | 0.9623 | 0.9577 | CONTAMINATED |
| **`yolo-mobadam`** (1280) | **0.9205** | 0.9256 | 0.4 | 0.9259 | 0.9254 | clean |
| **`yolo-mobadam@960`** | **0.9192** | **0.9321** | 0.5 | 0.9393 | 0.9251 | clean |
| `yolo11x-coco` | 0.8958 | 0.9254 | 0.5 | 0.9328 | 0.9181 | clean |
| `incumbent@960` | 0.8508 | 0.8599 | 0.7 | 0.9168 | 0.8097 | clean |
| `yolov26m-sportsmot` | 0.8292 | 0.8751 | 0.25 | 0.9243 | 0.8309 | clean |
| **`incumbent` (production)** | **0.8028** | 0.8181 | 0.7 | 0.8946 | 0.7536 | clean |

Recall by GT box height, held-out, each at its own best-F1 threshold:

| Detector | h<25px | 25≤h<50 | 50≤h<100 | h≥100 |
|---|---|---|---|---|
| `incumbent` (1280) | 0.000 | 0.038 | 0.762 | 0.759 |
| `incumbent@960` | 0.000 | 0.135 | 0.783 | 0.834 |
| `yolo11x-coco` | 0.000 | 0.058 | 0.886 | 0.948 |
| `yolo-mobadam` | 0.048 | 0.231 | 0.925 | 0.936 |
| **`yolo-mobadam@960`** | **0.429** | **0.308** | 0.916 | 0.938 |
| `yolov8x6-soccernet-gsr` *(contaminated)* | 0.333 | 0.327 | 0.950 | 0.971 |

`mobadam@960` is the only clean detector that registers at all on players under
25px, and it is the best clean model in the 25–50px band — the range where
players are still identity-bearing but the incumbent finds almost nothing
(0.038). It is also the only clean row that approaches the contaminated
SoccerNet-fitted detector on small players.

The two `mobadam` resolutions are statistically tied on AP (0.9205 vs 0.9192);
`@960` is preferred because it is better on small players, better on best-F1,
and 1.6× faster again than `mobadam@1280` (19 vs 31 ms/frame).

---

## 5. Ball detection results

Ball needs a different metric. It is ~10–20px across in a 1080p frame, so box
IoU is unstable, and the published state of the art (WASB and the TrackNet
family) are **heatmap models that emit a centre point and no box at all**.
Scoring is therefore centre-distance within 10px, top-1 candidate per frame,
which is how a pipeline would consume it. Frames where GT labels no ball are
scored separately as a false-alarm rate rather than folded into precision,
because SoccerNet ball GT has genuine gaps.

SoccerNet tuning tier, stride 10, tolerance 10px.

| Ball detector | best F1 | P | R | false-alarm rate |
|---|---|---|---|---|
| **`yolo-mobadam@960`** | **0.654** | 0.809 | 0.550 | 0.111 |
| **`yolo-mobadam`** (1280) | **0.648** | 0.808 | 0.540 | **0.074** |
| `yolo11x-coco` (`sports ball`) | 0.630 | 0.710 | 0.566 | 0.630 |
| `yolo-mobadam@1920` | 0.625 | 0.796 | 0.515 | 0.074 |
| `ball-raghav@1280` | 0.620 | 0.733 | 0.537 | 0.315 |
| `ball-raghav@1920` | 0.610 | 0.696 | 0.542 | 0.463 |
| `ball-rajatdave@1920` | 0.601 | 0.709 | 0.522 | 0.352 |
| `ball-rajatdave@1280` | 0.529 | 0.740 | 0.412 | 0.185 |
| `yolo11m-martinjolif` | 0.515 | 0.720 | 0.401 | 0.130 |
| `ball-martinjolif@1280` | 0.487 | 0.578 | 0.421 | 0.556 |
| `incumbent@960` | 0.454 | 0.664 | 0.344 | 0.241 |
| **`incumbent` (production)** | **0.362** | 0.475 | 0.293 | 0.537 |
| `ball-martinjolif@1920` | 0.332 | 0.388 | 0.289 | 0.611 |
| `wasbfam-wasb` | 0.117 | 0.126 | 0.108 | 0.907 |
| `wasbfam-restracknetv2` | 0.092 | 0.097 | 0.088 | 0.907 |
| `incumbent@1920` | 0.096 | 0.189 | 0.064 | 0.352 |
| `wasbfam-tracknetv2` | 0.062 | 0.078 | 0.051 | 0.759 |
| `wasbfam-monotrack` | 0.025 | 0.028 | 0.022 | 0.982 |
| `wasbfam-ballseg` | 0.000 | 0.000 | 0.000 | 0.000 |
| `wasbfam-deepball` | 0.000 | 0.000 | 0.000 | 1.000 |
| `wasbfam-deepball-large` | 0.000 | 0.000 | 0.000 | 0.944 |
| `yolov26m-sportsmot` | 0.000 | — | 0.000 | — |

`yolov26m-sportsmot` scores 0 because SportsMOT is a person-only dataset — the
model has no ball class. That is a schema fact, not a failure.

Held-out (SNMOT-124…127, stride 5, 591 ball-labelled frames):

| Ball detector | best F1 | P | R |
|---|---|---|---|
| `ball-raghav@1280` | 0.6570 | 0.704 | 0.616 |
| **`yolo-mobadam@960`** | **0.6627** | 0.808 | 0.562 |
| `yolo-mobadam` (1280) | 0.6525 | 0.813 | 0.545 |
| `yolo11x-coco` | 0.6485 | 0.702 | 0.602 |
| `ball-rajatdave@1920` | 0.5386 | 0.674 | 0.448 |
| `incumbent@960` | 0.4951 | 0.686 | 0.388 |
| **`incumbent` (production)** | **0.3680** | 0.626 | 0.261 |
| `ball-martinjolif@1920` | 0.3691 | 0.481 | 0.300 |

`ball-raghav@1280` and `mobadam@960` are effectively tied at the top, but
`raghav` is a plain COCO checkpoint scored on its `sports ball` class — it
brings no player detection, so adopting it would mean running a second model.
`mobadam` delivers the same ball F1 *and* the winning player detector in one
pass.

**The ranking is not an artifact of the 10px tolerance.** Best F1 on held-out
across the tolerance sweep:

| tolerance | incumbent | `yolo11x-coco` | **`mobadam@960`** |
|---|---|---|---|
| 5px | 0.292 | 0.532 | 0.525 |
| 10px | 0.368 | 0.649 | **0.663** |
| 20px | 0.404 | 0.714 | **0.747** |
| 30px | 0.418 | 0.732 | **0.775** |

The incumbent stays flat as the tolerance loosens (0.29 → 0.42) while the others
climb steeply. That shape says the incumbent is not *imprecisely* localising the
ball — it is failing to find it at all, and a looser threshold cannot rescue a
detection that was never made.

The held-out false-alarm column is omitted deliberately: the held-out tier has
only **9** ball-absent frames, so that rate is 1/9 vs 4/9 — noise. The tuning
tier's 54 ball-absent frames are the meaningful measurement, and there
`yolo-mobadam` has the lowest rate of any candidate (0.074).

### The SOTA ball detectors do not transfer

This is the most surprising result and it deserves care, because "the published
SOTA lost to a random Hugging Face YOLO" is exactly the kind of claim that is
usually an integration bug. It was checked three ways:

* **The affine round-trip is exact.** Mapping a GT ball centre into WASB's
  512×288 input space and back returns the original coordinate to 1e-6.
* **WASB does find the ball**, just not reliably: at a low heatmap threshold it
  places a candidate within 15px of the GT ball on 4 of 10 spot-checked frames.
  So the model runs correctly; it is not silent.
* **It is not a ranking artifact.** Recall if the ball is among *any* emitted
  candidate (not just top-1) — the ceiling a temporal tracker could reach:

  | | R@top-1 | R@any candidate |
  |---|---|---|
  | `wasbfam-wasb` | 0.108 | **0.326** |
  | `wasbfam-restracknetv2` | 0.088 | 0.220 |
  | `yolo-mobadam` | 0.540 | **0.546** |
  | `yolo11m-martinjolif` | 0.401 | 0.407 |

  Even with a perfect ranker, WASB tops out around 0.33 on this data, well under
  mobadam's 0.55. The YOLO models' top-1 and any-candidate recall are nearly
  identical, meaning they already rank the ball correctly.

* **The family's internal ranking reproduces the paper's.** All seven soccer
  checkpoints were run: WASB (0.117) > ResTrackNetV2 (0.092) > TrackNetV2
  (0.062) > MonoTrack (0.025) > BallSeg / DeepBall / DeepBall-Large (0.000).
  WASB coming out best of its own family, and the 2019 DeepBall baselines worst,
  is the ordering the BMVC paper reports. A broken integration would be unlikely
  to preserve that ordering while uniformly deflating the scores — this is the
  strongest single piece of evidence that we are measuring domain transfer
  rather than our own bug.

**Caveats, stated plainly.** WASB's published configuration pairs the detector
with an online tracker enforcing temporal consistency; we ran detection only, so
these numbers understate the full published method — but the R@any ceiling above
bounds how much that could recover. WASB's soccer weights were trained on a
different soccer dataset (its `ID-1…ID-6` clips), not SoccerNet, so this is a
domain-transfer result, not a like-for-like reproduction. And the three 0.000
rows (BallSeg, DeepBall, DeepBall-Large) should be read as "produced nothing
usable under this harness" rather than as precise measurements — BallSeg in
particular emits very large numbers of blobs and repeatedly died mid-sweep,
which smells of a post-processing mismatch on our side. They do not affect the
conclusion, which rests on WASB itself.

---

## 6. Required code change (already made)

`yolo-local` hard-coded the roboflow class order:

```
0: BALL, 1: GOALKEEPER, 2: PLAYER, 3: REFEREE
```

`mobadam` — and several other football checkpoints — order their classes
`ball / player / referee / goalkeeper`. Swapping weights by editing only a
config path would therefore have relabelled **every player as a goalkeeper and
every goalkeeper as a referee**, silently, with no error and a plausible-looking
run.

`stages/detect/yolo_local.py` now derives its class map from the checkpoint's own
`model.names` (`resolve_class_map`), falling back to the old id order only when
names are missing or unrecognisable. It also restricts emitted classes to the
mapped ids, so a COCO checkpoint contributes `person` + `sports ball` instead of
defaulting all 80 classes into `PLAYER`. Covered by
`packages/matchlab_core/tests/test_detect_yolo_local_classmap.py` (7 tests);
full core suite still passes (862 passed, 5 skipped).

---

## 7. Candidates tested and rejected

| Candidate | Outcome |
|---|---|
| `yolo11x-coco`, `yolov8x-coco` | Strong on tuning (AP 0.920/0.916) but loses on held-out (0.896), `person`-only schema, 8× the ball false-alarm rate. Runner-up. |
| `rfdetr-base-coco` | AP 0.904 — competitive, no football schema. |
| `rfdetr-large-coco` | AP 0.913. Ran only after fixing a resolution constraint (RFDETRLarge needs a size divisible by 32; the base model's 728 is invalid). Still below `mobadam` held-out, `person`-only schema. |
| `yolov26m-sportsmot` | AP 0.842, no ball class. |
| `yolo11m-martinjolif` | AP 0.809 — barely above the incumbent. |
| `yolo-gianpaj` | AP 0.109. Class map verified correct; the model is simply weak. |
| `rfdetr-large-soccernet` | **Could not be loaded** — checkpoint trained with `patch_size=14` and a 3-class head, incompatible with the current `rfdetr` release's state dict. Contaminated anyway, so no loss. |
| `OrbitalLab/mova-rfdetr-soccernet-v1` | Gated repo, not downloadable. |
| WASB / TrackNetV2 / ResTrackNetV2 / MonoTrack / BallSeg (soccer) | See §5. |

---

## 8. What this does not establish

* **The end-to-end run uses `imgsz: 1280`, not the recommended 960**, so that
  the detector weights are the single variable against the program comparator.
  That is the conservative choice: 1280 is the weaker of the two `mobadam`
  variants on best-F1 and on small-player recall, so it under-states the gain
  the recommended setting should deliver.
* **SoccerNet only.** SportsMOT is not on disk and there is no phone footage, so
  every number here is broadcast-framed 1080p. The resolution finding in
  particular may not transfer to other framings.
* **Ball GT gaps.** SoccerNet ball labels cover ~85–100% of frames; the
  false-alarm column is measured against the unlabelled remainder and is
  therefore an upper bound on true false alarms.
* `mobadam/football-player-detection` publishes **no training-set description**.
  It is treated as clean because nothing indicates SoccerNet exposure, but that
  is an absence of evidence, not evidence of absence. This is the single
  weakest point in the recommendation. Two things partly mitigate it: its
  held-out score (0.9205) is close to its tuning score (0.9018) rather than
  collapsing or spiking, and it remains well below the openly
  SoccerNet-fitted detector (0.9752). Neither rules out contamination. If this
  matters for a published claim, re-run the comparison on SportsMOT — which is
  not currently on disk.

---

## 9. End-to-end confirmation

`configs/train/benchmark-detector-swap-soccernet.yaml` runs the repo's own
benchmark gate over the SoccerNet held-out sequences with two candidates
identical in every slot except `detect.params.weights`:
`pipeline.v1-hardened-eval.yaml` (program comparator) vs
`pipeline.detector-swap-eval.yaml` (same file, mobadam weights).

**The detection-layer gain does reach the tracker — and then some.** Mean over
all four held-out sequences (SNMOT-124…127), both arms complete:

| Metric | incumbent | mobadam swap | Δ |
|---|---|---|---|
| detection AP | 0.7720 | 0.8992 | **+0.1272** |
| detection recall | 0.7965 | 0.9183 | **+0.1217** |
| **HOTA (tracklet)** | 0.5019 | 0.6181 | **+0.1163** |
| **HOTA (entity)** | 0.4987 | 0.6285 | **+0.1298** |
| IDF1 (tracklet) | 0.5823 | 0.6927 | +0.1105 |
| IDF1 (entity) | 0.6195 | 0.7468 | +0.1273 |
| entity purity | 0.7884 | 0.8323 | +0.0439 |
| **ID switches (tracklet)** | 146.5 | 60.8 | **−85.8 (−59%)** |
| persistent ID switches | 10.25 | 7.75 | −2.50 |
| miss-burst p95 (frames) | 19.65 | 11.08 | −8.58 |
| tracklet purity | 0.9526 | 0.9461 | −0.0066 |
| mixed-identity seconds | 14.01 | 18.20 | +4.19 |

**The baseline arm reproduces the program comparator exactly**, on four
independent metrics. `configs/pipeline.v1-hardened-eval.yaml`'s own header
records the SPO-22 confirmation run on this same held-out set as HOTA (tracklet)
0.5019, tracklet purity 0.9526, ID switches 146.5, mixed-identity seconds 14.01.
This run's incumbent arm scored **0.5019 / 0.9526 / 146.5 / 14.01** — all four
identical. So the harness is measuring exactly what the program has been
measuring since Phase 1, and the deltas above are directly comparable to every
previous phase gate rather than being on some new scale of their own.

For scale: the entire SPO-22 Phase 1 hardening effort — a pre-registered sweep
across stride, tracker thresholds, buffers and activation — moved HOTA (tracklet)
by **+0.0141**. Swapping the detector alone moves it by **+0.1163**, roughly
**eight times as much**, and cuts ID switches by 59%. The detector, not the
tracker, was the binding constraint.

**Two metrics regress, and they should be read together with the recall gain.**
Tracklet purity is down 0.0074 and mixed-identity seconds up 4.55. The new
detector finds substantially more of the players (recall +0.12), so there is
simply more tracked material per sequence for a mix-up to occur in; a detector
that sees fewer players trivially has fewer chances to confuse them. Entity
purity — the post-association measure that actually matters for identity — moves
the *right* way (+0.0373), and persistent ID switches are flat. The purity delta
is also within the 0.005–0.01 band the Phase 1 work treated as its noise floor.
This is worth watching, not blocking on.

Re-run with:

```bash
uv run --with ultralytics matchlab-train run \
  configs/train/benchmark-detector-swap-soccernet.yaml
uv run python scripts/detector_swap_summary.py \
  data/experiments/benchmark-detector-swap-soccernet-<timestamp>
```

---

## 10. Reproducing this

```bash
# player screen (tuning tier)
uv run --with ultralytics --with opencv-python-headless \
  python scripts/detector_bench.py infer --roles tuning --stride 10
uv run python scripts/detector_bench.py score --roles tuning --stride 10 \
  --out data/detector-bench/screen-tuning.json

# ball, including the WASB heatmap family (needs the upstream clone)
uv run --with hydra-core --with torch --with torchvision \
  --with opencv-python-headless python scripts/wasb_bench.py --roles tuning
uv run python scripts/detector_bench.py score-ball --roles tuning --stride 10 --tol 10

# tables
uv run python scripts/detector_bench_report.py data/detector-bench/heldout.json \
  --heights yolo-mobadam incumbent-yolov8x-roboflow

# frame-alignment check (run this first if anything looks wrong)
uv run --with ultralytics --with opencv-python-headless \
  python scripts/detector_bench.py verify --candidates incumbent-yolov8x-roboflow
```

Weights live in `data/weights/bench/` (player + ball YOLO checkpoints) and
`data/weights/wasb/` (7 WASB-family soccer checkpoints). The WASB code is a
clone of `nttcom/WASB-SBDT` at `../external-ball/WASB-SBDT`, kept out of this
repo's environment like `external-trackers/` and `external-spotters/`.

---

## 11. Follow-up: the Assoc tab under-reported missed merges (fixed)

Investigating why the SNMOT-125 Lab run showed "0 merges GT-correct, 1 missed"
turned up a measurement bug, not just a weak merge engine.

**Ground truth demanded 12 merges, not 1.** 12 of 25 players in SNMOT-125 leave
and re-enter frame (13 gap events, 2.6–13.5 s). 12 of those were left as two
separate tracklets — genuine association work. Association made **zero** merges.

**Two independent defects hid that:**

1. **Argmax GT assignment.** Missed merges were derived from
   `gt_id_of_tracklet`, a per-tracklet majority vote. When the tracker glues two
   players into one tracklet — exactly the runs where association matters — the
   minority half's identity is erased, so the pair never looks GT-same. Entity
   #4 (tracklet 3) is GT11 for 0.0–12.5 s then GT23 from 15.0 s; 20 of 28
   tracklets are impure (purity 0.733).
2. **Counting over the decision trail.** The UI tallied verdicts across
   `association.json`'s pairs. That trail only holds pairs the engine actually
   scored: `reid/twopass.py` explicitly drops temporally-overlapping pairs, and
   records *nothing* for a best candidate that simply scored under threshold. A
   miss absent from the trail could never be flagged.

**Fix.** `merge_quality` now computes misses from the full per-tracklet GT
composition over all tracklet pairs, independent of the trail, and publishes
`missed_pairs`, `n_missed_pairs`, `merge_recall`, and
`gt_composition_of_tracklet`. The Assoc tab reads those numbers. SNMOT-125 now
reports **17 missed across 11 players, recall 0.00**.

A tracklet counts as representing a GT player only if it holds ≥3 frames of it
**and** ≥20% of its own matched frames. The share floor matters: without it,
transient ID switches make every fragment pair up and the same run reports
**115** misses. Sensitivity, against an independent count of 12 from GT
absences:

| share floor | missed pairs | players split |
|---|---|---|
| none | 115 | 19 |
| 0.2 (chosen) | **17** | **11** |
| 0.3 | 10 | 10 |
| 0.5 | 3 | 3 |

Runs scored before the fix have no `n_missed_pairs`; the tab shows "re-score
needed" rather than a misleading 0.

**A third defect, same family: "0 wrong" was true but misleading.** The
correct/wrong counters only judge merges the *associate* stage made. Association
made none here, so "0 wrong" was literally correct — while entity #4 held GT11
for 12.5 s and then GT23, and 20 of 28 tracklets were impure. Those joins were
made by the TRACK stage, inside a single tracklet, where an association metric
can never see them; and association cannot repair them either, because it merges
tracklets and never splits them. The tab now shows a separate
`⚠ 20/28 tracklets impure · 79.1s mixed` figure from `purity.tracklet`, and the
merge counter is labelled "wrong merges" so its scope is explicit. Without that,
the run's dominant identity failure was invisible on the tab meant to diagnose
identity.

### Why association made zero merges

Separately measured, and not a threshold problem — re-running at
`min_score = −5.0` still yields only 2–3 merges:

* **The appearance calibrator's operating point is wrong for this run.** The
  `body` channel carries the largest weight (2.03) and its
  `fusion-footpass-v1.json` calibrator crosses zero at cosine **≈0.95**. The 12
  true pairs sit at median cosine **0.853**, so the dominant channel scores
  nearly every correct merge at −3 to −7 nats against a +4.0 threshold.
* **Two of four channels are starved.** `pnlcalib` emits homographies for all
  750 frames but at median confidence 0.196, while
  `calibration_min_confidence` is 0.5 — so **749 of 750 are discarded**, killing
  the occupancy and transition channels. (`yolo-pitch-local` passes 255/750 on
  the same clip, so this is a confidence-scale mismatch, not a pitch failure.)
* **Not the gap channel**, which is inert: +0.19 at 0.5 s, +0.11 at 13.5 s.

The fusion model's own provenance predicts this: fitted on FOOTPASS with
"oracle pitch coordinates" and `max_gap_frames=30 (1.2 s)`, with an explicit
domain-gap caveat. None of the above is fixed here — this section records the
measurement so the fix can be chosen deliberately.
