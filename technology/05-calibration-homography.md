# 5. Camera Calibration & Pitch Homography (Blocker 4)

> **Role in the pipeline:** module E. Maps image pixels → real pitch coordinates (the homography), so
> that player detections become *positions on a minimap* and distances/speeds become metric. Without
> it there is no minimap, no spatial stat, and GS-HOTA's localization term collapses. Marked 🟡 in
> [`../docs/07`](../docs/07-technology-maturity-deep-dive.md): works on operated cameras, **fragile on
> amateur handheld**.
>
> **Note on evidence:** the verified top-25 did **not** cover geometry — the deep-research synthesis
> flagged this as an open area. The findings below are `[EXTRACTED]` (primary sources, single-vote,
> not adversarially verified). Treat the specific numbers as directional.

---

## 5.1 Why this is the binding constraint on a moving phone

Recall [00](00-overview.md) §0.3: even on broadcast, **localization (GS-DetA ~49.5) is the weak half**,
far below association. Pitch registration is a big part of that localization term, and it is precisely
what amateur capture breaks:

- **Keypoints leave the frame.** Standard registration detects known pitch landmarks (line
  intersections, the centre circle, penalty-box corners) and solves a homography from them. A
  ground-level phone zoomed on the action often has **few or zero** of these landmarks in view — the
  far touchline, halfway line, and circle are simply off-screen. With too few correspondences the
  homography is under-constrained and jumps frame-to-frame.
- **Low oblique angle** compresses the far field, amplifying small keypoint errors into large pitch-
  coordinate errors.
- **Handheld motion** means the homography changes every frame and per-frame independent estimation
  jitters.

---

## 5.2 How the SOTA architectures work

### Keypoint-heatmap registration (the dominant form)
A network predicts **heatmaps for a predefined set of pitch keypoints**; detected keypoints give
pixel↔pitch correspondences; a homography (or full camera matrix) is fit. `[EXTRACTED]`

- **ViT-tiny encoder-decoder** predicting court-keypoint heatmaps achieves sub-meter projection error
  on broadcast — **0.74 m median (WorldCup), 0.26 m mean (TS-WorldCup)** — a concrete monocular
  baseline. `[EXTRACTED — PMC10534887 / "Individual locating … single moving view"]`
- This is the family behind the `sn-gamestate` calibration options (**TVCalib**, **"No Bells Just
  Whistles"**). `[VERIFIED 3-0 — sn-gamestate baseline composition]`

### PnLCalib — points *and lines*, with geometric refinement
`[EXTRACTED — arXiv:2404.08401]`
Estimates **full 3D camera calibration (intrinsics + extrinsics)** from a single sports-field image:
1. An **HRNetv2 backbone** detects a predefined **keypoint grid** *and* **field-line extremities**.
2. An initial projection matrix is computed via **RANSAC + DLT** (robust linear solve).
3. A **non-linear least-squares refinement** jointly minimizes **point reprojection error** *and*
   **point-on-line** constraints.

> **Why points+lines matters for amateur footage:** when few keypoints are visible, **lines** (the
> touchline, goal line) still constrain the solution. A registration that exploits line geometry is
> inherently more robust to the off-frame-keypoint problem than one relying on point intersections
> alone — making PnLCalib-style methods the better starting point here than pure keypoint heatmaps.

### BHITK — Bayesian sequential homography (temporal stability)
`[EXTRACTED — arXiv:2311.10361, "Video-based Sequential Bayesian Homography Estimation"]`
Instead of estimating each frame independently, BHITK **relates each frame's homography to the next
via an affine transformation while explicitly modeling keypoint uncertainty** — a Bayesian
filter over the homography sequence. This directly attacks handheld jitter and brief keypoint dropout:
when a frame has too few landmarks, the prior from neighbouring frames carries the estimate through.

---

## 5.3 Starting-point candidates

| Candidate | Mechanism | Why for amateur handheld |
|-----------|-----------|--------------------------|
| **A. PnLCalib (points + lines + NLS)** | HRNetv2 keypoints+lines → RANSAC/DLT → NLS refine | Lines survive when points leave frame; full camera model |
| **B. BHITK temporal wrapper** | Bayesian frame-to-frame homography + keypoint uncertainty | Bridges keypoint dropout, kills handheld jitter |
| **C. Keypoint-heatmap net (TVCalib / ViT-tiny)** | per-frame heatmap → homography | The `sn-gamestate` default; fine-tune on amateur angles |
| **D. Fine-tune on low/oblique angles + synthetic** | retrain C/A on ground-level viewpoints | Closes the angle-distribution gap ([08](08-amateur-data-strategy.md)) |

**Recommended rework** `[REASONED]`: combine **A + B** — a points-and-lines per-frame estimator
(robust to sparse landmarks) wrapped in a Bayesian temporal filter (robust to dropout and jitter) —
and **fine-tune (D)** on amateur low-angle footage, including **synthetic UE5 field imagery with
controlled camera angles** (SoccerSynth-Field, see [08](08-amateur-data-strategy.md)) to cover oblique
viewpoints the broadcast datasets never contain.

---

## 5.4 Graceful degradation (product design)

`[REASONED]` Some amateur clips will have *no* reliable registration (camera never shows enough
pitch). The product must degrade: **image-space stats that don't need a homography** — event counts
(passes, shots via spotting), possession, per-player touch counts — remain valid even when
*positional/metric* stats (distance covered, heatmaps, speed) cannot be computed. Tie this to the
attribution design in [07](07-event-attribution.md): the "what/who" layer is less calibration-
dependent than the "where" layer, so a clip with bad registration still yields a usable per-player
event sheet.
