# Full-TDLP measurement spike — SOTA-vs-SOTA and marginal value of appearance

**Issue:** SPO-32 follow-up / SPO-34 evidence · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 3 · **Date:** 2026-07-19 · **Config:** `configs/train/benchmark-phase3-tdlp-full-sportsmot.yaml`

**Status: measured.** Resolves the SPO-32 TDLP+ReID protocol caveat via the as-published route and supplies the SPO-34 SOTA-vs-SOTA evidence.

## What was run

Full TDLP (`bbox+kps+appearance`, HF `Robotmurlock/tdlp_sportsmot`, weights sha256 `e0f755ed…`)
and TDLP-bbox were run over **identical inputs**: CAMELTrack's released SportsMOT tracker-state
(its own YOLOX detector + pose + KPR-ReID) converted to TDLP's ExtraFeatures format via TDLP's
own `feature_extraction.py` (canonical parser + normalization). Because TDLP features are keyed
by scene-name + frame (not global video-id), all **6/6** held-out sequences aligned (unlike the
CAMEL state-load, which lost one). Both are **as-published reference rows** (CAMELTrack detections,
not our frozen dets; SportsMOT-trained → selection-only, non-shippable). MIT code, run in the
isolated env — never enters the dependency tree.

## Result 1 — marginal value of appearance+pose WITHIN TDLP (identical input)

| TDLP on CAMELTrack features (6-seq) | purity | mixed-track s | HOTA(t) | IDF1(t) | ID-sw(t) |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbox-only | 0.8680 | 42.10 | 0.8230 | 0.8523 | 27.5 |
| full (bbox+kps+appearance) | **0.9527** | **13.51** | **0.8993** | **0.9373** | **5.5** |
| Δ | **+0.085** | **−68%** | +0.076 | +0.085 | −22.0 |

Appearance+pose is **exactly what fixes TDLP-bbox's purity collapse.** The SPO-32 finding
(TDLP-bbox over-connects → contamination) was a *missing-appearance* problem, not an inherent
TDLP flaw: adding the learned appearance/pose cues turns the same association head from
purity-0.87 into purity-0.95. This is the clean marginal-value-of-appearance measurement the
SPO-32 pair was meant to produce (here via the as-published route, appearance = TDLP-native).

## Result 2 — SOTA-vs-SOTA: TDLP-full vs CAMELTrack (identical CAMELTrack features)

Both consume the **same** CAMELTrack detections + pose + appearance; the only difference is the
**association head** (TDLP link-prediction vs CAMEL transformer). Matched 5-seq set (the seqs
CAMELTrack scored in SPO-35):

| candidate (5-seq, identical features) | purity | mixed-track s | HOTA(t) | IDF1(t) | ID-sw(t) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAMELTrack (CAMEL head) | 0.9407 | 18.27 | 0.8931 | 0.9302 | 6.0 |
| **TDLP-full (link-prediction head)** | **0.9680** | **10.12** | **0.9097** | **0.9496** | **5.4** |
| Δ (TDLP-full − CAMEL) | **+0.027** | **−45%** | **+0.017** | **+0.019** | −0.6 |

**TDLP-full's association head beats CAMEL's on every metric on identical input** — including our
primary purity/mixed-track axis. This is the cleanest association-algorithm comparison available:
same detector, same pose, same appearance embeddings, only the learned matcher differs.
(Full-TDLP 6-seq means: purity 0.9527 / mixed 13.51 / HOTA 0.8993 / IDF1 0.9373.)

## Caveats

- **As-published, not frozen-det parity.** Both use CAMELTrack's own detector+pose+appearance, so
  these rows are not detection-controlled against the frozen-detection parity candidates
  (comparator, SPO-31). The TDLP-full-vs-CAMEL comparison *is* controlled (identical features).
- **SportsMOT only.** No SoccerNet transfer for this route (no CAMELTrack SoccerNet state).
- **Non-shippable** on multiple axes (NC-trained association weights; research-only KPR-ReID; pose
  front-end) — reference/ceiling evidence, not a shippable artifact.
- **TDLP is offline** (whole-clip), CAMELTrack online. Both compatible with the offline
  upload-and-process product (ADR 002).

## Bearing on decisions

- **SPO-32:** the TDLP+ReID marginal-appearance question is now answered (appearance is decisive
  within TDLP). Recorded as the as-published complement to the frozen-det TDLP-bbox row.
- **SPO-34:** for *architecture selection* in the shippable build (new PRD), the **TDLP
  link-prediction head is the measured winner** over the CAMEL head on identical multi-cue input.

Raw outputs: `data/experiments/benchmark-phase3-tdlp-full-sportsmot-*/result.json` (gitignored).
Extraction/driver: `external-trackers/{extract_tdlp_features,run_tdlp_frozen}.py`.
