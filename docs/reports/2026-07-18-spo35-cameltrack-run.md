# SPO-35 — CAMELTrack runnable as-published reference row (both tiers)

**Issue:** SPO-35 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 3 candidates · **Pre-registration:** [`2026-07-18-phase3-preregistration.md`](2026-07-18-phase3-preregistration.md) §8–§9 · **Comparator:** [`2026-07-18-spo30-comparator-run.md`](2026-07-18-spo30-comparator-run.md) · **Date:** 2026-07-18

**Status: SportsMOT as-published reference scored (5/6 held-out) — establishes the SOTA ceiling. SoccerNet transfer: tier limitation, documented (§4). Reference row only — NOT a promotion candidate, NOT bound by the SPO-29 §4 guardrails.**

## Integration path (import adapter, reference row)

CAMELTrack ([TrackingLaboratory/CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack),
commit `46a74bb`, **Apache-2.0**) was run in a fully isolated `uv` environment via its TrackLab
framework — code and weights **never enter this repo's dependency tree**. It is a native
multi-cue tracker (appearance + pose keypoints + motion); the pose input breaks the
frozen-detection parity protocol, so per the pre-registration (§9) it is a **clearly-labeled
as-published reference row**, not an input-parity candidate.

**SportsMOT (native, reference).** Run with CAMELTrack's released SportsMOT **tracker-state**
(`states/sportsmot-val.pklz` on HF — the authors' own YOLOX detector + pose estimator + KPR-ReID
embeddings, i.e. the exact off-the-shelf inputs behind their published HOTA 80.3) and the
`camel_bbox_app_kps_sportsmot.ckpt` checkpoint (sha256 `99db5f9b…`), CAMEL association only. This
reproduces the published pipeline faithfully — the **detections are CAMELTrack's own, not our
frozen MixSort YOLOX**, which is exactly why it is a reference and not a parity row. MOT output
imported via `pitchlab-train import-tracklets` with an `ExternalProvenance` sidecar
(`reference_only=true`, `comparison_class=as_published`, pose source recorded); scored with the
full metric stack. Config: `configs/train/benchmark-phase3-spo35-sportsmot.yaml`.

*Alignment note.* The released state is keyed by video-index over the full 45-video val set, so
scoring our held-out subset required reconstructing the full 45-video sequence layout (real dirs
for the 9 ingested, stubs for the rest) to preserve the state's cumulative frame indexing. 5 of
our 6 held-out sequences aligned; **`v_0kUtTtmLaJA_c004` came back empty** under this
reconstruction (an isolated indexing quirk) and is excluded — the numbers below are the 5-sequence
means, with the comparator and TDLP-bbox re-averaged over the **same 5** for a fair comparison.

## Results — SportsMOT held-out (5 seqs, same sequences across rows)

| row | class | purity | mixed-track s | HOTA(t) | IDF1(t) | ID-sw(t) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hardened BoT-SORT (comparator, frozen dets) | matched_data | **0.9534** | **16.65** | 0.7861 | 0.7984 | 31.0 |
| TDLP-bbox (frozen dets, SPO-32) | matched_data | 0.9002 | 33.08 | 0.8168 | 0.8624 | 33.6 |
| **CAMELTrack (bbox+app+kps, own dets)** | **as_published** | 0.9407 | 18.27 | **0.8931** | **0.9302** | **6.0** |

## Interpretation — the SOTA ceiling keeps purity while lifting association

CAMELTrack posts **HOTA 0.893 / IDF1 0.930 with just 6 ID-switches** — far above the comparator
(0.786 / 0.798, 31 switches) and above TDLP-bbox (0.817 / 0.862) — **while holding purity at
0.941 and mixed-track at 18.3 s**, both within a whisker of the comparator (0.953 / 16.7 s). This
is the reference's job: it bounds what strong multi-cue learned association can reach on this tier.

The contrast with SPO-32 is the substantive finding. TDLP-bbox bought its HOTA/IDF1 gain by
**over-connecting** — purity collapsed −0.05 and mixed-track doubled. CAMELTrack reaches a *much*
higher HOTA/IDF1 with **near-comparator purity**: multi-cue association (appearance + pose +
motion) does **not** force the purity-for-continuity trade that bbox-only learned association did.
So the ceiling is not "high HOTA requires sacrificing purity" — it is achievable to get both.

**Caveat — read this as a ceiling, not a head-to-head.** Part of CAMELTrack's margin is its
**own detector** (its released state's YOLOX, not our frozen MixSort dets) and its pose/appearance
cues; this row is not detection-controlled and is **not** comparable to the parity candidates on
the detection axis. It shows a *ceiling*, not a promotion delta — and it is not subject to the §4
guardrails or the primary-delta gate (§9). Per SPO-29 §7, it also does **not** bear on the
"marginal value of appearance" question (that stays scoped to the SPO-31 pair, appearance = ours).

## 4. SoccerNet transfer — tier limitation (recorded, not silently skipped)

No SoccerNet CAMELTrack tracker-state exists (HF ships states for DanceTrack, SportsMOT, MOT17,
BEE24 only), so the light state-replay path used for SportsMOT is unavailable. The alternative —
running CAMELTrack's full native pipeline (YOLO detector → pose → KPR-ReID → CAMEL) on SoccerNet
sequences — is **blocked by a framework bug**: CAMEL's `preprocess`
(`cameltrack/cameltrack.py:141`) does `np.stack` over per-detection cue arrays that the live
detector+pose+reid stages emit at inconsistent shapes (`ValueError: all input arrays must have the
same shape`), independent of dataloader workers. The released states sidestep this because their
cues are pre-computed and shape-consistent; the live pipeline does not.

Rather than hand-patch the tracker's association preprocessing (which would change its behaviour
and undermine "as-published"), the **SoccerNet CAMELTrack transfer row is recorded as a tier
limitation**, per the issue's instruction to "record any tier limitation explicitly rather than
silently skipping." The SportsMOT native row above stands as the CAMELTrack reference for the
SPO-34 gate.

## Provenance / licensing (recorded per axis)

- **Code:** Apache-2.0 (runnable; not entered into this repo's dependency tree).
- **Weights:** `camel_bbox_app_kps_sportsmot.ckpt` — trained on CC BY-NC 4.0 SportsMOT →
  **selection-only, non-shippable** on the training-data axis (same posture as MixSort YOLOX /
  SPO-25 and TDLP / SPO-32).
- **Pose/appearance source:** CAMELTrack released SportsMOT tracker-state (own YOLOX + pose +
  KPR-ReID) — native multi-cue, recorded in the sidecar.
- **Comparator ≠ shipping** restated: reference-row weights are non-shippable; the production
  tracker/detector question stays open to Phase 5.

Raw outputs: `data/experiments/benchmark-phase3-spo35-sportsmot-*/result.json` (gitignored).
External run tooling: `external-trackers/` (throwaway, outside the repo tree).
