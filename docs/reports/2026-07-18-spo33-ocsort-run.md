# SPO-33 — OC-SORT candidate over frozen detections (both tiers)

**Issue:** SPO-33 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 3 candidates · **Pre-registration:** [`2026-07-18-phase3-preregistration.md`](2026-07-18-phase3-preregistration.md) · **Comparator:** [`2026-07-18-spo30-comparator-run.md`](2026-07-18-spo30-comparator-run.md) · **Date:** 2026-07-18

**Status: scored — does NOT beat the comparator; documented and dropped per the pre-registered gate.**

## Integration path

**Registered in-repo track stage** (`stages/track/ocsort.py`, `@register(StageKind.TRACK,
"ocsort")`), not the import adapter — the pinned `trackers==2.4.0` package (Apache-2.0)
already provides `OCSORTTracker` with the exact `update(detections, frame=None) ->
sv.Detections` contract our BoT-SORT wrapper speaks. Fail-loud constructor + all parameters
config-exposed (SPO-13/15 conventions); the per-frame update loop and source-index survival
are shared with BoT-SORT via `stages/track/_assembly.py`. No CMC (never decodes pixels);
default state estimator XCYCSR. **License: Apache-2.0** (roboflow/trackers reimplementation;
the original OC-SORT is MIT).

## Results (identical frozen detections as the comparator; held-out both tiers)

**SoccerNet** (hosted-incumbent frozen dets):

| candidate | purity | mixed-track s | HOTA(t) | IDF1(t) | crop-yield/player | runtime/seq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hardened BoT-SORT (comparator) | 0.9257 | 23.00 | 0.5187 | 0.6052 | 419 | 45.5 s |
| OC-SORT | 0.9177 | 25.34 | 0.4954 | 0.5668 | 393 | 35.1 s |

**SportsMOT** (MixSort YOLOX frozen dets):

| candidate | purity | mixed-track s | HOTA(t) | IDF1(t) | crop-yield/player | runtime/seq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hardened BoT-SORT (comparator) | 0.9455 | 17.79 | 0.7854 | 0.8013 | 573 | 27.0 s |
| OC-SORT | 0.9183 | 25.67 | 0.7286 | 0.7190 | 562 | 20.5 s |

## Verdict against the pre-registered primary bar

The bar: mixed-track-seconds ≥15% **relative reduction** AND purity Δ ≥ +0.01, on ≥1 tier and
non-inferior on the other.

| tier | mixed-track vs comparator | purity Δ | meets bar? |
| --- | ---: | ---: | :--: |
| SoccerNet | **−10.2%** (worse: +2.3 s) | **−0.008** | no |
| SportsMOT | **−44.3%** (worse: +7.9 s) | **−0.027** | no |

OC-SORT regresses both primary metrics on both tiers — it does not approach the promotion
bar. Secondary metrics agree (HOTA/IDF1 both lower). It is **faster** (20–35 s vs 27–46 s/seq;
no CMC, no pixel decode) and crop-yield is comparable, but the compute tiebreak only applies
between primary-tied candidates, which this is not.

## Interpretation

OC-SORT is the lightweight motion-model ablation: observation-centric SORT with no camera-
motion compensation and no appearance. Its loss quantifies what BoT-SORT's extra machinery
buys — most of the gap is the **CMC** BoT-SORT applies on moving broadcast/phone footage
(Phase 1 already measured CMC as load-bearing: disabling it cost −0.045 HOTA / −0.048 purity
on SoccerNet). The purity/mixed-track regression here is consistent: without CMC the motion
model drifts under camera motion and fragments/contaminates more.

**Decision (pre-registered gate, applied):** OC-SORT is a documented loser — not promoted,
never enters the dependency tree beyond the already-permissive `trackers` package it shares
with BoT-SORT. Recorded here so the ablation's value (isolating the motion-model contribution)
is preserved.

Raw outputs: `data/experiments/benchmark-phase3-ladder-{soccernet,sportsmot}-*/result.json`
(gitignored). Ladder configs: `configs/train/benchmark-phase3-ladder-{soccernet,sportsmot}.yaml`.
