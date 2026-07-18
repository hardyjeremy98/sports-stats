# SPO-34 — Phase 3 exit gate: tracker winner under the promotion hierarchy

**Issue:** SPO-34 · **PRD:** [`docs/prds/tracklet-modernization.md`](../prds/tracklet-modernization.md) Phase 3 · **Pre-registration:** [`2026-07-18-phase3-preregistration.md`](2026-07-18-phase3-preregistration.md) · **Date:** 2026-07-19

**Status: DECIDED — CONFIRMED (Jeremy, 2026-07-19); gate closed.** Two outcomes: (1) no
frozen-detection parity candidate clears the promotion bar — the hardened baseline remains the
interim shippable tracker; (2) for the shippable build
([`shippable-multi-cue-tracklet-system.md`](../prds/shippable-multi-cue-tracklet-system.md)),
the **TDLP link-prediction association head is the selected architecture** — confirmed as
definitive (TDLP wins on benchmarks, reproduced here on our purity metric on identical features,
and is offline, matching the product). SPO-40/42 point at the TDLP head.

## 1. Frozen-detection parity candidates (promotion-eligible, SPO-29 primary bar)

Bar: mixed-track ≥15% relative reduction AND Δpurity ≥ +0.01 vs the SPO-30 comparator, consistent
across both tiers. All consume identical frozen detections.

| candidate | SportsMOT purity / mixed | SoccerNet purity / mixed | meets bar? |
| --- | --- | --- | :--: |
| hardened BoT-SORT (SPO-30, comparator) | 0.9455 / 17.79 | 0.9257 / 23.00 | baseline |
| BoT-SORT + body-ReID, ours (SPO-31) | 0.9520 / 14.55 | 0.9348 / 19.94 | **no** (purity < +0.01; consistent, sub-bar) |
| TDLP-bbox (SPO-32) | 0.8807 / 36.49 | 0.7991 / 65.52 | **no** (regresses both) |
| OC-SORT (SPO-33) | 0.9183 / 25.67 | 0.9177 / 25.34 | **no** (regresses both) |

**No parity candidate clears the promotion bar.** SPO-31 (our online body-ReID) is the only one
directionally positive on both tiers but lands sub-bar on purity; the pre-registration permits an
`appearance_weight` sweep, which remains the open lever if we want to promote an *in-repo* stage.
TDLP-bbox and OC-SORT regress purity outright. **Decision:** promote nothing off-the-shelf; the
**hardened BoT-SORT baseline (SPO-30) stands as the interim, shippable, runnable-everywhere
tracker** (purity-equivalent to the reference ceiling; see §3).

## 2. As-published reference rows (ceilings, not promotion candidates)

Not detection-controlled against §1 (they use their own detector/pose/appearance); non-shippable
(NC weights + research-only ReID). SportsMOT held-out.

| reference | purity | mixed-track s | HOTA(t) | IDF1(t) | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| CAMELTrack (SPO-35, 5-seq) | 0.9407 | 18.27 | 0.8931 | 0.9302 | native multi-cue |
| **TDLP-full (5-seq, same features)** | **0.9680** | **10.12** | **0.9097** | **0.9496** | link-prediction head |
| TDLP-bbox on CAMEL features (5-seq) | 0.8908 | 38.44 | 0.8425 | 0.8750 | appearance ablated |
| Deep-EIoU (SPO-27) | — | — | published only | — | unlicensed, never executed |

Two clean findings (identical CAMELTrack features; only the association head differs):

- **Appearance+pose is decisive within a learned head:** TDLP-bbox → TDLP-full is +0.077 purity /
  −74% mixed (5-seq). The SPO-32 "TDLP over-connects" result was a missing-appearance artifact.
- **TDLP's link-prediction head beats CAMEL's transformer head on every metric on identical
  input** (+0.027 purity, −45% mixed, +0.017 HOTA). This is the cleanest possible
  association-algorithm comparison.

## 3. What this means together

The reference ceiling (TDLP-full ≈ 0.97 purity / HOTA 0.91) sits well above the baseline's HOTA
(~0.79) while the **baseline already matches the ceiling on purity** (0.945 vs 0.95). So the gap a
new tracker must close is **association completeness (fragmentation/HOTA), not purity** — and
fragmentation is what the offline associator repairs. This confirms the program strategy: keep the
baseline as the shippable interim, and build a multi-cue tracker to capture the association gain.

## 4. Decisions

1. **Interim/shippable tracker: hardened BoT-SORT baseline (SPO-30).** No off-the-shelf candidate
   is promoted (none clears the bar; all SOTA options are non-shippable). Downstream systems
   develop against the baseline (runnable everywhere, purity-equivalent).
2. **Architecture for the shippable build (SPO-40): the TDLP link-prediction head**, on the
   measured evidence that it beats the CAMEL head on identical multi-cue features, on our primary
   purity metric. Feeds `shippable-multi-cue-tracklet-system.md` (SPO-40/42).
3. **Optional in-repo lever retained:** an `appearance_weight` sweep on SPO-31 (our BoT-SORT+ReID)
   is the only path to promoting a *shippable-today* improvement over the baseline, if wanted
   before the build lands.

## 5. Caveats binding these decisions

- The TDLP-vs-CAMEL result is **SportsMOT-only** and **as-published** (its own detector/pose/
  appearance). It selects an *architecture to build*, not a checkpoint to ship — the shippable
  version retrains on permissive data (SPO-38/39/40) and its quality is re-established at the Bar A
  gate (SPO-44).
- TDLP is **offline** (whole-clip); acceptable for the offline product (ADR 002) and consistent
  with the existing pipeline, but a different integration shape than an online tracker — noted for
  SPO-42.
- No product-domain (phone-footage) evidence exists for any candidate; domain validation is the
  deferred Bar B.
