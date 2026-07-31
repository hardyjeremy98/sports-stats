# Split stage, Phase 0: measured, not built — the oracle ceiling is too small to justify it yet

**Date:** 2026-08-01
**Question:** should the long-deferred "split & hygiene" stage (cut tracklets where the
tracker welded two players together, then let re-ID reassemble the fragments) be built?
**Answer:** not yet. One experiment decides it, and it needs a real detector.
**Raw:** `2026-08-01-split-contamination-shape.json`, `2026-08-01-split-localiser.json`,
`2026-08-01-split-oracle-ceiling.json`
**Related:** `2026-08-01-anchorless-merge-ab.md` (found while doing this work)

## Why this was worth asking

The split stage is §1 of the Notion "Re-ID Post-Tracklet System" spec, deferred by PRD
decision and never built. Its own status note argues it matters more than the page assumed,
because merging propagates contamination it did not create. The merge stage has been the
whole programme; nothing had ever measured what it is merging.

## Substrate — read this before any number below

All 8 SNMOT runs use `detect: oracle`. **0 of 76,933 tracklet frames fail to match a GT box
at IoU ≥ 0.5** — the tracklet boxes *are* the GT boxes. Two consequences run through
everything here:

- The contamination measured is purely **tracker association error**. The dominant
  real-world mechanism — one detection box straddling two players — is *structurally
  absent*.
- The geometric localiser (below) is evaluated on perfect geometry at exactly the moments
  where a real detector merges or drops boxes.

So these numbers characterise a tracker fed perfect boxes, not the system that would ship.

## 0a — the shape of contamination

Per-frame GT identity per tracklet (best IoU ≥ 0.5), collapsed into identity segments.

**The 51%-impure figure is inflated, and the corrected figure is threshold-dependent.**
Half of all tracklets are "impure" by raw purity, but 43.6% of identity segments are ≤5
frames. Filtering those as noise:

| segments ≤N frames treated as noise | 0 | 1 | 2 | 3 | **5** | 8 | 12 | 25 |
|---|---|---|---|---|---|---|---|---|
| cuts needed | 444 | 217 | 148 | 115 | **95** | 69 | 60 | 46 |
| tracklets needing ≥1 cut | 50.2% | 32.4% | 28.5% | 24.2% | **22.2%** | 20.3% | 19.3% | 16.9% |

The ordering is stable — most tracklets need no cut at all — but the headline moves between
17% and 50% depending on a free parameter. Quote the curve, not a point.

**And the filter is not obviously legitimate here.** With oracle boxes there is no IoU
labelling jitter, so a 4-frame alien run is a genuine association error, not noise. The
conservative reading is the unfiltered one: 444 cuts, 50.2% of tracklets.

At the 5-frame setting: 161 of 207 tracklets need 0 cuts, 27 need 1, and a tail reaches one
tracklet needing 9. Real alien segments (n=79) have median duration 37 frames (1.48 s), 33
of them under 1 s, max 272 frames.

## 0b — can geometry localise swaps without GT?

Two GT-free cues: **overlap** (runs of frames where another tracklet's box overlaps this one
above an IoU threshold — the close pass) and **gap** (a temporal hole inside the tracklet).
Ground truth is the boundary between consecutive real identity segments; tolerance ±12
frames.

| cue | recall | precision | events per true swap |
|---|---|---|---|
| overlap only (IoU ≥ 0.10) | 0.674 | 0.063 | 5.2 |
| gaps only | 0.379 | 0.271 | 1.5 |
| **overlap + gaps** | **0.979** (93/95) | 0.110 | 6.7 |

The two cues are strongly complementary — 0.674 and 0.379 combine to 0.979 — which is a
real finding and was not obvious.

**But most of that recall is bought by coverage, not by localisation.** The candidate
windows cover **57.0% of all tracklet frames** (median window 45 frames including tolerance,
p90 139). A shuffled-window null with identical event counts and identical widths scores
**recall 0.546**. So the informative margin is 0.979 against 0.546, not against zero, and
precision of 0.110 per event sits *below* the base rate implied by 57% frame coverage. As a
frame-level classifier the cue is not better than chance; it is useful only as a
cheap over-generating proposal mechanism.

**A further caveat on the ground truth itself:** the "true" cut point is the midpoint of the
handover interval between segments, and that interval has median width 4 frames but p75 25
and p90 102. For 24 of 95 boundaries it is wider than the ±12 tolerance, so the target is
genuinely ambiguous a quarter of the time. This understates recall (a cue can sit on the
real swap and miss the midpoint) and degrades the oracle splitter below.

## 0c — the oracle-split ceiling (the go/no-go)

Cut every tracklet at its true identity boundaries, then merge. This is the best any
splitter could do. **Anchorless** — the anchored version of this table was measuring GT
jersey labels reassembling fragments (see the companion report), and inflated the gain
roughly fivefold.

| arm | tuning | held-out | mean purity | entities |
|---|---|---|---|---|
| nosplit / pairwise | 0.9071 | 0.8792 | 0.9201 | 194 |
| nosplit / two-pass@−1 | 0.9022 | 0.8720 | 0.9201 | 202 |
| nosplit / no merge | 0.8734 | 0.8404 | 0.9201 | 228 |
| oracle-split / pairwise | 0.9182 | 0.8871 | 0.9819 | 249 |
| oracle-split / two-pass@−1 | 0.9083 | 0.9053 | 0.9819 | 239 |
| oracle-split / no merge | 0.8671 | 0.8471 | 0.9819 | 323 |

Paired per-sequence deltas (oracle-split minus nosplit), n=8:

| merge arm | mean Δ IDF1 | sd | t(7) | improved |
|---|---|---|---|---|
| no merge | **+0.0002** | 0.0153 | 0.03 | 3/8 |
| pairwise | +0.0095 | 0.0181 | 1.48 | 5/8 |
| two-pass@−1 | +0.0197 | 0.0214 | 2.60 | 7/8 |

Best arm to best arm — `nosplit/pairwise` → `oracle-split/two-pass@−1` — is **+0.0136,
6 of 8 sequences**, and the winning engine was chosen after seeing the result.

**Splitting alone is IDF1-neutral (+0.0002).** That is expected rather than damning:
splitting trades an identity error for a fragmentation error and IDF1 prices them
similarly. The stage's value proposition was always split-*then*-merge. But it does mean
every measurable gain is contingent on the merge engine's ability to reassemble, and that
engine is the one this programme has already characterised as evidence-limited.

## Verdict: do not build it yet

The entire theoretical ceiling — perfect cuts, perfect detections, best-of-both engines,
post-hoc engine selection — is **+0.0136 mean entity IDF1 at n=8**. A real splitter fed by
a localiser with a 0.979-vs-0.546 margin captures some fraction of that.

For scale, two things found *while* measuring this are worth more:

- Reverting the merge default is worth **+0.027** (see companion report).
- Enabling any merge at all is worth **+0.039** held-out (no-merge 0.8404 → pairwise 0.8792).

Building a stage whose best case is a third of a bug already on the floor would be the wrong
order of work.

## What would reopen it — one experiment

**Re-measure the ceiling on real detections.** One config, the `no merge` and `pairwise`
arms, 8 sequences. Everything above says contamination on oracle boxes is tracker-only; the
straddling-box mechanism that a real detector introduces is precisely the one splitting is
best suited to fix, and it is absent from this measurement. If the real-detector ceiling
exceeds **+0.03** mean entity IDF1, build the stage; if it stays under, close it permanently
and record that.

**This is currently blocked**, which is why it is not in this report:
`data/weights/football-player-detection.pt` is absent and `ultralytics` is not installed
(deliberately, per `CLAUDE.md` — supply per-invocation with `uv run --with ultralytics`).
It needs a Roboflow fetch.

## If it is built, the design the evidence supports

Not the spec's within-tracklet DBSCAN as primary detector — median alien segment is 37
frames inside tracklets with median length 309, and 33 of 79 are under 25 frames, which is
thin for density clustering. Instead **geometry proposes, appearance disposes**: overlap +
gap events generate candidates (0.979 recall, 6.7:1 over-generation), appearance decides at
those points. This also solves the sparse-invocation problem the mask-propagation page
raises — candidates come free rather than requiring segmentation everywhere.

Two things to carry over regardless:

- **Splitting errs in the safe direction.** A false cut fragments identity (→ abstention);
  a missed cut leaves a silent player swap. The product invariant ranks these unequally, so
  the split stage should run at a much more permissive threshold than the merge stage. That
  argument stands on its own and is not priced into IDF1 at all — purity goes 0.9201 →
  0.9819 in every split arm, and neither IDF1 nor MOTA can see it. If the case for this
  stage is ever remade, **it should be made on the ADR 004 semantic-identity layer, not on
  IDF1.**
- Do not bundle mask propagation with the first build, per the existing spec.

## Limits

- n=8 sequences, 4 held out; deltas of +0.01–0.02 at sd ~0.02 are not distinguishable from
  noise at this sample size.
- Oracle detection boxes and oracle team labels throughout.
- The claim "splitting manufactures the many-tracklets-per-player regime two-pass needs" was
  considered and **rejected**: after oracle splitting, two-pass is worse than pairwise on 5
  of 8 sequences, and the held-out inversion rests entirely on SNMOT-124 and SNMOT-125.
- The flicker threshold, the ±12 tolerance and the IoU threshold are all free parameters;
  the first is swept above, the other two are not.
