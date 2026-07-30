# Jersey OCR as a pairwise merge channel

**Date:** 2026-07-30
**Status:** Design, approved for planning
**Scope:** one new evidence channel for the re-ID merge decision. No anchor source, no naming
change, no shipped-default change.

## Why

Every merge channel measured so far — KPR appearance, PRTreID appearance, formation-relative
occupancy — has the same shape: a usable ranking body and an overlapping confident tail. That is
three distinct signals failing the same way, and the diagnosis in
`implementation-status.md` finding (e) is that the binding constraint is the upper tail, not
average separability. Adding a fourth weak-positive channel would be a fourth attempt at the
same thing.

Jersey OCR is different in kind: it is the first available channel that can produce **strong
negative evidence**. A confidently-read 7 against a confidently-read 9 is direct evidence
*against* a merge. Body, occupancy, continuity, gap and transition all range from "weakly for"
to "neutral"; none can veto. A channel that removes wrong merges attacks the tail directly,
which is what the operating curve is actually limited by.

## Constraints this design must respect

- **ADR 001 stands unamended.** Identity must not require jersey OCR. This channel abstains at
  exactly zero evidence when digits are absent or unreadable — by construction, not by a gate
  (see the likelihood ratio below). A pairwise channel needs no roster and no numbered kit, so
  it degrades to the existing non-OCR behaviour with no code path change.
- **ADR 003.** Missing evidence is neutral, never a penalty. `fuse()` already treats `None` this
  way.
- **ADR 002.** Numbers are decided per tracklet from aggregated per-crop evidence, never per
  frame.
- Optional modality status must be recorded in `implementation-status.md` with its **measured
  coverage**, so no future reader mistakes a benchmark result that leaned on numbered kits for a
  body-ID result.

## Build vs reuse

The reference system is Koshkina & Elder, *A General Framework for Jersey Number Recognition in
Sports Video* (CVPRW 2024) — 87.45% tracklet accuracy on the SoccerNet jersey test set, 79.31%
on the challenge set, and still the standard baseline (SoccerNet retired the standalone jersey
task; 2026 entrants used the same ViTPose + PARSeq + voting recipe).

Three of its four stages already exist here, so **only the recogniser is built**:

| Reference stage | Their model | Decision |
|---|---|---|
| Main-subject filter | Centroid-ReID + 3.5σ outlier rejection | **Reuse `crops.py`.** They reject contaminant crops statistically because they receive a bag of images. `sample_quality_crops` knows per-frame box overlap directly (`max_isolation_iou`), which is a measurement rather than an inference. |
| Localisation | ViTPose | **Reuse `pose/rtmpose.py`.** COCO-17 keypoints give the same torso quad, Apache-2.0, already attached to detections. |
| Recognition | PARSeq, fine-tuned on jersey digits | **Build.** The only real gap. |
| Legibility | ResNet34 binary classifier | **Build, but soft.** `part_visibility` says a torso is unoccluded, not that digits are readable (player facing away, motion blur, small crop). Their classifier is weak anyway — 94.5% accuracy at 71.7% F1 — and in this framework it need not be a hard gate; it becomes a per-crop weight. |
| Consolidation | confidence-weighted vote, or per-digit log-likelihood + temperature + count prior | **Reuse `reid/evidence.py`.** Their probabilistic variant is a hand-rolled sketch of the calibrated-LLR layer already here. |

Not used: SAM (only in their hockey label-generation path), Centroid-ReID, and their five-conda-env
`setup.py`.

**Weights.** PARSeq is Apache-2.0 and loaded in-tree; their jersey fine-tune is worth roughly six
points of digit accuracy (85.4% → 91.4% image-level on hockey) and is *weights only*, so it needs
no adoption of their CC BY-NC harness. **Consequence to state in every report from this channel:**
the checkpoint was fine-tuned on hockey and SoccerNet, so on SNMOT it is train-adjacent. That
weakens accuracy claims made on SoccerNet-derived data and must be disclosed, not hidden. Under
the repo's research posture the licence records provenance and gates nothing.

**Environment.** No `external-spotters/`-style isolation. PARSeq runs in-tree from a downloaded
checkpoint, supplied per invocation in the `rtmlib` idiom (`uv run --with ...`) rather than
declared as a dependency.

## The channel

### Per-tracklet likelihood, not a number

The reader emits a **likelihood vector** `L_t(n)` over `n ∈ {0…99}` for each tracklet — never a
decision. Per crop, PARSeq's per-position character probabilities give a distribution over
strings; these are combined across the tracklet's gated crops as a weighted sum of
log-likelihoods, the weight being the crop's legibility × quality. This is ADR 002's
tracklet-level aggregation and the paper's probabilistic consolidation, in the currency
`evidence.py` already uses.

Two priors enter here, both from the data rather than invented: the digit-count prior (single vs
double digit — 39% single in the reference dataset) and the number-frequency prior `prior(n)`.

### Pairwise likelihood ratio

For a candidate pair, marginalise over the unknown true number:

```
        Σ_n  prior(n) · L_a(n) · L_b(n)              ← one number generated both reads
LR  =  ──────────────────────────────────────────
       (Σ_n prior(n)·L_a(n)) · (Σ_m prior(m)·L_b(m)) ← two independent numbers
```

`log LR`, passed through `saturate()`, is the channel's contribution to `fuse()`.

Four required properties fall out of this rather than being engineered — which is the reason to
derive it instead of porting the paper's vote:

1. **Illegible is exactly neutral.** Flat vectors make numerator and denominator agree, so the
   LLR is 0. ADR 003 abstention with no gate and no threshold.
2. **Common numbers carry less weight.** Dividing by `prior(n)` means agreement on a frequent
   number is weaker evidence than agreement on a rare one — the same impostor-population
   informativeness argument already written into `evidence.py`'s docstring.
3. **Disagreement is strong negative evidence.** The property this channel exists for.
4. **Cross-team number collisions self-handle** via the prior, with the existing team channel
   doing the rest. No special case.

### Calibration: analytic LR plus one fitted temperature

**Not `LLRCalibrator`.** Two reasons, both binding:

- **Precedent.** `multi_input.py` deliberately excludes `transition` from `CHANNELS` because "it
  does not have a scalar raw score to hand to `LLRCalibrator`: the `TransitionPrior` IS its
  calibrator". Jersey has that exact shape — a joint likelihood, not a scalar similarity.
- **Sample size.** The GT-tracklet substrate carries 153 true re-entry pairs. A 20-bin histogram
  density ratio on that is per-bin noise, and a degenerate operating curve produced by the
  calibrator rather than the signal is a mistake already made once on this codebase.

So: the analytic LR above, with a **single scalar temperature** on the per-crop log-likelihoods,
fitted on real impostor pairs from the harness. One parameter is defensible at n=153; twenty bins
are not.

### Crop selection

`sample_quality_crops` is reused with a raised `per_tracklet` (32 rather than 8) — the reference
system reads every legible frame of ~482-frame tracklets, and 8 crops would starve the estimate.
**Temporal spread is retained** even though pure digit legibility would favour taking the best N
regardless of time: spread is the defence against an entire tracklet's read coming from one
instant of systematic misreading, which is the primary risk below.

## Code placement

- `matchlab_core/reid/jersey.py` — **pure**: gated crops + keypoints in, `L_t(n)` out; plus the
  pairwise LR. No model, no I/O, so both halves are testable against hand-computed vectors.
  Mirrors `pair_features.py`'s posture.
- `matchlab_core/ocr/parseq.py` — the reader front-end, alongside `pose/rtmpose.py` and following
  its lazy-import, loud-on-missing-dependency, provenance-recording convention.
- `matchlab_train/experiments/` — the measurement experiment, following
  `position_evidence.py`, using `reid/frontier.py` for the operating curve.
- `multi_input.py` — `jersey` added as a `transition`-style channel (own calibrator, outside
  `CHANNELS`).

## Evaluation

**The substrate problem, stated up front.** The `multi_input` fusion harness runs on FOOTPASS
tactical h5 data. There are **no pixels** — no FOOTPASS video is on disk. So the primary
channel-comparison harness *cannot* evaluate this channel. The venue is the SPO-85 GT-tracklet
harness on SNMOT, where the oracle track stage yields clean fragments and SNMOT GT already
carries `track.jersey`, so labels are free. That substrate is small (153 true pairs over 323
fragments from 198 tracks, 8 sequences), so `ingest-soccernet` must be scaled past `--limit 8`
before any tail figure is trusted.

Gates, in order. Each must pass before the next is run:

1. **Reproduce the reference metric on reference data.** Run the reader against the SoccerNet
   jersey test set and land near **87.45% tracklet accuracy**. Requires downloading that dataset
   (not currently on disk). If it does not reproduce, the fault is local wiring, and discovering
   that later on SNMOT crops would make the failure unattributable.
2. **Reader accuracy on SNMOT crops** — the actual target data, per tracklet, against GT jersey,
   densely rather than in aggregate. Expect well below 87%: SNMOT crops are smaller than that
   dataset's curated ones. **Report legible-fraction per tracklet**, because coverage sets the
   channel's ceiling and is the number that belongs in `implementation-status.md`.
3. **Channel alone.** AUC and the zero-wrong frontier via `reid/frontier.py`, in the same
   protocol as position (AUC 0.771; 14 of 13,016 needed merges at the zero-wrong frontier) so the
   comparison is direct. **Report wrong merges prevented separately** — that is this channel's
   actual claim and AUC cannot see it.
4. **Fused, ablated in and out**, weights via `fit_fusion_weights`. Do-no-harm on
   unnumbered/illegible pairs holds by construction; verify it empirically regardless.

Pre-registered before any measurement: the reader-reproduction bar in gate 1, the coverage figure
gate 2 must report, and the wrong-merges-prevented statistic gate 3 is judged on. Scope every
negative finding to the decision rule tested.

## Risks

**Correlated OCR errors break the independence assumption — the main risk.** The LR's denominator
assumes two different players' misreads are independent. They are not: 6↔8, 1↔7, and
single-vs-double-digit truncation are systematic. Two different players both misread as "1" will
present as strong same-player evidence, which is precisely a confident wrong merge — the failure
this channel was adopted to reduce. Mitigations: fit the temperature on real impostor pairs
rather than a synthetic prior; retain temporal spread in crop selection; and **inspect the
confident-impostor tail case by case** rather than trusting the aggregate, since the tail governs
merge safety.

**Train-adjacency of the checkpoint** (above) — disclosed in every report, not mitigated.

**Small substrate.** 153 true pairs cannot resolve a tail. Scaling ingest is a prerequisite for
gate 3's headline, not an optional improvement.

## Out of scope

- The anchor path (`AnchorSource` → roster naming). The seam exists and a real jersey source would
  drop into it, but it is a separate decision and separate evaluation.
- Any change to shipped `reid-engine` defaults.
- Retraining or fine-tuning PARSeq ourselves.
