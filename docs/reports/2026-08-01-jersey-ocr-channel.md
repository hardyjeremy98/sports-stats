# Jersey OCR as a re-ID merge channel

**Date:** 2026-08-01
**Status:** Measured, partially wired (OFF by default), several improvements queued unwired
**Scope:** SPO-90s jersey-OCR merge-channel work, 2026-07-30/31. Design:
[`docs/superpowers/specs/2026-07-30-jersey-ocr-merge-channel-design.md`](../superpowers/specs/2026-07-30-jersey-ocr-merge-channel-design.md)
(and its Amendment 1). Ledger:
`.superpowers/sdd/2026-07-30-jersey-ocr-merge-channel/progress.md`.

## Why

Every merge channel measured so far — KPR appearance, PRTreID appearance, formation-relative
occupancy — has the same shape: a usable ranking body with an overlapping confident tail, so
none can veto a wrong merge. Jersey OCR is the first channel that can produce strong *negative*
evidence: a confidently-read 7 against a confidently-read 9 argues directly against a merge.
ADR 001 stands unamended throughout this work — this is a merge-evidence channel, not an
identity foundation, and it is designed to abstain at exactly zero evidence with no gate.

## Gate 1 failure: 0.1367

The design's PARSeq recogniser + ViTPose-equivalent (RTMPose) crop + length-prior + legibility
weighting pipeline was run against 1211 SNMOT-derived tracklets pre-registered against an 0.8445
bar (reference: 87.45% tracklet accuracy, Koshkina & Elder CVPRW 2024). First run scored
**0.1367** (117 correct / 676 wrong / 63 abstained / 355 illegible-per-GT). The plan called this
a hard stop.

## Three root causes, all spec-level, none implementation bugs

Diagnosis traced the failure to three defects in the original design, not in how it was coded:

1. **Length prior collapsed everything to single digits.** Renormalising 10 single-digit
   candidates to 39% of mass and 90 double-digit candidates to 61% gives each single-digit
   candidate a ~9.5x per-candidate advantage. Under diffuse evidence the argmax was forced
   single-digit regardless of the pixels — 80% of predictions were {1, 2, 3} at confidence 1.0.
   Fix: `single_digit_prior` defaults to `None`.
2. **Crop region.** The full player crop (median 120h×51w) resized to 128×32 destroys the
   digits. A band sweep on 300 legible tracklets (vote, prior=None) found: full crop 0.0800,
   y[0.00,0.55] 0.1867, y[0.10,0.45] 0.2300, **y[0.12,0.50] 0.2533 (best)**, y[0.20,0.45] 0.2467,
   y[0.05,0.35] 0.0833. RTMPose torso crops scored 0.137 — worse than any fixed band, because
   RTMPose keypoints are unreliable at ~120×51px. RTMPose was removed from this path entirely
   rather than kept as a fallback option.
3. **No legibility gate — the dominant defect.** Visual inspection of montages showed many
   tracklets are mostly front-facing crops with no visible number, aggregated at weight 1.0
   alongside genuinely legible crops. Adding a ResNet34 legibility classifier
   (`data/weights/legibility-resnet34-soccer.pth`) as a hard gate, measured on 450 tracklets
   (band y[0.12,0.50] + vote): gate none → acc 0.1622 (73 correct / 377 wrong); gate ≥0.5 →
   0.5111 (230/40); **gate ≥0.9 → 0.5222 (235/29)** — wrong reads fell 13x from ungated.

A fourth candidate cause — the consolidation method itself (confidence-weighted vote vs the
spec's probabilistic LLR aggregation) — was tested by bisection on 300 legible tracklets with
identical crops and identical model: both scored 24/300. Consolidation was never the fault; the
defects were entirely upstream, in what was fed to the model.

## Repair to 0.8026

Once the three causes were fixed, crop *budget* turned out to be the dominant remaining lever,
because the legibility gate discards most candidates and the reader was starved. Crops must also
be evenly strided across the tracklet, not the first N: 12 crops → coverage 0.361; 40 → acc
0.5644 / coverage 0.447; **100 → acc 0.8178 / coverage 0.887**; 250 → 0.8133 (plateau). On the
full 1211-tracklet gate-1 set with the final config (band y[0.12,0.50], legibility ≥0.9, 100
strided crops): **972 correct / 127 wrong / 112 abstained = accuracy 0.8026**; legible-only
precision 0.8938, coverage 0.8692.

This is against a pre-registered bar of 0.8445 (reference 0.8745) — **gate 1 failed its own
bar**. The decision to proceed anyway was made by the controller while the plan's author was
unavailable, and is disclosed here per that decision's own condition: the bar existed to catch
wiring errors and caught three real ones; the residual 4.2pp gap is attributed to two named,
unreplicated differences from the reference — it uses a ViTPose-derived torso crop (this work
uses a fixed band after RTMPose was rejected) and consumes every frame of ~482-frame tracklets
(this work samples 100). This attribution is not independently re-verified; it is the stated
rationale for proceeding, not a closed finding.

Also measured at this stage: correlated-error risk is real. P(identical wrong value | both
wrong, different true numbers) = 0.0687 vs ~0.01–0.05 under independence; wrong reads have
attractors ("27" absorbed 9 distinct true numbers, "8" and "10" 6 each). An unconditioned
false-veto rate (same-number pairs read as different values) of 0.1244 was measured on this
curated split and reported at the time as disqualifying for the veto property — **this figure
did not reproduce on the SNMOT substrate** (see below) and must not be cited going forward.

## Rule sweep and review

An 868-fragment evidence cache (16/16 tune/held clip split) was used to sweep rule families
offline. Winner class: soft weights (`band_legibility^a * conf^b`) with a Σw-normalised
posterior (ρ) and top1–top2 margin abstention (τ), which Pareto-dominated the shipped hard-gate
rule on held-out data — e.g. `soft a2 b1 rho1 tau4`: precision 0.993 / coverage 0.586 vs shipped
hard@0.9's precision 0.963 / coverage 0.535, better on both axes. Tune and held-out results
agreed closely (no overfit signature).

An independent reviewer (opus) re-scored the rule family at the pair level on held-out clips,
including zero-evidence ties: shipped rule 0.707 AUC / 0.281 pairs touched; the winning rule
`a1 b1 rho0.5 tau2` 0.800 AUC / 0.422 touched, veto precision 0.999–1.0 maintained. A τ=0 control
confirmed margin abstention is load-bearing (veto drops to 0.984, zero-wrong prefix drops to 0
without it). A pre-registered two-pass TTA re-read of abstainers was tested and **rejected** by
its own pre-registered rule: only 4 correct / 2 wrong recovered from 292 abstainers (precision
0.67, far below the 0.96 headline) — an honest negative confirming abstained fragments are
genuinely unreadable near the substrate's visibility ceiling, not an artifact of a weak reader.

The winning rule (`rho1`, log-odds margin `tau2`, principle-chosen rather than held-out-shopped)
was wired into `tracklet_likelihood` and `JerseyReader`'s soft weights (default `margin_tau: 0`
preserves old behaviour unless explicitly enabled), and both shipped experiments were re-run
end-to-end, regenerating the previously-stale gate-1 artifact.

## SNMOT gate 3 and the fusion result

**Reader frontier on SNMOT** (32 clips, 1234 fragments, 24386 pairs, oracle-fragment
tracklets): shipped rule precision 0.953 / coverage 0.637 (legible: 436 correct / 23 wrong / 409
abstained); full precision/coverage curve spans 1.0@0.465 to 0.80@0.84 as the margin threshold
relaxes. ROC-AUC 0.9859 — a different class from position evidence (0.771) or comparable
appearance figures. **Veto precision 0.9994** on 4888+ fired pairs — this is the number that
supersedes the earlier curated-split 0.1244 false-veto figure, which did not reproduce here.
Do-no-harm holds by construction and empirically: thousands of both-flat pairs, zero violations,
max|LLR| exactly 0.0.

**Fusion ablation — the goal gate (commit `06f78ec`, adversarially reviewed and survived, with
corrections in `7c3900f`).** Held-out 16 clips, 7474 pairs, OSNet body arm (PRTreID not yet
integrated into this harness), shipped jersey rule, exact prefix scan (not the buggy
`zero_wrong_frontier` function — see Open items):

| arm | AUC | @ zero-wrong | @ ≤10 wrong |
|---|---|---|---|
| body-only | 0.822 | 0 | 13 |
| jersey-only | 0.787 | 14 | 45 |
| **fused (sum)** | **0.887** | **50** | **69** |

The zero-wrong gain is spread across 15 of 16 clips (max 9 in any single clip), and a
shuffled-body control refutes a tie-breaking artifact explanation (0–5 vs 50 with real body
scores). No tune/held leakage, no label leakage, no LLR saturation ties were found on review.
Do-no-harm was re-verified on 4280 held-out zero-jersey-evidence pairs (an earlier report cited
7199, which was the all-clips count, not the held-out count — corrected in review).

**Honest headline: fused 50 vs jersey-alone 14 vs body-alone 0, at zero wrong merges, held-out,
exact prefix scan.**

## Engine wiring and the unit-mismatch fix

Task 8 wired the channel into `reid_engine.py` (commit `a61fdba`). Review caught an IMPORTANT
defect before it shipped: the engine added the jersey log-likelihood-ratio (range ±6 nats, from
`saturate()`'s clamp) directly to cosine similarity (range ~[-1, 1]), so one confident channel
disagreement could act as an absolute veto — violating the `LOG_CLAMP` no-absolute-veto invariant
and *not* the configuration Task 7 had actually measured (which fused matched LLR units). Fix
round 1 (commit `70982d0`) bounded the contribution to
`jersey_weight * (llr / LOG_CLAMP)`, default `jersey_weight: 0.15` (below the 0.20 merge band),
added a no-absolute-veto test, and documented true LLR-space fusion as a principled follow-up
rather than the shipped shortcut. The same round corrected a report that had cited the
superseded 0.1244 false-veto figure as justification for keeping the channel default-off; the
real justification is the unvalidated real-tracker substrate (see Open items).

**Current state: the jersey channel is live in `reid_engine.py` behind `jersey_enabled: bool =
False` (default OFF), with bounded influence (±0.15 similarity units), no-absolute-veto tested,
provenance recorded, and an eval config shipped.**

## Measured-but-unwired improvements queue

These were measured in offline ideation rounds but not implemented in the shipped path:

- **Suffix decomposition** (+5–9pp headline hope; measured gain smaller but real): treating a
  single-digit read as a suffix constraint on a 25%-weighted mixture moved the frontier outward
  at mid-coverage — held tune point t1: 0.906/0.717 vs baseline 0.814/0.806 (wrong 71→32); t2:
  0.975/0.586 vs 0.924/0.666 (wrong 24→7). Follow-ups needed: fit the mixture weight properly,
  add true per-position pairwise partial-agreement scoring.
- **Garbage-mixture calibration**: attractors are real, not a base-rate artifact — "3" appears
  at 22x excess over its GT frequency among illegible reads, "1" at 12x. A calibration term for
  this has a genuine target but is not yet implemented.
- **Epsilon-clamped `pair_llr` numerator**: the measured correlated-error rate (eps=0.0687) caps
  achievable positive LLR at ~2.7 nats if applied; not yet wired.
- Also queued and lower-priority: n_eff-by-runs correction (P1), entity-weighted coverage (P7),
  char-temperature calibration (P3, blocked on cache v2 storing raw logits instead of
  renormalised probabilities), clip-aware EOS marginalisation (P5, precision-only lever,
  measured: rho1/tau2 held 0.924/0.666 → 0.951/0.645, wrong 24→15, but does not recover
  length-error failures as corrects).
- Rejected, not queued: orientation-aware sampling (P8) — legibility is not temporally
  concentrated (median 0.49), so this has no lever to pull; two-pass TTA re-read — already
  tested and rejected above; oracle roster restriction — modest gain (0.68→0.72 at full
  coverage) even given the true roster, so deprioritized.

## Open items

- **Real-tracker substrate.** Every number above is measured on oracle-fragment tracklets (the
  SPO-85 GT-tracklet harness, oracle TRACK stage) — clean fragments by construction. Nothing has
  been measured on real-tracker output tracklets, which carry contamination and fragmentation
  the oracle substrate does not. This is the single largest gap between "strong channel" and
  "improves re-ID" as a product claim.
- **PRTreID body arm.** The fusion comparison above uses OSNet for the body channel; the re-ID
  engine's current default backbone is PRTreID (per the SPO-85 report), and that combination has
  not been re-measured with jersey fused in.
- **LLR-space engine fusion.** The shipped wiring uses a bounded-but-ad-hoc
  `jersey_weight * (llr / LOG_CLAMP)` contribution to similarity space to avoid an absolute
  veto; principled LLR-space fusion (matching Task 7's actually-measured configuration) remains
  a follow-up, not done.
- **`zero_wrong_frontier` function bug, undiagnosed.** `reid/frontier.py`'s
  `zero_wrong_frontier` returned an internally incoherent `correct=0, wrong=0, threshold=None`
  on the SNMOT gate-3 run despite AUC 0.9836 — incoherent on its face, since a non-trivial AUC
  with zero merges recovered by the frontier function is not possible if the function is working.
  Prime suspect: `saturate()`'s tanh clamp bounds LLR at ±6.0, so confidently-agreeing pairs pile
  up at one value and mutual-best+margin admission cannot resolve the tie — the same mechanism
  that once tied 19.7% of decisions on the body channel. **This bug has not been diagnosed or
  fixed. It is excluded from every claim in this report and in `implementation-status.md`; all
  merge-count figures above (gate 3, fusion ablation) were produced by an exact prefix scan
  instead of this function.** Any future consumer of `zero_wrong_frontier` on this channel must
  diagnose this before trusting its output.
- **Checkpoint train-adjacency.** The PARSeq checkpoint is fine-tuned on hockey + SoccerNet, so
  every accuracy figure measured on SoccerNet-derived data (SNMOT) is train-adjacent, not an
  independent-domain result. Disclosed per-figure above; not mitigated.

## Commits

`f845d14`..`4f19a9b` (Task 1), `..a149fc2` (Task 2), `..203bb6d` (Task 3, incl. probe fix
round), Task 4 gate-1 diagnosis/repair (no single commit — cache + sweep artifacts), `4bed9d3`
(Task 6 wiring), `06f78ec` + `7c3900f` (Task 7 fusion, adversarially reviewed), `a61fdba` +
`70982d0` (Task 8 engine wiring, unit-mismatch fix).
