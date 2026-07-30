# Two-pass merging shipped as the default associate engine — ties the incumbent on SNMOT

**Date:** 2026-07-31
**Code:** `matchlab_core/reid/twopass.py`, `reid-engine` `merge_strategy="two-pass"` (default)
**Model:** `configs/reid/fusion-footpass-v1.json`
**Raw:** `2026-07-31-twopass-threshold-calibration.json`, `2026-07-31-twopass-heldout.json`,
`2026-07-31-twopass-ab-snmot120.json`

## What shipped

The research system measured on FOOTPASS is now the product default. Tracklets accumulate
into identity threads (pass 1, causal, time-ordered), then whole threads merge against each
other (pass 2, repeated to convergence). Four calibrated log-likelihood-ratio channels in
nats: appearance, pitch occupancy, elapsed time, and a bounded-diffusion transition prior.

Only two hard constraints, both physically certain: same team, and tracklets that never
overlap in time. Motion feasibility is a **scored channel** here, not the veto it is in the
`pairwise` gate list -- measured flip counts showed the veto costing far more correct merges
than it prevented wrong ones.

Calibrators, prior and weights (`body 2.03 / gap 0.95 / transition 0.47 / occupancy 0.33`)
are fitted offline on all three FOOTPASS matches, so there is no run-time fitting step.
`merge_strategy="pairwise"` still selects the incumbent single-pass union-find engine.

## Threshold calibration (declared split, no peeking)

Chosen on **tuning** sequences SNMOT-116/118/120/123 only; **held-out** SNMOT-124/125/126/127
scored once afterwards at the chosen value. All arms re-run only the associate stage over
frozen upstream artifacts, so detections, tracklets, teams, calibration and features are
byte-identical across every arm.

**Tuning:**

| arm | merges | wrong | precision | mean IDF1 |
|---|---|---|---|---|
| pairwise | 18 | 3 | 0.833 | 0.9104 |
| two-pass @ 8.0-4.0 | 11 | 0 | 1.000 | 0.9018 |
| **two-pass @ 2.0 (selected)** | 15 | 1 | 0.933 | 0.9096 |
| two-pass @ <=1.0 | 16 | 2 | 0.875 | 0.9096 |

**Held-out (SNMOT-124/125/126/127), scored once at 2.0 — replicates tuning:**

| arm | merges | wrong | precision | mean IDF1 |
|---|---|---|---|---|
| pairwise | 18 | 3 | 0.833 | 0.8837 |
| **two-pass @ 2.0** | 13 | 1 | **0.923** | 0.8824 |

Per sequence: SNMOT-124 5p/5ok vs 7p/6ok; SNMOT-125 7p/7ok vs 8p/8ok; SNMOT-126 1p/0ok vs
3p/1ok; SNMOT-127 both arms merge nothing (IDF1 0.9897 — the tracker already produced one
tracklet per player, so there is no re-ID decision to make). SNMOT-127 was scored after the
other three, at the same threshold; the threshold was not re-selected.

## The finding

**Two-pass does not beat the incumbent on SNMOT. It ties it.** Mean IDF1 differs by 0.0016 on
held-out and 0.0008 on tuning -- noise at n=4. What it delivers at the same IDF1 is
**one wrong merge instead of three**, which is the trade the product's own invariant asks for:
a silent player swap is worse than a temporarily unknown player.

Per sequence at 2.0 on tuning, it matched pairwise's IDF1 exactly on SNMOT-116/120/123 and
lost 0.003 on SNMOT-118 by declining a 1-of-2 coin-flip merge.

## Why the FOOTPASS gain cannot appear on this benchmark

SNMOT clips are ~750 frames and carry roughly **1.2 tracklets per player** (SNMOT-120: 33
tracklets, 26 GT players, at most 7 merges possible). Accumulation is the entire mechanism
behind the FOOTPASS result -- pooling many tracklets into a richer description of a player --
and there is nothing to accumulate when a player has one tracklet. FOOTPASS halves carry
~2,100 tracklets for ~26 players. These are different regimes.

This was stated before the runs, not after seeing the numbers. **The benchmark is
structurally incapable of discriminating the engines**, and the tie should be read as "no
evidence either way on short broadcast clips", not as "the FOOTPASS result did not transfer".

The threshold grid also saturates below 1.0 on both splits: once the hard constraints bind,
the threshold stops mattering.

## Also worth recording

The FOOTPASS-fitted threshold of 4.0 is **too strict for SNMOT** -- it makes 11 merges where
tuning wants 15. This is the domain gap recorded in the model's provenance block, showing up
exactly as predicted. The shipped default remains 4.0/2.0 because it was fitted, not tuned;
2.0/0.0 is the SNMOT-calibrated value and is the right setting for that footage.

## Limits

- Validated on 4 held-out SNMOT sequences with oracle boxes; on one of them (SNMOT-127) neither engine makes a single merge, so it carries no signal. No phone footage, no long match.
- The fusion model is fitted on FOOTPASS with oracle pitch coordinates and oracle team labels.
- The real test of accumulation needs footage with many tracklets per player. Nothing in the
  current benchmark set provides it.

## Open, unbuilt

- E0 separability diagnostic on the merge frontier (would confirm the evidence-limited
  conclusion directly rather than by inference).
- Separate calibration for symmetric thread-thread pairs; pass 2 reuses calibrators fitted on
  asymmetric thread-vs-single-tracklet pairs. Most likely-real remaining modelling error.
- A margin rule in pass 2 (pass 1 has one, pass 2 has none).
- `min_frames=50` in the research harness removes the hardest 7-13.5% of decisions.
