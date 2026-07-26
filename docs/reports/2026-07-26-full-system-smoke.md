# Full-system smoke test on real tracklets — new defaults (2026-07-26)

**Scope:** first end-to-end runs of the post-merge defaults (PRTreID backbone + permissive
merging) on **real TDLP-full tracklets**, rather than the GT-fragment substrate every prior
SPO-85 number came from. Two tuning sequences, ~20 min each; held-out remains quarantined.
Code: `main` @ `dd696f9`. Config: `configs/pipeline.tdlp-full-reid-oracle.yaml`.

**Detector substituted:** `data/weights/` does not exist in this tree, so the real-detector
config (`pipeline.tdlp-full-reid.yaml`, yolo-local) cannot run. Oracle detections stand in.
Everything else is the real path — real TDLP-full tracker, the new PRTreID second pass, real
kit-colour teams, permissive reid-engine, full evaluation.

## Result: the system works, and the extrapolation holds — at roughly double the predicted cost

| run | tracklets → entities | IDF1 tracklet | IDF1 entity | assoc gain | purity tracklet | purity entity | Δ purity | merge precision |
|---|---|---|---|---|---|---|---|---|
| systest-116 | 33 → 26 | 0.7380 | 0.8020 | **+0.0645** | 0.8184 | 0.8062 | **−0.0122** | 0.857 |
| systest-120 | 33 → 25 | 0.8750 | 0.9240 | **+0.0484** | 0.9753 | 0.9482 | **−0.0271** | 0.778 |

Functional health: both runs completed; every artifact written (`frame_features.npz`,
`association.json` with a full decision trail, `players.json`, `naming.json`,
`reid_detail.json`, `annotated.mp4`); `frame_features` shape `(1, 256)` with
`reid_model: prtreid` on both, confirming the second feature-gen pass is live in the real
pipeline.

## The finding worth acting on

**Merging costs 2–4.5× more entity purity on real tracklets than the GT substrate predicted.**
The GT-fragment measurement showed −0.006; here it is −0.012 and −0.027.

That is the compounding effect flagged repeatedly while the defaults were being set: GT
fragments are pure by construction, so a wrong merge joins two clean tracklets. A real
tracklet already contains swaps (tracklet purity 0.818 on SNMOT-116), so merging propagates
existing contamination as well as adding its own.

SNMOT-120 is the sharper case: the tracker handed it near-clean input (tracklet purity 0.975)
and merging still cost 2.7 points of entity purity at 0.778 merge precision — 7 correct
merges and 2 wrong. On a clean substrate a wrong merge has nothing to hide behind.

The IDF1 gain does transfer (+0.048 to +0.065 versus +0.068 on GT fragments), so the trade
remains positive on identity metrics. But the ratio is materially worse than the numbers the
default was chosen on.

## Caveats

- **Two sequences, both tuning.** This is a smoke test, not a benchmark. No held-out sequence
  was touched.
- Oracle detections, so detector error is excluded; a real detector would add its own
  fragmentation and its own contamination.
- The real tracker fragments *less* than natural-gap GT fragmentation (33 tracklets vs 49 GT
  fragments on SNMOT-116), so there are fewer merge opportunities and each one carries more
  weight.

## Suggested follow-up

Re-run the operating-point sweep on **real** tracklets rather than GT fragments. The margin
dial was calibrated on a substrate whose contamination cost is half what the real one shows,
so the chosen point (margin 0.0) may not be where the real curve wants to sit. That is a
measurement, not a guess, and the harness already exists.
