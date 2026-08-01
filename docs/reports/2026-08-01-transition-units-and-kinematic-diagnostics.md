# Transition-prior units bug + kinematic-evidence diagnostics (2026-08-01)

Branch: `reid-conditioning` (commit `ecbc54e`). Follows the measure-first plan for
"use position/velocity/candidate-density properly in re-ID": three diagnostics on
existing runs, then only the changes the measurements supported.

**Substrate for everything below** (read the caveats before generalising): the six
calibrated SNMOT sequences — tuning `fixA-{120,123,125}`, held-out
`heldout-{124,126,127}` (oracle detections + tdlp-full), plus `gate2base-*` (oracle
detections + IoU tracker) for the short-gap regime. GT-pinned team labels (the
kit-colour gate's false vetoes are SPO-75/SPO-87's problem and were isolated out).
Anchorless in-process replays of `merge_threads_two_pass` with
`configs/reid/fusion-footpass-v1.json`. Oracle boxes, P=1 PRTreID features,
30-second clips: none of this has been re-measured on a real detector.

## Headline: the transition channel was a constant, not evidence

`_tracklet_evidence` built `entry_xy`/`exit_xy` in raw pitch **centimetres**;
`displacement()` expects **normalised [0, 1]** endpoints (the convention the
FOOTPASS prior was fitted in) and multiplies by pitch metres itself. Every
displacement came out ~1e5 "metres", the LLR saturated at −6 nats, and the channel
contributed exactly −2.82 nats (0.4707 × −6) to **every pair scored by the shipped
two-pass engine** since it became the default. The tell in the diagnostics: median
transition contribution identical (−2.82) for true-positive, true-negative, and
wrong pairs. A representative pair scores −6.4e7 nats raw as shipped, +1.53
normalised.

Measured A/B (same harness, conf floor 0, right/wrong = GT-classified merge edges,
missed = consecutive same-player tracklet pairs left unlinked):

| operating point | broken: right/wrong/missed | fixed: right/wrong/missed |
|---|---|---|
| pass1 4.0 / pass2 2.0 | 5 / 0 / 17 | 10 / 3 / 12 |
| 2.0 / 1.0 | 6 / 0 / 16 | 14 / 3 / 8 |
| 0.0 / 0.0 | 8 / 0 / 14 | 15 / 3 / 7 |

The broken engine's precision 1.0 is vacuous — it barely merged. The 3 wrong edges
at every threshold are appearance-driven (body cosine ≥ 0.96 between different
players on the P=1 substrate) and exist with or without the fix; autopsy below.

This is the third instance of the matched-units failure class (after the jersey
raw-nats veto and the LLR/cosine mix): **channels fused as a sum must assert their
input conventions.** The fix normalises the endpoints at the source and adds a
regression test pinning entry/exit to the xs/ys convention.

## D1 — kinematic residuals at true gap sites (velocity channel: don't build)

For every true gap site (tracklet a → same-GT-player tracklet b) and its
engine-eligible wrong candidates: residual of each candidate's entry to (static)
a's exit point, (vel) exit + v̂·Δt with v̂ from a 1 s LSQ window backed off the
endpoint, (bidir) forward/backward extrapolation met at the time midpoint.
Top-1 = true candidate has the smallest residual.

IoU-tracker substrate (n=118 sites — TDLP substrate has only 21, nearly all >7 s):

| Δt bin | n | top-1 static | top-1 vel | top-1 bidir | med r_true / r_best-wrong (m) |
|---|---|---|---|---|---|
| 0–1 s | 47 | **0.94** | 0.92 | 0.81 | 0.7 / 7.3 |
| 1–3 s | 20 | 0.60 | 0.60 | 0.74 | 6.5 / 7.6 |
| 3–7 s | 21 | 0.62 | 0.50 | 0.40 | 7.3 / 8.4 |
| 7–15 s | 30 | 0.10 | 0.04 | 0.21 | 26.4 / 14.5 |

- **Position alone is nearly decisive under 1 s and dead past 7 s.** The static
  endpoint prior — exactly what the (now fixed) transition channel encodes — owns
  the winnable regime.
- **Velocity extrapolation adds nothing anywhere.** Endpoint velocity is dominated
  by calibration jitter (median implied per-frame speed 3.6 m/s on projected GT;
  11.5% of steps >12 m/s). Velocity continuity |Δv|: true 3.27 vs wrong 4.04 m/s
  median — barely separated. Bidirectional helps only in the 1–3 s bin (0.74 vs
  0.60, n≈20).
- **TDLP consumes the short gaps**: what survives to re-ID on the shipped tracker
  is 90% >7 s, adversarially selected against kinematics — SUSHI's "position is
  short-gap-only" prior, measured here on the exact surviving population.
- Local density is 0–1 players near the exit point at almost every SNMOT site;
  the crowded-regime concern doesn't arise on this data.

**Decision: the velocity-extrapolated channel is not built.** Reconsider only
after calibration smoothing/denoising lands, and validate on a substrate with
short-gap sites the tracker hasn't already consumed (post-severing fragment
re-joining, or FOOTPASS).

## D2 — candidate-set audit (decision rule: no signal to fix here)

At every operating point from threshold 0.0 to 4.0, margin bars 0/1/2 nats change
nothing on this substrate. Pre-fix the engine made essentially no wrong merges
(nothing for a margin to separate); post-fix the 3 wrong edges are not
margin-separable either:

- fixA-120 `0→31` and heldout-126 `4→13`: the true player **never returns** —
  the correct hypothesis was "none", and appearance voted same at cosine 0.963+.
- heldout-124 `5→25`: the true continuation (27) was scored and lost by ~11 nats
  (−8.37 vs +2.61) — an appearance deficit, not a decision-rule miss.

The recall side is the real finding: 21 of 29 scored true pairs had **negative**
fused totals pre-fix; channel attribution put the constant −2.82 transition tax on
all of them (dropping it alone rescued 7/21) with low cross-view body cosines
carrying the rest. The pass-1 margin machinery added in the SPO-87-adjacent work
already exists; extending it into a full hypothesis-set rule has no measurable
signal on 30-second clips and should be evaluated on FOOTPASS, where merges are
plentiful.

## D3 — camera coverage (the exclusion argument doesn't bind on broadcast)

Fraction of (gap frame × 5 m-disc sample) points of the exit neighbourhood that
are on-camera during the gap: median **0.05** on the TDLP substrate (0.15 on the
IoU substrate); <50% covered at 76–100% of sites; ≥95% covered at 0–15%. On
panning broadcast, a re-ID gap mostly *is* the camera leaving — "no one else was
seen moving there" is unavailable as evidence, and a visibility-conditioned
"unseen player" hypothesis would carry nearly all the mass. Static tripod phone
footage is the setting where closed-world exclusion becomes strong; nothing to
build until such footage exists.

## Also landed: occupancy sample-size shrinkage (opt-in)

`FusionModel.occupancy_shrink_n0` (default 0.0 = off) scales the occupancy
contribution by n_min/(n_min+n0). JS distance between sparse footprints is biased
high — on short clips the channel voted against *every* merge (median −1.08 nats
on true pairs). Shrinkage is a no-op in the FOOTPASS fitted regime (n in the
thousands) and fades the channel toward neutral where evidence is thin (ADR 003
applied continuously). Measured at n0=750, 4.0/2.0: tuning right 6→9 (wrong flat
at 1), held-out unchanged (4/2/4). Direction right, cost zero, out-of-sample gain
unproven — hence opt-in until validated at FOOTPASS scale.

## Infrastructure finding: `calibration_min_confidence: 0.5` starves the engine

The calibrator's confidence on these runs lives in 0.15–0.40; the engine default
0.5 keeps ~3% of frames, so occupancy/transition see almost nothing (and the
2026-07-30 run set was produced that way). The homographies are 100% in-bounds on
projected GT regardless of confidence. Diagnostics used floor 0; in the replay,
floor 0 vs 0.5 flipped the one wrong merge at 2.0/1.0 to right. The default and
the calibrator's confidence scale need to be reconciled — flagged, not changed.

## Deliberately not done

- **SPO-75 (kit-colour gate false vetoes):** a parallel session is actively on it
  (SPO-87 team-slot measurement, uncommitted work in this repo today); diagnostics
  here pinned teams to GT instead so the contamination could not leak in.
- **Gap-conditioned body calibration** (GHOST-style): requires a FOOTPASS refit;
  SNMOT cannot validate it.
- **A fusion-model refit is NOT required by the units bug** (correcting an
  earlier claim in this session): FOOTPASS tactical X/Y are normalised [0, 1]
  (verified: value range −0.06..1.03), so `fit_from` fitted the prior, the
  calibrators, and the weights through `displacement()` in the *correct*
  convention. The bug lived only in the serving path (`_tracklet_evidence`).
  The shipped weights are valid as fitted.
- **Occupancy/transition double-counting:** post-fix autopsies show them voting in
  opposite directions on the wrong merges; no evidence of the double-count yet.
  Re-examine on FOOTPASS.

## Caveats

Oracle detections (no detection dropout), P=1 features with visibility ≡ 1.0,
30-second broadcast clips, n=6 sequences, GT teams. Merge counts are small
integers — treat the A/B directions as robust (they are monotone across operating
points) and the exact figures as indicative. Nothing here is validated on a real
detector substrate (the standing Step-B gap).

## Addendum (2026-08-02): FOOTPASS-scale validation of the margin rule and shrinkage

LOSO over the three FOOTPASS matches (fit on two, evaluate both halves of the
third; `MAX_GAP_FRAMES=30`, pass 1 only, `min_score 4.0`), 12,595 fragments:

| arm | merges | precision | coverage | wrong |
|---|---|---|---|---|
| base (margin 0) | 8,746 | 0.9743 | 0.6849 | 225 |
| **margin 0.5** | 8,417 | **0.9815** | 0.6640 | 156 |
| **margin 1.0** | 7,886 | **0.9888** | 0.6268 | 88 |
| shrink n0=250 | 8,551 | 0.9704 | 0.6670 | 253 |
| shrink n0=750 | 8,502 | 0.9685 | 0.6618 | 268 |
| threshold 4.5 (margin 0) | 8,293 | 0.9796 | 0.6530 | 169 |
| threshold 5.0 | 7,797 | 0.9808 | 0.6147 | 150 |
| threshold 6.0 | 6,803 | 0.9832 | 0.5377 | 114 |

Two findings, both decisive:

1. **The winner-margin rule strictly dominates the absolute-threshold
   frontier.** Margin 0.5 beats threshold 4.5 on BOTH precision and coverage;
   margin 1.0 beats every threshold up to 6.0 on both axes (−61% wrong merges
   vs base for −5.8 pts coverage). Tightening the threshold discards
   uncontested-but-moderate evidence while keeping contested-but-strong pairs;
   the margin bar does the reverse, which is the correct direction — it is the
   candidate-relative denominator the decision was missing. The two-pass engine
   already plumbs `min_margin` (pass 1); the shipped default is 0.0.
   Recommendation: 0.5–1.0 for full-match footage. Not flipped here — default
   changes are Jeremy's call.

2. **Occupancy shrinkage is HARMFUL at FOOTPASS scale**: wrong merges rise
   225→253/268 while coverage falls. FOOTPASS query fragments are only ~200
   frames, so n0=250–750 mutes the channel throughout — and on full halves
   occupancy's negative evidence does real veto work that shrinkage silences.
   Combined with the SNMOT result (tuning +3 right, held-out flat), the honest
   status is: direction unproven, keep `occupancy_shrink_n0=0.0` (off). If the
   clip-regime miscalibration is worth fixing, the right form is a calibrator
   conditioned on n (refit), not a post-hoc scale on the LLR.

Margin in pass 2 remains unplumbed and unmeasured (the harness arms here are
pass-1 only).

## Addendum 2 (2026-08-02): pass-2 margin — implemented, but no dominance there

`pass2_min_margin` now plumbs the winner-margin rule into pass 2 (thread pair
must beat each side's best alternative partner; 0.0 = legacy greedy, the
default). Measured LOSO on FOOTPASS, on top of pass 1 at 4.0/margin 0.5:

| pass-2 arm | precision | coverage | wrong |
|---|---|---|---|
| score 4.0, margin 0 | 0.9763 | 0.7651 | 231 |
| score 4.0, margin 0.5 | 0.9812 | 0.6807 | 162 |
| score 4.0, margin 1.0 | 0.9812 | 0.6724 | 160 |
| score 7.0, margin 0 | 0.9801 | 0.7071 | 179 |
| score 8.0, margin 0 | 0.9809 | **0.6931** | 168 |

**The margin rule does NOT dominate in pass 2** — threshold 8.0 matches margin
0.5's precision at ~1.2 pts more coverage. Unlike pass 1 (one query, a small
natural candidate set, occasional genuine ties), agglomeration pairs are
pervasively contested at close scores, so the margin throttles coverage about
as indiscriminately as a threshold does. Margin dominance is a pass-1
phenomenon. Use `pass2_min_score` for the pass-2 trade; `pass2_min_margin`
stays available (on SNMOT through the full engine it removed 1 of 3 wrong
merges at zero cost) but earns no default.

Recommended operating recipe from this session's measurements, full-match
footage: pass 1 `min_score 4.0` + `merge_min_margin 0.5`, pass 2
`pass2_min_score` chosen by the precision/coverage trade you want (4.0 →
0.9763/0.765, 8.0 → 0.9809/0.693), `pass2_min_margin 0.0`.
