# 3a — attacking direction and half boundary

2026-08-05. Code: `matchlab_core/formation/direction.py`, tests
`test_formation_direction.py` (19), harness
`matchlab_train.experiments.direction_eval`. Substrate: FOOTPASS VAL, 3 val
matches × 2 halves. Design + review trail:
`docs/superpowers/specs/2026-08-05-attack-direction-design.md`.

## What was blocked and is now unblocked

Occupancy compares formation-relative footprints; attack direction reverses at
half-time, rotating a player's footprint 180°. Measured 2026-08-02: cross-half
raw same-player AUC 0.34–0.38 vs 0.74–0.80 rotated; fused LOMO 0.855 without
occupancy, 0.854 served raw, **0.888 with an explicit per-half flip**. The flip
was adopted as direction but deferred for want of a boundary estimate. This is
that estimate.

## Prior art: there is no published bar

- kloppy, floodlight, SoccerCPD and EFPI all take direction and periods from
  **provider metadata**; none estimates them.
- The one public CV implementation, the SoccerNet GSR baseline (sn-gamestate),
  compares *"the average 2D positions of each team on the pitch"* — the same
  centroid-ordering cue used here — but publishes **no isolated accuracy**, and
  its 30-second sequences never contain a flip.
- Boundary comparison bar: SoccerNet game-clock OCR, **90% of half-starts
  within ±2 s** — a cue needing a broadcast clock overlay, absent on the target
  amateur footage.

## Results

**Cue (V1b).** `d = mean_x(club A) − mean_x(club B)`, absolute pitch-normalised
x, observable players only. Per-probe sign accuracy **0.887–0.968** per half;
conditioned on play third, **worst third 0.855–0.940**. The pan bias is
*selection truncation*, not an additive offset, and it does bite — but modestly.

**Boundary (V2), the headline.** Unequal-φ concatenation (H1 truncated to
φ ∈ {0.25…0.85}) with **no half-time gap**, so the true boundary is not the
midpoint and the timeline has no empty region to find:

| | value |
|---|---|
| resolved | **15/15** |
| median abs error | **12.0 s** |
| max abs error | 54.6 s |
| within ±30 s | 87% |
| **constant-midpoint baseline, median** | **758.6 s** |

The centre-bias trap — `score(t)` peaks at n/2 under the null and `min_half`
narrows toward the centre, so equal butted halves would let a midpoint guess
score perfectly — is excluded by ~75×.

**Absolute direction (V1).** **12/12** correct (club × epoch, n=6 halves).

**Cost (V3).** 143–229 frames probed out of ~150,000, **~0.12%**. Coarse
spacing 30 s; the per-probe error autocorrelation decays below 0.1 by 5–10 s,
so 30 s probes are effectively independent. 120 s spacing still resolves 3/3 at
75 probes; 60 s is the only ragged point (12.4 s median).

**Downstream (V5) — the number that matters.** The mirror decision *is*
`same_epoch`, so agreement with the known-boundary oracle decides whether the
+3.4 transfers. Over all fragment pairs (GT observable spans ≥ 50 frames):

| game | pairs | correct | abstained | **wrong** | cross-half correct |
|---|---|---|---|---|---|
| game_18 | 9,827,961 | 96.29% | 3.71% | **0.000%** | 100.00% |
| game_24 | 9,506,980 | 97.49% | 2.51% | **0.000%** | 100.00% |
| game_47 | 7,040,628 | 96.51% | 3.49% | **0.000%** | 100.00% |

**Zero wrong mirror bits in 26.4M pairs**; every cross-half pair correct. The
2.5–3.7% abstentions are fragments straddling the boundary band, which fall
back to `mirror="off"` — today's behaviour, safe but collecting nothing.

## Robustness (V4) — and one uncovered failure

- Independent symmetric label flips: resolves to p=0.20, abstains at p=0.35.
  This is the *easy* model (it shrinks |d| but leaves the sign unbiased).
- **Bursty/asymmetric** (one club absorbed into the other over a contiguous
  window straddling half-time): fine to a 10-minute window (≤21 s error). At
  **20 minutes it produced a 361 s error whose permutation z (9.2), sign
  agreement (0.96/0.89) and separation (0.078) were all indistinguishable from
  a clean match** — no confidence statistic could flag it.

  Diagnosis: those probes correctly *abstain* (one club has no observable
  players), leaving a 20-minute blackout centred on the boundary. The error is
  genuine ambiguity that the reported CI was not expressing. Fixed by widening
  `ci_frames` to the probe gap bracketing the boundary; the error is now inside
  the reported band, so `epoch_of_fragment` returns `None` there instead of a
  confident wrong epoch. Regression-tested.

## Abstention

Permutation z (scale-free; raw `separation` is in pitch fractions and is kept
as a diagnostic only), opposite-sign check, per-segment sign agreement, and a
sub-boundary guard for the two-epoch scope limit — calibrated, not guessed:
across 24 clean two-epoch synthetics the largest sub-z was 1.53, against 4.1–4.7
for three-flip input; threshold 3.0.

**Scope limit:** two epochs. Extra time (4 epochs), warm-up footage, and
reverse-angle replays all violate it; the guard abstains rather than segmenting.
FOOTPASS contains none of these, so the substrate cannot show that failure.

## Substrate bug found

**FOOTPASS H2 frame indices continue the match timeline; they do not restart at
zero** (game_18_H1 ends 75307, H2 starts 75525).
`footpass_match_harness.load_match` offsets H2 by `h1_span + HALF_BREAK_FRAMES`
on top of already-global indices, double-counting H1's span and opening a
~50-minute void between the halves. Only the gap channel sees the half-time
offset, so the blast radius is contained to that channel's cross-half
distances — but they are wrong by ~50 minutes, not by the intended 15.
Not fixed here (it would move the published cross-half numbers); flagged.

Also promoted to an assertion: `club_of_side` resolves the flip by
`mean(votes_flip) >= 0.5`, so an exact tie declares a flip. `direction_eval`
now verifies directly that each resolved club's goalkeeper really does change
ends between halves.

## Not done / next

- **`occupancy_mirror` default is unchanged.** Flipping it needs the fitted
  `FusionModel` re-scored end to end and the best2 replay gate. V5 says the
  bits are right; it does not re-measure the fused frontier.
- Never seen real tracker output — GT observable spans and synthetic
  corruption only. The transfer test stands open, as it did for the centroid
  module.
- The absolute-direction output (12/12) is unvalidated *as evidence* for 3b;
  role templates must establish their own.
