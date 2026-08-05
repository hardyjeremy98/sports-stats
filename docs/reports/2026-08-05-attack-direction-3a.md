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

**Boundary (V2).** Unequal-φ concatenation (H1 truncated to φ ∈ {0.25…0.85})
with **no half-time gap**. That matches the substrate: FOOTPASS's halves are
separated by only **9–79 s** (game_18 8.7 s, game_24 78.7 s, game_47 11.4 s),
and the videos are ~100–108 min for two ~50 min halves, so the footage is
trimmed at half-time. There is no break to find — the swap is a cut. (Which
makes the 15-minute-gap variant the *unrealistic* one, not the easy contrast I
first labelled it.)

**15/15 resolved, median error 3.0 s, max 22.0 s, 67% within 5 s, 100% within
30 s, and the reported confidence covers the error 15/15.** On the real
geometry (φ=1.0): **8.0 / 1.3 / 0.9 s**. Cost ~370 probes, 0.25% of frames.

> **This replaced fine-scale bisection, which was the cause of the earlier
> 12.0 s median and a 54.6 s worst case.** Bisection needs a reliable oracle
> per step, but the per-probe sign is 0.86–0.97 accurate with errors correlated
> over 5–10 s (V3b), so once the bracket shrinks below that the majority vote
> degenerates to a single ~90% coin — and bisection is greedy with no
> backtracking, so one wrong call is unrecoverable. The worst match for
> per-probe accuracy (game_18, 0.887–0.908) was the worst for boundary error,
> which is the signature.
>
> Now: bracket, then **scan densely and run the same change-point statistic
> across the window**, pooling every sample instead of making ~5 irreversible
> decisions. The window is sized by the *coarse* profile (candidates within 10%
> of the maximum, padded 2 steps) rather than a fixed ±1 step — a fixed window
> assumes the coarse argmax is within one step, and on game_18 φ=0.70 it landed
> **142 s away**, so the scan searched the wrong place entirely and confidently
> reported 99.8 s. With the adaptive window that case is 11.1 s.

> **Read n as 3, not 15.** The five φ values per game share H2 entirely and use
> nested prefixes of the same H1; the detected change point is the same
> physical sign flip. They are correlated perturbations of 3 matches, not 15
> observations. **The defensible statements are "3/3 resolved" and "max error
> 54.6 s"**; a median quoted to 0.1 s is overprecision.
>
> **The 758.6 s midpoint baseline is engineered by this protocol and must not
> be quoted as a margin.** The φ-sweep deliberately moves the boundary away
> from the centre, so `|n/2 − b|` is large *by construction*. On the real
> FOOTPASS geometry (φ=1.0) the constant-midpoint baseline errs by only
> **32 / 112 / 101 s**. The φ-sweep is a legitimate *robustness* protocol and
> it does exclude the centre-bias trap; it is not a source of a baseline ratio.
>
> V2 also passes `min_half_seconds=8*60` rather than the module default of
> 20 min, to admit φ=0.25 — so these numbers do not characterise the shipped
> default configuration.

**Absolute direction (V1).** 12/12 correct — but that is **3 independent bits,
one per game**: `attacks_positive_x(club1, e)` is by construction the negation
of `attacks_positive_x(club0, e)`, and epoch 1 is the sign flip of epoch 0.

**Cost (V3).** 143–229 frames probed out of ~150,000, **~0.12%**. Coarse
spacing 30 s; the per-probe error autocorrelation decays below 0.1 by 5–10 s,
so 30 s probes are effectively independent. 120 s spacing still resolves 3/3 at
75 probes; 60 s is the only ragged point (12.4 s median).

**Downstream (V5).** The mirror decision *is* `same_epoch`. Over all fragment
pairs (GT observable spans ≥ 50 frames), against the known-boundary oracle,
with a **constant-midpoint estimator run through the identical enumeration** as
the baseline:

| game | err | ci | pairs | correct | abstain | **wrong** | midpoint **wrong** |
|---|---|---|---|---|---|---|---|
| game_18 | 8.0 s | 46 s | 9,827,961 | 96.47% | 3.53% | **0.000%** | 0.000% |
| game_24 | 1.3 s | 15 s | 9,506,980 | 97.95% | 2.05% | **0.000%** | 3.770% |
| game_47 | 0.9 s | 15 s | 7,040,628 | 97.56% | 2.44% | **0.000%** | 3.122% |

(The dense scan tightened the band, so fewer fragments land in the undecidable
zone: abstentions fell from 3.71/2.51/3.49% to 3.53/2.05/2.44%.)

> **"26.4M pairs, zero wrong" is 3 bits of evidence, not 26.4M.** Cold review
> established this and it is exactly right. `epoch_of_fragment` returns 0 iff
> `end < b̂ − ci` and 1 iff `start > b̂ + ci`, so **once `|b̂ − b_true| ≤ ci`,
> zero wrong bits is mathematically guaranteed** — the enumeration cannot
> produce a single error. It multiplies one per-game fact by ~10⁷ and reports
> the product as sample size. The same applies to "100% of cross-half pairs
> correct" (cross-half is ~49.9% of pairs, so not a negligible subset — but
> still the same three bits re-counted).
>
> It is also partly **self-consistency**: `ci` is set by the estimator itself,
> and was *widened* in response to the blackout case below. The band was tuned
> until it covered the error; the pair metric then reports coverage as accuracy.
>
> V5 also runs at φ=1.0 — equal butted halves, the configuration the V2
> protocol exists to avoid. **The honest claim is: eliminates ~3 percentage
> points of wrong mirror bits versus guessing the midpoint, on 2 of 3 games;
> on game_18 the midpoint is already perfect and marginally better on yield.**
>
> Finally, the enumeration is all unordered pairs; `pair_table` keeps only
> same-club, non-overlapping, ordered pairs (~4× fewer). The rate should carry
> over, but 26.4M is not the candidate set the +3.4 was measured on.

**The abstentions are NOT simply "safe".** They are concentrated around the
boundary, so they are disproportionately cross-half — where raw occupancy is
**anti-informative (AUC 0.34–0.38), not neutral**. Falling back to
`mirror="off"` is safe as a whole-match default because raw occupancy roughly
cancels overall; on this subset it is the actively harmful regime. **The
correct consumer behaviour is to zero occupancy for undecidable pairs**, now
documented on `same_epoch`.

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
~50-minute void between the halves. For game_18 that places H2's first fragment
at 173,333 — a **65.4-minute inter-half void instead of 15**.

**This contaminates the +3.4 that justifies this whole work item**, and my
first write-up understated it as "contained to the gap channel". The fused LOMO
was *body + gap + transition*, so the gap channel is **inside** the fusion that
produced 0.855 / 0.854 / 0.888. Mitigations: the corruption is a constant
additive shift applied uniformly to every cross-half pair, so per-channel
cross-half AUCs are rank-invariant and unaffected, and all three arms share it,
so the *delta* almost certainly survives in sign. What does not survive cleanly
is the **magnitude** — a logistic gap weight fitted against a feature that
pushes every cross-half pair into an implausible-gap regime changes how much
headroom occupancy has to fill, and that headroom is the +3.4. (Pair ordering
is unaffected: a uniform positive shift preserves H1 < H2. Had `max_gap_s` been
set rather than left `None`, cross-half pairs would have been silently
annihilated.)

Not a blocker for 3a — the estimator never touches `load_match` — but **"+3.4"
should not be cited as a clean number until refitted.** Not fixed here; fixing
it moves published cross-half numbers and deserves its own change.

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
