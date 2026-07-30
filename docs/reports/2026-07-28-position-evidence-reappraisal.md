# Position evidence for re-ID: reappraisal after adversarial review

**Date:** 2026-07-28
**Data:** FOOTPASS val, 3 matches × 2 halves, 13,170 fragments, 13,016 answerable episodes
**Protocol:** 3-fold match rotation — calibrate / fit / evaluate are always three *different*
matches, so no identity crosses a fold
**Code:** `25a3d67` (calibrator), `9cb4576` (fusion weights + transition prior)
**Substrate:** GT observability spans — perfect detection and perfect within-fragment
tracking. Every number here is optimistic against a real tracker.

Four independent review agents audited the position/zone-occupancy work. This report
records what survived, what did not, and what replaced it.

---

## 1. The headline result was mostly an artefact. Retracted.

The prior report claimed position added **+1.35 rank-1** over body ID (a "23% relative error
reduction") with a conditional effect of **+206** episodes. Both are void.

`LLRCalibrator` had three compounding defects that destroyed the body channel's ordering:

1. Equal-count quantile bins cannot resolve a tail. Where one class runs out of samples,
   every bin holds identical counts and Laplace smoothing maps them all to one value.
2. Per-bin noise put small downward steps (−0.0005 nats) into an otherwise increasing
   curve, inverting **17.07%** of adjacent candidate pairs.
3. `LOG_CLAMP` hard-clipped whatever survived.

Result: **19.71%** of body-ID decisions had a top-2 tie. The decisive test — perturb the
body score by `1e-9`, far too small to reorder any genuinely distinct pair, so any change is
a tie being broken:

| arm | rank-1 | vs body |
|---|---|---|
| body (as shipped) | 94.23% | — |
| body + 1e-9 **uniform random noise** | 94.81% | **+0.58** |
| body + 1e-9 position (tie-break only) | 95.37% | **+1.14** |
| body + position (full) | 95.58% | +1.35 |
| **raw cosine, uncalibrated** | 95.37% | +1.14 |

**84% of position's reported gain was tie-breaking**, and simply not calibrating the body
channel recovered the same amount. The calibrator was destroying more evidence than every
position channel put together contributed.

## 2. The fixed calibrator

Logistic backbone (defined and strictly monotone everywhere, including past the data) +
histogram correction weighted per bin by total count + isotonic regression in the direction
the fitted slope reports, returned as block-centroid **knots** so the fit interpolates
rather than steps + `tanh` saturation instead of a clip.

| | before | after |
|---|---|---|
| body top-2 tie fraction | 19.71% | **0.00%** |
| body rank-1 | 94.23% | **95.37%** |
| gain from 1e-9 noise | +0.58 | **+0.00** |
| **position's contribution** | +1.35 | **+0.28** |

Body now scores exactly what the raw cosine does, which is what an order-preserving
transform must do. Position's honest contribution over a non-degenerate body channel is
**+0.28**, roughly a fifth of the retracted figure.

## 3. What the combiner is worth — the main positive finding

A unit sum of LLRs assumes conditional independence and a shared scale. Neither holds:
occupancy and continuity correlate at 0.31 on impostors, and `gap`'s calibrated LLR spans a
twentieth of body's range. One weight per channel, fitted as a conditional logit on a
held-out match:

| arm | rank-1 | vs body |
|---|---|---|
| body | 95.37% | — |
| body + occupancy | 95.54% | +0.17 |
| body + position (**unit sum**, incl. transition) | 94.92% | **−0.45** |
| body + position (**fitted weights**) | **96.26%** | **+0.89** |

**Adding a channel to the unit sum made it worse.** The fit recovers **+1.34** over the sum.
Learned weights across folds:

| channel | weight |
|---|---|
| body | 2.45 – 2.95 |
| gap | 4.73 – 11.73 |
| occupancy | 0.75 – 1.36 |
| continuity | 0.49 – 0.93 |
| transition | 0.07 – 0.16 |

`gap` needs a large weight purely to undo its scale; `transition` is crushed to ~0.1 because
it is largely redundant with gap+continuity. The fit discovers both. The sum cannot, which
is exactly why the sum was damaged.

### Full ablation, all three combiners

`docs/reports/2026-07-28-multi-input-fixed.json`

| inputs | sum | **fitted** | set scorer |
|---|---|---|---|
| body | 95.37% | — | 95.05% |
| occupancy | 41.02% | | |
| continuity | 29.10% | | |
| gap | 14.12% | | |
| **transition** ⚠️ | **51.26%** | | |
| body + occupancy | 95.54% | 96.16% | |
| body + transition | 95.57% | 95.59% | |
| **body + occupancy + transition** | 95.64% | **96.27%** | |
| body + occupancy + continuity | 95.47% | 96.21% | |
| body + occupancy + continuity + gap | 95.65% | 96.20% | 95.83% |
| body + occ + cont + gap + transition | 94.92% | 96.26% | 95.85% |
| occupancy + continuity + gap | 55.71% | 59.80% | |
| occupancy + transition | 60.25% | 60.59% | |
| occupancy + continuity + gap + transition | 61.21% | 61.74% | **63.96%** |

Three things to read off it:

- **The best configuration is `body + occupancy + transition` with fitted weights, 96.27%.**
  Continuity and gap add nothing once the transition prior exists — it conditions the same
  physics on the gap instead of treating displacement and elapsed time as separate channels.
- **The transition prior is the strongest single position channel** (51.26% vs occupancy's
  41.02%), and removing its Δt conditioning costs −24.7 pp, so the conditioning *is* the
  signal. **⚠️ SUPERSEDED (2026-07-28): every `transition` figure on this page was measured
  on an UNBOUNDED channel; see §10 for the corrected result.** `_saturate` (tanh, ±6 nats) is applied inside
  `LLRCalibrator.llr`, so all four calibrated channels are bounded, but the transition prior
  is appended raw (`multi_input.py:216`, `bootstrap_threads.py:391`) and reaches −3754 nats
  against body's −3.87, with 17.0% of rows beyond −6 and its sd inflated 17.9×. Because
  tanh is a monotone NONLINEAR compression, bounding changes the channel's shape and not
  merely its units, so these numbers do not transfer. Re-measurement in progress.
- **The set scorer wins only where no channel dominates** — position-only, 63.96% vs 61.74%.
  With body present the 4-parameter conditional logit beats it at a fraction of the cost.
  Keep it as a research arm; do not make it the default.

The conditional effect printed by that run (351 fixes / 410 breaks, net −59) is computed
under the **unit sum** and should not be read as position's effect; it needs re-deriving
under fitted weights.

## 4. Accumulation is a bigger lever than the position representation

Today a candidate is one fragment: median **8.2 s** (12.2 s mean) touching, on game_18_H1, a
median of **3** of 96 pitch cells (p10 = 1) against the **22** a whole player's territory
covers. Representing a candidate by everything
seen of it so far instead (oracle-threaded, no future information — so a ceiling):

| arm | recent fragment | longest fragment | **accumulated** |
|---|---|---|---|
| body | 81.61% | — | **94.41%** (+12.80) |
| occupancy | 44.13% | 45.0% | **51.57%** (+7.44) |

The `longest` control matters: it gives the candidate its single best-observed fragment, so
if the gain were merely "more frames" it would capture most of it. It does not.

**The larger share goes to appearance, not position.** A thread's mean embedding is a far
better prototype than any single fragment's — which was not the hypothesis under test.

Independently corroborated by the formation agent from a different direction: closed-set
player ID from position alone, accumulated territory **50.67%** vs nearest single fragment
**41.18%** (chance 7.94%).

## 5. Dead ends, measured and closed

**Joint one-to-one assignment (Bialkowski / EFPI lineage).** The intuition — centre backs
merge cleanly, leave the pool, and the remaining left/right back disambiguation becomes
easy — is correct as physics and irrelevant here. The pool is never cut to 4; it is ~550 by
construction. Enumerating actual argmax collisions between co-ending queries gives a hard
ceiling of **47 / 13,016 = +0.36 pp**; measured achievement **+0.09 pp**. Global (whole-half)
exclusivity is **−6.7 pp**. Not built.

**Role discovery as an input.** GT `ROLE_ID` is a player alias on this substrate (110 of 132
team-role slots hold exactly one player), so it stays analysis-only — already known. What is
new: unsupervised role discovery agrees with GT at 0.418 while the trivial per-player mean
position reaches 0.900. The per-player accumulated model dominates the role taxonomy; a role
layer is a detour. Not built.

**`impostor_field_llr`.** Proven rank-neutral as a channel: margin-over-runner-up is an
affine, strictly increasing transform *within* an episode, so it cannot reorder anything
(occupancy 40.08% → 40.08%, identical to the episode). It was also unit-inconsistent,
clamping a z-score and summing it with nats. Superseded by fitted weights.

## 6. Substrate limits that bound every number here

- **Role ≡ identity on FOOTPASS.** Occupancy's AUC of 0.782 is computed against a negative
  set that is 99.2% different-role. Restricted to same-role impostors it collapses to
  **0.630**. For the amateur-football target, where roles are fluid, the bijection does not
  hold and this substrate cannot distinguish "position identifies the player" from "position
  identifies the role slot, which here *is* the player".
- **Fragments never contain an ID switch.** Position noise turns out to be irrelevant (σ = 5 m
  moves occupancy AUC by +0.004, below the 8.7 m grid resolution) and truncation to 2 s costs
  little, but neither degradation tests a fragment that fuses two players — which is what a
  real tracker produces and what would corrupt a footprint far more.
- **A "fragment" here is not a tracker tracklet.** It is a GT observability span, split
  whenever the player is off-camera for more than 2 frames; a tracker instead bridges short
  occlusions with its motion buffer and can carry an ID switch inside one tracklet. Measured
  under a matched >=2 s filter, real tracklets on SNMOT have a median duration of **10.0 s**
  against these fragments' **8.2 s**. So this substrate is MORE fragmented than reality --
  more joins required, less evidence in each -- which makes the accumulation result
  conservative, while every other difference (no ID switches, perfect detection, GT team
  gate) runs the other way.
- **The candidate field is oracle-purified** by a GT team gate.
- **Analyst-level selection on the eval set.** Formation-relative coordinates, `max_bins=200`
  and the interpolation fix were each chosen by their effect on these same val halves. The
  spec's C/T/D/E discipline was not followed; there is no dev block.

## 7. Where this leaves position as an input

Position is real evidence and it is not free of value: **+0.89** with a proper combiner,
against **+1.35** claimed and **+0.13** under a naive sum with the fixed calibrator. But the
two larger levers found in this review are not about position at all — they are the
**combiner** (+1.34 over the sum) and **accumulation** (+12.8 on appearance). The honest
summary is that position was over-credited by a broken calibrator, and that fixing the
machinery around the evidence mattered more than adding evidence.

## 8. End-to-end: what the system actually does

> **⚠️ SUPERSEDED 2026-07-30.** The numbers in this section were measured on fragments cut at
> an 80 ms absence, which is far finer than a tracker's output and mixes in a large population
> of trivially easy re-links. Rebuilt with a 1.2 s buffer so the units are tracker-shaped,
> precision falls **96.63% → 88.31%** at matched coverage and wrong merges rise 3.7×. See
> [`2026-07-30-tracker-shaped-tracklets.md`](2026-07-30-tracker-shaped-tracklets.md). The
> accumulation finding survives and widens; the transition prior's fitted weight rises 6×.

`bootstrap_threads.py` runs the merging without any oracle — fragments in time order, each
joining a thread or starting one, so later decisions are judged against evidence the system
built for itself. Scored as correct/wrong merge decisions rather than rank-1, because a
ranking metric cannot see a thread that quietly fused two players 40 minutes earlier.

All 6 halves, 13,016 merges required:

| operating point | merges made | correct | wrong | precision | coverage |
|---|---|---|---|---|---|
| conservative | 9,202 | 8,986 | 216 | **97.7%** | 69.0% |
| aggressive | 11,603 | 10,980 | 623 | 94.6% | 84.4% |

Against the single-fragment control, accumulation wins on **both** axes in 8 of 8 half ×
threshold comparisons — more correct merges *and* fewer wrong ones. There is no
precision/recall trade being made.

Identity is not 69% solved in any useful sense: at that operating point each player is still
split across 20–30 unlinked threads. What the system delivers reliably is "this is
consistently one person", not "this is *the* person from earlier in the match".

### A second, thread-to-thread pass

Pass 1 always compares a thread against a **lone fragment**, so half the evidence in every
comparison is an 8-second smudge. Pass 2 agglomerates thread-to-thread — greedy, re-scored
each round because every merge strengthens the surviving thread — so both sides are
accumulated. Pass-1 threshold fixed at 4.0, pass-2 threshold swept:

| arm | correct | wrong | precision | coverage |
|---|---|---|---|---|
| single-fragment control | 7,477 | 508 | 93.64% | 57.4% |
| pass 1 only | 8,986 | 216 | 97.65% | 69.0% |
| + pass 2 @ 4 | 9,132 | 222 | 97.63% | 70.2% |
| + pass 2 @ 2 | 9,635 | 264 | 97.33% | 74.0% |
| **+ pass 2 @ 0** | **10,293** | 359 | **96.63%** | **79.1%** |
| + pass 2 @ −2 | 10,885 | 533 | 95.33% | 83.6% |

**The two-pass system dominates the one-pass frontier.** One pass at its aggressive setting
gave 94.63% precision at 84.4% coverage; pass 2 @ −2 gives 95.33% at 83.6% — better
precision at the same coverage. Lowering pass 1's threshold is a strictly worse way to buy
coverage than adding pass 2.

Pass 2 is also far more precise than pass 1 at the same threshold: at 4.0 it adds 146 correct
merges for 6 wrong across all six halves. That is the accumulated-evidence effect appearing
on *both* sides of the comparison rather than one, and it is why the pass-2 threshold can be
set well below pass 1's.

Thread purity falls sharply at −2 (92.6% → 80.9% on game_24), which is the honest signal that
−2 is past the useful point; **pass 2 @ 0** is the operating point to prefer.

## 9. Fragment purity — the assumption that turned out to be backwards

Every number above rests on fragments that never contain two players. The stated worry was
that a contaminated fragment would permanently poison a thread's accumulated territory and
appearance, and that accumulation would therefore be *fragile* to real tracker output.

Measured by splicing a same-team player over the tail of a fraction of the evaluated
fragments, keeping the majority player's label, and remapping the exit point to the donor
(game_18, conservative operating point):

| contamination | accumulated precision | coverage | single-fragment precision | coverage |
|---|---|---|---|---|
| 0% | 98.66% | 67.1% | 95.10% | 55.3% |
| 5% | 98.17% | 65.7% | 92.99% | 53.2% |
| 15% | 96.59% | 63.6% | 87.79% | 48.9% |
| 30% | 92.57% | 60.1% | 78.17% | 42.7% |

**The worry was wrong, and in the useful direction.** Accumulation degrades roughly 3× more
slowly than the single-fragment representation (−6.1 vs −16.9 points of precision at 30%
contamination). Pooling over many fragments dilutes a corrupted one; representing a
candidate by a single fragment means a corrupted fragment *is* the candidate. Accumulation
is not merely better on clean data, it is the more robust representation under exactly the
failure mode that was expected to break it.

Still untested: real detector output (missed and spurious boxes), a learned team classifier
rather than the GT gate, and phone footage. Contamination here is simulated from GT
positions, not observed from a tracker.

Open: re-measure the merge frontier on the fixed calibrator (the −598 regression attributed
to position was measured on a different pair population and is unattributable).

## 10. The transition channel was mis-scaled; bounding it is a free win

Two agents argued opposite sides of "drop `gap` and let the transition prior carry the
physics". Both abandoned their assigned positions, independently, and converged.

**The channel was never on the same scale as the others.** `saturate` (tanh, ±6 nats) runs
inside `LLRCalibrator.llr`, so the four calibrated channels are bounded; the transition prior
was appended raw and bypassed it, reaching **−3754 nats** against body's −3.87, with 17.0% of
rows beyond −6 and its sd inflated **17.9×**. `fit_fusion_weights` standardises by sd, so its
reported weight (0.0121) was in raw units and NOT comparable to the others'. It did not mean
the channel was inert.

Bounding it identically, verified end to end on all 6 halves:

| transition | correct | wrong | precision | coverage |
|---|---|---|---|---|
| unbounded (shipped) | 10,293 | 359 | 96.63% | 79.08% |
| **tanh ±6** | **10,392** | **358** | **96.67%** | **79.84%** |
| tanh ±25 | 10,308 | 380 | 96.44% | 79.19% |
| channel deleted | 10,299 | 353 | 96.69% | 79.13% |

**+99 correct merges for −1 wrong.** Note the fourth row: unbounded, the channel was
statistically indistinguishable from **deleting it**. Its fitted weight rises 21× when bounded
(0.0121 → 0.2552), into the same range as its neighbours.

**`gap` stays, and is more load-bearing after the fix, not less.** At ±6, dropping `gap` costs
**+204 wrong merges** (358 → 562). At matched coverage the un-bounded comparison is +147 wrong
(+41%), worse in 6/6 halves (sign test p=0.031). On the impostor population `gap` and
`transition` correlate **−0.17** (Spearman −0.31 to −0.43): complementary, not redundant.
Transition fires precisely where `gap` is *wrong* — on short gaps, where `gap` says +0.36 nats.

**The margin that started this is noise.** 96.27 vs 96.20: paired over 6 halves, mean +0.067
against a per-half sd of 0.536, exact permutation **p = 0.78**, and it loses on 3 of 6. The
whole top cluster spans 0.11 points against a between-half sd of 1.3–1.5. The ranking ablation
cannot license removing a channel.

**The channel's advertised rationale is backwards.** `transition.py` claimed the value was
that it "can rule an identity out; it can never assert one". Leave-one-out flip counts at ±6
over 13,170 decisions: **enabling** merges 218 right / 8 wrong; the impossibility **veto** 1
correct block against **31** blocks of correct merges. The half that was advertised is the half
that costs — and saturation preferentially truncates it (the veto reaches −3754, the enabling
side caps near +3.6 by construction), which is *why* bounding helps.

**Deployment.** Missing calibration is free: at the measured 4.4% coverage the prior abstains
to 0 and the system collapses to the identical no-transition configuration (357 wrong, both
arms). **Wrong calibration is the risk** — at 5 m endpoint error the bounded arm reaches 436
wrong, worse than not having the channel at all (353). The status quo was immune only because
it was inert. So the channel should be gated on calibration **confidence**, not coverage: a
low-confidence solve must abstain rather than supply a plausible-but-wrong endpoint. Untested;
this is the open item.

## 11. Reachability as a hard constraint: rejected, and the shipped gate is a latent hazard

Two agents argued gate-vs-score. **Both abandoned their assigned positions**, independently,
and converged on: keep it a score.

**The gate loses on ORACLE coordinates, before calibration error enters.** All 6 halves,
thr 4.0, pass2@0, against the 10,392 correct / 358 wrong baseline:

| envelope | correct | wrong |
|---|---|---|
| none (score only) | 10,392 | 358 |
| 8.0 m/s | 10,368 | 382 |
| 9.5 m/s | 10,385 | 368 |
| 12.0 m/s | 10,385 | 370 |
| 15.0 m/s | 10,381 | 374 |

Every envelope is worse on both axes, and the effect is non-monotone in the threshold — the
signature of trajectory churn rather than a designed effect. Only **20** decisions across six
halves had their top-1 vetoed, yet one half swings 46 merges: **a gate's system-level effect is
not the sum of its vetoes, it is dominated by trajectory divergence.** At this sample size the
oracle-coordinate effect is honestly *unmeasurable*, not *free*.

**A veto is not an abstention.** This kills the risk-asymmetry argument that motivated the
proposal. The system takes an argmax over *surviving* candidates, so deleting the right answer
promotes the runner-up — an impostor. Instrumented over all 6 halves at 5 m endpoint noise: 229
top-1 vetoes, 217 of them false (95%), of which 107 abstained, 87 recovered a correct merge
elsewhere, and **23 merged the wrong player** — a wrong merge the ungated system did not make.
On oracle coordinates the ratio is 1 in 6. The product's preference for abstention is therefore
an argument for **scores that can be outvoted**, not for constraints that silently re-route the
decision.

Corollary worth keeping: the brief's claim that a veto is "unrecoverable by construction" is
also false — ~40% of false vetoes recover a correct merge from another candidate.

**A hard gate is bit-identical to a large finite score.** Independently measured by both agents:
the 9.5 m/s gate produces merge-for-merge identical output to a −10 nat penalty (agent A) and a
−50 nat penalty (agent B), at both noise levels. A hard constraint is a score with an unbounded
coefficient. It adds no capability — only the removal of the dial that lets other evidence
overrule an estimated position.

**Physics cannot see the errors we actually make.** **92.7% of the 358 wrong merges imply under
6 m/s.** And 99.7% of gate firings are inert, hitting candidates the scorer already rejects.
The ceiling was always tiny: in the pool the system actually faces, the impossible region holds
0.6% of impostors.

**It is a short-gap rule in a physics costume.** 97.3% of firings are at gaps under 5 s, 26.7%
under 1 s — exactly where positional error dominates, since speed = |d|/dt amplifies endpoint
error by 1/dt. At a 0.5 s gap, a genuine 6 m/s run needs only **0.88 m of error per endpoint**
to fabricate a 9.5 m/s impossibility. The repo's own measured camera-motion recovery error is
**0.5–6.1 m**. Break-even is ~1.5–2 m and we are not inside it. Below a quarter-second gap the
gate deletes the *majority* of true continuations even on oracle coordinates.

### The shipped pipeline already has this gate

`matchlab_core/reid/gates.py::MotionFeasibilityGate`, wired in `configs/pipeline.tdlp-full-reid.yaml`:
`max_speed_cm_s: 900.0` (9.0 m/s), `max_speed_px_s: 800.0`, `soft_gap_s: 15.0`,
`calibration_min_confidence: 0.5`. **Position A was not a proposal; it is production.**

It has never been exercised in metric mode: the pitch branch needs calibration on *both*
endpoint frames, and PnLCalib clears confidence ≥0.5 on 4.4% of SNMOT frames, so SPO-59,
SPO-73, SPO-85 and the smoke runs all scored on the pixel bound alone. Its authors hedged two of
the three regimes that matter — long gaps (soft beyond 15 s) and missing calibration (coverage
required). **The unhedged one is calibration present but WRONG**, which is precisely where the
gate's own precision falls from 70% to 5%. So improving calibration coverage will *activate* a
veto measured to be net harmful. That is a latent regression, not a latent improvement.

### Two larger levers this surfaced

**The team gate is the biggest unbudgeted risk in the stack.** Only *"times don't overlap"* is
genuinely certain — team comes from a classifier in deployment, and SPO-73 measured a
kit-colour gate falsely vetoing 19% of true re-entry pairs. Simulated here by flipping fragment
team labels (game_18): 5% error costs **−136** correct merges, 10% costs **−201**. That is
**20–30× the entire reachability-gate effect in either direction**. The general principle:
**a hard constraint inherits the error rate of whatever estimates it.** Non-overlap is estimated
by frame indices (exact); team by a classifier (19% measured pair false-veto); reachability by
calibration (0.5–6 m). Reachability is the worst-estimated of the three.

**A one-sided prior beats both the gate and the status quo.** Letting the transition channel
enable merges but never penalise them, with weights refitted:

| transition channel | correct | wrong | precision | coverage |
|---|---|---|---|---|
| two-sided, tanh ±6 (shipped) | 10,392 | 358 | 96.67% | 79.84% |
| one-sided (enable-only) | **10,535** | 373 | 96.58% | **80.94%** |

**+143 correct for +15 wrong** — the marginal merges are 90.5% correct. Not a Pareto win, so it
is a product decision, but it is the same trade the permissive defaults already made
deliberately. It confirms §10's flip-count finding from the opposite direction: reachability's
negative side should be **weakened**, not hardened. Caveat: the flip counts predicted +31/−1 and
the truth was +143/+15 — they over-predicted the gain 2× and missed the wrong-merge cost
entirely, so flip counts are not counterfactuals.
