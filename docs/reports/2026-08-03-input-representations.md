# Richer input representations at the re-ID merge frontier (2026-08-03)

Branch `worktree-reid-input-representations`. Spec:
[`docs/superpowers/specs/2026-08-03-richer-input-representations.md`](../superpowers/specs/2026-08-03-richer-input-representations.md).
Follow-up to [`2026-08-02-fusion-usage-audit.md`](2026-08-02-fusion-usage-audit.md) Phase C.

Revision 2, after a cold review of the results found one real bug (experiment
2's early stopping never restored the best model), one bad comparison
(experiment 1's AUC column), one unshipped claim, and several undisclosed
protocol deviations. All are corrected or disclosed below; the numbers here are
post-fix.

## The claim under test

Every "re-ID here is evidence-limited" negative on this stack was measured on
**hand-reduced scalar inputs**: a pooled-prototype cosine, a JS distance between
blurred footprints, gap seconds, and a linear-velocity residual, each squashed
through a 1-D calibrator and summed with one scalar weight. Learned trackers
(SUSHI, TDLP, CAMELTrack, GHOST) consume raw per-detection or sequence-valued
cues instead. So the negative was **representation-scoped**.

Five experiments each remove one part of the reduction.

## Substrate (measured, not estimated)

`MAX_GAP_FRAMES=30`, `COORDS="rel"`, `min_frames=50`, oracle pitch coords and
teams, PRTreID embeddings, LOSO over game_18/24/47.

| | fragments | identities | rows | episodes | positives | field median |
|---|---|---|---|---|---|---|
| **total (6 halves)** | 12,595 | **154 player-halves** | 88,685 | 12,479 | 12,441 | 5–8 |

Per fold: fit ≈ 8.3k episodes / 59k rows, test ≈ 4.2k episodes / 29k rows.
Fragment `n_frames` median 178–300 (p10 70, p90 468–979) — a ~13× dynamic range.
**154 independent identities in one broadcast style is the binding constraint on
everything below**, not the row count.

## Resolution: what this substrate can and cannot detect

**The pre-registered MDE gate FAILED.** `measure_mde` reports
`can_resolve_v3_sized_effect: false`: the unpaired 95% precision interval is
0.011–0.016 wide (full width; the spec asked for half-width, so 0.0055–0.0078
half-width), against a v3-sized effect of 0.0024. The spec said that statement
goes in the report either way, so here it is.

The operative statistic is the **paired** delta, where both arms are re-scored
on the same resampled clusters and the substrate variance cancels. That
interval is much narrower — but it is **not one number**, and the first draft's
single "~0.005 threshold" was wrong. Measured per-arm 95% widths at median
coverage, player-within-half clusters (n ≈ 52 per fold):

| arm class | paired 95% CI width |
|---|---|
| pooling arms (exp 4) | **0.0008 – 0.0030** |
| appearance-set arms (exp 0) | 0.0027 – 0.0366 |
| MLP arms (exp 1) | 0.0027 – 0.0066 |
| trajectory arms (exp 2) | 0.0020 – 0.0047 |
| cohort arms (exp 3) | 0.0036 – 0.0110 |

So exp 4 is resolved to ±0.001 while `raw_softmax` and `mutual_nn` are resolved
only to ±0.02. **Every arm below is judged against its own interval**, not a
shared threshold.

Deltas are reported over a **coverage band** (8 points, 0.40–0.90), never a
single matched point — see the next section for why that matters.

## The instrument, validated

**It reproduces the audit's published figure.** Shipped as
`--stage reproduce` → `…-reproduce-v3.json`. Pooled LOSO pass-1 threading
frontier (margin 0.5) at `TRANS_NEG_CLAMP=6`, the clamp in force when the audit
was written:

| coverage | 0.45 | 0.50 | 0.55 | 0.60 | **0.65** | 0.70 | 0.75 | 0.80 |
|---|---|---|---|---|---|---|---|---|
| flat | .9890 | .9884 | .9875 | .9860 | **.9825** | .9780 | .9752 | .9702 |
| v3 | .9877 | .9871 | .9870 | .9862 | **.9848** | .9785 | .9729 | .9690 |
| Δ | −.0013 | −.0014 | −.0005 | +.0002 | **+.0023** | +.0004 | −.0023 | −.0012 |

At coverage 0.65: **0.9825 → 0.9848**, against the audit's published "0.9825 vs
0.9849". A harness that can re-derive a reference result it did not produce is
worth grading new arms with.

**What this says about the v3 claim — corrected.** The audit described v3 as
above the flat frontier "through the operating band". It is above at two of
eight points and below at five. But the honest verdict is **not** "it never
won": every one of those eight deltas, positive and negative alike, is smaller
than v3's own paired 95% CI width (0.0035–0.0046). The correct statement is
**the v3 effect is not resolvable at this substrate's resolution, in either
direction**, and quoting a single matched-coverage point overstated what was
measured. (v3 was not adopted regardless — it failed the end-to-end gate — so no
decision changes.)

That is the methodological finding this round contributes, and it is why every
arm below is reported band-wide with its own interval.

## Scoreboard

Pooled = mean of the three fold band-means (a descriptive summary; the spec's
"pooled with a fold effect" was not implemented — see Limitations). Verdicts use
each arm's own CI width. **No arm proceeded to the end-to-end best2 gate**,
because the protocol only sends screening-metric winners there and there were
none.

| # | arm | pooled band Δ | per fold | own CI width | verdict |
|---|---|---|---|---|---|
| — | incumbent (control) | 0 | — | — | — |
| **0** | mutual-NN | **−0.1170** | −.183 / −.068 / −.100 | .022–.037 | **negative-and-resolvable** |
| 0 | median | **−0.0272** | −.020 / −.031 / −.031 | .011–.019 | **negative-and-resolvable** |
| 0 | top-1 | **−0.0218** | −.026 / −.023 / −.017 | .009–.015 | **negative-and-resolvable** |
| 0 | q90 | **−0.0140** | −.009 / −.021 / −.012 | .009–.017 | **negative-and-resolvable** |
| 0 | top-3 | **−0.0114** | −.009 / −.017 / −.009 | .008–.010 | **negative-and-resolvable** |
| 0 | top-5 | **−0.0096** | −.006 / −.015 / −.008 | .007–.011 | **negative-and-resolvable** |
| 0 | *proto (plumbing control)* | −0.0010 | −.002 / +.000 / −.002 | .003–.005 | control passes |
| **1** | MLP raw, BCE | −0.0006 | −.005 / +.003 / +.000 | .003–.006 | negative-but-underpowered |
| 1 | MLP on-LLR, BCE | −0.0005 | −.004 / +.003 / −.001 | .003–.007 | negative-but-underpowered |
| 1 | MLP on-LLR, softmax | −0.0006 | −.002 / +.003 / −.002 | .003–.004 | negative-but-underpowered |
| 1 | MLP raw, softmax | **−0.0909** | −.120 / −.068 / −.084 | .024–.041 | uncalibrated-score artefact |
| 1 | *zero-hidden (machinery)* | −0.0088 | −.018 / −.001 / −.008 | .005–.011 | **control FAILS** |
| 1 | *context-shuffled (null)* | −0.0030 | −.007 / +.001 / −.003 | .004–.005 | null ≈ real arm |
| **2** | GRU NCE, banded, clamp 6 | **+0.0016** | +.001 / +.003 / +.001 | .002–.003 | negative-but-underpowered |
| 2 | GRU NCE, static, clamp 6 | +0.0016 | +.001 / +.003 / +.001 | .002–.004 | negative-but-underpowered |
| 2 | GRU MLE, banded, clamp 6 | +0.0014 | +.001 / +.003 / +.001 | .003–.004 | negative-but-underpowered |
| 2 | GRU NCE, banded, clamp 0 | +0.0013 | +.001 / +.003 / +.001 | .002–.005 | negative-but-underpowered |
| 2 | GRU MLE, banded, clamp 0 | +0.0012 | +.001 / +.003 / +.000 | .002–.004 | negative-but-underpowered |
| **3** | cohort z-score | **−0.0101** | −.023 / −.002 / −.006 | .006–.011 | negative-and-resolvable |
| 3 | cohort rank | −0.0043 | −.008 / +.000 / −.005 | .006–.008 | negative-but-underpowered |
| 3 | cohort margin | +0.0016 | −.003 / +.007 / +.001 | .004–.009 | negative-but-underpowered |
| **4** | pool count-weighted | −0.0004 | −.001 / +.000 / −.001 | **.002–.003** | **negative-and-resolvable** |
| 4 | pool shrunk n/(n+200) | +0.0000 | −.001 / +.001 / −.000 | **.002–.003** | **negative-and-resolvable** |
| 4 | *pool mean (control)* | +0.0000 | −.001 / +.001 / −.000 | .001–.002 | control passes |

**Nothing wins.** Six arms are resolvably *worse*; the rest are flat within
their own intervals. The largest positive anywhere is +0.0016, on an arm whose
own CI is ±0.001–0.002 per fold and which is negative on one fold.

## What each experiment showed

### Experiment 0 — set-to-set appearance: the reduction is doing work

The per-frame PRTreID embeddings were still on disk (`<key>/feat/`, ~363 MB per
half), so this needed no re-extraction — only a join back to identity, now keyed
by `(player_id, frame)` (the durable fix `check_appearance_alignment`'s own
docstring asks for). Validated against a **disconfirming** control:
rebuilt-vs-stored prototype cosine **mean 0.9994, min 0.912**, versus a shuffled
pairing at **mean 0.742, max 0.993**. (The first attempt joined *zero* rows — a
frame-numbering off-by-one. A mean-cosine check would have hidden it. An empty
join now raises.)

Every set statistic is resolvably worse than the mean, and the plumbing control
(`proto`) is flat, so this is the statistic and not the cap.

**Corrected interpretation.** The first draft claimed a clean monotone ladder
("the more a statistic averages, the better"). That is **contradicted by its own
data**: `median` is the most heavily-averaging statistic and is the second-worst
arm (−0.0272, worse than `top1`). The defensible statement is narrower —
**statistics closer to the cosine-of-means do better, and every departure from
it costs** — which supports the conclusion (pooling is not the bottleneck)
without the mechanistic story.

Two confounds now disclosed:
- top-k is taken over the flattened |a|×|b| matrix with |a| capped at 64 and |b|
  median ~7, so a top-k order statistic partly encodes **set size** (thread age,
  fragment length), not appearance.
- `mutual_nn` is a fraction over |a| frames and is therefore heavily quantised
  at these sample counts — its −0.117 is likely in part a calibration artefact
  (this project's own `quantised-scores-destroy-tail-resolution` failure mode),
  not a clean statement about set-level agreement.

`appearance_visibility` is **constant 1.0** in every pickle — the bridge ran
`--no-pose`, so PRTreID's part-visibility signal was never computed. The
visibility-weighted arms (0d, and the quality half of 4b) are **not tested**.

### Experiment 1 — learned edge scorer: nothing to fuse

Flat, with three diagnostics:

- **The context-shuffled permutation null performs as well as the real arm**
  (−0.0030 vs −0.0005, overlapping intervals). Per-side frame counts, fragment
  counts and field size contribute nothing; whatever the MLP does, it does with
  the four LLRs alone. Direct evidence that the context interaction a scalar sum
  cannot express is **absent**, not merely unexpressed.
- **AUC, corrected.** The first draft compared the MLP's *fused* AUC (0.984)
  against the incumbent's *raw appearance channel* (0.950) — apples to oranges.
  The incumbent's **fused** AUC is **0.9655**. The MLP is still ahead (0.984 vs
  0.966), so the "better model, unmoved frontier" dissociation survives, but it
  is a 0.019 gap, not 0.034.
- **`raw_softmax` collapses to −0.0909 — and this is not a finding about
  objectives.** The spec pre-registered a post-hoc isotonic calibration on every
  arm precisely to prevent this, and **that calibration was never implemented**
  (see Limitations). The collapse is the predicted failure mode of an
  uncalibrated score under a global threshold, occurring because the
  pre-registered remedy was skipped. It is real evidence that the episode
  softmax is unsuitable *uncalibrated* — which is why `on_llr_softmax`, whose
  inputs are already calibrated LLRs in nats, does not collapse — but it is not
  a clean experiment.

**The machinery control FAILED.** The zero-hidden-unit arm does not land on the
linear frontier; it is 0.0088 below. The incumbent is a two-stage estimator
(per-channel calibration on all rows, then 4 softmax weights); the learned arms
use one-stage BCE with early stopping. So the learned family carries an
estimator handicap of ~0.009 and the MLP arms claw back to parity from there.
**A real gain of up to ~0.009 could be masked by this**, and experiment 1 does
not exclude it. This is the weakest null in the set.

### Experiment 2 — trajectory motion: better forecaster, unmoved frontier

Re-run after fixing a real bug: `state_dict()` returns live tensor references,
so the early-stopping checkpoint was mutated by continued training and the
restore was a no-op returning the **last** epoch. Every first-run exp-2 number
came from an unselected model. Now fixed, tested, and re-run.

**The sequence model is a strictly better forecaster.** Held-out log-likelihood
of the true re-entry point, learned Gaussian vs the incumbent's
bounded-diffusion prior (nats/pair):

| gap bin | n (same-pairs), per fold | Δ log-lik NCE | Δ log-lik MLE |
|---|---|---|---|
| < 2 s | 133 / 236 / **81** | +0.59 / +1.16 / +1.31 | +0.89 / +1.18 / +1.36 |
| 2–7 s | 1739 / 1822 / 1414 | +0.59 / +0.76 / +0.73 | +0.61 / +0.77 / +0.74 |
| 7–30 s | 1675 / 1679 / 1415 | +0.17 / +0.22 / +0.21 | +0.14 / +0.27 / +0.24 |
| > 30 s | 855 / 588 / 804 | −0.19 / −0.08 / −0.11 | −0.07 / +0.05 / +0.01 |

Consistent on 3/3 folds, monotonically decaying with the gap, turning negative
past 30 s where a trajectory genuinely carries no information. Sanity bounds
healthy: median |μ| 3.5–8.0 m, σ growing 2.5 m → 34 m. (Note the marquee <2 s
figure rests on an 81-pair bin on one fold; the 2–7 s row, at ~1,500 pairs per
fold, is the load-bearing one at +0.6 to +0.8 nats.)

**Fused frontier effect: +0.0009 to +0.0016**, against per-fold CI widths of
0.002–0.005.

**The clamp objection, measured and answered.** The shipped `TRANS_NEG_CLAMP=0`
flattens all negative transition evidence, and a sharper numerator improves
mostly the negative side — so the arm's upside could have been discarded by
construction. Measured: **56–60% of the arm's transition rows are negative**, so
clamp 0 does discard most of the improvement. Running clamp 6 (which leaves only
7–9% at the bound) recovers that half and moves the frontier from +0.0013 to
+0.0016. **The improvement is expressible and still does not matter.**

Both other paired controls were run — the dt-banded impostor denominator
(`transition.py` OUTSTANDING ISSUE 1) adds +0.0001–0.0003, and NCE, which
optimises the ratio the frontier consumes, does no better than plain MLE. The
**ego-motion control the spec required was not run** (see Limitations).

### Experiment 3 — cohort normalisation: AUC is not the frontier

Field-relative transforms lift the raw appearance channel AUC enormously —
0.950 → **0.997** for the margin form — while the fused frontier is
flat-to-resolvably-worse (−0.0101, −0.0043, +0.0016).

The mechanism is clean: within-field normalisation makes every field's best
candidate look alike (rank 1.0, margin > 0) whether or not it is the right
answer. It improves *ranking within a decision*, which AUC rewards, and destroys
*comparability across decisions*, which the global threshold needs. This also
retro-explains the prior expectation: "margin-over-runner-up governs merge
quality" is about ranking, and the audit had already shown ranking is not the
bottleneck (candidate recall 1.00, 7/8 top-1).

### Experiment 4 — pooling: measured out, and the best-resolved null here

Frame-count weighting and shrinkage are within ±0.001 of the plain mean and of
each other, with the **tightest intervals in the study** (0.0008–0.0030). The
audit's "pooling weights differ — cosmetic, noted" is now measured. Weights sum
over **embedded members only**; merge-order independence and the
unembedded-member trap are both covered by tests.

**This null holds on clean threads only** — the contamination arm the spec
required was not run (see Limitations).

## Cross-check: one arm, pass 1 only, no interval

The static pair frontier scores against oracle-grown threads (purity 1.0); the
threading frontier scores against threads the system built for itself. Exp 3's
margin arm was re-run through the sequential threading path:

| fold | static band Δ | threading band Δ |
|---|---|---|
| game_18 | −0.0027 | −0.0289 |
| game_24 | **+0.0069** | −0.0031 |
| game_47 | +0.0005 | −0.0094 |

The threading frontier is more negative on all three folds, and the single fold
that looked positive on the screening metric inverts. That disposes of the one
apparent positive in the study.

**What this does NOT establish.** The first draft generalised this to "the
screening metric is optimistic, so all the negatives are understated". That is
unearned: this is **one arm, pass 1 only** (`pass2=False`), under a **fixed
`min_margin=0.5` and a fixed threshold grid** rather than each arm's own
quantiles, with **three point estimates and no interval**, and no known-flat
control run through both metrics. The poisoning mechanism predicts harm
proportional to an arm's *wrong merges*, which says nothing about arms that are
flat. Read it as: the one arm that looked positive on the screening metric does
not survive real threading. Nothing more.

## Verdict on the representation caveat

**The caveat narrows but does not come off.**

What is established on this substrate is stronger than "richer inputs did not
help" — it is that **richer inputs measurably improved the underlying models and
the frontier still did not move**:

- the trajectory model predicts re-entry better by +0.6 to +0.8 nats/pair in the
  bin carrying ~1,500 pairs per fold, and by more at shorter gaps;
- the cohort margin channel reaches 0.997 AUC against the incumbent's 0.950;
- the MLP arms reach 0.984 fused AUC against the incumbent's 0.966;

and all three are flat at the merge frontier. Combined with the audit's finding
that surviving misses are evidence-dead, the picture is consistent: the binding
constraint is **how much identity evidence the appearance channel carries about
the hard pairs**, and no re-expression of the available cues reaches it.

Experiment 0 is the load-bearing arm because it is the only model-free one, and
experiment 4 the best-resolved: removing the appearance pooling makes things
resolvably worse, and re-weighting it does nothing to ±0.001.

### What this cannot conclude

> Over ~12.5k episodes from **154 player-halves in 3 matches of one broadcast
> style**, with a **frozen off-domain PRTreID extractor**, no re-expression of
> these cues — set-to-set appearance, learned fusion, sequence motion, cohort
> normalisation, or prototype pooling — beats linear-over-LLR fusion at matched
> coverage, at per-arm resolutions of ±0.001 to ±0.02 precision.

Named and still untested:

1. **Domain-adaptive appearance** — GHOST's actual result is per-sequence
   statistic re-estimation; fine-tuning PRTreID on FOOTPASS identities is
   untried. "Appearance is exhausted" remains "this *frozen* extractor's
   representation is exhausted". Since experiment 0 shows per-frame embeddings
   are too noisy to use individually, improving the **extractor** is the
   indicated next move, not re-summarising its output.
2. **Learned graph-level association** — SUSHI's hierarchical message passing,
   TDLP's global link prediction: an edge score that depends on other edges and
   on transitivity. Nothing here does. `global_assignment.py` on main is the
   adjacent probe.
3. **Part-visibility-gated appearance** — blocked on the `--no-pose` extraction.
4. **The exp-1 estimator gap** — a ~0.009 masked gain is not excluded.

## Limitations and protocol deviations

Pre-registered but **not done**:

- **Post-hoc isotonic calibration on every arm.** The spec called this the fix
  without which three arms are strawmen. It was never implemented. `raw_softmax`
  is the visible consequence; the BCE arms are less affected because BCE is
  already a proper scoring rule.
- **Nested CV for exp-1 architecture selection.** Width/depth/LR were fixed, not
  selected on the inner split. The spec explicitly calls an untuned architecture
  a strawman. Only early stopping used the inner split.
- **Exp 2's ego-motion control** (tail expressed relative to the observed-player
  centroid). Endpoints are where the *camera* lost the player, so the GRU may
  partly have learned the pan. Not run, so not excluded.
- **Exp 4 under contamination.** Its null is a null on oracle-clean threads and
  does not license count-weighting on real tracker output.
- **Exp 4c (learned attention).** Not run; 4a/4b were flat to ±0.001, which was
  the pre-registered condition for skipping it.
- **Exp 3's field guards** — no `log field_size` breakdown, no check that the
  effect is not concentrated in 2-candidate fields, no fit-vs-serve field-size
  comparison.
- **The threading frontier per arm.** Run once, as a one-arm cross-check.
- **"Pooled with a fold effect" as the primary statistic.** Pooled numbers here
  are means of fold band-means and carry **no interval**; intervals are per fold.

Other disclosures:

- **Exp-1 machinery control failed** (zero-hidden 0.0088 below linear).
- **NaN band points.** Where an arm's hull does not span the baseline's coverage,
  the band mean averages fewer than 8 points: `raw_softmax` 2/1/1 dropped,
  `on_llr_softmax` 0/0/3, `zero_hidden` 0/0/2, `margin` 0/0/2, `rank` 1/0/0.
  Those arms are graded on sub-bands. Direction favours the truncated arms, so
  the large negatives are if anything understated.
- **`margin` on game_47 has `unreachable_frac` 0.26** — a quarter of bootstrap
  resamples cannot reach matched coverage. Its interval there is weak, and it is
  the arm used for the cross-check.
- 3 folds share a fit match pairwise, so fold deltas are positively correlated
  and per-fold agreement counts for less than it looks.
- **First-run exp-2 numbers were produced with the early-stopping bug** and are
  superseded by the numbers here.

## Artefacts

No config changed and no fitted artefact shipped — nothing won, so no contract
block or `validate_serving` check was required. Every JSON records its substrate
(`max_gap_frames`, `coords`, `trans_neg_clamp`, stage).

| file | contents |
|---|---|
| `…-mde.json` | pre-registered MDE (gate result: **fails**) |
| `…-calibration.json` | paired v3-vs-flat, static frontier |
| `…-reproduce-v3.json` | audit Phase C reproduction, threading frontier, clamp 6 |
| `…-appearance.json` | experiments 0 and 4 |
| `…-exp1.json` | experiment 1 |
| `…-exp2.json` | experiment 2, per-bin log-likelihoods + clamp saturation |
| `…-exp3.json` | experiment 3 |
| `…-crosscheck.json` | static vs threading frontier, one arm |
