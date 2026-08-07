# 3b — role assignment (`formation/roles.py`)

Status: design v2, 2026-08-05. **v1 was substantially wrong; see "What v1 got
wrong" below.** Cold review ran code against the substrate and refuted four of
its premises.

## The question changed

v1 asked "how do I build role assignment". The measured answer to a question it
never asked makes that premature:

> **A trivial baseline — per-fragment mean (x, y), nearest of 12 fitted role
> anchors, no assignment, no Gaussians, no per-frame anything — scores 0.662
> exact-role excluding GK on VAL** (0.688 including GK). Chance is 0.091.

So the first deliverable is not a system, it is a **headroom probe**: establish
the trivial floor and the realistic ceiling under actual broadcast visibility.
If the gap is small, 3b as an accuracy-improvement exercise is not worth
building, and we say so.

There is a second reason to probe before building. Role here is derived from
mean position — and **mean position has already been measured as a re-ID merge
channel on this repo**: AUC 0.771, but **14 merges out of 13,016 at zero-wrong**.
If role is a discretisation of a signal already known to be nearly useless for
merging, the discretisation has to be where the value comes from, and that is a
claim requiring its own evidence.

## Substrate facts (corrected — v1's were wrong)

Verified by cold review running code, not by me:

- `COL.ROLE`, 13 labels. **TRAIN: 192 team-halves, 9 distinct role-sets**
  (v1 said 80 and 5 — I sampled 40 halves and reported it as the whole set).
- **3 TRAIN team-halves have only 10 distinct roles**, so "exactly 11 per
  team-half" is false.
- **VAL has 12 team-halves and exactly ONE role-set**:
  `GK LB LCB RCB LM RM AM LW RW CF RB`. **MCB and DM never appear in VAL.**
- **Zero mid-half role changes** — this one v1 got right, confirmed across every
  player in every TRAIN and VAL half.
- Visibility: **mean 3.95 observable per team-frame, median 5** (v1 said ~7).
  **P(k < 3) = 0.363**, so a third of team-frames fall below
  `formation_relative`'s `min_observed=3` and return NaN. P(k=11) = 0.001.
- **Visibility is strongly role-correlated**: GK **0.094**, RCB 0.273, LCB
  0.309, CF 0.334, LB 0.372, RB 0.367, AM 0.458, RM 0.462, LM 0.472.

Consequences that reshape the design:

1. **The formation-latent arm is unmeasurable on VAL** — one formation, so a
   constant predictor is 100% correct and restriction can only lose. Move it to
   held-out TRAIN halves carrying the rare formations, or drop it.
2. **Two of 13 roles are never a correct answer on VAL** yet remain assignable.
3. **GK is 9% visible**, so it is both trivially separable when seen and almost
   never seen — and it is largely absent from the observed centroid, biasing
   that centroid in a role-dependent way.

## Direction: oracle on this substrate, and say so

v1 said "canonicalise using 3a". **3a cannot run here.** Each FOOTPASS H5 key is
a single half, genuinely one epoch, so `estimate_direction` returns
`single-epoch` with `epoch_signs=None` and `attacks_positive_x` returns `None` —
v1's rule would abstain 100% of fragments.

Every number below therefore uses **oracle direction** from `COL.TEAM` /
goalkeeper x. This must be named per arm (`oracle-side`, not `3a`). The
3a→3b integration is **unmeasured**; exercising it requires concatenating H1+H2
into one timeline so a flip exists, which is a separate arm.

## Coordinate frame is an ARM, not a premise

v1 asserted formation-relative because occupancy uses it. Measured:
**absolute pitch coordinates tie or beat it** (0.669 vs 0.662 exGK).

The occupancy argument does not transfer. Formation-relative exists there to
cancel a *selection effect* when comparing two fragments' footprints; matching a
fragment to a *static template* has no such cancellation, and the absolute frame
carries the template's anchor for free — while formation-relative adds the
centroid's own error (served relative-x retains only r²=0.575).

Also: `formation_relative(zoom)` applies **no scale normalisation** —
`0.5 + zoom*(xy - c)` with a fixed scalar. Team spread varies hugely with phase
of play, so the same player lands at different relative radii. EFPI's one
substantive finding is that **unscaled matching against fixed templates produces
wrong labels** (a CB labelled DM, a left CM labelled LW), fixed by rescaling to
template extent.

**Frame arms:** `absolute` | `formation-relative` | `formation-relative +
spread-normalised` (divide by team RMS radius) | `hybrid` (absolute plus
centroid offset as 4-D). The centroid-arm sweep (oracle / module / observed) is
only meaningful *conditional on* the frame, so frame is the outer axis.

**v1's pre-registered prediction stands but is now secondary:** 3b should be
more centroid-sensitive than occupancy, because template matching has no
pairwise cancellation. If formation-relative loses outright, the prediction is
moot for the chosen arm and should be reported as such rather than quietly
dropped.

## Estimator: fragment-level, not per-frame

v1 specified per-frame Hungarian then argmax aggregation. Wrong twice:

- With k ≈ 4 of 11, a per-frame assignment is near ill-posed.
- Aggregating hard argmaxes discards the per-frame likelihood exactly where it
  is most uncertain, producing a histogram of winners, not a posterior — so
  v1's abstention threshold would threshold an uncalibrated quantity, which is
  this repo's recorded "quantised scores destroy tail resolution" failure.

**Accumulate per-frame log-likelihoods over the fragment and solve once** (per
fragment, or per sliding window). ADR 002 requires evidence to be *aggregated*
at tracklet level; it does not require the estimator to be per-frame. Per-frame
Hungarian becomes an arm, not the default.

## The assignment constraint must justify itself

Measured: **Hungarian loses to unconstrained argmax** (0.648 vs 0.662 relative;
0.627 vs 0.669 absolute). That is the forced-choice failure already recorded
here (split re-match took swaps 54→72): a hard one-to-one constraint against a
lopsided prior turns one confident correct assignment into two wrong ones — and
it is least justified precisely when k is small, which is always.

The literature's constraint assumes **full visibility**: Bialkowski runs EM +
Hungarian over all 10 outfield players every frame with identity known;
SoccerCPD requires all ten simultaneously measured. Neither faces k ≈ 4.

**The genuinely novel piece, and the one thing worth building:** nobody
published assigns roles to a partially-observed subset while modelling *which*
roles are absent. Visibility here is strongly role-correlated and emphatically
not missing-at-random. So the formulation to test is a rectangular assignment
with **per-role log-occupancy priors** fitted on TRAIN
(`-log P(role r observable | k)`), or equivalently dummy rows with
role-specific costs — not a bare rectangular Hungarian that lets the cost matrix
alone decide which roles go unfilled.

## Baselines — mandatory, in this order, in every table

1. **chance** (1/11 = 0.091).
2. **prior** — always the most common role.
3. **fragment-mean nearest-anchor, unconstrained** — *the trivial floor,
   measured at 0.662 exGK*. Computed at the same aggregation level as the
   headline. **Any arm that does not beat this by more than the paired
   match-clustered CI is not a result.**
4. **unconstrained argmax** with the full model — isolates whether the
   assignment constraint helps or hurts.

Report **excluding GK as the headline** (GK is 9% visible but trivially
separable when seen, so including it inflates every arm); GK-inclusive is a
footnote.

## Comparison bars — none of them are targets

- **SoccerCPD 86.5% position accuracy** — verified, but 10/10 visibility, known
  stable identity, and **one-minute segments** (~1,500 frames of evidence per
  decision). A full-information ceiling, not a like-for-like bar.
- **Bialkowski 75.33%** — that is *formation-cluster* accuracy vs expert labels;
  the paper reports **no per-player role accuracy at all**.
- **EFPI — no quantitative evaluation exists.** Do not cite as an accuracy
  reference.
- A 2024 Frontiers survey states outright that evaluation metrics vary widely
  and quantitative evaluation is hampered by absent ground truth. We are
  defining our own metric; say so.

## Metrics

Exact role accuracy (fragment-level, VAL, exGK) primary. Plus **line** accuracy
(defence/mid/attack) and **flank** accuracy (left/centre/right), since exact
role may be too strict and these say *how* it fails. Full confusion matrix, not
a scalar: the literature reports only depth confusions (4-4-2 ↔ 4-2-3-1),
because everyone else has known direction and full observation, which pins the
lateral axis for free. **We remove exactly that, so left/right confusion is
likely our dominant failure and is unmeasured anywhere in the literature.**
Abstention rate beside every accuracy. n = 6 halves / 3 matches; cluster by
match.

## Pre-committed decision rules

- **Headroom first.** If the best deployable arm does not beat the 0.662 trivial
  floor by more than the match-clustered CI, **report that and stop** — do not
  ship a system whose contribution is inside the noise of two floats.
- If Hungarian loses or ties, the one-to-one constraint is mis-specified for
  partial visibility; the fix is occupancy priors, **not more search**.
- If `module` recovers < half of `observed`→`oracle`, the deployable centroid is
  3b's bottleneck; say so and do not present oracle numbers as the result.
- Any re-ID claim requires the merge-level measurement, separately.

## What v1 got wrong (kept deliberately)

Substrate counts (80/5 vs 192/9) from sampling 40 halves and reporting it as
the whole; "exactly 11 roles per team-half"; k ≈ 7 vs measured 3.95;
formation-relative asserted rather than tested; 3a named as the direction source
when it cannot run on single-half keys; per-frame Hungarian + argmax
aggregation; and no trivial baseline, which is the omission that would have made
everything else look better than it is.

## Scope boundary

Land the headroom probe first. Build the estimator only if the probe shows
headroom. **Do not wire role into any re-ID channel in this change**, and do not
flip any default.
