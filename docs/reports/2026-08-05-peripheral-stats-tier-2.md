# Peripheral Statistics — Tier 2 (stats 12–20), results

Branch `peripheral-stats-tier-2`, cut from `worktree-peripheral-stats-tier-1` (not from `main`).
Plan: [`docs/prds/peripheral-stats-tier-2.md`](../prds/peripheral-stats-tier-2.md).
Source: Notion "📊 Peripheral Statistics", Tier 2.

**Ground-truth labels only.** No detector, tracker or spotter output is consumed anywhere.
xT is fitted on the 96 FOOTPASS train halves and applied to the 6 val halves; the 48 train
games and 3 val games are disjoint.

Reproduce: `uv run python -c "from matchlab_train.experiments.tier2_stats import run; run()"`
→ `data/reports/tier2-stats/`.

---

## The three results worth reading

### 1. The surface is geometry; the *differences* are not. Measure the baseline at the level the stat is consumed

The plan's rule R4 required a trivial baseline beside every correlation. Running it produced a
number that looked decisive and, taken alone, would have been **wrong**:

| R4 baseline (i) — fitted xT vs a pure −distance-to-goal surface | Spearman ρ |
|---|---|
| **per zone** (192 zones) | **0.9902** |
| per **action** delta (4 448 rated actions, whole val split) | 0.8696 |
| per **player** total, successes only (154 player-halves) | **0.6550** |
| per **player** total, risk-adjusted | 0.6900 |

| R4 baseline (ii) — per-player **action count** in place of the xT total | Spearman ρ |
|---|---|
| vs xT total | 0.4443 |
| vs risk-adjusted total | 0.1816 |

Baseline (ii) is favourable and was **omitted from the first version of this report**, which
cited R4 as its justification while running half of it. It matters: if a player's xT total were
a touch counter wearing a model, it would show here. It does not — and the risk-adjusted total
is nearly independent of volume (ρ = 0.18), which is the strongest thing this branch can say
for §12d's risk-adjusted convention.

The first row alone says "the fitted grid is a relabelled distance map". The last row says
something much weaker: distance-to-goal reproduces about two thirds of the player ranking, and
**a third of it is information the geometry does not contain**.

The gap between the rows is not noise, it is arithmetic — and it is the *same* point the plan
already makes in §12c about percentiles. An action's value is `xT(end) − xT(start)`, a
**difference between two nearby zones**, so a deviation far too small to move a rank-correlation
over zone *levels* can dominate the difference. Reporting the zone-level ρ as the headline
measures agreement at the one level where the stat is never used.

**Correction recorded rather than quietly fixed:** an earlier draft of this report led with
"the fitted xT surface is a relabelled distance-to-goal map (ρ = 0.99)". That claim is
withdrawn. R4 was applied — and then applied at the wrong level, which is a subtler version of
the failure R4 exists to prevent. The rule needs the amendment: **evaluate a trivial baseline at
the level the statistic is consumed at, not at the level it is convenient to compute.**

**Where that leaves §12.** xT here adds real but moderate information over pure geometry for
player ranking (ρ = 0.655 means the two orderings genuinely differ), and essentially none for
describing the pitch. It remains uncalibrated, and no claim is made that it beats geometry by a
useful margin on any decision.

**A second correction, to the stated mechanism.** An earlier version explained the geometry as
"`g(z)` is `xg()` at the centroid and `xg()` is a function of location, so the only
non-geometric information is `T`". That is **false**, and measurement says the opposite:

| | Spearman ρ vs −distance to goal |
|---|---|
| fitted xT | **+0.9902** |
| `s(z)` — the per-zone shot share, estimated from 1 093 train shots | **+0.7404** |
| `g(z)` — the xG prior | **−0.2525** |

`g(z)` is *anti*-correlated with distance-to-goal, and with the fitted surface (ρ = −0.2885).
The geometry in the result comes from **`s(z)`, which is data**, not from `g`. The original
inference was wrong.

**And that anti-correlation is itself a latent defect in Tier 1's xG model, surfaced here.**
`location_only_xg` returns **0.531 at (3.3 m, 65.2 m) — 106 m from the goal being attacked**,
against **0.181 at the penalty spot**. The Soccermatics fit's quadratic terms extrapolate
absurdly far outside the 8 451-shot sample's support, and Tier 1 never saw it because it only
ever evaluated `xg()` at real shot locations. It is invisible in this fit only because
`s(z) = 0` exactly in those zones, so the nonsense value is multiplied by zero — one changed
shrinkage parameter away from contaminating the surface. `xg()` should be domain-clamped, and
`xt.py`'s "a shape error in `g` does not survive at all" is doing more work than was admitted.

The stability gate (train-half vs train-half at matched n) passes at **ρ = 0.9918** against a
pre-registered ≥ 0.60. On its own that is nearly uninformative for the same reason: a surface
monotone in distance-to-goal would be perfectly stable too. It is reported as a
regression-grade check, not as evidence the model transfers.

### 2. The failure-handling fork is resolvable here, and success-only is degenerate

**The pre-registered criterion (R6) is the per-player rank correlation between the arms, with
equivalence declared at ρ ≥ 0.95.** Measured:

| | Spearman ρ, socceraction vs singh |
|---|---|
| per half | 0.713, 0.318, 0.482, 0.398, 0.356, 0.292 |
| **pooled, xT total** (154 player-halves) | **0.4333** |
| pooled, risk-adjusted | 0.4251 |
| pre-registered equivalence gate | ≥ 0.95 |

0.43 is nowhere near 0.95, so the arms are **not** equivalent and the fork is resolvable on this
data. The choice reorders more than half the player ranking.

> **Correction:** the first version of this report resolved the fork on **max absolute
> difference across the 192-zone surface** (0.11892 for the arm against 0.02227 for the widest
> `max_gap_s` perturbation) and reported the ratio as "5.3×". That statistic was not
> pre-registered, was chosen after seeing the data, and is not a sound comparison as
> constructed: it compares a single order statistic across two perturbations of different kinds
> *and* different scales — a binary modelling fork versus a ±50%/±100% nudge of one heuristic
> constant — and the max over 192 zones is dominated by the goal-mouth cell. The registered
> statistic above reaches the same conclusion on better evidence. The max-abs-diff figures are
> retained below only as a supporting ablation.

The inferencer ablation the plan required — because pass outcome and end point are both
`INFERRED` by `chains.py`, never labelled — is still the right check, and it passes:

| perturbation | max abs change in the surface |
|---|---|
| arm: absorbing-failure → success-only | 0.11892 |
| inferencer: `max_gap_s` 10 s → 5 s | 0.02227 |
| inferencer: `max_gap_s` 10 s → 20 s | 0.01450 |
| **grid resolution: 16×12 → 8×6** | **0.16 (larger than the arm)** |

The last row was computed by the runner and **absent from the first version of this report**,
which is selective reporting: coarsening the grid — a specification taken verbatim from Singh,
with no more justification than the failure arm has — moves the surface *more* than the fork
this section calls decisive. It does not overturn the R6 result, which is measured per player
rather than per zone, but a reader shown only the first three rows cannot see that a third knob
dominates the second.

What the arms produce:

| arm | iterations | zones with zero leakage | xT range | centre channel |
|---|---|---|---|---|
| absorbing-failure (socceraction) | 46 | 0 / 192 | 0.0039 – 0.4689 | monotone 0.006 → 0.451 |
| success-only (Singh) | 160 | **191 / 192** | 0.0907 – 0.4689 | **plateau 0.1222–0.1228 over bands 0–8, decaying to 0.1110, then dips** |

With `T`'s rows summing to 1 the chain has no absorbing state: possession never ends, so every
zone inherits nearly the same long-run scoring probability and the surface stops carrying
positional information. It is not even monotone — band 12 (0.0907) sits *below* a team's own
goal-mouth value (0.1227). Absorbing-failure is the default.

Two claims from the first plan draft are **withdrawn** rather than quietly fixed: that
absorbing-failure "guarantees a contraction" (it does not — a zone whose every observed move
completed has zero leakage, which is why `min_leakage` is now a measured diagnostic), and that
success-only fails to converge (it converges to the minimal non-negative fixed point).

### 3. The §14 taxonomy is substantially a knob, and §19's corner detector is not validatable here

**Threshold sweep** (val split, 752 chains). Each row moves one threshold:

| variant | unclassified | high_turnover | counter_attack | direct_attack | build_up |
|---|---|---|---|---|---|
| **default** | **627** | **63** | **39** | **5** | **18** |
| high turnover 30 m | 663 | 26 | 39 | 5 | 19 |
| high turnover 50 m | 573 | 122 | 36 | 5 | 16 |
| counter directness 0.50 | 586 | 63 | 84 | 1 | 18 |
| counter directness 0.90 | 653 | 63 | 12 | 6 | 18 |
| build-up 5 passes | 610 | 63 | 39 | 5 | 35 |
| build-up 15 passes | 639 | 63 | 39 | 5 | 6 |

Counter-attacks range 12–84 (7×) on the directness threshold alone, build-up 6–35, high
turnovers 26–122. **Every one of these thresholds is a published provider value**, and they
still span most of the plausible range — because the providers fitted them against their own
labelled taxonomies, which this ground truth does not have. No per-type count from it should be
quoted as a measurement.

> **The label-permutation null on this taxonomy is circular, and its result is withdrawn.** An
> earlier version reported "the between-type conversion spread is separable from a
> label-permutation null (spread 0.571, p = 1/2001), so the taxonomy is not noise". The spread
> is `shot-ending / chains` by type, and its two extremes are `build_up` (0.611, n=18) and
> `unclassified` (0.040, n=627) — but `build_up`'s *definition* requires
> `ends_in_shot or has_box_touch`. The label partly encodes the outcome being measured, so no
> permutation could fail. p = 1/2001 is a tautology, not evidence. Re-running it on the two
> outcome-free types (`high_turnover`, `counter_attack`) is the test that would mean something,
> and it has not been run.

**`unclassified` is 83.4%** (627/752), far outside the pre-registered 25–60% band. Reported,
not tuned; a characterisation test asserts the band is *breached* so retuning fails the suite.
The mechanism was measured: only 289 of 752 chains are shorter than 3 own events, while **77
chains carry 10+ own passes and just 18 become `build_up`** — the providers' definitions are
conjunctions, so the terminal condition rejects long chains, not the pass count.

**The §19 corner detector is reported as FAILED.** Its permutation null hits the floor
(21 of 99 crosses within 3 m of a flag, all 21 also following a >10 s gap, p = 1/2001) — which
read as a strong result until R4 was applied to it:

| region tested | detections | p |
|---|---|---|
| ≤3 m of the attacked **corner flag** (the detector) | 21 | 0.0005 |
| ≤3 m of the **nearest touchline** | **21** | **0.0005** |
| ≤3 m of the attacked **byline** | 22 | 0.0005 |

and p stays at the floor for every radius from 2 m to 10 m — 10 m admits the very box crosses
the radius exists to exclude.

**The disposition is FAILED, but the first version gave the wrong reason for it.** The
touchline region is not an independent control that happened to tie: `touchline ≤ 3 m` is
*implied by* `corner_flag ≤ 3 m`, so the control **strictly contains** the detector's region,
and after the gap filter the two select the **identical 21 events**. A nested superset cannot
discriminate, so it was never going to separate them; it is uninformative by construction
rather than a matched baseline that scored equally. Building a genuinely disjoint control — a
mid-touchline band excluding the corner arcs — is the test that was actually needed.

The reason the detector fails stands regardless, and it is the simpler one: **there is no
corner label and no negative class**, so nothing here can distinguish a corner from any other
restart taken near the flag. Separation from a null is necessary, not sufficient.

---

## A data discovery: FOOTPASS writes sentinel coordinates for some crosses

Found while auditing the detections above. **5 rows out of 9 917 540 in the val tactical h5 sit
at an exact corner of the unit square, and all 5 carry `CLS == 3` (cross).** Coordinates here
are not clamped — X runs −0.035 to 1.018, Y −0.646 to 1.066 — so an exactly integral corner is
not a boundary effect. Under the marginal rates actually present in this file (42 rows with
x ∈ {0,1} and 183 with y ∈ {0,1}) the expected count is **7.7 × 10⁻⁴ rows**, against 5 observed.
(An earlier version said ~1e-5, which does not reproduce; and the marginals are themselves
contaminated by the same mechanism, so this is a soft bound either way.)

FOOTPASS is writing a sentinel or imputed position for a specific subset of crosses. Whether it
means "corner kick" or "position unknown" is not decidable from this data, but **2 of the 21
corner detections are sentinels rather than measurements**, and this is the closest thing to a
corner label the dataset contains. Worth investigating before anything is built on it.

---

## Per-stat results

### §12 xT — see above. §13 momentum

Ships, but note what its gate is worth: §12's stability gate is train-half vs train-half at
ρ = 0.9918 against a ≥ 0.60 bar, and a pure distance-to-goal surface scores ρ = 0.986 and 0.990
against those same two halves. **The gate clears geometry by about 0.003**, so passing it is not
evidence of anything about §13. §13 is **a presentation choice, not a measurement**, and says so
on the returned object. Structure — per-minute bins, per-club
maximum, `[0, 0.1]` cap, recency weighting over ~3–4 minutes, inter-club difference — follows
Opta's published Match Momentum. The kernel family, half-life, and the use of xT in place of
Opta's undisclosed possession-value model are **ours**. Antisymmetric by construction, so one
signed series ships and both-club aggregates are forbidden.

### §12d — the credit convention changes who is top

On `game_18_H1`, applying the train grid:

| convention | players with a negative total | top player |
|---|---|---|
| successes only (socceraction's rating behaviour) | 4 of 23 | 203 |
| risk-adjusted (failures debited `xT(start)`) | **17 of 23** | **102** |

Most players give up more xT through failed moves than they create through successful ones. A
player card built on the successes-only total alone would show far more net-positive
contributors than the data supports, which is an artefact of the convention. Both totals ship
and neither is privileged; §12e's highlight ranking defaults to risk-adjusted, because a failed
move that surrendered a valuable position is a worse moment than a neutral one.

Credit coverage is 720 of 839 move actions rated (14.2% unrated), surfaced rather than dropped.

### §14/§15 — a corrected spec clause changed the headline

A cold review found the counter-attack origin clause implemented **backwards**. StatsBomb
play_pattern 6 says the possession "started with an open play turnover **outside the
counter-attacking team's final third**"; a team's final third is the one it *attacks*, so that
is an upper bound (`x < 2L/3`). It was implemented as a lower bound (`x ≥ L/3`) against a
docstring that had inserted the word "own" and rewritten "final" as "defensive" — excluding
precisely the deep regains that are the canonical counter-attack.

Counter-attacks **13 → 39** (per half 8, 4, 5, 10, 3, 9; 3 shot-ending across the split).
Reported as a whole-split count with the number of halves attached; no per-half rate is emitted.

### §16 field tilt

One club's share per half: 0.406, 0.566, 0.210, 0.248, 0.517, 0.647. The definition is **ours**
— no provider definition could be verified (StatsBomb's article returned an empty page,
`statsbomb.com` redirects off-host, Opta's glossary omits the term, and the one source that
opened contradicts itself within a sentence).

Safe from the cross-club frame bug because each club's count is computed in its own
attack-normalised frame and only the *counts* are combined — the only Tier 2 stat with that
property. Sums to 1 across clubs, so only one share is reported.

### §17 PPDA — abstains at team-half granularity

| | |
|---|---|
| in-zone defensive actions, whole val split | **25** (17 blocks, 8 tackles) |
| team-halves with a zero denominator | **4 of 12** |
| pooled PPDA | **80.8** (2021 passes / 25 actions) |
| top-five-league season average (Wyscout) | ~11.0 |

No per-half or per-team-half number is reported anywhere; the metric returns `None` at a zero
denominator and is excluded from the headline set. The pooled figure is biased high by roughly
the factor visible above, because interceptions, challenges and fouls have no class in this
ground truth — tackle and block are the only defensive actions available.

The plan's stated *mechanism* was wrong and is corrected in place: it claimed blocks barely
reach the zone. They are the **larger** share (17 vs 8), at a similar retention rate (23% vs
32%). The conclusion is unaffected; the reason was an assumption measurement contradicted.

### §18 high turnovers

63 across the split (per half 15, 9, 9, 10, 8, 12), 9 shot-ending. **No time window** — Opta's
construction is structural, and the 5-second threshold that circulates belongs to StatsBomb's
*counterpress*, a different metric. The distance metric is a genuine choice, not a detail:
radial-to-goal-centre gives 63, the x-line variant **92** — a 46% swing on an ambiguity the
source does not resolve.

### §19 set pieces — mostly abstains

Throw-ins are the only labelled restart: **922 train / 49 val live**, from 1730 / 97 raw.
**46.7% of throw-in labels are replays**, and that loss is *systematic*, not random —
broadcasts cut to replays during dead-ball periods, which is exactly when throw-ins are taken.
The survivors are therefore a biased subsample. (The join was checked: 6070 val rows → 6070
unique keys, zero collisions, so the rate is real and not a join defect.)

Corners, free kicks, goal kicks and penalties return `None` with a reason, never 0. **0 of 65
val shots are set-piece-attributable**, and `shots_open_play` is `None` rather than 65 — with no
corner or free-kick class, the source cannot say a shot came from open play.

### §20 goalkeeper

`ROLE == 1`, stable within a half. On `game_18_H1` the two keepers attempted 14 and 19 passes
(13 and 18 completed) and faced 9 and 5 shots for 1.09 and 0.76 xG faced.

Length buckets are **ours and unvalidatable** — FBref returned HTTP 403 to two independent
research passes, so the commonly quoted yard thresholds could not be attached to any source and
are not cited. Saves, saves-vs-xG-faced, post-shot xG, claims and punches all abstain with
reasons: there is no shot outcome, no on-goal contact point, and no claim/punch class.

---

## Recall-sensitivity sweep (the build-order table)

`game_18_H1`, 10 trials per cell, both loss models. Gating = highest drop rate tolerated within
10% movement.

| stat | tolerates (**crowd-biased**) | tolerates (uniform) | uniform @20% | crowd-biased @20% |
|---|---|---|---|---|
| `ratio_field_tilt` | **40%** | 40% | 0.027 | 0.051 |
| `ratio_unclassified` | **20%** | 20% | 0.032 | 0.058 |
| `count_chains` | **10%** | 40% | 0.059 | 0.130 |
| `count_rated_actions` | **10%** | 5% | 0.202 | 0.174 |
| `count_counter_attacks` | **5%** | 10% | 0.138 | 0.400 |
| `count_high_turnovers` | **0%** | 20% | 0.080 | 0.213 |
| `count_build_up` | **0%** | 5% | 0.520 | 0.540 |

**The gating column is now derived from the crowd-biased model**, and the uniform column is
shown beside it to make the gap visible. The first version published a single "tolerates"
column computed as the max over *both* models, so a stat qualified if *either* passed — which
meant the published build-order was derived from the easier model while the text argued the
harder one is what matters. `count_chains` read as tolerating 40% while its own crowd-biased
row showed 13% movement at a 20% drop; `count_high_turnovers` and `count_build_up` do not
survive even the smallest sweep step under crowd-biased loss.

Tier 1's central finding is **directionally reproduced**: ratios tolerate 20–40% event loss,
counts 0–10%, and chain-relational counts degrade substantially faster under crowd-biased loss
than uniform. The two largest ratios (`count_counter_attacks` 2.9×, `count_high_turnovers`
2.7×) are **not resolvable at this sample size** and the earlier "~3×" is withdrawn as a point
estimate: they rest on 10 trials over baselines of 8 and 15 events on a single half, so movement
is quantised in steps of 1/8 and 1/15 and no interval was computed. The stats with large
baselines (`count_rated_actions`, n=720) show no such ratio. The direction is supported; the
factor is not.

Signed stats (`signed_net_xt`) are reported as **absolute movement in stat units** under rule
R5, because the relative metric explodes near zero and the old zero-skip silently dropped rows.

> A first run of this sweep produced a crowd-biased column **identical** to the uniform one,
> because the events were loaded without off-ball context and the crowd model weights by
> players within 10 m — so it degenerated to uniform in silence. Fixed; the harder loss model
> is the one that matters.

---

## What this branch does not establish

* **No calibration claim** for xT or xG. There are no goal labels anywhere in this data.
* **No evidence that possession value beats geometry** — ρ = 0.99 against distance-to-goal.
* **No validated attack taxonomy.** Provider thresholds are cited, but citing a threshold does
  not validate a classifier against a taxonomy this data does not contain.
* **No corner detection.** The detector does not beat a corner-free control.
* **No per-half PPDA**, no per-half counter-attack rate, no rate at n < 10 anywhere.
* **Team-level split cleanliness is unverifiable**: `PLAYER_ID` is match-local (32 distinct
  values across 48 games), so no cross-match club key exists. Val is out-of-sample at the
  *match* level only.
* **Two pre-registered obligations were breached and are reported as breaches, not omissions.**
  R1 requires a stratified null for §15 counter-attacks; the 39-counter-attack headline has
  none, so it is a corrected count, not a validated one. R7 requires this branch to run its own
  mutation suite with a stated target survivor rate; it did not — the mutation findings recorded
  below came from cold reviewers. Since R6 designates the mutation run as §12's only real
  validation, **§12 currently has no reported validation of its own.**
* **R6's "val reported separately with a bootstrap CI"** was not done either. No val-fitted
  surface and no CI exist.
* **`FIFA_PITCH` is a fixed 105 × 68 spec** applied across 48 real pitches of differing size, so
  a pooled fitted grid's zone boundaries do not correspond to identical physical locations
  across matches. Unmeasurable from this data.

## Defects found by cold review, recorded because each is a trap

1. **The xT cache validated nothing.** Keyed only on `(split, arm)` — no code hash, no
   parameters. With a stale file present, mutating `s(z)` to identically zero left the entire
   characterisation suite green, because not one test called `fit()`. Now fingerprinted on the
   source of every module the fit depends on.
2. **Momentum applied the half offset twice**, rendering H2 minute 0 at minute 90. Both axis
   tests were blind to it.
3. **The counter-attack spec clause was inverted** (above).
4. **The rule-0 frame test was anchored at y = W/2**, the fixed point of `y → W − y`, so the
   mirror-instead-of-rotation mutation passed it.
5. **Two tests compared a constant to itself** (momentum's cap and bin width) — the exact
   self-referential assertion that let a mutated coefficient reach a commit on Tier 1.
6. **`corner_null_test`'s default permutation count was never exercised**; 20 permutations would
   floor p at 0.048, under alpha, carrying no evidence. Now refused below 200.
7. **R4's second baseline was never computed**, and R6's registered arm criterion was replaced
   by a statistic chosen after seeing the data. Both are now run; both reach the same
   conclusions on better evidence. A pre-registered rule is not self-executing.
8. **The §14 permutation null was circular** — `build_up`'s definition contains the outcome the
   null was testing, so it could not have failed.
9. **The corner detector's "control" was a nested superset** of the detector's own region,
   selecting the identical events, so it could never have discriminated either way.
10. **The sensitivity gating column took the max over both loss models**, publishing a build
    order derived from the *easier* one while the text argued the harder one is what matters.
11. **Cross-match event interleaving.** `half` is per-match and `frame_idx` per-half, so pooling
   matches before chaining interleaves them: `game_18_H1` spans frames 64–75306 and
   `game_24_H1` spans 5397–77162. Measured on those two halves — 419 chains instead of 248,
   with 61 chains containing events from *both matches*. `build_chains` now refuses such a
   stream outright rather than handling it defensively.

Two research agents were briefed independently on xT's failure handling and returned
**contradictory** answers; one had read only socceraction's documentation page, the other its
source. It was settled by reading the code first-hand. A third reading, from a cold reviewer who
consulted only Singh's blog, was correct about the blog and silent about the code. All three are
recorded in §12b of the plan, because no one of them was sufficient.

## Next

The source doc's build order puts Tier 3 (ball-local pitch control, pressure events,
line-breaking passes) after this, gated on whether in-frame multi-player pitch coordinates are
stable enough to justify the scoping discipline. Before that, two things here deserve closing:
the sentinel-coordinate discovery above, and an xT arm that can beat the distance-to-goal
baseline — without which §12 is an engine that has not yet earned its complexity.
