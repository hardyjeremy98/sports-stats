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

### 1. The fitted xT surface is a relabelled distance-to-goal map (ρ = 0.99)

This is the finding that most constrains what §12 can be used for, and it exists only because
the plan's rule R4 required a trivial baseline beside every correlation.

| quantity | value |
|---|---|
| Spearman ρ, fitted xT vs **−distance to goal centre** (192 zones) | **0.9900** |
| Spearman ρ, train-half vs train-half at matched n (stability gate) | 0.9918 |
| pre-registered stability gate | ≥ 0.60 |

The stability gate passes overwhelmingly — and that is nearly meaningless on its own, because a
surface that is a monotone function of distance-to-goal is trivially stable too. Read together,
the two numbers say the fit is **as close to plain geometry as it is to itself across splits**.

This is not a surprise once §12c is taken seriously: `g(z)` on this ground truth is `xg()`
evaluated at the zone centroid, and `xg()` here is a deterministic function of location
(`is_header` and `is_set_piece_origin` both default False, `defenders_in_lane` is unavailable).
So the only non-geometric information entering the surface is the transition matrix `T`, and it
moves the result by very little.

**What that licenses.** xT as built is a defensible *relative* action-value surface and an
adequate ranking function for §12e's highlight selector. It is **not** evidence that a
possession-value model adds anything over distance-to-goal on this data, and no such claim is
made. A future arm that beats the distance baseline by a stated margin would be a real result;
this one does not.

### 2. The failure-handling fork is resolvable here, and success-only is degenerate

The plan pre-registered that the arm comparison could only be reported as a modelling result if
it survived a sweep over the *inferencer* — because pass outcome and end point are both
`INFERRED` by `chains.py`, never labelled. It survives comfortably:

| perturbation | max abs change in the surface |
|---|---|
| **arm**: absorbing-failure → success-only | **0.11892** |
| inferencer: `max_gap_s` 10 s → 5 s | 0.02227 |
| inferencer: `max_gap_s` 10 s → 20 s | 0.01450 |

The arm choice moves the surface **5.3× more** than the widest inferencer perturbation, so the
fork is not an artefact of the heuristic. What the arms produce:

| arm | iterations | zones with zero leakage | xT range | centre channel |
|---|---|---|---|---|
| absorbing-failure (socceraction) | 46 | 0 / 192 | 0.0039 – 0.4689 | monotone 0.006 → 0.451 |
| success-only (Singh) | 160 | **191 / 192** | 0.0907 – 0.4689 | **flat ~0.118 across 12 of 16 bands, then dips** |

With `T`'s rows summing to 1 the chain has no absorbing state: possession never ends, so every
zone inherits nearly the same long-run scoring probability and the surface stops carrying
positional information. It is not even monotone — band 12 (0.091) sits *below* a team's own
goal-mouth value (0.123). Absorbing-failure is the default.

Two claims from the first plan draft are **withdrawn** rather than quietly fixed: that
absorbing-failure "guarantees a contraction" (it does not — a zone whose every observed move
completed has zero leakage, which is why `min_leakage` is now a measured diagnostic), and that
success-only fails to converge (it converges to the minimal non-negative fixed point).

### 3. The §14 taxonomy is substantially a knob, and §19's detector fails its trivial baseline

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
labelled taxonomies, which this ground truth does not have. The between-type conversion spread
is separable from a label-permutation null (spread 0.571, p = 1/2001), so the taxonomy is not
noise; but no per-type count from it should be quoted as a measurement.

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
| ≤3 m of the **nearest touchline** — corner-free control | **21** | **0.0005** |
| ≤3 m of the attacked **byline** — corner-free control | 22 | 0.0005 |

and p stays at the floor for every radius from 2 m to 10 m. The test establishes only that
*crosses from wide or deep positions follow stoppages*. This is the repo's own "coverage
metrics multiply, they don't measure" lesson recurring: R4 was written into this plan and then
not applied to the one stat it was written for.

---

## A data discovery: FOOTPASS writes sentinel coordinates for some crosses

Found while auditing the detections above. **5 rows out of 9 917 540 in the val tactical h5 sit
at an exact corner of the unit square, and all 5 carry `CLS == 3` (cross).** Coordinates here
are not clamped — X runs −0.035 to 1.018, Y −0.646 to 1.066 — so an exactly integral corner is
not a boundary effect; under the marginal rates you would expect ~1e-5 such rows.

FOOTPASS is writing a sentinel or imputed position for a specific subset of crosses. Whether it
means "corner kick" or "position unknown" is not decidable from this data, but **2 of the 21
corner detections are sentinels rather than measurements**, and this is the closest thing to a
corner label the dataset contains. Worth investigating before anything is built on it.

---

## Per-stat results

### §12 xT — see above. §13 momentum

Ships (its gate is §12's stability, which passed) but is **a presentation choice, not a
measurement**, and says so on the returned object. Structure — per-minute bins, per-club
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

| stat | tolerates | uniform @20% | crowd-biased @20% | ratio |
|---|---|---|---|---|
| `ratio_field_tilt` | **40%** | 0.027 | 0.051 | 1.9× |
| `count_chains` | 40% | 0.059 | 0.130 | 2.2× |
| `ratio_unclassified` | 20% | 0.032 | 0.058 | 1.8× |
| `count_high_turnovers` | 20% | 0.080 | 0.213 | **2.7×** |
| `count_counter_attacks` | 10% | 0.138 | 0.400 | **2.9×** |
| `count_rated_actions` | 10% | 0.202 | 0.174 | 0.9× |
| `count_build_up` | **5%** | 0.520 | 0.540 | 1.0× |

Tier 1's central sensitivity finding **reproduces on Tier 2**: ratios tolerate 20–40% event
loss, counts 5–10%, and chain-relational counts degrade ~3× faster under crowd-biased loss than
uniform. `count_build_up` is the most fragile stat in the tier — it needs 10 consecutive passes
*and* a terminal condition to survive, so it fails even a 5% drop.

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
7. **Cross-match event interleaving.** `half` is per-match and `frame_idx` per-half, so pooling
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
