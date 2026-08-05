# Peripheral Statistics — Tier 2 (stats 12–20)

Source: Notion "📊 Peripheral Statistics" (curated 2026-08-05), Tier 2 = "the narrative
layer" — team-level and value-based stats built on Tier 1. This document is the
implementation plan and the honest coverage record for what is and is not derivable from
ground truth.

Builds directly on `worktree-peripheral-stats-tier-1` (`matchlab_core.stats`), which is a
prerequisite and is not yet merged to `main`. This branch is cut from it, not from `main`.

**Revised after two independent cold reviews**, which between them found three blockers, and
which corrected five of this document's own data claims. Every correction is recorded inline
rather than silently patched, because each one is a trap for the next reader.

## Scope

| # | Stat | Family |
|---|------|--------|
| 12 | Expected Threat (xT) | possession-value engine |
| 13 | Team momentum chart | derived from 12 |
| 14 | Possession sequences + attack-type classification | chain taxonomy |
| 15 | Counter-attack stats | subset of 14 |
| 16 | Field tilt | territory ratio |
| 17 | PPDA | pressing |
| 18 | High turnovers, shot-ending high turnovers | pressing |
| 19 | Set-piece breakdown | restart family |
| 20 | Goalkeeper metrics | role-specific |

Notion's rationale for building 12 early is that it is **three** product surfaces, not one:
player ratings, the momentum chart, **and a highlight selector** ("ranked per action it is an
automatic highlight selector … the top-N xT-delta events in a player's match"). The third is
in scope — see §12e. The first draft dropped it silently.

**Ground-truth labels only.** No tracker, detector or spotter output is consumed anywhere on
this branch, exactly as in Tier 1. Out of scope: Tier 3, Tier 4 (never), UI, pipeline stage
registration, and any artifact contract — nothing in the pipeline produces this sheet yet, so
registering an `ArtifactName` would be a dangling contract.

## Ground-truth substrate

Tier 1 used the FOOTPASS **val** split only. Tier 2 needs a *fitted* model (xT), and fitting
and evaluating on the same 6 halves would be in-sample.

### Every count in this document is replay-filtered, and labelled `live`

The first draft quoted replay-filtered and unfiltered counts interchangeably, and once
estimated a filtered count by applying the *global* 9.38% replay rate to a class whose actual
replay rate is 0.4%. Both reviews caught it. **Replays are wildly non-uniform by class**, so
the global rate must never be used to scale a class. Measured directly from
`playbyplay_{train,val}.json`:

| class | train total | train replay | train **live** | val live | replay % |
|---|---|---|---|---|---|
| carry | 35 527 | 2 823 | **32 704** | 2 239 | 7.9 |
| pass | 45 621 | 4 466 | **41 155** | 2 755 | 9.8 |
| cross | 2 156 | 219 | **1 937** | 99 | 10.2 |
| **throw-in** | 1 730 | 808 | **922** | **49** | **46.7** |
| shot | 1 097 | 4 | **1 093** | 65 | **0.4** |
| header | 3 642 | 162 | **3 480** | 149 | 4.4 |
| tackle | 285 | 21 | **264** | 25 | 7.4 |
| block | 1 269 | 65 | **1 204** | 75 | 5.1 |

Train: 96 halves (48 matches), 91 327 raw → **82 759 live** events. Val: 6 halves, 6 070 raw
→ 5 456 live.

**The throw-in replay rate is 46.7% and it is real, not a join defect.** Review E1 flagged it
as suspicious — 47% on throw-ins against 0.4% on shots inverts the intuition that shots get
replayed most. It was checked: the `(half, frame, team, shirt, cls)` join produces **6 070
unique keys from 6 070 val rows, zero collisions and zero conflicting flags**, so no live
event is being overwritten by a replay row. The explanation is the opposite of the intuition
and is self-consistent: the flag marks events occurring *while the broadcast was showing a
replay*, and broadcasts cut to replays during **dead-ball periods** — which is exactly when
throw-ins are taken, and never when a shot is in progress.

This has a consequence §19 must carry: the replay filter removes **half of all throw-ins, and
does so systematically** — the surviving 49 val throw-ins are only those taken during live
coverage, which is a biased subsample, not a smaller random one.

### Split discipline

The xT grid is fitted on train and applied to val, never fitted on val. The 48 train games and
the 3 val games are **disjoint** (verified, empty intersection), so the split is clean **at
the match level**.

**It is not verifiable at the team level, and the first draft overclaimed this.** `PLAYER_ID`
runs 100–215 in both splits with `PLAYER_ID // 100 ∈ {1, 2}`; there is no cross-match club or
player key anywhere in the tactical h5. With 50 matches presumably drawn from one competition,
val clubs very likely also appear in train. For a global, team-agnostic xT surface this is
weak exposure, but the claim is narrowed to **match-disjoint**, and any per-club comparison
inherits the caveat.

### `PLAYER_ID` is match-local, not global

There are **only 32 distinct `PLAYER_ID` values across all 48 train games**, and the train/val
sets overlap 32/32. `PLAYER_ID` is a *within-match* key reused across matches. "Club 1" in
`game_18` and "club 1" in `game_24` are different clubs.

Tier 1 never hit this because it aggregated strictly per `(match_id, half)`. Tier 2 fits
across 96 halves, so **every cross-match aggregation keys on `(match_id, player_id)` or
`(match_id, club_id)`**, never on the bare id — a `dict[int, ...]` keyed on player id across
matches silently merges 48 different people into 32 buckets. xT *fitting* is unaffected (it
pools by location, not identity). A test pins the guard.

### Sample sizes that constrain the design

`game_18_H1`, live: 905 events, 132 chains, mean own-events per chain 6.5, median 3.5,
**14 live shots**, and **12 of 132 chains end in a shot**.

> Correction: the first draft said "7 of 132 chains, against 15 shot events". Both were wrong.
> 15 was the replay-inclusive shot count. The 7 came from testing `chain.events[-1]`, which
> per `Chain`'s own docstring **deliberately includes the opponent's contest events** — a
> chain ending `pass → shot → opponent block` has a block as its last event. The correct test
> is `own_events[-1]`, which gives 12. This is exactly the trap `Chain` documents, and the
> first draft fell in it while quoting the number to argue the sample was desperately thin. It
> is thin; it was overstated by 1.7×.

### What the ground truth does NOT contain

* **No goal labels anywhere.** Binding constraint on §12; propagates to §13 and to §20's saves.
* **No corner, free-kick, goal-kick or penalty class.** Throw-in is the only labelled restart.
* **No interception, challenge or foul class.** Tackle and block are the only defensive
  actions — the binding constraint on §17.
* **No shot outcome.** Post-shot xG is not computable.
* **No stoppage marker of any kind** (`chains.py` rule 3, verbatim: "the ground truth carries
  no stoppage marker at all"). Binding constraint on §14's start/end causes.
* **No ball position** independent of the acting player. Every location is a player location.
* **No labelled pass outcome or end point** — both are inferred by `chains.py` and stamped
  `outcome_source = INFERRED`. Binding constraint on §12; see §12b.

`ROLE` **is** present, stable within a half (verified: 0 multi-role players in `game_18_H1`),
`ROLE == 1` = goalkeeper, 2 per half. This is what makes §20 partially computable.

## Design commitments (inherited, non-negotiable)

* **An abstention is not a zero.** Every unmeasurable stat returns `None` with a reason string.
* **Ratios over counts.** Tier 1 measured it: ratios tolerate 20–40% event loss within 10%
  movement, counts only 5–10%, and chain-relational stats degrade ~3× faster under
  crowd-biased loss. Tier 2 is *more* chain-relational, so the sweep is extended to every
  Tier 2 stat under both loss models.
* **Every stat declares its contract** in a `Tier2StatSpec`, including what it cannot prove.
* **Source-agnostic.** `matchlab_core.stats` imports no dataset.

---

## Cross-cutting rule 0 — the frame-of-reference blocker

**This is the most dangerous defect either review found, and it is silent.**

Every event is attack-normalised **by its acting club** (`footpass_events.py`, keyed on pitch
side, then `zones.normalised_to_cm`). So an opponent's pass at `x = 20 m` and our pressing
tackle at `x = 20 m` are **at opposite ends of the pitch**. Any stat whose numerator and
denominator come from *different clubs* is silently comparing two mutually rotated frames, and
produces a number that looks entirely reasonable and means nothing.

Affected: **§17 PPDA** (opponent passes ÷ our defensive actions — the two halves of the
formula are in rotated frames), **§18** (a regain's location is in the regaining club's frame,
but the turnover *site* from the loser's perspective is the rotated point), and any part of
**§14** whose start cause reasons about where the opponent lost the ball.

**Not** affected: **§16 field tilt** — each club's final-third touch count is computed in that
club's own frame and only the *counts* are combined, never the coordinates. The plan states
this explicitly so a reader does not generalise §16's safety to §17, where it is false.

**Fix, mandatory before any cross-club stat is written:** add
`zones.to_opponent_frame(p, pitch) -> PitchPoint` implementing the 180° rotation
`(x, y) → (L − x, W − y)`. It does not exist in the tree today. Every cross-club stat calls it.
A test asserts that a tackle and the pass it dispossessed map to the same pitch point — under
a reflection-instead-of-rotation bug that test fails, which is the point.

## Cross-cutting evidence rules

The first review's summary finding: Tier 1's lessons had been applied to the stats where they
were *learned* and not to the stats where they *recur*. These rules bind every section.

### R1 — A rule-detected class needs a stratified null

Tier 1's take-on detector produced a rate a stratified null explained 57% of. **Every Tier 2
quantity detected by a threshold rule rather than read from a label carries the same
requirement**: §14 start cause, §14 attack type, §15, §18, §19 corners.

Citing a provider's threshold does **not** validate a classifier — providers fitted their
thresholds against their own labelled taxonomies, which this ground truth does not have. A rate
not separable from its null is reported as **failed**, as take-ons were.

### R2 — No rate is rendered below n = 10

Below a denominator of 10, emit the raw `(numerator, denominator)` pair and a `sample_starved`
flag, surfaced in the report. Binds §14 per-type conversion (0–3 per cell per half), §15,
§17 (0–2 per team-half), §18 (0–3 across the whole val split), §20's length buckets.

### R3 — Identities are declared, never reported as measurements

Tier 1 killed the ground-duel win rate for being 0.5 by construction. Three Tier 2 stats have
that shape:

* **§13 momentum is antisymmetric** — club B's series is exactly club A's negated. Every
  both-club aggregate is identically 0, and standard deviation, peak magnitude and
  zero-crossing count are one number reported twice. **One signed series only.**
* **§16 field tilt sums to 1** across clubs, so the two-club mean is 0.5 by construction.
  **One club's share only**; the other is `1 − x` and is not a second observation.
* **§14's overall `shots / chains` is invariant to the taxonomy.** Only the between-type
  contrast is informative; it is pooled across all 6 halves and carries a label-permutation
  null.

### R4 — Every reported correlation carries its trivial baseline

Any §12 correlation is reported alongside the same statistic for (i) a surface that is a
monotone function of distance-to-goal only, and (ii) per-player **action count** in place of xT
totals — a high-volume player accumulates xT roughly in proportion to touches. **The excess
over baseline is the result.**

### R5 — Signed stats need their own sweep metric

`sensitivity.py` reports `|value − baseline| / |baseline|` and skips exactly-zero baselines.
§13 and per-bin net `xt_delta` are signed and cross zero by construction, so the relative
metric explodes and the zero-skip silently drops bins — a sweep table whose missing rows read
as coverage. Signed quantities are swept on **absolute movement in stat units**; skipped bins
are counted and reported.

### R6 — Pre-registered exit criteria

The first draft contained six claims that could never fail. Each now has a threshold, fixed
**before** the numbers are seen:

| was unfalsifiable | pre-registered criterion |
|---|---|
| "a low correlation is a finding" | Spearman ρ < 0.6 on per-zone values between train halves ⇒ §12 is reported as **not transferring**, and §13 does not ship |
| "the characterisation digest carries weight" | a digest cannot fail on the commit that creates it. It is a **regression tripwire, not validation**; §12's only real check is the mutation run |
| "`unclassified` is a real outcome, not a dumping ground" | expected 25–60% of chains; outside that band is a **finding**, reported, not tuned |
| "§13 is a presentation choice" | §13 ships only if §12 passes its ρ ≥ 0.6 gate; otherwise it is withdrawn |
| "the difference is measured and reported" | per-player **rank correlation between the two arms**; ρ ≥ 0.95 ⇒ declared equivalent on this data and the fork reported as unresolvable |
| "the direction is reported, magnitude not invented" | §17 abstains at team-half granularity outright (see §17) |

### R7 — This branch owns its mutation run

Tier 1's measured lesson: structural tests caught 8 of 20 coefficient mutations. Delegating
mutation testing to reviewers is how that was *discovered*, not how it should be *prevented*.
This branch runs its own mutation suite against **a copy of the tree** (never the shared
worktree — a mutated coefficient reached a commit that way on Tier 1), with a stated target
survivor rate, and reports survivors.

---

## §12 — Expected Threat (xT)

### Model

Grid possession-value model. For each zone `z`: `s(z)` shot share, `m(z)` move share,
`g(z)` shot value, `T(z → z')` move destination distribution.

```
xT(z) = s(z)·g(z) + m(z)·Σ_z' T(z → z')·xT(z')
```

iterated from all-zeros to a max-delta tolerance of `1e-5` (socceraction's default `eps`).
Singh reports "4-5 iterations to be sufficient" but explicitly hedges it as dataset-dependent
and states no stopping rule, so the count is **not** hard-coded; tolerance is the rule and the
achieved count is recorded.

Grid **16 × 12**, verbatim from the source ("we're working with a 16x12 grid on the pitch,
which gives us 192 zones"), 16 along the length. Pitch orientation is **not** specified by the
source, so "attacks toward increasing x" is our convention, imposed by `zones.py`.

### §12a — The action set is stipulated, not natural

`s(z) + m(z) = 1` holds only over a **stipulated** denominator of
`{pass, cross, carry, throw-in} ∪ {shot}`. The first draft called this "by construction",
which reads as a property of the data. It is not.

Excluded, and the exclusions are material: **headers (3 480 live train events, 4.2%)**,
tackles (264), blocks (1 204). An aerial ball into the box is therefore **invisible to `T`** —
the model cannot represent the most valuable class of delivery in amateur football. Defensible
as a simplification, unacceptable as an unstated one.

`TAKE_ON` is in `schema.BALL_MOVING` but is a Tier 1 *derived*, `unvalidated=True` quantity.
**Decision: excluded from the fit**, because including it injects an unvalidated derivation
into the fitted grid. The divergence from `BALL_MOVING` is deliberate and the module names it.

Consequently `s(z)` is **"shot share of the stipulated action set"**, not "probability of
shooting from `z`", and the docstring must say so.

A zone whose only events are tackles has `0/0` — `s` and `m` are **undefined, not zero**. A
low-support reporting flag applied afterwards does not help: NaNs propagate silently through
value iteration. Such zones shrink `s(z)` toward the global rate, and the shrinkage is recorded.

### §12b — Failure handling: what is ours, what is cited

Two constructions:

* *success-only* — `T` rows built from completed moves, normalised to 1; failures dropped.
* *absorbing-failure* — every attempt contributes; failures go to an absorbing zero-value
  state, so rows sum to the zone's success rate.

**What the sources say.** Singh's blog carries, verbatim from
`https://karun.in/blog/expected-threat.html`:

> "Note: for the purposes of this simplified model, we consider only 'successful' moves, i.e.
> moves that were completed without possession being lost. You could, quite easily, consider
> all attempted moves as well, though at the cost of making your model slightly more complex."

That says failures are excluded from what is *counted*. It does **not** state `T`'s
denominator, so it does not by itself settle the row normalisation.

`socceraction`, the most widely used open implementation, does settle its own behaviour — in
code, fetched and read directly at
`https://raw.githubusercontent.com/ML-KULeuven/socceraction/master/socceraction/xthreat.py`:

```python
move_actions = get_move_actions(actions)      # NOT filtered on result
start_counts[vc.index] = vc                   # denominator: ALL attempts from cell i
transition_matrix[i, vc2.index] = vc2 / start_counts[i]   # numerator: successes only
```

Row `i` sums to the **success rate**, not to 1; `action_prob` likewise uses all attempts. So
socceraction implements absorbing-failure, and `get_successful_move_actions` is used only for
*rating*, not fitting.

**Labelling decision.** Singh's stated simplification is success-only, so the §12 headline
does **not** call the default "Karun Singh's formulation" while defaulting to a variant he
excludes. The default is a *possession-value variant*; success-only is the **Singh arm**;
absorbing-failure is the **socceraction arm**. Both are implemented; the difference is measured
against R6's ρ ≥ 0.95 criterion.

> Provenance note, recorded because it is the exact failure Tier 1 was burned by. Two research
> agents were briefed independently on this question and returned **contradictory** answers.
> One reported socceraction "agrees" that rows sum to 1 over successes — it had read only the
> *documentation page*, not the source. The other quoted the code. The code was then read
> first-hand to settle it. Do not cite socceraction's docs page as corroboration here; and note
> that socceraction cites Singh directly, so even where it agrees it is a reimplementation of
> the same method, not independent evidence. A third source, one cold reviewer, read the blog
> alone and concluded "the reference is success-only" — correct about the blog, silent about
> the code. All three readings are in this paragraph because no one of them is sufficient.

**The convergence argument is withdrawn.** The first draft claimed absorbing-failure
"*guarantees* the value iteration is a contraction" and success-only does not. Both halves are
wrong:

* Absorbing-failure guarantees a contraction only if *every* zone has strictly positive
  observed failure probability. A low-support zone with 2 observed moves, both completed, has
  `m(z) = 1` and zero leakage. This is an **empirical property of the fitted counts, asserted
  at fit time**, not a construction guarantee.
* Success-only does not lose *convergence*. With `M = diag(m)·T`, value iteration from
  `xT ≡ 0` with non-negative rewards is monotone non-decreasing and bounded above by `max g`,
  so it converges to the minimal non-negative fixed point — which on a closed class of
  zero-shot zones is 0, the correct answer. Singh's 4–5 iterations confirm it is a non-issue.

**The fork is semantic, not numerical**, and that is how it is stated: success-only estimates
*P(score within n actions | possession survives the moves)*; absorbing-failure estimates
*P(score before losing possession)*. Different quantities, both defensible.

**And the fork is over an inferred bit.** FOOTPASS labels no pass outcome and no end point;
both come from `chains.py` stamped `INFERRED`. The two arms differ *only* in how they treat
that heuristic, and the same heuristic supplies `T`'s destinations. So the ablation runs over
the **inferencer's** parameters too — at minimum `DEFAULT_MAX_GAP_S = 10.0`. **If the two arms
differ by less than the spread induced by moving the gap threshold, the honest statement is
that this data cannot resolve the fork**, and that is what the report will say.

`UNKNOWN` outcomes are **excluded from `T`**, not folded into failure — an abstention is not a
failure; counting it either way biases every zone. The dropped count is recorded.

**Measured caveat on carries.** On `game_10_H1`, inferred carry completion is 316/322 (98%),
against 400/482 (83%) for passes and 2/11 for crosses. Carry outcome is inferred from whether
the same club has the next possession-defining event, and a carry is usually followed by the
same player's own pass. So carries contribute almost no leakage, and the absorbing discount is
carried almost entirely by passes and crosses. This is a property of the *outcome inference*,
not of football, and the two arms' similarity must not be read as robustness.

### §12c — `g(z)` is a geometric prior, not an estimate

The first draft described `g(z)` as "the mean `xg()` over shots observed in `z`, with a
smoothed fallback". That description is misleading and the review is right to reject it.

On FOOTPASS, `xg()` is a **deterministic function of shot location alone**: `_header_flag`
returns `None` on this source so `is_header` defaults False, `is_set_piece_origin` defaults
False, and `defenders_in_lane` is `None` unless off-ball context is read (which §12 must not
read — see §12f). So "mean `xg()` over shots in `z`" is just `xg()` at the zone's shot
centroid; and **139 of 192 zones contain zero shots**, so a fallback would fire for ~72% of the
grid.

**`g(z)` therefore carries no information from the data at all. It is a smooth geometric
prior.** It is implemented as exactly that — `xg()` evaluated at the zone centroid, honest and
simpler — and described as that everywhere.

**What survives the substitution, stated as the theorem it is.** In matrix form
`xT = (I − diag(m)T)⁻¹ diag(s) g`, so **`xT` is *linear* in `g`**. Therefore:

* **Survives:** any statement invariant to a positive *scalar* multiple of `g`. The known xG
  failure mode is "professional coefficients overstate amateur conversion by an unknown
  factor" — a uniform scale error leaves every zone ranking, every `xt_delta` ranking and every
  player ranking **exactly** unchanged. This is a stronger position than the first draft
  claimed.
* **Does not survive:** any *shape* error in `g` — location-dependent miscalibration, which is
  precisely what transferring a fixed geometry model to a different shot distribution produces.
  Nothing about xT is robust to that.

**The percentile defence is withdrawn.** `percentile_within` rescues xG because xG is one value
per shot and percentile is invariant under *any* monotone distortion. A player's xT total is
`Σ (xT(end) − xT(start))` — a sum of **differences**, and monotone transformation does not
commute with differencing and summing. Reusing Tier 1's "report as a percentile" sentence here
imports a guarantee that does not exist.

**The ablation is retargeted.** "xG model vs flat constant `g`" only tests whether geometry
matters at all, which it trivially does. The test that matters is a **shape perturbation**:
re-fit with the distance/angle coefficients perturbed within a plausible band and report how
far per-player xT ranks move. That directly bounds exposure to the only error class that is
not scale-invariant.

### §12d — Per-action credit, consistent with the chosen model

`xt_delta(action) = xT(zone(end)) − xT(zone(start))` for **completed** moves.

The first draft credited failures `0.0` while arguing for absorbing-failure *because* "a
possession that ends in a turnover is worth zero from that point" — a direct
self-contradiction the review caught. With non-negative credit, a player accrues credit for
risky passes that come off and no debit for those that don't, so **volume dominates quality**
in every player total.

Also, the stated mitigation ("the count of failures is carried alongside so a consumer can
compute either convention") does not work: reconstructing `−Σ xT(start)` needs the failed
moves' **start zones**, not a count.

**Fix:** carry `failed_xt_at_start` — the summed `xT(start)` over failed moves — so both
conventions are computable exactly, and report **both** the non-negative total (socceraction's
rating behaviour, which returns `NaN` for failures) and the risk-adjusted total
`Σ_success Δ − Σ_fail xT(start)`. Neither is privileged as *the* number.

Shots are credited on the `s`-side separately and not double-counted as moves. Moves with no
reconstructed end point are excluded from the numerator **and counted**, so credit coverage is
visible.

### §12e — Highlight ranking (Notion's third surface)

Top-N `xt_delta` events per player per match, which Notion names as a reason to build xT
early. One function over the credit output.

**It is the surface most damaged by §12d**: under non-negative credit the top-N is a list of
successful long forward passes and nothing else. It therefore ships against the
**risk-adjusted** credit, and the report shows both rankings side by side so the difference is
visible rather than asserted.

### §12f — Tier 3 leakage guard

`xg()` internally calls `defenders_in_lane(p, event.opponents, pitch)`. If §12 passes events
carrying off-ball context, `g(z)` **silently becomes a Tier 3 quantity** and the fitted grid is
no longer reproducible from event data alone. This would have been discovered late and
invalidated the fit.

**Pin: §12 evaluates `xg()` on off-ball-stripped events only.** A test asserts the fitted grid
is bit-identical whether or not the source events carry `teammates` / `opponents`.

### §12g — What `T` actually estimates

`_fill_end_points` reconstructs a pass's end as **the next actor's position at the next event's
frame** (up to 10 s later), and a carry's end as the same player's position at that frame.
These are reconstructions, correctly flagged in Tier 1 — but §12 *fits a model* on them. So
`T(z→z')` is **not "where the ball arrived"**; it is "where the receiver was when they next did
something", which is systematically further forward. Tier 1's progressive stats carry this
bias; xT compounds it and re-exports it as per-action credit. An ablation over `max_gap_s` for
the destination reconstruction specifically is required.

Related correction: the first draft argued sparsity from "~950 events per half against
~1 600–2 000 in a commercial feed". That compares unlike vocabularies — FOOTPASS is 43%
`drive`/carry, which commercial feeds largely do not label. **Live passes are 41 155/96 ≈ 429
per half**, comparable to a commercial feed's pass density. The sparsity argument was
overstated and is withdrawn in that form.

### §12h — Resolution: the constraint is shots, and coarsening does not relieve it

The first draft argued for coarsening from "77 000 moves / 192² ≈ 2 per cell". The arithmetic
is right and **the statistic is the wrong one** — `T` is estimated **row-wise**, and
destinations are highly local rather than uniform over 192 cells. Measured on live train events
(76 711 moves, 1 093 shots), attack-normalised:

| grid | zones | zones with 0 shots | median shots/zone | median moves/zone |
|---|---|---|---|---|
| 16×12 | 192 | **139** | 0 | 427 |
| 12×8 | 96 | 63 | 0 | 894 |
| 8×6 | 48 | 27 | 0 | 1 733 |
| 6×4 | 24 | **12** | 0 | 3 432 |

Two conclusions the first draft had backwards:

* **Moves are not the constraint.** 427 per origin zone at Singh's own 16×12 is ample.
* **Shots are, and coarsening does not fix it**, because shots are spatially *concentrated*,
  not uniformly sparse. Even at 6×4, **half the pitch has exactly zero shots**, so `s(z) = 0`
  identically over most of the defensive half at *every* resolution. The proposed
  "resolution ablation" cannot answer the question it was posed.

**Decision: decouple the three resolutions.** `T` at 16×12; `s(z)` as a heavily smoothed
low-order model in `x` (the data supports roughly a 4–6 band model, not a 192-cell one); `g(z)`
not estimated from data at all (§12c). Keep 16×12 as the reporting grid.

### §12i — Validation

1. **Structural** — monotone increase toward goal along the centre channel; bounded in
   `[0, max g]`; convergence delta below tolerance; positive leakage asserted at fit time.
2. **Asymmetry, not symmetry.** The first draft proposed "symmetry about the long axis under
   y-reflection". A mirror-instead-of-rotation bug produces a surface y-flipped **for one club
   only**; pooled over both clubs that is invisible to a y-symmetry test, and asserting
   y-symmetry as a pass criterion means **the buggy build passes**. The test is not merely weak,
   it is anti-correlated with the failure mode. Replaced with: fit per-club grids and assert any
   left/right asymmetry has the **same sign** for both clubs — under the bug it flips for one.
3. **Stability** — train-half vs train-half at **matched n** (not train-vs-val: val is 5 456
   live events against a 192-cell matrix, so a low train-vs-val correlation would measure val's
   sample size, not the surface). Val reported separately with a bootstrap CI. Gated on R6's
   ρ ≥ 0.6. Every correlation carries R4's two baselines.
4. **Ablations** — Singh arm vs socceraction arm (gated on the inferencer sweep, §12b);
   `g` shape perturbation (§12c); `max_gap_s` destination reconstruction (§12g).
5. **Mutation run, owned by this branch** (R7). Per R6 this is §12's *only* real validation:
   the characterisation digest is a regression tripwire that cannot fail on the commit creating
   it, and Tier 1 measured structural tests catching 8/20 mutations.

## §13 — Team momentum chart

Per R3, **one signed series**, positive toward `momentum.club_id`; both-club aggregates
forbidden.

### What is published, and what is ours

**No canonical cross-provider definition exists** — both researchers converged on that
independently. Two fully specified published implementations exist and they disagree on the
value model:

* **Opta / Stats Perform "Match Momentum"** (`theanalyst.com/articles/what-is-match-momentum`),
  built on their *possession value* model — "the likelihood of the team in possession scoring
  within the next 10 seconds". Aggregation, verbatim: "We look at the maximum possession value
  for each team in every minute of the game so far (capped between zero and 0.1)." Smoothing:
  "weighted for how recently they occurred… Only the most recent three to four minutes have
  significant impact here." Series: "the difference between these values for each team". **The
  kernel is not disclosed.**
* An open xT-based reimplementation (Kapich) uses a 4-minute window, exponential decay
  `0.25`, `clip(xT, 0, 0.1)`, per-minute max, Gaussian post-smooth `sigma = 1`. It explicitly
  derives from Opta's approach, so it corroborates the *structure* only because it copied it —
  not independent evidence, exactly as socceraction is not independent of Singh.

**Adopted structure** (the two implementations' shared shape, cited): per-minute bin, **maximum
per club per bin** rather than sum, clipped to `[0, 0.1]`, recency-weighted over a ~3–4 minute
effective window, series = inter-club difference.

**Declared as ours:** the kernel family and half-life, and the use of xT in place of Opta's
undisclosed possession-value model.

### The time axis, which the first draft left undefined

`MatchEvent.t = frame_idx / 25` and **`frame_idx` is per-half**, so a chart concatenating H1
and H2 overlays the two halves. An explicit half offset is required, and the kernel must be
**re-truncated at the half boundary** — otherwise H2's opening minutes borrow from H1's closing
minutes. The first draft specified edge renormalisation carefully and then left the axis itself
undefined.

### Edge handling

A kernel truncated at the start of a half must be **renormalised over its available support**,
or the opening minutes are damped toward zero and every chart shows a slow start that is a
rendering artefact. Test: a constant input must produce a constant output **including at
boundary bins** — and at the half boundary, not only at the series start.

Per R6, §13 ships only if §12 passes its stability gate.

## §14 — Possession sequences and attack-type classification

Per chain: a start cause, an end cause, and exactly one type label.

### Start and end causes must abstain, not invent

`chains.py` rule 3 is explicit that the ground truth carries **no stoppage marker at all**; the
chain builder's only non-club-change split is `t_gap > 10 s`. So a `dead_ball` end cause and a
`restart` start cause would be, apart from labelled throw-ins, **relabelled time gaps** —
indistinguishable from a camera cut, a replay boundary, or untracked play. That converts an
abstention into a named cause, which is exactly what `xg._is_set_piece_origin` was rewritten to
avoid.

**Decision:** the vocabulary is `regain` (live possession change), `restart` (**labelled
throw-in only**), `gap` (a time split, explicitly not a cause), `half_start`, `unknown`; and
end causes `shot`, `turnover`, `gap`, `half_end`, `stream_end`. No `dead_ball` category exists.

### Type definitions

No provider defines a counter-attack by a **time** threshold — verified independently by both
researchers. The first draft's "reaches a shot within a time threshold" was unsourced and is
removed.

| type | definition | source |
|---|---|---|
| `build_up` | open-play sequence with **10+ passes**, ending in a shot or a touch in the opposition box | Opta, verbatim: "open-play sequences that contain 10+ passes and either end in a shot or have at least one touch in the opposition's box" |
| `direct_attack` | starts just inside own half, **≥50% of movement towards the opposition's goal**, ends in a shot or box touch | Opta, verbatim; "just inside the team's own half" is **not quantified in metres by the source** — that value is ours |
| `counter_attack` | open-play turnover **outside the counter-attacking team's own final third**, **≥75% direct towards goal**, travelling **≥18 yards** toward goal | StatsBomb Open Data Events v4.0.0 spec, play_pattern 6 — fetched and read first-hand |
| `high_turnover` | possession starting in open play **≤40 m from the opponent's goal** | Opta, verbatim; the Notion source doc independently states "~40 m" |
| `set_piece` | **abstains on FOOTPASS** | — |
| `unclassified` | none of the above | — |

**Provider disagreement is recorded, not averaged.** StatsBomb requires ≥75% directness for a
*counter-attack*; Opta requires ≥50% for a *direct attack*. Different metrics from different
taxonomies, not interchangeable; origin zones differ too. StatsBomb's is used for
`counter_attack` (the only fully numeric definition of that class), Opta's for `direct_attack`
and `build_up`, and **the source is stamped on each classified chain** so no consumer can mix
them unknowingly.

**Precedence, explicit:** `high_turnover` → `counter_attack` → `direct_attack` → `build_up` →
`unclassified`. Exactly one label per chain; per R6, `unclassified` is expected at 25–60%.

**`set_piece` abstains** to stay consistent with Tier 1, which made `_is_set_piece_origin`
abstain for want of a class. A §14 `set_piece` type would reintroduce the claim Tier 1 removed
and the two would then disagree.

**Pass counts are computed over `Chain.own_events`, not `Chain.events`** — `Chain`'s docstring
warns that `events` deliberately contains the opponent's contest events. Two sentences of plan,
one line of code that is easy to get wrong, and the same trap this document already fell into
once (see the 7-vs-12 correction above).

### Validation

Per R1 the taxonomy has no labelled ground truth: its null is a **label permutation across
chains**, and a **threshold-sweep table** is reported showing type counts as a function of each
threshold, so a reader can see whether the taxonomy is a measurement or a knob. Per R3 the
headline is the between-type contrast pooled across all 6 halves — the overall conversion rate
is invariant to the classification. Per R2, per-type cells are raw pairs.

## §15 — Counter-attacks

A **predicate over the §14-classified chain**, not a second parallel pass, so the two cannot
disagree. Thresholds are StatsBomb's, as tabulated.

Per R1 it needs a duration- and start-location-matched null. Per R2 its counts are expected at
0–2 per half, so it is reported as a whole-split count with the number of halves attached and
**no per-half rate**.

The Notion doc's rationale — amateur football is largely a transition game — is a product
argument and does not lower the evidence bar.

## §16 — Field tilt

**No provider definition could be verified.** Both researchers failed independently: StatsBomb
via Hudl returned an empty error page, `statsbomb.com` redirects off-host, and Opta's glossary
does not define the term. The one source opened (Driblab) is internally inconsistent within a
single sentence — "touches" then "touches and passes". The widely repeated "completed
final-third passes share" formula appeared **only in search snippets** and is attributed to
nobody here.

**Definition declared as ours**, following the Notion doc's own wording:

```
field_tilt(club) = club final-third touches / (both clubs' final-third touches)
```

**Why this is safe from rule 0** (and §17 is not): each club's final-third touch count is
computed in **that club's own attack-normalised frame**, and only the resulting *counts* are
combined — never the coordinates. It is the only cross-club Tier 2 stat with that property.

Per R3, one club's share only. The first draft called this "the most robust stat in Tier 2";
**withdrawn** — part of the apparent robustness is the sums-to-1 identity, and an
unbiased-loss cancellation argument is trivially satisfied by a quantity pinned to 0.5 in
aggregate.

Tier 1's swept `ratio_field_tilt_box_share` is `box_touches / len(events)`, **not** a two-club
share, so that sweep result does not transfer; the sweep must target the per-club share.

Per R6 the test with teeth is Tier 1's club-level attack-normalisation test — a
reflection-instead-of-rotation bug mirrors the wings and is otherwise undetectable. Tier 1's
side-flip test is vacuous here because sides never flip.

## §17 — PPDA — **abstains at team-half granularity**

Both reviews independently measured the denominator and it is fatal to a per-half number.
Defensive actions (tackle ∪ block) inside the PPDA zone, **both clubs pooled**, per val half:

| half | tackles raw | tackle+block in zone |
|---|---|---|
| game_18_H1 | 6 | 4 |
| game_18_H2 | 4 | 4 |
| game_24_H1 | 3 | 4 |
| game_24_H2 | **0** | **0** |
| game_47_H1 | 6 | 6 |
| game_47_H2 | 7 | 7 |

**25 in-zone defensive actions across the entire val split** — ~4 per half for both clubs,
**~2 per team-half**, and `game_24_H2` is a **division by zero**. The first draft quoted the
raw class counts (285/26) and mitigated by "printing the denominator"; the in-zone count is the
one that matters and it is an order of magnitude worse.

The mechanism is structural, not bad luck: **a block occurs near the shooter's target goal,
i.e. structurally outside the PPDA zone**, so the block class barely contributes and the metric
rests on ~4 tackles per half.

**Decisions:**

* PPDA returns `None` — never `inf`, never `0` — at zero denominator, and abstains outright at
  **team-half granularity**.
* **No per-half PPDA number appears anywhere in the report.** It is attempted only pooled over
  the split, with the in-zone denominator stated, if at all.
* **Excluded from the headline set.**
* Because the metric is undefined at the reporting granularity, the declared-bias statement
  below is the *only* content §17 carries, and the report says so.

### Definition and provider disagreement

Trainor / StatsBomb (primary, fetched): numerator = opposition passes beyond the `x = 40` line;
denominator = "Tackles, Interceptions, Challenges (failed tackles), Fouls" beyond that line.
`x = 40` is an Opta 0–100 pitch-length coordinate — **"40% of the pitch" is arithmetic on that
coordinate system, not the article's wording** — i.e. 42 m from the pressing team's own goal on
this 105 m pitch.

Opta's present-day definition differs: outside the pressing team's own **defensive third**
(≈35 m), not 42 m. **Different numbers on the same match.** Trainor's is the default (original,
explicit action list); Opta's zone is a flag; the choice is stamped and the two are never mixed.

The "x = 40 explains >90% of variation" line circulating in search snippets was **not found in
the article text** by the researcher who fetched it, and is not cited.

### Frame of reference

Per rule 0 this is the stat most exposed. The zone is defined relative to the **pressing** team
while events are stamped in the **acting** club's frame, so "beyond x=40 for the presser" is
`x_opp < 0.60 × length` in the opponent's own frame. Implemented via `to_opponent_frame`, with
a dedicated test.

### Declared bias

Available defensive actions are **tackle and block only**; interceptions, challenges and fouls
have no class. The denominator is an under-count, so **PPDA is biased high** — it reads as
*less* pressing than reality. The magnitude is not estimable here and is not invented.

## §18 — High turnovers

Opta, verbatim: "The number of possessions that start in open play and begin 40 metres or less
from the opponent's goal", and "The number of shot-ending or goal-ending sequences that begin
with a high turnover".

**There is no time window; the first draft's "within M seconds" is removed as unsourced.** Both
researchers confirmed independently that Opta's construction is *structural* — the sequence
itself must end in a shot. The 5-second threshold that circulates belongs to StatsBomb's
**counterpress**, a defensive-action attribute, and must not be imported.

Ambiguities the source does not settle, handled explicitly:

* "40 metres from the opponent's goal" does not disambiguate radius-to-goal-centre from an
  x-line. Implemented as **radial distance to the goal centre** (consistent with
  `zones.distance_to_goal_cm`), with the x-line variant as a reported ablation.
* Opta defines it on **possessions**, which it distinguishes from sequences. `chains.py`
  produces sequence-like chains, so numbers will differ from Opta's by construction. Stated.

Per rule 0, a regain's location is in the regaining club's frame; where the *losing* club's
perspective is needed, `to_opponent_frame` is used. Per R1 the null preserves the
regain-location distribution and shuffles the shot-follows indicator. Per R2 the shot-ending
subset is expected at 0–3 across the whole val split and is reported as a raw count.

## §19 — Set-piece breakdown

**Mostly abstains, and the plan says so up front.**

The schema gains `CORNER`, `FREE_KICK`, `GOAL_KICK`, `PENALTY` alongside `THROW_IN`, in the
same "reserved for a source that has it" spirit as Tier 1's `TAKE_ON` / `FOUL_WON`. The
breakdown is implemented once, source-agnostically: restarts taken, shots and xG from set
pieces vs open play, per-taker delivery success.

On FOOTPASS:

* **Throw-ins are labelled** — but only **922 train / 49 val live**, not the 1 730 / 97 the
  first draft quoted. **46.7% of throw-in labels are replays**, and per the substrate section
  that loss is *systematic* (broadcasts cut to replays during dead-ball periods), so the
  survivors are a **biased subsample**, not a smaller random one. This is the one set-piece
  class that is real and it loses half its sample.
* **Corners, free kicks, goal kicks and penalties have no class** — `None` with a reason.

**Corner detector, strictly separate from the measured part.** A cross starting within a small
radius of a corner flag, following a time gap. Motivated by observed data: crosses in
`game_18_H1` start at (105.3, 66.3), (104.6, 66.1), (103.6, 1.2), (102.4, 68.2) m — goal line
at the touchline. **But the same half's 21 live crosses also include (101.5, 14.0) and
(102.3, 56.5), which are box positions, not corner arcs**, so the radius must be tight enough
to exclude those on a base of ~21 crosses per half.

Per R1 it ships behind a flag, out of headline tables, with a stratified null (a
location-shuffled control preserving the cross-location distribution). **Pre-registered per
R6: at ~21 crosses per half the null test is likely to be underpowered, and "underpowered,
therefore not reported" is an expected outcome, not a surprise** — it is recorded now so it
cannot later be presented as a finding either way.

## §20 — Goalkeeper metrics

`ROLE == 1`, stable within a half, 2 per half. Measured live on `game_18_H1`: the two keepers
have **34 passes and 24 carries** between them — not the 42/27 the first draft quoted, which
were replay-inclusive.

Computed:

* **Distribution completion by length** — GK passes bucketed short/medium/long with the Tier 1
  inferred outcome. Per R2 the long bucket is single-digit per half and is emitted as a raw
  pair.
* **Shots faced and xG faced** — shots by the opposing club, attributed to the keeper defending
  the goal being attacked, via `to_opponent_frame` per rule 0.

**The length buckets are ours, and this is a verification gap rather than a choice.** FBref
returned **HTTP 403** to both researchers on both the glossary and the stats pages, and
archive.org was unreachable, so the widely repeated Short 5–15 yd / Medium 15–30 yd /
Long >30 yd thresholds **could not be attached to any source** and are not cited. The StatsBomb
open-data spec records pass length in yards but defines no buckets. Round metric buckets are
used, declared as ours, and exposed as a parameter.

Abstains, with reasons:

* **Saves, saves vs xG faced** — requires shot outcome. No class. `None`.
* **Post-shot xG** — requires the on-goal contact point; its definition also could not be
  obtained from any primary source (FBref 403). `None`.
* **Claims and punches** — no class. `None`.

Keeper-of-record attribution has **no labelled ball position to check against** — every
location in this data is a player location — so it is stated as an inference, not a
measurement.

---

## Validation power (R6) — the test that would catch a wrong implementation

Written out because the plan must not imply validation by proximity to a review protocol.

| stat | test that fails if the implementation is subtly wrong |
|---|---|
| §12 xT | **Mutation run only** (R7). The digest catches change, not error; structural tests caught 8/20 on Tier 1's xG; the y-symmetry test was *anti-correlated* with the mirror bug and is replaced by the per-club asymmetry-sign test. |
| §13 momentum | Edge-renormalisation test at series start **and half boundary**; sign-convention test. Wrong half-life or kernel family: **none**. |
| §14 taxonomy | Threshold-sweep table + label-permutation null. "Exactly one label" passes under any wrong rule and is not validation. |
| §15 | Duration/start-matched null. Subset-consistency with §14 is coherence between two things that can be wrong together — **not** validity. |
| §16 field tilt | Tier 1's club-level attack-normalisation test (catches reflection-vs-rotation). |
| §17 PPDA | `to_opponent_frame` round-trip test + zero-denominator abstention test. Beyond that **none** — and the stat abstains. |
| §18 | Radial-vs-x-line ablation + null. |
| §19 throw-ins | Labelled: a live-count test against the h5 bites. Corner detector: **null baseline only, pre-registered as likely underpowered**. |
| §20 | Bucket thresholds: **none** — ours and unvalidatable. Keeper attribution: **none** — no labelled ball position. |

## Known exposures, stated rather than discovered later

* **`FIFA_PITCH` is a fixed 105 × 68 spec** and all coordinates are `[0,1]`-normalised onto it.
  Fine for Tier 1's within-match ratios; for a **grid fitted across 48 matches on
  differently-sized real pitches**, zone boundaries do not correspond to the same physical
  locations across matches. Probably second-order, but it is a fitting-specific exposure Tier 1
  did not have, and it is unmeasurable from this data.
* **Headers are excluded from `T`** (§12a) — 3 480 live train events invisible to the model.
* **`T` estimates receiver-next-action position, not ball arrival** (§12g).
* **`g(z)` carries no data information** (§12c).

## Replay-filter regression on the train join

Tier 1 found the replay flag lives only in `playbyplay_*.json`, that the natural h5-vs-JSON
cross-check passes perfectly while the numbers are wrong, and that the test with teeth toggles
the filter and asserts a downstream count moves. The train join is now load-bearing for the
fitted grid, so a **train-side filter-toggle test** is required, asserting movement in a
**composition** statistic of the fitted grid (per-zone move counts), not a headline scalar —
Tier 1 showed an unchanged headline is not evidence the filter did nothing.

## Deliverables

* `matchlab_core/stats/xt.py` — grid model, fitting, value iteration, per-action credit,
  highlight ranking.
* `matchlab_core/stats/momentum.py` — §13.
* `matchlab_core/stats/sequences.py` — §14, §15, §18 chain taxonomy.
* `matchlab_core/stats/team.py` — §16, §17, team-level folding.
* `matchlab_core/stats/setpieces.py` — §19.
* `matchlab_core/stats/keeper.py` — §20.
* `matchlab_core/stats/zones.py` — **additive: `to_opponent_frame`** (rule 0).
* `matchlab_core/stats/schema.py` — additive: restart types, `Tier2StatLine`, `Tier2TeamLine`,
  `Tier2StatSheet`, `Tier2StatSpec`.
* `matchlab_core/stats/compute.py` — `compute_tier2`, `TIER2_STAT_REGISTRY`.
* `matchlab_core/stats/sensitivity.py` — Tier 2 metrics, with R5's signed-stat metric.
* `matchlab_train/experiments/tier2_stats.py` — fit on train, evaluate on val, ablations.
* Tests, **structural and characterisation separated by file** per the Tier 1 review finding.
* `docs/reports/YYYY-MM-DD-peripheral-stats-tier-2.md`.

## Review protocol

Cold reviewers are briefed to (a) **fetch every cited source and verify the number digit by
digit**, and check that claimed corroboration is genuinely independent — different authors,
different sample, not a re-post or a reimplementation; and (b) **mutate the implementation
against a copy of the tree** and report survivors. Mutation testing never runs against the
shared worktree. Per R7 this branch also runs its own mutation suite rather than outsourcing it.

## Explicit non-goals

* No calibration claim of any kind for xT or xG. There is nothing here to calibrate against.
* No per-90 or per-match extrapolation from partial coverage.
* No Tier 3 or Tier 4 stat, including by accident — see §12f, which pins the one path by which
  Tier 3 data would otherwise have leaked into the fitted grid.
