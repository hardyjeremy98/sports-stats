# Peripheral Statistics, Tier 1 — implementation and ground-truth results

Branch `worktree-peripheral-stats-tier-1`. Plan: [`docs/prds/peripheral-stats-tier-1.md`](../prds/peripheral-stats-tier-1.md).
Source of the stat list: Notion "📊 Peripheral Statistics" (curated 2026-08-05).

**Ground truth only.** No detector, tracker or spotter output is consumed anywhere in this
branch. Every number below comes from FOOTPASS labels. That is the point: if a stat is wrong
here, it is wrong in its definition or its chain logic, and perception cannot be blamed.

## What was built

All eleven Tier 1 stats, as a source-agnostic library over a canonical event stream
(`matchlab_core.stats`), plus the three cross-cutting requirements the source doc mandates:
coverage denominators on every stat line, a per-stat abstention declaration in code, and the
recall-sensitivity sweep it calls "probably the first concrete task of the phase".

| Module | Contents |
|---|---|
| `stats/schema.py` | `MatchEvent`, outcome + provenance enums, `Tier1StatLine/Sheet`, `StatSpec` registry types |
| `stats/zones.py` | thirds, box, goal angle/distance, pressure, attack normalisation |
| `stats/chains.py` | possession chains, receiver and outcome inference |
| `stats/xg.py` | stat 1 |
| `stats/creation.py` | stats 2–4 |
| `stats/progression.py` | stats 5–7 |
| `stats/duels.py` | stats 8–10 |
| `stats/passing.py` | stat 11 |
| `stats/compute.py` | orchestrator, coverage, stat registry |
| `stats/sensitivity.py` | recall-sensitivity sweep |
| `matchlab_train/datasets/footpass_events.py` | FOOTPASS GT → `MatchEvent` adapter |
| `matchlab_train/experiments/tier1_stats.py` | end-to-end runner, writes `data/reports/tier1-stats/` |

## The four findings that matter

### 1. The ground-truth adapter has to read two files, and the obvious test cannot tell you

**10.1% of FOOTPASS val events (614/6070) are broadcast replays** that duplicate live play,
and the `replay` flag exists **only** in `playbyplay_val.json` — never in the tactical HDF5.
An event-derived statistic built from the h5 alone silently double-counts them.

The trap is the verification, not the bug. The two streams are otherwise *identical* —
verified exact on the full `(frame, team, shirt, class)` tuple set for all six halves, zero
symmetric difference — so the natural cross-check ("does the h5 agree with the JSON?") passes
perfectly while the numbers are wrong. The tests that bite instead toggle the filter and
assert the downstream count moves.

A second trap sits behind the first: on `game_18_H1` the chain **count** is 132 with or
without the filter, while the composition differs (possession changes 93 → 89, gap splits
38 → 42). An unchanged headline number is not evidence that a filter did nothing.

### 2. `TEAM` is pitch side, not club — and the natural regression test is vacuous

`TEAM` is 0-on-the-left in every val half. What flips at the interval is the **club↔side
binding**: 18/18, 21/21 and 21/21 of the players present in both halves take the opposite
`TEAM` value in H2. Aggregating on `(TEAM, SHIRT)` therefore merges the two clubs across
halves, and shirt numbers additionally collide within a half (7 collisions in `game_24_H2`).
The stable keys are `PLAYER_ID` and club = `PLAYER_ID // 100`.

The originally planned test — "H1 and H2 produce the same attacking direction" — passes
whether or not the code is right, because sides never flip. The test with teeth asserts the
same **club** maps to opposite raw sides across halves *and* that its shots still concentrate
at `+x` after normalisation.

Attack normalisation is a **180° rotation** (`x → L−x` *and* `y → W−y`). The `1−x` mirror in
the upstream convention is a reflection that swaps the left and right wings — every flank
stat would come out mirrored for one club, undetectably.

### 3. An opponent's block is not a turnover

Chain segmentation originally started a new possession on any club change, including an
opponent's block, tackle or challenging header. That turned a single deflection into a
turnover *and* a recovery, and split the pass and the shot that followed it into different
chains, hiding real key passes.

Treating an opposing contest event as an interruption *inside* the possession moved
`game_18_H1` from 168 chains / 125 possession changes to **132 chains / 89 possession
changes** — 29% of "possession changes" were deflections. It also recovered one key pass and
three shot-creating actions in that half alone, and it exposed a live misattribution bug: the
turnover had been credited to the defender who made the block rather than the player who lost
the ball.

### 4. The take-on detector mostly measures how far the player ran

Stat 8 has no ground-truth class **and no negative class**, so it cannot be scored — which is
exactly why it needed a null baseline rather than a plausible-looking rate.

The mechanism was a scale mismatch: the proximity test used a 3 m radius while the "beaten
the defender" test only asked whether the opponent was behind the carry's *end* point, so any
carry longer than 3 m satisfied it automatically — 62.5% of detections, and 54 of 160 beat
nobody who had ever been in front. The number looked like a measurement and was substantially
a restatement of "did the player move forward".

Requiring the defender to be ahead at the start *and* behind at the end
(`start.x < opponent.x < end.x`) removes the vacuous regime. Null baselines, 20 trials,
strata = start third × x-gain band:

| | detections | rate/carry | unstratified null | matched null | residual |
|---|---|---|---|---|---|
| before | 160 | 0.0715 | 0.0306 | 0.0465 | 0.0250 (35%) |
| after | 70 | 0.0313 | 0.0116 | 0.0177 | 0.0136 (43%) |

**The matched null still explains 57% of the rate.** The detector is no longer *dominated* by
carry length; it is still substantially nuisance, and no available ground truth can move it
past "inspectable". `take_on_null_rates()` ships with the module so the finding is
reproducible rather than a claim in a report, and a test asserts the matched null stays below
the observed rate — deliberately a weak bound, because a tight one would be a lie about this
stat. Stat 8 should not be reported to anyone yet.

The same discipline applied to stat 9 found a different kind of empty number: the ground-duel
win rate is **0.5 by construction** — every duel has two opposing participants and exactly
one winner, so the population rate is 100/200 whatever the players did. Only per-player and
per-club splits carry information. And stat 10's location skew turned out to depend on an
undocumented choice: anchoring a turnover at the pass's reconstructed *end* gives
def/mid/final 77/193/229, while anchoring at its *start* gives 143/194/162. Turnovers skew
forward of recoveries under both — that comparison is safe — but "turnovers concentrate in
the final third" holds only under the `end` anchor, so the anchor is now a named parameter
that must be quoted alongside the split.

## Results — FOOTPASS val, ground truth, replay-filtered

*(Per-half sheets and the full sweep are written to `data/reports/tier1-stats/`.)*

| Half | events raw → live | replays | chains | shots | xG | key passes | SCA | prog. passes | recoveries | passes |
|---|---|---|---|---|---|---|---|---|---|---|
| game_18_H1 | 1042 → 905 | 137 | 132 | 14 | 1.75 | 12 | 25 | 42 | 87 | 398/456 |
| game_18_H2 | 837 → 716 | 121 | 118 | 12 | 1.09 | 6 | 16 | 31 | 68 | 286/336 |
| game_24_H1 | 924 → 867 | 57 | 116 | 9 | 0.58 | 6 | 15 | 34 | 73 | 375/426 |
| game_24_H2 | 1052 → 970 | 82 | 140 | 5 | 0.31 | 3 | 9 | 24 | 99 | 410/471 |
| game_47_H1 | 1202 → 1108 | 94 | 115 | 10 | 1.05 | 10 | 20 | 35 | 81 | 525/586 |
| game_47_H2 | 1013 → 890 | 123 | 131 | 15 | 1.84 | 12 | 28 | 49 | 86 | 400/447 |

*(Small deltas vs the first published table — prog. passes ±1–3, recoveries −1/−2 per half —
come from two second-round review fixes: live end points are no longer read from replay
successors, and a contest event founding a chain after a stoppage no longer counts as a
possession change.)*

Pass completion runs 84–90% per half, which is broadcast-professional-shaped. xG per club-match lands 0.25–1.97; four of six sit in or just under the plausible 1–3 band, and the two low ones are both `game_24`, which has 14 shots in a full match against a typical ~25 — a shot-supply property of this ground truth, not a coefficient problem.

Live event mix across the six halves (replays removed): pass 2755, carry 2239, header 149,
cross 99, block 75, shot 65, throw-in 49, tackle 25.

**Coverage is not incidental, and it corroborates the source doc's premise.** Each player has
exactly one tactical row per frame, so the denominator is exact. Mean in-frame visibility per
half runs **31.5% – 48.8%** — squarely in the "20–40% of the match, non-uniformly" range the
source doc gives as its reason for refusing Tier 4 totals, here measured rather than asserted,
on broadcast footage that is *more* generous than the amateur single-camera target. Within a
half the spread is severe: one `game_18_H1` player is observable for **8.9%** of it. Every
stat line carries its own denominator for this reason.

**Progressive passes were corrected downward by ~65% during review** (`game_18_H1`: 58/71 per club → 16/27) once FBref's "excludes passes from the defending 40%" clause and its completion requirement were both applied. The first draft had adopted the box clause from the same sentence while dropping the exclusion clause beside it. A remaining known deviation, flagged by the implementer rather than the reviewer: FBref's "from its furthest point in the last six passes" is not implemented, so these counts are an **upper bound** on the FBref rule.

## Recall-sensitivity sweep — the build-order table

Drop X% of events and measure each stat's mean relative movement over 10 trials, under two
loss models: **uniform**, and **crowd-biased**, where an event's drop probability rises with
the number of players near it — because that is where real detectors fail.

Mean relative movement at a 10% drop (and at 40%):

| Stat | uniform | crowd-biased |
|---|---|---|
| `ratio_pass_completion` | 0.008 (0.023) | 0.007 (0.015) |
| `ratio_progressive_share` | 0.045 (0.170) | 0.066 (0.218) |
| `ratio_field_tilt_box_share` | 0.043 (0.109) | 0.132 (0.592) |
| `count_recoveries` | 0.068 (0.322) | 0.116 (0.513) |
| `count_passes_attempted` | 0.106 (0.434) | 0.114 (0.466) |
| `count_final_third_entries` | 0.106 (0.471) | 0.131 (0.557) |
| `count_progressive_actions` | 0.129 (0.515) | 0.173 (0.582) |
| `count_touches_in_opp_box` | 0.107 (0.387) | 0.223 (0.755) |
| `count_shots` | 0.121 (0.393) | 0.307 (0.850) |
| `count_sca` | 0.128 (0.492) | 0.300 (0.856) |
| `total_xg` | 0.130 (0.454) | 0.314 (0.908) |
| `count_key_passes` | 0.217 (0.583) | 0.350 (0.892) |

**Gating table** — the highest drop rate each stat tolerates within 10% movement, where
"tolerates" now means *every rate up to it passed under **every** loss model*. (The first
published version of this table let a later lucky pass override an earlier failure and
quietly keyed on the optimistic uniform model; the second review round caught both, plus a
normalisation bug that made the crowd-biased sweep apply only about half its nominal stress.
The corrected table is harsher, and the harsher version is the honest one.)

| tolerates 40% | tolerates 10% | tolerates 5% | tolerates 0% — fails even a 5% drop |
|---|---|---|---|
| pass completion % | progressive share | field-tilt share, recoveries, passes attempted, progressive actions, final-third entries | key passes, SCA, shots, xG, box touches |

Two things fall out, and they set the build order:

1. **The source doc's "prefer ratios over counts" principle is measured, not assumed.** Pass
   completion moves 0.8% when a tenth of the event stream vanishes; the raw pass count moves
   ~11%, essentially one-for-one. Ratios sit at the top of the table, counts at the bottom.
2. **The shot-anchored family fails outright under crowd-biased loss.** Key passes, SCA,
   shots, xG and box touches all move ≥ 22% at a 5–10% crowd-biased drop — versus ~12% under
   uniform loss — because a key pass needs *two* specific events to survive and crowded
   events cluster near the box, precisely where shots and the passes that create them live.
   A sweep run only against uniform loss, or with the earlier optimistic gating, reported
   these stats as merely "5%-tolerant"; honestly gated, they tolerate **no** measured loss.

So: pass completion and progressive share are buildable against a weak event detector; the
volume counts need ~95% recall; and the creation family (stats 2–4), xG and box touches need
near-perfect recall **in crowds specifically** before their numbers mean anything.

## What is not derivable from this ground truth

| Stat | Status |
|---|---|
| **xG (1)** | Model runs; **calibration entirely untested** — there are no goal labels anywhere in this GT, so no xG value here has ever been checked against an outcome. Coefficients are published and cited, not fitted here. Report as a percentile. |
| **xA (3)** | Abstains (`None`) unless an xG model is injected; inherits every xG caveat. |
| **GCA (4)** | `None`, reason `no_goal_labels`. Never 0 — that would claim every player failed to create a goal. |
| **SCA (4)** | A strict subset of FBref's: shot rebounds and fouls won have no class in this GT. |
| **Take-ons (8)** | No class, no negative class, no validation possible. Flagged `unvalidated`; kept out of the headline set. |
| **Duels (9)** | Sample-starved: **one** aerial duel across three matches at the 1.0 s definition. The ground-duel win rate is 50% *by construction* — every duel resolves and exactly one club wins — so any aggregate is a mathematical identity, not a measurement. |
| **Pass outcomes, end points** | Inferred from chain continuation and neighbouring positions, never labelled. Stamped `outcome_source = inferred` throughout. |
| **Under-pressure, defenders-in-lane, take-ons** | Read all-22 GT positions. That is an isolation condition of this harness, declared per stat via `requires_offball_positions`. |

## Corrections made to existing documentation

* `docs/reference/footpass-setup.md` — the action-class order was **wrong**: it listed
  "Shot, Header, Throw-in" where the data has throw-in at 4, shot at 5, header at 6.
  Anything built against the old order labelled every shot a throw-in. Settled empirically
  (class 4 is 86.6% within 5% of a touchline; class 5 is 0%).
* Same doc — `LEFT_TO_RIGHT` reframed as pitch side, with the club↔side rebinding, the stable
  keys, and the 180°-rotation requirement recorded.
* Same doc — the replay flag's location and prevalence added.
* `docs/prds/peripheral-stats-tier-1.md` — an early estimate that replays accounted for
  ~21.5% of possession changes was corrected to the measured 7.25% across the split; the
  original figure counted raw chain boundaries, a looser denominator.

## Verification

`uv run pytest packages -q` → **1461 passed, 5 skipped**. `uv run ruff check packages` clean.
181 of those tests are new and belong to this branch.

Each stat was implemented and then **cold-reviewed by an agent with no stake in it**, briefed
to re-derive every reported number from the raw data without importing the module under
review, and to mutate the implementation and report which mutations the tests failed to
catch. That second instruction did most of the work:

| What the review found | Effect |
|---|---|
| FBref's "excludes passes from the defending 40%" clause dropped, and the completion requirement missing | progressive passes 58/71 → 16/27 per club on one half |
| A mutation making *any* forward carry progressive passed all 36 tests | would have inflated progressive carries 14/25 → 74/85 undetected |
| Take-on null baseline | 65% of the detector was carry length; rule corrected, 57% still is |
| Turnover credited to the blocker, not the player who lost the ball | real misattribution, fixed |
| The SCA window walked past an earlier shot and credited an action from before it | one credit reassigned; the two definitions in the file now agree a shot is a barrier |
| Cross-club guards were untested — deleting them left 42/42 tests passing | five direct tests added on raw slices, mutations now die |
| `compute_tier1` folded an xA abstention to `0.0`, and emitted `gca=None` even when goal labels were supplied | both fixed; `None` no longer carries two meanings on one sheet |
| **A citation in the xG module was fabricated** — a claimed second study "triangulating" the header coefficient is the same analysis by the same authors on the same sample, and does not contain the value attributed to it | corrected; see below |

The last one is the reason cold review was worth its cost. The xG module claimed its header
coefficient (−1.16) was "triangulated against" a second study reporting −1.29, with the
conservative value chosen. The reviewer pulled both PDFs: the second paper contains no −1.29
anywhere, reports the identical −1.16 on the identical 10,709-shot sample by the identical
four authors, and is the same analysis re-posted. Wrong number, and no independence to
triangulate with. It is the single claim in the branch that no amount of reading the code
could have falsified — and it existed precisely to reassure a reader that a number had been
externally checked. The module now states plainly that −1.16 rests on **one unrefereed
preprint** with no corroborating fit, and names the re-post so nobody cites it as a second
observation.

Two related honesty fixes followed from the same review: 29 of the 65 val shots (45%) carry
an **unfitted** coefficient (`defenders_in_lane`), so the cited-only totals are now asserted
alongside the defaults (max delta +0.309, band membership unchanged); and
`is_set_piece_origin` now **abstains** instead of returning `False`, because FOOTPASS has no
corner or free-kick class — a corner arrives labelled `CROSS` and a free kick `PASS`, so a
negative there was a claim the vocabulary cannot support.

**Process hazard worth recording:** mutation testing rewrote `stats/xg.py` in place while
other work was staging files, and a mutated coefficient (`defender_in_lane = -0.75` instead
of `-0.25`) **did reach a commit** before being caught by diffing the working tree against
`HEAD`; it was amended out. Mutation testing in a shared worktree must run against a copy,
and a commit made while agents are editing needs its diff read, not just its test run — the
suite was green on the mutated value because the mutation only fires on shots the assertions
tolerate.

**Second review round (final state, 2026-08-05).** After the fixes above landed, five fresh
cold reviewers re-reviewed the committed state of every module, confirmed all first-round
fixes present, and found no blockers — but the foundation reviewer demonstrated five further
defects by direct probe, none of which the (then-green) suite pinned:

| Second-round finding | Effect of the fix |
|---|---|
| Live end points were reconstructed from **replay successors** (3.8% of live ball-moving events) — a position from re-broadcast footage, disagreeing with the outcome the chain layer resolves | prog. passes/recoveries shift by ±1–3 per half |
| A contest event **founding** a chain after a stoppage (nothing to interrupt) manufactured a turnover charged to a club that never held the ball | recoveries −1/−2 per half |
| `gating_table` let a later lucky pass override an earlier failure, and keyed on the optimistic uniform model | gating table substantially harsher (see above) |
| The crowd-biased drop clipped probabilities at 1.0 without renormalising — a nominal 40% drop realised ~21% | crowd-biased rows now apply full stress |
| `shots` was gated behind `xg_fn`, so `shots=0` silently meant "not counted" — the exact zero/abstention conflation the schema warns about | shots always counted; only xG needs the model |

Plus, from the other reviewers: duel winners were resolved across half boundaries and dead
balls (now unresolved, matching the chain layer's guard); the SCA window walked past a
throw-in restart (now a creditable barrier, like the shot rebound); the abstention counters
the passing/progression modules return were dropped at the sheet fold (now on
`Tier1StatSheet.abstentions`); and a handful of surviving mutations got pinning tests
(same-club contest in an SCA window, the aerial-duel reuse guard, the ground-duel type set).

Every one of those behaviors was demonstrable while the whole suite was green — which is the
recurring lesson of this branch: the suite proves what it pins, and nothing else.

The remaining NIT-level review items were then also implemented rather than left as notes:
the FBref decompositions (passing/carrying distance, PPA vs CrsPA, the pressure-unknown
buckets) now have real slots on `Tier1StatLine` instead of living only in the module
dataclasses; `load_half_events` requires the play-by-play argument explicitly (an unfiltered
stream must be written out as `None`, never produced by forgetting an optional); an
actor-less shot earns no key pass (without a shooter identity the self-pass check cannot
fire); `percentile_within` rejects NaN and gained its first consumer — the runner emits
`xg_percentiles` as the reportable xG number, per the source doc's mandate; duel timestamps
and the 3 m take-on boundary are pinned; and the cross-chain throw-in precedence in the xG
set-piece rule plus the counter-bucketing divergence between the passing and progression
modules are stated where the code lives.

Independent re-derivation agreed exactly on every headline number in every module: the xG
val-split distribution and all six club-match totals, the duel and take-on counts and null
baselines, the creation counts, and the progression and passing tables.

**Known weakness in the verification itself, stated rather than buried:** the xG structural
tests (monotonicity, bounds, symmetry) catch only 8 of 20 coefficient mutations — flipping
the intercept's sign passes all of them. The test that actually detects damage is the
val-split totals characterisation test, which is a regression test, not a validity test.
Structural validation is weaker than it reads.

## Follow-ups, not done here

* No pipeline stage is registered and no new `ArtifactName` was added — nothing in the
  pipeline produces this sheet yet, and registering an artifact no stage writes would be a
  dangling contract. `ArtifactName.STATS`/`StatLine` (the possession-heuristic surface)
  remains separate; merging the two is deliberate follow-up work.
* No UI wiring.
* Tier 2 (xT engine, momentum) and Tier 3 (ball-local pitch control) untouched. The source
  doc puts xT next, and notes it is one model feeding three product surfaces.
* The stats are validated against ground-truth *labels*, never against pipeline output. The
  sweep is the stand-in for that, and it is a model of event loss, not a measurement of this
  pipeline's event loss.
