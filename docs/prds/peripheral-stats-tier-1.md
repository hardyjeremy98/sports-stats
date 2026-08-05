# Peripheral Statistics — Tier 1 (stats 1–11)

Source: Notion "📊 Peripheral Statistics" (curated 2026-08-05), Tier 1 = event-chain stats.
This document is the implementation plan and the honest coverage record for what is and is
not derivable from ground truth. Revised 2026-08-05 after a cold review that found four
plan-breaking errors (recorded inline below, because each one is a trap for the next reader).

## Scope

Implement Tier 1 stats 1–11 as a source-agnostic library over a canonical event stream in
pitch coordinates, plus the cross-cutting machinery the doc mandates (coverage denominators,
per-stat abstention declaration, recall-sensitivity sweep). **Ground-truth labels only** — no
tracker, detector or spotter output is consumed anywhere in this branch.

Out of scope: Tier 2 (xT, momentum), Tier 3 (pitch control), Tier 4 (never), UI, pipeline
stage registration.

## Ground-truth substrate

FOOTPASS (SoccerNet SN-PCBAS-2026) val split: 3 matches × 2 halves, 6070 events.

Two GT files carry the **same** event stream — verified exact on the full
`(frame, team, shirt, class)` tuple set for all 6 halves, zero symmetric difference:

* tactical HDF5 `data/footpass/tactical/val_tactical_data.h5` — `CLS` on the acting
  player's row, which also carries that player's pitch `X, Y` **and all 22 players'
  positions at that frame**.
* `data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json` —
  `[frame, team, shirt, class, has_bbox, replay]`.

**Both are required inputs.** The h5 alone is not sufficient: the `replay` flag exists only
in the JSON, and **10.1% of val events (614/6070) are broadcast replays** (per half:
137/121/57/82/94/123). Replays duplicate live play, so an unfiltered stream double-counts
events and fabricates possession changes at replay boundaries. The adapter joins the two on
`(frame, team, shirt, class)` — exact and unique on all 6 halves — and excludes `replay == 1`
by default. Every headline number is reported with and without the filter, and the delta is
published.

> Cold-review trap #1: the obvious cross-check (h5 stream == JSON stream) passes perfectly
> and proves nothing, because the streams are identical. The test that bites is toggling the
> replay filter and asserting the possession-change count moves.

Event classes (`CLS` 1..8): `drive, pass, cross, throw-in, shot, header, tackle, block`.
Val counts: pass 3059, drive 2470, header 162, cross 111, throw-in 97, block 78, shot 67,
tackle 26. (`docs/reference/footpass-setup.md` lists a different order and is wrong; the data
settles it — class 4 is 86.6% within 5% of a touchline, class 5 is 0% touchline and 64% near
a goal end. That doc is corrected in this branch.)

### Identity keys — not `(TEAM, SHIRT)`

`TEAM` is **pitch side** (0 = left, 1 = right) in every half, not club. The club↔`TEAM`
binding **rebinds at half-time**: verified 18/18, 21/21, 21/21 players change `TEAM` value
between halves in the three val matches. `PLAYER_ID` is stable across halves and
`PLAYER_ID // 100 ∈ {1, 2}` is the stable club key. Shirt numbers also collide across teams
within a half (7 collisions in `game_24_H2`).

* Player aggregation keys on `PLAYER_ID`.
* Team aggregation keys on club = `PLAYER_ID // 100`.
* `(TEAM, SHIRT)` is used **only** to join the play-by-play row to the h5 row, never to
  aggregate.

> Cold-review trap #2: `docs/reference/footpass-setup.md` and this repo's memory both say
> "TEAM flips at half-time", which reads as *sides* flipping. Sides never flip. Corrected in
> the setup doc in this branch.

### What the GT does NOT contain

| Missing | Consequence |
|---|---|
| **Goal outcome** | xG cannot be fitted or calibrated here; GCA returns `None` with reason `no_goal_labels`, never a misleading 0. |
| **Pass outcome flag** | Inferred from the chain, labelled `outcome_source = inferred`. |
| **Take-on class** | Stat 8 has no GT label *and no negative class*; detector is validated on constructed cases only and carries an explicit `unvalidated` flag. Excluded from headline tables. |
| **Duel outcome** | Inferred from possession retention. |
| **Stoppage / period markers** | Chain boundaries need an explicit time guard (below). |

## Coordinates

Stored in **centimetres**, matching `matchlab_core.pitch` (`FIFA_PITCH` = 10500 × 6800 cm)
and the existing `distance_cm` convention. FOOTPASS normalised coords convert through one
tested function; no scattered `/100`.

**Attack normalisation is a 180° rotation, not a mirror**: teams attacking `−x` are
transformed `x → L − x` **and** `y → W − y`. The `1 − x` idiom quoted in the upstream setup
doc is a reflection that silently swaps left and right wings, mirroring every flank stat for
one team. A round-trip test asserts a left-flank action stays on the left flank.

Attack direction is the direct GT fact: `TEAM == 0` attacks `+x`, `TEAM == 1` attacks `−x`,
in every half. (The originally planned goalkeeper-mean-x cue was dropped: it agrees, but it
estimates something the data states outright, and the H1/H2 regression test built on it is
vacuous — nothing flips. The test that has teeth: the same **club** maps to opposite `TEAM`
values in H1 vs H2, and its shots concentrate near `+x` after normalisation in both.)

Coordinates exceed `[0, 1]` in the raw data (X ∈ [−0.035, 1.018], Y ∈ [−0.646, 1.066];
`game_24_H2` has a Y 44 m off the pitch). Positions outside a 300 cm plausibility margin —
the same margin `stages/fuse/minimap.py` uses — are **rejected, counted, and surfaced** in
the coverage block; they are not silently clamped into a zone.

## Architecture

Library (source-agnostic) in `packages/matchlab_core/src/matchlab_core/stats/`:

```
schema.py       MatchEvent, EventOutcome, StatEventType, ActorKey, Tier1StatLine/Sheet
zones.py        thirds, penalty area, goal geometry, distance/angle, pressure, cm<->norm
chains.py       possession-chain segmentation; contest handling; receiver + outcome inference
xg.py           shot features + logistic xG (stat 1)
creation.py     key passes, xA, SCA/GCA (stats 2-4)
progression.py  progressive passes/carries, entries, box touches (stats 5-7)
duels.py        take-ons, duels, recoveries/turnovers (stats 8-10)
passing.py      decomposed pass completion (stat 11)
compute.py      orchestrator -> Tier1StatSheet + coverage denominators
sensitivity.py  recall-sensitivity sweep
```

FOOTPASS GT adapter and the sweep/report runner live in **`matchlab_train`**
(`datasets/footpass_events.py`, `experiments/tier1_stats.py`) — dataset adapters belong
there, train may import core, and this keeps `matchlab_core.stats` free of any FOOTPASS
import. (The alternative, `core/stats/sources/footpass.py`, would invert the package
dependency.)

**No new `ArtifactName`.** `ArtifactName.STATS → stats.json` and `schemas/events.py::StatLine`
already exist and are written by the possession-heuristic events stage. This branch registers
no stage, so a new enum member would be a contract nothing produces. `Tier1StatSheet` is a
separate, richer schema written by the experiment runner to its own report path; superseding
or merging the older `StatLine` surface is a follow-up, recorded and not silently done.

Tests land flat in `packages/matchlab_core/tests/test_*.py` (repo convention — no test
subdirectories).

## Stat definitions (anchored, not invented)

Anchors: FBref (SCA/GCA, progressive actions, thirds), StatsBomb open-data spec (xG feature
set), Opta box/zone conventions.

1. **xG** — logistic on `[log distance to goal centre, visible goal angle, is_header,
   is_set_piece_origin, defenders_in_lane]`. **No goal labels exist in this GT**, so
   coefficients are literature-shaped and fixed, never fitted. Validated *structurally*:
   monotone decreasing in distance, increasing in angle, bounded `(0,1)`, penalty-spot value
   in the 0.7–0.8 band. Calibration is explicitly untested and the stat is reported as a
   within-userbase percentile, per the source doc.
2. **Key passes / chances created** — pass or cross immediately preceding a shot by the same
   club within the same chain.
3. **xA** — the xG of the shot a key pass created, credited to the passer.
4. **SCA / GCA** — the two offensive actions (pass, cross, drive, take-on) preceding a shot,
   same club, same chain, 2-action window. **GCA is `None` (`no_goal_labels`)** on this GT.
5. **Progressive passes / carries** — FBref rule: ≥10 yd (914 cm) closer to the opponent goal
   centre, or ending in the penalty area; carries ending in the own defensive third excluded;
   passes into the box always count.
6. **Final-third / penalty-area entries** — start outside, end inside, split pass vs carry.
7. **Touches in the opposition penalty area** — on-ball event starting inside the box.
8. **Take-ons** — no GT class, no negative class. Detector: during a `drive`, an opponent
   within 3 m of the carrier at start who is behind the carrier (attack direction) at the end.
   Consumes all-22 GT context, so it is an **isolation condition**, not a general capability:
   flagged `requires_offball_positions` and `unvalidated`, excluded from headline tables.
9. **Duels won %** — aerial = opposing `header` events within 1.0 s; ground = `tackle` /
   `block` against the carrier; won by the club holding the ball at the next
   possession-defining event. **Sample-starved on this GT and reported as such: 2 of 162
   headers (1.2%) have an opposing header within 1.0 s, and there are 26 tackles in the whole
   val split (~8.7/match vs a real rate of 15–20).** The definition ships with its counts
   attached; the number itself is not usable evidence at this n, and the results report says
   so rather than printing a percentage.
10. **Recoveries and turnovers, with pitch location** — chain-boundary possession changes;
    gaining actor gets a recovery, losing actor a turnover, both stamped with pitch xy and
    third. Contest events and replay boundaries are excluded before boundaries are counted.
    *Measured, correcting an earlier estimate in this document:* the replay filter moves live
    possession changes by **7.25% across the val split** (4.3% on `game_18_H1`), not the
    ~21.5% first claimed — that figure counted **raw chain boundaries**, a different and much
    looser denominator. The contest-interruption fix absorbed part of the replay filter's
    effect, since an event inside a replay segment could break a chain by either route.
    A cautionary detail: on `game_18_H1` the chain **count** is 132 either way, while the
    composition differs (possession changes 93 → 89, gap splits 38 → 42). An unchanged
    headline count is not evidence that a filter did nothing.
11. **Pass completion, decomposed** — overall, by direction (forward / lateral / back, ±15°
    bands), by third of origin, and under pressure (nearest opponent < 5 m at release — again
    all-22 GT context, flagged like stat 8).

### Chain construction — the part that quietly decides four stats

* **Contest events (`block`, `tackle`, contested `header`) do not settle possession.** A
  `pass → tackle(opponent)` means the pass **completed** and the receiver was then
  dispossessed; a naive next-event-team flip marks the pass incomplete. Outcome resolution
  skips contest events to the next possession-defining event. This matters for 195 of the 484
  opposite-team successors in val (`block` 48, `header` 83, `throw-in` 64).
* **Same-actor successors** (13/3166 passes, 0.4%) cannot be receipts; excluded from receiver
  inference explicitly.
* **Chains need a time guard.** Inter-event gaps: median 36 frames, p90 102, p99 928 (37 s),
  max 3950 (158 s); 3.8% exceed 10 s, and the GT has no stoppage marker at all. A chain
  breaks on a gap > **10 s** (parameter, swept in the sensitivity harness); the number of
  chains this splits is reported, not assumed.

## Cross-cutting requirements (from the source doc)

* **Coverage denominator per player-match** — fraction of the half with `ROI_X` non-NaN
  (in-frame) in the tactical GT, emitted on every stat line.
* **Abstention class per stat** — each stat declares whether it tolerates actor-less events;
  encoded in the stat registry and asserted in tests.
* **Ratios over counts** — completion %, duel %, shares are first-class; counts always carry
  their coverage denominator.
* **Recall-sensitivity sweep** — drop X ∈ {5, 10, 20, 40}% of events, uniformly and biased
  toward crowded frames (misses cluster in crowds), and measure each stat's relative movement.
  Publishes the gating table the source doc calls "probably the first concrete task of the
  phase". A deliverable, not an extra.

## Verification plan

* Unit tests per stat module, hand-computed expectations on tiny synthetic streams.
* Real-GT consistency tests: chain totals reconcile with event counts; pass outcomes
  partition exactly; club-flip direction test; rotation round-trip (left flank stays left).
* **Disconfirming tests, not just confirming ones**: toggling the replay filter must move the
  recovery/turnover count by ~20%; disabling contest handling must move pass completion;
  removing the y-flip must break the flank test.
* Every numeric claim in the results report traced to a script that re-derives it.
* `uv run pytest packages -q` and `uv run ruff check packages` clean before merge.

## Build order

1. Foundation: schema, zones, chains, FOOTPASS adapter (with replay join), coverage.
2. Recall-sensitivity harness.
3. Stats 5–7, 11 (geometry/chain queries; lowest risk).
4. Stats 2–4, 10 (chain-relational).
5. Stats 1, 8, 9 (model- or off-ball-context-dependent; highest caveat load).
6. Sweep run, results report, doc corrections (`footpass-setup.md` class order and TEAM
   framing; `docs/implementation-status.md`).
