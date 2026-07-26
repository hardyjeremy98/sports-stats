# Position evidence for re-ID: real signal, but a body signal — not a tail signal

**Date:** 2026-07-27 · **Author:** Claude (autonomous session)
**Spec:** [`docs/superpowers/specs/2026-07-27-position-evidence-reid.md`](../superpowers/specs/2026-07-27-position-evidence-reid.md)
**Plan:** [`docs/superpowers/plans/2026-07-27-position-evidence-reid.md`](../superpowers/plans/2026-07-27-position-evidence-reid.md)
**Branch:** `spo-position-evidence-reid`
**Substrate:** FOOTPASS tactical data — 48 train / 3 val complete matches, GT track identity,
GT pitch position, GT tactical role. Fragments built from observability spans (`ROI_X` non-NaN).
Do-no-harm gating was explicitly waived for this work by Jeremy.

## Headline

1. **Position carries real re-ID evidence.** Pooled AUC **0.771** on held-out matches
   (pre-registered H1 bar: ≥0.70). **PASS.**
2. **But only once the play-location confound is removed.** Absolute occupancy scores 0.655
   (inconclusive); formation-relative scores 0.771. The +0.116 comes entirely from
   subtracting the team's observable centroid.
3. **Position alone cannot decide merges.** At the zero-wrong frontier it repairs **14 of
   13,016** needed merges (0.1%). Gating to ≥10 s fragments does not rescue it (**11 of
   5,385**). At the loose point it is 1,376 correct / 1,301 wrong — 51% precision.
4. **So position has the same shape as appearance: good ranking, overlapping tail.** AUC is a
   property of the distribution's body; do-no-harm is set by its extreme tail. This is the
   third distinct signal (KPR, PRTreID, occupancy) to show the pattern.
5. **The design's actual claim — fusion — remains untested**, and this work does not bear on
   it. See "What this does and does not license" below.

## Prior finding confirmed by direct inspection

**Calibration has never run in any benchmarked re-ID experiment.** `manifest.json` for the
2026-07-26 smoke runs records `calibrate: static-demo, status: skipped`, and
`configs/pipeline.tdlp-full-reid-oracle.yaml` / `configs/pipeline.gt-tracklets-reid.yaml`
both carried `enabled: false`. The merge engine's pitch-metric motion branch
(`reid/gates.py:84`) only applies where calibration covers both endpoint frames, so SPO-59,
SPO-73, SPO-85 and the smoke runs all scored on the GMC pixel bound alone. Position has
contributed nothing to B2 to date — not as evidence, not even as a veto. Now enabled
(PnLCalib, `pitch: fifa`).

## Merge-failure triage (the go/no-go for this design)

The 2026-07-26 smoke runs' wrong merges, judged against GT track identity and GT team:

| run | correct | wrong | wrong same-team | wrong cross-team |
|---|---|---|---|---|
| systest-116 | 6 | 1 | **1** | 0 |
| systest-120 | 7 | 2 | **1** | 1 |

- **2 of 3 wrong merges are same-team** (tracklets 4+33, GT 5/15; and 0+31, GT 2/27) — same-kit
  teammates, exactly the case appearance cannot separate. Both at long gaps (227 and 263
  frames ≈ 9–10.5 s), which is also appearance's weakest regime.
- **1 of 3 is a goalkeeper merged across teams** (GT 11 goalkeeper / GT 25 player, right/left).
  Keepers wear a different kit from their own team, so this is a **team-gate defect**, not an
  appearance failure — SPO-75 territory.

This justified building the position channel: the errors are concentrated where the design
aims. It does not, on its own, establish that position fixes them.

## Phase A — pre-registered falsification test

Fragments from observability spans, ≥2 s. Calibrator fitted on TRAIN halves, evaluated on VAL
halves, never the same half. 240k same-player and 1.15M different-player pairs.

| | Variant A (absolute) | Variant B (formation-relative) |
|---|---|---|
| **H1 pooled AUC** | 0.6549 — INCONCLUSIVE | **0.7713 — PASS** |
| AUC, <4 s fragments | 0.6398 | 0.7432 |
| AUC, 4–10 s | 0.6595 | 0.7784 |
| AUC, 10–30 s | 0.6831 | 0.8164 |
| AUC, >30 s | 0.7170 | **0.8684** |
| **H2 distant/same-role ratio** | 1.079 — FAIL | 1.294 — INCONCLUSIVE |
| same-role is weakest bucket | yes | no (within noise of near-role) |

**Why variant B works.** A player is observable only when play is near them, so raw occupancy
partly measures *where the ball was*, not where the player's role is. This is a selection
effect, not a coordinate shift — FOOTPASS positions are pitch coordinates, so the camera does
not translate them. Subtracting the team's **observable** centroid per frame removes the shared
play-location component and leaves formation-relative position. Only observable teammates enter
the centroid, because the deployed system cannot see the others either.

**H2 is directionally right but weak.** Distant-role impostors carry 29% more evidence than
same-role ones (1.158 vs 0.895 mean |LLR|), short of the 1.5× bar. The pair-dependence the
design predicted is present but modest, and same-role vs near-role is within noise
(0.895 vs 0.889 on n=4,186 vs 53,420). **Per the pre-registered consequence, the
role-conditional global layer (Phase D) is not justified and is dropped.**

## Merge accounting — the number that matters

Mutual-best + margin over the calibrated position LLR, swept to its own frontier per half:

| half | merges needed | zero-wrong correct | loose correct / wrong |
|---|---|---|---|
| game_18_H1 | 2,530 | 1 | 255 / 288 |
| game_18_H2 | 2,283 | 1 | 206 / 280 |
| game_24_H1 | 2,219 | 4 | 239 / 190 |
| game_24_H2 | 2,242 | 2 | 250 / 194 |
| game_47_H1 | 1,802 | 1 | 217 / 162 |
| game_47_H2 | 1,940 | 5 | 209 / 187 |
| **total** | **13,016** | **14** | **1,376 / 1,301** |

Budget curve (pooled correct at a per-half wrong-merge budget): 0→14, 1→17, 5→46, 10→71,
25→125, 50→308. Gating fragments to ≥10 s: 11 correct at zero-wrong of 5,385 needed.

**Read this as a scoped negative:** *formation-relative occupancy, under mutual-best + margin,
cannot clear a zero-wrong bar on full broadcast halves.* It is not a claim that position is
useless, and it is explicitly not a claim about fusion.

## Two implementation bugs the measurement caught

Recorded because both would have produced a **false negative** attributed to the signal:

1. **The LLR was piecewise-constant.** With 20 histogram bins the calibrated score took ~20
   distinct values, so hundreds of candidate pairs tied at the ceiling and the operating curve
   jumped from "merge nothing" to "merge everything" (0 → 390 correct with nothing between).
   Fixed by interpolating between bin centres. Recall at the loose point tripled (390 → 1,376).
2. **Bin resolution was capped far below what the data supported.** The adaptive rule
   `bins = clip(n/100, 4, 20)` was designed for small samples; with 1.4M pairs it starved the
   tail, where merge safety is decided. Fixed by plumbing `max_bins` (200) at the call site.

The general lesson matches the standing one: a quantised or over-smoothed score destroys tail
resolution, and the tail is the whole game.

## What this does and does not license

**Licenses:**
- Formation-relative occupancy as the position representation (variant A is strictly worse).
- Dropping the role-conditional global assignment layer (H2 did not clear its bar).
- Treating position as a **quality-gated evidence channel** (ADR 003), not a decider — its
  strength scales with fragment duration (0.743 → 0.868 from <4 s to >30 s).

**Does not license:**
- Any claim about **fusion**, which is what the design actually proposes. Appearance and
  position could still combine well: the impostor that beats you on appearance is often a
  different player from the one that beats you on position, so the joint tail may be cleaner
  than either. That is a real hypothesis and it is **untested** — FOOTPASS has no video, so
  there are no appearance embeddings on this substrate.
- Any statement about SoccerNet or the shipped pipeline. FOOTPASS supplies identity and GT
  positions, so this is a development/isolation substrate, never a gate.
- Any change to shipped merge defaults. None were made.

## What fusion needs

Appearance and position must exist on the same fragments. Two routes:

1. **FOOTPASS + video** — `videos_fullHD_VAL.zip` (11.3 GB, 3 matches); the 352×640 tier now
   downloading is for B4/TAAD and its crops are far too small for re-ID embeddings. Run PRTreID
   over GT boxes, join on fragment id. Cleanest, because position is GT-exact.
2. **SoccerNet + calibration** — now unblocked (PnLCalib enabled). Weaker: 30-second clips give
   players almost no time to express a positional footprint, which is precisely the regime
   where this channel is weakest (AUC 0.743 under 4 s).

Route 1 is the real experiment. Route 2 is the one that tests the shipped path.

## Reproduce

```bash
uv run python -m matchlab_train.experiments.position_evidence --relative              # Phase A
uv run python -m matchlab_train.experiments.position_evidence --relative --frontier   # merges
uv run pytest packages/matchlab_core/tests/test_reid_{occupancy,evidence,merge_frontier}.py \
              packages/matchlab_core/tests/test_{footpass_loader,position_evidence_experiment}.py -q
```

Raw results: `2026-07-27-phase-a-raw.json` (variant A), `2026-07-27-phase-a-raw-relative.json`
(variant B), `2026-07-27-phase-c-frontier-position{,-10s}.json`.
