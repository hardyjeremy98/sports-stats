# PCBAS Phase 0 — ingest and metric gates

**Date:** 2026-07-27
**Branch:** `worktree-spo-action-spotting-prd` @ `6cf663c`
**Data:** FOOTPASS / SoccerNet SN-PCBAS-2026 tactical HDF5, `data/footpass/tactical/`
**Plan:** [`docs/superpowers/plans/2026-07-27-player-centric-action-spotting.md`](../superpowers/plans/2026-07-27-player-centric-action-spotting.md) Tasks 1–4
**Linear:** SPO-96

Both Phase 0 gates pass. No video, no GPU, no training was involved — which is the
point: these are the cheapest falsifiable checks available, and everything measured
later is meaningless if either fails.

---

## Gate A — the metric reproduces the reference exactly

The reference ships its own VAL ground truth and the final event lists both published
arms produced. Scoring those artifacts with `matchlab_core.pcbas.eval.score_halves`
(δ=12, τ=0.15, `identity="shirt"`) must reproduce the numbers its own
`evaluation.py` prints. It does, to the integer.

| arm | TP | FP | FN | GT | micro-F1 | macro-F1 | TP without bbox |
|---|---:|---:|---:|---:|---:|---:|---:|
| TAAD | 3,822 | 8,754 | 2,248 | 6,070 | **0.4100** | 0.2445 | 33 |
| TAAD + DST | 4,267 | 1,539 | 1,803 | 6,070 | **0.7186** | 0.4926 | 390 |

Per-class F1:

| class | TAAD | TAAD + DST |
|---|---:|---:|
| drive | 0.408 | 0.724 |
| pass | 0.571 | 0.758 |
| cross | 0.235 | 0.610 |
| throw-in | 0.262 | 0.647 |
| shot | 0.232 | 0.602 |
| header | 0.147 | 0.293 |
| tackle | 0.029 | **0.061** |
| block | 0.071 | 0.246 |

The full 8×3 per-class `(TP, FP, GT)` table is asserted, not F1 alone — two different
matching rules can round to the same F1 but not to the same table.
`packages/matchlab_train/tests/test_pcbas_eval_reproduces_reference.py`.

Three details had to be faithful or the counts drift:

1. Sub-threshold predictions are dropped **before** matching, so they cannot even
   become false positives.
2. Detections sort by descending score **globally**, not per class.
3. Matching happens **per half key**. Frame indices run continuously across a match's
   halves but overlap between matches (`game_18` spans 32–149,181; `game_24` spans
   5,397–157,023), so a pooled scorer would let a `game_18` prediction satisfy a
   `game_24` ground-truth event.

### macro-F1 is ours, and it is much lower

The reference computes micro-F1 and per-class precision/recall only. macro-F1 is our
addition, reported alongside micro because either alone misleads: `pass` and `drive`
are 5,529 of VAL's 6,070 events, so micro is essentially a two-class score. In the
0.719 arm, `tackle` recall is **0.038** — one true positive out of 26. Any report
quoting micro alone hides that.

---

## Gate B — the ingest reproduces the published event counts

`matchlab-train footpass-stats --h5 <split>_tactical_data.h5`, reading the tactical
HDF5 directly rather than the reference's derived JSON.

| | expected | measured | |
|---|---|---|---|
| VAL halves | 6 (3 matches) | **6** | pass |
| VAL events | 6,070 | **6,070** | pass |
| TRAIN halves | 96 (48 matches) | **96** | pass |
| TRAIN events | 91,327 | **91,327** | pass |
| VAL events with a bbox | ~82.5% | **82.5%** | pass |
| slots seen | ⊆ 0..25 | VAL 22, TRAIN **26** | pass |

Stronger than the totals: VAL's **per-class** counts match the reference's own
playbyplay GT JSON exactly (drive 2,470 / pass 3,059 / cross 111 / throw-in 97 /
shot 67 / header 162 / tackle 26 / block 78). Our HDF5 reader and their JSON exporter
independently agree on every class.

### Class distribution

| class | VAL | TRAIN |
|---|---:|---:|
| pass | 3,059 | 45,621 |
| drive | 2,470 | 35,527 |
| header | 162 | 3,642 |
| cross | 111 | 2,156 |
| throw-in | 97 | 1,730 |
| block | 78 | 1,269 |
| shot | 67 | 1,097 |
| tackle | **26** | **285** |

**Correction to the plan.** It stated TRAIN has ~390 tackles and ~1,000 shots.
Measured: **285** tackles and **1,097** shots. The 390 was a transcription of DST's
`TP_noBB` count, not a tackle count. Tackle is thinner than planned for — 285 training
examples against a 500-per-class-per-epoch sampler cap, so tackle can never fill its
quota.

---

## Two findings that change the design

### 1. The "11 of 13 roles" fact is VAL-specific, not a dataset property

Earlier work measured 22 of 26 slots occupied and concluded roles 4 (MCB) and 8 (DM)
are unused. That holds for VAL's three matches — missing slots are exactly
`[3, 7, 16, 20]`, i.e. roles 4 and 8 on both sides — but **TRAIN uses all 26**. The
formations in VAL happen not to field a middle centre-back or a dedicated defensive
midfielder; other matches do.

Consequence: the model must support all 26 slots, and any Phase 3 role assigner must
too. A 22-slot shortcut would train fine and fail on unseen formations. The ADR 008
claim stands (role slots are tactical, not roster slots) but its supporting figure is
now scoped to the split it was measured on.

### 2. Row-level and event-level bbox coverage differ by half

| | rows with a bbox | events with a bbox |
|---|---:|---:|
| VAL | 41.0% | 82.5% |
| TRAIN | 38.3% | 81.4% |

These must never be quoted interchangeably. Roughly 60% of player-frames are
off-screen, but actions happen where the camera looks, so only ~18% of *actions* are
on an unseen player. `PCBASSplitStats` reports both separately for that reason, and
`PCBASReport` carries `tp_without_bbox` so the split stays visible in every score.

The ~82.5% figure is a **soft** bound on a purely visual model, not a ceiling: the
reference's TAAD arm recovers 33 of the 1,062 no-bbox VAL events and TAAD+DST recovers
**390**. A sequence model reasoning over slots can infer actions it never saw.

---

## What is not yet verified

- **CHALLENGE** carries 13 columns (no `class`) and cannot be scored locally.
  3 matches, ~5,595 withheld events by subtraction from the published 102,992 total.
- Frame-to-video alignment is verified on **VAL only** (see the Task 5 findings in the
  plan). `videos_352x640_TRAIN.zip` is still downloading; the montage check must be
  re-run on one TRAIN match before any training claim.

## Reproduce

```bash
D=/home/jeremy/code/MatchDay/lab/data/footpass/tactical
uv run matchlab-train footpass-stats --h5 $D/val_tactical_data.h5   --out /tmp/val.json
uv run matchlab-train footpass-stats --h5 $D/train_tactical_data.h5 --out /tmp/train.json  # ~3 min
uv run pytest packages/matchlab_train/tests/test_pcbas_eval_reproduces_reference.py -q
```
