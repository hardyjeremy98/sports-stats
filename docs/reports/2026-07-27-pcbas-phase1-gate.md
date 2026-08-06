# PCBAS Phase 1 gate — stage 1 (TAAD-alone) on FOOTPASS VAL

**Date:** 2026-07-27
**Branch:** `worktree-spo-action-spotting-prd`
**Checkpoint:** `action_head_selected.pt` = epoch 12 of 20 (5.30 h budget reached)
**Linear:** SPO-96 · **Build:** [`2026-07-27-pcbas-phase1-build.md`](2026-07-27-pcbas-phase1-build.md)

The first end-to-end measured number: a trained action head run over all six VAL
halves and scored with the reference-validated player-centric metric.

**This is stage 1 only. DST is built but untrained**, so this is comparable to the
reference's TAAD arm (0.4100) and **not** to its headline TAAD+DST (0.7186).

---

## Result

| | ours | reference TAAD | delta |
|---|---:|---:|---:|
| **micro-F1** | **0.3274** | 0.4100 | −0.083 |
| macro-F1 | 0.2056 | 0.2445 | −0.039 |
| precision | **0.3568** | 0.3039 | **+0.053** |
| recall | 0.3025 | 0.6297 | −0.327 |
| TP | 1,836 | 3,822 | |
| FP | 3,310 | 8,754 | |
| FN | 4,234 | 2,248 | |

**80% of the reference's micro-F1 on 60% of its training schedule** (12 epochs of 20),
with a model that was still improving at its final epoch.

Identical under both identity modes — `by_slot` and `by_shirt` agree to four decimals
(see below), so the figure is unambiguous.

### Per class

| class | ours F1 | reference F1 | GT |
|---|---:|---:|---:|
| drive | 0.317 | 0.408 | 2,470 |
| pass | 0.453 | 0.571 | 3,059 |
| cross | 0.095 | 0.235 | 111 |
| **throw-in** | **0.316** | 0.262 | 97 |
| shot | 0.145 | 0.232 | 67 |
| **header** | **0.212** | 0.147 | 162 |
| **tackle** | **0.030** | 0.029 | 26 |
| **block** | **0.078** | 0.071 | 78 |

We **beat the reference on four of eight classes** — every one of them a rare class.
The deficit is concentrated in `drive`, `pass` and `cross`. That is why macro-F1 is
closer to parity than micro-F1: micro is dominated by `pass` and `drive`, which are
5,529 of the 6,070 events.

---

## The gap is entirely recall — and it is a model-quality gap, not an operating point

We are **more precise than the reference** and less than half as sensitive.

| half | GT | our predictions |
|---|---:|---:|
| game_18_H1 | 1,042 | 862 |
| game_18_H2 | 837 | 657 |
| game_24_H1 | 924 | 704 |
| game_24_H2 | 1,052 | 758 |
| game_47_H1 | 1,202 | 1,147 |
| game_47_H2 | 1,013 | 1,018 |
| **total** | **6,070** | **5,146** |

We emit **fewer events than exist**. The reference emits 12,576 — more than twice the
ground truth — trading precision for double the recall.

**The conservatism is in the `argmax`, not the threshold.** Of 5,146 decoded events,
**zero** were removed by the τ=0.15 confidence floor: every surviving peak already
scored above it. So the model is not producing low-confidence actions that get
filtered — it is not producing them at all, because background wins the argmax over
all 9 classes.

That makes decode-time class balance the natural suspect. It was tested, and it is
not the answer — see below.

### Measured: the decode threshold is NOT the lever

The obvious inference was that we simply fire too rarely, and that re-decoding the
saved logits on action-class probability instead of a global argmax would spend our
surplus precision on the missing recall. **Measured on the saved VAL logits, that is
wrong**, and it is worth recording so it is not re-attempted:

| decode | preds | micro-F1 | macro-F1 | precision | recall |
|---|---:|---:|---:|---:|---:|
| argmax (current) | 5,146 | 0.3274 | **0.2056** | 0.357 | 0.303 |
| P(action) ≥ 0.45 | 4,615 | 0.3255 | 0.2074 | 0.377 | 0.286 |
| P(action) ≥ 0.35 | 7,226 | **0.3400** | 0.1995 | 0.313 | 0.372 |
| P(action) ≥ 0.25 | 11,922 | 0.3171 | 0.1783 | 0.239 | 0.470 |
| P(action) ≥ 0.15 | 22,123 | 0.2486 | 0.1435 | 0.158 | 0.577 |

Best case is +0.013 micro-F1, and macro-F1 gets *worse*. `argmax` is already
near-optimal for macro and within a hair of optimal for micro; no change is warranted.

**The decisive comparison is at matched prediction volume.** At threshold 0.25 we emit
11,922 predictions — essentially the reference's 12,576 — and recall **0.470 against
its 0.630**. Firing at the same rate, we find about 25% fewer real events.

So the deficit is not an operating-point choice: **our logits are genuinely weaker**.
That rules out the cheap explanation and points the remaining gap at training —
principally the 8 unrun epochs (the model was still improving when the budget stopped
it), and possibly the affine/scale/crop augmentations dropped because
`albumentations` is not packaged here.

---

## Two findings

### The ADR 008 roster remap is lossless on VAL

`by_slot` and `by_shirt` produce **identical** TP/FP/FN and identical F1 to four
decimals, and `unknown_shirt` is **0 on every half**. Within a half each occupied slot
maps to exactly one shirt, so the two groupings partition the events the same way.

The export-time remap therefore costs nothing measurable here. That is a real result
for ADR 008 — the slot-native model loses nothing by being reported in shirt terms —
but it is measured on ground-truth roles. Phase 3 replaces those with assigned roles,
and the remap's cost there is a separate, unmeasured question.

### Off-screen recovery is negligible for any visual model

Our stage 1 recovers **9** of the 1,062 no-bbox VAL events; the reference's TAAD
recovers **33**. Both are noise next to the **390** that TAAD+DST recovers.

This is the strongest available argument for the two-stage architecture, and it is not
about accuracy on visible players at all: roughly 17.5% of actions happen to a player
the camera is not showing, and no amount of additional visual training reaches them.
Only the sequence stage can.

---

## Honest scoping

- **12 of 20 epochs.** Training stopped on a 5.30 h wall-clock budget; `result.json`
  records `epochs_run` vs `epochs_planned`. The final epoch won the checkpoint
  selection on both micro and macro F1, so the schedule was **not** saturated — the 8
  unrun epochs would likely have helped.
- **Checkpoint selected on the task metric, not validation loss.** Val loss is
  dominated by background cells and is measurably the wrong criterion: it would have
  selected epoch 8 (clip micro-F1 0.346) over epoch 12 (0.400), a 16% relative loss.
- **The clip-level metric used for selection is not this number.** It scores 5
  candidate players over 50 frames; this scores 26 slots over ~150,000 frames per half.
  The clip figure (0.400) resembling the reference's 0.4100 is a coincidence of two
  different measurements and must not be quoted as a reproduction.
- **No control run.** The reference ships no checkpoints, logits or video, so running
  it unmodified would cost a full retrain. The available control — its shipped anchor
  list — passes exactly on both splits.

## Reproduce

```bash
E=packages/matchlab_train/src/matchlab_train/experiments
uv run matchlab-train run $E/pcbas_infer_logits.yaml   # ~50 min, 6 halves
uv run matchlab-train run $E/pcbas_score.yaml
```
