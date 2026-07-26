# PCBAS Phase 1 — build report

**Date:** 2026-07-27
**Branch:** `worktree-spo-action-spotting-prd`
**Plan:** [`docs/superpowers/plans/2026-07-27-player-centric-action-spotting.md`](../superpowers/plans/2026-07-27-player-centric-action-spotting.md) Tasks 5–9
**Linear:** SPO-96 · **Phase 0:** [`2026-07-27-pcbas-phase0-ingest.md`](2026-07-27-pcbas-phase0-ingest.md)

Every component of the two-stage player-centric spotter is implemented and tested.
Stage 1 has been run end to end on real footage. **No trained model exists yet** — the
TRAIN video finished downloading late and training is the long pole. This report
records what is built and what is measured; the reproduction gate itself is Task 10
and is explicitly **not** claimed here.

---

## What is built

| component | module | tests |
|---|---|---|
| Column schema, class order, role slots | `matchlab_core/pcbas/schema.py` | 10 |
| Slot-attributed event models | `matchlab_core/pcbas/events.py` | (via eval) |
| Player-centric metric | `matchlab_core/pcbas/eval.py` | 20 |
| `(9,26,T)` contract + window averaging | `matchlab_core/pcbas/logits.py` | 17 |
| Logits → events (softmax, NMS) | `matchlab_core/pcbas/decode.py` | 16 |
| Action head (X3D-S + FPN + ROI pooling) | `matchlab_core/pcbas/action_head.py` | 9 |
| DST sequence denoiser (2 encoders) | `matchlab_core/pcbas/denoiser.py` | 19 |
| Tactical HDF5 ingest + roster remap | `matchlab_train/datasets/footpass_pcbas.py` | 17 |
| Match video reader | `matchlab_train/datasets/footpass_video.py` | 11 |
| Clip sampler | `matchlab_train/datasets/footpass_clips.py` | 17 |
| DST window dataset + pitch symmetry | `matchlab_train/datasets/footpass_windows.py` | 28 |
| 4 experiments (train / infer / score / denoise) | `matchlab_train/experiments/pcbas_*.py` | 15 |

Full suite: **1,247 passed, 15 skipped**. `ruff` clean.

---

## Measured end to end

The whole stage-1 chain has been exercised on real FOOTPASS footage, with a
deliberately under-trained model, to prove the plumbing before committing hours of GPU
time to it.

| step | measurement |
|---|---|
| Training loop | 2.2 clip/s; loss falls; checkpoints written and reloaded |
| Clip read (seek + 50 frames) | **0.04 s** — clip I/O is not a training bottleneck |
| Training step, batch 1 | **0.39 s**, peak **6.2 GiB**; batch 2 OOMs |
| Inference | **0.32 s** per 50-frame window at stride 25, full frame coverage |
| Decode → score | runs; produces a report; exports the reference's JSON format |

### The GPU decided the recipe

An RTX 4060 Ti with ~12.5 GiB usable (the desktop holds ~3.7 GiB) cannot fit the
reference's batch of 6. Micro-batch drops to 1 and `accum_steps` rises to 48, holding
the effective batch at the reference's 48. That keeps the deviation a memory
workaround rather than a hyperparameter change — which matters, because a
hyperparameter change would make any reproduction miss unattributable between the
recipe and the implementation.

The only real consequence is BatchNorm statistics, now computed over one clip:
50×44×80 samples per channel for the 3D norms and M·T = 250 for the 1D norm. Both
remain well conditioned.

---

## Bugs the tests caught during implementation

Recorded because each would have produced a plausible model rather than a crash.

**1. The loss ignored the observability mask.** A masked cell's pooled feature is
forced to exactly zero, so including it trains the classifier to map the zero vector
to background — on roughly 60% of cells, since that is how often a player is
off-screen. Found by reading the reference's loss after writing my own.

**2. An invented class weighting.** I added softened inverse-frequency class weights
and a test immediately showed absent classes driving `pass` to a weight of 3.5e-5,
effectively deleting the most common class from the loss. Fixing the arithmetic was
the wrong repair: the reference uses flat `[0.05, 0.95 × 8]` weights and handles
balance in the *sampler*. A second frequency correction in the loss double-counts it.
Reverted to the reference, and the episode is recorded in `class_weights`'s docstring
so it is not re-invented.

**3. Inference rescanned 1.6M rows per window.** 0.28 s of a 0.45 s window against a
0.13 s model forward. Sorting once and slicing with `searchsorted` took it to 0.32 s.

**4. Sub-window averaging halved the sequence edges.** The reference divides its whole
overlap region by 2, including the first and last 25 frames of every half, which only
one tiling covers. `WindowAccumulator` divides by actual per-frame coverage, so the
denoiser does not receive systematically under-confident edges.

**5. `memory_key_padding_mask` was never passed.** The reference omits it too.
Measured: PyTorch's nested-tensor encoder fast path happens to make padded memory
positions independent of their input, so the omission is currently harmless — but
that is an implementation detail, not a guarantee. Now passed explicitly.

---

## Design decisions worth their own line

**`PCBASEvent` lives in `matchlab_core`, not `matchlab_train`.** The plan had the
metric importing the event model from the train package, inverting the dependency
direction.

**The module is `footpass_pcbas.py`, not `footpass.py`.** An unmerged branch
(`spo-position-evidence-reid`) already has a `datasets/footpass.py` re-ID
observability loader. Different responsibility, and taking the name would guarantee a
conflict over a file neither branch owns.

**`half_to_events` returns a new `PCBASEvents`, not `EventGroundTruth`.**
`GroundTruthEvent` has no player field, so it cannot carry the slot this whole task is
about. `to_event_ground_truth()` is the explicit lossy downcast. This is the schema
gap the design doc named, now concrete.

**Two identity modes, always both reported.** `slot` is what the model predicts;
`shirt` is the reference's exchange identity, reachable only through the ADR 008
per-frame roster remap. Only `shirt` is comparable to the reference's 0.4100 / 0.7186.
The gap between them is the measured cost of the remap rather than an assumed zero.

**Both DST encoders exist behind one interface.** `flat` is the reference's; `attn` is
the PAVE-style per-player attention the 2026 winner used. The structural difference is
*asserted*, not described: the attention encoder's parameter count is independent of
the number of slots, which is why it can generalise to an unseen formation. Since
TRAIN uses all 26 slots but VAL only 22, that property is not academic.

---

## What is NOT done

- **No trained model.** The action head has run only as a 2-epoch smoke on VAL.
- **No Phase 1 gate.** Task 10 compares against the reference's VAL micro-F1 (0.4100
  TAAD, 0.7186 TAAD+DST). Nothing here may be quoted as a reproduction.
- **The denoiser has never been trained.** It needs stage-1 logits for all 96 TRAIN
  halves — roughly 8 GPU-hours of inference before its own training starts. That
  prerequisite, not the training, is the schedule risk.
- **ROI alignment on a TRAIN match is unverified.** Verified on VAL only; the check
  runs automatically when extraction completes (`scratchpad/align_check_train.py`).
- **The control run is not done.** The plan requires running the reference unmodified
  on one match, so that a gate miss is attributable between our reimplementation, our
  ingest, and the setup.

## Reproduce

```bash
# from the MAIN checkout, not a worktree -- configs use paths relative to data/
E=packages/matchlab_train/src/matchlab_train/experiments
uv run matchlab-train run $E/pcbas_action_head.yaml    # stage 1 training
uv run matchlab-train run $E/pcbas_infer_logits.yaml   # (9,26,T) over VAL
uv run matchlab-train run $E/pcbas_score.yaml          # decode + score
uv run matchlab-train run $E/pcbas_denoiser.yaml       # stage 2 (needs TRAIN logits)
```
