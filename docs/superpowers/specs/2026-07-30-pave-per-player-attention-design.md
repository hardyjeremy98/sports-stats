# PAVE — per-player attention, and the stage-1 changes underneath it

**Date:** 2026-07-30
**Branch:** `worktree-spo-action-spotting-prd`
**Linear:** SPO-96 (Phase 2)
**Supersedes nothing.** Amends the Phase 2 row of
[`2026-07-27-player-centric-action-spotting-design.md`](2026-07-27-player-centric-action-spotting-design.md)
§3, which specified "export-time roster remap + PAVE attention" from a one-line
description of PAVE. The actual paper is now in hand and says something different.

**Prior state:** [Phase 1 gate](../../reports/2026-07-27-pcbas-phase1-gate.md),
[DST investigation](../../reports/2026-07-28-pcbas-dst-investigation.md).

---

## 1. What PAVE actually is

**"SoccerNet 2026 Player-Centric Ball Action Spotting: Per-Player Attention with
Agreement-Based Ensembling"** — Faisal Altawijri, Ismail Mathkour (TAHAKOM),
[arXiv:2606.28389](https://arxiv.org/abs/2606.28389), 23 Jun 2026. Two pages,
first place on the PCBAS challenge leaderboard at **58.94 macro-F1**
([SoccerNet 2026 results](https://arxiv.org/abs/2607.07320), Table 2).

Its own progression, from the paper's Table 1:

| Configuration | VAL macro-F1 | Challenge macro-F1 |
|---|---:|---:|
| DST with **improved TAAD** | 0.541 | 48.6 |
| + per-player attention | 0.549 | — |
| + spatial-first order | 0.567 | 55.1 |
| 4-model ensemble | **0.609** | **58.9** |

"Val F1" is macro-F1: the paper's per-class Table 2 averages to exactly these
figures (Model A's eight classes mean 0.567; the ensemble's mean 0.609). Its
"6 validation games" are halves, so the split is our VAL — 3 matches, 6,070
events. That makes the comparison to our numbers legitimate, with the caveat in
§6.

PAVE is **three separable contributions**, not one.

### 1.1 Stage-1 (TAAD) improvements — §2 of the paper

An **L=4 temporal transformer** on the ROI-pooled per-player features: project to
`d=256`, four encoder layers with 8 heads, feedforward 1024, GELU, **pre-norm**,
learnable positional embeddings (up to 50 frames), then project to 512 before the
9-class head. The stated purpose is "to reason about temporal patterns across
frames rather than classifying each frame independently."

Plus five training changes: a fixed **warmup scheduler bug** (the LR stopped
increasing prematurely and stayed at its warmup value); cosine annealing for all
layers **except** X3D, which takes a fixed `5e-5` after warmup; both LRs divided
by `√8`; warmup set to 100 steps **with no gradient accumulation**; background
class weight reduced to **0.05**; and tracklet count raised from **4 to 6 at
epoch 16**.

The resulting logits are frozen input to every DST experiment.

### 1.2 Per-player attention — §3.1–3.2

A two-stage attention block **before** the main transformer encoder, on features
reshaped to per-player representations:

- **Stage 1 (Spatial)** — cross-player attention at each frame. All 26 players at
  a timestep attend to each other through one transformer encoder layer,
  `d_p=64`, 4 heads.
- **Stage 2 (Temporal)** — per-player self-attention across frames. Each player's
  Stage-1 representation is processed independently over time with sinusoidal
  positional encoding, through another encoder layer.

Then: "The attended features are mean-pooled across players and concatenated with
the game-state logits per frame, then projected and **added to the main encoder
input**."

Two findings the paper reports explicitly, both of which contradict our current
stub:

1. **Game-state channels only.** "We found that using only game-state features
   (positions, velocities, visibility) for the per-player attention, excluding
   TAAD logits, outperforms using all channels."
2. **Spatial-first ordering is worth more than the attention itself** — spatial
   then temporal beats temporal then spatial by **+1.87% macro-F1 on validation
   (0.549 → 0.567)**, against the attention block's own +0.008.

### 1.3 Ensemble — §4

Four DST variants (A: `d=512`, 1 attention layer, `d_p=64`; B: `d=768`; C: 2
attention layers, `d_p=128`; D: parallel rather than sequential branches, summed),
combined by **Weighted Event Fusion**: group predictions by (team, shirt, class),
cluster within ±12 frames by greedy assignment sorted on descending score, score
each cluster as the mean of contributing models × `(n/N)^0.5`, then discard any
event supported by fewer than 2 models — **except tackle**, which bypasses
agreement filtering entirely because the filter was deleting the only correct
tackle predictions.

All four models: 6 encoder/decoder layers, 8 heads, dropout 0.1, LR `2.5e-4` with
exponential decay ×0.1 at epochs 3, 6, 8, **10 epochs**, best epoch chosen by a
checkpoint sweep.

### 1.4 Two incidental confirmations

§3.4 reports that tuning per-class thresholds on the 6 validation games
**overfits** — 48.0 vs 48.6 on challenge — and that τ=0 with default thresholds
wins. That independently corroborates our own
["the decode threshold is NOT the recall lever"](../../reports/2026-07-27-pcbas-phase1-gate.md)
finding. Neither is a lever; we adopt no threshold work.

The paper states tackle has "26 occurrences across 96 training games". Our ingest
gate, which passes exactly against the published event counts, measures **26 in
VAL and ~390 in TRAIN**. We trust our ingest and note the discrepancy rather than
resolving it.

---

## 2. What transfers to us, and what does not

| PAVE change | Our state | Decision |
|---|---|---|
| **L=4 temporal transformer** | `action_head.py:116` is `Conv1d(192→512, k=3)` — a **3-frame** receptive field | **Adopt.** The one architectural change, and §3 argues it is the one that matches our measured deficit. |
| Cosine anneal head, fixed `5e-5` X3D | `pcbas_action_head.py:70` steps **both** groups ×0.1 at epoch 10 | **Adopt.** |
| Warmup 50 → 100 steps | `warmup_steps: 50` | **Adopt.** |
| Tracklets 4 → 6 at epoch 16 | `nb_tracklets: 4`, fixed | **Adopt.** |
| Warmup scheduler bug fix | We never had the bug — `lr_scale` (`pcbas_action_head.py:101`) ramps to 1.0 and stays | **No-op.** We reimplemented rather than vendored, so we did not inherit it. |
| Background class weight 0.05 | Already 0.05 (`pcbas_action_head.py:73`) | **No-op.** |
| **Both LRs ÷ √8** | — | **Reject.** This is *coupled* to "no gradient accumulation": they went from effective batch 48 (6×8) to 6 and applied sqrt-scaling. We are micro-batch 1 × accum 48 because one `(1,3,50,352,640)` clip peaks at 6.2 GiB on a 16 GiB card. Our effective batch stays 48, so copying the LR without the batch change is a de-tuning wearing a fix's clothes. |
| Per-player attention | `SlotAttentionEncoderEmbedding` (`denoiser.py:93`) is a *guess* — built, shape-tested, never trained | **Rebuild** per §4.2. |
| DST LR: exp decay ×0.1 at epochs 3/6/8 | — | **Reject, already refuted here.** `pcbas_denoiser.py:56-62` records that copying those epoch numbers annealed us to `2.5e-7` before convergence and flatlined the run: **micro-F1 0.048 vs 0.119**. The reference takes ~2,000 optimiser steps per epoch to our ~200, so its "epoch 3" is 6,000 steps in and ours is 600. Do not re-run this hypothesis. |
| 4-model ensemble + Weighted Event Fusion | — | **Out of scope** (§7). |
| τ / per-class threshold tuning | Independently found not to be a lever | **Reject** (§1.4). |

---

## 3. Why stage 1 goes first

Two facts decide the ordering.

**PAVE's own numbers say stage 1 is where the distance is.** Their improved-TAAD
DST scores **0.541** VAL macro against the reference's **0.4926** — so their
stage-1 work alone is worth **+0.048**, more than the attention (+0.008) and the
ordering (+0.018) combined. Everything after that is refinement on a working
system.

**Our DST cannot currently measure a change of that size.** PAVE's attention
delta is +0.008 macro-F1 on a DST scoring 0.541. Ours scores **0.060**. At that
operating point a +0.008 change is indistinguishable from seed noise, and the
Phase 2 gate as originally written ("PAVE arm ≥ flat arm on macro-F1") would be
decided by a coin flip. That is a negative finding scoped to a decision rule that
cannot resolve it — we would be measuring PAVE's encoder under a test blind to the
effect size it reports, and recording the result as though the test had been
capable of seeing it.

There is also a free control available first. Our last stage-1 run stopped at
**12 of 20 epochs** on a wall-clock budget, still improving at the last one, and
"finish stage 1's schedule" was already item 1 on the DST investigation's own
next-steps list. On a faster GPU the full schedule may fit. Running the
**unchanged** config to completion tells us how much of the 0.3274 → 0.4100 gap
was simply the truncated schedule, before we attribute any of it to architecture.

---

## 4. Architecture

Two files change. Both follow the pattern `denoiser.py` already uses: a `Literal`
kind selector with the existing behaviour as the default, so the control arm stays
runnable from the same code and the ablation is a config flag, not a branch.

### 4.1 `matchlab_core/pcbas/action_head.py` — temporal transformer

```python
TemporalKind = Literal["conv", "transformer"]
```

- `conv` (default, the control): today's `Conv1d(192→512, k=3)` + BN + GELU.
- `transformer`: `Linear(192→256)` → learnable positional embedding (max 50
  frames) → 4 × `TransformerEncoderLayer(d_model=256, nhead=8,
  dim_feedforward=1024, activation="gelu", norm_first=True, dropout=0.1)` →
  `Linear(256→512)`.

`Linear(512, 9)` is unchanged in both. The transformer runs on
`pooled.reshape(b*m, T, 192)` — post-ROI-pool features, not video — so its cost
is negligible beside the X3D pass.

**Dropout 0.1 is our choice; the paper does not state stage-1 dropout.** Recorded
as a deviation, matching the DST value.

**Two hazards the conv did not have.** The pooled tensor is *zeroed* wherever the
observability mask is 0 (`action_head.py:93`), and a player is off-screen roughly
60% of the time. A conv over zeros is harmless; attention over zeros is not — the
model would attend to fabricated frames.

1. Pass `src_key_padding_mask = ~mask` so absent frames are excluded from
   attention.
2. **A player observed in zero frames yields an all-masked sequence, which makes
   PyTorch attention return NaN.** Detect this case and emit zeros for that
   player without calling the transformer. Those cells are masked out of the loss
   anyway, but a NaN propagates through the whole batch's gradient and would
   present as an unexplained training collapse.

### 4.2 `matchlab_core/pcbas/denoiser.py` — per-player attention

The current `SlotAttentionEncoderEmbedding` **replaces** the flat projection
(`denoiser.py:132`). PAVE **adds** to it. That difference is not cosmetic: as an
additive branch, disabling the attention recovers the flat arm *exactly*, which
makes the ablation a pure addition rather than a substitution of two things at
once.

```python
emb = FlatEncoderEmbedding(src) + PerPlayerAttentionBranch(src)
```

`PerPlayerAttentionBranch`, faithful to §3.1:

1. `slots = src[..., :364].reshape(B, T, 26, 14)`. Our layout is slot-major, which
   the DST investigation already noted is *required* for this encoder.
2. **Game-state channels only** by default: `slots[..., :5]` — x, y, vx, vy,
   observable. `Linear(5 → d_p)`. A `use_logits_in_attention` flag keeps the
   14-channel variant available, since that is a claim of theirs worth being able
   to re-test rather than assume.
3. **Spatial:** reshape `(B*T, 26, d_p)`, one `TransformerEncoderLayer(d_p, 4
   heads)`.
4. **Temporal:** reshape `(B*26, T, d_p)`, add `sinusoidal_positional_encoding` at
   `d_p` over **window-local** frame indices — the same convention whose violation
   cost 3.4× in bug 1 — then a second encoder layer.
5. **Pool:** mean over the 26 slots → `(B, T, d_p)`.
6. **Concat + project:** concatenate the 234 TAAD logit channels
   (`slots[..., 5:]` flattened) → `Linear(d_p + 234 → hidden_dim)`.

**Step 6 is our reading of an ambiguous sentence.** The paper says "concatenated
with the game-state logits per frame", which is not a quantity that exists — the
game-state channels and the TAAD logits are different things, and the attention
has just excluded the latter. Their Figure 1 labels the same box "Pool + Concat
logits". We read it as re-introducing the 234 TAAD logit channels the attention
deliberately skipped, which is the only reading under which the branch carries
information the flat projection does not already have. Recorded as an
interpretation so that a B1 miss stays attributable to it.

`order: Literal["spatial_first", "temporal_first", "parallel"]` — the first two
are the §3.2 ablation, `parallel` is their Model D (branches on the raw input,
summed).

New params: `attn_order`, `attn_dim` (`d_p`, default 64), `attn_layers`,
`attn_use_logits`. `encoder: flat` remains the default.

Absent slots carry `ABSENT_FILL = -15.0`, not zero — deliberately out-of-range so
"absent" is learnable rather than confusable with the origin. That survives the
rebuild unchanged; no slot masking is added, because the fill *is* the signal.

---

## 5. Phasing and gates

Every gate states a number or an explicit "report, no bar". Arms run in order;
each changes one thing against the arm before it.

| Phase | Arm | Gate |
|---|---|---|
| **0** | Migrate to the office PC; re-run `pcbas-action-head` **unchanged**, 20 epochs, no `max_hours` | Completes 20 epochs. VAL micro/macro **reported, no bar** — this is the control every later arm is measured against. |
| **A1** | + temporal transformer (§4.1), nothing else | **Carry forward only if VAL micro-F1 > Phase 0 control.** Target: reference parity, micro **0.4100** / macro **0.2445**. |
| **A2** | + cosine anneal / fixed X3D LR / warmup 100 / tracklets 4→6 at epoch 16 | Beats A1 on VAL micro-F1. If it does not, keep A1's checkpoint and record the schedule changes as not transferring. |
| **B0** | Re-infer TRAIN+VAL logits from the best stage-1; retrain **flat** DST, **two seeds** | Reports the new flat baseline **and the run-to-run variance**. See §5.1. |
| **B1** | + per-player attention, spatial-first, game-state-only (§4.2) | **Macro-F1 gain must exceed the B0 seed spread.** Anything smaller is unresolvable and is recorded as such, not as a win. |
| **B2** | temporal-first, identical otherwise | Fidelity check — see §5.2. |

### 5.1 The noise floor is a prerequisite, not a nicety

PAVE's headline ordering effect is **+0.018 macro-F1**, and the attention block's
own contribution is **+0.008**. We have never measured our DST's run-to-run
variance. If it is wider than 0.018, no ablation of this encoder means anything,
and we would spend days interpreting noise.

Two identical flat-DST runs at different seeds cost one extra DST run — cheap
beside a stage-1 run — and they also **discharge the retracted oracle
experiment's outstanding debt**, whose stated fix was to re-run the real-input arm
at a matched budget. B0 does that by construction: both seeds run the same epochs
on the same input.

### 5.2 B2 tests our reimplementation, not our score

B2 is the only self-validating check available. PAVE's sole internal ablation of
the attention block is the ordering, and it is a *relative* claim: spatial-first
ahead of temporal-first by +0.018. That is testable even when our absolute score
is far from theirs.

If spatial-first and temporal-first are indistinguishable beyond the B0 noise
floor, our module is not doing what theirs does — a reimplementation fault, and
a far more useful thing to learn than "the number did not go up."

---

## 6. Comparability — what any of these numbers can claim

The Phase 1 spec's warning stands and tightens here. **58.94 and 46.41 are
challenge-split leaderboard figures**; D6 commits us to VAL because CHALLENGE
labels are withheld. Every bar above is set against VAL.

PAVE's VAL figures (0.541 / 0.567 / 0.609) are *their own* reproduction, not
shipped predictions we can re-score. Our 0.4926 reference comparand comes from the
reference's shipped VAL predictions and is reproducible locally. So PAVE's VAL
numbers are a **target to aim at, never a gate to pass** — a miss against them
could be theirs, ours, or the difference between two reproductions.

The prior comparability caveats are unchanged: FOOTPASS *supplies* tracking,
jersey and role as inputs, so these baselines measure a strictly shorter pipeline
than MatchDay's, and per [ADR 008](../../decisions/008-role-slots-are-not-roster-slots.md)
role slots are not roster slots.

---

## 7. Out of scope

- **The 4-model ensemble and Weighted Event Fusion (§1.3).** It is worth +0.042
  VAL macro — the single largest contribution after stage 1 — and it is still the
  last thing to build: it costs 4× the training and it can only average away
  variance in models we do not yet have. Revisit when B1 lands.
- **Solo tackle exception.** Belongs with the ensemble; meaningless without it.
- **τ / per-class threshold tuning** (§1.4).
- **Export-time roster remap (ADR 008).** Named alongside PAVE attention in the
  Phase 2 row of the earlier spec, but independent of it. Unchanged and untouched
  here.
- **Stage-1 augmentations** (affine/scale/crop, dropped because `albumentations`
  is not packaged). Item 2 on the DST investigation's next-steps list, and a
  genuine confound with A1 if bundled. It gets its own arm later, or none.

---

## 8. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Machine change confounds the control** | The 0.3274 control was measured at micro-batch 1 with BatchNorm over a single clip, on a 16 GiB card. A different effective batch changes BN statistics and the architecture in one move | The office PC has the **same 16 GiB VRAM**, so micro-batch 1 × accum 48 is pinned unchanged and the control carries. **Do not raise the micro-batch**, however much headroom a faster GPU appears to offer. |
| **Attention delta below the noise floor** | Would produce a confident, unfalsifiable Phase 2 conclusion | B0 measures the floor *before* B1 is interpreted (§5.1) |
| **NaN from fully-absent players** | Presents as an unexplained training collapse, not as a masking bug | Explicit guard and a unit test (§4.1) |
| **Bundling A1 and A2** | Four changes in one run, unattributable either way | Separate arms; A2 is gated on A1 |
| **Both machines committing to the branch** | Divergent `history.json` and experiment outputs | After migration the office PC is the **only** place this branch advances |

---

## 9. Testing

- **Attention-off equivalence.** With the attention branch disabled, the encoder
  output is bitwise identical to the flat arm. This is what makes B1 a pure
  addition; without it the ablation compares two changes.
- **Fully-absent player yields no NaN**, in both `action_head` and the encoder.
- **Absent-slot fill survives** the reshape into per-slot tokens — `ABSENT_FILL`
  reaches the projection, rather than being silently zeroed.
- **Window-local frame convention** in the temporal attention's positional
  encoding, asserted directly rather than inferred from a shape.
- The existing `build_tokens` → `tokens_to_events` round trip must still pass
  unchanged. It is the test whose absence cost 3.4×.
- Shape/dtype contracts for `(9, 26, T)` fp16 across both stages, as today.
