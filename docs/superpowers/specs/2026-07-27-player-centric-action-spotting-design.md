# Player-centric action spotting (B3+B4) — architecture review and design

**Date:** 2026-07-27
**Status:** design, pending review
**Supersedes in part:** `docs/prds/action-spotting-possession-transition.md` Phase 2
(`possession-peral`) — see "What this supersedes" below.
**Notion:** [B4 — Reproduce FOOTPASS TAAD→DST, then swap jersey ID for roster slot](https://app.notion.com/p/3a9a932a314581c39697c162f5c9d525),
[B3 — Action Spotting](https://app.notion.com/p/3a6a932a31458161a3c1ed47f97a3160),
[B4 — Action Attribution](https://app.notion.com/p/3a6a932a31458183a42ec45a5ac56cfa)
**Linear:** project *Possession-Transition Spotting (B3)*; related SPO-76, SPO-83, SPO-95

---

## Part 1 — Architecture review

### 1.1 What the reference system actually is

Read from `github.com/JeremieOchin/FOOTPASS` at Apache-2.0. Every number below is
from the source, not the paper.

**Stage 1 — TAAD** (`models/model_TAAD_baseline.py`, 71 lines)

```
input   (B, 3, T=50, 352, 640) video  +  (B, M, T, 5) ROIs  +  (B, M, T) masks
        X3D-S (pretrained, pytorchvideo)
        FPN: upsample+concat blocks 4→3→2  →  (B, 192, T, 44, 80)   [stride 8]
        roi_align(output 4×2, spatial_scale 0.125)   ← per-player pooling
        mask out unobserved players
        Conv1d(192→512) over time  →  FC  →  9 classes
output  (B, 9, M, T)
```

The single most important structural fact: **TAAD does not crop player pixels.**
It runs the whole frame through one backbone and pools each player off a shared
stride-8 feature map. Resolution enters through the feature map, which is why
352×640 is the design point and not a compromise.

**Stage 2 — DST** (`models/model_DST.py`, 126 lines)

Not a per-frame denoiser — a **sequence-to-sequence Transformer** that *translates*
a window of noisy per-player logits into a clean event list.

```
encoder src   (T=750, 1116)  = 130 slot-kinematics ‖ 234 slot-logits ‖ 752 one-hot frame index
                               where 130 = 26 slots × 5  and  234 = 26 slots × 9
                               the 5 channels are x, y, vx, vy, observed-flag
                               (4 kinematic + 1 observability; sentinel -15.0 when unobserved)
                               sinusoidal positional encoding is added on top
decoder tgt   per event: action one-hot(10) ‖ role one-hot(27) ‖ timestamp one-hot(752)
              6 enc + 6 dec layers, d_model 512, 8 heads, autoregressive at inference
```

The 364-wide block is the *dataset's* per-frame feature vector; the model prepends a
752-wide one-hot absolute frame index before embedding
(`model_DST.py:35`, `train_DST.py:198`). The fifth channel being an **observability
flag**, not kinematics, is worth naming: it is the reference's own mechanism for
quality-gating missing evidence, which is ADR 003's principle arrived at independently.

**Stage 3 — export** (`utils/metric_utils.py`): the model predicts a **role slot**,
never a jersey. Jersey is attached afterwards by a per-frame role→shirt lookup, then
events export as `(frame, team, shirt_number, class, score)`.

### 1.2 The three findings that shape our design

**(a) The slot is a TACTICAL ROLE slot, not a roster slot — and it is not a stable
player key.** An earlier draft of this document called the reference "already
roster-slot native" and treated ADR 007's substitution as a lookup-table swap. **That
was a category error**, and the measurements below refute it.

`slot = left_to_right * 13 + (role_id - 1)`, M=26 (`DST_Dataset.py:113-114`). Measured
on `val_tactical_data.h5`:

| fact | measurement |
|---|---|
| `left_to_right` **flips at half time** | **17 of 18** shirts present in both halves of `game_18` swap sides; `role_id` is unchanged for 17 of 18 |
| only **11 of 13** roles are used per side | roles 4 and 8 never appear; 22 of 26 slots occupied, in both games checked |
| substitutes **reuse** a slot within a half | 1 slot in `game_18_H1`; ~15% of VAL events fall in slots shared by two players |

Three consequences, none of which the earlier draft survived:

1. **ADR 007's "per-match one-to-one assignment between slot and identity" is
   impossible.** The relation is per-**half**, and piecewise-constant *in time* within
   a half because of substitutions. FOOTPASS's own export uses exactly that — a
   per-frame role→shirt lookup — which is the tell we should have read.
2. **ADR 007's mechanism claim — that DST's identity channel is an "anchor-agnostic"
   token needing "a key, not a name" — is falsified by the source.** The channel
   carries *tactical position*. A left-back's positional prior is the entire reason
   the sequence stage lifts precision. Substituting an arbitrary roster index into the
   26-channel encoder would destroy the prior, not preserve it.
3. Therefore the roster-slot substitution is an **export-time remapping only**, and it
   changes the metric's matching key from `(team, shirt_number, class)`
   (`metric_utils.py:482`) to a roster key — so those numbers are non-comparable to any
   FOOTPASS figure *in addition to* the longer-pipeline caveat in §1.6.

**This supersedes ADR 007's mechanism.** Recorded as ADR 008; a spec cannot amend an
ADR (CLAUDE.md governance), so the ADR is the authority and this section merely cites
it.

**(a2) `left_to_right` is attacking direction — we do not produce it, and getting it
wrong fails silently.** It is not `teams.json`. It needs pitch geometry, a side
determination, and half-boundary detection, and it **inverts at half time**. Our team
stage emits unordered cluster labels with no side semantics, and
`implementation-status.md` lists team/role accuracy metrics as **Not implemented**.

A side error shifts all 26 slots by 13 at once, so per-team role accuracy can read
100% while every attribution is wrong. **Side/half assignment is therefore a
first-class capability with its own gate**, not an arrow in a diagram.

**(b) Role assignment is the linchpin, and we do not have it.** M=26 role slots are
required at *TAAD inference* as well as at DST — `run_TAAD_on_matches.py` builds
`(26, T, 5)` ROI tensors indexed by slot. FOOTPASS supplies role; MatchDay must
produce it. Nothing else in the chain is blocked on data we lack. **This is the
critical path, not the visual model.**

**(c) A soft visual-recall bound, not a hard ceiling.** Of 6,070 VAL actions, **1,062
(17.5%) have no bounding box at the event frame** — verified against
`playbyplay_GT/playbyplay_val.json`.

An earlier draft of this document called that a hard ~82.5% ceiling on any visual
model. **That was wrong**, and the reference's own code refutes it: `metric_utils.py`
splits true positives into `TP_bb` / `TP_nobb` precisely because no-box events *can*
be matched — the ±12-frame tolerance and the temporal Conv1d let a prediction land on
one. Measured on the reference's shipped predictions: **TAAD alone recovers 33** of
the 1,062; TAAD+DST recovers **390**.

Two further corrections to that draft: "no bounding box" is **not** the same as
"off-screen" — the reference tracks them as separate flags, and only **614 VAL events
(10.1%)** are on replay frames with no visible players at all.

The two-stage argument survives, and is in fact sharpened: the sequence stage lifts
no-box recovery from 33 to 390, a **12×** improvement on exactly the subset local
visual evidence handles worst. That is a measured argument for the split, where the
"hard ceiling" version was an overstated one.

Note also that our ceiling is strictly *below* FOOTPASS's, since our own detector and
tracker misses stack on top of the dataset's.

**(d) Slot *stability*, not slot accuracy, is the likeliest killer.** DST accumulates
30 seconds (750 frames) of history against a fixed slot index. Its value depends on
that index meaning the same physical player for the whole window. Against our pipeline
the index is perturbed by:

- **tracklet fragmentation** — `implementation-status.md` records 20.7% missing
  frame-time with gaps **up to 482 consecutive frames**, which is **64% of a single
  DST window**;
- re-ID merge/split churn;
- substitutions reusing a slot mid-window (measured above);
- the half boundary, which re-permutes every slot at once.

Per-frame role accuracy can look respectable while the slot index is effectively
re-permuted several times per window. **Role accuracy and slot stability are different
quantities and need separate metrics** — slot-switch rate per entity per window, and
window-level slot purity.

### 1.3 Target architecture

```
video ─┬─> detect ─> track ──────────────> tracklets.json
       │                    ├─> team ────> teams.json   (cluster labels, NO side semantics)
       │                    ├─> calibrate ─> homography
       │                    └─> fuse/minimap ─> pitch x,y + velocity
       │                                   │
       │                     ┌─────────────▼──────────────┐
       │                     │ SIDE + HALF  (NEW)          │  team → left_to_right,
       │                     │ inverts at half time        │  half boundary detection
       │                     └─────────────┬──────────────┘
       │                     ┌─────────────▼──────────────┐
       │                     │ ROLE ASSIGNMENT (NEW)       │  11-of-13 per side, per half
       │                     │ → 26 slots + STABILITY      │  ← critical path
       │                     └─────────────┬──────────────┘
       └────────────────────────> ACTION HEAD (TAAD-equivalent)
                                           │  (9, 26, T) logits
                                           ▼
                                 SEQUENCE DENOISER (DST/PAVE)
                                           │  (action, role_slot, frame)
                                           ▼
                     ┌─────────────────────▼──────────────────────┐
                     │ EXPORT-TIME remap: (half, slot, interval)   │  NOT a bijection
                     │   → identity        [needs an anchor]       │  see §1.2a
                     └─────────────────────┬──────────────────────┘
                                           ▼
                     events.json + spotting.json  [+ identity field — NEW]
```

**What we already own — with the record's own caveats attached:**

| capability | caveat that the earlier draft omitted |
|---|---|
| detect, track | 20.7% missing frame-time; gaps up to 482 frames = 64% of a DST window |
| team | **no accuracy metric exists** ("Not implemented"); no side semantics |
| calibrate, fuse/minimap | merged at `12aca03`; **absolute** pitch accuracy has never been measured — Gate 2 tracks the error's time-derivative, not its magnitude, and role priors are absolute-position priors |
| B2 re-ID engine | its do-no-harm gate passed **using oracle jersey anchors from GT**. On FOOTPASS there is no non-OCR anchor, so `slot → identity` has **no runtime input** — ADR 007's per-match assignment is an *evaluation device*, not a pipeline stage |
| artifact/stage machinery, event scoring | sound |

**What is new — the earlier draft named one of these; there are six:**

1. **Side + half determination** (§1.2a2)
2. **Role assignment** (11-of-13, per half)
3. **Slot stability** as a measured property (§1.2d)
4. Action head
5. Sequence denoiser
6. FOOTPASS ingest + the macro-F1 metric

Plus two schema gaps neither draft acknowledged: `GroundTruthEvent` has **no player
field** (it is class+time+half), and `SpottedEvent` has **no identity field** — and
the latter is the *frozen external exchange contract* shared with the T-DEED CLI, so
adding one means versioning that contract, not editing a model.

**Deliberately unowned in this design:** "calibrated confidence" appears in the B4
Notion page as the gate for auto-attribution vs VLM vs HITL. It has no design here and
no phase. It is named so it is not mistaken for solved.

### 1.4 Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **In-repo dependency, not an isolated env** | The reference ships **contradictory licence statements** — `LICENSE` is Apache-2.0, `README.md:308-312` says the baselines are CC BY-NC 4.0. Unresolved, and per CLAUDE.md's research posture it **gates nothing** either way. The `external-spotters/` isolation exists for GPL T-DEED and a conflicting dependency set; neither applies. torch is already in the `cv` extra with in-repo model precedent (`stages/track/tdlp`, the associate embedders). |
| D2 | **Adapt, don't vendor wholesale** | We take the architecture and the `(9,26,T)` contract, and write our own modules against our schemas. Vendoring 4,260 lines of research code we cannot test is worse than reimplementing 200 lines we can. |
| D3 | **`(9, 26, T)` fp16 is the frozen stage-1↔stage-2 contract** | It is the reference's own interface (`avg_logits_*.npy`), so freezing it lets either side be replaced independently and makes stage 2 developable against synthetic logits. **Correction:** an earlier draft claimed this lets stage 2 be scored against *reference* logits before our stage 1 exists. It does not — the repo ships **no checkpoints and no logits** (`TAAD_predictions/` and `TAAD_h5/` hold only placeholder `.txt` files, and `run_TAAD_on_matches.py:22` defaults to an absent `curr_model_19.pt`). What *is* shipped is `playbyplay_PRED/playbyplay_{TAAD,DST}_val.json` — final event lists, which validate the **metric** end-to-end (see D8). |
| D8 | **Validate our metric against the reference's shipped predictions before anything else** | `playbyplay_PRED/*_val.json` + `playbyplay_GT/playbyplay_val.json` are in the repo and need no video, no NDA and no GPU. Reproducing the reference's own scores exactly is the cheapest possible correctness gate, and it catches metric bugs before they can be mistaken for model quality. |
| D4 | **Role assignment gets its own module and its own gate** | It is the critical path and the only wholly-new capability. It must be measurable in isolation against FOOTPASS's supplied roles before anything downstream consumes it. |
| D5 | **Two new stage slots: `role` and reuse of `spotting`** | Role is a per-entity property like `team`, and belongs in its own slot after `fuse`. The action head + denoiser together fill the existing `spotting` slot. |
| D6 | **Score on FOOTPASS VAL only; CHALLENGE is unusable** | CHALLENGE tactical data ships with 13 columns — no `class`. Labels are withheld. VAL (3 matches, 6,070 events) is the only labelled held-out data. |
| D7 | **Synthetic-noise training of DST is a development harness, never a result** | Real stage-1 errors are correlated in time, space and class; i.i.d. injected noise is not. Any DST number from synthetic input is explicitly not reportable. |

### 1.5 What this supersedes, and by what authority

**A spec cannot supersede a PRD or an ADR.** `docs/superpowers/specs/` does not appear
in CLAUDE.md's precedence list at all. So this section only *records* supersessions
enacted elsewhere:

- **ADR 008** supersedes ADR 007's mechanism claim (§1.2a). The ADR is the authority.
- The **PRD carries a supersession banner** on its Phase 2 (`possession-peral`), added
  2026-07-27. Peral's block 2 is a smoother over a possession likelihood; DST is a
  tactical seq2seq translator — same slot, materially different capability, and DST has
  released code and an external benchmark.
- `docs/reference/footpass-pcbas-acquisition.md` previously claimed the same
  supersession. That claim is now redundant; the PRD banner is authoritative.

The possession track is **not** discarded. `possession-viterbi` and
`transition_to_events` remain the calibration-free, role-free path — the only path that
runs on tiers with no pitch keypoints, which today is every tier except FOOTPASS.

It is *a candidate* input channel to the denoiser, with a caveat that must travel with
the offer: the possession signal has **no measured possessor accuracy on any tier**
(`docs/reports/2026-07-27-b3-possession-denoise-ablation.md`), and its results are
oracle-input only.

### 1.6 Honest limits, carried forward

- **Not leaderboard-comparable.** FOOTPASS supplies tracking, jersey and role; we
  produce all three. Any figure we quote measures a strictly longer pipeline.
- **τ=0.15 is one point on a curve.** The benchmark's fixed high-recall operating
  point is the opposite end from MatchDay's abstention design. The internal
  precision-vs-abstention curve stays the frontier measurement (B2 lesson).
- **Two classes are near-unlearnable at this data scale.** VAL has 26 tackles and 67
  shots; TRAIN is ~15× larger, so a few hundred each. Per-class results for `tackle`,
  `shot`, `cross` and `throw-in` must be reported with counts, never as a bare macro
  average.
- **~82.5% stage-1 recall ceiling** (§1.2c).

---

## Part 2 — Component design

### 2.1 FOOTPASS ingest (`matchlab_train/datasets/footpass.py`)

Reads `data/footpass/tactical/{train,val}_tactical_data.h5`. Column spec confirmed
twice over: by `tactical_data_format.txt` (which ships with the **HF dataset**, not
the code repo — fetched 2026-07-27), and independently by the reference code, which
unpacks the same layout in four places (`TAAD_Dataset.py:178`, `DST_Dataset.py:68`,
`metric_utils.py:158`, `run_TAAD_on_matches.py:88`):

```
0 frame  1 player_id  2 left_to_right  3 shirt_number  4 role_id
5 x  6 y  7 speed_x  8 speed_y  9 roi_x  10 roi_y  11 roi_width  12 roi_height  13 class
```

`class`: `0 background, 1 drive, 2 pass, 3 cross, 4 throw-in, 5 shot, 6 header, 7 tackle, 8 block`.
`roi_*` are **full-HD pixel coords** and are NaN when the player is off-screen (59% of
rows). CHALLENGE has 13 columns — no `class`.

Produces our schemas: `Tracklet` (from `roi_*`), `TeamAssignment` (from
`left_to_right`), pitch positions (from `x,y`), plus a `FootpassRoster` mapping
`(frame, slot) → shirt_number` for export, and `EventGroundTruth`.

### 2.2 The metric (`matchlab_core/pcbas/eval.py`)

**What the reference actually computes:** greedy matching at `delta=12` frames and
`conf_thresh=0.15`, with the TP key `(team, shirt_number, class)`; then **per-class
precision and recall, and a single micro-F1**. It computes no per-class F1 and no
macro-F1 — the string "macro" appears nowhere in it (`metric_utils.py:604-617,696`).

**Macro-F1 is our addition, and must be labelled as ours.** We report both, because
micro-F1 on this class distribution is dominated by `drive` and `pass` (5,529 of 6,070
VAL events) and can look healthy while every rare class is at zero.

Measured by running the reference's own `evaluation.py` on its shipped VAL predictions:

| arm | micro-F1 | macro-F1 (8 classes) |
|---|---:|---:|
| TAAD | 41.0 | 24.45 |
| TAAD + DST | **71.86** | **49.26** |

These are the reproduction targets — reproducible from artifacts in the repo, unlike
the paper's headline figure (see §3 gate).

Our version additionally supports keying identity on **roster slot** instead of
`shirt_number`, for ADR 007. Per §1.2a that changes the scoring key, so a roster-slot
number and a jersey number are **not** the same measurement and must never be placed
in one column without a note.

This is a **different metric from the existing `action_spotting_eval.average_map`** —
that one is class+time avg-mAP for SoccerNet-ball, with no identity term at all. Both
stay; conflating them is how a number gets quoted for the wrong task.

### 2.3 Role assignment (`matchlab_core/roles/`) — the critical path

Input: per-entity pitch trajectories + team. Output: assignment of each entity to one
of 13 tactical roles per team, i.e. a permutation onto 26 slots.

Baseline: mean pitch position per entity over a window, Hungarian assignment against
per-role position priors estimated from FOOTPASS TRAIN. The structural point of
SPO-95 holds — roles are a joint assignment, not independent per-player labels — but
the earlier draft's "each role used once per team" is **wrong**: measured, only **11
of 13 roles appear per side per half** (roles 4 and 8 unused in both games checked),
and 22 of 26 slots are occupied. Forcing 13 assignments would mis-assign whenever the
formation's unused roles differ from the priors'. So: **rectangular Hungarian with
dummy rows plus a formation-conditioned role subset** — and choosing that subset is
itself a modelling problem this design has not solved.

**Two gates, in isolation, before anything consumes it:**

1. **Role accuracy** against FOOTPASS's supplied `role_id` on VAL, reported per role.
2. **Slot stability** (§1.2d) — slot-switch rate per entity per 750-frame window and
   window-level slot purity, measured on our pipeline against supplied roles.

Gate 2 is the one that is easy to skip and expensive to skip, because an assignment
can be accurate per frame and useless per window.

Prerequisite gate: **side/half determination must be 100% per half on VAL.** It is a
binary per team per half; anything less than perfect makes every downstream number
uninterpretable rather than degraded.

### 2.4 Action head (`matchlab_core/spotting/action_head.py`)

X3D-S + FPN + `roi_align` + temporal Conv1d, per §1.1. Trains on `(B,3,50,352,640)`
clips with M=4 sampled tracklets; runs inference over all 26 slots with 50%-overlap
window averaging. Emits `(9, 26, T)` fp16 — the D3 contract.

### 2.5 Sequence denoiser (`matchlab_core/spotting/denoiser.py`)

Encoder-decoder Transformer per §1.1, framespan 750. Two variants behind one
interface: `flat` (reference DST, 364-dim concatenated slot vector) and `attn`
(PAVE-style per-player attention over role slots). The Notion page flags the flat
encoding as exactly what the 2026 winner replaced, so both must be measurable.

Pitch-symmetry augmentation (X, Y, XY flips with role-slot remap tables) is part of
the reference recipe and is carried over.

---

## Part 3 — Phasing

Every gate below states a **number**, because a gate whose bar is set after the result
is not a gate. Where the earlier draft said "within a stated band" without stating one,
or "measured" instead of a threshold, that is corrected.

| Phase | Deliverable | Needs video? | Gate (falsifiable) |
|---|---|---|---|
| **0a** | Metric vs the reference's shipped VAL predictions | no | **exact** reproduction of micro-F1 41.0 (TAAD) and 71.86 (TAAD+DST) to 2 d.p. |
| **0b** | Ingest + GT export | no | **6,070** VAL / **91,327** TRAIN events; per-class counts; 82.5% of VAL events carry a box |
| **0c** | ROI geometry check | VAL only | sampled ROIs land on players — **done 2026-07-27**, see plan Task 5 |
| **1** | Reproduce TAAD→DST on supplied roles | yes (24 GB) | VAL **micro-F1 ≥ 65** and **macro-F1 ≥ 42** (vs reference 71.86 / 49.26 — a ~10% relative band, stated now) |
| **2** | Export-time roster remap + PAVE attention | yes | each change ablated separately; PAVE arm ≥ flat arm on macro-F1 |
| **3a** | Side + half determination | yes | **100%** per team per half on VAL |
| **3b** | Role assignment from our pipeline | yes | role accuracy **≥ 70%** per side; **kill threshold < 50%** |
| **3c** | Slot stability | yes | median slot-switches per entity per 750-frame window **≤ 1**; window slot purity **≥ 0.8** |
| **3d** | End-to-end decay | yes | reported, no bar — this one is genuinely exploratory |
| **4** | Pipeline integration + Lab | no | smoke test only; **explicitly not a quality gate** |

The Phase 1 band deserves a note on comparability: **46.41 and PAVE's 58.94 are
challenge-split leaderboard figures**, and D6 commits us to VAL because CHALLENGE
labels are withheld. Comparing a VAL number to a challenge number can pass or fail for
split reasons alone. The bars above are therefore set against the **reference's own
VAL scores, reproduced locally from its shipped predictions** — the only comparand
that isolates our implementation from split effects.

Phases 0–1 are planned task-by-task in the accompanying plan. **Phases 2–4 are
specified here but deliberately not decomposed into steps** — each is a separate
subsystem whose design depends on Phase 1's measured outcome, and writing bite-sized
steps for them now would be inventing detail rather than recording it.

## Part 4 — Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Slot instability inside the DST window** | The likeliest killer, and distinct from role *accuracy*. Tracklet gaps reach 482 frames = 64% of a window | Gate 3c measures it directly, before the end-to-end run |
| **Side/half determination is wrong** | Shifts all 26 slots by 13 at once; per-team role accuracy still reads 100% while every attribution is wrong | Gate 3a at 100%, no tolerance |
| **Role assignment is poor on our inputs** | Blocks the entire chain (§1.2b) | Gate 3b with a stated kill threshold; the possession track remains the fallback |
| **Absolute pitch accuracy has never been measured** | Role priors are absolute-position priors, and Gate 2 only tracks the error's time-derivative | Must be measured before Phase 3; currently an unquantified dependency |
| **Class imbalance ~1,600 : 1** | Per-player-frame positive rate is 0.06%; a model predicting only background scores 99.94% | The reference's own recipe is the mitigation — balanced resampling at 500/class/epoch, resampled every epoch. Any deviation must be recorded |
| **We reimplement rather than vendor (D2), then miss the Phase 1 band** | The miss is unattributable between our reimplementation, our ingest and the setup | Run the reference code **once, unmodified, as a control** before adopting D2 for the production path |
| **Synthetic-noise DST flatters itself** | Would produce a great number that means nothing | D7: not reportable, full stop |
| **Macro-F1 hides rare-class collapse** | tackle n=26 on VAL (TRAIN ~390); shot n=67 (TRAIN ~1,000) | Always report per-class with counts. Only `tackle` is genuinely near-unlearnable — the earlier draft's "a few hundred each" was wrong for `shot` |
| **`SpottedEvent` has no identity field** | It is the frozen external contract shared with the T-DEED CLI; attribution would be silently dropped | Version the contract deliberately, or carry identity on `Event.attrs` (scalars only) |
| **No abstention path anywhere** | ADR 003 requires quality-gated evidence with abstention valid. Role is always assigned, every slot always emits logits, τ=0.15 always fires | Unresolved. At minimum a "role unknown" slot and a low-quality-tracklet gate — **this is a design gap, not a mitigated risk** |
| **16 GB VRAM on an RTX 4060 Ti** | batch 6 at T=50 is what the reference uses; whether that was chosen for 16 GB is unknown | Gradient accumulation (8 steps) is already in the recipe |
| **`pytorchvideo` is effectively unmaintained** | D1 puts it in the `cv` group | Pin it; the X3D-S weights are the only thing we need from it |
| **Scope creep into a full retrain** | This is a 4-phase program | Phase gates are hard; each phase reports before the next starts |
