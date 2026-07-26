# FOOTPASS / PCBAS — acquisition and B3-B4 retarget

**Status:** OPTIONAL, not blocking. B3-B4 progresses without it — the SNMOT
action-label tier gives a working event benchmark with no agreement required
(see the report linked at the bottom). What the NDA buys is specifically the
ability to **score player attribution**, which no other benchmark measures, plus
comparability with published baselines. Everything downstream of acquisition is
specified here so a build starts the moment it lands.

## Why this dataset

SoccerNet 2026 replaced Team Ball Action Spotting **and** Game State
Reconstruction. The five-task slate is now: Ball Action Anticipation,
**Player-Centric Ball-Action Spotting (PCBAS)**, Novel View Synthesis, Spiideo
SynLoc, VQA. Plain Ball Action Spotting is off the slate entirely.

That matters because the SPO-83 gate's criterion 1, `docs/prds/action-spotting-possession-transition.md`
and `configs/pipeline.possession-heuristic-eval.yaml` are all anchored to pass
avg-mAP@1 against SoccerNet Ball Action Spotting — a de-slated task.

PCBAS scores **action + responsible player**. Every prior benchmark scored class
and time only, which is the sole reason the PRD deferred attribution:

> `SpottedEvent` carries no `player_id`; avg-mAP scores class+time only. Player
> attribution is … a **B4** measurement concern, not scored here.

The possession-transition track has emitted `player_id` on every derived event
since SPO-78. It has been producing, unscored, exactly what PCBAS measures.
**B3 and B4 should be treated as one bucket from here.**

## What blocks acquisition

| Asset | Where | Gate |
|---|---|---|
| Annotations + videos | Hugging Face `SoccerNet/SN-PCBAS-2026` | Gated — "You have to accept the conditions to access its files and content" |
| Videos | SoccerNet website | Separate NDA form |

The HF dataset is **235 GB**, modality video, no dataset card published. Local
free space at the time of writing: 246 GB — enough, but with no margin. Prefer
pulling annotations first and videos only if the visual stage is actually built.

**Neither gate can be cleared by an agent.** Accepting an NDA is a legal act
requiring the account holder, and the HF download needs credentials.

### Steps for a human

1. Create / sign in to a Hugging Face account.
2. Open `https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026` and accept the
   conditions (shares contact info with SoccerNet).
3. Complete the SoccerNet NDA form for video access if videos are needed.
4. `hf auth login` (or set `HF_TOKEN`), then pull **annotations only** first —
   they are what the DST stage needs and are a small fraction of the 235 GB.
5. Place under `data/footpass/` and tell the agent; ingest is specified below.

## Dataset facts (from the paper and baseline repo)

- 54 complete men's matches, 2023/24 (Ligue 1, Bundesliga, Serie A, La Liga, UCL)
- 102,992 events — 48 train matches (91,327), 3 valid (6,070), 3 test (5,595)
- Full HD, 25 fps
- Events are tuples `(frame, team, jersey, class)`; frame 0-based, team 0=left /
  1=right, jersey = shirt number
- 8 action classes: Drive, Pass, Cross, Shot, Header, Throw-in, Tackle, Block
- Each player carries one of **13 tactical roles**, from formations, trajectories
  and expert annotation
- Baselines: TAAD (visual), TAAD+GNN, TAAD+DST
- Best 2026 entry: PAVE 58.94 macro-F1 vs 46.41 baseline

## Three caveats that shape what we can claim

1. **Leaderboard numbers are not comparable to ours.** FOOTPASS *supplies*
   tracking, jersey and role as inputs. MatchDay produces all three, so any
   number we get measures a strictly longer pipeline than the baselines do.
2. **The metric requires jersey-number match**, which collides with ADR 001
   (identity without OCR). **Settled by [ADR 007](../decisions/007-roster-slot-identity-for-attribution-benchmarks.md):**
   roster-slot identity substitutes for the DST mechanism (it needs a stable
   anchor token, not a number's semantics), and the benchmark is scored under a
   per-match optimal assignment following ADR 004's precedent. Any resulting
   figure is an **upper bound**, never leaderboard-comparable.
3. **Evaluation sits at one fixed high-recall operating point** (τ=0.15), the
   opposite end of the curve from MatchDay's abstention design. Per the B2 lesson
   — compare along the curve, not at a point — a single τ=0.15 number says little
   about the regime the product depends on. The internal precision-vs-abstention
   curve stays the frontier measurement.

## The architecture to build (recommended shape)

The FOOTPASS ablation is the strongest external evidence the project has: tactical
reasoning lifted precision from **~25% to ~68%** on *unchanged* visual predictions
(Drive F1 34% → 68%). The accuracy lives in stage 2, not stage 1.

```
stage 1  per-player tubes  ->  noisy per-player action logits      (TAAD; visual)
stage 2  whole-sequence denoising with tactical priors             (DST)
         role + position + velocity + team  ->  clean event list
```

**This supersedes the PRD's Phase 2 `possession-peral` spec in part.** Peral's
block 2 (Conv-TasNet + TDNN over a Gaussian-smoothed likelihood window) is a
*smoother*; DST is a tactical-prior sequence model. Same slot, materially
different capability. The PRD's hyperparameters remain valid for stage 1 tubes.

### What maps onto what we already have

| DST input | MatchDay source | Status |
|---|---|---|
| per-player action logits | `possession-heuristic-image` possessor timeline | built, weak |
| team | `teams.json` / oracle team stage | built |
| position | image-space only; pitch coords need calibration | **gap** |
| velocity | derivable from tracklets | trivial |
| tactical role (13) | nothing equivalent | **gap** |

The position gap is the B0 geometry question again — the fourth independent time
this track has hit it. The calibration stages already exist in this branch
(`stages/calibrate/{yolo_pitch_local,roboflow_keypoints,static}.py` +
`matchlab_core/pitch.py`, feeding `fuse/minimap`); what is missing is
`data/weights/football-pitch-detection.pt`, not code and not a merge. SNMOT
carries no pitch keypoints either, so pitch coordinates for the DST stage need
either those weights on footage that has them, or a calibrated tier.

## What was built instead, while blocked

Acquisition is gated, so the measurable work went to the one event ground truth
that is *not* gated — see
[`../reports/2026-07-27-b3-snmot-action-localization.md`](../reports/2026-07-27-b3-snmot-action-localization.md).
SNMOT clips carry `actionClass` + `actionPosition` in `gameinfo.ini`, making the
already-downloaded tracking tier a sparse action-spotting benchmark:

```bash
uv run matchlab-train spot-localization --signal ball-trajectory --out loc.json
uv run matchlab-train spot-localization --signal possession       --out loc-poss.json
```

Ball-contact classes: ball-trajectory median **2.0 frames (0.08 s)**, 87% within
0.2 s. That is the first real event-GT number this track has ever had.
