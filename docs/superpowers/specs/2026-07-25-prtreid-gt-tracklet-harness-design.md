# GT-tracklet re-ID harness + soccer-trained embedder comparison — design

**Date:** 2026-07-25 · **Status:** design approved in-session (Jeremy), not yet implemented
**Relates to:** SPO-74 (soccer-finetuned embedder), SPO-73 (mutual-best merge rule, verdict
superseded by its own oracle-team amendment), SPO-75 (kit-colour team gate), PRD
[sports-stats#1](https://github.com/hardyjeremy98/sports-stats/issues/1)

## Problem

SPO-73 located the anchorless-merging bottleneck in the embedder rather than the decision
rule: tracker-frozen KPR embeddings fail do-no-harm under both an absolute threshold
(SPO-59) and mutual-best + margin (SPO-73), and on held-out the decision statistics invert —
the single most confident candidate pair was a wrong same-team merge. SPO-74 responds by
proposing to finetune an embedder. That skips a cheaper rung: **a soccer-trained part-based
re-ID model already exists and is obtainable**, and we have never measured whether swapping
it in clears the bar.

Two measurement defects block that comparison today.

1. **The substrate contaminates the measurement.** Every association experiment in the repo
   runs on TDLP-full tracklets, which carry known residual impurity (tracklet purity 0.9093
   held-out, cross-exit re-links). The merge layer is therefore measured on inputs that
   already contain silent swaps, and "the engine merged wrongly" cannot be separated from
   "the tracker handed it a contaminated tracklet". True pairs are recovered by a GT-argmax
   analyzer rather than known exactly.
2. **The substrate no longer exists on disk.** `data/experiments/` is absent; the frozen base
   run `benchmark-reid-b2-base-20260724-035932` that every replay arm overrides to is gone.
   Only `recon-SNMOT-124/125/126` (tracklets + features) survive; SNMOT-127 and all tuning
   sequences are lost. Rebuilding costs ~14 min/sequence of GPU.

## What we are building

A **GT-tracklet re-ID harness**: ground-truth tracks fragmented at their natural gaps,
producing a merge task with purity 1.0 by construction and exactly known correct pairs, on
which candidate embedders are compared by retrieval metrics. No tracker run is required.

Then a **five-way embedder comparison** on that harness — the incumbent KPR against
soccer-trained PRTreID and three in-repo controls — with a pre-registered decision and
interpretation rule.

### Scope decisions (settled in the design session)

- **Identity embedding only.** PRTreID also has team-affiliation and role heads; neither is
  adopted here. The harness pins teams to oracle anyway, so a team head would buy nothing for
  this measurement and would add a second changed variable. Team/role adoption is follow-on
  work against SPO-75.
- **Natural gaps only** for fragmentation. Synthetic cuts would give a controllable
  difficulty curve, but the player stays visible through an artificial gap, so the crops
  either side stay clean and the degraded-re-entry failure class — 5 of 21 true held-out
  pairs in SPO-73 — disappears from the test.
- **Retrieval first, engine run for the winner only.** Retrieval metrics are seconds per
  embedder once features exist and attribute cleanly to the embedder; the engine run
  confirms that better ranking converts into correct merges under the real rule and gates.
- **All five arms**, not just KPR vs PRTreID. The controls are what make a PRTreID win
  interpretable — without them, a win cannot distinguish "in-domain soccer training helped"
  from "any different backbone would have helped", which is precisely the question that
  decides whether SPO-74's finetune is worth funding.

### Non-goals

- **Not a do-no-harm gate, and it cannot become one.** GT tracklets are pure by construction
  and easier than real ones; absolute numbers will not transfer to the real pipeline. The
  held-out do-no-harm gate remains a separate later step, and that one does need the
  TDLP-full substrate rebuilt, because do-no-harm is a claim about the real pipeline.
- **No tracker changes.** TDLP-full stays frozen.
- **No anchor work.** This experiment is anchorless throughout; the anchor layer and the
  naming decoder are untouched.
- **No split/hygiene stage.** Still deferred by PRD decision.
- **The Sinkhorn removal (SPO-72) is not in scope.**

## Components

### 1. `matchlab_core/gt_fragments.py` — fragmentation (pure)

`GroundTruth → (list[Tracklet], fragment_id → gt_track_id map)`. A GT track is split wherever
consecutive annotated frames differ by more than `gap_frames`. No video, no I/O, no model —
analytically testable against hand-computed expectations on toy inputs, the house pattern.

The fragment→GT map is exact and is what makes the harness better than the existing setup:
no GT-argmax inference, so a wrong merge is wrong by construction rather than by attribution.

### 2. `stages/track/oracle.py` — the `oracle` TRACK stage

The GT twin of `tdlp-full`, and the missing member of the oracle family (`detect` has one,
`team` has one, `track` does not). Emits `tracklets.json` + `frame_features.npz`.

Params:

| param | default | meaning |
|---|---|---|
| `gt_path` | `None` | explicit GT, else sibling `<video>.gt.json` (same resolution order as oracle detect) |
| `gap_frames` | `2` | split threshold, in frames |
| `min_fragment_frames` | `1` | drop fragments shorter than this |
| `include_classes` | player, goalkeeper, referee | matches `bridge.DEFAULT_INCLUDE_CLASSES` |
| `features.backend` | `external` | `external` or `in-repo` |
| `features.model` | `kpr` | `kpr`/`prtreid` (external) or `osnet`/`solider`/`dinov2` (in-repo) |
| `features.device` | `cuda:0` | passed to whichever backend runs |

A missing GT is a loud error, never silent empty output — same convention as the oracle
detector: an oracle run without GT is meaningless.

### 3. Feature backends

**External (`kpr`, `prtreid`).** Reuses the existing `tdlp_full/bridge.py` — `stage_sequence`
already turns run frames + detections into the MOT layout, and `gen_features.py` already
takes `--img-dir` + `--det-file` and embeds whatever boxes it is given. With `detect: oracle`,
those boxes are GT boxes, so KPR features over GT crops need no new external machinery.
`gen_features.py` gains `--reid-model {kpr,prtreid}` and a no-pose fast path (pose is a TDLP
feature; retrieval does not need it, and it is a large share of the runtime). The no-pose path
writes zero keypoints at confidence 0 to keep the `frame_features` schema intact; verified safe
because the re-ID engine reads embeddings and visibility only — no module under
`matchlab_core/reid/` references keypoints. Any future consumer that needs them must run the
pose path.

**In-repo (`osnet`, `solider`, `dinov2`).** `get_embedder()` over crops cut with the existing
`stages/team/_crops.py`, reshaped `(N, D) → (N, 1, D)` with visibility ones. **No
`BodyEmbedder` interface change is needed** — `frame_features` is already `(N, P, D)` and P=1
is a valid case. The part-based models arrive through the external bridge, which already
emits 6-part output.

Features join to fragments by `(frame_idx, detection index)`; `det.txt` ordering per frame is
preserved in the emitted pickles, and the bridge already owns the local↔source frame-index
mapping.

### 4. `matchlab_train` experiment `reid-retrieval`

Reads a run's `frame_features.npz` + `tracklets.json` + GT and reports the metrics below to
`result.json`. No engine run, no pipeline execution.

### 5. Configs

- `configs/pipeline.gt-tracklets-reid.yaml` — detect `oracle`, track `oracle`, team `oracle`,
  associate `per-tracklet` (no-op) by default, everything downstream disabled.
- `configs/train/reid-retrieval-tuning.yaml` — the five-arm comparison over tuning sequences.

## Data flow

```
<video>.gt.json ──► gt_fragments (split at natural gaps ≥ gap_frames)
                      ├──► tracklets.json           (purity 1.0 by construction)
                      └──► fragment → GT-track map  (exact; no GT-argmax inference)

GT boxes ──► det.txt ──► [external venv] gen_features --reid-model {kpr,prtreid}
        └──► crops   ──► [in-repo] get_embedder(osnet|solider|dinov2)
                              │
                              └──► frame_features.npz (N, P, D) + visibility
                                   joined by (frame_idx, det index)

frame_features + fragment map ──► reid-retrieval ──► rank-1 / mAP / margins per arm
                                                          │  winner only
                                                          ▼
                                          reid-engine run ──► association.json + eval.json
```

**Cost.** The GT substrate saves the *tracker* run, not the embedding run: per the
`tdlp-full` stage comments, feature-gen dominates the ~14 min/sequence. Budget roughly
1.5 h per external embedder across the 8 tuning sequences; the in-repo controls are much
cheaper (no pose, no external venv).

## Protocol (pre-registered)

Registered on the Linear issue before any arm executes. No amendment after seeing results
except by a recorded amendment, per repo convention.

**Splits.** SoccerNet tier manifest `configs/datasets/soccernet.json`, hash-guarded. All
development and embedder selection on **tuning SNMOT-116–123**. Held-out SNMOT-124–127 is
untouched by this experiment. The surviving `recon-SNMOT-124/125/126` run dirs are moved out
of `data/runs/` before any work starts, so held-out artifacts cannot be picked up
accidentally.

**Frozen substrate.** Oracle detections, oracle teams, GT tracklets with **`gap_frames: 2`**
(the registered value, matching the engine's `overlap_tolerance_frames`) and
`min_fragment_frames: 1`. `gap_frames` sets task difficulty; tuning it after seeing results
would be tuning the benchmark, so it is registered once and recorded in provenance.
Fragmentation is computed once per sequence and reused byte-identically across all arms —
*verified*, the way SPO-59 verified its replay, not assumed.

**Arms.** `kpr` (incumbent baseline), `prtreid` (hypothesis), `osnet`, `solider`, `dinov2`
(controls). Identical fragments; features recomputed per arm; everything else pinned.

**Primary metric — rank-1 on re-entry pairs.** For each fragment with at least one same-player
partner, does its top-1 candidate belong to the same GT track? A fragment with several
partners scores correct if top-1 is *any* of them; the denominator is the number of fragments
that have at least one gate-passing same-track partner. The candidate pool is
restricted to **gate-passing fragments** — temporal non-overlap, oracle team consistency,
motion feasibility — the same pool the merge rule would see. Ranking against all fragments
would flatter every arm by counting opponents as distractors, which is the easy case.

**Secondary metrics.** mAP; same-player vs different-player affinity separation; and the
top-1 margin distribution split by whether top-1 is correct — the exact statistic SPO-73
found inverting on held-out. Breakdowns by gap length, fragment length, and crop size, where
the two known failure classes (same-kit lookalikes; degraded re-entry crops) should separate
if the kit-dominance mechanism hypothesis is right.

**Decision rule.** Highest tuning rank-1 wins. A gap below **0.02** counts as a tie and breaks
toward the incumbent (KPR).

**Interpretation rule** (registered in advance — this is what the controls are for):

- PRTreID beats KPR *and* all controls → in-domain soccer training is the cause; SPO-74's
  finetune is justified as an incremental step from a soccer-trained base.
- A control matches PRTreID's gain over KPR → the cause is backbone/model, not soccer
  training; SPO-74 is re-scoped before any GPU is spent on finetuning.
- Nothing beats KPR → the embedder hypothesis is wrong on this substrate and the bottleneck
  lies elsewhere (gates, representation, or the merge rule).

**Engine confirmation, winner only.** `reid-engine` on GT tracklets, anchorless, at the
operating point already pre-registered on SPO-73's oracle-team amendment (`mutual-best`,
`merge_min_margin: 0.07`, `min_similarity: 0.95`). Reports correct/wrong edges and entity
purity. Tuning only.

**Provenance per row.** Git revision, weights hashes, dataset manifest hash, fragmentation
params. Aggregation refuses on any inconsistency, as elsewhere in the benchmark harness.

## Error handling

- **Missing GT** — loud error naming the resolution order attempted. Never silent empty output.
- **Missing or unloadable PRTreID weights** — loud error naming the acquisition step, in the
  convention used for missing extras (`stages/identity/face.py`). No silent fallback to KPR:
  a benchmark arm that quietly runs a different model than its label is worse than a crash.
- **Fragment/feature join mismatch** — if any fragment frame lacks a feature row, or counts
  disagree, fail the run rather than dropping rows. Silent row loss would bias retrieval.
- **Degenerate fragments** — GT tracks that never gap produce a single fragment with no
  partner. These are excluded from the rank-1 denominator (no correct answer exists) and the
  count is reported, so the metric's base is visible rather than implied.
- **External subprocess failure** — existing `bridge.run_external` behaviour: fail loudly with
  the command and a stderr tail.

## Testing

Following the house pattern — assert external behaviour against hand-computed expectations on
small synthetic inputs, never internal call structure.

- `gt_fragments`: toy GroundTruth with known gap structure → expected fragment boundaries and
  fragment→track map; `gap_frames` boundary cases (gap exactly at threshold, gap one over);
  `min_fragment_frames` filtering; a track with no gaps yielding one fragment.
- Feature join: synthetic det/feature pairs → correct `(frame_idx, det index)` association;
  a deliberately mismatched pair asserting the loud failure.
- In-repo backend reshape: `(N, D) → (N, 1, D)` with visibility ones, round-tripped through
  `FrameFeatures` write/read.
- Retrieval analyzer: a hand-built fragment set with analytically known rank-1, mAP, and
  margin values, including the gate-restricted pool (asserting that gate-vetoed distractors
  are excluded from ranking) and the degenerate-fragment exclusion.
- Oracle track stage: one end-to-end smoke test on synthetic video — runs, emits both
  artifacts, fragment count matches the pure function's output.

No UI tests (no frontend test infrastructure); the frontend build's type-check remains the
regression net if artifact types change, which this design does not require.

## Risks and open items

1. **PRTreID weight acquisition is unverified.** The `prtreid` repo has no model zoo yet
   ("will be updated soon"); the practical route is TrackLab's automatic download when running
   the sn-gamestate baseline, or the Zenodo tracker states. First implementation step is to
   confirm acquisition; if it fails, the fallback is the SoccerNet ReID benchmark baseline
   weights, and if that also fails the experiment reduces to KPR + in-repo controls and the
   PRTreID question stays open. This is a real dependency on an external artifact, not a
   formality.
2. **`prtreid_api` availability in the CAMELTrack venv is unverified.** TrackLab ships a
   PRTreID wrapper and the venv already has tracklab + KPR; if the wrapper is absent, adding
   it to that isolated env is the fix, consistent with the repo's env-isolation pattern.
3. **SoccerNet-ReID's positive pairs are not re-entry pairs.** Its identity labels do not hold
   across actions — a player has a different identity per action, and evaluation matches only
   within an action, across camera viewpoints at one instant. So a model trained on it learns
   "same player, same moment, different angle", not "same player, twenty seconds later, from a
   degraded crop". Prediction: PRTreID should help the same-kit-lookalike failure class more
   than the degraded-crop class. PRTreID was also evaluated on SoccerNet-Tracking, whose
   identities do persist within a sequence; whether that is in its *training* mix is unconfirmed
   and worth checking, as it would change this prediction.
4. **The kit-dominance mechanism is a hypothesis, not a finding.** It is consistent with the
   measured statistics (different-player median affinity 0.767, p90 0.912 — a high floor for
   "different people") but has not been verified on our data. The per-part breakdown in the
   secondary metrics is what would confirm or kill it.
5. **GT tracklets are easier than real ones.** Restated because it is the most likely way this
   work gets over-read: a win here is evidence for advancing to a real-substrate gate, not
   evidence that the pipeline improved.
6. **License provenance.** `prtreid` code is under the Hippocratic License
   HL3-LAW-MEDIA-MIL-SOC-SV. Recorded as provenance; gates nothing under the repo's research
   posture.

## Consequences for existing work

- The `oracle` TRACK stage gives every future association experiment a contamination-free
  substrate, removing the tracker-impurity confound permanently rather than only for this run.
- SPO-74 is re-sequenced: download-and-measure before finetune, with the finetune's scope
  determined by this experiment's interpretation rule.
- The TDLP-full substrate rebuild is deferred, not cancelled — it is still required for the
  held-out do-no-harm gate that follows a winner.
