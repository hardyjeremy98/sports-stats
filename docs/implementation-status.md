# Implementation Status

**Status:** Canonical factual inventory  
**Last verified:** 2026-07-15  
**Purpose:** Distinguish implemented behavior from prototypes, stubs, research candidates, and plans.

Update this document when a capability is added, removed, materially changed, or measured. Product
intent belongs in [`player-identity-vision.md`](player-identity-vision.md); detailed historical
research belongs in [`../technology/`](../technology/).

## Status vocabulary

- **Implemented:** Runs through the normal pipeline and produces its intended artifact.
- **Prototype:** Runs, but has known evidence, licensing, evaluation, or robustness limitations.
- **Stub:** Registered interface or placeholder that intentionally does not implement the feature.
- **Planned:** Agreed direction with no runnable implementation.
- **Research candidate:** Documented option that has not been selected or implemented.

## Pipeline and identity capabilities

| Capability | Status | Current implementation | Primary location |
|---|---|---|---|
| Player and ball detection | Implemented | Roboflow inference, local YOLO, synthetic detector | `pitchlab_core/stages/detect/` |
| Short-term tracking | Implemented | BoT-SORT and dependency-free IoU tracker | `pitchlab_core/stages/track/` |
| Learned query-propagation tracking | Stub | `learned-motr` raises `NotImplementedError` | `stages/track/learned_stub.py` |
| Team classification | Implemented | Lab-space kit colour and SigLIP/KMeans variants | `pitchlab_core/stages/team/` |
| Camera calibration | Implemented | Static, Roboflow keypoint, and local YOLO variants | `pitchlab_core/stages/calibrate/` |
| Cross-tracklet association | Prototype | Greedy union-find using team/time/speed constraints and mean torso colour | `stages/associate/global_embed.py` |
| Association null baseline | Implemented | One player entity per tracklet | `stages/associate/identity_fallback.py` |
| Body re-ID association | Planned | Registry seam exists; no learned body embedding is wired in | — |
| Face identity | Prototype | InsightFace anchors from largest boxes, weighted embedding, greedy clustering | `stages/identity/face.py` |
| Optional face-crop upscaling | Prototype | RealESRGAN path in face resolver | `stages/identity/face.py` |
| Jersey OCR resolver | Research candidate | Schema supports `jersey`; no registered resolver exists | — |
| Structured visual attributes | Research candidate | No extractor, schema, or artifact exists | — |
| Gait identity | Research candidate | No temporal identity model or artifact exists | — |
| Quality-guided multimodal fusion | Planned | No fusion implementation; would require a composite or revised inference flow | — |
| Match-level constrained optimizer | Planned | Only local pair filtering and greedy merging exist | — |
| Roster enrollment and assignment | Planned | No roster model or assignment workflow exists | — |
| Identity-specific human QA | Planned | Current QA corrects event attribution, not identity entities | — |
| Minimap fusion | Implemented | Homography projection using associated entity IDs | `stages/fuse/minimap.py` |
| Event attribution | Prototype | Possession heuristic and contested-event QA | `stages/events/possession.py` |
| Learned action spotting | Stub | Registered no-op implementation | `stages/events/spotting_stub.py` |

Paths in this table are relative to `packages/pitchlab_core/src/pitchlab_core/` unless otherwise
stated.

## Evaluation

### Implemented

- SoccerNet Tracking ground-truth ingestion with boxes, track IDs, role, team, and optional jersey.
- MOT evaluation at two levels:
  - **Tracklet:** raw tracker IDs.
  - **Entity:** post-association `player_id` groupings.
- IDF1, ID precision/recall, MOTA, switches, fragmentations, misses, false positives, precision, and
  recall.
- Association IDF1 gain and ID-switch delta.
- Per-instance ID-switch records used by the Lab failure browser.
- Automatic worker evaluation when a registered video has ground truth.

Primary locations:

- `packages/pitchlab_core/src/pitchlab_core/gt.py`
- `packages/pitchlab_core/src/pitchlab_core/evaluation.py`
- `packages/pitchlab_server/src/pitchlab_server/worker.py`

### Not implemented

- Semantic roster-identity scoring.
- Identity-label comparison against GT jersey or roster records.
- Team and role accuracy metrics.
- Cluster purity, completeness, roster coverage, or abstention curves.
- Anchor coverage and per-modality quality diagnostics.
- HOTA, AssA, DetA, or GS-HOTA.
- Event-attribution ground-truth evaluation.
- Batch aggregation of GT metrics across an experiment set.

Changing `identity.impl` from `none` to `face` does not currently change `eval.json`. The evaluator
uses entity `player_id`, not `PlayerIdentity.label`.

## Experiment tooling

### Implemented

- YAML-selected stage implementations and parameters.
- Registry endpoints and Lab controls for launching stage variants.
- Run manifests containing resolved pipeline configuration and stage timing.
- Same-video run selection from the run viewer.
- A run-diff API and UI for config, headline metric, timeline, and stat differences.
- CLI `eval-pipelines` experiment that runs two configurations over multiple clips.
- SoccerNet ingestion and QA-label export commands.

### Limitations

- `eval-pipelines` aggregates artifact counts, not ground-truth identity metrics.
- There is no parameter-sweep or repeated-seed experiment.
- Pipeline stages do not share persisted identity embeddings.
- Model and weight provenance are not consistently represented as first-class run metadata.
- The CLI pipeline path does not automatically score against ground truth.

## Lab UI

### Implemented

- Source-video playback with client-side prediction overlays.
- Tracklet, team, player identity, calibration, minimap, event, and GT layers.
- Tracklet and player lists with selection and seeking.
- Face evidence thumbnails with frame, score, and upscaled metadata in the tooltip.
- Tracklet-versus-entity evaluation metrics.
- Clickable ID-switch instances that seek to the failure time.
- Run diff for configuration and headline metrics.
- Event-attribution QA with accept, correct, and reject actions.

### Limitations

- Evidence thumbnails do not open a detailed anchor inspector or seek directly to their frame.
- Entity-level switch inspection does not fully highlight the associated entity path.
- Diff has no synchronized side-by-side video or fixed/new switch comparison.
- QA cannot merge/split entities, label same/different tracklet pairs, or assign roster identities.
- There is no embedding, affinity, or constraint-decision inspector.
- Label export is CLI-only and contains event-attribution labels rather than re-ID pairs.

## Known findings

Measured local findings recorded by the repository guidance:

- Kit-colour association is ineffective for player-level identity.
- Remaining ID switches are substantially a tracker-level problem that simple post-association
  cannot repair.

Do not generalize these findings beyond the evaluated data. Link future claims to an experiment
report, run set, dataset split, and code/model revision.

## Immediate next milestones

1. Add semantic roster or cluster identity evaluation.
2. Make batch experiments aggregate GT metrics.
3. Add a learned body re-ID associator baseline.
4. Persist reusable anchor quality and identity evidence.
5. Add identity-failure and anchor inspection to the Lab.
6. Add identity QA that exports same/different and roster-assignment labels.

## Maintenance checklist

When implementation changes:

1. Update the relevant table row and status.
2. Add or remove the implementation path.
3. Update evaluation and UI capability lists where affected.
4. Link measured claims to their experiment report.
5. Update the verification date only after inspecting the implementation.
6. Do not mark a researched model as implemented merely because it appears in `technology/`.
