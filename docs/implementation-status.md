# Implementation Status

**Status:** Canonical factual inventory  
**Last verified:** 2026-07-17  
**Purpose:** Distinguish implemented behavior from prototypes, stubs, research candidates, and plans.

Update this document when a capability is added, removed, materially changed, or measured. Product
intent belongs in [`player-identity-vision.md`](player-identity-vision.md); detailed historical
research belongs in [`../technology/`](../technology/).

A tracklet-quality measurement and hardening program (tracker-level ID switches, GT-contamination
metrics, oracle-detection ceiling experiments) is underway; plan and phase gates are in
[`docs/prds/tracklet-modernization.md`](prds/tracklet-modernization.md) and tracked as the SPO
project in Linear.

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
| Oracle (ground-truth) detection | Implemented | Emits a video's GT boxes as detections instead of running a real detector, to isolate tracker/association behavior from detection quality (the "tracker ceiling" experiment); GT resolved from an explicit `gt_path` param or the sibling `<video>.gt.json` convention, loud error if neither exists; optional seed-deterministic dropout/jitter knobs (off by default); metadata-only, no frame decode | `pitchlab_core/stages/detect/oracle.py` (`impl: oracle`), `configs/pipeline.oracle-eval.yaml` |
| Short-term tracking | Implemented | BoT-SORT (via `roboflow/trackers`, pinned `==2.4.0`) and dependency-free IoU tracker. BoT-SORT construction fails loudly (`RuntimeError` naming the class, kwargs, and installed version) on constructor-signature drift instead of silently falling back to a zero-argument constructor; all 13 `BoTSORTTracker` constructor kwargs are exposed as `Params` (shipped configs state them explicitly); person/goalkeeper/referee class is carried through tracking via a `source_idx` entry in `sv.Detections.data` (reconstructed by nearest box centre previously; now index-based, with a fail-loud guard if the tracker drops/truncates that payload) | `pitchlab_core/stages/track/botsort.py` |
| Learned query-propagation tracking | Stub | `learned-motr` raises `NotImplementedError` | `stages/track/learned_stub.py` |
| Team classification | Implemented | Lab-space kit colour and SigLIP/KMeans variants | `pitchlab_core/stages/team/` |
| Camera calibration | Implemented | Static, Roboflow keypoint, and local YOLO variants | `pitchlab_core/stages/calibrate/` |
| Cross-tracklet association | Prototype | Greedy union-find using team/time/speed constraints and mean torso colour; records per-pair decisions (affinity, rejection reason) to `association.json` | `stages/associate/global_embed.py` |
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
| Identity-specific human QA | Implemented | Pair same/different/unsure verdicts (seeded from association near-misses and eval ID switches), entity merge/split flags, and roster labels, stored as annotations that never mutate run artifacts; exportable as re-ID training pairs via `pitchlab-train export-reid` | `web/src/components/IdentityQATab.tsx` + `pitchlab_server/api/identity_qa.py` |
| Minimap fusion | Implemented | Homography projection using associated entity IDs | `stages/fuse/minimap.py` |
| Event attribution | Prototype | Possession heuristic and contested-event QA | `stages/events/possession.py` |
| Learned action spotting | Stub | Registered no-op implementation | `stages/events/spotting_stub.py` |

Paths in this table are relative to `packages/pitchlab_core/src/pitchlab_core/` unless otherwise
stated.

`TrackletFrame` (`schemas/tracks.py`) carries a `source` provenance field —
`"observed"`/`"predicted"`/`"interpolated"`, defaulting to `"observed"` so artifacts written before
this field existed still parse — currently always `"observed"` from the shipped tracker
implementations; no stage yet writes `predicted`/`interpolated` frames.

## Provenance and reproducibility

- **Run provenance recorder** (SPO-10 part 1): every `manifest.json` written by
  `PipelineRunner` carries a `provenance` block that is append-only after run completion
  (only `evaluation_set_hash`/`evaluation_set_source` are filled in later, in place, by the
  GT auto-scoring hook — everything else is fixed when the run finishes) — git revision (`-dirty`
  suffixed on an unclean tree, `"unknown"` if git is unavailable), installed versions of
  `DEFAULT_PACKAGE_NAMES`, one `StageProvenance` (resolved params + a `ModelProvenance`
  list) per stage that actually ran, and `evaluation_set_hash`/`evaluation_set_source`.
  Every declared field is always present — unknown values are the literal string
  `"unknown"` (or `null` where the schema allows it), never an absent key. `Stage.
  provenance()` is a hook (default `[]`), overridden by `yolo-local` detect,
  `yolo-pitch-local` calibrate, `roboflow` detect, `roboflow-keypoints` calibrate, the
  `global-reid` associator's OSNet embedder, `siglip` team classification, and `face`
  identity — each reports architecture, local weights path + streaming SHA-256 (when
  applicable), lineage, and per-axis (`code`/`weights`/`training_data`) license status.
  `evaluation_set_hash` is `"unknown"` on every manifest as written by the pipeline runner
  itself (GT linkage is server-side); the server's GT auto-scoring hook
  (`pitchlab_server/evaluation.py::evaluate_run_against_gt`, called by the worker on run
  completion and by `POST /api/runs/{id}/evaluate`) writes the canonical-JSON hash of the
  scored GT file back into the manifest in place before scoring. `hash_evaluation_set` /
  `hash_dataset_manifest` hash `json.dumps(json.loads(text), sort_keys=True,
  separators=(",", ":"))`, so formatting differences never change the hash but any semantic
  change always does; `check_evaluation_set` is a refusal primitive (raises, naming both
  hashes and a context string, on mismatch) for a future benchmark runner (SPO-17, not
  implemented) to call before aggregating two runs together — landing the primitive does
  not itself gate any aggregation path yet. Consumer-facing schema doc: `docs/provenance.md`;
  `web/src/lib/types.ts` mirrors the schema by hand. No benchmark or comparison numbers
  depend on this yet — Phase 0 instrumentation.
  `packages/pitchlab_core/src/pitchlab_core/provenance.py`,
  `packages/pitchlab_core/src/pitchlab_core/runner.py`.
- **Hosted-detection response cache** (SPO-10 part 2): `RoboflowDetector` gains `cache_dir`
  (default `data/cache/hosted-detections`) and `cache_mode` (`off` / `readwrite` / `replay`)
  params; both shipped roboflow configs (`configs/pipeline.v1.yaml`,
  `configs/pipeline.v1-iou-baseline.yaml`) default to `readwrite`. Cache keys are
  `sha256(model_id, confidence, sha256-of-raw-pixel-bytes, shape, dtype)` — frame index is
  deliberately excluded, so identical pixel content recurring across frames/strides/runs
  hits the same entry; entries store the post-conversion `xyxy`/`scores`/`class_id` arrays
  (`hosted-detections/v1` JSON), not the raw hosted-API response object. `readwrite` fills
  the cache as it runs; `replay` is cache-hits-only, needs no `ROBOFLOW_API_KEY` and never
  constructs the hosted model in `prepare()`, and raises `RuntimeError` naming the key,
  frame index, and cache dir on a miss instead of silently falling back to the network.
  Both the player-model path and the tiled ball-detection path go through the cache.
  `HostedDetectionCache.content_hash()` (an order-independent sha256 over the directory's
  sorted (key, file-sha256) pairs) is recorded into `ModelProvenance.detections_cache_hash`
  (`null` when caching is off or the stage doesn't cache), refreshed by the runner both
  after `prepare()` and after the stage finishes executing, so a cold cache warmed during a
  `readwrite` run reflects its actual output rather than the pre-run empty-cache hash.
  `packages/pitchlab_core/src/pitchlab_core/stages/detect/hosted_cache.py`.
- **External tracklet exchange** (SPO-18): a frozen-detections export and a MOT-tracklet
  importer let an external MOT research tracker consume this repo's detections and have its
  output scored by the existing evaluator, without becoming a registered pipeline stage.
  Pure conversion (no CLI logic, no DB, no network) in
  `packages/pitchlab_core/src/pitchlab_core/exchange.py`; CLI wrappers `pitchlab-train
  export-detections` / `pitchlab-train import-tracklets`
  (`packages/pitchlab_train/src/pitchlab_train/cli.py`).
  - `export_frozen_detections(run_dir, out_dir)` reads `<run_dir>/detections.jsonl` +
    `manifest.json` and writes a standard MOT `det.txt` (1-based frames, `id=-1`, xywh boxes
    fixed-formatted `%.2f`/conf `%.6f` so repeated exports of the same run are byte-identical)
    plus a `detections_provenance.json` sidecar (`schema: "frozen-detections/v1"`,
    `sort_keys=True`): `det_txt_sha256`, video meta (`frame_count`/`sample_stride`/`fps`), the
    class filter applied, and the source run's detect-stage provenance carried over verbatim
    from the manifest (the literal string `"unknown"` if the manifest has none). Ball
    detections are excluded by default (`DEFAULT_INCLUDE_CLASSES = player, goalkeeper,
    referee`); `--include-ball` adds them back.
  - `import_mot_tracklets(mot_path, sidecar_path, out_run_dir, ...)` parses
    `frame,track_id,x,y,w,h,conf[,...]` MOT rows (1-based frame, conf defaulting to `1.0` if
    the column is absent; all imported tracklets are labeled `DetectionClass.PLAYER` since MOT
    carries no per-row class) into `tracklets.json` in the native `Tracklet` schema — the raw
    tracklet layer only, deliberately no `players.json` — plus a `manifest.json` whose
    `track` stage `StageProvenance.impl` is `"external:<system>"` (params = the full sidecar
    dump, one `ModelProvenance` built from the sidecar's system/variant/commit/weights/
    license), and a verbatim copy of the sidecar as `external_provenance.json`. A required
    `ExternalProvenance` sidecar (`system`, `license: LicenseAxes`, and `reference_only: bool`
    are mandatory, no defaults) must accompany the import; missing/malformed/incomplete
    sidecars, malformed or duplicate MOT rows, and a non-empty `out_run_dir` (never silently
    overwritten) are all refused loudly naming the offending path/field/row. When a
    `frozen_detections_dir` is supplied and the sidecar states `frozen_detections_sha256`, the
    import cross-checks it against that export's `det_txt_sha256` via the existing
    `check_evaluation_set` refusal primitive. Since no `players.json` is written, the
    semantic-identity evaluation layer is correctly skipped (stays `null`) when an imported
    run is scored — only the raw-tracklet MOT metrics apply.
  - `reference_only` is recorded on the sidecar (and copied into `external_provenance.json`)
    but nothing currently reads it to exclude a run from anything — enforcement is the
    benchmark runner (SPO-17, not implemented; same caveat as `check_evaluation_set` above).
  - Tested (`packages/pitchlab_core/tests/test_exchange.py`): export row/ball-filter/sidecar/
    determinism/missing-input cases; import fixture parsing, all refusal paths above
    (parametrized over each required sidecar field), and a hash-mismatch case; an
    export-then-import round trip asserting boxes and track ids survive exactly at `det.txt`'s
    2-decimal-place precision; and an end-to-end case (`test_evaluate_run_scores_imported_
    tracklets`) confirming `evaluation.evaluate_run` scores an imported run's raw tracklets
    against GT and skips the identity layer. No benchmark numbers — Phase 0 instrumentation,
    same as the other provenance work above.

## Evaluation

### Implemented

- SoccerNet Tracking ground-truth ingestion with boxes, track IDs, role, team, and optional jersey.
- SportsMOT ground-truth ingestion (SPO-11, `gt.py::load_sportsmot_sequence`): standard
  MOT17-style `gt.txt` + `seqinfo.ini`; players only (no ball/referee distinction, no team
  labels); rows with `conf == 0` (MOT's ignore-region convention, not a real detection) are
  skipped.
- SoccerTrack ground-truth ingestion (SPO-11, `gt.py::load_soccertrack_sequence`): parses
  the 3-header-row (TeamID/PlayerID/attribute) bounding-box CSV directly, since SoccerTrack
  carries no `seqinfo`; caller supplies fps/width/height (read from the ingested video via
  `probe()`). Deterministic, collision-free `track_id = team_id * 1000 + player_id`
  (`player_id` bound-checked `< 1000`; ball fixed at `9999`); NaN-safe cell parsing.
  **Caveat, unverified against real data:** SoccerTrack CSV frame numbers are assumed
  already 0-based (unlike MOT's 1-based `gt.txt` convention) — this has not been checked
  against an actual released SoccerTrack file, only against fixture data written to match
  the assumption.
- MOT evaluation at two levels:
  - **Tracklet:** raw tracker IDs.
  - **Entity:** post-association `player_id` groupings.
- IDF1, ID precision/recall, MOTA, switches, fragmentations, misses, false positives, precision, and
  recall.
- Association IDF1 gain and ID-switch delta.
- Per-instance ID-switch records used by the Lab failure browser.
- Automatic worker evaluation when a registered video has ground truth.
- Run-diff fixed/introduced/persisted ID-switch comparison between two runs on the same video
  (`diff_switch_instances`, greedy nearest-time matching within a group of `(level, gt_track_id)`).
- A third, semantic identity evaluation layer (ADR 004): does `PlayerIdentity.label` correspond
  to the right person, judged against GT tracks. Computes label coverage and abstention rate
  over matched entities, plus cluster purity and completeness — each labeled entity's full
  GT-overlap mass is assigned to its single argmax-overlap GT track (not smeared across every
  track it incidentally touched) before purity/completeness are aggregated. `null` when the
  identity stage didn't run; non-null with coverage 0 and purity `null` means it ran but
  abstained everywhere. Folds `identity_coverage` and `cluster_purity` into `runs.metrics`.
  Coverage's denominator is all matched real entities, including referees — a run that labels
  every player but abstains on the referee reports coverage < 1.0, not 1.0.
- Run-grouping benchmark: `GET /api/benchmark` aggregates completed, GT-scored runs by
  `(config_name, normalized-config hash)` into a config × GT-video matrix of per-cell
  mean/range across the eight benchmark metric keys (idf1/mota/idsw at tracklet and entity
  level, association gain, identity coverage, cluster purity).
- A fourth, orthogonal layer (`eval.json`'s `purity` block, tracklet-modernization SPO-6): direct
  per-tracklet GT-contamination measurement, at both tracklet and post-association entity level —
  where `merge_quality`'s majority-vote already discards exactly this signal. Per tracklet: GT
  composition, purity (majority-GT-id fraction of matched frames), and mixed-identity duration in
  stride-aware seconds; aggregated (pre- and post- min-track-length filter) into mean purity,
  fraction impure, total mixed seconds, tracklets-per-GT-player, and track-length distributions.
  `min_track_length` is discovered in order: the manifest's resolved track-stage config, then the
  registered track impl's own `Params.min_length` pydantic default, then `null` with a `note`
  explaining filtering happens upstream (in the track stage) and can't be recovered here.
  `headline_metrics` gains `tracklet_purity` and `mixed_track_seconds`.
- A fifth, independent backend (`eval.json`'s `hota` block, tracklet-modernization SPO-7): HOTA,
  DetA, AssA, and LocA at both tracklet and entity level, via a vendored slice of TrackEval
  (`pitchlab_core/hota.py` + `pitchlab_core/_vendor/trackeval/`, upstream
  `JonathonLuiten/TrackEval` @ commit `12c8791`, MIT license, only the HOTA metric-math path kept,
  numpy-2 patched (`np.float` removed alias)). Scores the same per-frame GT/prediction structures
  the motmetrics IDF1/MOTA accumulators use; the two backends are never reconciled against each
  other. `headline_metrics` gains `hota_tracklet` and `hota_entity`.
- A sixth, LEVEL-INDEPENDENT layer (`eval.json`'s `detection` block, SPO-9): scores the
  detector itself rather than the tracker, computed once per run (not per tracklet/entity
  level) via a standalone pure-numpy module (`pitchlab_core/detection_eval.py`, no
  motmetrics/scipy/trackeval), following the HOTA-adapter structural precedent — its own
  module, lazily imported, folded into `eval.json` under one top-level key. Deterministic
  confidence-descending greedy matching per frame (VOC-style single-pass assignment, not a
  global Hungarian solve) at a configurable IoU threshold; documented tie-breaks (equal-IoU
  GT candidates resolve to the lower `gt_track_id`; equal-confidence detections resolve by
  stable input order). Reports: operating-point precision/recall and average precision (AP,
  VOC2010+ all-point precision-envelope interpolation) over the emitted confidence range;
  a per-player-height-bin breakdown at fixed pixel edges (TP/FN bucketed by GT box height,
  FP by the detection's own height, since an FP has no GT to inherit a height from);
  consecutive miss-burst-length distributions per GT track (own-frame-presence-relative, not
  raw video-frame-relative), with a `burst_seconds_p95` derived from `stride`/`fps`;
  duplicate-detection rate (a second detection overlapping a GT box another detection in the
  same frame already claimed); and GT-residual temporal box jitter — center and height terms
  computed from `det_center - gt_center` between temporally adjacent matched frames of the
  same GT track, isolating detector box instability from real player motion. Every declared
  output key is always present; mathematically undefined values (e.g. precision with zero
  detections) are `None`, never NaN, so the result stays JSON-clean.
  - **Honest caveats:** jitter is measured only through GT association — it is a residual
    signal (detector noise around the GT box), not an absolute detector-jitter measurement,
    and both the miss-burst chains and the jitter adjacency chains are relative to each GT
    track's own evaluated-frame presence list, so frames where the GT track has no box at all
    are silently absent from that track's chain rather than breaking or extending it — this
    compresses (never inflates) the frame span the burst/jitter statistics actually cover.
  - Wired into `evaluate_run` via `_load_detections`, which reads `<run_dir>/detections.jsonl`
    (person classes only — player/goalkeeper/referee, mirroring `_SCORED_ROLES`; ball is
    excluded symmetrically), restricted to the same `eval_frames` the tracklet/entity/HOTA/
    purity layers already score against. `result["detection"]` is `null` when
    `detections.jsonl` doesn't exist at all (imported external-tracker runs —
    `exchange.py::import_mot_tracklets` writes no detections file), never a crash or a
    fabricated score; a malformed row raises `ValueError` naming the file and line number.
    `headline_metrics` gains `detection_ap` / `detection_recall` /
    `detection_miss_burst_p95` only when the block is present; `GET /api/benchmark`'s
    `BENCHMARK_METRIC_KEYS` gains `detection_ap` / `detection_recall`; `web/src/lib/types.ts`
    gains `DetectionEval` (+ `DetectionHeightBin`/`DetectionMissBursts`/
    `DetectionBurstSummary`) and `EvalResult.detection`.
  - Tested: analytic, hand-computed-arithmetic unit tests per metric
    (`test_detection_eval.py`, no motmetrics/scipy import needed at all — mirrors
    `test_hota.py`'s style) plus `evaluate_run` wiring/integration tests
    (`test_gt_eval.py`): the block present and correctly filtered/aggregated on a
    near-perfect synthetic run, `null` when `detections.jsonl` is absent, and a loud
    `ValueError` naming file+line on a malformed row.
  - **No benchmark numbers exist for this layer yet — Phase 0 instrumentation**, same as the
    other provenance/exchange work in this document: capability and test coverage only, no
    measured detection-quality figures on real footage.
- SoccerNet tuning/held-out split manifest (`configs/datasets/soccernet.json`,
  `configs/datasets/README.md`): 12 registered sequences — SNMOT-116..123 as `tuning` (already used
  in July-2026 re-ID/threshold work, permanently ineligible for held-out promotion), SNMOT-124..127
  as `held_out` (ingested 2026-07-16 specifically for this manifest, never referenced by a tuning
  config or experiment). Byte-stable (`sort_keys=True`, stable sequence ordering) so the file can
  be hashed as the identity of "which sequences an evaluation used."

Primary locations:

- `packages/pitchlab_core/src/pitchlab_core/gt.py`
- `packages/pitchlab_core/src/pitchlab_core/evaluation.py`
- `packages/pitchlab_core/src/pitchlab_core/hota.py`
- `packages/pitchlab_core/src/pitchlab_core/_vendor/trackeval/`
- `packages/pitchlab_core/src/pitchlab_core/detection_eval.py`
- `packages/pitchlab_server/src/pitchlab_server/worker.py`
- `packages/pitchlab_server/src/pitchlab_server/evaluation.py`
- `packages/pitchlab_server/src/pitchlab_server/api/benchmark.py`
- `configs/datasets/soccernet.json`

### Not implemented

- Identity-label comparison against GT jersey or roster records.
- Team and role accuracy metrics.
- Abstention/coverage curves (trend over time or across a run set — the underlying coverage and
  abstention-rate numbers exist per run, but nothing plots them across a batch).
- Anchor coverage and per-modality quality diagnostics.
- GS-HOTA (plain HOTA/DetA/AssA/LocA is implemented, see above).
- Event-attribution ground-truth evaluation.

Changing `identity.impl` from `none` to `face` now changes `eval.json`: the semantic identity
layer scores `PlayerIdentity.label` against GT-track argmax assignment (coverage, cluster
purity/completeness). The tracklet/entity MOT layers are unaffected — they still key off entity
`player_id` groupings, not the label.

## Experiment tooling

### Implemented

- YAML-selected stage implementations and parameters.
- Registry endpoints and Lab controls for launching stage variants.
- Run manifests containing resolved pipeline configuration and stage timing.
- Same-video run selection from the run viewer.
- A run-diff API and UI for config, headline metric, timeline, and stat differences.
- CLI `eval-pipelines` experiment that runs two configurations over multiple clips.
- SoccerNet ingestion and QA-label export commands.
- `pitchlab-train ingest-sportsmot` / `ingest-soccertrack` (SPO-11): mirror
  `ingest-soccernet`'s register-as-Lab-video pattern (stitch frames or copy the source
  video, write a `.gt.json`, register a `Video` row with `gt_path` set) and additionally
  write/merge a `configs/datasets/<tier>.json` split-manifest entry for each ingested
  sequence — the first programmatic writer of that file format (`soccernet.json` remains
  hand-maintained). Frame-stitching is shared via `pitchlab_train/datasets/stitch.py`
  (used by both `ingest-soccernet` and `ingest-sportsmot`; `ingest-soccertrack` needs no
  stitching since SoccerTrack ships pre-encoded video, discovered by same-directory,
  same-stem `*.mp4`/`*.csv` pairing). `packages/pitchlab_train/src/pitchlab_train/datasets/
  {sportsmot,soccertrack,stitch}.py`.
- `pitchlab_train.datasets.manifest.update_tier_manifest` (SPO-11): deterministic
  (`sort_keys=True`, `sequences` grouped `"tuning"` entries first then `"held_out"`,
  ascending by name within each group) merge-writer for `configs/datasets/<tier>.json`.
  Verifies every `video`/`gt` path exists before writing anything; records paths relative
  to the repo root, raising loudly (naming both the path and the root) rather than falling
  back to an absolute, machine-specific path if one isn't actually under it; refuses
  (`RuntimeError` naming the sequence) to flip a sequence already recorded `"tuning"` to
  `"held_out"` — promotion the other way (`"held_out"` to `"tuning"`) is allowed. Covered
  by an end-to-end scoreability test (`test_end_to_end_scoreability_of_ingested_sequence`)
  that ingests a fixture sequence, builds a run whose tracklets echo the ingested ground
  truth exactly, and asserts `pitchlab_core.evaluation.evaluate_run` recovers near-1.0
  IDF1/HOTA against it — confirms the ingest's ground truth is scoreable through the
  existing evaluator, not a benchmark measurement on real detector/tracker output.
  `packages/pitchlab_train/src/pitchlab_train/datasets/manifest.py`.
- `pitchlab-train export-reid`: exports identity-QA "same"/"different" pair verdicts (unsure
  pairs excluded) as re-ID training pairs with copied crop images, cross-run crop-name-collision
  safe.
- `GET /api/benchmark` (`pitchlab_server/api/benchmark.py`): read-only, no-schema-change
  aggregation of every completed, GT-scored run into a config × video mean/range matrix — the
  batch-GT-metrics aggregation that `eval-pipelines` doesn't do (see limitation below), surfaced
  in the Lab at `/lab/benchmark`.

### Limitations

- `eval-pipelines` aggregates artifact counts, not ground-truth identity metrics — `GET
  /api/benchmark` covers batch GT-metric aggregation for runs already in the run table, but
  there is still no CLI experiment that launches a sweep and scores it against GT directly.
- There is no parameter-sweep or repeated-seed experiment.
- Pipeline stages do not share persisted identity embeddings.
- Model and weight provenance are not consistently represented as first-class run metadata.
- The CLI pipeline path does not automatically score against ground truth.

## Lab UI

### Implemented

- Source-video playback with client-side prediction overlays.
- Tracklet, team, player identity, calibration, minimap, event, and GT layers.
- Tracklet and player lists with selection and seeking.
- Face evidence thumbnails with frame, score, and upscaled metadata in the tooltip, clickable to
  open an anchor-frame inspector.
- Anchor-frame inspector with full-frame context (crop box outlined on the source frame), a ~4×
  zoom panel, a raw-vs-upscaled crop pair (when raw persistence fired), quality metadata (face
  score, upscaled flag, crop box size, frame/clock time), a jump-to-tracklet action, and prev/next
  navigation across a player's evidence without closing. Degrades gracefully for older runs
  lacking crop geometry (`box`).
- Tracklet-versus-entity evaluation metrics.
- A filterable, sortable identity-failure browser: ID-switch instances filter by level and GT
  track, sort by time or GT track, and click-to-inspect highlights the tracklet, entity, and GT
  track together (dual highlighting) in the video overlay.
- Timeline markers for ID-switch instances, color-coded by level (tracklet vs entity).
- Re-evaluate a run against ground truth on demand from the run viewer (and from the eval-missing
  empty state, for runs that predate a video's ground truth).
- Run diff for configuration and headline metrics, with a same-video guard (enforced server-side;
  the dashboard also disables cross-video checkbox selection client-side).
- Synchronized diff playback: two VideoOverlays (A master, B follower) driven by shared play/pause/
  seek and a drift-correction loop, one shared layer-toggle row, and seekable timelines carrying
  both runs' own eval markers plus fixed/introduced switch markers.
- Fixed/introduced/persisted switch comparison in the diff view, browsable per category and
  click-to-seek (reuses the run viewer's switch-instance row component).
- Diff identity metrics: tracklet/entity IDF1, tracklet/entity IDSW, entity MOTA, and association
  IDF1 gain, shown side by side with directional deltas, once both runs have an eval artifact.
- Dashboard filters (video, config, status, ground-truth-only) and sortable metric/time columns.
- Event-attribution QA with accept, correct, and reject actions.
- Association decision artifact (`association.json`): per-pair affinity, distance, and
  rejection-reason recording alongside the accepted merge edges, from the associate stage
  (`stages/associate/global_embed.py`).
- Association inspector tab in the run viewer: impl/params header with merged/rejected pair
  counts, a rejection-reason constraint summary (gate vs colour-distance), an entity-to-tracklet
  mini-gantt, and a filterable/sortable candidate-pair list that seeks to and highlights both
  tracklets of a pair in the video overlay.
- Identity QA sub-tab (`IdentityQATab`) in the run viewer's QA tab: a same/different/unsure pair
  queue seeded from association near-misses (`color_too_far`, sorted closest-miss-first, plus
  `span_conflict`/`speed_implausible`) and eval entity-level ID switches (prev/new tracklets
  derived deterministically around the switch frame), deduped against already-labeled pairs by
  unordered tracklet-pair equality; crop-strip evidence per tracklet with seek/highlight
  fallbacks when no evidence crops exist; keyboard shortcuts (s/d/u) that act on the first
  visible candidate, guarded against firing while an input is focused; a manual pair-entry
  degradation path for runs without association/eval artifacts; a labels-collected counter; and
  a run-scoped recent-labels list with undo (delete).
- PlayersTab entity actions: multi-select checkboxes with a floating "flag merge" action row,
  per-entity inline tracklet multi-select "flag split", and a per-entity roster-label input —
  all annotations that do not mutate the run's entities, with UI copy saying so.
- Cross-run identity-label browser on the global QA page (`pages/LabQA.tsx`): a read-only table
  of every pair/merge/split/roster label across all runs, with a run-viewer link and delete.
- Identity block in the run viewer's Eval tab: coverage, abstention rate, labeled/matched entity
  counts, cluster count, and cluster purity/completeness (`—` when null), shown when the run's
  eval carries the semantic identity layer; a one-line "identity stage not run" note when it's
  `null`.
- Identity coverage and cluster purity rows (A/B/delta, higher-is-better) in the run-diff's
  identity metrics table, shown when either compared run has a non-null identity layer.
- Benchmark matrix view (`pages/LabBenchmark.tsx`, `/lab/benchmark`): rows are config groups
  (name, short config hash, run count), columns are GT videos plus a trailing per-group mean;
  a metric-picker (IDF1 entity/tracklet, IDSW entity, MOTA entity, association gain, identity
  coverage, cluster purity) color-scales every cell through the shared confidence ramp (inverted
  for IDSW, where lower is better), shows the mean with a `±(range/2)` sub-line when a cell has
  more than one run, and a missing metric renders as `—` with a hint to re-evaluate for identity
  metrics; clicking a cell expands an inline row linking to that cell's runs.

### Limitations

- Merge/split/roster flags are annotations only; nothing currently consumes them to actually
  re-associate entities or rewrite a roster short of the offline `export-reid` pair pipeline.

## Known findings

Measured local findings recorded by the repository guidance:

- Kit-colour association is ineffective for player-level identity.
- Remaining ID switches are substantially a tracker-level problem that simple post-association
  cannot repair.
- **Offline association currently adds GT contamination relative to raw tracklets, on one
  measured sequence.** Run `06a067a478f2` (video SNMOT-116, `configs/datasets/soccernet.json`
  `tuning`-role sequence, config `v1-local-eval`: local YOLO detection (`yolo-local`,
  `football-player-detection.pt`) + BoT-SORT tracking + `global-color` association): tracklet
  IDF1 0.4305, HOTA (tracklet) 0.3617, mean tracklet purity 0.9056 (`frac_impure` 0.1831,
  pre/post min-length-filter identical); entity-level (post-association) mean purity 0.6592
  (`frac_impure` 0.68). I.e. association merges raise tracklet-level IDF1 only marginally
  (entity IDF1 0.4234, a −0.0071 "gain") while roughly doubling the fraction of impure identity
  groups. Numbers reproduced by calling `evaluation.evaluate_run` directly against this run's
  stored artifacts and GT — the run's persisted `eval.json` predates the purity/HOTA evaluator
  and does not yet contain these blocks; re-run `POST /api/runs/{id}/evaluate` to refresh it.
  Code revision: `tracklet-modernization` branch @ `1f759cb` (2026-07-16).
- **Oracle-detection smoke test: near-zero tracker/association error when detection is perfect,
  but the headline IDF1/HOTA from this run are not a ceiling figure.** Run `oracle-smoke-spo8`
  (video SNMOT-116, first 100 of 750 frames only, config `pipeline.oracle-eval.yaml`: oracle
  detector (`impl: oracle`, GT boxes, no dropout/jitter) + dependency-free `iou` tracker at its
  defaults (`iou_threshold=0.2`, `max_age_frames=20`, `min_length=5`)): all 19 tracklets pure
  (mean purity 1.0, `frac_impure` 0.0) and zero ID switches within the processed window. Tracklet
  IDF1 0.30 / HOTA (tracklet) 0.19 from the same run are **coverage-diluted, not a ceiling
  measurement**: the GT/metric window spans the full 750-frame sequence while only the first 100
  frames were processed, so recall (and therefore IDF1/HOTA) is suppressed by frames the run
  never attempted, not by tracking error. The genuine oracle-ceiling number is the Phase 0 exit
  gate's full-sequence oracle run (Linear SPO-21), not this smoke test. No `eval.json` is
  persisted for this run; the IDF1/HOTA/purity figures above were computed by calling
  `evaluation.evaluate_run` directly against this run's stored artifacts and GT. Code revision:
  `tracklet-modernization` branch @ `1f759cb` (2026-07-16).

Do not generalize these findings beyond the evaluated data. Link future claims to an experiment
report, run set, dataset split, and code/model revision.

## Immediate next milestones

1. Add a learned body re-ID associator baseline.
2. Persist reusable anchor quality and identity evidence.
3. Add identity-label comparison against GT jersey/roster records, and abstention/coverage
   curves (currently only single-run snapshot numbers exist, not trends across frames or a
   run set).

## Maintenance checklist

When implementation changes:

1. Update the relevant table row and status.
2. Add or remove the implementation path.
3. Update evaluation and UI capability lists where affected.
4. Link measured claims to their experiment report.
5. Update the verification date only after inspecting the implementation.
6. Do not mark a researched model as implemented merely because it appears in `technology/`.
