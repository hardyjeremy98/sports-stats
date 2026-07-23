# Implementation Status

**Status:** Canonical factual inventory  
**Last verified:** 2026-07-19  
**Purpose:** Distinguish implemented behavior from prototypes, stubs, research candidates, and plans.

Update this document when a capability is added, removed, materially changed, or measured. Product
intent belongs in [`../../docs/player-identity-vision.md`](../../docs/player-identity-vision.md); detailed historical
research belongs in [`../../docs/technology/`](../../docs/technology/).

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

> **Default / current-best tracklet system (as of 2026-07-19): the hardened BoT-SORT baseline.**
> Config [`configs/pipeline.v1-hardened-eval.yaml`](../configs/pipeline.v1-hardened-eval.yaml)
> (frozen-detection variant `pipeline.v1-hardened-frozen-eval.yaml`), impl **`botsort`**
> (`stages/track/botsort.py`, via `roboflow/trackers==2.4.0`). Fully licensing-clean (a
> heuristic tracker — Kalman motion model + IoU gating + camera-motion compensation, **no
> learned weights, no NC/research data**) and runnable on any box. Tuned params (SPO-22):
> `track_activation_threshold 0.25`, `lost_track_buffer_s 1.0`, `minimum_consecutive_frames 3`,
> `min_length 5`, `enable_cmc true` (`sparseOptFlow`, `cmc_downscale 2`),
> `minimum_iou_threshold_first/second/unconfirmed 0.2/0.5/0.3`, `high_conf_det_threshold 0.4`,
> `instant_first_frame_activation true`, `state_estimator xcycwh`.
> Measured on held-out sequences over **frozen reference detections** (tracker isolated from
> detection): SportsMOT **IDsw 31 / HOTA 0.785 / purity 0.945**; SoccerNet **IDsw 144 / HOTA
> 0.519 / purity 0.926** — purity-equivalent to the non-shippable SOTA TDLP reference ceiling.
> **Selected at the SPO-34 Phase-3 gate**: no candidate (BoT-SORT+body-ReID/SPO-31, OC-SORT,
> TDLP) cleared the pre-registered promotion bar, so this baseline stands as the interim
> shippable tracker. `botsort-reid` (SPO-31) is a retained optional lever (marginally better
> IDsw, sub-bar on purity).
>
> **The shippable multi-cue TDLP program (SPO-36–44) was CLOSED/superseded (2026-07-20)** by a
> pivot to **research mode** (adopt SOTA directly, drop the shippable/licensing-clean goal).
> Outcome: the **SOTA TDLP-full** tracker (MixSort YOLOX + RTMPose + KPR + released TDLP head)
> runs in the isolated `external-trackers/` env and its results are imported + GT-scored in the
> Lab — **SportsMOT (in-domain) IDsw 6–9 / HOTA 0.85–0.92; SoccerNet (cross-domain, oracle
> dets) ≈ on par with BoT-SORT (HOTA ~0.75)**. Finding: TDLP-full's edge is domain-bound to
> SportsMOT, and detection is the dominant real-world bottleneck. It is a **research/local
> tool**, not a native in-repo stage, and not shippable (NC/research weights). The hardened
> BoT-SORT baseline above remains the default. Full close-out:
> [`docs/reports/2026-07-20-sota-tdlp-research-outcome.md`](reports/2026-07-20-sota-tdlp-research-outcome.md).

| Capability | Status | Current implementation | Primary location |
|---|---|---|---|
| Player and ball detection | Implemented | Roboflow inference, local YOLO, synthetic detector | `matchlab_core/stages/detect/` |
| Oracle (ground-truth) detection | Implemented | Emits a video's GT boxes as detections instead of running a real detector, to isolate tracker/association behavior from detection quality (the "tracker ceiling" experiment); GT resolved from an explicit `gt_path` param or the sibling `<video>.gt.json` convention, loud error if neither exists; optional seed-deterministic dropout/jitter knobs (off by default); metadata-only, no frame decode | `matchlab_core/stages/detect/oracle.py` (`impl: oracle`), `configs/pipeline.oracle-eval.yaml` |
| SportsMOT frozen comparator detection (SPO-25) | Prototype | `yolox-local` stage loads a vendored, inference-only MixSort YOLOX-X (`matchlab_core/vendor/mixsort_yolox/`, fetched from `github.com/MCG-NJU/MixSort` @ pinned commit `a078f5bf6ae9fbeecbc1384479d5f02ab8b9e7f6`, MIT repo / Apache-2.0 upstream YOLOX code) against the frozen checkpoint `data/weights/mixsort/yolox_x_sports_train.pth.tar` (sha256 `58547880fb73b9f9ac5674547781c6a87071906376286da301f9b0e19b50ed1c`). Same "fail loud on missing weights" / `provenance()` idiom as `yolo-local`. **Selection-only, non-shippable**: the checkpoint was fine-tuned on SportsMOT (CC BY-NC 4.0) — it exists to freeze a stronger detection floor for tracker selection (Phase 2/3), never to ship. Measured: mean detection_ap 0.9844 vs the hardened incumbent's 0.2641 over the same 9 same-protocol SportsMOT sequences; see [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md). | `matchlab_core/vendor/mixsort_yolox/`, `matchlab_core/stages/detect/yolox_local.py`, `configs/pipeline.yolox-sportsmot-eval.yaml` |
| Short-term tracking (**DEFAULT tracker**) | Implemented | **The current best / default tracklet system — hardened BoT-SORT (see the callout above `configs/pipeline.v1-hardened-eval.yaml`).** BoT-SORT (via `roboflow/trackers`, pinned `==2.4.0`) and dependency-free IoU tracker. BoT-SORT construction fails loudly (`RuntimeError` naming the class, kwargs, and installed version) on constructor-signature drift instead of silently falling back to a zero-argument constructor; all 13 `BoTSORTTracker` constructor kwargs are exposed as `Params` (shipped configs state them explicitly); person/goalkeeper/referee class is carried through tracking via a `source_idx` entry in `sv.Detections.data` (reconstructed by nearest box centre previously; now index-based, with a fail-loud guard if the tracker drops/truncates that payload) | `matchlab_core/stages/track/botsort.py` |
| Learned query-propagation tracking | Stub | `learned-motr` raises `NotImplementedError` | `stages/track/learned_stub.py` |
| Multi-cue learned tracking (in-repo `tdlp-shippable`) | Experimental — **CLOSED/superseded, not the default** | The shippable-equivalent program was retired 2026-07-20 (research-mode pivot; see the callout above). This in-repo stage (weaker DINOv2 appearance) does **not** beat the default BoT-SORT and is not maintained; the SOTA TDLP-full path (external env, imported to the Lab) replaced it for research. `tdlp-shippable` (`StageKind.TRACK`): the assembled licensing-clean multi-cue tracker — RF-DETR detections + RTMPose keypoints + DINOv2 global appearance → **vendored TDLP link-prediction head** (`matchlab_core/_vendor/tdlp/`, MIT @50344b9, arch only; local `global_appearance` encoder replaces the research-only KPR 6-part) → in-repo offline association loop (SciPy Hungarian, no `motrack`). **Runs end-to-end on arbitrary video** (verified: `configs/pipeline.tdlp-shippable-smoke.yaml`, 40 frames → 24 tracklets). Head behind a swappable interface: random-init (plumbing, logs loudly) or a checkpoint. **No shippable checkpoint yet** — a preliminary NC-eval-tier-tuning-trained head (`matchlab_train/tdlp_head_train.py`, corrupt-and-recover) yields the first Bar A number (non-shippable); the shippable retrain (SPO-40) is **blocked on permissive association-training data (SPO-39, HITL)**. See [`docs/reports/2026-07-19-spo42-assembled-shippable-tdlp.md`](reports/2026-07-19-spo42-assembled-shippable-tdlp.md). | `stages/track/tdlp/`, `_vendor/tdlp/`, `stages/associate/embedders/dinov2.py`, `pose/rtmpose.py`, `stages/detect/rfdetr.py` |
| Team classification | Implemented | Lab-space kit colour and SigLIP/KMeans variants | `matchlab_core/stages/team/` |
| Camera calibration | Implemented | Static, Roboflow keypoint, and local YOLO variants | `matchlab_core/stages/calibrate/` |
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
| Identity-specific human QA | Implemented | Pair same/different/unsure verdicts (seeded from association near-misses and eval ID switches), entity merge/split flags, and roster labels, stored as annotations that never mutate run artifacts; exportable as re-ID training pairs via `matchlab-train export-reid` | `web/src/components/IdentityQATab.tsx` + `matchlab_server/api/identity_qa.py` |
| Minimap fusion | Implemented | Homography projection using associated entity IDs | `stages/fuse/minimap.py` |
| Event attribution | Prototype | Possession heuristic and contested-event QA | `stages/events/possession.py` |
| Learned action spotting (SPO-45/46) | Prototype | `tdeed` `EventSpotter` runs an external action spotter via a subprocess bridge over a documented CLI contract (`docs/reference/spotting-exchange-contract.md`), writing a dedicated `spotting.json` artifact in the spotter's native ball-action taxonomy (no `EventType` mapping applied). The real model (GPL-3.0 T-DEED, SoccerNet-trained non-commercial weights) is isolated in a sibling `external-spotters/` env (`docs/reference/external-spotters-setup.md`) and is **reference/internal only, never shipped** — mirrors the `ultralytics`/`external-trackers/` posture. A permissive in-repo reference CLI (`matchlab_core/spotting/reference_cli.py`) stands in for dev/test with no GPU or real model. | `stages/events/tdeed.py`, `spotting/bridge.py` |

Paths in this table are relative to `packages/matchlab_core/src/matchlab_core/` unless otherwise
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
  (`matchlab_server/evaluation.py::evaluate_run_against_gt`, called by the worker on run
  completion and by `POST /api/runs/{id}/evaluate`) writes the canonical-JSON hash of the
  scored GT file back into the manifest in place before scoring. `hash_evaluation_set` /
  `hash_dataset_manifest` hash `json.dumps(json.loads(text), sort_keys=True,
  separators=(",", ":"))`, so formatting differences never change the hash but any semantic
  change always does; `check_evaluation_set` is a refusal primitive (raises, naming both
  hashes and a context string, on mismatch) now used by the `benchmark` experiment's
  evaluation-set-consistency gate (SPO-17 part 2, see Experiment tooling below) before
  aggregating rows together. Consumer-facing schema doc: `docs/provenance.md`;
  `web/src/lib/types.ts` mirrors the schema by hand. No benchmark or comparison numbers
  depend on this yet — Phase 0 instrumentation.
  `packages/matchlab_core/src/matchlab_core/provenance.py`,
  `packages/matchlab_core/src/matchlab_core/runner.py`.
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
  `packages/matchlab_core/src/matchlab_core/stages/detect/hosted_cache.py`. **Exercised at
  tier scale (SPO-26, Phase 2):** the SoccerNet tier's incumbent (`roboflow`,
  `football-players-detection-3zvbc/11`) was captured `readwrite` at confidence 0.1 over all
  12 tuning/held-out sequences (9000 cache entries, 36 MB), then replayed `cache_mode:
  replay` with `ROBOFLOW_API_KEY` unset — completed with zero network access and produced a
  byte-identical exported `det.txt` versus the original capture, confirming the cache is
  genuinely frozen-and-replayable, not merely off. Per-sequence `detections_cache_hash`
  values recorded mid-run are warm-up-time snapshots, not the tier's final cache identity —
  the completed cache directory's own `content_hash()` is the identity to cite. Report:
  [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md).
- **External tracklet exchange** (SPO-18): a frozen-detections export and a MOT-tracklet
  importer let an external MOT research tracker consume this repo's detections and have its
  output scored by the existing evaluator, without becoming a registered pipeline stage.
  Pure conversion (no CLI logic, no DB, no network) in
  `packages/matchlab_core/src/matchlab_core/exchange.py`; CLI wrappers `matchlab-train
  export-detections` / `matchlab-train import-tracklets`
  (`packages/matchlab_train/src/matchlab_train/cli.py`).
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
  - `reference_only` is recorded on the sidecar (and copied into `external_provenance.json`);
    enforcement is the benchmark runner's `ImportCandidate` validation (SPO-17 part 2, see
    Experiment tooling below): a `reference_only: true` import is refused outright if its
    `comparison_class` is `"matched_data"`.
  - Tested (`packages/matchlab_core/tests/test_exchange.py`): export row/ball-filter/sidecar/
    determinism/missing-input cases; import fixture parsing, all refusal paths above
    (parametrized over each required sidecar field), and a hash-mismatch case; an
    export-then-import round trip asserting boxes and track ids survive exactly at `det.txt`'s
    2-decimal-place precision; and an end-to-end case (`test_evaluate_run_scores_imported_
    tracklets`) confirming `evaluation.evaluate_run` scores an imported run's raw tracklets
    against GT and skips the identity layer. No benchmark numbers — Phase 0 instrumentation,
    same as the other provenance work above.
  - **Frozen-detections `INDEX.json` convention (Phase 2, SPO-25/26):** exporting every
    sequence of a tier produces `data/exchange/frozen-detections/<tier>/<seq>/{det.txt,
    detections_provenance.json}` (gitignored data, not a new code artifact); a hand-built
    per-tier `INDEX.json` (sorted keys) maps sequence name to `det_txt_sha256`, `n_rows`,
    `frame_count`, the run's `evaluation_set_hash`, and — for hosted tiers — a
    `detections_cache_hash`. This is a reporting convention over the existing exporter and
    manifest fields, not a new schema or CLI command; both tiers' index tables are
    reproduced in [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md)
    §2. Re-export determinism (same run dir → byte-identical `det.txt`) and one fp32/GPU
    repeat-inference determinism check (bitwise-identical) are recorded in that report §4.

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
  mean/range across the fifteen benchmark metric keys (idf1 at tracklet and entity level,
  mota at entity level, idsw at tracklet and entity level, association gain, merge precision,
  identity coverage, cluster purity, hota at tracklet and entity level, tracklet purity,
  mixed-track seconds, detection AP and recall).
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
  (`matchlab_core/hota.py` + `matchlab_core/_vendor/trackeval/`, upstream
  `JonathonLuiten/TrackEval` @ commit `12c8791`, MIT license, only the HOTA metric-math path kept,
  numpy-2 patched (`np.float` removed alias)). Scores the same per-frame GT/prediction structures
  the motmetrics IDF1/MOTA accumulators use; the two backends are never reconciled against each
  other. `headline_metrics` gains `hota_tracklet` and `hota_entity`.
- A sixth, LEVEL-INDEPENDENT layer (`eval.json`'s `detection` block, SPO-9): scores the
  detector itself rather than the tracker, computed once per run (not per tracklet/entity
  level) via a standalone pure-numpy module (`matchlab_core/detection_eval.py`, no
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
- A seventh annotation pass (`eval.json`'s per-instance `attribution` + top-level
  `attribution` context block, SPO-19, `matchlab_core/attribution.py`): every per-instance
  ID-switch record carries an evidence-based layer attribution — `detection`,
  `online_association`, `refinement` (reserved for the Phase 4 refined-tracklet layer,
  never emitted today), `offline_association` — or an explicit `ambiguous` tag, with the
  evidence basis recorded per instance (`oracle_input` / `oracle_comparison` /
  `tracklet_counterpart` / `entity_only` / `insufficient_evidence`). Rules are
  deterministic and conservative: a pristine oracle-detections run (detect impl `oracle`,
  zero dropout/jitter, read from the manifest's resolved config) attributes its tracklet
  switches to `online_association` by construction; an oracle-run comparison (greedy
  nearest-`t` one-to-one matching per GT track, the same matcher `diff_switch_instances`
  now shares from `matchlab_core.attribution.match_instances`) attributes disappears→
  `detection` / persists→`online_association`; entity-level switches inherit a matched
  tracklet counterpart's layer or are `offline_association` when association introduced
  them; everything else is `ambiguous` — no path silently defaults to a specific layer.
  Oracle enrichment is explicit, never inferred: the benchmark runner pairs candidates via
  `PipelineCandidate.oracle_candidate` (validated at expansion: pristine oracle detect,
  identical resolved track config and stride; rows record `attribution_oracle`
  enriched/unavailable), and the Lab re-scores via
  `POST /api/runs/{id}/evaluate?oracle_run_id=<scored oracle run of the same video>`.
  An oracle payload must self-describe (`attribution.oracle_input == true` in its own
  eval.json) or enrichment refuses loudly; sequence/stride/IoU mismatches also refuse.
  The Lab failure browser renders a per-switch layer pill (tooltip = evidence; eval.json
  files predating SPO-19 show `unattributed`) and a layer filter; `web/src/lib/types.ts`
  gains `AttributionLayer`/`EvalInstanceAttribution` and `EvalResult.attribution`.
  **Honest caveat:** oracle comparison is categorization support, not causal proof — a
  baseline switch matched within tolerance to an oracle-run switch is attributed
  `online_association` even though the two could in principle have different causes; the
  matched oracle instance is recorded in the evidence so the claim is inspectable.
  Tested on handcrafted payloads and run dirs with known causes plus benchmark/API
  integration (`test_attribution.py`, `test_benchmark_runner.py`, `test_api.py`).
- SoccerNet tuning/held-out split manifest (`configs/datasets/soccernet.json`,
  `configs/datasets/README.md`): 12 registered sequences — SNMOT-116..123 as `tuning` (already used
  in July-2026 re-ID/threshold work, permanently ineligible for held-out promotion), SNMOT-124..127
  as `held_out` (ingested 2026-07-16 specifically for this manifest, never referenced by a tuning
  config or experiment). Byte-stable (`sort_keys=True`, stable sequence ordering) so the file can
  be hashed as the identity of "which sequences an evaluation used."
- SportsMOT tuning/held-out split manifest (`configs/datasets/sportsmot.json`, SPO-16): 9
  sequences drawn from the SportsMOT `val` split (HuggingFace MCG-NJU/SportsMOT) — 6 `held_out`
  (2 each football/basketball/volleyball) + 3 `tuning` (1 per sport), all from distinct source
  videos so no clip correlation crosses the held-out/tuning boundary. Ingested 2026-07-17 via
  `matchlab-train ingest-sportsmot`; one held-out sequence scored end-to-end through the
  oracle-eval pipeline to confirm the GT is scorable. **License: CC BY-NC 4.0 — non-commercial /
  research only.** This tier is an evaluation benchmark only; SportsMOT must never train shipped
  models or be redistributed with the product (see CLAUDE.md → Licensing boundaries). Upstream
  distribution is also CodaLab-agreement-gated; the HF `val.tar` object served publicly, so the
  gate was not clicked through — **commercial use needs an explicit recorded licensing sign-off,
  which is still open** (raised at SPO-16, deferred to the owner of product licensing risk).
- Action-spotting scoring (SPO-47/49): a timestamped-event ground-truth representation
  (`EventGroundTruth`, `matchlab_core/event_gt.py`) distinct from the box/track
  `GroundTruth` used everywhere else, an `ingest-soccernet-ball` CLI that registers SoccerNet
  Ball Action Spotting matches as Lab videos with event GT, and a `soccernet-ball` eval tier
  manifest (`configs/datasets/soccernet-ball.json`). Scored by a dedicated avg-mAP@1 metric
  (tolerance-window matching, per-class AP, mean; `matchlab_core/action_spotting_eval.py`),
  wired into `eval.json` (its own action-spotting result shape, not the MOT suite),
  `runs.metrics` (`spotting_map_at_1`), and `GET /api/benchmark`'s metric keys — the server's
  GT auto-scoring hook (`matchlab_server/evaluation.py::evaluate_run_against_gt`) picks this
  path automatically whenever a run's video GT is event-shaped
  (`event_gt.is_event_ground_truth`), the same way it already auto-picks the MOT path for
  track GT. **No benchmark number has been measured yet.** Running the real T-DEED weights
  against SoccerNet Ball Action Spotting (the GPU pass that would produce a real `avg-mAP@1`)
  is a pending human-gated step (SPO-50) — nothing in this repo invents or estimates that
  number. Two open caveats, unverified against real downloaded data:
  - The `ingest-soccernet-ball` adapter assumes **one video per match**; the real dataset
    release is understood to package each match as **two half-video files** sharing one
    `Labels-ball.json` — unverified since no real copy has been downloaded to check against
    (see the adapter's module docstring and `configs/datasets/soccernet-ball.json`'s notes).
    The adapter will need extending before a real ingest if this holds.
  - The `soccernet-ball` tier's non-commercial/eval-only licensing note is an **inference**,
    not a confirmed reading of SoccerNet's ball-action-specific terms — CLAUDE.md's existing
    licensing-boundaries section classifies SportsMOT and SoccerNet-tracking, not ball-action
    data specifically. Treat it as provisional and get an explicit human licensing sign-off
    before any use beyond internal benchmarking (mirrors the still-open SportsMOT sign-off
    above).

Primary locations:

- `packages/matchlab_core/src/matchlab_core/gt.py`
- `packages/matchlab_core/src/matchlab_core/evaluation.py`
- `packages/matchlab_core/src/matchlab_core/attribution.py`
- `packages/matchlab_core/src/matchlab_core/hota.py`
- `packages/matchlab_core/src/matchlab_core/_vendor/trackeval/`
- `packages/matchlab_core/src/matchlab_core/detection_eval.py`
- `packages/matchlab_server/src/matchlab_server/worker.py`
- `packages/matchlab_server/src/matchlab_server/evaluation.py`
- `packages/matchlab_server/src/matchlab_server/api/benchmark.py`
- `configs/datasets/soccernet.json`
- `packages/matchlab_core/src/matchlab_core/event_gt.py`
- `packages/matchlab_core/src/matchlab_core/action_spotting_eval.py`
- `packages/matchlab_train/src/matchlab_train/datasets/soccernet_ball.py`
- `configs/datasets/soccernet-ball.json`

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
- `matchlab-train ingest-sportsmot` / `ingest-soccertrack` (SPO-11): mirror
  `ingest-soccernet`'s register-as-Lab-video pattern (stitch frames or copy the source
  video, write a `.gt.json`, register a `Video` row with `gt_path` set) and additionally
  write/merge a `configs/datasets/<tier>.json` split-manifest entry for each ingested
  sequence — the first programmatic writer of that file format (`soccernet.json` remains
  hand-maintained). Frame-stitching is shared via `matchlab_train/datasets/stitch.py`
  (used by both `ingest-soccernet` and `ingest-sportsmot`; `ingest-soccertrack` needs no
  stitching since SoccerTrack ships pre-encoded video, discovered by same-directory,
  same-stem `*.mp4`/`*.csv` pairing). `packages/matchlab_train/src/matchlab_train/datasets/
  {sportsmot,soccertrack,stitch}.py`.
- `matchlab_train.datasets.manifest.update_tier_manifest` (SPO-11): deterministic
  (`sort_keys=True`, `sequences` grouped `"tuning"` entries first then `"held_out"`,
  ascending by name within each group) merge-writer for `configs/datasets/<tier>.json`.
  Verifies every `video`/`gt` path exists before writing anything; records paths relative
  to the repo root, raising loudly (naming both the path and the root) rather than falling
  back to an absolute, machine-specific path if one isn't actually under it; refuses
  (`RuntimeError` naming the sequence) to flip a sequence already recorded `"tuning"` to
  `"held_out"` — promotion the other way (`"held_out"` to `"tuning"`) is allowed. Covered
  by an end-to-end scoreability test (`test_end_to_end_scoreability_of_ingested_sequence`)
  that ingests a fixture sequence, builds a run whose tracklets echo the ingested ground
  truth exactly, and asserts `matchlab_core.evaluation.evaluate_run` recovers near-1.0
  IDF1/HOTA against it — confirms the ingest's ground truth is scoreable through the
  existing evaluator, not a benchmark measurement on real detector/tracker output.
  `packages/matchlab_train/src/matchlab_train/datasets/manifest.py`.
- `matchlab-train export-reid`: exports identity-QA "same"/"different" pair verdicts (unsure
  pairs excluded) as re-ID training pairs with copied crop images, cross-run crop-name-collision
  safe.
- `GET /api/benchmark` (`matchlab_server/api/benchmark.py`): read-only, no-schema-change
  aggregation of every completed, GT-scored run into a config × video mean/range matrix — the
  batch-GT-metrics aggregation that `eval-pipelines` doesn't do (see limitation below), surfaced
  in the Lab at `/lab/benchmark`.
- `matchlab-train run` `benchmark` experiment (SPO-17, `matchlab_train/experiments/
  benchmark.py`): the PRD's decision-making backbone — a parallel system to `GET
  /api/benchmark` above, by design (no `matchlab_server`/DB imports, offline-first). Loads a
  `configs/datasets/<tier>.json` manifest, expands a candidate matrix (pipeline candidates:
  a config path + dotted-path overrides + parameter sweeps; import candidates: sequence name
  → an `exchange.import_mot_tracklets` run dir), runs/scores each candidate over each
  selected sequence, and emits one provenance-stamped row per (candidate, sequence). Rows
  then pass three pure gates before any aggregate is computed — missing provenance (empty
  `git_revision`/`stage_impls`), provenance inconsistency within one candidate's rows
  (differing `stage_impls` or model identity set across its sequences), and evaluation-set
  hash mismatch across candidates scoring the same sequence — each a loud `RuntimeError`
  naming the offending run_id(s)/candidate/hashes (`check_evaluation_set`-style); a candidate
  whose config or import provenance is invalid refuses at expansion, before any run/score
  happens. Surviving rows fold into `summary.tables = {"matched_data": {...}, "as_published":
  {...}}` (mean/median per headline metric, `n_sequences`, a `by_role` sub-breakdown when a
  candidate has both `tuning` and `held_out` rows; zero completed rows still produces an
  entry with `n_sequences: 0, metrics: None` — never omitted) — the two tables are never
  merged, and an `ImportCandidate` with `reference_only: true` is refused outright if it
  names `comparison_class: "matched_data"` (a reference-only external system can never enter
  a shipping comparison). An optional `Params.compare = {baseline: <matched_data candidate>}`
  plus `Params.tolerances` (pre-registered per gate issue, never a hardcoded default in code)
  produces `summary.comparison.verdicts`: per other matched_data candidate, per tolerance
  metric, `"improved"`/`"regressed"`/`"within_tolerance"` (a small `LOWER_IS_BETTER` set —
  `idsw_tracklet`, `idsw_entity`, `mixed_track_seconds`, `detection_miss_burst_p95` — inverts
  the delta sign) or `"unavailable"` when the metric never got a headline value. Covered by a
  PRD-mandated golden integration suite (`tests/test_benchmark_golden.py`): one end-to-end run
  (2 rendered clips, 2 pipeline candidates, 1 import candidate) asserting exact row shape,
  table separation/disjointness, import-row confinement to `as_published`, and deterministic
  tolerance verdicts derived from that run's own measured deltas — plus refusal-path unit
  tests (`tests/test_benchmark_task9.py`) on the pure gate/aggregation/comparison functions.
  No benchmark numbers yet — the pre-registered gate thresholds this feeds are a later phase.

### Limitations

- `eval-pipelines` aggregates artifact counts, not ground-truth identity metrics — `GET
  /api/benchmark` covers batch GT-metric aggregation for runs already in the run table.
  Launching a sweep and scoring it against GT directly is now covered by the `benchmark`
  experiment (SPO-17, documented above), not by `eval-pipelines`.
- There is no repeated-seed experiment — the `benchmark` experiment's candidate/sweep
  expansion is deterministic (one row per candidate × sequence), so run-to-run variance from
  re-running the same candidate isn't measured.
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
- A filterable, sortable identity-failure browser: ID-switch instances filter by level, GT
  track, and attributed layer (SPO-19), sort by time or GT track, and click-to-inspect
  highlights the tracklet, entity, and GT track together (dual highlighting) in the video
  overlay. Every switch row shows its layer-attribution pill (evidence in the tooltip;
  `unattributed` for eval.json files predating SPO-19).
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
- HOTA-family and track-composition row groups in the run viewer's Eval tab's Metric × level
  table, alongside the existing IDF1/MOTA rows and sharing their tracklet/entity columns:
  "HOTA family" (HOTA, DetA, AssA — LocA deliberately omitted) reads `eval.json`'s `hota`
  block; "Track composition" (mean purity, impure fraction, mixed-identity duration,
  tracklets-per-GT-player mean/max) reads the `purity` block's `post_filter` aggregate at each
  level. A missing value distinguishes its two causes by tooltip: `undefined` = the run predates
  the metric (re-evaluate to backfill), `null` = the evaluator abstained because nothing matched
  GT. Below the table, a caption states the `min_track_length` the purity filter used, or that
  it could not be discovered and no filtering was applied.
- Identity coverage and cluster purity rows (A/B/delta, higher-is-better) in the run-diff's
  identity metrics table, shown when either compared run has a non-null identity layer.
- Benchmark matrix view (`pages/LabBenchmark.tsx`, `/lab/benchmark`): rows are config groups
  (name, short config hash, run count), columns are GT videos plus a trailing per-group mean;
  a metric-picker (IDF1 entity/tracklet, HOTA entity/tracklet, IDSW entity, MOTA entity,
  association gain, merge precision, identity coverage, tracklet purity, mixed-identity time,
  cluster purity — the two purity metrics are labeled by layer so the matrix never shows a bare,
  ambiguous "Purity") color-scales every cell through the shared confidence ramp (inverted for
  IDSW and mixed-identity time, where lower is better), shows the mean with a `±(range/2)`
  sub-line when a cell has more than one run, and a missing metric renders as `—` — hinting to
  re-evaluate only for the metrics whose absence really means the run predates the layer that
  computes them (identity coverage, cluster purity, HOTA, tracklet purity, mixed-identity time),
  not for ones a null value is a real answer for; clicking a cell expands an inline row linking
  to that cell's runs.

### Limitations

- Merge/split/roster flags are annotations only; nothing currently consumes them to actually
  re-associate entities or rewrite a roster short of the offline `export-reid` pair pipeline.

## Known findings

Measured local findings recorded by the repository guidance:

- Kit-colour association is ineffective for player-level identity.
- Remaining ID switches are substantially a tracker-level problem that simple post-association
  cannot repair.
- **Phase 3 tracker benchmark + SPO-34 exit gate (2026-07-19).** On frozen detections, no
  off-the-shelf candidate cleared the pre-registered promotion bar (BoT-SORT+body-ReID/SPO-31
  directionally positive but sub-bar on purity; TDLP-bbox/SPO-32 and OC-SORT/SPO-33 regress) —
  the hardened BoT-SORT baseline stands as the interim shippable tracker. As-published references
  (CAMELTrack, full TDLP) run via the import adapter establish the SOTA ceiling: on **identical
  CAMELTrack multi-cue features** (only the association head differs), full **TDLP's
  link-prediction head beat CAMELTrack's transformer head on every metric** (SportsMOT held-out,
  5-seq: purity 0.968 vs 0.941, mixed-track 10.1 vs 18.3 s, HOTA 0.910 vs 0.893), and
  **appearance+pose is decisive within TDLP** (bbox-only 0.868 → full 0.953 purity; the SPO-32
  "TDLP over-connects" result was a missing-appearance artifact). SPO-34 selected the **TDLP
  link-prediction head** as the architecture for the shippable build
  (`docs/prds/shippable-multi-cue-tracklet-system.md`). All SOTA weights are non-shippable
  (CC BY-NC SportsMOT training data + research-only ReID); the build retrains on permissive data.
  Reports: `docs/reports/2026-07-19-{spo34-phase3-exit-gate,tdlp-full-spike}.md`,
  `docs/reports/2026-07-18-spo3{0,1,2,3,5}-*.md`.
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

- **Phase 0 exit gate (SPO-21): detection is the dominant error source; the tracker ceiling is
  high but scene-dependent.** Full report: [`docs/reports/2026-07-17-phase0-exit-gate.md`](reports/2026-07-17-phase0-exit-gate.md).
  Current baseline (`v1-local-eval`: `yolo-local` `football-player-detection.pt` + BoT-SORT,
  `sample_stride=2`) vs. its oracle-detection counterpart (`pipeline.oracle-botsort-eval.yaml`:
  GT boxes + identical BoT-SORT) over **held-out** sequences: SoccerNet SNMOT-124–127 (manifest
  hash `7dfe09fdc5cc`) and SportsMOT 6-seq subset (`581ecb80614c`), IoU 0.5, `device=cuda`.
  Mean HOTA (tracklet) baseline→oracle: SoccerNet 0.488→0.836, SportsMOT 0.170→0.857. On
  SportsMOT basketball/volleyball the football detector yields ~0 tracklets (HOTA ≈ 0) — total
  cross-sport detection failure. Every ID switch attributed via oracle comparison (SPO-19), 100%
  coverage: 75% (SoccerNet) / 63% (SportsMOT) attributed to the **detection** layer, remainder to
  online association — so 63–75% of switches vanish under perfect detection. This **refines** the
  earlier "remaining ID switches are substantially a tracker-level problem" finding above: with a
  real detector, they are *substantially a detection-layer problem*; the tracker-level residual is
  the minority (25–37%). The oracle ceiling is high (HOTA ~0.84–0.86, MOTA ~0.99) but not
  uniformly near-perfect — HOTA 0.73–0.78 with 77–92 switches on crowded scenes even with GT
  boxes, vs. 0.97 / 10 switches on easy scenes. Repeat runs of the baseline agreed exactly
  (max |Δ| 0.0 across all metrics; pre-registered tolerances ratio 0.005 / ID-switch 1 /
  mixed-identity 0.5 s), so measured deltas are signal not noise. Code revision: `spo-21-phase0-gate`
  @ `0d2274c`. Stop/go decision recorded in the gate report §6.

- **Phase 1 exit gate (SPO-22): parameter hardening closes only ~4% of the gap to the oracle
  ceiling; two PRD-assumed axes are inert or backwards.** Full report:
  [`docs/reports/2026-07-17-phase1-hardening.md`](reports/2026-07-17-phase1-hardening.md).
  Pre-registered OAT sweeps (sample stride, detector confidence, lost-track buffer, activation
  threshold, min length, CMC, `high_conf_det_threshold`) + combination candidates over SoccerNet
  held-out (manifest `7dfe09fdc5cc`), 100 rows / 0 failures, IoU 0.5, `device=cuda`. The
  rule-selected **hardened baseline** is `configs/pipeline.v1-hardened-eval.yaml` (stride 1,
  detector confidence 0.4, `high_conf_det_threshold` 0.4) — **the program comparator for later
  phases**. Confirmed on both tiers: SoccerNet HOTA (tracklet) 0.4878→0.5019, purity
  0.8985→0.9526, mixed-identity 32.2→14.0 s; SportsMOT HOTA 0.1702→0.1881, purity 0.8552→0.8868,
  mixed-identity 12.3→5.9 s. Raw ID-switch counts *rise* (121→147 SoccerNet) because stride 1
  doubles the frames and therefore the switch opportunities, while the duration-weighted measure
  of the same failure halves — do not read the switch count alone. Measured findings:
  (a) **`track_activation_threshold` is inert** in this pipeline (0.15/0.25/0.4 give
  byte-identical artifacts — it only gates track *spawning*, the detector's confidence floor
  leaves nothing to gate, second association recovers what it declines, `min_length` removes
  short spawns); (b) the PRD's low-score-association hypothesis is **refuted for the current
  detector** — every low-floor probe regressed (conf 0.1: ΔHOTA −0.0128, Δpurity −0.0268), the
  pre-tracker floor wants to go *up* (0.4), not down (re-test after Phase 2's YOLOX, whose
  low-score detections may be better calibrated); (c) `min_length` does **not** buy purity
  (pre/post-filter identical at every value, purity flat 0.8984→0.8976 across 3/5/10 — impurity
  lives in long tracklets); (d) camera-motion compensation is **load-bearing** (disabling costs
  −0.0445 HOTA / −0.0479 purity); (e) `lost_track_buffer_s` is redundant once other axes are
  combined. The +0.0141 HOTA gain is **~4% of the 0.348 gap** to the Phase 0 oracle-detection
  ceiling (0.836), i.e. ~96% of the gap survives tuning — independently corroborating Phase 0's
  detection-first finding. Code revision: `spo-22-phase1-gate` (off `main` `2ab2e18`).

- **Phase 2 frozen reference detections (SPO-25/26): the imported YOLOX closes the SportsMOT
  detection-attributable gap; SoccerNet incumbent is now frozen and replayable.** Full report:
  [`docs/reports/2026-07-17-phase2-frozen-detections.md`](reports/2026-07-17-phase2-frozen-detections.md).
  Same-protocol (stride 1, IoU 0.5, `device=cuda`) comparison over all 9 SportsMOT tuning +
  held-out sequences: the frozen MixSort YOLOX-X (`yolox-local`, checkpoint sha256
  `58547880fb73...ed1c`) scores mean detection_ap 0.9844 vs. the Phase 1 hardened incumbent's
  0.2641 (medians 0.9866 vs. 0.0014); on the 6 cross-sport basketball/volleyball sequences the
  incumbent is 0.0000–0.0031 AP while YOLOX is 0.9832–0.9978, confirming and closing the Phase 0
  cross-sport detection failure under same-protocol (stride-1) conditions; on football the
  incumbent's existing 0.79 mean AP improves to 0.97, narrowing rather than eliminating that
  gap. Tracker-headline deltas track detection closely (idf1_entity 0.82 vs. 0.17). Determinism:
  export byte-identical on re-export; one fp32/GPU repeat-inference check bitwise-identical.
  Separately, the SoccerNet tier's hosted incumbent was frozen via the existing response cache
  (12 sequences, confidence 0.1, 9000 entries) and proven replayable — zero network, byte-
  identical `det.txt` — with no code change beyond the existing cache. The YOLOX weights remain
  **selection-only, non-shippable** (CC BY-NC 4.0 training data); this closes the Phase 0 stop/go
  decision's Phase 2 scope and hands inputs to the SPO-28 gate (HITL, not yet decided). Code
  revision: `phase2-frozen-detections` branch off `main` `5c9229a`.

Do not generalize these findings beyond the evaluated data. Link future claims to an experiment
report, run set, dataset split, and code/model revision.

## Immediate next milestones

1. Add a learned body re-ID associator baseline.
2. Persist reusable anchor quality and identity evidence.
3. Add identity-label comparison against GT jersey/roster records, and abstention/coverage
   curves (currently only single-run snapshot numbers exist, not trends across frames or a
   run set).
4. Run the pending SPO-50 human-gated benchmark pass (real T-DEED weights, GPU, the
   `external-spotters/` env) to get the first measured `avg-mAP@1` number, then design and
   scope the follow-up **shippable clean-room T-DEED-equivalent spotter** — a permissively
   licensed, permissively trained retrain of the ball-action-spotting capability, the same
   reference→shippable sequence the tracklet program followed (SPO-32/35 → the shippable
   multi-cue tracklet system, `docs/prds/shippable-multi-cue-tracklet-system.md`) — per the
   Out of Scope section of `docs/prds/reference-action-spotting-tdeed.md`.

## Maintenance checklist

When implementation changes:

1. Update the relevant table row and status.
2. Add or remove the implementation path.
3. Update evaluation and UI capability lists where affected.
4. Link measured claims to their experiment report.
5. Update the verification date only after inspecting the implementation.
6. Do not mark a researched model as implemented merely because it appears in `../../docs/technology/`.
