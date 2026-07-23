# Run provenance (SPO-10)

Every `manifest.json` written by `PipelineRunner` carries a `provenance` block: what produced
the run, so results stay comparable months later. The block is append-only after run
completion -- only `evaluation_set_hash`/`evaluation_set_source` are filled in later, in
place, by the GT auto-scoring hook (see [Refresh timing](#refresh-timing)); everything else
is fixed when the run finishes. This document is written
for implementers of the benchmark runner (SPO-17) and any future import adapter that needs
to read or reason about this block — it is the consumer-facing contract, not an
implementation walkthrough. Code: `packages/matchlab_core/src/matchlab_core/provenance.py`.

## The "unknown" convention

**Every declared field is always present in the output.** A value the recorder could not
determine is the literal string `"unknown"` (or `null` for the small number of fields typed
as optional — see below), never an absent key. This means:

- A consumer can safely read `manifest["provenance"]["git_revision"]` etc. without a
  `KeyError`/`.get()` guard **for any manifest written since this feature landed, or read back
  through pydantic** (`RunManifest.model_validate`, which fills in the all-`"unknown"` default
  via `Field(default_factory=RunProvenance)` for older manifests missing the key). A manifest
  served as raw JSON without that revalidation step (e.g. `matchlab_server`'s
  `GET /api/runs/{id}` today, which does a plain `json.loads`) may be a pre-feature manifest
  with **no `provenance` key at all** -- guard for its absence there.
- `"unknown"` is a real, meaningful value — it says "we could not determine this," not "we
  forgot to look." Treat two runs both reporting `"unknown"` for the same field as
  **not** proven identical on that axis; only equal *known* values are a positive match.
- `null` is reserved for fields with a genuine "this doesn't apply" meaning (e.g.
  `weights_sha256: null` for a hosted model with no local weights file to hash) — it is
  distinct from `"unknown"` (`"we don't know"`).

## Schema

### `RunProvenance` (`manifest.json["provenance"]`)

| Field | Type | Meaning |
|---|---|---|
| `git_revision` | `str` | Short SHA of the commit the runner ran from, `-dirty` suffixed if the working tree had uncommitted changes at run time. `"unknown"` if git was unavailable (e.g. no `.git`, git not installed). |
| `package_versions` | `dict[str, str]` | Installed version of every package in `DEFAULT_PACKAGE_NAMES` (`matchlab-core`, `torch`, `trackers`, `supervision`, `inference`, `ultralytics`, `transformers`, `numpy`, `opencv-python`, `motmetrics`), via `importlib.metadata.version`. A package that isn't installed still appears, with value `"unknown"`. |
| `stages` | `dict[str, StageProvenance]` | One entry per stage that actually executed (skipped/absent stages have no entry), keyed by the `StageKind` string value (`"detect"`, `"track"`, ...). |
| `evaluation_set_hash` | `str` | `sha256` of the canonical form of the ground-truth JSON this run was scored against (see [Canonical hashing](#canonical-hashing-rule)). `"unknown"` when the run has no associated GT at manifest-write time — true for every run at pipeline-completion time (the pipeline runner cannot see GT linkage, which is server-side), and updated in place by the server's auto-scoring hook once a GT-backed run is actually scored. |
| `evaluation_set_source` | `str \| null` | Path to the GT file (or, for the benchmark runner, a `configs/datasets/<tier>.json` manifest) that was hashed. `null` until `evaluation_set_hash` is known. |

### `StageProvenance`

| Field | Type | Meaning |
|---|---|---|
| `impl` | `str` | The registered implementation name (`registry.build`'s second argument), e.g. `"yolo-local"`, `"botsort"`. |
| `params` | `dict` | The stage's **fully resolved** params — the pydantic `Params` model dumped after defaults were filled in, not the raw YAML. This is where resolved confidence/NMS/tiling/TTA/transform settings live; a stage with no tunable params simply reports its (possibly empty) params dict. |
| `models` | `list[ModelProvenance]` | Every model this stage instance carries. Empty for model-free/heuristic stages (trackers, heuristic team classifiers, the event engine, etc). |

### `ModelProvenance`

| Field | Type | Meaning |
|---|---|---|
| `architecture` | `str` | Model family, e.g. `"yolo"`, `"osnet"`, `"insightface-buffalo_l"`. `"unknown"` when the stage genuinely cannot determine it (e.g. an arbitrary hosted model id). |
| `revision` | `str` | Model id / checkpoint version string. For hosted detectors this is the Roboflow model id (`"football-players-detection-3zvbc/11"`); for local weights it is `"unknown"` (no version is pinned anywhere in the repo today). |
| `weights_path` | `str \| null` | Local filesystem path to the weights file, when one exists. `null` for hosted models with no local file. |
| `weights_sha256` | `str \| null` | Streaming SHA-256 of `weights_path`'s bytes, when the file exists at manifest-write time. `null` when there is no local file, **or** when `weights_path` is set but the file is missing (e.g. weights not yet downloaded). |
| `lineage` | `str` | Free-text pretraining/fine-tuning lineage description, e.g. `"pretrained (MSMT17), no fine-tuning"`, `"hosted (unpinned)"`. `"unknown"` by default. |
| `training_commit` | `str \| null` | Commit the model was trained/fine-tuned at, when applicable (`matchlab-train` experiment lineage). |
| `training_config` | `str \| null` | Path to the training config used, when applicable. |
| `training_seed` | `int \| null` | Training-time random seed, when applicable. |
| `dataset_split_manifest` | `str \| null` | Path to the `configs/datasets/<tier>.json` the model's training/eval split came from, when applicable. |
| `dataset_split_manifest_sha256` | `str \| null` | `hash_dataset_manifest` of the above. |
| `detections_cache_hash` | `str \| null` | `HostedDetectionCache.content_hash()` (SPO-10 part 2) — the fingerprint of the frozen hosted-API responses backing this model's detections, when the stage has response caching enabled (`RoboflowDetector` with `cache_mode` != `"off"`). `null` when caching is off or the stage doesn't cache at all. The cache's own location and mode (`cache_dir`, `cache_mode`) live in the stage's `params` snapshot, not here — this field is only the content fingerprint. Refreshed by the runner both after `prepare()` and after the stage finishes executing (see [Refresh timing](#refresh-timing) below), so a cold cache warmed during a `readwrite` run is reflected, not the pre-run empty-cache hash. |
| `license` | `LicenseAxes` | See below. |

### `LicenseAxes`

License / commercial-use status recorded **per axis** — code, weights, and training data are
independent facts; a permissive code license says nothing about the weights or training
data. Each field is a free-text status string, default `"unknown"`.

| Field | Meaning |
|---|---|
| `code` | The runtime/inference code's license, e.g. `"AGPL-3.0 (ultralytics, local-eval only, non-shippable)"`, `"proprietary hosted API (Roboflow)"`, `"MIT (insightface)"`. |
| `weights` | The distributed weights/checkpoint's license, e.g. `"research-only (buffalo_l pack; commercial deployment needs a licensed pack)"`. |
| `training_data` | The training dataset's license/usage terms, when known. Almost always `"unknown"` today — no stage currently records this axis with confidence. |

### Refresh timing

`stage.provenance()` is called by `PipelineRunner._exec` at three points, each overwriting
`StageProvenance.models` in the in-memory manifest (which is then re-saved):

1. Immediately when the stage is built, before `prepare()` runs.
2. Immediately after `prepare()` returns, so fields only `prepare()` can resolve (e.g. a
   downloaded weights path) are captured.
3. Immediately after the stage's main method (`detect()`, `track()`, ...) returns
   successfully, so fields that only finish populating as the stage *runs* — e.g.
   `detections_cache_hash`, which grows as a `readwrite`-mode cache is filled frame by
   frame during `detect()` — reflect the stage's actual output, not its pre-run state.

A stage instance's `provenance()` is expected to be safe to call repeatedly and reflect
current state each time (no memoization that would go stale across these three calls).

## Canonical hashing rule

Two hashing utilities (`hash_evaluation_set`, `hash_dataset_manifest`) both hash the
**canonical form** of a JSON document:

```python
obj = json.loads(text)
canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

Consequence: **key order and whitespace in the source file never change the hash; any
semantic change to the parsed value always does.** Two GT files that differ only in
formatting (pretty-printed vs. minified, keys reordered) hash identically. A single changed
box coordinate, added/removed track, or renamed field changes the hash.

`sha256_file(path)` is a separate, simpler primitive: a streaming SHA-256 over a file's raw
bytes (used for weights files, which are binary, not JSON).

## The refusal primitive: `check_evaluation_set`

```python
def check_evaluation_set(expected_hash: str, actual_hash: str, context: str) -> None
```

Raises `RuntimeError` naming both hashes and the context string when they differ; no-op when
equal. This mirrors the existing embedder-provenance gate in
`matchlab_train.experiments.reid_ablation._sweep_one` ("Embedder provenance mismatch"): name
both values, no silent fallback to "close enough."

This function is the primitive the benchmark runner (SPO-17,
`matchlab_train.experiments.benchmark`) calls before aggregating two runs' metrics
together — if their `evaluation_set_hash`es differ, the runs were scored against different
ground truth and must not be silently averaged. Wired in as
`_check_evaluation_set_consistency`: it runs once per pair of completed rows scoring the same
sequence, before any of `summary.tables`/`summary.comparison` is computed.

## Worked example

```json
{
  "run_id": "a1b2c3d4",
  "...": "... (rest of manifest.json unchanged) ...",
  "provenance": {
    "git_revision": "5da1f12-dirty",
    "package_versions": {
      "matchlab-core": "0.1.0",
      "torch": "2.4.1",
      "trackers": "2.4.0",
      "supervision": "0.24.0",
      "inference": "0.29.0",
      "ultralytics": "unknown",
      "transformers": "4.44.2",
      "numpy": "1.26.4",
      "opencv-python": "4.10.0.84",
      "motmetrics": "1.4.0"
    },
    "stages": {
      "detect": {
        "impl": "yolo-local",
        "params": {
          "weights": "data/weights/football-player-detection.pt",
          "ball_weights": "",
          "imgsz": 1280,
          "confidence": 0.3,
          "ball_buffer_size": 10,
          "ball_max_gap_frames": 30
        },
        "models": [
          {
            "architecture": "yolo",
            "revision": "unknown",
            "weights_path": "data/weights/football-player-detection.pt",
            "weights_sha256": "9f2a...c81e",
            "lineage": "unknown",
            "training_commit": null,
            "training_config": null,
            "training_seed": null,
            "dataset_split_manifest": null,
            "dataset_split_manifest_sha256": null,
            "detections_cache_hash": null,
            "license": {
              "code": "AGPL-3.0 (ultralytics, local-eval only, non-shippable)",
              "weights": "unknown",
              "training_data": "unknown"
            }
          }
        ]
      },
      "track": { "impl": "botsort", "params": { "...": "..." }, "models": [] }
    },
    "evaluation_set_hash": "6b1e...f0a2",
    "evaluation_set_source": "data/videos/soccernet/SNMOT-124.gt.json"
  }
}
```

Notes on this example:

- `ultralytics` shows `"unknown"` even though the pipeline clearly used a YOLO detector:
  `ultralytics` is deliberately not a declared workspace dependency (AGPL boundary, see
  `CLAUDE.md`) and is only present in an environment that opted in with
  `--with ultralytics`. This is correct, honest behavior, not a bug — a run's
  `package_versions` block records what was actually importable in that process, which for
  `ultralytics` specifically depends on how the runner was invoked. Note also that the
  fixed name `"opencv-python"` in `DEFAULT_PACKAGE_NAMES` is a specific PyPI distribution
  name; an environment that only has `opencv-python-headless` installed (the core
  dependency pinned in `matchlab_core`'s `pyproject.toml`) and not `opencv-python` itself
  will report `"unknown"` for that key even though OpenCV is clearly usable — check
  `package_versions` for the exact distribution name you care about, not just the logical
  capability.
- `track` here shows an empty `models` list — BoT-SORT is a heuristic tracker with no
  weights, not an oversight.
- `evaluation_set_hash`/`evaluation_set_source` are `"unknown"`/`null` on every manifest as
  written by the pipeline runner itself; they are filled in afterward, in place, by the
  server's GT auto-scoring hook (`matchlab_server.evaluation.evaluate_run_against_gt`) the
  first time a GT-backed run is scored.

## Related: external tracklet exchange (SPO-18)

`matchlab_core.exchange` defines two sibling sidecar schemas for exchanging data with
external MOT trackers rather than the pipeline's own `RunProvenance`: `frozen-detections/v1`
(a `det.txt` export's `detections_provenance.json`) and `ExternalProvenance` (the sidecar an
imported external tracker's output must carry, folded into the imported run's
`StageProvenance.params` and copied verbatim to `external_provenance.json`). Both reuse this
document's `LicenseAxes` and `check_evaluation_set` primitives rather than redefining them.
Full field-level detail lives in `exchange.py`'s module and class docstrings and in
[`implementation-status.md`](implementation-status.md); this is a pointer, not a duplicate
spec.

## Related: the benchmark runner (SPO-17)

`matchlab_train.experiments.benchmark` (`BenchmarkExperiment`) is the main consumer of this
module: it calls `hash_dataset_manifest` on the selected `configs/datasets/<tier>.json`,
`hash_evaluation_set` per sequence's GT, and `check_evaluation_set` (via
`_check_evaluation_set_consistency`) to refuse aggregating rows scored against different
ground truth. Full behavior — candidate matrix, gates, tables, tolerances — is documented in
[`implementation-status.md`](implementation-status.md), not here.

## What is out of scope here

- **Hosted-response caching** (Task 2 of SPO-10) is implemented in
  `matchlab_core.stages.detect.hosted_cache` (`HostedDetectionCache`) and wired into
  `RoboflowDetector`, not in this module — `provenance.py` only defines the
  `ModelProvenance.detections_cache_hash` field it reports into (see above). Caching for
  local detectors, eviction/GC policy, and the benchmark runner's use of the cache hash
  remain out of scope.
- **Candidate matrix, aggregation tables, and tolerance-comparison logic** for the benchmark
  runner (SPO-17) live in `matchlab_train.experiments.benchmark`, not this module — this file
  only defines the `check_evaluation_set`/hashing primitives that runner calls (see above).
