"""Core benchmark runner (SPO-17 part 1): the offline experiment every later
stop/go gate runs through. Loads a dataset tier manifest (`configs/datasets/
<tier>.json`), expands a candidate matrix (pipeline candidates with config
overrides + parameter sweeps), runs each candidate over each selected
sequence, scores every completed run with the full evaluation stack
(`pitchlab_core.evaluation`), and emits provenance-stamped per-sequence rows.

Aggregation, provenance-consistency gates, tolerance comparisons, and import
candidates are Task 9 -- this module only records rows the aggregator will
later fold together (`summary.note` says so explicitly). Offline-first: no
`pitchlab_server` imports, no DB -- dataset-manifest path resolution below is
deliberately reimplemented against pure paths rather than importing
`pitchlab_server.settings`.

Refusals are loud and name what's wrong (path, candidate, role) rather than
silently dropping a row: a missing manifest/video/gt, an empty role
selection, an unknown override path, or a duplicate candidate name all raise
before any pipeline runs. Once running, a single candidate's pipeline failure
(e.g. a bad GT path fed to the oracle detector) does NOT abort the
experiment -- it becomes a `status: "failed"` row, also listed in
`summary.failed_rows`, so a failed run can never look like a scored one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pitchlab_core.config import PipelineConfig
from pitchlab_core.evaluation import evaluate_run, headline_metrics
from pitchlab_core.gt import GroundTruth
from pitchlab_core.provenance import hash_dataset_manifest, hash_evaluation_set
from pitchlab_core.runner import PipelineRunner
from pitchlab_core.schemas.run import RunManifest, StageKind, StageStatus
from pydantic import BaseModel, Field, field_validator

from pitchlab_train.experiments.base import Experiment
from pitchlab_train.registry import register

_ROLES = ("tuning", "held_out")


class PipelineCandidate(BaseModel):
    """One entry of `Params.candidates`, already validated for `kind ==
    "pipeline"` -- Task 9 adds an ImportCandidate kind alongside this one,
    which is why `Params.candidates` stays `list[dict]` rather than
    `list[PipelineCandidate]`."""

    name: str
    kind: Literal["pipeline"] = "pipeline"
    config: str  # pipeline YAML path
    overrides: dict = Field(default_factory=dict)  # dotted-path -> value
    comparison_class: Literal["matched_data", "as_published"] = "matched_data"


class SweepSpec(BaseModel):
    candidate: str  # base candidate name
    param: str  # dotted path, e.g. "stages.track.params.lost_track_buffer_s"
    values: list


class ManifestSequence(BaseModel):
    """One resolved `configs/datasets/<tier>.json` sequence: `video`/`gt` are
    absolute (or at least resolvable-as-given) filesystem paths, already
    checked to exist."""

    name: str
    role: str
    video: str
    gt: str


class Params(BaseModel):
    dataset_manifest: str  # path to configs/datasets/<tier>.json
    roles: list[str] = Field(default_factory=lambda: ["tuning"])
    max_sequences: int | None = None  # cap AFTER role filter, manifest order
    candidates: list[dict]
    sweeps: list[SweepSpec] = Field(default_factory=list)
    device: str = "cpu"
    iou_threshold: float = 0.5
    # Pre-registered per gate issue; Task 9 consumes these for tolerance
    # comparisons -- this task only parses and records them verbatim.
    tolerances: dict[str, float] = Field(default_factory=dict)

    @field_validator("roles")
    @classmethod
    def _validate_roles(cls, v: list[str]) -> list[str]:
        bad = sorted(set(v) - set(_ROLES))
        if bad:
            raise ValueError(f"Unknown role(s) {bad}; roles must be a subset of {list(_ROLES)}")
        return v

    @field_validator("candidates")
    @classmethod
    def _validate_candidates_nonempty(cls, v: list[dict]) -> list[dict]:
        if not v:
            raise ValueError("Params.candidates must have at least one entry")
        return v


@register("benchmark")
class BenchmarkExperiment(Experiment):
    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()

        manifest_path = Path(p.dataset_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        dataset_manifest_hash = hash_dataset_manifest(manifest_path)

        sequences = _load_manifest_sequences(manifest_path, p.roles)
        n_available = len(sequences)
        max_sequences_applied = False
        if p.max_sequences is not None and n_available > p.max_sequences:
            sequences = sequences[: p.max_sequences]
            max_sequences_applied = True

        candidates = _expand_candidates(p.candidates, p.sweeps)

        rows: list[dict] = []
        failed_rows: list[dict] = []
        for candidate in candidates:
            # Loaded once per candidate (not per sequence): every sequence
            # this candidate runs against shares the same resolved config,
            # mirroring reid_ablation.py's per-variant config reuse.
            config = _load_pipeline_config(candidate)
            for seq in sequences:
                run_id = f"{candidate.name}-{seq.name}"
                run_dir = workdir / "runs" / run_id
                runner = PipelineRunner(
                    run_id=run_id,
                    video_path=seq.video,
                    config=config,
                    run_dir=run_dir,
                    device=p.device,
                )
                manifest = runner.run()

                full_eval: dict | None = None
                eval_path: str | None = None
                if manifest.status == StageStatus.COMPLETED:
                    gt_text = Path(seq.gt).read_text()
                    # Mirrors pitchlab_server.evaluation's GT auto-scoring
                    # hook: stamp the evaluation-set identity into the run's
                    # manifest.json before scoring -- these are native
                    # pipeline runs, only the server did this stamping
                    # before now.
                    manifest.provenance.evaluation_set_hash = hash_evaluation_set(gt_text)
                    manifest.provenance.evaluation_set_source = seq.gt
                    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))

                    gt = GroundTruth.model_validate_json(gt_text)
                    full_eval = evaluate_run(run_dir, gt, p.iou_threshold)
                    eval_file = run_dir / "eval.json"
                    eval_file.write_text(json.dumps(full_eval))
                    eval_path = str(eval_file.relative_to(workdir))

                row = _row_from_run(candidate, seq, run_id, manifest, full_eval, eval_path)
                rows.append(row)
                if row["status"] == "failed":
                    failed_rows.append({"run_id": run_id, "error": row["error"]})

        summary: dict = {
            "n_rows": len(rows),
            "n_failed": len(failed_rows),
            "failed_rows": failed_rows,
            "sequences": [{"name": s.name, "role": s.role} for s in sequences],
            "note": "aggregates land in Task 9",
        }
        if p.max_sequences is not None:
            summary["max_sequences"] = p.max_sequences
            summary["max_sequences_applied"] = max_sequences_applied

        result = {
            "dataset_manifest": p.dataset_manifest,
            "dataset_manifest_hash": dataset_manifest_hash,
            "roles": p.roles,
            "tolerances": p.tolerances,
            "candidates": [c.model_dump(mode="json") for c in candidates],
            "rows": rows,
            "summary": summary,
        }
        self.write_result(workdir, result)
        return result


# ---------------------------------------------------------------------------
# Pure functions -- unit-testable without running any pipeline.
# ---------------------------------------------------------------------------


def _repo_root_from_manifest_path(manifest_path: Path) -> Path:
    """The directory containing `configs/`, found by walking up from the
    manifest path to the ancestor literally named "configs" and returning
    its parent. Mirrors `pitchlab_train.datasets.manifest.update_tier_manifest`'s
    `settings.config_dir.parent` anchor (the same root the manifest writer
    resolved paths against) without importing pitchlab_server settings --
    this module is offline-first, no server/DB imports."""
    for ancestor in manifest_path.resolve().parents:
        if ancestor.name == "configs":
            return ancestor.parent
    raise RuntimeError(
        f"Dataset manifest path {manifest_path} is not under a 'configs' "
        "directory; cannot resolve its sequence video/gt paths relative to "
        "the repo root."
    )


def _load_manifest_sequences(manifest_path: str | Path, roles: list[str]) -> list[ManifestSequence]:
    """Load `configs/datasets/<tier>.json`, select sequences matching
    `roles` (manifest order preserved), and resolve/verify each entry's
    video/gt path against the repo root (the directory containing
    `configs/`). Loud refusals name the missing path/manifest/roles -- never
    a silently-empty or partially-resolved sequence list."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text())
    repo_root = _repo_root_from_manifest_path(manifest_path)

    selected_raw = [s for s in raw.get("sequences", []) if s["role"] in roles]
    if not selected_raw:
        raise RuntimeError(
            f"No sequences with role(s) {roles} found in dataset manifest {manifest_path}"
        )

    sequences: list[ManifestSequence] = []
    for entry in selected_raw:
        name = entry["name"]
        video_path = repo_root / entry["video"]
        gt_path = repo_root / entry["gt"]
        if not video_path.exists():
            raise FileNotFoundError(
                f"Sequence {name!r} video not found: {video_path} "
                f"(dataset manifest {manifest_path})"
            )
        if not gt_path.exists():
            raise FileNotFoundError(
                f"Sequence {name!r} ground truth not found: {gt_path} "
                f"(dataset manifest {manifest_path})"
            )
        sequences.append(
            ManifestSequence(name=name, role=entry["role"], video=str(video_path), gt=str(gt_path))
        )
    return sequences


def _descend(obj, part: str, full_path: str):
    """One dotted-path segment: attribute access for pydantic models (field
    must exist), dict lookup for dicts (keys typed `StageKind` -- i.e.
    `PipelineConfig.stages` -- are looked up by their string value, so a
    stage that isn't configured for this pipeline, or isn't a real
    StageKind at all, refuses loudly by name)."""
    if isinstance(obj, BaseModel):
        if part not in type(obj).model_fields:
            raise ValueError(
                f"Unknown override path '{full_path}': no field '{part}' on {type(obj).__name__}"
            )
        return getattr(obj, part)
    if isinstance(obj, dict):
        if obj and isinstance(next(iter(obj)), StageKind):
            try:
                key = StageKind(part)
            except ValueError:
                raise ValueError(
                    f"Unknown override path '{full_path}': no such stage '{part}'"
                ) from None
            if key not in obj:
                raise ValueError(
                    f"Unknown override path '{full_path}': pipeline has no stage "
                    f"'{part}' configured"
                )
            return obj[key]
        if part not in obj:
            raise ValueError(f"Unknown override path '{full_path}': key '{part}' not found")
        return obj[part]
    raise ValueError(
        f"Unknown override path '{full_path}': cannot descend into "
        f"{type(obj).__name__} at '{part}'"
    )


def _set_override(config: PipelineConfig, path: str, value) -> None:
    """Apply one dotted-path override (e.g.
    "stages.track.params.lost_track_buffer_s") to a loaded PipelineConfig,
    in place. Structural validity (does this stage/field exist) is checked
    eagerly here; whether `value` is itself sensible for the target stage's
    own Params model is a runtime concern (surfaces as a failed row when the
    pipeline actually runs), not validated here."""
    parts = path.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Unknown override path '{path}': expected at least two dotted segments"
        )
    obj = config
    for part in parts[:-1]:
        obj = _descend(obj, part, path)
    last = parts[-1]
    if isinstance(obj, BaseModel):
        if last not in type(obj).model_fields:
            raise ValueError(
                f"Unknown override path '{path}': no field '{last}' on {type(obj).__name__}"
            )
        setattr(obj, last, value)
    elif isinstance(obj, dict):
        # A stage's `params` dict is data, not schema -- new keys are valid
        # (the stage's own Params model validates them when constructed).
        obj[last] = value
    else:
        raise ValueError(
            f"Unknown override path '{path}': cannot set '{last}' on {type(obj).__name__}"
        )


def _load_pipeline_config(candidate: PipelineCandidate) -> PipelineConfig:
    """Load `candidate.config` and apply its overrides, refusing loudly (by
    path) if the config file is missing or an override path is structurally
    invalid. Called both during candidate expansion (to validate eagerly,
    before any pipeline runs) and once per candidate during the actual
    run."""
    path = Path(candidate.config)
    if not path.exists():
        raise FileNotFoundError(f"Candidate '{candidate.name}': pipeline config not found: {path}")
    config = PipelineConfig.from_yaml(path)
    for dotted_path, value in candidate.overrides.items():
        _set_override(config, dotted_path, value)
    return config


def _expand_candidates(raw_candidates: list[dict], sweeps: list[SweepSpec]) -> list[PipelineCandidate]:
    """Validate `raw_candidates` (kind == "pipeline" only, in this task),
    check each candidate's config path + overrides eagerly (refusing at
    expansion, before any pipeline runs), then expand `sweeps` into derived
    candidates: each `SweepSpec` value produces
    `f"{candidate}@{param.rsplit('.',1)[-1]}={value}"` with the sweep's
    override merged on top of the base candidate's own overrides. The base
    candidate itself is always retained. Duplicate resulting names (within
    the raw candidates, or introduced by a sweep) refuse loudly."""
    base_candidates: list[PipelineCandidate] = []
    seen_names: set[str] = set()
    for entry in raw_candidates:
        kind = entry.get("kind", "pipeline")
        if kind != "pipeline":
            raise ValueError(
                f"Unsupported candidate kind '{kind}' for candidate "
                f"{entry.get('name', '?')!r} (only 'pipeline' is implemented in "
                "this task; Task 9 adds more kinds)."
            )
        candidate = PipelineCandidate(**entry)
        if candidate.name in seen_names:
            raise RuntimeError(f"Duplicate candidate name '{candidate.name}'")
        seen_names.add(candidate.name)
        _load_pipeline_config(candidate)  # eager validation, result discarded
        base_candidates.append(candidate)

    by_name = {c.name: c for c in base_candidates}
    expanded: list[PipelineCandidate] = list(base_candidates)
    for sweep in sweeps:
        base = by_name.get(sweep.candidate)
        if base is None:
            raise RuntimeError(
                f"Sweep references unknown base candidate '{sweep.candidate}' "
                f"(known candidates: {sorted(by_name)})"
            )
        param_key = sweep.param.rsplit(".", 1)[-1]
        for value in sweep.values:
            name = f"{sweep.candidate}@{param_key}={value}"
            if name in seen_names:
                raise RuntimeError(
                    f"Duplicate candidate name '{name}' produced by sweep expansion "
                    f"(candidate={sweep.candidate!r}, param={sweep.param!r})"
                )
            seen_names.add(name)
            derived = base.model_copy(
                update={"name": name, "overrides": {**base.overrides, sweep.param: value}}
            )
            _load_pipeline_config(derived)  # eager validation, result discarded
            expanded.append(derived)
    return expanded


def _provenance_summary(manifest: RunManifest) -> dict:
    """Flatten `manifest.provenance` into the row's compact
    `provenance_summary`: git revision, evaluation-set hash, stage impls,
    and every model's identity (architecture/revision/weights hash/
    detections-cache hash), across all stages."""
    prov = manifest.provenance
    stage_impls = {stage: sp.impl for stage, sp in prov.stages.items()}
    model_identities = [
        {
            "architecture": m.architecture,
            "revision": m.revision,
            "weights_sha256": m.weights_sha256,
            "detections_cache_hash": m.detections_cache_hash,
        }
        for sp in prov.stages.values()
        for m in sp.models
    ]
    return {
        "git_revision": prov.git_revision,
        "evaluation_set_hash": prov.evaluation_set_hash,
        "stage_impls": stage_impls,
        "model_identities": model_identities,
    }


def _row_from_run(
    candidate: PipelineCandidate,
    seq: ManifestSequence,
    run_id: str,
    manifest: RunManifest,
    full_eval: dict | None,
    eval_path: str | None,
) -> dict:
    """Build one per-sequence row -- the unit Task 9 aggregates -- from an
    already-produced RunManifest (+ eval result, when scored). Pure: takes
    no run_dir, does no I/O, so it's unit-testable with a hand-built
    RunManifest and eval dict, no pipeline execution required. A failed run
    never carries `metrics`/`eval_path`/`provenance_summary` -- it must
    never look like a scored one."""
    row: dict = {
        "candidate": candidate.name,
        "comparison_class": candidate.comparison_class,
        "sequence": seq.name,
        "role": seq.role,
        "run_id": run_id,
        "config_name": manifest.config_name,
        "n_tracklets": manifest.metrics.get("n_tracklets"),
    }
    if manifest.status == StageStatus.COMPLETED:
        row["status"] = "completed"
        row["metrics"] = headline_metrics(full_eval)
        row["eval_path"] = eval_path
        row["provenance_summary"] = _provenance_summary(manifest)
    else:
        row["status"] = "failed"
        row["error"] = (manifest.error or "")[:500]
    return row
