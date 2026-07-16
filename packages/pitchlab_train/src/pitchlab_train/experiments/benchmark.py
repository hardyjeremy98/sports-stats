"""Core benchmark runner (SPO-17): the offline experiment every later stop/go
gate runs through. Loads a dataset tier manifest (`configs/datasets/
<tier>.json`), expands a candidate matrix (pipeline candidates with config
overrides + parameter sweeps, plus external-import candidates -- see
`ImportCandidate`), runs/scores each candidate over each selected sequence
with the full evaluation stack (`pitchlab_core.evaluation`), and emits
provenance-stamped per-sequence rows (part 1, SPO-17). Those rows then pass
through provenance aggregation gates (`_check_missing_provenance`,
`_check_provenance_consistency`, `_check_evaluation_set_consistency`), a
matched_data/as_published table split (`_build_tables`), and an optional
tolerance-band comparison against a baseline candidate (`_compute_comparison`,
part 2) before landing in `summary.tables` / `summary.comparison`.
Offline-first: no `pitchlab_server` imports, no DB -- dataset-manifest path
resolution below is deliberately reimplemented against pure paths rather than
importing `pitchlab_server.settings`.

Refusals are loud and name what's wrong (path, candidate, run_id, hash) rather
than silently dropping a row or aggregating over broken provenance: a missing
manifest/video/gt, an empty role selection, an unknown override path, a
duplicate candidate name, a provenance-less import, a reference-only system
entering a matched_data comparison, or inconsistent/missing provenance across
a candidate's rows all raise. The one exception is a single pipeline
candidate's *runtime* pipeline failure (e.g. a bad GT path fed to the oracle
detector), which does NOT abort the experiment -- it becomes a `status:
"failed"` row, also listed in `summary.failed_rows`, so a failed run can
never look like a scored one and is excluded from aggregates.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Literal

from pitchlab_core.config import PipelineConfig
from pitchlab_core.evaluation import evaluate_run, headline_metrics
from pitchlab_core.exchange import ExternalProvenance
from pitchlab_core.gt import GroundTruth
from pitchlab_core.provenance import (
    check_evaluation_set,
    hash_dataset_manifest,
    hash_evaluation_set,
)
from pitchlab_core.runner import PipelineRunner
from pitchlab_core.schemas.run import RunManifest, StageKind, StageStatus
from pydantic import BaseModel, Field, ValidationError, field_validator

from pitchlab_train.experiments.base import Experiment
from pitchlab_train.registry import register

_ROLES = ("tuning", "held_out")

# Headline metrics (pitchlab_core.evaluation.headline_metrics) where a LOWER
# value is the better outcome: identity switches, mixed-identity screen-time,
# and the detector's worst-case (p95) consecutive-miss streak. Every other
# headline metric (IDF1, MOTA, HOTA, purity, coverage, AP, recall, ...) is
# higher-is-better. Tolerance verdicts below invert deltas for this set.
LOWER_IS_BETTER = frozenset(
    {"idsw_tracklet", "idsw_entity", "mixed_track_seconds", "detection_miss_burst_p95"}
)


class PipelineCandidate(BaseModel):
    """One entry of `Params.candidates`, already validated for `kind ==
    "pipeline"` -- `ImportCandidate` below is the other kind, which is why
    `Params.candidates` stays `list[dict]` rather than
    `list[PipelineCandidate]` (each entry's `kind` picks which model
    validates it, in `_expand_candidates`)."""

    name: str
    kind: Literal["pipeline"] = "pipeline"
    config: str  # pipeline YAML path
    overrides: dict = Field(default_factory=dict)  # dotted-path -> value
    comparison_class: Literal["matched_data", "as_published"] = "matched_data"


class ImportCandidate(BaseModel):
    """An external tracker's output, already materialized as a run dir by
    `pitchlab_core.exchange.import_mot_tracklets` -- this runner never
    produces or owns that run dir, and never stamps/rewrites its
    manifest.json. `runs` maps sequence name -> that run dir; each entry's
    `external_provenance.json` is read and validated at expansion time (see
    `_validate_import_candidate`): a provenance-less import, or a
    reference-only system entered into a `matched_data` comparison_class,
    are both refused before any scoring happens (the PRD rule: reference-only
    systems can never silently enter a shipping comparison).

    Caveat: scoring is the one exception to "never produces or owns that run
    dir" -- `BenchmarkExperiment.run` writes/overwrites `eval.json` inside
    this externally-owned import dir on every benchmark run that includes
    it, same as it does for native pipeline run dirs. That file always
    reflects only the LAST benchmark scoring pass (whatever GT and
    `Params.iou_threshold` that pass used), not a durable per-import record
    -- manifest.json and external_provenance.json remain untouched.

    `comparison_class` has no default (unlike PipelineCandidate's
    "matched_data") -- an import's table membership must be stated
    explicitly, never assumed."""

    name: str
    kind: Literal["import"] = "import"
    runs: dict[str, str] = Field(min_length=1)  # sequence name -> imported run dir
    comparison_class: Literal["matched_data", "as_published"]


class SweepSpec(BaseModel):
    candidate: str  # base candidate name
    param: str  # dotted path, e.g. "stages.track.params.lost_track_buffer_s"
    values: list


class Compare(BaseModel):
    """`Params.compare`: names the matched_data candidate every other
    matched_data candidate is compared against, per-metric, using
    `Params.tolerances` (see `_compute_comparison`)."""

    baseline: str


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
    # Pre-registered per gate issue (never scientific-looking hardcoded
    # defaults in code) -- see `_compute_comparison` for how these turn into
    # improved/regressed/within_tolerance verdicts against `compare.baseline`.
    tolerances: dict[str, float] = Field(default_factory=dict)
    compare: Compare | None = None

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
        sequences_by_name = {s.name: s for s in sequences}

        rows: list[dict] = []
        failed_rows: list[dict] = []
        for candidate in candidates:
            if isinstance(candidate, ImportCandidate):
                for seq_name, run_dir_str in candidate.runs.items():
                    seq = sequences_by_name.get(seq_name)
                    if seq is None:
                        raise RuntimeError(
                            f"Import candidate '{candidate.name}': sequence "
                            f"'{seq_name}' is not among the selected sequences "
                            f"{sorted(sequences_by_name)} (roles={p.roles})"
                        )
                    run_dir = Path(run_dir_str)
                    run_id = f"{candidate.name}-{seq.name}"
                    ext_prov = _read_external_provenance(run_dir / "external_provenance.json")
                    raw_manifest = json.loads((run_dir / "manifest.json").read_text())

                    gt_text = Path(seq.gt).read_text()
                    evaluation_set_hash = hash_evaluation_set(gt_text)
                    gt = GroundTruth.model_validate_json(gt_text)
                    full_eval = evaluate_run(run_dir, gt, p.iou_threshold)
                    eval_file = run_dir / "eval.json"
                    eval_file.write_text(json.dumps(full_eval))

                    row = _row_from_import(
                        candidate,
                        seq,
                        run_id,
                        run_dir,
                        raw_manifest,
                        ext_prov,
                        full_eval,
                        str(eval_file),
                        evaluation_set_hash,
                    )
                    rows.append(row)
                continue

            # Loaded once per candidate (not per sequence): every sequence
            # this candidate runs against shares the same resolved config,
            # mirroring reid_ablation.py's per-variant config reuse.
            config = _load_pipeline_config(candidate)
            for seq in sequences:
                # Sweep-derived candidate names contain "@"/"=" (e.g.
                # "base@max_age_frames=10") -- both are legal path
                # characters on every filesystem this project targets, so
                # they're kept as-is rather than sanitized; run_dir below
                # becomes a real, readable directory name.
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

        # Provenance aggregation gates (Task 9): refuse loudly, before any
        # aggregate is computed, rather than silently aggregating a result
        # whose provenance can't be trusted.
        _check_missing_provenance(rows)
        _check_provenance_consistency(rows)
        _check_evaluation_set_consistency(rows)

        tables = _build_tables(candidates, rows)
        comparison = _compute_comparison(p.compare, p.tolerances, tables)

        summary: dict = {
            "n_rows": len(rows),
            "n_failed": len(failed_rows),
            "failed_rows": failed_rows,
            "sequences": [{"name": s.name, "role": s.role} for s in sequences],
            "tables": tables,
            "comparison": comparison,
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
        if obj and isinstance(next(iter(obj)), StageKind):
            # `obj` is PipelineConfig.stages itself (StageKind-keyed) -- a
            # bare "stages.<stage>" path would otherwise silently insert a
            # plain-string key alongside the StageKind-keyed entries (a
            # no-op override: PipelineRunner only ever looks stages up by
            # StageKind, so the inserted key is dead). Whole-StageConfig
            # replacement via override is questionable anyway -- refuse and
            # name a real target instead of coercing/accepting it.
            raise ValueError(
                f"Unknown override path '{path}': '{last}' would replace a "
                f"whole stage config on the stages map -- target a field "
                f"under it instead, e.g. '{path}.impl' or '{path}.params.<key>'"
            )
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


def _read_external_provenance(sidecar_path: Path) -> ExternalProvenance:
    """Read + validate one imported run dir's `external_provenance.json`,
    refusing loudly (naming the sidecar path) on any structural problem: a
    missing file, malformed JSON, or an incomplete/invalid ExternalProvenance
    schema -- a provenance-less external result is refused outright, same
    contract as `exchange._load_external_provenance` (which this mirrors;
    this module's refusal style is `RuntimeError` throughout, not
    `ValueError`)."""
    if not sidecar_path.exists():
        raise RuntimeError(f"No external_provenance.json found at {sidecar_path}")
    try:
        raw = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"external_provenance.json at {sidecar_path} is not valid JSON: {exc}"
        ) from exc
    try:
        return ExternalProvenance.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(f"Invalid external_provenance.json at {sidecar_path}: {exc}") from exc


def _validate_import_candidate(candidate: ImportCandidate) -> None:
    """Eager provenance validation for one `ImportCandidate`, at expansion
    time (before any scoring): every run dir in `candidate.runs` must carry
    a valid `external_provenance.json` (a provenance-less external result is
    refused outright -- the SPO-18 import contract), and a `reference_only`
    system can never enter a `matched_data` comparison_class (the PRD rule:
    a reference-only system can never silently enter a shipping
    comparison)."""
    for seq_name, run_dir in candidate.runs.items():
        sidecar_path = Path(run_dir) / "external_provenance.json"
        try:
            ext_prov = _read_external_provenance(sidecar_path)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Import candidate '{candidate.name}' (sequence '{seq_name}'): {exc} "
                "-- provenance-less input is refused"
            ) from exc
        if ext_prov.reference_only and candidate.comparison_class == "matched_data":
            raise RuntimeError(
                f"Import candidate '{candidate.name}': reference_only=True "
                f"(sequence '{seq_name}', {sidecar_path}) cannot enter "
                "comparison_class='matched_data' -- reference-only systems can "
                "never silently enter a shipping comparison"
            )


def _expand_candidates(
    raw_candidates: list[dict], sweeps: list[SweepSpec]
) -> list[PipelineCandidate | ImportCandidate]:
    """Validate `raw_candidates` (kind == "pipeline" or "import"), check each
    pipeline candidate's config path + overrides eagerly and each import
    candidate's provenance eagerly (refusing at expansion, before any
    pipeline runs or scoring), then expand `sweeps` into derived pipeline
    candidates: each `SweepSpec` value produces
    `f"{candidate}@{param.rsplit('.',1)[-1]}={value}"` with the sweep's
    override merged on top of the base candidate's own overrides. The base
    candidate itself is always retained. Duplicate resulting names (within
    the raw candidates, or introduced by a sweep) refuse loudly. Sweeps only
    apply to pipeline candidates -- one targeting an import candidate name
    refuses loudly rather than silently doing nothing."""
    base_candidates: list[PipelineCandidate | ImportCandidate] = []
    seen_names: set[str] = set()
    for entry in raw_candidates:
        kind = entry.get("kind", "pipeline")
        if kind == "pipeline":
            candidate: PipelineCandidate | ImportCandidate = PipelineCandidate(**entry)
        elif kind == "import":
            candidate = ImportCandidate(**entry)
        else:
            raise ValueError(
                f"Unsupported candidate kind '{kind}' for candidate "
                f"{entry.get('name', '?')!r} (must be 'pipeline' or 'import')."
            )
        if candidate.name in seen_names:
            raise RuntimeError(f"Duplicate candidate name '{candidate.name}'")
        seen_names.add(candidate.name)
        if isinstance(candidate, PipelineCandidate):
            _load_pipeline_config(candidate)  # eager validation, result discarded
        else:
            _validate_import_candidate(candidate)
        base_candidates.append(candidate)

    by_name = {c.name: c for c in base_candidates}
    expanded: list[PipelineCandidate | ImportCandidate] = list(base_candidates)
    for sweep in sweeps:
        base = by_name.get(sweep.candidate)
        if base is None:
            raise RuntimeError(
                f"Sweep references unknown base candidate '{sweep.candidate}' "
                f"(known candidates: {sorted(by_name)})"
            )
        if not isinstance(base, PipelineCandidate):
            raise RuntimeError(
                f"Sweep references import candidate '{sweep.candidate}' -- sweeps "
                "only support pipeline candidates"
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


def _provenance_summary_from_dict(prov: dict) -> dict:
    """Shared core of `_provenance_summary` (native runs, from a
    `RunManifest.provenance` dump) and `_import_provenance_summary` (import
    runs, from the imported run dir's raw-dict manifest -- never
    RunManifest-valid, see module docstring): git revision, evaluation-set
    hash, stage impls, and every model's identity (architecture/revision/
    weights hash/detections-cache hash), across all stages. Missing keys
    default to "unknown"/`None` the same way the pydantic schema they
    normally come through would -- never a KeyError on a raw dict."""
    stages = prov.get("stages", {})
    stage_impls = {stage: sp.get("impl", "unknown") for stage, sp in stages.items()}
    model_identities = [
        {
            "architecture": m.get("architecture", "unknown"),
            "revision": m.get("revision", "unknown"),
            "weights_sha256": m.get("weights_sha256"),
            "detections_cache_hash": m.get("detections_cache_hash"),
        }
        for sp in stages.values()
        for m in sp.get("models", [])
    ]
    return {
        "git_revision": prov.get("git_revision", "unknown"),
        "evaluation_set_hash": prov.get("evaluation_set_hash", "unknown"),
        "stage_impls": stage_impls,
        "model_identities": model_identities,
    }


def _provenance_summary(manifest: RunManifest) -> dict:
    """Flatten `manifest.provenance` into the row's compact
    `provenance_summary` (native pipeline run)."""
    return _provenance_summary_from_dict(manifest.provenance.model_dump(mode="json"))


def _import_provenance_summary(
    raw_manifest: dict, ext_prov: ExternalProvenance, evaluation_set_hash: str
) -> dict:
    """`provenance_summary` for an import row: the raw-dict manifest's
    provenance block (stage impls, model identities -- the external system's
    identity, per SPO-18) plus the ExternalProvenance's system/variant/
    license axes. `evaluation_set_hash` is always the raw manifest's literal
    "unknown" (`import_mot_tracklets` never computes it -- the runner doesn't
    own that manifest); it's overridden here with the hash this row actually
    verified against the sequence's GT file, which is the value every
    consistency gate below compares -- "the row carries the truth"."""
    summary = _provenance_summary_from_dict(raw_manifest.get("provenance", {}))
    summary["evaluation_set_hash"] = evaluation_set_hash
    summary["external"] = {
        "system": ext_prov.system,
        "variant": ext_prov.variant,
        "license": ext_prov.license.model_dump(mode="json"),
        "reference_only": ext_prov.reference_only,
    }
    return summary


def _row_from_import(
    candidate: ImportCandidate,
    seq: ManifestSequence,
    run_id: str,
    run_dir: Path,
    raw_manifest: dict,
    ext_prov: ExternalProvenance,
    full_eval: dict,
    eval_path: str,
    evaluation_set_hash: str,
) -> dict:
    """Build one per-sequence row for an import candidate -- same shape as
    `_row_from_run`'s completed-row branch (candidate/comparison_class/
    sequence/role/run_id/config_name/n_tracklets/status/metrics/eval_path/
    provenance_summary), so aggregation treats native and import rows
    uniformly. Import rows are always "completed": there is no PipelineRunner
    execution to fail here, only scoring an already-materialized run dir --
    a structural problem there is a bug to fix, not a soft per-row failure,
    so it raises rather than becoming a failed row."""
    tracklets_path = run_dir / "tracklets.json"
    n_tracklets = len(json.loads(tracklets_path.read_text())) if tracklets_path.exists() else None
    return {
        "candidate": candidate.name,
        "comparison_class": candidate.comparison_class,
        "sequence": seq.name,
        "role": seq.role,
        "run_id": run_id,
        "config_name": raw_manifest.get("config_name", "unknown"),
        "n_tracklets": n_tracklets,
        "status": "completed",
        "metrics": headline_metrics(full_eval),
        "eval_path": eval_path,
        "provenance_summary": _import_provenance_summary(raw_manifest, ext_prov, evaluation_set_hash),
    }


# ---------------------------------------------------------------------------
# Provenance aggregation gates (Task 9) -- pure functions over completed
# rows, called before any aggregate is computed. PRD: "refuses to aggregate
# runs whose provenance is missing or inconsistent." Failed rows are never
# inspected here (they carry no provenance_summary at all, by design).
# ---------------------------------------------------------------------------


def _check_missing_provenance(rows: list[dict]) -> None:
    """Refuse if any completed row lacks a usable provenance_summary: no
    git_revision recorded, or an empty stage_impls. The literal string
    "unknown" is a legitimate recorded value in this codebase's provenance
    convention ("asked, found nothing" -- see provenance.py's module
    docstring) and passes this check; only an actually empty/absent value
    fails it."""
    for row in rows:
        if row["status"] != "completed":
            continue
        run_id = row["run_id"]
        candidate = row["candidate"]
        summary = row.get("provenance_summary")
        if not summary:
            raise RuntimeError(
                f"Run '{run_id}' (candidate '{candidate}') is missing "
                "provenance_summary entirely -- refusing to aggregate"
            )
        if not summary.get("git_revision"):
            raise RuntimeError(
                f"Run '{run_id}' (candidate '{candidate}') has no "
                "provenance_summary.git_revision recorded -- refusing to aggregate"
            )
        if not summary.get("stage_impls"):
            raise RuntimeError(
                f"Run '{run_id}' (candidate '{candidate}') has an empty "
                "provenance_summary.stage_impls -- refusing to aggregate"
            )


def _model_identity_set(model_identities: list[dict]) -> frozenset[tuple[str, str, str | None]]:
    """(architecture, revision, weights_sha256) per model -- deliberately
    excludes detections_cache_hash: a cache-hit/miss difference across
    sequences is not a model identity change."""
    return frozenset(
        (m["architecture"], m["revision"], m.get("weights_sha256")) for m in model_identities
    )


def _check_provenance_consistency(rows: list[dict]) -> None:
    """Refuse if two completed rows of the SAME candidate disagree on
    stage_impls or model identity set -- e.g. a candidate that silently ran
    against different code or weights on different sequences, which would
    make its aggregate meaningless (the embedding-artifact-gate precedent;
    message style follows `check_evaluation_set`)."""
    by_candidate: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] == "completed":
            by_candidate.setdefault(row["candidate"], []).append(row)

    for candidate_name, crows in by_candidate.items():
        first = crows[0]
        first_impls = first["provenance_summary"]["stage_impls"]
        first_models = _model_identity_set(first["provenance_summary"]["model_identities"])
        for other in crows[1:]:
            other_impls = other["provenance_summary"]["stage_impls"]
            if other_impls != first_impls:
                raise RuntimeError(
                    f"Provenance mismatch for candidate '{candidate_name}': run "
                    f"'{first['run_id']}' has stage_impls {first_impls} but run "
                    f"'{other['run_id']}' has {other_impls} -- runs of one candidate "
                    "must share identical stage implementations"
                )
            other_models = _model_identity_set(other["provenance_summary"]["model_identities"])
            if other_models != first_models:
                raise RuntimeError(
                    f"Provenance mismatch for candidate '{candidate_name}': run "
                    f"'{first['run_id']}' has model identities {sorted(first_models)} but "
                    f"run '{other['run_id']}' has {sorted(other_models)} -- runs of one "
                    "candidate must share identical model identities"
                )


def _check_evaluation_set_consistency(rows: list[dict]) -> None:
    """Refuse if two completed rows of the SAME sequence (native or
    imported) disagree on evaluation_set_hash -- they would be scored
    against different ground truth without anyone noticing."""
    by_sequence: dict[str, list[dict]] = {}
    for row in rows:
        if row["status"] == "completed":
            by_sequence.setdefault(row["sequence"], []).append(row)

    for seq_name, srows in by_sequence.items():
        first = srows[0]
        expected = first["provenance_summary"]["evaluation_set_hash"]
        for other in srows[1:]:
            actual = other["provenance_summary"]["evaluation_set_hash"]
            check_evaluation_set(
                expected,
                actual,
                context=f"sequence {seq_name}: {first['run_id']} vs {other['run_id']}",
            )


# ---------------------------------------------------------------------------
# Aggregation + matched_data/as_published table separation (Task 9)
# ---------------------------------------------------------------------------


def _mean_median(values: list[float]) -> dict[str, float]:
    return {"mean": round(statistics.fmean(values), 4), "median": round(statistics.median(values), 4)}


def _aggregate_metric_block(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Mean/median per headline metric key, numeric values only: a key with
    no non-null value across `rows` (e.g. merge_precision when purity
    couldn't be computed, or every identity_* key on import rows, which have
    no identity layer at all) is simply absent -- there is nothing to
    aggregate, structurally, not a missing measurement."""
    keys: set[str] = set()
    for row in rows:
        keys.update(row["metrics"])
    block: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = [row["metrics"][key] for row in rows if row["metrics"].get(key) is not None]
        if values:
            block[key] = _mean_median(values)
    return block


def _aggregate_candidate_rows(candidate_name: str, comparison_class: str, rows: list[dict]) -> dict:
    """One candidate's aggregate over its own completed rows: mean/median
    per headline metric, n_sequences, and a per-role sub-breakdown when the
    completed rows span more than one role. Zero completed rows -> the
    aggregate is still present, with `n_sequences: 0` and `metrics: None`
    (never omitted -- a candidate that produced nothing is a fact worth
    recording, not a silently-missing row in the table)."""
    completed = [r for r in rows if r["candidate"] == candidate_name and r["status"] == "completed"]
    if not completed:
        return {"n_sequences": 0, "metrics": None}
    agg: dict = {
        "n_sequences": len(completed),
        "metrics": _aggregate_metric_block(completed),
    }
    roles = {r["role"] for r in completed}
    if len(roles) > 1:
        agg["by_role"] = {
            role: {
                "n_sequences": sum(1 for r in completed if r["role"] == role),
                "metrics": _aggregate_metric_block([r for r in completed if r["role"] == role]),
            }
            for role in sorted(roles)
        }
    return agg


def _build_tables(
    candidates: list[PipelineCandidate | ImportCandidate], rows: list[dict]
) -> dict[str, dict[str, dict]]:
    """`summary.tables`: `{"matched_data": {candidate: aggregate},
    "as_published": {candidate: aggregate}}` -- two separate tables, never
    merged (a `comparison_class` value outside the two refuses at expansion
    already, via the pydantic Literal). Every candidate appears in exactly
    one table, including candidates with zero completed rows."""
    tables: dict[str, dict[str, dict]] = {"matched_data": {}, "as_published": {}}
    for candidate in candidates:
        agg = _aggregate_candidate_rows(candidate.name, candidate.comparison_class, rows)
        tables[candidate.comparison_class][candidate.name] = agg
    return tables


# ---------------------------------------------------------------------------
# Tolerance-band comparison (Task 9)
# ---------------------------------------------------------------------------


def _verdict(delta: float, tol: float, metric: str) -> str:
    signed = -delta if metric in LOWER_IS_BETTER else delta
    if signed > tol:
        return "improved"
    if signed < -tol:
        return "regressed"
    return "within_tolerance"


def _compute_comparison(
    compare: Compare | None, tolerances: dict[str, float], tables: dict[str, dict[str, dict]]
) -> dict | None:
    """`summary.comparison`: for every matched_data candidate other than
    `compare.baseline`, per metric in `tolerances`, delta = candidate_mean -
    baseline_mean and a verdict (`_verdict`, LOWER_IS_BETTER-aware). A metric
    named in `tolerances` but absent from either aggregate (no headline value
    ever recorded, or the candidate has zero completed rows) verdicts
    "unavailable" with a null delta -- present, never omitted. Only
    matched_data candidates are ever compared: as-published rows are
    reference-only by definition, so the baseline must itself be
    matched_data, and no as_published candidate ever appears in `verdicts`
    -- no comparison across tables, ever. `None` when `compare` is unset."""
    if compare is None:
        return None
    baseline_name = compare.baseline
    matched = tables["matched_data"]
    as_published = tables["as_published"]
    if baseline_name not in matched:
        if baseline_name in as_published:
            raise RuntimeError(
                f"compare.baseline '{baseline_name}' is an as_published candidate; "
                "the tolerance-comparison baseline must be matched_data (as-published "
                "rows are reference-only by definition)"
            )
        raise RuntimeError(f"compare.baseline '{baseline_name}' is not a known candidate")

    baseline_agg = matched[baseline_name]
    baseline_metrics = baseline_agg["metrics"] or {}
    verdicts: dict[str, dict[str, dict]] = {}
    for candidate_name, agg in matched.items():
        if candidate_name == baseline_name:
            continue
        candidate_metrics = agg["metrics"] or {}
        per_metric: dict[str, dict] = {}
        for metric, tol in tolerances.items():
            baseline_metric = baseline_metrics.get(metric)
            candidate_metric = candidate_metrics.get(metric)
            if baseline_metric is None or candidate_metric is None:
                per_metric[metric] = {"delta": None, "verdict": "unavailable"}
                continue
            delta = round(candidate_metric["mean"] - baseline_metric["mean"], 4)
            per_metric[metric] = {"delta": delta, "verdict": _verdict(delta, tol, metric)}
        verdicts[candidate_name] = per_metric

    return {"baseline": baseline_name, "tolerances": tolerances, "verdicts": verdicts}


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
