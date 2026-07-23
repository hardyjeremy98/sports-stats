"""Provenance recording: what produced a run, so results stay comparable
months later (Phase 0 of the tracklet-modernization program, SPO-10 part 1).

Every declared field is always present in recorded output — unknown values
are the literal string "unknown" (or null where the schema allows it) rather
than an absent key, so a benchmark runner reading a manifest never has to
special-case a missing key. See docs/provenance.md for the full field-by-field
contract and the canonical-hashing rule this module implements.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

# Packages whose versions are always recorded by the runner (present in the
# output even when a package is absent — see collect_package_versions).
DEFAULT_PACKAGE_NAMES: list[str] = [
    "matchlab-core",
    "torch",
    "trackers",
    "supervision",
    "inference",
    "ultralytics",
    "transformers",
    "numpy",
    "opencv-python",
    "motmetrics",
]


class LicenseAxes(BaseModel):
    """License / commercial-use status, recorded per axis — code, weights,
    and training data licenses are independent facts; one being permissive
    says nothing about the others."""

    code: str = "unknown"
    weights: str = "unknown"
    training_data: str = "unknown"


class ModelProvenance(BaseModel):
    architecture: str = "unknown"
    revision: str = "unknown"  # model id / checkpoint version string
    weights_path: str | None = None
    weights_sha256: str | None = None  # null = no local weights file (e.g. hosted)
    lineage: str = "unknown"  # pretraining/fine-tuning lineage description
    training_commit: str | None = None
    training_config: str | None = None
    training_seed: int | None = None
    dataset_split_manifest: str | None = None  # path
    dataset_split_manifest_sha256: str | None = None
    # Hosted-detection response cache content hash (SPO-10 part 2): the
    # `HostedDetectionCache.content_hash()` backing this hosted model's
    # detections, when the stage has caching enabled. Deliberately a
    # dedicated field, not a repurposing of dataset_split_manifest -- that
    # field means training-data split lineage, a different fact, and a
    # future benchmark runner reading it needs that meaning to stay
    # unambiguous. `null` = no cache in play (cache_mode="off" or the stage
    # doesn't cache at all).
    detections_cache_hash: str | None = None
    license: LicenseAxes = Field(default_factory=LicenseAxes)


class StageProvenance(BaseModel):
    impl: str
    params: dict = {}  # resolved params snapshot
    models: list[ModelProvenance] = []


class RunProvenance(BaseModel):
    git_revision: str = "unknown"
    package_versions: dict[str, str] = {}
    stages: dict[str, StageProvenance] = {}  # keyed by StageKind value
    evaluation_set_hash: str = "unknown"
    evaluation_set_source: str | None = None  # path of the GT/dataset file hashed


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_hash(text: str) -> str:
    """sha256 over `json.dumps(json.loads(text), sort_keys=True,
    separators=(",", ":"))` — key order and whitespace never change the hash;
    any semantic change to the parsed value always does."""
    obj = json.loads(text)
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_evaluation_set(gt_json_text: str) -> str:
    """sha256 over the canonical form of a GT JSON document (see
    `_canonical_json_hash`)."""
    return _canonical_json_hash(gt_json_text)


def hash_dataset_manifest(path: str | Path) -> str:
    """Same canonicalization as `hash_evaluation_set`, applied to a
    `configs/datasets/<tier>.json` file — the benchmark runner's identity for
    "which sequences an evaluation run used"."""
    return _canonical_json_hash(Path(path).read_text())


def git_revision(repo_root: str | Path) -> str:
    """Short SHA, `-dirty` suffixed when the working tree has uncommitted
    changes; `"unknown"` on any failure (git missing, not a repo, ...) rather
    than raising — provenance must never crash a pipeline run."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        if not sha:
            return "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
        return f"{sha}-dirty" if status.strip() else sha
    except Exception:
        return "unknown"


def collect_package_versions(names: list[str]) -> dict[str, str]:
    """Resolved installed version per package name; absent packages are
    recorded as "unknown" (present in the dict — never silently omitted)."""
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


def check_evaluation_set(expected_hash: str, actual_hash: str, context: str) -> None:
    """Refusal primitive for the benchmark runner: raise when two
    evaluation-set hashes disagree, naming both hashes and the context so
    the mismatch is diagnosable at a glance (mirrors
    matchlab_train.experiments.reid_ablation._sweep_one's embedder-provenance
    gate). No-op when the hashes are equal. Wired into
    `matchlab_train.experiments.benchmark._check_evaluation_set_consistency`
    (SPO-17), which calls this once per pair of completed rows scoring the
    same sequence."""
    if expected_hash != actual_hash:
        raise RuntimeError(
            f"Evaluation-set mismatch in {context}: expected hash "
            f"'{expected_hash}' but got '{actual_hash}'."
        )
