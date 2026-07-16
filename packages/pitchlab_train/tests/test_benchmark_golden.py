"""PRD-mandated golden integration suite for Task 9 (SPO-17 part 2): one
deterministic end-to-end benchmark run over a synthetic 2-clip dataset with
2 pipeline candidates + 1 import candidate, asserting exact row shape/count,
matched_data/as_published table separation, tolerance-band verdicts, and the
refusal paths named in the task brief:

  (a) import candidate missing external_provenance.json
  (b) reference_only=true entering comparison_class="matched_data"
  (c) provenance inconsistency within one candidate's rows
  (d) evaluation-set hash mismatch within one sequence's rows
  (e) compare.baseline naming an as_published or unknown candidate

(c) and (d) build tampered row dicts from the golden run's own real rows and
call the pure gate functions directly, rather than re-running any pipeline
(per the task brief). (a), (b), (e) go through the full experiment/Params
stack since they refuse before any pipeline executes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pitchlab_core.demo import render_demo_video
from pitchlab_core.exchange import import_mot_tracklets
from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from pitchlab_core.schemas.geometry import Box
from pitchlab_train.config import ExperimentConfig
from pitchlab_train.experiments.benchmark import (
    LOWER_IS_BETTER,
    _check_evaluation_set_consistency,
    _check_provenance_consistency,
    _compute_comparison,
)
from pitchlab_train.registry import build

pytest.importorskip("motmetrics")  # every test in this module scores a real run

REPO = Path(__file__).parents[3]
STUB_CONFIG = str(REPO / "configs" / "pipeline.stub.yaml")

_FPS = 10.0
_SEQ_LENGTH = 10  # duration_s=1 * fps=10
_GT_BOX = Box(x1=100, y1=100, x2=140, y2=220)


def _write_gt(clip_path: Path) -> None:
    gt = GroundTruth(
        source="golden-suite",
        sequence=clip_path.stem,
        fps=_FPS,
        width=320,
        height=240,
        seq_length=_SEQ_LENGTH,
        tracks=[
            GroundTruthTrack(
                track_id=1,
                role="player",
                frames=[GroundTruthFrame(frame_idx=f, box=_GT_BOX) for f in range(_SEQ_LENGTH)],
            )
        ],
    )
    (clip_path.parent / f"{clip_path.stem}.gt.json").write_text(gt.model_dump_json())


def _write_dataset(tmp_path: Path) -> Path:
    """2 rendered demo clips + hand-built GT: clip-a (role=tuning),
    clip-b (role=held_out) -- both roles present so the per-role aggregate
    breakdown is also exercised by this same golden run."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    for name in ("clip-a", "clip-b"):
        render_demo_video(videos_dir / f"{name}.mp4", duration_s=1, fps=_FPS, width=320, height=240)
        _write_gt(videos_dir / f"{name}.mp4")

    configs_dir = tmp_path / "configs" / "datasets"
    configs_dir.mkdir(parents=True)
    manifest_path = configs_dir / "tier.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "golden-tier",
                "tier": "golden",
                "source_split": "test",
                "created": "2026-07-17",
                "sequences": [
                    {
                        "name": "clip-a",
                        "video": "videos/clip-a.mp4",
                        "gt": "videos/clip-a.gt.json",
                        "role": "tuning",
                    },
                    {
                        "name": "clip-b",
                        "video": "videos/clip-b.mp4",
                        "gt": "videos/clip-b.gt.json",
                        "role": "held_out",
                    },
                ],
                "notes": [],
            }
        )
    )
    return manifest_path


def _write_import_run(tmp_path: Path, name: str, *, reference_only: bool) -> Path:
    """A hand-written MOT tracklet, exactly matching clip-a's static GT box
    for every frame, imported via `import_mot_tracklets` -- a perfect-score
    external run (deterministic: idf1_tracklet == 1.0), independent of the
    stub pipeline's synthetic-detector randomness."""
    mot_path = tmp_path / f"{name}-mot.txt"
    lines = [
        f"{f + 1},1,100.00,100.00,40.00,120.00,0.900000" for f in range(_SEQ_LENGTH)
    ]
    mot_path.write_text("\n".join(lines) + "\n")

    sidecar_path = tmp_path / f"{name}-sidecar.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "system": "TDLP",
                "variant": "bbox-only",
                "repo_url": "https://example.com/tdlp",
                "commit": "abc123",
                "weights": "tdlp-v1.pt",
                "weights_sha256": "deadbeef",
                "license": {"code": "MIT", "weights": "unknown", "training_data": "unknown"},
                "reference_only": reference_only,
                "notes": "golden-suite fixture",
            }
        )
    )

    out_run_dir = tmp_path / f"{name}-run"
    import_mot_tracklets(
        mot_path, sidecar_path, out_run_dir, fps=_FPS, frame_count=_SEQ_LENGTH, sample_stride=1
    )
    return out_run_dir


def _candidates(import_run_dir: Path, *, import_comparison_class: str = "as_published") -> list[dict]:
    return [
        {"name": "base", "config": STUB_CONFIG, "comparison_class": "matched_data"},
        {
            "name": "variant",
            "config": STUB_CONFIG,
            "comparison_class": "matched_data",
            "overrides": {
                "stages.track.params.iou_threshold": 0.9,
                "stages.track.params.max_age_frames": 1,
            },
        },
        {
            "name": "ext",
            "kind": "import",
            "runs": {"clip-a": str(import_run_dir)},
            "comparison_class": import_comparison_class,
        },
    ]


def _run(tmp_path: Path, manifest_path: Path, import_run_dir: Path, **param_overrides) -> dict:
    params = {
        "dataset_manifest": str(manifest_path),
        "roles": ["tuning", "held_out"],
        "candidates": _candidates(import_run_dir),
        "device": "cpu",
    }
    params.update(param_overrides)
    config = ExperimentConfig(
        name=f"golden-{len(param_overrides)}-{param_overrides.get('tolerances', '')}",
        task="benchmark",
        params=params,
        output_dir=str(tmp_path / "exp"),
    )
    return build(config.task, config).run()


@pytest.fixture(scope="module")
def golden_fixture(tmp_path_factory):
    """Built once per module: dataset + import run dir are expensive-ish to
    render/import and are read-only inputs to every test below."""
    tmp_path = tmp_path_factory.mktemp("golden")
    manifest_path = _write_dataset(tmp_path)
    import_run_dir = _write_import_run(tmp_path, "ext", reference_only=True)
    return tmp_path, manifest_path, import_run_dir


@pytest.fixture(scope="module")
def probe_result(golden_fixture):
    """One real end-to-end run (no tolerances/compare) -- the golden rows
    and aggregates every other test in this module inspects or tampers."""
    tmp_path, manifest_path, import_run_dir = golden_fixture
    return _run(tmp_path / "probe", manifest_path, import_run_dir)


# ---------------------------------------------------------------------------
# Golden row/table shape
# ---------------------------------------------------------------------------


def test_golden_row_count_and_status(probe_result):
    # 2 pipeline candidates x 2 sequences + 1 import candidate x 1 sequence.
    assert len(probe_result["rows"]) == 5
    assert probe_result["summary"]["n_failed"] == 0
    assert all(row["status"] == "completed" for row in probe_result["rows"])


def test_golden_row_shape_uniform_across_native_and_import(probe_result):
    expected_keys = {
        "candidate",
        "comparison_class",
        "sequence",
        "role",
        "run_id",
        "config_name",
        "n_tracklets",
        "status",
        "metrics",
        "eval_path",
        "provenance_summary",
    }
    for row in probe_result["rows"]:
        assert set(row) == expected_keys


def test_golden_import_row_shape_and_metrics(probe_result):
    ext_row = next(r for r in probe_result["rows"] if r["candidate"] == "ext")
    assert ext_row["sequence"] == "clip-a"
    assert ext_row["comparison_class"] == "as_published"
    # A perfect-match MOT import against the same static GT box -> exact score.
    assert ext_row["metrics"]["idf1_tracklet"] == 1.0
    ext = ext_row["provenance_summary"]["external"]
    assert ext["system"] == "TDLP"
    assert ext["reference_only"] is True
    assert ext["license"] == {"code": "MIT", "weights": "unknown", "training_data": "unknown"}


def test_golden_import_row_eval_path_is_workdir_relative(probe_result):
    """eval_path for an import row follows the same workdir-relative
    convention as native rows -- never an absolute path into the
    externally-owned import dir."""
    ext_row = next(r for r in probe_result["rows"] if r["candidate"] == "ext")
    assert ext_row["eval_path"] == f"runs/{ext_row['run_id']}/eval.json"
    assert not Path(ext_row["eval_path"]).is_absolute()


def test_golden_import_dir_never_gets_eval_json(golden_fixture, probe_result):
    """The import dir is externally owned and must stay read-only: scoring
    an import candidate must never write eval.json (or anything else) into
    it, no matter how many benchmark runs (this module runs several against
    the same shared `golden_fixture` import dir) have scored it."""
    _, _, import_run_dir = golden_fixture
    assert not (import_run_dir / "eval.json").exists()
    assert {p.name for p in import_run_dir.iterdir()} == {
        "tracklets.json",
        "manifest.json",
        "external_provenance.json",
    }


def test_golden_tables_present_and_disjoint(probe_result):
    tables = probe_result["summary"]["tables"]
    assert set(tables) == {"matched_data", "as_published"}
    assert set(tables["matched_data"]) == {"base", "variant"}
    assert set(tables["as_published"]) == {"ext"}
    # Never merged.
    assert not set(tables["matched_data"]) & set(tables["as_published"])


def test_golden_import_row_confined_to_as_published(probe_result):
    tables = probe_result["summary"]["tables"]
    assert "ext" not in tables["matched_data"]
    assert tables["as_published"]["ext"]["n_sequences"] == 1


def test_golden_matched_candidates_have_per_role_breakdown(probe_result):
    """clip-a is role=tuning, clip-b is role=held_out -- both native
    candidates ran against both, so their aggregates get a by_role split."""
    tables = probe_result["summary"]["tables"]
    for name in ("base", "variant"):
        agg = tables["matched_data"][name]
        assert agg["n_sequences"] == 2
        assert set(agg["by_role"]) == {"tuning", "held_out"}


# ---------------------------------------------------------------------------
# Tolerance-band comparison, deterministic verdicts
# ---------------------------------------------------------------------------


def _pick_metrics(probe_result) -> tuple[str, float, str, float]:
    """From the probe run's real (deterministic, seeded) measured deltas
    between 'base' and 'variant', pick one metric with a nonzero delta (tol
    set to half that delta -> forces improved/regressed) and one other
    numeric metric shared by both aggregates (tol set far above its delta
    -> forces within_tolerance), without any hardcoded expected metric
    value -- the tolerances are derived from what this run actually
    measured, mirroring how a real gate's pre-registered tolerance is
    chosen against a real baseline."""
    tables = probe_result["summary"]["tables"]["matched_data"]
    base_metrics = tables["base"]["metrics"]
    variant_metrics = tables["variant"]["metrics"]
    shared = sorted(set(base_metrics) & set(variant_metrics))

    metric_a = None
    delta_a = 0.0
    for m in shared:
        d = variant_metrics[m]["mean"] - base_metrics[m]["mean"]
        if abs(d) > 1e-9:
            metric_a = m
            delta_a = d
            break
    assert metric_a is not None, (
        f"No metric differed between base and variant overrides -- fixture needs a "
        f"stronger override. Shared metrics: {shared}"
    )

    metric_b = next(m for m in shared if m != metric_a)
    delta_b = variant_metrics[metric_b]["mean"] - base_metrics[metric_b]["mean"]

    tol_a = abs(delta_a) / 2
    tol_b = abs(delta_b) * 10 + 1.0
    return metric_a, tol_a, metric_b, tol_b


@pytest.fixture(scope="module")
def tolerance_result(golden_fixture, probe_result):
    tmp_path, manifest_path, import_run_dir = golden_fixture
    metric_a, tol_a, metric_b, tol_b = _pick_metrics(probe_result)
    result = _run(
        tmp_path / "tol",
        manifest_path,
        import_run_dir,
        tolerances={metric_a: tol_a, metric_b: tol_b},
        compare={"baseline": "base"},
    )
    return result, metric_a, delta_sign(probe_result, metric_a), metric_b


def delta_sign(probe_result, metric: str) -> str:
    tables = probe_result["summary"]["tables"]["matched_data"]
    delta = tables["variant"]["metrics"][metric]["mean"] - tables["base"]["metrics"][metric]["mean"]
    lower_is_better = metric in LOWER_IS_BETTER
    signed = -delta if lower_is_better else delta
    return "improved" if signed > 0 else "regressed"


def test_golden_tolerance_verdict_improved_or_regressed_matches_measured_sign(tolerance_result):
    result, metric_a, expected_verdict, _metric_b = tolerance_result
    verdict = result["summary"]["comparison"]["verdicts"]["variant"][metric_a]
    assert verdict["verdict"] == expected_verdict
    assert verdict["delta"] != 0


def test_golden_tolerance_verdict_within_tolerance(tolerance_result):
    result, _metric_a, _expected, metric_b = tolerance_result
    verdict = result["summary"]["comparison"]["verdicts"]["variant"][metric_b]
    assert verdict["verdict"] == "within_tolerance"


def test_golden_tolerance_comparison_excludes_as_published_candidate(tolerance_result):
    result, *_ = tolerance_result
    assert "ext" not in result["summary"]["comparison"]["verdicts"]


def test_golden_tolerance_comparison_baseline_recorded(tolerance_result):
    result, *_ = tolerance_result
    assert result["summary"]["comparison"]["baseline"] == "base"


# ---------------------------------------------------------------------------
# Refusal paths
# ---------------------------------------------------------------------------


def test_refuses_import_candidate_missing_provenance_sidecar(golden_fixture):
    tmp_path, manifest_path, import_run_dir = golden_fixture
    bad_run_dir = tmp_path / "bad-import-no-sidecar"
    bad_run_dir.mkdir()
    (bad_run_dir / "tracklets.json").write_text("[]")
    (bad_run_dir / "manifest.json").write_text("{}")

    config = ExperimentConfig(
        name="golden-refuse-a",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning", "held_out"],
            "candidates": [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"clip-a": str(bad_run_dir)},
                    "comparison_class": "as_published",
                }
            ],
        },
        output_dir=str(tmp_path / "exp-refuse-a"),
    )
    with pytest.raises(RuntimeError, match="external_provenance.json"):
        build(config.task, config).run()


def test_refuses_reference_only_import_into_matched_data(golden_fixture):
    tmp_path, manifest_path, import_run_dir = golden_fixture
    config = ExperimentConfig(
        name="golden-refuse-b",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning", "held_out"],
            "candidates": [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"clip-a": str(import_run_dir)},
                    "comparison_class": "matched_data",
                }
            ],
        },
        output_dir=str(tmp_path / "exp-refuse-b"),
    )
    with pytest.raises(RuntimeError, match="reference_only"):
        build(config.task, config).run()


def test_refuses_provenance_inconsistency_within_candidate(probe_result):
    """(c): tamper one of 'base's two real rows' stage_impls, then call the
    pure gate function directly -- no pipeline re-run needed."""
    base_rows = [copy.deepcopy(r) for r in probe_result["rows"] if r["candidate"] == "base"]
    assert len(base_rows) == 2
    base_rows[1]["provenance_summary"]["stage_impls"] = dict(
        base_rows[1]["provenance_summary"]["stage_impls"], track="botsort"
    )
    with pytest.raises(RuntimeError) as exc_info:
        _check_provenance_consistency(base_rows)
    msg = str(exc_info.value)
    assert "base" in msg
    assert base_rows[0]["run_id"] in msg
    assert base_rows[1]["run_id"] in msg


def test_refuses_evaluation_set_hash_mismatch(probe_result):
    """(d): tamper one of two rows scoring the same sequence ('clip-a', from
    'base' and 'ext') to carry a different stamped evaluation_set_hash, then
    call the pure gate function directly."""
    clip_a_rows = [copy.deepcopy(r) for r in probe_result["rows"] if r["sequence"] == "clip-a"]
    assert len(clip_a_rows) >= 2
    tampered = clip_a_rows[1]
    original_hash = tampered["provenance_summary"]["evaluation_set_hash"]
    tampered["provenance_summary"]["evaluation_set_hash"] = "tampered-hash"
    with pytest.raises(RuntimeError) as exc_info:
        _check_evaluation_set_consistency(clip_a_rows)
    msg = str(exc_info.value)
    assert "clip-a" in msg
    assert clip_a_rows[0]["run_id"] in msg
    assert tampered["run_id"] in msg
    assert original_hash in msg
    assert "tampered-hash" in msg


def test_refuses_compare_baseline_as_published(probe_result):
    """(e), part 1: compare.baseline naming an as_published candidate."""
    from pitchlab_train.experiments.benchmark import Compare

    tables = probe_result["summary"]["tables"]
    with pytest.raises(RuntimeError, match="ext"):
        _compute_comparison(Compare(baseline="ext"), {"idf1_tracklet": 0.01}, tables)


def test_refuses_compare_baseline_unknown(probe_result):
    """(e), part 2: compare.baseline naming a candidate that doesn't exist
    in either table."""
    from pitchlab_train.experiments.benchmark import Compare

    tables = probe_result["summary"]["tables"]
    with pytest.raises(RuntimeError, match="nope"):
        _compute_comparison(Compare(baseline="nope"), {"idf1_tracklet": 0.01}, tables)


def test_refuses_import_candidate_unknown_sequence_name(golden_fixture):
    """Import candidate's `runs` must be a subset of the selected sequences
    -- an unknown sequence name refuses at run() time."""
    tmp_path, manifest_path, import_run_dir = golden_fixture
    config = ExperimentConfig(
        name="golden-refuse-unknown-seq",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning", "held_out"],
            "candidates": [
                {
                    "name": "ext",
                    "kind": "import",
                    "runs": {"clip-nonexistent": str(import_run_dir)},
                    "comparison_class": "as_published",
                }
            ],
        },
        output_dir=str(tmp_path / "exp-refuse-unknown-seq"),
    )
    with pytest.raises(RuntimeError, match="clip-nonexistent"):
        build(config.task, config).run()
