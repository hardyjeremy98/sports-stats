"""Offline unit + end-to-end tests for the benchmark runner (SPO-17 part 1):
manifest loading, candidate/sweep expansion, and row-building as pure
functions, plus one full pipeline run confirming the wiring (provenance
stamping, scoring, failure surfacing) actually works end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pitchlab_core.config import PipelineConfig
from pitchlab_core.demo import render_demo_video
from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from pitchlab_core.provenance import hash_evaluation_set
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.run import RunManifest, StageKind, StageStatus
from pitchlab_train.config import ExperimentConfig
from pitchlab_train.experiments.benchmark import (
    ManifestSequence,
    Params,
    PipelineCandidate,
    SweepSpec,
    _expand_candidates,
    _load_manifest_sequences,
    _load_pipeline_config,
    _row_from_run,
)
from pitchlab_train.registry import available, build
from pydantic import ValidationError

REPO = Path(__file__).parents[3]
STUB_CONFIG = str(REPO / "configs" / "pipeline.stub.yaml")
ORACLE_CONFIG = str(REPO / "configs" / "pipeline.oracle-eval.yaml")


def test_task_registered():
    assert "benchmark" in available()


# ---------------------------------------------------------------------------
# _load_manifest_sequences
# ---------------------------------------------------------------------------


def _write_manifest_tree(tmp_path: Path, sequences: list[dict], write_files: bool = True) -> Path:
    """A tmp `<root>/configs/datasets/tier.json` manifest plus (optionally)
    the video/gt files each entry's relative path points at, so path
    resolution against "the directory containing configs/" is exercised the
    same way it is against the real repo's configs/datasets/*.json."""
    configs_dir = tmp_path / "configs" / "datasets"
    configs_dir.mkdir(parents=True)
    manifest_path = configs_dir / "tier.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "test-tier",
                "tier": "tier",
                "source_split": "test",
                "created": "2026-07-17",
                "sequences": sequences,
                "notes": [],
            }
        )
    )
    if write_files:
        for seq in sequences:
            video = tmp_path / seq["video"]
            gt = tmp_path / seq["gt"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"fake-video")
            gt.write_text("{}")
    return manifest_path


def test_load_manifest_sequences_filters_by_role(tmp_path):
    manifest_path = _write_manifest_tree(
        tmp_path,
        [
            {"name": "A", "video": "data/videos/a.mp4", "gt": "data/videos/a.gt.json", "role": "tuning"},
            {"name": "B", "video": "data/videos/b.mp4", "gt": "data/videos/b.gt.json", "role": "held_out"},
        ],
    )
    seqs = _load_manifest_sequences(manifest_path, ["tuning"])
    assert [s.name for s in seqs] == ["A"]
    assert seqs[0].role == "tuning"
    # Resolved relative to the directory containing configs/, not the CWD.
    assert seqs[0].video == str(tmp_path / "data/videos/a.mp4")
    assert seqs[0].gt == str(tmp_path / "data/videos/a.gt.json")


def test_load_manifest_sequences_missing_manifest_raises():
    with pytest.raises(FileNotFoundError, match="nope.json"):
        _load_manifest_sequences("configs/datasets/nope.json", ["tuning"])


def test_load_manifest_sequences_missing_video_raises(tmp_path):
    manifest_path = _write_manifest_tree(
        tmp_path,
        [{"name": "A", "video": "data/videos/a.mp4", "gt": "data/videos/a.gt.json", "role": "tuning"}],
        write_files=False,
    )
    (tmp_path / "data" / "videos").mkdir(parents=True)
    (tmp_path / "data" / "videos" / "a.gt.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="a.mp4"):
        _load_manifest_sequences(manifest_path, ["tuning"])


def test_load_manifest_sequences_missing_gt_raises(tmp_path):
    manifest_path = _write_manifest_tree(
        tmp_path,
        [{"name": "A", "video": "data/videos/a.mp4", "gt": "data/videos/a.gt.json", "role": "tuning"}],
        write_files=False,
    )
    (tmp_path / "data" / "videos").mkdir(parents=True)
    (tmp_path / "data" / "videos" / "a.mp4").write_bytes(b"fake-video")
    with pytest.raises(FileNotFoundError, match="a.gt.json"):
        _load_manifest_sequences(manifest_path, ["tuning"])


def test_load_manifest_sequences_empty_role_selection_raises(tmp_path):
    manifest_path = _write_manifest_tree(
        tmp_path,
        [{"name": "A", "video": "data/videos/a.mp4", "gt": "data/videos/a.gt.json", "role": "tuning"}],
    )
    with pytest.raises(RuntimeError, match="held_out"):
        _load_manifest_sequences(manifest_path, ["held_out"])


# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------


def test_params_rejects_unknown_role():
    with pytest.raises(ValidationError, match="staging"):
        Params(dataset_manifest="x.json", roles=["staging"], candidates=[{"name": "a", "config": STUB_CONFIG}])


def test_params_rejects_empty_candidates():
    with pytest.raises(ValidationError):
        Params(dataset_manifest="x.json", candidates=[])


def test_params_records_tolerances_verbatim():
    p = Params(
        dataset_manifest="x.json",
        candidates=[{"name": "a", "config": STUB_CONFIG}],
        tolerances={"idf1_entity": 0.02},
    )
    assert p.tolerances == {"idf1_entity": 0.02}


# ---------------------------------------------------------------------------
# _load_pipeline_config (override application)
# ---------------------------------------------------------------------------


def test_load_pipeline_config_applies_override():
    candidate = PipelineCandidate(
        name="a", config=STUB_CONFIG, overrides={"stages.track.params.max_age_frames": 99}
    )
    cfg = _load_pipeline_config(candidate)
    assert cfg.stages[StageKind.TRACK].params["max_age_frames"] == 99
    # Base config on disk is untouched.
    assert PipelineConfig.from_yaml(STUB_CONFIG).stages[StageKind.TRACK].params["max_age_frames"] != 99


def test_load_pipeline_config_unknown_stage_raises():
    candidate = PipelineCandidate(
        name="a", config=STUB_CONFIG, overrides={"stages.frobnicate.params.x": 1}
    )
    with pytest.raises(ValueError, match="frobnicate"):
        _load_pipeline_config(candidate)


def test_load_pipeline_config_malformed_path_raises():
    candidate = PipelineCandidate(name="a", config=STUB_CONFIG, overrides={"onlyonepart": 1})
    with pytest.raises(ValueError, match="onlyonepart"):
        _load_pipeline_config(candidate)


def test_load_pipeline_config_whole_stage_replacement_raises_not_silent_noop():
    """A 2-segment path bottoming out directly on `config.stages` (e.g.
    "stages.track") must refuse loudly, not silently insert a plain-string
    key alongside the StageKind-keyed entries (which PipelineRunner would
    never look up -- a silent no-op override)."""
    candidate = PipelineCandidate(
        name="a", config=STUB_CONFIG, overrides={"stages.track": {"impl": "iou", "params": {}}}
    )
    with pytest.raises(ValueError, match="stages.track"):
        _load_pipeline_config(candidate)


def test_load_pipeline_config_missing_config_file_raises():
    candidate = PipelineCandidate(name="a", config=str(REPO / "configs" / "pipeline.nope.yaml"))
    with pytest.raises(FileNotFoundError, match="pipeline.nope.yaml"):
        _load_pipeline_config(candidate)


# ---------------------------------------------------------------------------
# _expand_candidates
# ---------------------------------------------------------------------------


def test_expand_candidates_base_only():
    expanded = _expand_candidates([{"name": "base", "config": STUB_CONFIG}], [])
    assert [c.name for c in expanded] == ["base"]
    assert expanded[0].comparison_class == "matched_data"


def test_expand_candidates_config_missing_refuses_at_expansion():
    with pytest.raises(FileNotFoundError, match="pipeline.nope.yaml"):
        _expand_candidates(
            [{"name": "base", "config": str(REPO / "configs" / "pipeline.nope.yaml")}], []
        )


def test_expand_candidates_duplicate_base_names_raise():
    with pytest.raises(RuntimeError, match="dup"):
        _expand_candidates(
            [
                {"name": "dup", "config": STUB_CONFIG},
                {"name": "dup", "config": STUB_CONFIG},
            ],
            [],
        )


def test_expand_candidates_sweep_retains_base_and_derives_names():
    raw = [{"name": "base", "config": STUB_CONFIG, "overrides": {"stages.track.params.min_length": 5}}]
    sweeps = [SweepSpec(candidate="base", param="stages.track.params.max_age_frames", values=[10, 20])]
    expanded = _expand_candidates(raw, sweeps)
    names = [c.name for c in expanded]
    assert names == ["base", "base@max_age_frames=10", "base@max_age_frames=20"]
    derived = expanded[1]
    # Sweep override applied on top of the base candidate's own overrides.
    assert derived.overrides == {
        "stages.track.params.min_length": 5,
        "stages.track.params.max_age_frames": 10,
    }


def test_expand_candidates_sweep_unknown_base_raises():
    sweeps = [SweepSpec(candidate="missing", param="stages.track.params.x", values=[1])]
    with pytest.raises(RuntimeError, match="missing"):
        _expand_candidates([{"name": "base", "config": STUB_CONFIG}], sweeps)


def test_expand_candidates_sweep_duplicate_name_raises():
    raw = [
        {"name": "base", "config": STUB_CONFIG},
        {"name": "base@max_age_frames=10", "config": STUB_CONFIG},
    ]
    sweeps = [SweepSpec(candidate="base", param="stages.track.params.max_age_frames", values=[10])]
    with pytest.raises(RuntimeError, match="base@max_age_frames=10"):
        _expand_candidates(raw, sweeps)


def test_expand_candidates_unsupported_kind_raises():
    with pytest.raises(ValueError, match="widget"):
        _expand_candidates([{"name": "base", "config": STUB_CONFIG, "kind": "widget"}], [])


# ---------------------------------------------------------------------------
# _row_from_run
# ---------------------------------------------------------------------------


def _fake_manifest(status: StageStatus, **kw) -> RunManifest:
    from pitchlab_core.schemas.run import VideoMeta

    defaults = dict(
        run_id="r1",
        created_at="2026-07-17T00:00:00Z",
        video=VideoMeta(path="x.mp4", fps=25.0, frame_count=10, width=64, height=64, duration_s=0.4),
        config={},
        config_name="cfg",
        status=status,
    )
    defaults.update(kw)
    return RunManifest(**defaults)


def _fake_eval_result() -> dict:
    return {
        "levels": {
            "tracklet": {"idf1": 0.5, "mota": 0.4, "num_switches": 2},
            "entity": {"idf1": 0.6, "mota": 0.5, "num_switches": 1},
        },
        "hota": {"tracklet": {"hota": 0.5}, "entity": {"hota": 0.6}},
        "association": {"idf1_gain": 0.1, "merge_precision": None},
        "purity": {"tracklet": {"post_filter": {"mean_purity": None, "total_mixed_seconds": 0.0}}},
    }


def test_row_from_run_completed_row_shape():
    candidate = PipelineCandidate(name="cand-a", config=STUB_CONFIG)
    seq = ManifestSequence(name="seq-1", role="tuning", video="v.mp4", gt="v.gt.json")
    manifest = _fake_manifest(StageStatus.COMPLETED, metrics={"n_tracklets": 4})
    manifest.provenance.git_revision = "abc123"
    manifest.provenance.evaluation_set_hash = "deadbeef"

    row = _row_from_run(candidate, seq, "cand-a-seq-1", manifest, _fake_eval_result(), "runs/cand-a-seq-1/eval.json")

    assert row["candidate"] == "cand-a"
    assert row["comparison_class"] == "matched_data"
    assert row["sequence"] == "seq-1"
    assert row["role"] == "tuning"
    assert row["run_id"] == "cand-a-seq-1"
    assert row["status"] == "completed"
    assert row["config_name"] == "cfg"
    assert row["n_tracklets"] == 4
    assert row["eval_path"] == "runs/cand-a-seq-1/eval.json"
    assert row["metrics"]["idf1_entity"] == 0.6
    assert row["provenance_summary"]["git_revision"] == "abc123"
    assert row["provenance_summary"]["evaluation_set_hash"] == "deadbeef"
    assert "error" not in row


def test_row_from_run_failed_row_has_no_metrics():
    candidate = PipelineCandidate(name="cand-a", config=STUB_CONFIG)
    seq = ManifestSequence(name="seq-1", role="tuning", video="v.mp4", gt="v.gt.json")
    manifest = _fake_manifest(StageStatus.FAILED, error="boom" * 200)

    row = _row_from_run(candidate, seq, "cand-a-seq-1", manifest, None, None)

    assert row["status"] == "failed"
    assert row["error"] == ("boom" * 200)[:500]
    assert "metrics" not in row
    assert "eval_path" not in row
    assert "provenance_summary" not in row


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def _write_gt_for_clip(clip_path: Path, seq_length: int, fps: float) -> None:
    gt = GroundTruth(
        source="test",
        sequence=clip_path.stem,
        fps=fps,
        width=960,
        height=540,
        seq_length=seq_length,
        tracks=[
            GroundTruthTrack(
                track_id=1,
                role="player",
                frames=[
                    GroundTruthFrame(frame_idx=f, box=Box(x1=100, y1=100, x2=140, y2=220))
                    for f in range(seq_length)
                ],
            )
        ],
    )
    (clip_path.parent / f"{clip_path.stem}.gt.json").write_text(gt.model_dump_json())


def test_benchmark_end_to_end(tmp_path):
    pytest.importorskip("motmetrics")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    for name in ("clip-a", "clip-b"):
        render_demo_video(videos_dir / f"{name}.mp4", duration_s=2, fps=20, width=960, height=540)
        _write_gt_for_clip(videos_dir / f"{name}.mp4", seq_length=40, fps=20)

    manifest_path = _write_manifest_tree(
        tmp_path,
        [
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
                "role": "tuning",
            },
        ],
        write_files=False,
    )

    config = ExperimentConfig(
        name="test-benchmark",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning"],
            "candidates": [{"name": "stub-base", "config": STUB_CONFIG}],
            "sweeps": [
                {
                    "candidate": "stub-base",
                    "param": "stages.track.params.max_age_frames",
                    "values": [30],
                }
            ],
            "device": "cpu",
        },
        output_dir=str(tmp_path / "exp"),
    )
    result = build(config.task, config).run()

    assert result["dataset_manifest_hash"]
    assert len(result["candidates"]) == 2
    assert len(result["rows"]) == 4  # 2 candidates x 2 sequences

    expected_hash_by_seq = {
        name: hash_evaluation_set((videos_dir / f"{name}.gt.json").read_text())
        for name in ("clip-a", "clip-b")
    }

    for row in result["rows"]:
        assert row["status"] == "completed"
        assert "idf1_tracklet" in row["metrics"]
        assert row["provenance_summary"]["git_revision"]
        assert row["provenance_summary"]["evaluation_set_hash"] == expected_hash_by_seq[row["sequence"]]

    result_dir = next((tmp_path / "exp").glob("*/result.json")).parent
    run_dirs = list((result_dir / "runs").iterdir())
    assert run_dirs
    run_dir = next(d for d in run_dirs if d.name.endswith("-clip-a"))
    stamped = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    assert stamped.provenance.evaluation_set_hash == expected_hash_by_seq["clip-a"]
    assert (run_dir / "eval.json").exists()

    assert result["summary"]["n_failed"] == 0
    # Aggregation, gates, and tolerance comparisons land in Task 9 --
    # summary.tables is always present (even with a single matched_data
    # candidate) and summary.comparison is None when Params.compare is unset.
    assert set(result["summary"]["tables"]) == {"matched_data", "as_published"}
    assert set(result["summary"]["tables"]["matched_data"]) == {
        "stub-base",
        "stub-base@max_age_frames=30",
    }
    assert result["summary"]["tables"]["matched_data"]["stub-base"]["n_sequences"] == 2
    assert result["summary"]["tables"]["as_published"] == {}
    assert result["summary"]["comparison"] is None


def test_benchmark_max_sequences_caps_after_role_filter_in_manifest_order(tmp_path):
    """max_sequences caps AFTER role filtering, in manifest order -- clip-a
    is listed first in the manifest, so a cap of 1 must keep clip-a and drop
    clip-b, and the cap must be recorded in the summary."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    for name in ("clip-a", "clip-b"):
        render_demo_video(videos_dir / f"{name}.mp4", duration_s=1, fps=20, width=320, height=240)
        _write_gt_for_clip(videos_dir / f"{name}.mp4", seq_length=20, fps=20)

    manifest_path = _write_manifest_tree(
        tmp_path,
        [
            {"name": "clip-a", "video": "videos/clip-a.mp4", "gt": "videos/clip-a.gt.json", "role": "tuning"},
            {"name": "clip-b", "video": "videos/clip-b.mp4", "gt": "videos/clip-b.gt.json", "role": "tuning"},
        ],
        write_files=False,
    )

    config = ExperimentConfig(
        name="test-benchmark-cap",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning"],
            "max_sequences": 1,
            "candidates": [{"name": "stub-base", "config": STUB_CONFIG}],
        },
        output_dir=str(tmp_path / "exp"),
    )
    result = build(config.task, config).run()

    assert len(result["rows"]) == 1
    assert result["rows"][0]["sequence"] == "clip-a"
    assert result["summary"]["sequences"] == [{"name": "clip-a", "role": "tuning"}]
    assert result["summary"]["max_sequences"] == 1
    assert result["summary"]["max_sequences_applied"] is True


def test_benchmark_surfaces_runtime_failure_without_aborting_experiment(tmp_path):
    pytest.importorskip("motmetrics")
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    render_demo_video(videos_dir / "clip-a.mp4", duration_s=2, fps=20, width=960, height=540)
    _write_gt_for_clip(videos_dir / "clip-a.mp4", seq_length=40, fps=20)

    manifest_path = _write_manifest_tree(
        tmp_path,
        [{"name": "clip-a", "video": "videos/clip-a.mp4", "gt": "videos/clip-a.gt.json", "role": "tuning"}],
        write_files=False,
    )

    config = ExperimentConfig(
        name="test-benchmark-fail",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning"],
            "candidates": [
                {"name": "good", "config": STUB_CONFIG},
                {
                    "name": "bad-gt",
                    "config": ORACLE_CONFIG,
                    "overrides": {"stages.detect.params.gt_path": str(tmp_path / "nope.gt.json")},
                },
            ],
        },
        output_dir=str(tmp_path / "exp"),
    )
    result = build(config.task, config).run()

    by_candidate = {row["candidate"]: row for row in result["rows"]}
    assert by_candidate["good"]["status"] == "completed"
    assert by_candidate["bad-gt"]["status"] == "failed"
    assert "nope.gt.json" in by_candidate["bad-gt"]["error"]
    assert "metrics" not in by_candidate["bad-gt"]

    assert result["summary"]["n_failed"] == 1
    failed_run_ids = [f["run_id"] for f in result["summary"]["failed_rows"]]
    assert failed_run_ids == ["bad-gt-clip-a"]


def test_benchmark_config_path_missing_refuses_at_expansion_not_as_failed_row(tmp_path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    render_demo_video(videos_dir / "clip-a.mp4", duration_s=1, fps=20, width=320, height=240)
    _write_gt_for_clip(videos_dir / "clip-a.mp4", seq_length=20, fps=20)
    manifest_path = _write_manifest_tree(
        tmp_path,
        [{"name": "clip-a", "video": "videos/clip-a.mp4", "gt": "videos/clip-a.gt.json", "role": "tuning"}],
        write_files=False,
    )

    config = ExperimentConfig(
        name="test-benchmark-badcfg",
        task="benchmark",
        params={
            "dataset_manifest": str(manifest_path),
            "roles": ["tuning"],
            "candidates": [{"name": "ghost", "config": str(tmp_path / "no-such-config.yaml")}],
        },
        output_dir=str(tmp_path / "exp"),
    )
    with pytest.raises(FileNotFoundError, match="no-such-config.yaml"):
        build(config.task, config).run()
