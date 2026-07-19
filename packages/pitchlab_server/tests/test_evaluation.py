"""Unit tests for the pure eval-diff helper used by the run-diff endpoint,
and the evaluation-set provenance hook in `evaluate_run_against_gt`."""

import json
from pathlib import Path

import pytest
from pitchlab_server.evaluation import diff_switch_instances


def _inst(t, level="tracklet", gt_track_id=1, **kw):
    return {
        "level": level,
        "kind": "id_switch",
        "frame_idx": int(t * 10),
        "t": t,
        "gt_track_id": gt_track_id,
        "gt_label": kw.get("gt_label", "home_7"),
        "prev_id": kw.get("prev_id", 100),
        "new_id": kw.get("new_id", 101),
    }


def test_none_inputs_return_none():
    assert diff_switch_instances(None, {"instances": []}) is None
    assert diff_switch_instances({"instances": []}, None) is None
    assert diff_switch_instances(None, None) is None


def test_missing_instances_key_returns_none():
    assert diff_switch_instances({}, {"instances": []}) is None
    assert diff_switch_instances({"instances": []}, {}) is None


def test_identical_instance_sets_all_persisted():
    inst_a = _inst(10.0)
    inst_b = _inst(10.0)
    result = diff_switch_instances({"instances": [inst_a]}, {"instances": [inst_b]})
    assert result["fixed"] == []
    assert result["introduced"] == []
    assert result["persisted"] == [{"a": inst_a, "b": inst_b}]
    assert result["counts"] == {"fixed": 0, "introduced": 0, "persisted": 1}


def test_within_tolerance_is_persisted():
    inst_a = _inst(10.0)
    inst_b = _inst(10.6)
    result = diff_switch_instances(
        {"instances": [inst_a]}, {"instances": [inst_b]}, tol_s=1.0
    )
    assert result["persisted"] == [{"a": inst_a, "b": inst_b}]
    assert result["fixed"] == []
    assert result["introduced"] == []


def test_outside_tolerance_is_fixed_and_introduced():
    inst_a = _inst(10.0)
    inst_b = _inst(12.0)
    result = diff_switch_instances(
        {"instances": [inst_a]}, {"instances": [inst_b]}, tol_s=1.0
    )
    assert result["fixed"] == [inst_a]
    assert result["introduced"] == [inst_b]
    assert result["persisted"] == []
    assert result["counts"] == {"fixed": 1, "introduced": 1, "persisted": 0}


def test_same_t_different_level_not_matched():
    inst_a = _inst(10.0, level="tracklet")
    inst_b = _inst(10.0, level="entity")
    result = diff_switch_instances({"instances": [inst_a]}, {"instances": [inst_b]})
    assert result["fixed"] == [inst_a]
    assert result["introduced"] == [inst_b]
    assert result["persisted"] == []


def test_different_gt_track_id_not_matched():
    inst_a = _inst(10.0, gt_track_id=1)
    inst_b = _inst(10.0, gt_track_id=2)
    result = diff_switch_instances({"instances": [inst_a]}, {"instances": [inst_b]})
    assert result["fixed"] == [inst_a]
    assert result["introduced"] == [inst_b]
    assert result["persisted"] == []


def test_empty_instance_lists_return_empty_buckets():
    result = diff_switch_instances({"instances": []}, {"instances": []})
    assert result == {
        "fixed": [],
        "introduced": [],
        "persisted": [],
        "counts": {"fixed": 0, "introduced": 0, "persisted": 0},
    }


def test_greedy_matching_prefers_closest_pairs_first():
    # Group (tracklet, 1) has two A instances and two B instances. The globally
    # sorted greedy matcher should pair by closest distance first, even when the
    # enumeration order would pair differently. Naive enumeration-order greedy
    # would incorrectly match (10.0↔10.4, 10.5↔10.05) instead of the correct
    # (10.0↔10.05, 10.5↔10.4).
    a1 = _inst(10.0)
    a2 = _inst(10.5)
    b1 = _inst(10.4)
    b2 = _inst(10.05)
    result = diff_switch_instances(
        {"instances": [a1, a2]}, {"instances": [b1, b2]}, tol_s=1.0
    )
    assert result["counts"] == {"fixed": 0, "introduced": 0, "persisted": 2}
    # Verify exact pairings (sorted by closest distance first):
    # - (10.0, 10.05): distance 0.05 (closest)
    # - (10.5, 10.4): distance 0.1 (second closest)
    persisted = result["persisted"]
    assert len(persisted) == 2
    # Extract pairs for easier verification
    pairs = [(p["a"]["t"], p["b"]["t"]) for p in persisted]
    assert (10.0, 10.05) in pairs, f"Expected (10.0, 10.05) in {pairs}"
    assert (10.5, 10.4) in pairs, f"Expected (10.5, 10.4) in {pairs}"


def test_counts_consistent_with_list_lengths():
    a_insts = [_inst(1.0), _inst(5.0, gt_track_id=2)]
    b_insts = [_inst(1.05), _inst(20.0, gt_track_id=3)]
    result = diff_switch_instances({"instances": a_insts}, {"instances": b_insts})
    assert result["counts"]["fixed"] == len(result["fixed"])
    assert result["counts"]["introduced"] == len(result["introduced"])
    assert result["counts"]["persisted"] == len(result["persisted"])


# --- evaluate_run_against_gt: evaluation-set provenance hook (SPO-10) ------


def _write_run_dir(root: Path, video_meta: dict) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "test-run",
        "created_at": "2026-07-16T00:00:00+00:00",
        "video": video_meta,
        "config": {},
        "config_name": "test",
        "status": "completed",
        # Deliberately no "provenance" key at all -- mirrors a manifest
        # written before this field existed, exercising the pydantic default.
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "tracklets.json").write_text(json.dumps([]))
    (run_dir / "players.json").write_text(json.dumps([]))
    return run_dir


def test_evaluate_run_against_gt_writes_evaluation_set_hash_into_manifest(tmp_path):
    pytest.importorskip("motmetrics")
    from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
    from pitchlab_core.provenance import hash_evaluation_set
    from pitchlab_server.evaluation import evaluate_run_against_gt
    from pitchlab_server.models import Run, Video

    gt = GroundTruth(
        source="unit-test",
        sequence="SEQ-1",
        fps=25.0,
        width=100,
        height=100,
        seq_length=10,
        tracks=[
            GroundTruthTrack(
                track_id=1,
                role="player",
                team="left",
                frames=[
                    GroundTruthFrame(
                        frame_idx=f, box={"x1": 10, "y1": 10, "x2": 50, "y2": 130}
                    )
                    for f in range(10)
                ],
            )
        ],
    )
    gt_text = gt.model_dump_json()
    gt_path = tmp_path / "seq.gt.json"
    gt_path.write_text(gt_text)

    run_dir = _write_run_dir(
        tmp_path,
        {
            "path": "seq.mp4", "fps": 25.0, "frame_count": 10,
            "width": 100, "height": 100, "duration_s": 0.4, "sample_stride": 1,
        },
    )
    run = Run(id="test-run", video_id=1, config_name="test", config_yaml="", run_dir=str(run_dir))
    video = Video(id=1, filename="seq.mp4", path="seq.mp4", gt_path=str(gt_path))

    result = evaluate_run_against_gt(run, video)
    assert result is not None  # scoring still happened (eval.json written)
    assert (run_dir / "eval.json").exists()

    on_disk = json.loads((run_dir / "manifest.json").read_text())
    prov = on_disk["provenance"]
    assert prov["evaluation_set_hash"] == hash_evaluation_set(gt_text)
    assert prov["evaluation_set_hash"] != "unknown"
    assert prov["evaluation_set_source"] == str(gt_path)


def test_evaluate_run_against_gt_scores_spotting_from_event_gt(tmp_path):
    # Event GT + spotting.json -> the spotting branch scores avg-mAP, writes
    # eval.json, and records the same provenance hash. No motmetrics needed,
    # and NO tracklets.json required (a pure spotting run has none).
    from pitchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
    from pitchlab_core.provenance import hash_evaluation_set
    from pitchlab_server.evaluation import evaluate_run_against_gt, merged_metrics
    from pitchlab_server.models import Run, Video

    gt = EventGroundTruth(
        source="unit-test",
        fps=25.0,
        events=[GroundTruthEvent(class_="PASS", frame_idx=25, t=1.0)],
    )
    gt_text = gt.model_dump_json()
    gt_path = tmp_path / "seq.events.json"
    gt_path.write_text(gt_text)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "test-run",
        "created_at": "2026-07-16T00:00:00+00:00",
        "video": {
            "path": "seq.mp4", "fps": 25.0, "frame_count": 25,
            "width": 100, "height": 100, "duration_s": 1.0, "sample_stride": 1,
        },
        "config": {}, "config_name": "test", "status": "completed",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    # A perfect single-class prediction; note: no tracklets.json on disk.
    (run_dir / "spotting.json").write_text(
        json.dumps([{"class": "PASS", "frame_idx": 25, "t": 1.0, "confidence": 0.9}])
    )

    run = Run(id="test-run", video_id=1, config_name="test", config_yaml="", run_dir=str(run_dir))
    video = Video(id=1, filename="seq.mp4", path="seq.mp4", gt_path=str(gt_path))

    result = evaluate_run_against_gt(run, video)
    assert result is not None
    assert result["kind"] == "action_spotting"
    assert result["avg_map"] == 1.0
    assert (run_dir / "eval.json").exists()

    prov = json.loads((run_dir / "manifest.json").read_text())["provenance"]
    assert prov["evaluation_set_hash"] == hash_evaluation_set(gt_text)
    assert prov["evaluation_set_source"] == str(gt_path)

    metrics = merged_metrics(run, result)
    assert metrics["spotting_map_at_1"] == 1.0


def test_evaluate_run_against_gt_event_gt_without_spotting_returns_none(tmp_path):
    from pitchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
    from pitchlab_server.evaluation import evaluate_run_against_gt
    from pitchlab_server.models import Run, Video

    gt = EventGroundTruth(
        source="unit-test",
        fps=25.0,
        events=[GroundTruthEvent(class_="PASS", frame_idx=25, t=1.0)],
    )
    gt_path = tmp_path / "seq.events.json"
    gt_path.write_text(gt.model_dump_json())

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "r"}))
    # No spotting.json -> nothing to score.

    run = Run(id="test-run", video_id=1, config_name="test", config_yaml="", run_dir=str(run_dir))
    video = Video(id=1, filename="seq.mp4", path="seq.mp4", gt_path=str(gt_path))

    assert evaluate_run_against_gt(run, video) is None


def test_evaluate_run_against_gt_no_gt_path_leaves_result_none(tmp_path):
    from pitchlab_server.evaluation import evaluate_run_against_gt
    from pitchlab_server.models import Run, Video

    run_dir = _write_run_dir(
        tmp_path,
        {
            "path": "seq.mp4", "fps": 25.0, "frame_count": 10,
            "width": 100, "height": 100, "duration_s": 0.4, "sample_stride": 1,
        },
    )
    run = Run(id="test-run", video_id=1, config_name="test", config_yaml="", run_dir=str(run_dir))
    video = Video(id=1, filename="seq.mp4", path="seq.mp4", gt_path=None)

    before = (run_dir / "manifest.json").read_text()
    assert evaluate_run_against_gt(run, video) is None
    # No GT -> the hook must not touch the manifest at all.
    assert (run_dir / "manifest.json").read_text() == before
