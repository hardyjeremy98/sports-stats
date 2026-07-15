"""Runs the eval-pipelines experiment end-to-end on a synthetic clip."""

from pathlib import Path

import pytest
from pitchlab_core.demo import render_demo_video
from pitchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from pitchlab_core.schemas.geometry import Box
from pitchlab_train.config import ExperimentConfig
from pitchlab_train.experiments.eval_pipelines import _summarize
from pitchlab_train.registry import available, build

REPO = Path(__file__).parents[3]


def test_tasks_registered():
    assert {"detector-rfdetr", "eval-pipelines"} <= set(available())


def test_eval_pipelines(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    render_demo_video(clips / "a.mp4", duration_s=5, fps=20, width=960, height=540)

    config = ExperimentConfig(
        name="test-eval",
        task="eval-pipelines",
        params={
            "config_a": str(REPO / "configs" / "pipeline.stub.yaml"),
            "config_b": str(REPO / "configs" / "pipeline.stub.yaml"),
            "clips_dir": str(clips),
            "max_clips": 1,
        },
        output_dir=str(tmp_path / "exp"),
    )
    result = build(config.task, config).run()
    assert result["summary"]["completed_pairs"] == 1
    assert result["clips"][0]["a"]["metrics"]["n_tracklets"] > 5
    # result.json written
    assert list((tmp_path / "exp").glob("*/result.json"))
    # No sibling GT for this clip -> no GT metrics anywhere.
    assert "gt_metrics" not in result["clips"][0]["a"]
    assert "gt_metrics" not in result["summary"]


def test_eval_pipelines_scores_clips_with_sibling_gt(tmp_path):
    pytest.importorskip("motmetrics")
    clips = tmp_path / "clips"
    clips.mkdir()
    render_demo_video(clips / "a.mp4", duration_s=2, fps=20, width=960, height=540)

    # Not positionally accurate to the synthetic footage — this only exercises
    # the sibling-GT discovery -> evaluate_run -> headline_metrics wiring, not
    # metric correctness (that's covered by pitchlab_core's own GT eval tests).
    gt = GroundTruth(
        source="test",
        sequence="a",
        fps=20.0,
        width=960,
        height=540,
        seq_length=40,
        tracks=[
            GroundTruthTrack(
                track_id=1,
                role="player",
                frames=[
                    GroundTruthFrame(frame_idx=f, box=Box(x1=100, y1=100, x2=140, y2=220))
                    for f in range(40)
                ],
            )
        ],
    )
    (clips / "a.gt.json").write_text(gt.model_dump_json())

    config = ExperimentConfig(
        name="test-eval-gt",
        task="eval-pipelines",
        params={
            "config_a": str(REPO / "configs" / "pipeline.stub.yaml"),
            "config_b": str(REPO / "configs" / "pipeline.stub.yaml"),
            "clips_dir": str(clips),
            "max_clips": 1,
        },
        output_dir=str(tmp_path / "exp"),
    )
    result = build(config.task, config).run()
    row = result["clips"][0]
    assert "idf1_entity" in row["a"]["gt_metrics"]
    assert "idf1_entity" in row["b"]["gt_metrics"]
    assert "idf1_entity" in result["summary"]["gt_metrics"]["a"]


def test_summarize_aggregates_gt_metrics_mean_and_median():
    per_clip = [
        {
            "clip": "x.mp4",
            "a": {
                "status": "completed",
                "metrics": {"n_tracklets": 4},
                "gt_metrics": {"idf1_entity": 0.8, "merge_precision": 1.0},
            },
            "b": {
                "status": "completed",
                "metrics": {"n_tracklets": 4},
                "gt_metrics": {"idf1_entity": 0.6, "merge_precision": 0.5},
            },
        },
        {
            "clip": "y.mp4",
            "a": {
                "status": "completed",
                "metrics": {"n_tracklets": 6},
                # merge_precision is None here (no merged pairs at all) -> must
                # not pollute the mean/median as if it were 0.
                "gt_metrics": {"idf1_entity": 0.9, "merge_precision": None},
            },
            "b": {
                "status": "completed",
                "metrics": {"n_tracklets": 6},
                "gt_metrics": {"idf1_entity": 0.7, "merge_precision": 1.0},
            },
        },
    ]
    summary = _summarize(per_clip)
    gt = summary["gt_metrics"]
    assert gt["a"]["idf1_entity"]["mean"] == pytest.approx(0.85)
    assert gt["a"]["idf1_entity"]["median"] == pytest.approx(0.85)
    assert gt["b"]["idf1_entity"]["mean"] == pytest.approx(0.65)
    # Only one clip contributed a numeric merge_precision for "a".
    assert gt["a"]["merge_precision"]["mean"] == pytest.approx(1.0)
    assert gt["b"]["merge_precision"]["mean"] == pytest.approx(0.75)
    # Artifact-count aggregation is unchanged.
    assert summary["mean"]["a"]["n_tracklets"] == pytest.approx(5.0)


def test_summarize_without_gt_metrics_matches_prior_behavior():
    per_clip = [
        {
            "clip": "x.mp4",
            "a": {"status": "completed", "metrics": {"n_tracklets": 4}},
            "b": {"status": "completed", "metrics": {"n_tracklets": 4}},
        },
    ]
    summary = _summarize(per_clip)
    assert "gt_metrics" not in summary
