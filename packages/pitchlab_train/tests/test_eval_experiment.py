"""Runs the eval-pipelines experiment end-to-end on a synthetic clip."""

from pathlib import Path

from pitchlab_core.demo import render_demo_video
from pitchlab_train.config import ExperimentConfig
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
