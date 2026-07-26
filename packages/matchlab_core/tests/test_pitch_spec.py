"""Behavioral tests for the pitch spec seam (SPO-61): a physically-correct
FIFA 105x68 m PitchSpec selectable per pipeline config, alongside the
existing non-physical roboflow template, plumbed through PipelineConfig ->
PipelineRunner -> StageContext.pitch and recorded in the run manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from matchlab_core.config import PipelineConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.pitch import FIFA_PITCH, SOCCER_PITCH, get_pitch
from matchlab_core.runner import PipelineRunner

CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.stub.yaml"


# --- get_pitch("fifa"): physically-correct FIFA 105x68 m geometry -----------


def test_fifa_pitch_dimensions():
    fifa = get_pitch("fifa")
    assert fifa.length == 10500
    assert fifa.width == 6800
    assert fifa.penalty_box_length == 1650
    assert fifa.penalty_box_width == 4032
    assert fifa.goal_box_length == 550
    assert fifa.goal_box_width == 1832
    assert fifa.centre_circle_radius == 915
    assert fifa.penalty_spot_distance == 1100
    assert len(fifa.vertices) == 32


def test_fifa_pitch_returns_same_object_as_constant():
    assert get_pitch("fifa") is FIFA_PITCH


def test_fifa_pitch_landmark_vertices():
    fifa = get_pitch("fifa")
    ln, w = 10500, 6800
    pbl, pbw = 1650, 4032
    gbl, gbw = 550, 1832
    ccr = 915
    psd = 1100

    # 1-indexed vertices per the docstring in _pitch_vertices.
    assert fifa.vertices[0] == (0, 0)  # 1 corner top-left
    assert fifa.vertices[5] == (0, w)  # 6 corner bottom-left
    assert fifa.vertices[8] == (psd, w / 2)  # 9 penalty spot left
    assert fifa.vertices[13] == (ln / 2, 0)  # 14 halfway top
    assert fifa.vertices[16] == (ln / 2, w)  # 17 halfway bottom
    assert fifa.vertices[24] == (ln, 0)  # 25 corner top-right
    assert fifa.vertices[29] == (ln, w)  # 30 corner bottom-right

    # Right-half vertices mirror their left-half counterparts across x=ln/2.
    left_penalty_corner = fifa.vertices[9]  # 10: penalty box top-left corner
    right_penalty_corner = fifa.vertices[17]  # 18: penalty box top-right corner
    assert right_penalty_corner == (ln - left_penalty_corner[0], left_penalty_corner[1])

    left_goal_corner = fifa.vertices[6]  # 7: goal box top-left corner
    right_goal_corner = fifa.vertices[22]  # 23: goal box top-right corner
    assert right_goal_corner == (ln - left_goal_corner[0], left_goal_corner[1])

    left_spot = fifa.vertices[8]  # 9: penalty spot left
    right_spot = fifa.vertices[21]  # 22: penalty spot right
    assert right_spot == (ln - left_spot[0], left_spot[1])

    left_circle = fifa.vertices[30]  # 31: centre circle left
    right_circle = fifa.vertices[31]  # 32: centre circle right
    assert left_circle == (ln / 2 - ccr, w / 2)
    assert right_circle == (ln / 2 + ccr, w / 2)

    # penalty-box and goal-box widths derive from the FIFA numbers given above.
    assert fifa.vertices[1] == (0, (w - pbw) / 2)  # 2 penalty box top-left y
    assert fifa.vertices[2] == (0, (w - gbw) / 2)  # 3 goal box top-left y
    assert fifa.vertices[6] == (gbl, (w - gbw) / 2)  # 7 goal box top-left corner
    assert fifa.vertices[9] == (pbl, (w - pbw) / 2)  # 10 penalty box top-left corner


# --- get_pitch("roboflow"): unchanged SOCCER_PITCH regression pin ----------


def test_roboflow_pitch_is_soccer_pitch():
    assert get_pitch("roboflow") is SOCCER_PITCH


def test_roboflow_pitch_vertices_unchanged():
    roboflow = get_pitch("roboflow")
    assert roboflow.length == 12000
    assert roboflow.width == 7000
    assert len(roboflow.vertices) == 32
    assert roboflow.vertices[0] == (0, 0)  # vertex 1
    assert roboflow.vertices[8] == (1100, 3500)  # vertex 9
    assert roboflow.vertices[31] == (6915, 3500)  # vertex 32


# --- get_pitch("nope"): unknown name -----------------------------------------


def test_get_pitch_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="nope"):
        get_pitch("nope")


def test_get_pitch_unknown_name_error_names_valid_options():
    with pytest.raises(ValueError, match="fifa"):
        get_pitch("nope")
    with pytest.raises(ValueError, match="roboflow"):
        get_pitch("nope")


# --- PipelineConfig.pitch -----------------------------------------------


def test_pipeline_config_defaults_to_roboflow_pitch():
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    assert config.pitch == "roboflow"


def test_pipeline_config_parses_fifa_pitch():
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    config.pitch = "fifa"
    reparsed = PipelineConfig.model_validate(config.model_dump(mode="json"))
    assert reparsed.pitch == "fifa"


# --- Runner/StageContext wiring + manifest recording ------------------------


@pytest.fixture(scope="module")
def fifa_run(tmp_path_factory):
    """The cheapest way to exercise PipelineRunner's constructor (which builds
    both ctx and the initial manifest) without running any stages: a tiny
    synthetic clip satisfies `probe()`, and we only inspect state set up in
    __init__."""
    tmp = tmp_path_factory.mktemp("fifa-wiring")
    video = render_demo_video(tmp / "clip.mp4", duration_s=1, fps=5, width=320, height=180)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    config.pitch = "fifa"
    runner = PipelineRunner(
        run_id="test-fifa", video_path=video, config=config, run_dir=tmp / "run"
    )
    return runner


def test_runner_wires_configured_pitch_into_stage_context(fifa_run):
    assert fifa_run.ctx.pitch is FIFA_PITCH


def test_runner_default_pitch_is_roboflow(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("roboflow-wiring")
    video = render_demo_video(tmp / "clip.mp4", duration_s=1, fps=5, width=320, height=180)
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    runner = PipelineRunner(
        run_id="test-roboflow", video_path=video, config=config, run_dir=tmp / "run"
    )
    assert runner.ctx.pitch is SOCCER_PITCH


def test_manifest_records_configured_pitch(fifa_run):
    assert fifa_run.manifest.config["pitch"] == "fifa"
