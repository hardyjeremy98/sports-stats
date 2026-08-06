"""SPO-80: the possession track's benchmark measurement path. The honest
"pass avg-mAP@1" is the PASS-class AP (not the diluted mean over all ~12
SoccerNet-ball GT classes, which a pass-only predictor would tank). `class_ap`
extracts it; the integration test proves possession-derived events are scorable
by the existing avg-mAP evaluator (SPO-49) with zero new metric code.
"""

from __future__ import annotations

import json
from pathlib import Path

from matchlab_core.action_spotting_eval import average_map, class_ap
from matchlab_core.config import PipelineConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas import PossessorFrame, Team
from matchlab_core.schemas.run import StageKind, StageStatus
from matchlab_core.stages.possession.events_from_possession import (
    events_to_spotted,
    transition_to_events,
)

CONFIGS = Path(__file__).parents[3] / "configs"


def _timeline(spans):
    frames = []
    for possessor, start, end in spans:
        for f in range(start, end + 1):
            frames.append(
                PossessorFrame(
                    frame_idx=f, t=f / 10.0, possessor_tracklet_id=possessor,
                    team=Team.HOME, confidence=0.9, margin=2.0,
                )
            )
    return frames


def test_class_ap_extracts_named_class():
    preds = [{"class": "PASS", "t": 1.0, "confidence": 0.9}]
    gt = [{"class": "PASS", "t": 1.0}]
    result = average_map(preds, gt, [1.0])
    assert class_ap(result, "PASS") == 1.0


def test_class_ap_absent_class_is_zero():
    result = average_map([], [{"class": "PASS", "t": 1.0}], [1.0])
    assert class_ap(result, "DRIVE") == 0.0


def test_pass_ap_not_diluted_by_other_gt_classes():
    # Full GT has PASS + DRIVE; a pass-only predictor scores PASS perfectly but
    # the 2-class mean is dragged down. class_ap isolates the honest pass number.
    preds = [{"class": "PASS", "t": 1.0, "confidence": 0.9}]
    gt = [{"class": "PASS", "t": 1.0}, {"class": "DRIVE", "t": 5.0}]
    result = average_map(preds, gt, [1.0])
    assert class_ap(result, "PASS") == 1.0
    assert result["avg_map"] < 1.0  # diluted mean over {PASS, DRIVE}


def test_possession_events_scored_by_avg_map():
    # Player 2 holds 0-9 then player 3 holds 10-19 -> a pass at t=0.9. A GT PASS
    # near that time should be a true positive at 1 s tolerance.
    events = transition_to_events(_timeline([(2, 0, 9), (3, 10, 19)]))
    spotted = events_to_spotted(events)
    gt = EventGroundTruth(
        source="test", fps=10.0,
        events=[GroundTruthEvent(class_="PASS", frame_idx=9, t=0.9)],
    )
    result = average_map(spotted, list(gt.events), [1.0])
    assert class_ap(result, "PASS") == 1.0


def test_smoke_config_selects_heuristic_possession():
    cfg = PipelineConfig.from_yaml(CONFIGS / "pipeline.possession-heuristic-smoke.yaml")
    assert cfg.stages[StageKind.POSSESSION].impl == "possession-heuristic-image"


def test_eval_config_selects_heuristic_possession_and_no_learned_spotter():
    cfg = PipelineConfig.from_yaml(CONFIGS / "pipeline.possession-heuristic-eval.yaml")
    assert cfg.stages[StageKind.POSSESSION].impl == "possession-heuristic-image"
    # No learned spotter clobbers spotting.json — the possession-derived passes
    # are what gets scored.
    spotting = cfg.stages.get(StageKind.SPOTTING)
    assert spotting is None or not spotting.enabled or spotting.impl == "none"


def test_smoke_config_runs_green_end_to_end(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("poss-smoke")
    video = render_demo_video(tmp / "clip.mp4", duration_s=3, fps=15, width=640, height=360)
    cfg = PipelineConfig.from_yaml(CONFIGS / "pipeline.possession-heuristic-smoke.yaml")
    runner = PipelineRunner(
        run_id="poss-smoke", video_path=video, config=cfg, run_dir=tmp / "run"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error
    rows = json.loads((tmp / "run" / "possession_timeline.json").read_text())
    for row in rows:
        PossessorFrame.model_validate(row)
