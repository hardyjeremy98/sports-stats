"""Pitch-space game-state metrics (SPO-69): project GT tracks through a run's
calibration and score coverage / implausible speed / teleports / in-bounds.

Unit tests build synthetic GroundTruth + synthetic FrameCalibration rows and
call `compute_gamestate_metrics` directly (its real input shape), so every
expected number is hand-computable without run-dir scaffolding. Integration
tests at the end go through `evaluate_run` to prove the section is wired into
eval.json + headline_metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.pitch import FIFA_PITCH
from matchlab_core.schemas.calibration import FrameCalibration
from matchlab_core.schemas.geometry import Box

IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _shift_h(dx: float, dy: float = 0.0) -> list[list[float]]:
    """Identity homography translated by (dx, dy) in projected space."""
    return [[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]


def _box_at(cx: float, y_bottom: float, w: float = 40.0, h: float = 120.0) -> Box:
    """Box whose bottom-center is (cx, y_bottom)."""
    return Box(x1=cx - w / 2, y1=y_bottom - h, x2=cx + w / 2, y2=y_bottom)


def _track(track_id: int, boxes: dict[int, Box], role: str = "player") -> GroundTruthTrack:
    frames = [GroundTruthFrame(frame_idx=f, box=b) for f, b in sorted(boxes.items())]
    return GroundTruthTrack(track_id=track_id, role=role, frames=frames)


def _gt(tracks: list[GroundTruthTrack], fps: float = 25.0) -> GroundTruth:
    return GroundTruth(source="synthetic", sequence="SYN", fps=fps, width=1920, height=1080,
                       tracks=tracks)


def _calib(frame_idx: int, homography, status: str, fps: float = 25.0) -> FrameCalibration:
    return FrameCalibration(
        frame_idx=frame_idx,
        t=frame_idx / fps,
        homography=homography,
        status=status,
        confidence=1.0,
    )


def test_identity_homography_stationary_box_is_clean():
    from matchlab_core.gamestate_eval import compute_gamestate_metrics

    # One player, stationary at bottom-center (520, 420), identity H every frame.
    boxes = {f: _box_at(520.0, 420.0) for f in range(10)}
    gt = _gt([_track(1, boxes)])
    calibration = [_calib(f, IDENTITY, "fresh") for f in range(10)]

    gs = compute_gamestate_metrics(gt, calibration, FIFA_PITCH, fps=25.0, stride=1)

    assert gs["coverage"] == 1.0
    assert gs["teleport_count"] == 0
    assert gs["teleports_at_refresh"] == 0
    assert gs["implausible_speed_rate"] == 0.0
    assert gs["in_bounds_rate"] == 1.0
    assert gs["n_steps"] == 9  # 10 frames -> 9 consecutive steps


def test_calibration_gap_reduces_coverage_and_breaks_step_chain():
    from matchlab_core.gamestate_eval import compute_gamestate_metrics

    # 6 sampled frames; frames 2 and 3 are ABSENT (no homography). The player
    # is at (500,300) for frames 0-1 then, having moved during the unobserved
    # gap, at (5000,300) for frames 4-5. If the gap were bridged, the 1->4 step
    # (45 m) would be a giant teleport -- it must NOT be, because 2/3 are absent
    # so 1 and 4 are not consecutive sampled frames with usable H.
    boxes = {0: _box_at(500.0, 300.0), 1: _box_at(500.0, 300.0),
             2: _box_at(700.0, 300.0), 3: _box_at(700.0, 300.0),
             4: _box_at(5000.0, 300.0), 5: _box_at(5000.0, 300.0)}
    gt = _gt([_track(1, boxes)])
    calibration = [
        _calib(0, IDENTITY, "fresh"), _calib(1, IDENTITY, "fresh"),
        _calib(2, None, "absent"), _calib(3, None, "absent"),
        _calib(4, IDENTITY, "fresh"), _calib(5, IDENTITY, "fresh"),
    ]

    gs = compute_gamestate_metrics(gt, calibration, FIFA_PITCH, fps=25.0, stride=1)

    assert gs["n_sampled_frames_gt_covered"] == 6
    assert gs["n_frames_with_homography"] == 4
    assert gs["coverage"] == pytest.approx(4 / 6, abs=1e-4)  # exactly the gap fraction
    # Only (0->1) and (4->5) are consecutive usable pairs; the gap boundary
    # never forms a step, so no false teleport despite the 45 m jump.
    assert gs["n_steps"] == 2
    assert gs["teleport_count"] == 0
    assert gs["implausible_speed_rate"] == 0.0


def test_homography_snap_flags_teleport_and_implausible_speed():
    from matchlab_core.gamestate_eval import compute_gamestate_metrics

    # Stationary box, but H jumps +2000 cm (20 m) between the two frames.
    boxes = {0: _box_at(520.0, 420.0), 1: _box_at(520.0, 420.0)}
    gt = _gt([_track(1, boxes)])
    calibration = [_calib(0, IDENTITY, "fresh"), _calib(1, _shift_h(2000.0), "fresh")]

    gs = compute_gamestate_metrics(gt, calibration, FIFA_PITCH, fps=25.0, stride=1)

    assert gs["n_steps"] == 1
    assert gs["teleport_count"] == 1
    assert gs["implausible_speed_rate"] > 0.0
    # Both rows are "fresh": no status transition, so not a refresh teleport.
    assert gs["teleports_at_refresh"] == 0


def test_moving_player_continuous_h_is_clean_at_stride_two():
    from matchlab_core.gamestate_eval import compute_gamestate_metrics

    # 5 m/s at fps 25, stride 2 -> dt = 2/25 s = 0.08 s -> 0.4 m = 40 cm per
    # consecutive sampled step. Sampled frames are the even ones (0,2,4,6,8);
    # GT is dense but only even frames carry a calibration row. Identity H, so
    # pitch cm == image px: x advances 40 per sampled step.
    boxes = {f: _box_at(500.0 + (f // 2) * 40.0, 420.0) for f in range(10)}
    gt = _gt([_track(1, boxes)])
    calibration = [_calib(f, IDENTITY, "fresh") for f in range(0, 10, 2)]

    gs = compute_gamestate_metrics(gt, calibration, FIFA_PITCH, fps=25.0, stride=2)

    assert gs["n_steps"] == 4
    assert gs["teleport_count"] == 0  # 0.4 m < 2 m
    assert gs["implausible_speed_rate"] == 0.0  # 5 m/s < 12 m/s


def test_in_bounds_rate_reflects_out_of_bounds_projection():
    from matchlab_core.gamestate_eval import compute_gamestate_metrics

    # Track 1 projects on-pitch (identity, x=520); track 2 projects far off the
    # pitch (x=20000 cm > 10500 + 500 margin) every frame -> half the positions
    # are out of bounds.
    boxes_in = {f: _box_at(520.0, 420.0) for f in range(4)}
    boxes_out = {f: _box_at(20000.0, 420.0) for f in range(4)}
    gt = _gt([_track(1, boxes_in), _track(2, boxes_out)])
    calibration = [_calib(f, IDENTITY, "fresh") for f in range(4)]

    gs = compute_gamestate_metrics(gt, calibration, FIFA_PITCH, fps=25.0, stride=1)

    assert gs["n_projected_positions"] == 8  # 2 tracks x 4 frames
    assert gs["n_in_bounds"] == 4
    assert gs["in_bounds_rate"] == 0.5


# --- run-dir / evaluate_run integration -------------------------------------


def _write_gs_run_dir(root: Path, gt: GroundTruth, calibration, *, pitch="fifa",
                      with_calibration=True) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    manifest = {
        "video": {"fps": gt.fps, "frame_count": 10, "sample_stride": 1},
        "config": {"pitch": pitch, "stages": {"track": {"impl": "iou", "params": {}}}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    # A single tracklet sitting on GT track 1 so the MOT eval has something to
    # score (evaluate_run requires tracklets.json).
    tracklets = [{
        "tracklet_id": 10,
        "cls": "player",
        "frames": [
            {"frame_idx": f, "box": {"x1": 500.0, "y1": 300.0, "x2": 540.0, "y2": 420.0},
             "confidence": 0.9}
            for f in range(10)
        ],
    }]
    (run_dir / "tracklets.json").write_text(json.dumps(tracklets))
    (run_dir / "players.json").write_text(json.dumps([]))
    if with_calibration:
        with open(run_dir / "calibration.jsonl", "w") as f:
            for c in calibration:
                f.write(c.model_dump_json() + "\n")
    return run_dir


def test_evaluate_gamestate_none_without_calibration_artifact(tmp_path):
    from matchlab_core.gamestate_eval import evaluate_gamestate

    boxes = {f: _box_at(520.0, 420.0) for f in range(10)}
    gt = _gt([_track(1, boxes)])
    run_dir = _write_gs_run_dir(tmp_path, gt, [], with_calibration=False)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert evaluate_gamestate(run_dir, gt, manifest) is None


def test_evaluate_run_omits_gamestate_without_calibration(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    boxes = {f: _box_at(500.0, 420.0) for f in range(10)}
    gt = _gt([_track(1, boxes)])
    run_dir = _write_gs_run_dir(tmp_path, gt, [], with_calibration=False)

    result = evaluate_run(run_dir, gt)
    assert "gamestate" not in result
    heads = headline_metrics(result)
    assert "gs_coverage" not in heads


def test_evaluate_run_folds_gamestate_and_gs_headline(tmp_path):
    pytest.importorskip("motmetrics")
    from matchlab_core.evaluation import evaluate_run, headline_metrics

    boxes = {f: _box_at(500.0, 420.0) for f in range(10)}
    gt = _gt([_track(1, boxes)])
    calibration = [_calib(f, IDENTITY, "fresh") for f in range(10)]
    run_dir = _write_gs_run_dir(tmp_path, gt, calibration)

    result = evaluate_run(run_dir, gt)
    gs = result["gamestate"]
    assert gs is not None
    assert gs["pitch"] == "fifa"
    assert gs["coverage"] == 1.0

    heads = headline_metrics(result)
    assert heads["gs_coverage"] == gs["coverage"]
    assert heads["gs_implausible_speed_rate"] == gs["implausible_speed_rate"]
    assert heads["gs_teleports"] == gs["teleport_count"]
    assert heads["gs_in_bounds_rate"] == gs["in_bounds_rate"]
