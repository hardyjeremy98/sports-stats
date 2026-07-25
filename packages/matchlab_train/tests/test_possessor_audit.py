"""SPO-83: GT -> oracle possession inputs, on a hand-built GroundTruth so every
surrounding input is pinned and a failure localizes to the adapter."""

from __future__ import annotations

from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.schemas import Box, DetectionClass, Team
from matchlab_train.datasets.possessor_audit import gt_to_possession_inputs


def _track(track_id, role, team, frames):
    """frames: dict frame_idx -> (x1, y1, x2, y2)."""
    return GroundTruthTrack(
        track_id=track_id,
        role=role,
        team=team,
        frames=[
            GroundTruthFrame(frame_idx=f, box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
            for f, b in sorted(frames.items())
        ],
    )


def _gt(tracks, seq_length=4, fps=25.0):
    return GroundTruth(
        source="soccernet-tracking",
        sequence="TEST-1",
        fps=fps,
        width=1920,
        height=1080,
        seq_length=seq_length,
        tracks=tracks,
    )


def test_players_and_goalkeepers_become_possessor_class_tracklets():
    gt = _gt([
        _track(1, "player", "left", {0: (0, 0, 10, 40)}),
        _track(2, "goalkeeper", "right", {0: (50, 0, 60, 40)}),
    ])
    tracklets, _, _ = gt_to_possession_inputs(gt)
    by_id = {t.tracklet_id: t for t in tracklets}
    assert by_id[1].cls == DetectionClass.PLAYER
    assert by_id[2].cls == DetectionClass.GOALKEEPER
    assert by_id[1].frames[0].confidence == 1.0
    assert by_id[1].frames[0].source == "observed"


def test_referees_are_kept_as_referee_class_tracklets():
    gt = _gt([_track(9, "referee", None, {0: (0, 0, 10, 40)})])
    tracklets, teams, _ = gt_to_possession_inputs(gt)
    assert tracklets[0].cls == DetectionClass.REFEREE
    assert teams[0].team == Team.REFEREE


def test_team_mapping_matches_the_oracle_team_stage():
    gt = _gt([
        _track(1, "player", "left", {0: (0, 0, 10, 40)}),
        _track(2, "player", "right", {0: (50, 0, 60, 40)}),
        _track(3, "player", None, {0: (80, 0, 90, 40)}),
    ])
    _, teams, _ = gt_to_possession_inputs(gt)
    by_id = {t.tracklet_id: t.team for t in teams}
    assert by_id == {1: Team.HOME, 2: Team.AWAY, 3: Team.UNKNOWN}


def test_ball_track_becomes_observations_at_box_centres():
    gt = _gt([_track(99, "ball", None, {0: (10, 20, 14, 24), 1: (30, 40, 34, 44)})])
    _, _, ball = gt_to_possession_inputs(gt)
    assert [b.frame_idx for b in ball] == [0, 1]
    assert (ball[0].xy.x, ball[0].xy.y) == (12.0, 22.0)
    assert ball[0].t == 0.0
    assert ball[1].t == 0.04  # 1 / 25 fps
    assert all(b.confidence == 1.0 and not b.interpolated for b in ball)


def test_unannotated_ball_frames_produce_no_observation():
    # Ball annotated on frames 0 and 3 only; 1 and 2 are genuine absence.
    gt = _gt([_track(99, "ball", None, {0: (10, 20, 14, 24), 3: (30, 40, 34, 44)})])
    _, _, ball = gt_to_possession_inputs(gt)
    assert [b.frame_idx for b in ball] == [0, 3]


def test_ball_is_not_a_possessor_candidate_tracklet():
    gt = _gt([
        _track(1, "player", "left", {0: (0, 0, 10, 40)}),
        _track(99, "ball", None, {0: (10, 20, 14, 24)}),
    ])
    tracklets, _, _ = gt_to_possession_inputs(gt)
    assert [t.tracklet_id for t in tracklets] == [1]


def test_no_ball_track_yields_no_observations():
    gt = _gt([_track(1, "player", "left", {0: (0, 0, 10, 40)})])
    _, _, ball = gt_to_possession_inputs(gt)
    assert ball == []
