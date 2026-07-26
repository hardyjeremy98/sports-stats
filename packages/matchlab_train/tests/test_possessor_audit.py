"""SPO-83: GT -> oracle possession inputs, on a hand-built GroundTruth so every
surrounding input is pinned and a failure localizes to the adapter."""

from __future__ import annotations

import pytest
from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.schemas import Box, DetectionClass, Team
from matchlab_train.datasets.possessor_audit import (
    ESTIMATORS,
    _possession_timeline,
    audit_sequence,
    audit_sequences,
    crossval_sequence,
    crossval_sequences,
    gt_to_possession_inputs,
)


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


def test_two_ball_tracks_resolve_to_one_observation_per_frame():
    """SoccerNet declares a second ball ('ball;2') on some sequences -- a spare
    ball on the pitch. BallObservation is the SINGLE resolved ball for a frame,
    so the adapter must not emit one row per GT ball track."""
    gt = _gt(
        [
            _track(8, "ball", None, {f: (10, 20, 14, 24) for f in range(4)}),
            _track(31, "ball", None, {2: (900, 500, 904, 504), 3: (902, 500, 906, 504)}),
        ],
        seq_length=4,
    )
    _, _, ball = gt_to_possession_inputs(gt)
    assert [b.frame_idx for b in ball] == [0, 1, 2, 3]
    assert len({b.frame_idx for b in ball}) == len(ball)


def test_ball_resolution_follows_the_continuous_track_not_the_stray():
    """Temporal continuity picks the ball consistent with recent history, so a
    far-side stray does not hijack the possession signal mid-sequence."""
    gt = _gt(
        [
            _track(8, "ball", None, {f: (10, 20, 14, 24) for f in range(4)}),
            _track(31, "ball", None, {2: (900, 500, 904, 504), 3: (902, 500, 906, 504)}),
        ],
        seq_length=4,
    )
    _, _, ball = gt_to_possession_inputs(gt)
    assert all(b.xy.x < 100 for b in ball)


def test_unannotated_gaps_are_never_interpolated():
    """Absence of GT ball is genuine absence -- filling it would fabricate
    possession labels the ground truth does not support."""
    gt = _gt([_track(8, "ball", None, {0: (10, 20, 14, 24), 5: (30, 40, 34, 44)})], seq_length=6)
    _, _, ball = gt_to_possession_inputs(gt)
    assert [b.frame_idx for b in ball] == [0, 5]
    assert not any(b.interpolated for b in ball)


def _scene(seq_length, ball_frames, name="TEST-1"):
    """Two players 100px apart; the ball sits on player 1 for `ball_frames`."""
    players = [
        _track(1, "player", "left", {f: (0, 0, 20, 40) for f in range(seq_length)}),
        _track(2, "player", "right", {f: (100, 0, 120, 40) for f in range(seq_length)}),
    ]
    ball = _track(99, "ball", None, {f: (8, 18, 12, 22) for f in range(ball_frames)})
    gt = _gt([*players, ball], seq_length=seq_length)
    gt.sequence = name
    return gt


def test_audit_sequence_reports_ball_coverage_and_profiles():
    audit = audit_sequence(_scene(10, 10), smooth_radius=0)
    assert audit.sequence == "TEST-1"
    assert audit.total_frames == 10
    assert audit.ball_gt_frames == 10
    assert audit.ball_coverage == pytest.approx(1.0)
    assert audit.excluded is False
    assert audit.profile.asserted_frames == 10
    assert audit.profile.coverage == pytest.approx(1.0)


def test_low_ball_coverage_sequence_is_excluded_but_still_listed():
    report = audit_sequences(
        [_scene(10, 10, "GOOD-1"), _scene(10, 1, "SPARSE-1")], smooth_radius=0
    )
    by_name = {s.sequence: s for s in report.sequences}
    assert by_name["GOOD-1"].excluded is False
    assert by_name["SPARSE-1"].excluded is True
    assert by_name["SPARSE-1"].ball_coverage == pytest.approx(0.1)
    assert len(report.sequences) == 2  # excluded sequences stay visible


def test_aggregate_covers_retained_sequences_only():
    report = audit_sequences(
        [_scene(10, 10, "GOOD-1"), _scene(10, 1, "SPARSE-1")], smooth_radius=0
    )
    assert report.aggregate.total_frames == 10  # SPARSE-1's 10 frames excluded
    assert report.aggregate.asserted_frames == 10


def test_exclusion_threshold_is_configurable_and_recorded():
    report = audit_sequences([_scene(10, 6, "MID-1")], min_ball_coverage=0.8)
    assert report.min_ball_coverage == 0.8
    assert report.sequences[0].excluded is True
    assert report.aggregate.total_frames == 0


def test_report_carries_a_no_accuracy_caveat():
    report = audit_sequences([_scene(10, 10)])
    assert "accuracy" in report.caveat.lower()


def _rally(seq_length=60):
    """Two players; the ball shuttles between them with sharp direction changes
    -- both signals should fire, so the pair is cross-validatable."""
    p1 = _track(1, "player", "left", {f: (0, 100, 40, 180) for f in range(seq_length)})
    p2 = _track(2, "player", "right", {f: (400, 100, 440, 180) for f in range(seq_length)})
    xy = {}
    for f in range(seq_length):
        leg, step = divmod(f, 20)
        x = 20 + 19 * step if leg % 2 == 0 else 400 - 19 * step
        xy[f] = (x, 140)
    ball = _track(99, "ball", None, {f: (v[0] - 2, v[1] - 2, v[0] + 2, v[1] + 2)
                                     for f, v in xy.items()})
    gt = _gt([p1, p2, ball], seq_length=seq_length)
    return gt


def test_crossval_sequence_reports_both_signals():
    cv = crossval_sequence(_rally(), smooth_radius=0)
    assert cv.sequence == "TEST-1"
    assert cv.report.n_events > 0
    assert cv.report.n_touches > 0
    assert cv.report.matched + cv.report.possession_only == cv.report.n_events
    assert cv.report.matched + cv.report.trajectory_only == cv.report.n_touches


def test_crossval_carries_a_no_ground_truth_caveat():
    report = crossval_sequences([_rally()])
    assert "corroboration" in report.caveat.lower()
    assert "ground truth" in report.caveat.lower()


def test_crossval_aggregate_sums_the_per_sequence_counts():
    report = crossval_sequences([_rally(), _rally()], smooth_radius=0)
    assert len(report.sequences) == 2
    assert report.aggregate.n_events == sum(s.report.n_events for s in report.sequences)
    assert report.aggregate.matched == sum(s.report.matched for s in report.sequences)


def test_crossval_with_no_ball_yields_no_touches_and_zero_agreement():
    gt = _gt([_track(1, "player", "left", {f: (0, 0, 20, 40) for f in range(20)})], seq_length=20)
    cv = crossval_sequence(gt)
    assert cv.report.n_touches == 0
    assert cv.report.agreement_rate == 0.0


# --------------------------------------------------------------------------
# Estimator dispatch -- the seam the denoiser ablation runs through
# --------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_possession_timeline_dispatches_to_every_declared_estimator(estimator):
    tracklets, teams, ball = gt_to_possession_inputs(_rally())
    timeline = _possession_timeline(estimator, tracklets, teams, ball)
    assert [f.frame_idx for f in timeline] == [b.frame_idx for b in ball]


def test_possession_timeline_rejects_unknown_estimator():
    tracklets, teams, ball = gt_to_possession_inputs(_rally())
    with pytest.raises(ValueError, match="unknown estimator"):
        _possession_timeline("nope", tracklets, teams, ball)


def test_possession_alias_resolves_to_the_heuristic():
    """The localisation CLI's historical `--signal possession` must keep meaning
    the SPO-79 heuristic, or every previously published number silently changes."""
    tracklets, teams, ball = gt_to_possession_inputs(_rally())
    alias = _possession_timeline("possession", tracklets, teams, ball)
    named = _possession_timeline("possession-heuristic-image", tracklets, teams, ball)
    assert [f.possessor_tracklet_id for f in alias] == [
        f.possessor_tracklet_id for f in named
    ]


def test_crossval_records_which_estimator_produced_the_events():
    report = crossval_sequences([_rally()], estimator="possession-viterbi")
    assert report.estimator == "possession-viterbi"
