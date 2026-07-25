"""SPO-83: label-risk profiler for weak possessor labels.

NOT an accuracy measure -- no per-frame possessor ground truth exists on any
tier. Every assertion here is about the label set's own structure. Timelines are
hand-built so each indicator's expected value is known by construction.
"""

from __future__ import annotations

import pytest
from matchlab_core.possession_profile import aggregate_profiles, profile_possessor_labels
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    PossessorFrame,
    Team,
    Tracklet,
    TrackletFrame,
)
from matchlab_core.stages.possession.heuristic_image import Params


def _row(frame_idx, tid, margin=50.0, team=Team.HOME, conf=0.9):
    """One timeline row. tid=None means the estimator abstained."""
    return PossessorFrame(
        frame_idx=frame_idx,
        t=frame_idx / 25.0,
        possessor_tracklet_id=tid,
        team=team if tid is not None else Team.UNKNOWN,
        confidence=conf if tid is not None else 0.0,
        margin=margin,
    )


def test_coverage_counts_asserted_rows():
    timeline = [_row(0, 1), _row(1, 1), _row(2, None), _row(3, None)]
    p = profile_possessor_labels(
        timeline, [], [], Params(min_margin_px=10.0), total_frames=10
    )
    assert p.total_frames == 10
    assert p.asserted_frames == 2
    assert p.coverage == pytest.approx(0.2)


def test_abstention_causes_sum_to_all_non_asserted_frames():
    timeline = [
        _row(0, 1),                  # asserted
        _row(1, None, margin=50.0),  # abstained, margin above min -> outside radius
        _row(2, None, margin=1.0),   # abstained, margin below min -> contested tie
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(min_margin_px=10.0), total_frames=7
    )
    a = p.abstention
    assert a.no_ball_observation == 4      # 7 total - 3 rows
    assert a.outside_radius == 1
    assert a.contested_tie == 1
    assert a.no_ball_observation + a.outside_radius + a.contested_tie == 6
    assert p.asserted_frames + 6 == p.total_frames


def test_empty_timeline_is_all_no_ball():
    p = profile_possessor_labels([], [], [], Params(), total_frames=5)
    assert p.asserted_frames == 0
    assert p.coverage == 0.0
    assert p.abstention.no_ball_observation == 5


def test_total_frames_below_row_count_is_a_programming_error():
    with pytest.raises(ValueError, match="total_frames"):
        profile_possessor_labels([_row(0, 1), _row(1, 1)], [], [], Params(), total_frames=1)


def test_zero_total_frames_does_not_divide_by_zero():
    p = profile_possessor_labels([], [], [], Params(), total_frames=0)
    assert p.coverage == 0.0


def test_contested_curve_counts_asserted_rows_below_each_tau():
    timeline = [
        _row(0, 1, margin=1.0),
        _row(1, 1, margin=6.0),
        _row(2, 1, margin=30.0),
        _row(3, None, margin=0.5),  # abstained -- must not enter the curve
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=4, tau_grid_px=(0.0, 2.0, 10.0, 40.0)
    )
    assert [pt.threshold for pt in p.contested_curve] == [0.0, 2.0, 10.0, 40.0]
    assert [pt.count for pt in p.contested_curve] == [0, 1, 2, 3]
    assert p.contested_curve[-1].fraction == pytest.approx(1.0)


def test_contested_curve_is_monotone_non_decreasing():
    timeline = [_row(i, 1, margin=float(i)) for i in range(20)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=20)
    counts = [pt.count for pt in p.contested_curve]
    assert counts == sorted(counts)


def test_contested_curve_with_no_asserted_rows_is_all_zero():
    timeline = [_row(0, None, margin=0.0)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=1)
    assert all(pt.count == 0 and pt.fraction == 0.0 for pt in p.contested_curve)


def _tracklet(tid, frame_boxes, conf=0.9, cls=DetectionClass.PLAYER):
    """frame_boxes: dict frame_idx -> (x1, y1, x2, y2)."""
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(
                frame_idx=f,
                box=Box(x1=b[0], y1=b[1], x2=b[2], y2=b[3]),
                confidence=conf,
            )
            for f, b in sorted(frame_boxes.items())
        ],
    )


def _ball_obs(frame, x, y):
    return BallObservation(
        frame_idx=frame, t=frame / 25.0, xy=Point(x=x, y=y), confidence=1.0
    )


def test_depth_discordance_flags_a_much_taller_runner_up():
    # Possessor (tid 1) is 20px tall; the runner-up (tid 2) is 80px tall --
    # a much nearer player sitting comparably close in pixels.
    tracklets = [
        _tracklet(1, {0: (0, 0, 10, 20)}),
        _tracklet(2, {0: (14, 0, 40, 80)}),
    ]
    ball = [_ball_obs(0, 5, 10)]
    p = profile_possessor_labels(
        [_row(0, 1)], tracklets, ball, Params(), total_frames=1,
        depth_ratio_grid=(1.2, 2.0, 8.0),
    )
    assert p.depth_evaluable_frames == 1
    assert [pt.count for pt in p.depth_discordance] == [1, 1, 0]
    assert p.depth_discordance[0].fraction == pytest.approx(1.0)


def test_depth_concordance_when_candidates_are_similar_height():
    tracklets = [
        _tracklet(1, {0: (0, 0, 10, 40)}),
        _tracklet(2, {0: (14, 0, 24, 42)}),
    ]
    ball = [_ball_obs(0, 5, 20)]
    p = profile_possessor_labels(
        [_row(0, 1)], tracklets, ball, Params(), total_frames=1,
        depth_ratio_grid=(1.2, 2.0),
    )
    assert p.depth_evaluable_frames == 1
    assert [pt.count for pt in p.depth_discordance] == [0, 0]


def test_single_candidate_frames_are_not_depth_evaluable():
    tracklets = [_tracklet(1, {0: (0, 0, 10, 20)})]
    ball = [_ball_obs(0, 5, 10)]
    p = profile_possessor_labels(
        [_row(0, 1)], tracklets, ball, Params(), total_frames=1
    )
    assert p.depth_evaluable_frames == 0
    assert all(pt.count == 0 for pt in p.depth_discordance)


def test_runner_up_is_nearest_other_not_rank_one_after_smoothing():
    # Smoothing made tid 2 the possessor even though tid 1 is nearest the ball.
    # The runner-up must then be tid 1 (nearest *other*), not tid 2 itself.
    tracklets = [
        _tracklet(1, {0: (0, 0, 10, 100)}),   # nearest, tall
        _tracklet(2, {0: (30, 0, 40, 20)}),   # possessor after smoothing, short
    ]
    ball = [_ball_obs(0, 5, 50)]
    p = profile_possessor_labels(
        [_row(0, 2)], tracklets, ball, Params(), total_frames=1,
        depth_ratio_grid=(2.0,),
    )
    assert p.depth_evaluable_frames == 1
    assert p.depth_discordance[0].count == 1  # 100/20 = 5.0 > 2.0


def test_segments_split_on_possessor_change():
    timeline = [_row(0, 1), _row(1, 1), _row(2, 2), _row(3, 2), _row(4, 2)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=5)
    assert p.segments.count == 2
    assert p.segments.total_segment_frames == 5
    assert p.segments.mean_frames == pytest.approx(2.5)


def test_segments_split_on_a_frame_gap_even_with_the_same_possessor():
    timeline = [_row(0, 1), _row(1, 1), _row(5, 1)]  # frames 2-4 unobserved
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=6)
    assert p.segments.count == 2


def test_below_te_counts_short_segments():
    # Segment lengths 1, 2, 3, 4 with te=3 -> two segments below threshold.
    timeline = [
        _row(0, 1),
        _row(1, 2), _row(2, 2),
        _row(3, 3), _row(4, 3), _row(5, 3),
        _row(6, 4), _row(7, 4), _row(8, 4), _row(9, 4),
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=10, te_frames=3
    )
    assert p.segments.count == 4
    assert p.segments.below_te_count == 2
    assert p.segments.below_te_fraction == pytest.approx(0.5)


def test_changes_per_second_uses_fps_and_span():
    # 25 rows spanning 1 second at 25 fps, alternating every 5 frames -> 4 changes.
    timeline = [_row(i, 1 + i // 5) for i in range(25)]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=25, fps=25.0
    )
    assert p.segments.changes == 4
    assert p.segments.span_seconds == pytest.approx(1.0)
    assert p.segments.changes_per_second == pytest.approx(4.0)


def test_abstention_rows_count_as_a_change():
    timeline = [_row(0, 1), _row(1, None), _row(2, 1)]
    p = profile_possessor_labels(timeline, [], [], Params(), total_frames=3)
    assert p.segments.changes == 2


def test_short_segment_switching_team_is_implausible():
    timeline = [
        _row(0, 1, team=Team.HOME),                       # 1-frame segment
        _row(1, 2, team=Team.AWAY), _row(2, 2, team=Team.AWAY),
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=3, te_frames=3
    )
    assert p.implausible_team_flips == 1


def test_long_segment_switching_team_is_plausible():
    timeline = [
        _row(0, 1, team=Team.HOME), _row(1, 1, team=Team.HOME),
        _row(2, 1, team=Team.HOME), _row(3, 1, team=Team.HOME),
        _row(4, 2, team=Team.AWAY),
    ]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=5, te_frames=3
    )
    assert p.implausible_team_flips == 0


def test_unknown_team_never_counts_as_a_flip():
    timeline = [_row(0, 1, team=Team.UNKNOWN), _row(1, 2, team=Team.AWAY)]
    p = profile_possessor_labels(
        timeline, [], [], Params(), total_frames=2, te_frames=3
    )
    assert p.implausible_team_flips == 0


def test_aggregate_sums_counts_and_recomputes_fractions():
    a = profile_possessor_labels(
        [_row(0, 1, margin=1.0), _row(1, 1, margin=100.0)], [], [], Params(),
        total_frames=4, tau_grid_px=(2.0,),
    )
    b = profile_possessor_labels(
        [_row(0, 1, margin=1.0)], [], [], Params(), total_frames=6, tau_grid_px=(2.0,),
    )
    agg = aggregate_profiles([a, b])
    assert agg.total_frames == 10
    assert agg.asserted_frames == 3
    assert agg.coverage == pytest.approx(0.3)
    assert agg.contested_curve[0].count == 2
    assert agg.contested_curve[0].fraction == pytest.approx(2 / 3)


def test_aggregate_rejects_mismatched_curve_grids():
    a = profile_possessor_labels([], [], [], Params(), total_frames=1, tau_grid_px=(2.0,))
    b = profile_possessor_labels([], [], [], Params(), total_frames=1, tau_grid_px=(5.0,))
    with pytest.raises(ValueError, match="grid"):
        aggregate_profiles([a, b])


def test_aggregate_of_nothing_is_empty_not_a_crash():
    agg = aggregate_profiles([])
    assert agg.total_frames == 0
    assert agg.coverage == 0.0
