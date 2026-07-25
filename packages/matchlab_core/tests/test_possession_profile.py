"""SPO-83: label-risk profiler for weak possessor labels.

NOT an accuracy measure -- no per-frame possessor ground truth exists on any
tier. Every assertion here is about the label set's own structure. Timelines are
hand-built so each indicator's expected value is known by construction.
"""

from __future__ import annotations

import pytest
from matchlab_core.possession_profile import profile_possessor_labels
from matchlab_core.schemas import PossessorFrame, Team
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
