"""SPO-83: the shared ball->player candidate geometry, extracted from the
estimator so the profiler cannot drift from it."""

from __future__ import annotations

from matchlab_core.possession_ranking import (
    POSSESSOR_CLASSES,
    dist_point_box,
    index_possessor_boxes,
    rank_candidates,
)
from matchlab_core.schemas import (
    BallObservation,
    Box,
    DetectionClass,
    Point,
    Tracklet,
    TrackletFrame,
)


def _player(tid, frame, xyxy, conf=0.9, cls=DetectionClass.PLAYER):
    return Tracklet(
        tracklet_id=tid,
        cls=cls,
        frames=[
            TrackletFrame(
                frame_idx=frame,
                box=Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                confidence=conf,
            )
        ],
    )


def test_distance_is_zero_inside_the_box():
    assert dist_point_box(Point(x=10, y=20), Box(x1=0, y1=0, x2=20, y2=40)) == 0.0


def test_distance_is_edge_distance_outside_the_box():
    assert dist_point_box(Point(x=30, y=20), Box(x1=0, y1=0, x2=20, y2=40)) == 10.0


def test_index_keeps_players_and_goalkeepers_only():
    tracklets = [
        _player(1, 0, (0, 0, 20, 40)),
        _player(2, 0, (30, 0, 50, 40), cls=DetectionClass.GOALKEEPER),
        _player(9, 0, (60, 0, 80, 40), cls=DetectionClass.REFEREE),
    ]
    boxes = index_possessor_boxes(tracklets)
    assert sorted(tid for tid, _, _ in boxes[0]) == [1, 2]
    assert DetectionClass.REFEREE not in POSSESSOR_CLASSES


def test_rank_orders_by_distance_and_carries_boxes():
    tracklets = [_player(1, 0, (0, 0, 20, 40)), _player(2, 0, (100, 0, 120, 40))]
    boxes = index_possessor_boxes(tracklets)[0]
    ball = BallObservation(frame_idx=0, t=0.0, xy=Point(x=10, y=20), confidence=1.0)
    ranked = rank_candidates(ball, boxes)
    assert [c.tracklet_id for c in ranked] == [1, 2]
    assert ranked[0].distance == 0.0
    assert ranked[0].box.y2 - ranked[0].box.y1 == 40.0
    assert ranked[0].box_confidence == 0.9


def test_rank_on_empty_candidates_is_empty():
    ball = BallObservation(frame_idx=0, t=0.0, xy=Point(x=0, y=0), confidence=1.0)
    assert rank_candidates(ball, []) == []
