from __future__ import annotations

from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.lab_view import build_lab_events


def _ev(frame, shirt, cls, score=1.0, ltr=0, has_bbox=None):
    return PCBASEvent(
        frame_idx=frame,
        left_to_right=ltr,
        shirt_number=shirt,
        class_id=cls,
        score=score,
        has_bbox=has_bbox,
    )


def _build(gt, pred, **kw):
    return build_lab_events(
        key="game_1_H1", game_id="game_1", half=1, fps=25.0, gt=gt, pred=pred, **kw
    )


def test_rows_and_report_come_from_one_matcher():
    out = _build([_ev(100, 7, 2), _ev(500, 9, 3)], [_ev(102, 7, 2), _ev(900, 4, 5)])
    kinds = [e.verdict for e in out.events]
    assert kinds.count("tp") == out.report.tp
    assert kinds.count("fp") == out.report.fp
    assert kinds.count("fn") == out.report.fn


def test_true_positive_is_anchored_at_the_prediction():
    out = _build([_ev(100, 7, 2)], [_ev(108, 7, 2)])
    (tp,) = out.events
    assert tp.verdict == "tp"
    assert tp.frame_idx == 108  # where the system claims it happened
    assert tp.gt_frame_idx == 100
    assert tp.frame_error == 8


def test_missed_event_is_anchored_at_ground_truth_and_has_no_score():
    out = _build([_ev(100, 7, 2)], [])
    (fn,) = out.events
    assert (fn.verdict, fn.frame_idx, fn.score) == ("fn", 100, None)


def test_offscreen_flag_survives_onto_the_row():
    out = _build([_ev(100, 7, 2, has_bbox=False)], [_ev(100, 7, 2)])
    assert out.events[0].has_bbox is False


def test_box_is_attached_when_the_player_is_observed():
    boxes = {(100, 0, 7): (10.0, 20.0, 30.0, 40.0)}
    out = _build([_ev(100, 7, 2)], [_ev(100, 7, 2)], boxes=boxes)
    assert out.events[0].box == (10.0, 20.0, 30.0, 40.0)
    # A different player at that frame must not inherit it.
    other = _build([_ev(100, 9, 2)], [_ev(100, 9, 2)], boxes=boxes)
    assert other.events[0].box is None


def test_events_are_ordered_by_frame():
    out = _build([_ev(500, 7, 2), _ev(100, 9, 3)], [_ev(300, 4, 5)])
    frames = [e.frame_idx for e in out.events]
    assert frames == sorted(frames)


def test_sub_threshold_prediction_is_not_a_row():
    out = _build([], [_ev(100, 7, 2, score=0.01)])
    assert out.events == []
