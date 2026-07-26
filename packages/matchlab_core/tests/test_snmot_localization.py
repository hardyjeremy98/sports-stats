"""B3: localisation error against SNMOT's one-action-per-clip ground truth.

Recall/localisation only -- with a single labelled action per 30s clip an
unmatched prediction is very likely a real unlabelled action, so precision, F1
and mAP are all unsupported here. These tests pin that the metric refuses to
imply otherwise.
"""

from __future__ import annotations

import pytest
from matchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
from matchlab_core.snmot_action_gt import snmot_localization_error


def _gt(class_="Corner", frame=100, fps=25.0):
    return EventGroundTruth(
        source="soccernet-tracking", sequence="T-1", fps=fps,
        events=[GroundTruthEvent(class_=class_, frame_idx=frame, t=frame / fps)],
    )


def test_error_is_distance_to_the_nearest_prediction():
    r = snmot_localization_error(_gt(), [90, 103, 400])
    assert r.matched is True
    assert r.error_frames == 3


def test_nearest_wins_regardless_of_order():
    assert snmot_localization_error(_gt(), [400, 101, 90]).error_frames == 1


def test_exact_hit_is_zero_error():
    assert snmot_localization_error(_gt(), [100]).error_frames == 0


def test_no_predictions_is_unmatched_not_zero_error():
    r = snmot_localization_error(_gt(), [])
    assert r.matched is False
    assert r.error_frames is None


def test_clip_with_no_labelled_action_is_not_scorable():
    gt = EventGroundTruth(source="soccernet-tracking", sequence="T-1", fps=25.0, events=[])
    r = snmot_localization_error(gt, [10, 20])
    assert r.scorable is False
    assert r.matched is False


def test_class_and_ball_contact_flag_are_carried():
    assert snmot_localization_error(_gt("Corner"), [100]).ball_contact is True
    assert snmot_localization_error(_gt("Yellow card"), [100]).ball_contact is False
    assert snmot_localization_error(_gt("Yellow card"), [100]).class_ == "Yellow card"


def test_error_seconds_uses_the_sequence_fps():
    r = snmot_localization_error(_gt(fps=50.0), [125])
    assert r.error_frames == 25
    assert r.error_seconds == pytest.approx(0.5)
