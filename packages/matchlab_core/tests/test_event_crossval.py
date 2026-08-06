"""B3: cross-validation of possession-derived events against ball-trajectory
touches. Two independent signals corroborating each other -- NOT a ground-truth
comparison. Hand-built inputs so each assertion is known by construction.
"""

from __future__ import annotations

import pytest
from matchlab_core.ball_kinematics import BallTouch
from matchlab_core.event_crossval import crossvalidate_events
from matchlab_core.schemas import Event, EventType, Team

FPS = 25.0


def _event(eid, frame, etype=EventType.PASS, confidence=0.6):
    return Event(
        event_id=eid,
        type=etype,
        frame_idx=frame,
        t=frame / FPS,
        player_id=1,
        team=Team.HOME,
        confidence=confidence,
    )


def _touch(frame, score=0.8):
    return BallTouch(
        frame_idx=frame,
        t=frame / FPS,
        score=score,
        turn=score,
        speed_change=0.0,
        interpolated=False,
    )


def test_exact_frame_match_is_matched():
    r = crossvalidate_events([_event(1, 100)], [_touch(100)], tolerance_frames=6)
    assert r.matched == 1
    assert r.possession_only == 0
    assert r.trajectory_only == 0
    assert r.agreement_rate == pytest.approx(1.0)


def test_match_within_tolerance():
    r = crossvalidate_events([_event(1, 100)], [_touch(104)], tolerance_frames=6)
    assert r.matched == 1


def test_no_match_outside_tolerance():
    r = crossvalidate_events([_event(1, 100)], [_touch(120)], tolerance_frames=6)
    assert r.matched == 0
    assert r.possession_only == 1
    assert r.trajectory_only == 1


def test_one_touch_cannot_match_two_events():
    r = crossvalidate_events([_event(1, 100), _event(2, 102)], [_touch(101)], tolerance_frames=6)
    assert r.matched == 1
    assert r.possession_only == 1
    assert r.trajectory_only == 0


def test_nearest_event_wins_the_touch():
    r = crossvalidate_events([_event(1, 100), _event(2, 105)], [_touch(104)], tolerance_frames=6)
    matched = [c for c in r.corroborations if c.matched_touch_frame is not None]
    assert len(matched) == 1
    assert matched[0].event_id == 2


def test_counts_are_disjoint_and_total_correctly():
    events = [_event(1, 10), _event(2, 50), _event(3, 90)]
    touches = [_touch(11), _touch(200)]
    r = crossvalidate_events(events, touches, tolerance_frames=6)
    assert r.matched + r.possession_only == r.n_events == 3
    assert r.matched + r.trajectory_only == r.n_touches == 2


def test_matched_events_gain_confidence_and_unmatched_lose_it():
    events = [_event(1, 10, confidence=0.5), _event(2, 90, confidence=0.5)]
    r = crossvalidate_events(
        events, [_touch(10)], tolerance_frames=6,
        corroboration_bonus=0.2, disagreement_penalty=0.3,
    )
    by_id = {c.event_id: c for c in r.corroborations}
    assert by_id[1].adjusted_confidence == pytest.approx(0.7)
    assert by_id[2].adjusted_confidence == pytest.approx(0.2)


def test_adjusted_confidence_is_clamped_to_unit_interval():
    r = crossvalidate_events(
        [_event(1, 10, confidence=0.95), _event(2, 90, confidence=0.05)],
        [_touch(10)], tolerance_frames=6,
        corroboration_bonus=0.5, disagreement_penalty=0.5,
    )
    for c in r.corroborations:
        assert 0.0 <= c.adjusted_confidence <= 1.0


def test_per_type_counts_are_reported():
    events = [_event(1, 10, EventType.PASS), _event(2, 50, EventType.RECEPTION)]
    r = crossvalidate_events(events, [_touch(10)], tolerance_frames=6)
    assert r.matched_by_type["pass"] == 1
    assert r.matched_by_type.get("reception", 0) == 0


def test_no_events_and_no_touches_is_zero_not_a_crash():
    r = crossvalidate_events([], [], tolerance_frames=6)
    assert r.n_events == 0
    assert r.agreement_rate == 0.0
    assert r.touch_recall == 0.0


def test_events_with_no_touches_are_all_possession_only():
    r = crossvalidate_events([_event(1, 10), _event(2, 20)], [], tolerance_frames=6)
    assert r.possession_only == 2
    assert r.agreement_rate == 0.0


def test_refine_snaps_an_event_to_its_corroborating_touch():
    from matchlab_core.event_crossval import refine_event_timing

    refined = refine_event_timing([_event(1, 100)], [_touch(97)], tolerance_frames=6)
    assert refined[0].frame_idx == 97
    assert refined[0].t == pytest.approx(97 / FPS)


def test_refine_leaves_uncorroborated_events_untouched():
    from matchlab_core.event_crossval import refine_event_timing

    refined = refine_event_timing([_event(1, 100)], [_touch(200)], tolerance_frames=6)
    assert refined[0].frame_idx == 100


def test_refine_does_not_mutate_the_input_events():
    from matchlab_core.event_crossval import refine_event_timing

    events = [_event(1, 100)]
    refine_event_timing(events, [_touch(97)], tolerance_frames=6)
    assert events[0].frame_idx == 100


def test_refine_records_the_shift_in_attrs():
    from matchlab_core.event_crossval import refine_event_timing

    refined = refine_event_timing([_event(1, 100)], [_touch(97)], tolerance_frames=6)
    assert refined[0].attrs["timing_refined_from"] == 100
    assert refined[0].attrs["timing_shift_frames"] == -3


def test_refine_keeps_events_frame_ordered():
    from matchlab_core.event_crossval import refine_event_timing

    refined = refine_event_timing(
        [_event(1, 100), _event(2, 130)], [_touch(97), _touch(133)], tolerance_frames=6
    )
    assert [e.frame_idx for e in refined] == sorted(e.frame_idx for e in refined)


def test_refine_with_no_touches_is_a_no_op():
    from matchlab_core.event_crossval import refine_event_timing

    refined = refine_event_timing([_event(1, 100)], [], tolerance_frames=6)
    assert [e.frame_idx for e in refined] == [100]
