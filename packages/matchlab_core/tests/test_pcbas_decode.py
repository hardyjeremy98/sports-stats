from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.pcbas.decode import decode_logits, softmax, suppress
from matchlab_core.pcbas.schema import N_SLOTS


def _logits(n_frames: int) -> np.ndarray:
    """All-background logits: class 0 wins everywhere."""
    a = np.zeros((9, N_SLOTS, n_frames), dtype=np.float32)
    a[0] = 5.0
    return a


def test_all_background_decodes_to_nothing():
    assert decode_logits(_logits(100)) == []


def test_a_single_peak_becomes_one_event():
    a = _logits(100)
    a[2, 3, 50] = 9.0  # slot 3, class 2 (pass)
    events = decode_logits(a)
    assert len(events) == 1
    assert (events[0].frame_idx, events[0].slot, events[0].class_id) == (50, 3, 2)
    assert 0.0 < events[0].score <= 1.0


def test_slot_maps_back_to_side_and_role():
    a = _logits(20)
    a[2, 16, 5] = 9.0  # slot 16 = right side, role 4
    e = decode_logits(a)[0]
    assert (e.left_to_right, e.role_id) == (1, 4)


def test_decoded_events_have_no_shirt_number():
    """The model is slot-native. Inventing a shirt here would bypass the ADR 008
    export-time remap and quietly assert an identity the model never predicted."""
    a = _logits(20)
    a[2, 0, 5] = 9.0
    assert decode_logits(a)[0].shirt_number is None


def test_nms_collapses_a_plateau_to_one_event():
    a = _logits(100)
    a[2, 0, 48:53] = 9.0  # five adjacent frames of the same class
    events = decode_logits(a, nms_window=25)
    assert len(events) == 1


def test_nms_keeps_the_highest_scoring_peak():
    a = _logits(100)
    a[2, 0, 40] = 8.0
    a[2, 0, 45] = 12.0  # higher
    events = decode_logits(a, nms_window=25)
    assert len(events) == 1
    assert events[0].frame_idx == 45


def test_events_further_apart_than_the_window_both_survive():
    a = _logits(200)
    a[2, 0, 30] = 9.0
    a[2, 0, 100] = 9.0
    assert len(decode_logits(a, nms_window=25)) == 2


def test_nms_does_not_suppress_across_classes():
    """A pass and a header 3 frames apart are two different events, not a duplicate."""
    a = _logits(100)
    a[2, 0, 50] = 9.0  # pass
    a[6, 0, 53] = 10.0  # header
    events = decode_logits(a, nms_window=25)
    assert {e.class_id for e in events} == {2, 6}


def test_nms_does_not_suppress_across_slots():
    """Two players can act simultaneously; that is the whole point of player-centric
    spotting. Suppressing across slots would delete one of them."""
    a = _logits(100)
    a[2, 0, 50] = 9.0
    a[2, 7, 50] = 9.0
    events = decode_logits(a, nms_window=25)
    assert {e.slot for e in events} == {0, 7}


def test_events_are_sorted_by_frame():
    a = _logits(300)
    a[2, 5, 200] = 9.0
    a[2, 1, 40] = 9.0
    frames = [e.frame_idx for e in decode_logits(a)]
    assert frames == sorted(frames)


def test_min_score_filters_weak_peaks():
    a = _logits(100)
    a[2, 0, 50] = 5.2  # barely beats the background logit of 5.0
    assert decode_logits(a, min_score=0.9) == []
    assert len(decode_logits(a, min_score=0.0)) == 1


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        decode_logits(np.zeros((8, 26, 10), dtype=np.float32))


# --- primitives -------------------------------------------------------------------


def test_softmax_is_a_distribution():
    p = softmax(np.array([[1.0], [2.0], [3.0]]), axis=0)
    assert p.sum() == pytest.approx(1.0)
    assert p[2] > p[1] > p[0]


def test_softmax_does_not_overflow_on_large_logits():
    """The reference's np_softmax omits the max subtraction and overflows above ~88."""
    p = softmax(np.array([[1000.0], [999.0]]), axis=0)
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)


def test_suppress_returns_ascending_frames():
    frames = np.array([10, 100, 200])
    kept = suppress(frames, np.array([0.5, 0.9, 0.7]), 5)
    assert kept == [10, 100, 200]


def test_suppress_prefers_the_higher_score_on_a_tie_window():
    frames = np.array([10, 12])
    assert suppress(frames, np.array([0.5, 0.9]), 5) == [12]
