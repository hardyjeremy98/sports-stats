from __future__ import annotations

import pytest
from matchlab_core.pcbas.denoiser import EOS_ACTION, NEUTRAL_ROLE, SOS_ACTION
from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.schema import slot_index
from matchlab_train.experiments.pcbas_denoise_infer import dedupe, tokens_to_events
from matchlab_train.registry import available


def test_experiment_is_registered():
    assert "pcbas-denoise-infer" in available()


# --- token -> event ---------------------------------------------------------------


def test_timestamp_maps_to_an_absolute_frame():
    """Timestamp index 0 is reserved for SOS, so window frame f is encoded as f+1.
    An off-by-one is invisible at delta=12 and would corrupt any tighter analysis."""
    events = tokens_to_events([(2, 5, 101, 0.9)], window_start=1000)
    assert len(events) == 1
    assert events[0].frame_idx == 1000 + 100


def test_action_index_shifts_back_to_class_id():
    """Decoder action 0-7 are classes 1-8; 8 is SOS and 9 is EOS."""
    assert tokens_to_events([(0, 1, 10, 0.5)], 0)[0].class_id == 1
    assert tokens_to_events([(7, 1, 10, 0.5)], 0)[0].class_id == 8


def test_role_index_is_the_slot():
    e = tokens_to_events([(1, slot_index(1, 4), 10, 0.5)], 0)[0]
    assert (e.slot, e.left_to_right, e.role_id) == (16, 1, 4)


@pytest.mark.parametrize(
    "token,reason",
    [
        ((EOS_ACTION, 3, 10, 0.9), "EOS"),
        ((SOS_ACTION, 3, 10, 0.9), "SOS"),
        ((2, NEUTRAL_ROLE, 10, 0.9), "neutral role"),
        ((2, 3, 0, 0.9), "timestamp 0 is SOS's slot, not a frame"),
    ],
)
def test_control_tokens_are_dropped(token, reason):
    assert tokens_to_events([token], 0) == [], reason


def test_score_is_carried_through():
    assert tokens_to_events([(2, 3, 10, 0.77)], 0)[0].score == pytest.approx(0.77)


# --- dedup ------------------------------------------------------------------------


def _ev(frame, slot=1, cls=2, score=0.9):
    return PCBASEvent(
        frame_idx=frame, left_to_right=slot // 13, role_id=(slot % 13) + 1,
        slot=slot, class_id=cls, score=score,
    )


def test_overlapping_windows_collapse_to_one_event():
    """Stride 375 against framespan 750 means every action is emitted about twice.
    The duplicate is a guaranteed false positive: a GT event matches only once."""
    assert len(dedupe([_ev(1000, score=0.9), _ev(1003, score=0.8)], 25)) == 1


def test_dedup_keeps_the_higher_scoring_duplicate():
    kept = dedupe([_ev(1000, score=0.6), _ev(1003, score=0.95)], 25)
    assert len(kept) == 1
    assert kept[0].frame_idx == 1003


def test_distinct_events_survive_dedup():
    assert len(dedupe([_ev(1000), _ev(1200)], 25)) == 2


def test_dedup_does_not_merge_across_slots_or_classes():
    """Two players can act at the same instant, and one player can head then pass."""
    assert len(dedupe([_ev(1000, slot=1), _ev(1000, slot=7)], 25)) == 2
    assert len(dedupe([_ev(1000, cls=2), _ev(1000, cls=6)], 25)) == 2


def test_dedup_output_is_sorted_by_frame():
    frames = [e.frame_idx for e in dedupe([_ev(2000), _ev(100), _ev(900)], 25)]
    assert frames == sorted(frames)


def test_dedup_of_nothing_is_nothing():
    assert dedupe([], 25) == []


def test_encode_decode_round_trip_recovers_exact_frames():
    """THE decisive plumbing test, and the one missing when DST's first end-to-end
    run scored 0.035: encode ground-truth events as decoder tokens, decode them
    back, and require the original absolute frames and slots.

    Tested in isolation, both halves looked right. Only the round trip pins the
    SOS-reserved timestamp offset (window frame f is encoded f+1) against the
    inverse (window_start + timestamp - 1). A mismatch here would be indisputably
    a bug; agreement means a poor score is the model's, not the plumbing's.
    """
    import numpy as np
    from matchlab_train.datasets.footpass_windows import build_tokens

    window_start = 74_088
    events = np.array(
        [[0, 0, 1], [37, 16, 2], [300, 25, 8], [749, 7, 5]], dtype=np.int64
    )
    actions, roles, frames = build_tokens(events, 750)

    tokens = [
        (int(a.argmax()), int(r.argmax()), int(f), 0.9)
        for a, r, f in zip(actions, roles, frames)
    ]
    decoded = tokens_to_events(tokens, window_start)

    assert [e.frame_idx for e in decoded] == [window_start + f for f in (0, 37, 300, 749)]
    assert [e.slot for e in decoded] == [0, 16, 25, 7]
    assert [e.class_id for e in decoded] == [1, 2, 8, 5]
