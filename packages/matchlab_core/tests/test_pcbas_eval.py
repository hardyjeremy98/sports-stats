from __future__ import annotations

import pytest
from matchlab_core.pcbas.eval import score_events, score_halves
from matchlab_core.pcbas.events import PCBASEvent


def _ev(frame, slot, cls, score=1.0, shirt=0):
    return PCBASEvent(
        frame_idx=frame,
        left_to_right=slot // 13,
        role_id=(slot % 13) + 1,
        slot=slot,
        shirt_number=shirt,
        class_id=cls,
        score=score,
    )


def test_exact_match_is_a_true_positive():
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 2)])
    assert r.macro_f1 == 1.0
    assert r.micro_f1 == 1.0


def test_match_within_delta():
    r = score_events([_ev(100, 0, 2)], [_ev(111, 0, 2)], delta=12)
    assert r.macro_f1 == 1.0


def test_no_match_outside_delta():
    r = score_events([_ev(100, 0, 2)], [_ev(113, 0, 2)], delta=12)
    assert r.macro_f1 == 0.0


def test_wrong_class_is_not_a_match():
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 3)])
    assert r.macro_f1 == 0.0


def test_wrong_slot_is_not_a_match():
    """Player-CENTRIC: right action, right time, wrong player is a miss.
    This is the whole difference from class+time avg-mAP."""
    r = score_events([_ev(100, 0, 2)], [_ev(100, 5, 2)])
    assert r.macro_f1 == 0.0


def test_low_confidence_predictions_are_discarded():
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 2, score=0.10)], conf_thresh=0.15)
    assert r.per_class[2].tp == 0
    assert r.per_class[2].fn == 1


def test_one_gt_absorbs_only_one_prediction():
    """Two predictions on one GT -> 1 TP, 1 FP, never 2 TP."""
    r = score_events([_ev(100, 0, 2)], [_ev(100, 0, 2), _ev(102, 0, 2)])
    assert (r.per_class[2].tp, r.per_class[2].fp) == (1, 1)


def test_greedy_matching_prefers_the_nearest_prediction():
    r = score_events([_ev(100, 0, 2)], [_ev(108, 0, 2, 0.9), _ev(101, 0, 2, 0.8)])
    assert r.per_class[2].tp == 1


def test_highest_scoring_prediction_claims_the_gt_first():
    """The reference sorts detections by DESCENDING SCORE, not by proximity. The
    0.9 prediction is farther away but claims the GT; the nearer 0.8 becomes an FP."""
    r = score_events([_ev(100, 0, 2)], [_ev(108, 0, 2, 0.9), _ev(101, 0, 2, 0.8)])
    assert (r.per_class[2].tp, r.per_class[2].fp) == (1, 1)


def test_macro_f1_averages_over_scored_classes_only():
    """Background is never scored, and absent classes must not silently count as 1.0."""
    r = score_events([_ev(1, 0, 2), _ev(50, 0, 5)], [_ev(1, 0, 2), _ev(50, 0, 5)])
    assert r.macro_f1 == 1.0
    assert 0 not in r.per_class


def test_empty_predictions_give_zero_not_a_crash():
    r = score_events([_ev(1, 0, 2)], [])
    assert r.macro_f1 == 0.0
    assert r.per_class[2].fn == 1


def test_per_class_counts_are_reported():
    """Rare classes (VAL has 26 tackles) must never be readable as a bare average."""
    r = score_events([_ev(1, 0, 7)], [_ev(1, 0, 7)])
    assert r.per_class[7].n_gt == 1
    assert r.per_class[7].class_name == "tackle"


def test_macro_and_micro_diverge_on_imbalanced_classes():
    """One perfect common class and one missed rare class: micro is dominated by
    the common one, macro is not. Reporting only one of them hides a dead class."""
    gt = [_ev(i, 0, 2) for i in range(0, 900, 100)] + [_ev(5000, 0, 7)]
    pred = [_ev(i, 0, 2) for i in range(0, 900, 100)]
    r = score_events(gt, pred)
    assert r.micro_f1 == pytest.approx(2 * 1.0 * (9 / 10) / (1.0 + 9 / 10))
    assert r.macro_f1 == pytest.approx(0.5)  # class 2 -> 1.0, class 7 -> 0.0


def test_shirt_identity_matches_on_team_and_jersey():
    """The reference's on-disk exchange format is shirt-native, not slot-native."""
    gt = [_ev(100, 0, 2, shirt=7)]
    pred = [_ev(100, 12, 2, shirt=7)]  # same side+shirt, different ROLE
    assert score_events(gt, pred, identity="shirt").macro_f1 == 1.0
    assert score_events(gt, pred, identity="slot").macro_f1 == 0.0


def test_slot_identity_requires_a_slot():
    ev = PCBASEvent(frame_idx=1, left_to_right=0, shirt_number=9, class_id=2)
    with pytest.raises(ValueError, match="slot"):
        score_events([ev], [ev], identity="slot")


def test_slot_is_derived_from_side_and_role_when_omitted():
    ev = PCBASEvent(frame_idx=1, left_to_right=1, role_id=4, shirt_number=9, class_id=2)
    assert ev.slot == 16


def test_inconsistent_slot_is_rejected():
    with pytest.raises(ValueError, match="slot"):
        PCBASEvent(
            frame_idx=1, left_to_right=1, role_id=4, slot=0, shirt_number=9, class_id=2
        )


def test_halves_are_matched_independently():
    """Frame indices are continuous across halves WITHIN a match but overlap ACROSS
    matches. Flattening every half into one pool would let a game_18 prediction
    satisfy a game_24 GT at the same frame."""
    gt = {"game_18_H1": [_ev(100, 0, 2)], "game_24_H1": [_ev(100, 0, 2)]}
    pred = {"game_18_H1": [_ev(100, 0, 2), _ev(100, 0, 2)], "game_24_H1": []}
    r = score_halves(gt, pred)
    assert (r.per_class[2].tp, r.per_class[2].fp, r.per_class[2].fn) == (1, 1, 1)


def test_half_with_no_predictions_counts_as_all_false_negatives():
    r = score_halves({"a": [_ev(1, 0, 2)]}, {})
    assert r.per_class[2].fn == 1
    assert r.micro_f1 == 0.0
