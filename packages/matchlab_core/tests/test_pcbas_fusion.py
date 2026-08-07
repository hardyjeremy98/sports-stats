"""Weighted Event Fusion tests (PAVE section 4).

The fusion is pure post-processing over per-model event lists, so it is fully
testable without a model. The tests that matter are the agreement filter and the
tackle exception: PAVE reports that the filter was deleting the only correct tackle
predictions, and tackle has 26 GT events in VAL and 174 trainable anchors in all of
TRAIN, so a filter that removes its handful of hits removes the class.
"""

from __future__ import annotations

import pytest
from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.fusion import fuse_model_events

DRIVE, PASS, TACKLE = 1, 2, 7


def ev(frame: int, *, slot: int = 3, class_id: int = DRIVE, score: float = 0.9,
       ltr: int = 1) -> PCBASEvent:
    return PCBASEvent(
        frame_idx=frame, left_to_right=ltr, slot=slot, class_id=class_id, score=score
    )


def test_two_models_agreeing_collapse_to_one_event():
    """The point of fusing: the same action found twice is one action."""
    fused = fuse_model_events([[ev(100, score=0.8)], [ev(105, score=0.6)]], delta=12)
    assert len(fused) == 1
    assert fused[0].class_id == DRIVE
    assert fused[0].slot == 3


def test_cluster_score_is_mean_times_sqrt_agreement_fraction():
    """PAVE's weighting: mean of contributing models x (n/N)^0.5.

    The penalty is what stops a lone confident model from outranking a quiet
    consensus in the metric's greedy match ordering.
    """
    models = [[ev(100, score=0.8)], [ev(104, score=0.6)], [], []]
    fused = fuse_model_events(models, delta=12)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(0.7 * (2 / 4) ** 0.5)


def test_full_agreement_leaves_the_mean_unpenalised():
    models = [[ev(100, score=0.8)], [ev(100, score=0.6)]]
    assert fuse_model_events(models, delta=12)[0].score == pytest.approx(0.7)


def test_single_model_support_is_discarded():
    """Fewer than 2 supporting models is dropped -- the agreement filter."""
    assert fuse_model_events([[ev(100)], [], [], []], delta=12) == []


def test_tackle_bypasses_the_agreement_filter():
    """PAVE exempts tackle explicitly: the filter was deleting the only correct
    tackle predictions. With 26 GT tackles in VAL, that removes the class."""
    fused = fuse_model_events([[ev(100, class_id=TACKLE)], [], [], []], delta=12)
    assert len(fused) == 1
    assert fused[0].class_id == TACKLE


def test_tackle_bypass_can_be_turned_off():
    """It is an exception worth being able to ablate rather than assume."""
    models = [[ev(100, class_id=TACKLE)], [], [], []]
    assert fuse_model_events(models, delta=12, solo_classes=()) == []


def test_events_further_apart_than_delta_stay_separate():
    """Two real actions by the same player must not merge into one."""
    models = [[ev(100, score=0.9), ev(400, score=0.9)],
              [ev(103, score=0.9), ev(402, score=0.9)]]
    fused = fuse_model_events(models, delta=12)
    assert [e.frame_idx for e in fused] == [100, 400]


def test_different_slots_and_classes_never_merge():
    """Grouping is by identity AND class, so a pass and a drive at the same frame
    are different events, as are two players acting simultaneously."""
    models = [[ev(100, slot=3), ev(100, slot=4), ev(100, class_id=PASS)],
              [ev(100, slot=3), ev(100, slot=4), ev(100, class_id=PASS)]]
    fused = fuse_model_events(models, delta=12)
    assert len(fused) == 3
    assert {(e.slot, e.class_id) for e in fused} == {(3, DRIVE), (4, DRIVE), (3, PASS)}


def test_clustering_seeds_on_the_highest_score_not_input_order():
    """Greedy assignment sorted on DESCENDING score, per PAVE. With a chain of
    events spaced under 2*delta, seeding from the wrong end merges a different
    partition -- and the representative frame comes from the seed."""
    models = [[ev(100, score=0.5)], [ev(110, score=0.99)], [ev(120, score=0.5)]]
    fused = fuse_model_events(models, delta=12)
    assert len(fused) == 1
    assert fused[0].frame_idx == 110  # the 0.99 seed, which reaches both others


def test_one_model_contributes_at_most_once_per_cluster():
    """A model that emitted two nearby events must not count as two supporters --
    that would let a single model satisfy the agreement filter by itself."""
    models = [[ev(100, score=0.9), ev(106, score=0.9)], [], [], []]
    assert fuse_model_events(models, delta=12) == []


def test_shirt_identity_grouping_is_available():
    """The reference's on-disk exchange is shirt-keyed, so fusion over remapped
    events must group on (team, shirt, class) as PAVE describes."""
    a = PCBASEvent(frame_idx=100, left_to_right=1, shirt_number=7, class_id=DRIVE, score=0.8)
    b = PCBASEvent(frame_idx=104, left_to_right=1, shirt_number=7, class_id=DRIVE, score=0.6)
    c = PCBASEvent(frame_idx=104, left_to_right=0, shirt_number=7, class_id=DRIVE, score=0.6)
    fused = fuse_model_events([[a], [b, c]], delta=12, identity="shirt")
    assert len(fused) == 1  # c is the OTHER team's number 7
    assert fused[0].shirt_number == 7 and fused[0].left_to_right == 1


def test_empty_input_is_empty_output():
    assert fuse_model_events([[], [], []], delta=12) == []
    assert fuse_model_events([], delta=12) == []


def test_output_is_sorted_by_frame():
    """The metric sorts by descending score globally, but a frame-ordered list is
    what every consumer here expects and what dedupe already returns."""
    models = [[ev(400), ev(100), ev(250)], [ev(402), ev(103), ev(248)]]
    fused = fuse_model_events(models, delta=12)
    assert [e.frame_idx for e in fused] == sorted(e.frame_idx for e in fused)
