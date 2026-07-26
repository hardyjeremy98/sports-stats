from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.pcbas.schema import CLS, LEFT_TO_RIGHT, ROI_X, ROLE_ID, slot_index
from matchlab_train.experiments.pcbas_infer_logits import (
    INFERENCE_DUMMY_BOX,
    slot_rois_masks,
)
from matchlab_train.registry import available


def _row(frame, pid, ltr, role, roi):
    r = np.full(14, np.nan, dtype=np.float64)
    r[0], r[1], r[2], r[4] = frame, pid, ltr, role
    if roi is not None:
        r[9], r[10], r[11], r[12] = roi
    r[13] = 0
    return r


def test_experiment_is_registered():
    assert "pcbas-infer-logits" in available()


def test_shapes_are_the_frozen_26_slot_contract():
    rows = np.stack([_row(0, 1, 0, 1, (960, 540, 60, 120))])
    rois, masks = slot_rois_masks(rows, [0, 1, 2])
    assert rois.shape == (26, 3, 5)
    assert masks.shape == (26, 3)


def test_slot_index_places_the_player_in_the_right_row():
    rows = np.stack(
        [
            _row(0, 1, 0, 1, (960, 540, 60, 120)),  # left GK  -> slot 0
            _row(0, 2, 1, 4, (300, 200, 60, 120)),  # right MCB -> slot 16
        ]
    )
    _, masks = slot_rois_masks(rows, [0])
    assert masks[slot_index(0, 1), 0] == 1.0
    assert masks[slot_index(1, 4), 0] == 1.0
    assert masks.sum() == 2.0


def test_unoccupied_slots_are_masked_with_an_in_bounds_dummy_box():
    """roi_align still reads every slot, so a masked slot's box must be valid
    geometry -- a degenerate or out-of-frame box would produce NaNs that the mask
    multiply cannot clean up."""
    rows = np.stack([_row(0, 1, 0, 1, (960, 540, 60, 120))])
    rois, masks = slot_rois_masks(rows, [0])
    empty = slot_index(1, 7)
    assert masks[empty, 0] == 0.0
    assert tuple(rois[empty, 0, 1:]) == pytest.approx(INFERENCE_DUMMY_BOX)
    assert rois[empty, 0, 3] > rois[empty, 0, 1]
    assert rois[empty, 0, 4] > rois[empty, 0, 2]


def test_off_screen_players_are_masked():
    rows = np.stack([_row(0, 1, 0, 1, None)])  # in the tactical data, no bbox
    _, masks = slot_rois_masks(rows, [0])
    assert masks.sum() == 0.0


def test_roi_frame_column_is_clip_local():
    """Absolute video frames in column 0 would send roi_align far outside the batch."""
    rows = np.stack([_row(5000, 1, 0, 1, (960, 540, 60, 120))])
    rois, _ = slot_rois_masks(rows, [4999, 5000, 5001])
    assert rois[0, :, 0].tolist() == [0.0, 1.0, 2.0]


def test_frames_outside_the_window_are_ignored():
    rows = np.stack(
        [
            _row(10, 1, 0, 1, (960, 540, 60, 120)),
            _row(99, 1, 0, 1, (960, 540, 60, 120)),
        ]
    )
    _, masks = slot_rois_masks(rows, [9, 10, 11])
    assert masks[slot_index(0, 1)].tolist() == [0.0, 1.0, 0.0]


def test_substitution_overlap_keeps_the_first_row():
    """During a substitution two players can briefly share one (side, role). The
    reference takes the first; recording that as behaviour rather than luck, because
    Phase 3 reads slot occupancy from tracking and will hit the same case."""
    rows = np.stack(
        [
            _row(0, 1, 0, 2, (300, 100, 60, 120)),
            _row(0, 2, 0, 2, (900, 100, 60, 120)),
        ]
    )
    rois, masks = slot_rois_masks(rows, [0])
    slot = slot_index(0, 2)
    assert masks[slot, 0] == 1.0
    # The first row's box (x=300), expanded by the 1.125 coefficient, not the second.
    assert rois[slot, 0, 1] == pytest.approx((300 - 0.125 * 60 / 2) / 3.0, abs=1e-3)


def test_out_of_range_roles_are_skipped_not_crashed():
    """Defensive: a role id outside 1..13 must not raise mid-inference over a
    90-minute half."""
    rows = np.stack([_row(0, 1, 0, 1, (960, 540, 60, 120))])
    rows[0][ROLE_ID] = 99
    _, masks = slot_rois_masks(rows, [0])
    assert masks.sum() == 0.0


def test_missing_side_or_role_is_skipped():
    rows = np.stack([_row(0, 1, 0, 1, (960, 540, 60, 120))])
    rows[0][LEFT_TO_RIGHT] = np.nan
    _, masks = slot_rois_masks(rows, [0])
    assert masks.sum() == 0.0


def test_label_column_is_not_consulted():
    """Inference must never read CLS. If it did, a leak would inflate every number
    and nothing downstream would reveal it."""
    a = np.stack([_row(0, 1, 0, 1, (960, 540, 60, 120))])
    b = a.copy()
    b[0][CLS] = 5
    assert np.array_equal(slot_rois_masks(a, [0])[0], slot_rois_masks(b, [0])[0])
    assert np.array_equal(slot_rois_masks(a, [0])[1], slot_rois_masks(b, [0])[1])


def test_boxes_are_scaled_and_expanded_like_training():
    rows = np.stack([_row(0, 1, 0, 1, (960, 540, 120, 240))])
    rois, _ = slot_rois_masks(rows, [0])
    x1 = rois[slot_index(0, 1), 0, 1]
    assert x1 == pytest.approx((960 - 0.125 * 120 / 2) / 3.0, abs=1e-3)


def test_degenerate_boxes_are_masked():
    rows = np.stack([_row(0, 1, 0, 1, (960, 540, 0.1, 0.1))])
    rois, masks = slot_rois_masks(rows, [0])
    assert masks.sum() == 0.0
    assert tuple(rois[slot_index(0, 1), 0, 1:]) == pytest.approx(INFERENCE_DUMMY_BOX)


def test_no_observable_rows_gives_all_masked_but_valid_geometry():
    rows = np.zeros((0, 14))
    rois, masks = slot_rois_masks(rows, [0, 1])
    assert masks.sum() == 0.0
    assert not np.isnan(rois).any()
    assert (rois[:, :, 3] > rois[:, :, 1]).all()


def test_roi_x_nan_rows_never_reach_the_box_builder():
    rows = np.stack([_row(0, 1, 0, 1, (960, 540, 60, 120))])
    rows[0][ROI_X] = np.nan
    rois, masks = slot_rois_masks(rows, [0])
    assert masks.sum() == 0.0
    assert not np.isnan(rois).any()
