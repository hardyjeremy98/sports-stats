from __future__ import annotations

import pytest
from matchlab_core.pcbas.schema import (
    CLASS_NAMES,
    CLS,
    FRAME,
    LEFT_TO_RIGHT,
    N_CLASSES,
    N_SLOTS,
    ROI_X,
    ROLE_ID,
    ROLE_NAMES,
    X_POS,
    slot_index,
    slot_to_role,
)


def test_class_order_is_the_reference_order():
    # Frozen: reordering silently corrupts every trained checkpoint and metric.
    assert CLASS_NAMES == [
        "background",
        "drive",
        "pass",
        "cross",
        "throw-in",
        "shot",
        "header",
        "tackle",
        "block",
    ]
    assert N_CLASSES == 9


def test_column_indices_match_tactical_data_format():
    # From the dataset's own tactical_data_format.txt
    assert (FRAME, LEFT_TO_RIGHT, ROLE_ID, X_POS, ROI_X, CLS) == (0, 2, 4, 5, 9, 13)


def test_slot_index_packs_team_and_role():
    assert slot_index(0, 1) == 0  # left team, role 1
    assert slot_index(0, 13) == 12
    assert slot_index(1, 1) == 13  # right team, role 1
    assert slot_index(1, 13) == 25
    assert N_SLOTS == 26


def test_slot_round_trips():
    for ltr in (0, 1):
        for role in range(1, 14):
            assert slot_to_role(slot_index(ltr, role)) == (ltr, role)


@pytest.mark.parametrize("ltr,role", [(0, 0), (0, 14), (2, 1), (-1, 5)])
def test_slot_index_rejects_out_of_range(ltr, role):
    with pytest.raises(ValueError):
        slot_index(ltr, role)


def test_slot_to_role_rejects_out_of_range():
    for bad in (-1, 26, 99):
        with pytest.raises(ValueError):
            slot_to_role(bad)


def test_role_names_cover_every_role_id():
    assert sorted(ROLE_NAMES) == list(range(1, 14))
    assert ROLE_NAMES[1] == "GK"
