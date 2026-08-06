from __future__ import annotations

import numpy as np
from matchlab_core.pcbas.decode import decode_logits
from matchlab_core.pcbas.logits import validate_logits
from matchlab_core.pcbas.schema import CLS, N_SLOTS, slot_index
from matchlab_train.experiments.pcbas_oracle_logits import (
    ACTION_LOGIT,
    BACKGROUND_LOGIT,
    oracle_logits,
)
from matchlab_train.registry import available


def _row(frame, ltr, role, cls=0, observed=True):
    r = np.full(14, np.nan, dtype=np.float64)
    r[0], r[1], r[2], r[4] = frame, 1, ltr, role
    r[5], r[6], r[7], r[8] = 0.5, 0.5, 0.0, 0.0
    if observed:
        r[9], r[10], r[11], r[12] = 100, 100, 20, 40
    r[13] = cls
    return r


def test_experiment_is_registered():
    assert "pcbas-oracle-logits" in available()


def test_output_satisfies_the_frozen_contract():
    rows = np.stack([_row(i, 0, 1) for i in range(50)])
    validate_logits(oracle_logits(rows, 50))


def test_background_wins_where_nothing_happens():
    rows = np.stack([_row(i, 0, 1) for i in range(50)])
    out = oracle_logits(rows, 50).astype(np.float32)
    assert (out.argmax(0) == 0).all()


def test_the_event_class_wins_at_the_event_frame_and_slot():
    rows = np.stack([_row(i, 0, 1) for i in range(50)])
    rows[20][CLS] = 5  # a shot
    out = oracle_logits(rows, 50, dilation=0).astype(np.float32)
    slot = slot_index(0, 1)
    assert out[:, slot, 20].argmax() == 5
    assert out[:, slot, 21].argmax() == 0  # neighbours stay background at dilation 0


def test_only_the_acting_slot_is_marked():
    """An oracle that lit every slot would make the role head trivial for the wrong
    reason -- the point is to test whether DST can READ a correct signal, not to
    remove the identity problem."""
    rows = np.stack(
        [_row(i, 0, 1) for i in range(50)] + [_row(i, 1, 4) for i in range(50)]
    )
    rows[20][CLS] = 5
    out = oracle_logits(rows, 50, dilation=0).astype(np.float32)
    acting = slot_index(0, 1)
    for slot in range(N_SLOTS):
        expected = 5 if slot == acting else 0
        assert out[:, slot, 20].argmax() == expected


def test_dilation_widens_the_peak_like_stage_one_training_does():
    rows = np.stack([_row(i, 0, 1) for i in range(50)])
    rows[20][CLS] = 5
    out = oracle_logits(rows, 50, dilation=1).astype(np.float32)
    slot = slot_index(0, 1)
    assert [int(out[:, slot, t].argmax()) for t in (18, 19, 20, 21, 22)] == [0, 5, 5, 5, 0]


def test_off_screen_events_are_included():
    """A PURE oracle, deliberately: it encodes every GT event including the 17.5%
    whose player has no bounding box. Those are unreachable for a real visual stage,
    so this is an upper bound on what stage 1 could ever hand DST."""
    rows = np.stack([_row(0, 0, 1, observed=False)])
    rows[0][CLS] = 3
    out = oracle_logits(rows, 1, dilation=0).astype(np.float32)
    assert out[:, slot_index(0, 1), 0].argmax() == 3


def test_decoding_the_oracle_recovers_the_events():
    """The end-to-end sanity check: our own decoder, unchanged, must read the oracle
    back out. If it cannot, the oracle is not a valid upper bound."""
    rows = np.stack([_row(i, 0, 1) for i in range(400)])
    for frame, cls in ((50, 2), (200, 5), (350, 7)):
        rows[frame][CLS] = cls
    events = decode_logits(oracle_logits(rows, 400, dilation=1).astype(np.float32))
    assert [(e.frame_idx, e.class_id) for e in events] == [(50, 2), (200, 5), (350, 7)]
    assert {e.slot for e in events} == {slot_index(0, 1)}


def test_logit_magnitudes_resemble_real_stage_one_output():
    """Measured on real VAL logits: background mean +3.9, action mean -3.6. Feeding
    DST an oracle on a wildly different scale would change the input distribution and
    confound the comparison."""
    assert 2.0 <= BACKGROUND_LOGIT <= 6.0
    assert ACTION_LOGIT > BACKGROUND_LOGIT
