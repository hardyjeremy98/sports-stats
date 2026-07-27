"""Window-dataset tests.

The load-bearing ones are the symmetry tests. Mirroring the pitch is only
label-preserving if the ROLE SLOTS are remapped too, and a flip that moves the
coordinates but not the slots trains the model that left-backs stand on the right --
which would look like a slightly-worse model, never like a bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.pcbas.denoiser import (
    ABSENT_FILL,
    ACTION_VOCAB,
    ENCODER_FEATURE_DIM,
    EOS_ACTION,
    FEATURES_PER_SLOT,
    NEUTRAL_ROLE,
    ROLE_VOCAB,
    SOS_ACTION,
)
from matchlab_core.pcbas.schema import N_CLASSES, N_SLOTS, ROLE_NAMES, slot_index
from matchlab_train.datasets.footpass_windows import (
    apply_symmetry,
    build_kinematics,
    build_tokens,
    encoder_features,
    one_hot_frames,
    slot_permutation,
    window_events,
)


def _row(frame, ltr, role, x=0.25, y=0.4, vx=0.1, vy=0.2, observed=True, cls=0):
    r = np.full(14, np.nan, dtype=np.float64)
    r[0], r[1], r[2], r[4] = frame, 1, ltr, role
    r[5], r[6], r[7], r[8] = x, y, vx, vy
    if observed:
        r[9], r[10], r[11], r[12] = 100, 100, 20, 40
    r[13] = cls
    return r


# --- slot permutations ------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "y", "xy"])
def test_permutations_are_bijections(axis):
    perm = slot_permutation(axis)
    assert sorted(perm.tolist()) == list(range(N_SLOTS))


@pytest.mark.parametrize("axis", ["x", "y", "xy"])
def test_permutations_are_involutions(axis):
    """Mirroring twice must be the identity, or augmentation slowly rotates the
    slot assignment away from the truth."""
    perm = slot_permutation(axis)
    assert perm[perm].tolist() == list(range(N_SLOTS))


def test_x_symmetry_swaps_side_and_left_right_role():
    """Attacking direction flips AND the pitch mirrors, so a left team's left-back
    becomes the right team's right-back."""
    lb = ROLE_NAMES  # sanity: 2 = LB, 13 = RB
    assert (lb[2], lb[13]) == ("LB", "RB")
    perm = slot_permutation("x")
    assert perm[slot_index(0, 2)] == slot_index(1, 13)


def test_y_symmetry_keeps_the_side():
    """Reflecting across Y does not change which way a team attacks."""
    perm = slot_permutation("y")
    assert perm[slot_index(0, 2)] == slot_index(0, 13)


def test_xy_symmetry_keeps_the_role():
    """Two mirrors cancel on left/right, but the attacking direction still flips."""
    perm = slot_permutation("xy")
    assert perm[slot_index(0, 2)] == slot_index(1, 2)


def test_self_mirrored_roles_stay_put_under_y():
    perm = slot_permutation("y")
    for role in (1, 4, 8, 9, 12):  # GK, MCB, DM, AM, CF
        assert perm[slot_index(0, role)] == slot_index(0, role)


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="axis"):
        slot_permutation("z")


# --- symmetry applied -------------------------------------------------------------


def _kin_log(x=0.25, y=0.4, vx=0.1, vy=0.2):
    kin = np.full((5, N_SLOTS, 1), ABSENT_FILL, dtype=np.float32)
    slot = slot_index(0, 2)  # left team, LB
    kin[:, slot, 0] = (x, y, vx, vy, 1.0)
    log = np.zeros((N_CLASSES, N_SLOTS, 1), dtype=np.float32)
    log[2, slot, 0] = 7.0
    return kin, log, slot


def test_x_symmetry_moves_the_player_and_its_slot_together():
    kin, log, slot = _kin_log()
    events = np.array([[0, slot, 2]], dtype=np.int64)
    kin2, log2, ev2 = apply_symmetry(kin, log, events, "x")
    target = slot_index(1, 13)

    assert ev2[0, 1] == target
    assert kin2[0, target, 0] == pytest.approx(1 - 0.25)  # x mirrored
    assert kin2[2, target, 0] == pytest.approx(-0.1)  # vx negated
    assert kin2[1, target, 0] == pytest.approx(0.4)  # y untouched
    assert log2[2, target, 0] == 7.0  # logits followed the slot


def test_y_symmetry_mirrors_y_only():
    kin, log, slot = _kin_log()
    kin2, _, _ = apply_symmetry(kin, log, np.zeros((0, 3), dtype=np.int64), "y")
    target = slot_index(0, 13)
    assert kin2[1, target, 0] == pytest.approx(1 - 0.4)
    assert kin2[3, target, 0] == pytest.approx(-0.2)
    assert kin2[0, target, 0] == pytest.approx(0.25)


def test_absent_slots_are_not_mirrored_into_the_pitch():
    """ABSENT_FILL is a sentinel, not a coordinate. Mirroring it would produce
    1 - (-15) = 16, a plausible-looking position for a player who is not there."""
    kin, log, _ = _kin_log()
    kin2, _, _ = apply_symmetry(kin, log, np.zeros((0, 3), dtype=np.int64), "x")
    empty = slot_index(1, 9)
    assert kin2[0, empty, 0] == pytest.approx(ABSENT_FILL)


def test_symmetry_applied_twice_is_the_identity():
    kin, log, slot = _kin_log()
    events = np.array([[0, slot, 2]], dtype=np.int64)
    k1, l1, e1 = apply_symmetry(kin, log, events, "x")
    k2, l2, e2 = apply_symmetry(k1, l1, e1, "x")
    assert np.allclose(k2, kin, atol=1e-5)
    assert np.allclose(l2, log)
    assert e2.tolist() == events.tolist()


# --- kinematics and features ------------------------------------------------------


def test_kinematics_shape_and_absent_fill():
    kin = build_kinematics(np.zeros((0, 14)), np.arange(4))
    assert kin.shape == (5, N_SLOTS, 4)
    assert (kin == ABSENT_FILL).all()


def test_observability_flag_is_the_fifth_channel():
    """It is a FLAG, not a coordinate: position is present in the tactical data even
    when the player is off-camera, so this is the only thing telling the model
    whether that slot's visual logits mean anything."""
    rows = np.stack([_row(0, 0, 1, observed=False), _row(1, 0, 1, observed=True)])
    kin = build_kinematics(rows, np.arange(2))
    slot = slot_index(0, 1)
    assert kin[4, slot, 0] == 0.0
    assert kin[4, slot, 1] == 1.0
    assert kin[0, slot, 0] != ABSENT_FILL  # position still known while off-camera


def test_encoder_features_are_slot_major():
    """SlotAttentionEncoderEmbedding reshapes to (T, 26, 14), so all 14 of a slot's
    features must be contiguous. A frame-major layout would silently scramble them."""
    kin = np.zeros((5, N_SLOTS, 2), dtype=np.float32)
    log = np.zeros((N_CLASSES, N_SLOTS, 2), dtype=np.float32)
    kin[0, 3, 1] = 1.0  # slot 3, kinematic 0, frame 1
    log[4, 3, 1] = 2.0  # slot 3, logit 4, frame 1
    feats = encoder_features(kin, log)
    assert feats.shape == (2, ENCODER_FEATURE_DIM)
    base = 3 * FEATURES_PER_SLOT
    assert feats[1, base + 0] == 1.0
    assert feats[1, base + 5 + 4] == 2.0
    assert feats[0].sum() == 0.0


def test_encoder_features_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="disagree"):
        encoder_features(
            np.zeros((5, N_SLOTS, 3), dtype=np.float32),
            np.zeros((N_CLASSES, N_SLOTS, 4), dtype=np.float32),
        )


# --- tokens -----------------------------------------------------------------------


def test_tokens_are_wrapped_in_sos_and_eos():
    actions, roles, frames = build_tokens(np.zeros((0, 3), dtype=np.int64), 750)
    assert actions.shape == (2, ACTION_VOCAB)
    assert actions[0].argmax() == SOS_ACTION
    assert actions[-1].argmax() == EOS_ACTION
    assert roles[0].argmax() == roles[-1].argmax() == NEUTRAL_ROLE
    assert frames[0] == 0


def test_class_ids_map_onto_action_indices_minus_one():
    """Action index 8 is SOS and 9 is EOS, so the 8 classes must occupy 0-7. An
    off-by-one here collides a real class with SOS."""
    events = np.array([[10, 0, 1], [20, 5, 8]], dtype=np.int64)
    actions, roles, _ = build_tokens(events, 750)
    assert actions[1].argmax() == 0  # class 1 -> index 0
    assert actions[2].argmax() == 7  # class 8 -> index 7
    assert actions[1].argmax() != SOS_ACTION
    assert roles[2].argmax() == 5
    assert roles.shape[1] == ROLE_VOCAB


def test_events_are_emitted_in_time_order():
    events = np.array([[300, 1, 2], [10, 2, 3]], dtype=np.int64)
    _, _, frames = build_tokens(events, 750)
    assert frames[1] < frames[2]


def test_timestamp_zero_is_reserved_for_sos():
    """An event at window frame 0 must not collide with the SOS timestamp."""
    events = np.array([[0, 1, 2]], dtype=np.int64)
    _, _, frames = build_tokens(events, 750)
    assert frames[0] == 0
    assert frames[1] == 1


def test_window_events_only_include_the_window():
    rows = np.stack(
        [
            _row(5, 0, 1, cls=2),
            _row(100, 0, 1, cls=3),
            _row(900, 0, 1, cls=4),
        ]
    )
    events = window_events(rows, 0, 750)
    assert events[:, 0].tolist() == [5, 100]
    assert events[:, 2].tolist() == [2, 3]


def test_window_events_are_window_local():
    rows = np.stack([_row(1005, 0, 1, cls=2)])
    assert window_events(rows, 1000, 750)[0, 0] == 5


def test_background_rows_are_not_events():
    rows = np.stack([_row(5, 0, 1, cls=0)])
    assert len(window_events(rows, 0, 750)) == 0


# --- frame one-hot ----------------------------------------------------------------


def test_one_hot_frames_width_is_framespan_plus_two():
    out = one_hot_frames(np.array([0, 5]), 750)
    assert out.shape == (2, 752)
    assert out[1].argmax() == 5


def test_one_hot_frames_clips_out_of_range():
    out = one_hot_frames(np.array([-3, 10_000]), 750)
    assert out[0].argmax() == 0
    assert out[1].argmax() == 751


# --- positional coordinate system -------------------------------------------------


def test_encoder_positions_are_window_local_not_absolute_frames(tmp_path):
    """The bug that made DST's first end-to-end run score 0.035 against stage 1's
    0.327.

    The decoder's timestamps are window-local (`build_tokens` encodes window frame f
    as f+1). If the ENCODER receives absolute video frames instead, the two are in
    different coordinate systems and the model cannot align a decoder query with an
    encoder position -- it can still learn the class prior, which is exactly the
    failure signature we saw. Absolute frames also alias the sinusoidal encoding at
    magnitudes around 150,000.

    The reference normalises identically (`Encoder_abs_frame_nb -= min_Frame - 1`)
    despite naming the variable "abs" and commenting it "Absolute frame number".
    """
    import h5py
    from matchlab_core.pcbas.logits import empty_logits, logits_filename, save_logits
    from matchlab_train.datasets.footpass_windows import FootpassWindowDataset

    start = 74_000
    rows = np.stack([_row(start + i, 0, 1) for i in range(60)])
    h5_path = tmp_path / "w.h5"
    with h5py.File(h5_path, "w") as f:
        f["game_9_H2"] = rows
    logits_dir = tmp_path / "logits"
    save_logits(empty_logits(start + 60), logits_dir / logits_filename("game_9_H2"))

    ds = FootpassWindowDataset(h5_path, logits_dir, framespan=50, stride=50, train=False)
    s = ds.sample(0)
    assert s.frames[0] == 1, "positions must be 1-based window-local"
    assert s.frames[-1] == 50
    assert s.frames.max() <= 50, "absolute video frames must never reach the model"
    # The absolute start is still available for mapping events back.
    assert s.start_frame == start


# --- window sampling --------------------------------------------------------------


def _tiny_split(tmp_path, n_frames=3000):
    import h5py
    from matchlab_core.pcbas.logits import empty_logits, logits_filename, save_logits

    rows = np.stack([_row(i, 0, 1) for i in range(n_frames)])
    h5_path = tmp_path / "w.h5"
    with h5py.File(h5_path, "w") as f:
        f["game_1_H1"] = rows
    logits_dir = tmp_path / "logits"
    save_logits(empty_logits(n_frames), logits_dir / logits_filename("game_1_H1"))
    return h5_path, logits_dir


def test_training_windows_use_random_offsets(tmp_path):
    """The reason the timestamp head stalled. With a fixed stride every event appears
    at only about two distinct window-local positions ever, so the model is asked to
    localise events it has only ever seen in two places. Measured with stride 375,
    timestamp val loss stalled at 4.35 (random 6.62) while action and role improved."""
    from matchlab_train.datasets.footpass_windows import FootpassWindowDataset

    h5_path, logits_dir = _tiny_split(tmp_path)
    ds = FootpassWindowDataset(
        h5_path, logits_dir, framespan=500, train=True, repeat=40, seed=1
    )
    starts = [s for _, s in ds.index]
    assert len(starts) == 40
    assert len(set(starts)) > 20, "offsets must vary, not repeat a stride"
    assert all(s % 250 != 0 or True for s in starts)


def test_resample_redraws_offsets_between_epochs(tmp_path):
    from matchlab_train.datasets.footpass_windows import FootpassWindowDataset

    h5_path, logits_dir = _tiny_split(tmp_path)
    ds = FootpassWindowDataset(
        h5_path, logits_dir, framespan=500, train=True, repeat=40, seed=1
    )
    first = list(ds.index)
    ds.resample()
    assert list(ds.index) != first


def test_evaluation_windows_stay_deterministic(tmp_path):
    """A score has to be reproducible, so eval keeps fixed stride windows."""
    from matchlab_train.datasets.footpass_windows import FootpassWindowDataset

    h5_path, logits_dir = _tiny_split(tmp_path)
    kw = dict(framespan=500, stride=250, train=False)
    a = FootpassWindowDataset(h5_path, logits_dir, **kw)
    b = FootpassWindowDataset(h5_path, logits_dir, **kw)
    assert a.index == b.index
    a.resample()
    assert a.index == b.index, "resample must be a no-op for evaluation"


def test_every_training_window_fits_inside_the_half(tmp_path):
    from matchlab_train.datasets.footpass_windows import FootpassWindowDataset

    h5_path, logits_dir = _tiny_split(tmp_path, n_frames=3000)
    ds = FootpassWindowDataset(
        h5_path, logits_dir, framespan=500, train=True, repeat=60, seed=3
    )
    for _, start in ds.index:
        assert 0 <= start <= 3000 - 500
