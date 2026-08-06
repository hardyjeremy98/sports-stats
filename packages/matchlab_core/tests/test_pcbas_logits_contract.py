from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.pcbas.logits import (
    LOGITS_DTYPE,
    WindowAccumulator,
    empty_logits,
    load_logits,
    logits_filename,
    save_logits,
    validate_logits,
    window_starts,
)


def test_empty_logits_shape_and_dtype():
    a = empty_logits(100)
    assert a.shape == (9, 26, 100)
    assert a.dtype == LOGITS_DTYPE


@pytest.mark.parametrize(
    "array,match",
    [
        (np.zeros((9, 26), dtype=np.float16), "3-D"),
        (np.zeros((9, 22, 10), dtype=np.float16), r"\(9, 26, T\)"),
        (np.zeros((8, 26, 10), dtype=np.float16), r"\(9, 26, T\)"),
        (np.zeros((9, 26, 10), dtype=np.float32), "float16"),
    ],
)
def test_contract_violations_are_rejected(array, match):
    """The contract is frozen. A 22-slot array would be a natural mistake -- VAL
    only ever occupies 22 slots -- and would silently misalign every later slot."""
    with pytest.raises(ValueError, match=match):
        validate_logits(array)


def test_round_trips_through_disk(tmp_path):
    a = empty_logits(20)
    a[2, 5, 7] = 3.5
    save_logits(a, tmp_path / "x.npy")
    b = load_logits(tmp_path / "x.npy")
    assert b.dtype == LOGITS_DTYPE
    assert b[2, 5, 7] == 3.5


def test_filename_matches_the_reference():
    assert logits_filename("game_18_H1") == "avg_logits_game_18_H1.npy"


# --- window planning --------------------------------------------------------------


def test_windows_tile_a_short_sequence_with_one_window():
    assert window_starts(0, 30, 50, 25) == [0]


def test_windows_never_run_past_the_last_frame():
    """MatchVideo.read_clip raises past the end, deliberately. The final window is
    pulled back to land exactly on the last frame."""
    starts = window_starts(0, 119, 50, 25)
    assert max(starts) + 50 - 1 <= 119
    assert max(starts) == 70


def test_windows_cover_every_frame():
    starts = window_starts(100, 399, 50, 25)
    covered = set()
    for s in starts:
        covered.update(range(s, s + 50))
    assert set(range(100, 400)) <= covered


def test_window_starts_respect_the_first_frame():
    assert window_starts(1000, 1200, 50, 25)[0] == 1000


def test_empty_range_gives_no_windows():
    assert window_starts(50, 10, 50, 25) == []


# --- overlap averaging ------------------------------------------------------------


def test_overlapping_windows_are_averaged():
    acc = WindowAccumulator(75)
    acc.add(np.full((9, 26, 50), 2.0, dtype=np.float32), 0)
    acc.add(np.full((9, 26, 50), 4.0, dtype=np.float32), 25)
    out = acc.result()
    assert out[0, 0, 0] == pytest.approx(2.0)  # covered once
    assert out[0, 0, 30] == pytest.approx(3.0)  # covered twice -> mean
    assert out[0, 0, 70] == pytest.approx(4.0)  # covered once


def test_edge_frames_are_not_halved():
    """The reference divides the whole overlap region by 2. Frames covered by only
    one window must be divided by 1, or every half arrives at the denoiser with
    systematically under-confident first and last 25 frames."""
    acc = WindowAccumulator(75)
    acc.add(np.full((9, 26, 50), 6.0, dtype=np.float32), 0)
    acc.add(np.full((9, 26, 50), 6.0, dtype=np.float32), 25)
    out = acc.result()
    assert out[0, 0, 0] == pytest.approx(6.0)
    assert out[0, 0, 74] == pytest.approx(6.0)
    assert out[0, 0, 40] == pytest.approx(6.0)


def test_uncovered_frames_stay_zero_and_are_reported():
    acc = WindowAccumulator(100)
    acc.add(np.full((9, 26, 50), 5.0, dtype=np.float32), 0)
    assert acc.coverage[80] == 0
    assert acc.result()[0, 0, 80] == 0.0


def test_offset_lets_a_half_start_at_an_absolute_frame():
    acc = WindowAccumulator(50, offset=1000)
    acc.add(np.full((9, 26, 50), 1.0, dtype=np.float32), 1000)
    assert acc.result()[0, 0, 0] == pytest.approx(1.0)


def test_window_outside_the_range_raises():
    acc = WindowAccumulator(50, offset=1000)
    with pytest.raises(ValueError, match="outside"):
        acc.add(np.zeros((9, 26, 50), dtype=np.float32), 999)
