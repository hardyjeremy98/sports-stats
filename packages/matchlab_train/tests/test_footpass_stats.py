from __future__ import annotations

import numpy as np
import pytest
from matchlab_train.datasets.footpass_pcbas import split_stats

h5py = pytest.importorskip("h5py")


def _row(frame, pid, ltr, shirt, role, roi, cls):
    r = np.full(14, np.nan, dtype=np.float64)
    r[0], r[1], r[2], r[3], r[4] = frame, pid, ltr, shirt, role
    r[5], r[6], r[7], r[8] = 0.5, 0.5, 0.0, 0.0
    if roi is not None:
        r[9], r[10], r[11], r[12] = roi
    r[13] = cls
    return r


@pytest.fixture
def tiny_h5(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as f:
        # H1: player 100 on-screen (a pass), player 101 off-screen (a shot)
        f["game_1_H1"] = np.stack(
            [
                _row(0, 100, 0, 7, 1, (960, 540, 60, 120), 0),
                _row(1, 100, 0, 7, 1, (960, 540, 60, 120), 2),
                _row(0, 101, 1, 9, 4, None, 0),
                _row(1, 101, 1, 9, 4, None, 5),
            ]
        )
        f["game_1_H2"] = np.stack([_row(2, 100, 0, 7, 1, (10, 10, 5, 5), 2)])
    return path


def test_counts_halves_rows_and_events(tiny_h5):
    s = split_stats(tiny_h5)
    assert s.n_halves == 2
    assert s.keys == ["game_1_H1", "game_1_H2"]
    assert s.n_rows == 5
    assert s.n_events == 3


def test_per_class_counts_use_class_names(tiny_h5):
    s = split_stats(tiny_h5)
    assert s.events_per_class == {"pass": 2, "shot": 1}


def test_bbox_fractions_are_reported_separately_for_rows_and_events(tiny_h5):
    """Rows and events have different bbox rates -- 59% of ROWS lack a box but only
    17.5% of EVENTS do. Conflating them would misstate the visual ceiling."""
    s = split_stats(tiny_h5)
    assert s.rows_with_bbox == 3
    assert s.frac_rows_with_bbox == pytest.approx(3 / 5)
    assert s.events_with_bbox == 2
    assert s.frac_events_with_bbox == pytest.approx(2 / 3)


def test_slots_seen_are_within_range(tiny_h5):
    s = split_stats(tiny_h5)
    assert s.slots_seen == [0, 16]
    assert all(0 <= v <= 25 for v in s.slots_seen)


def test_unlabelled_split_reports_no_events(tmp_path):
    """CHALLENGE has 13 columns and no labels. Stats must say so, not crash and not
    silently report zero events as if the split were empty."""
    path = tmp_path / "chal.h5"
    with h5py.File(path, "w") as f:
        f["game_9_H1"] = np.zeros((4, 13))
    s = split_stats(path)
    assert s.labelled is False
    assert s.n_events == 0
    assert s.n_rows == 4
