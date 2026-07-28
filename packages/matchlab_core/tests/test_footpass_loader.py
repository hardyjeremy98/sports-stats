"""FOOTPASS tactical loader: observability-based fragmentation.

The loader's whole job for B2 is turning `ROI_X is NaN` (player not visible in
frame) into tracklet-shaped spans, so these tests pin that behaviour precisely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matchlab_train.datasets.footpass import COL, FootpassHalf, load_half, observable_spans

NAN = float("nan")


def _rows(visible_by_frame: dict[int, bool], player_id: int = 100, role: int = 2) -> list[list]:
    return [
        [f, player_id, 0, 7, role, 0.1, 0.5, 0.0, 0.0, (1.0 if vis else NAN), 0.0, 0.0, 0.0, 0.0]
        for f, vis in sorted(visible_by_frame.items())
    ]


def _half(rows: list[list]) -> FootpassHalf:
    return FootpassHalf(game_id="game_1", half=1, rows=np.asarray(rows, dtype=np.float32))


def test_observable_spans_splits_on_offcamera_gap():
    half = _half(_rows({f: (f <= 2 or f >= 10) for f in range(12)}))
    assert observable_spans(half, 100, max_gap_frames=2) == [(0, 2), (10, 11)]


def test_observable_spans_bridges_gap_within_tolerance():
    half = _half(_rows({f: f != 3 for f in range(6)}))
    assert observable_spans(half, 100, max_gap_frames=2) == [(0, 5)]


def test_observable_spans_is_empty_when_never_visible():
    half = _half(_rows({f: False for f in range(6)}))
    assert observable_spans(half, 100) == []


def test_observable_spans_ignores_other_players():
    rows = _rows({f: True for f in range(4)}, player_id=100)
    rows += _rows({f: True for f in range(4)}, player_id=101)
    assert observable_spans(_half(rows), 100) == [(0, 3)]


def test_col_indices_match_documented_schema():
    assert (COL.FRAME, COL.PLAYER_ID, COL.TEAM, COL.ROLE) == (0, 1, 2, 4)
    assert (COL.X, COL.Y, COL.ROI_X) == (5, 6, 9)


def test_half_exposes_player_role_and_team():
    half = _half(_rows({f: True for f in range(3)}, player_id=100, role=11))
    assert half.player_ids == [100]
    assert half.role_of(100) == 11
    assert half.team_of(100) == 0


FOOTPASS_VAL = Path("data/footpass/tactical/val_tactical_data.h5")


@pytest.mark.skipif(not FOOTPASS_VAL.exists(), reason="FOOTPASS tactical data not downloaded")
def test_real_val_half_matches_documented_schema():
    half = load_half(FOOTPASS_VAL, "game_18_H1")
    assert half.rows.shape[1] == 14
    assert len(half.player_ids) >= 22
    spans = observable_spans(half, half.player_ids[0])
    assert len(spans) > 1, "a broadcast half must fragment the player at least once"
