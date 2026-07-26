from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.pcbas.schema import CLS
from matchlab_train.datasets.footpass_pcbas import (
    half_to_events,
    half_to_tracklets,
    roster_lookup,
)


def _row(frame, pid, ltr, shirt, role, x, y, vx, vy, roi, cls):
    r = np.full(14, np.nan, dtype=np.float64)
    r[0], r[1], r[2], r[3], r[4] = frame, pid, ltr, shirt, role
    r[5], r[6], r[7], r[8] = x, y, vx, vy
    if roi is not None:
        r[9], r[10], r[11], r[12] = roi
    r[13] = cls
    return r


def _arr():
    rows = []
    for f in range(4):
        rows.append(_row(f, 100, 0, 7, 1, 0.5, 0.5, 0.0, 0.0, (960, 540, 60, 120), 0))
        rows.append(_row(f, 101, 1, 9, 4, 0.6, 0.4, 0.1, 0.0, None, 0))  # off-screen
    rows[2][CLS] = 2  # frame 1, player 100 -> a pass
    return np.stack(rows)


def test_tracklets_only_include_observed_boxes():
    tracklets, _ = half_to_tracklets(_arr())
    by_id = {t.tracklet_id: t for t in tracklets}
    assert len(by_id[100].frames) == 4  # on-screen every frame
    assert 101 not in by_id or not by_id[101].frames  # never on-screen -> no boxes


def test_boxes_are_scaled_to_352x640():
    tracklets, _ = half_to_tracklets(_arr())
    box = {t.tracklet_id: t for t in tracklets}[100].frames[0].box
    assert box.x1 == pytest.approx(960 / 3.0, abs=1.0)
    assert box.y1 == pytest.approx(540 / 3.068181, abs=1.0)
    assert box.width == pytest.approx(60 / 3.0, abs=1.0)
    assert box.height == pytest.approx(120 / 3.068181, abs=1.0)


def test_team_assignment_is_emitted_for_every_tracklet():
    tracklets, teams = half_to_tracklets(_arr())
    assert {t.tracklet_id for t in tracklets} == {a.tracklet_id for a in teams}
    assert {a.team for a in teams} == {"home"}  # only the left side is ever on-screen


def test_events_extracted_with_slot_and_shirt():
    gt = half_to_events(_arr(), "game_1_H1")
    assert len(gt.events) == 1
    ev = gt.events[0]
    assert ev.frame_idx == 1
    assert ev.class_id == 2  # pass
    assert ev.slot == 0  # left team, role 1
    assert ev.shirt_number == 7


def test_half_key_is_parsed():
    gt = half_to_events(_arr(), "game_1_H2")
    assert (gt.key, gt.game_id, gt.half) == ("game_1_H2", "game_1", 2)


def test_background_rows_are_not_events():
    arr = _arr()
    arr[:, CLS] = 0
    assert half_to_events(arr, "k").events == []


def test_roster_lookup_maps_frame_and_slot_to_shirt():
    lut = roster_lookup(_arr())
    assert lut[(0, 0)] == 7
    assert lut[(0, 13 + 3)] == 9  # right team, role 4 -> slot 16


def test_off_screen_rows_still_produce_roster_and_events():
    """The 17.5% of actions on off-screen players must NOT be dropped -- they are
    exactly the cases only the sequence stage can recover."""
    arr = _arr()
    arr[1][CLS] = 5  # frame 0, player 101 (no bbox) -> a shot
    gt = half_to_events(arr, "k")
    assert any(e.class_id == 5 and e.slot == 16 for e in gt.events)


def test_unlabelled_split_is_rejected():
    with pytest.raises(ValueError, match="13 columns"):
        half_to_events(np.zeros((3, 13)), "challenge_key")


def test_events_carry_a_timestamp_from_fps():
    gt = half_to_events(_arr(), "game_1_H1", fps=25.0)
    assert gt.events[0].t == pytest.approx(1 / 25.0)


def test_downcast_to_event_ground_truth_drops_identity():
    """`EventGroundTruth` has no player field (spec gap). The downcast must be
    lossy-but-honest, not silently inventing one."""
    egt = half_to_events(_arr(), "game_1_H1").to_event_ground_truth()
    assert egt.kind == "action_events"
    assert [e.class_ for e in egt.events] == ["pass"]
    assert egt.events[0].half == 1
