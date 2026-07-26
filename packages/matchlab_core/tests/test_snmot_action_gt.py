"""B3: SoccerNet-tracking clips carry a labelled action (class + timestamp) in
gameinfo.ini. Parsing it turns the already-downloaded tracking tier into a
sparse action-spotting benchmark. Hand-built ini files so each assertion is
known by construction.
"""

from __future__ import annotations

import pytest
from matchlab_core.snmot_action_gt import (
    BALL_CONTACT_CLASSES,
    load_snmot_action_gt,
)

INI = """[Sequence]
name=SNMOT-116
actionPosition=975293
actionClass=Corner
visibility=visible
clipStart=969000
clipStop=999000
num_tracklets=2
trackletID_1= player team left;4
trackletID_2= ball;1
"""

SEQINFO = """[Sequence]
name=SNMOT-116
imDir=img1
frameRate=25
seqLength=750
imWidth=1920
imHeight=1080
"""


def _seq(tmp_path, ini=INI, seqinfo=SEQINFO, name="SNMOT-116"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "gameinfo.ini").write_text(ini)
    (d / "seqinfo.ini").write_text(seqinfo)
    return d


def test_action_frame_is_offset_from_clip_start(tmp_path):
    gt = load_snmot_action_gt(_seq(tmp_path))
    # (975293 - 969000) ms = 6.293 s -> 6.293 * 25 fps = 157.3 -> frame 157
    assert len(gt.events) == 1
    assert gt.events[0].frame_idx == 157
    assert gt.events[0].t == pytest.approx(6.293, abs=1e-3)
    assert gt.events[0].class_ == "Corner"


def test_kind_and_fps_are_carried(tmp_path):
    gt = load_snmot_action_gt(_seq(tmp_path))
    assert gt.kind == "action_events"
    assert gt.fps == 25.0
    assert gt.sequence == "SNMOT-116"
    assert gt.source == "soccernet-tracking"


def test_action_not_shown_yields_no_events(tmp_path):
    """visibility=not shown means the labelled action is not in the clip, so it
    must not become a ground-truth event -- scoring against it would penalise a
    spotter for missing something that is not there."""
    ini = INI.replace("visibility=visible", "visibility=not shown")
    gt = load_snmot_action_gt(_seq(tmp_path, ini=ini))
    assert gt.events == []


def test_missing_action_fields_yield_no_events(tmp_path):
    ini = "\n".join(ln for ln in INI.splitlines() if not ln.startswith("actionPosition"))
    gt = load_snmot_action_gt(_seq(tmp_path, ini=ini))
    assert gt.events == []


def test_action_outside_the_clip_window_is_dropped(tmp_path):
    """A timestamp before clipStart or after clipStop cannot be a frame in this
    clip; emitting it would create an unreachable ground-truth event."""
    ini = INI.replace("actionPosition=975293", "actionPosition=1200000")
    gt = load_snmot_action_gt(_seq(tmp_path, ini=ini))
    assert gt.events == []


def test_ball_contact_classes_are_declared_and_exclude_non_ball_actions():
    assert "Corner" in BALL_CONTACT_CLASSES
    assert "Shots on target" in BALL_CONTACT_CLASSES
    for non_ball in ("Yellow card", "Substitution", "Offside", "Red card"):
        assert non_ball not in BALL_CONTACT_CLASSES
