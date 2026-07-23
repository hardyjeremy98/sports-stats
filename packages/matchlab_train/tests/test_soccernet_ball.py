"""TDD for the SoccerNet Ball Action Spotting label parser (SPO-47):
`load_soccernet_ball_labels` converts a `Labels-ball.json` file into an
`EventGroundTruth` (matchlab_core.event_gt). Pure/fixture-tested, no
network -- the fixture is a tiny hand-written `Labels-ball.json` matching
the real format's documented shape:

    {"UrlLocal": "...",
     "annotations": [
       {"gameTime": "1 - 00:12", "label": "PASS", "position": "12340",
        "team": "...", "visibility": "visible"}, ...]}

`position` is milliseconds from the start of the half; `gameTime` is
"<half> - MM:SS". Conversion: `half = int(gameTime.split("-")[0])`,
`t = position_ms / 1000.0`, `frame_idx = round(t * fps)` using round-half-up
(not Python's banker's-rounding `round()` -- see the numeric assertion
below, where a value landing exactly on .5 must round up, not to even).
"""

from __future__ import annotations

import json
from pathlib import Path

from matchlab_train.datasets.soccernet_ball import load_soccernet_ball_labels

FIXTURE = {
    "UrlLocal": "england_epl/2015-2016/test-match/",
    "annotations": [
        {
            "gameTime": "1 - 00:12",
            "label": "PASS",
            "position": "12340",
            "team": "home",
            "visibility": "visible",
        },
        {
            "gameTime": "1 - 00:45",
            "label": "DRIVE",
            "position": "45000",
            "team": "away",
            "visibility": "visible",
        },
        {
            "gameTime": "2 - 00:02",
            "label": "SHOT",
            "position": "2000",
            "team": "home",
            "visibility": "not shown",
        },
    ],
}


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "Labels-ball.json"
    path.write_text(json.dumps(FIXTURE))
    return path


def test_load_soccernet_ball_labels_event_count_and_source(tmp_path):
    gt = load_soccernet_ball_labels(_write_fixture(tmp_path), fps=25, sequence="test-match")

    assert gt.source == "soccernet-ball"
    assert gt.sequence == "test-match"
    assert gt.fps == 25
    assert gt.kind == "action_events"
    assert len(gt.events) == 3


def test_load_soccernet_ball_labels_position_12340_at_fps25_rounds_half_up(tmp_path):
    """The spec's worked example: position "12340" ms @ fps 25 -> t=12.34,
    frame_idx=309. Note t*fps == 308.5 EXACTLY here, so this also pins down
    that rounding must be half-up (309), not Python's default
    round-half-to-even (which would give 308)."""
    gt = load_soccernet_ball_labels(_write_fixture(tmp_path), fps=25, sequence=None)

    first = gt.events[0]
    assert first.class_ == "PASS"
    assert first.half == 1
    assert first.t == 12.34
    assert first.frame_idx == 309


def test_load_soccernet_ball_labels_half_parsed_from_game_time(tmp_path):
    gt = load_soccernet_ball_labels(_write_fixture(tmp_path), fps=25)

    assert [e.half for e in gt.events] == [1, 1, 2]


def test_load_soccernet_ball_labels_class_values_verbatim(tmp_path):
    gt = load_soccernet_ball_labels(_write_fixture(tmp_path), fps=25)

    assert [e.class_ for e in gt.events] == ["PASS", "DRIVE", "SHOT"]


def test_load_soccernet_ball_labels_t_and_frame_idx_for_clean_values(tmp_path):
    gt = load_soccernet_ball_labels(_write_fixture(tmp_path), fps=25)

    second, third = gt.events[1], gt.events[2]
    assert second.t == 45.0
    assert second.frame_idx == 1125  # 45.0 * 25, exact
    assert third.t == 2.0
    assert third.frame_idx == 50  # 2.0 * 25, exact


def test_load_soccernet_ball_labels_keeps_events_regardless_of_visibility(tmp_path):
    """Documented default: visibility is NOT filtered on -- "not shown"
    events are kept, same as "visible" ones."""
    gt = load_soccernet_ball_labels(_write_fixture(tmp_path), fps=25)

    assert len(gt.events) == 3
    assert gt.events[2].class_ == "SHOT"  # the "not shown" one, still present


def test_load_soccernet_ball_labels_tolerates_missing_optional_keys(tmp_path):
    minimal = {
        "annotations": [
            {"gameTime": "1 - 00:00", "label": "PASS", "position": "0"},
        ]
    }
    path = tmp_path / "Labels-ball.json"
    path.write_text(json.dumps(minimal))

    gt = load_soccernet_ball_labels(path, fps=25)

    assert len(gt.events) == 1
    assert gt.events[0].t == 0.0
    assert gt.events[0].frame_idx == 0
