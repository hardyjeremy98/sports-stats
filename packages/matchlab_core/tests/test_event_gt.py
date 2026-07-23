"""TDD for the event ground-truth representation (SPO-47): `GroundTruthEvent`
/ `EventGroundTruth`, `load_event_ground_truth`, and `is_event_ground_truth`.

Distinct from the track/box-centric `GroundTruth` in `matchlab_core.gt` --
these model purely temporal action-spotting ground truth (a class + a time +
a half, no boxes/tracks). `class` is a Python keyword, so the field is named
`class_` with `alias="class"` -- the same convention as
`matchlab_core.schemas.spotting.SpottedEvent` (Task 2) -- so the on-disk key
is literally "class".
"""

from __future__ import annotations

import json

from matchlab_core.event_gt import (
    EventGroundTruth,
    GroundTruthEvent,
    is_event_ground_truth,
    load_event_ground_truth,
)
from matchlab_core.gt import GroundTruth


def test_ground_truth_event_model_dump_emits_class_key():
    event = GroundTruthEvent(class_="PASS", frame_idx=309, t=12.34, half=1)

    dumped = event.model_dump()

    assert dumped == {"class": "PASS", "frame_idx": 309, "t": 12.34, "half": 1}


def test_ground_truth_event_model_validate_accepts_class_key():
    event = GroundTruthEvent.model_validate(
        {"class": "SHOT", "frame_idx": 0, "t": 0.0, "half": None}
    )

    assert event.class_ == "SHOT"


def test_ground_truth_event_populate_by_name_allows_class_underscore():
    event = GroundTruthEvent(class_="DRIVE", frame_idx=1, t=0.04, half=2)

    assert event.class_ == "DRIVE"


def test_event_ground_truth_requires_kind_discriminator():
    gt = EventGroundTruth(source="soccernet-ball", fps=25.0, events=[])

    assert gt.kind == "action_events"


def test_event_ground_truth_round_trip_json_key_is_literally_class():
    gt = EventGroundTruth(
        source="soccernet-ball",
        sequence="SNMOT-BALL-001",
        fps=25.0,
        events=[GroundTruthEvent(class_="PASS", frame_idx=309, t=12.34, half=1)],
    )

    dumped = json.loads(gt.model_dump_json())

    assert dumped["kind"] == "action_events"
    assert dumped["events"][0]["class"] == "PASS"
    assert "class_" not in dumped["events"][0]


def test_load_event_ground_truth_reloads_faithfully(tmp_path):
    gt = EventGroundTruth(
        source="soccernet-ball",
        sequence="SNMOT-BALL-001",
        fps=25.0,
        events=[
            GroundTruthEvent(class_="PASS", frame_idx=309, t=12.34, half=1),
            GroundTruthEvent(class_="SHOT", frame_idx=50, t=2.0, half=2),
        ],
    )
    path = tmp_path / "SNMOT-BALL-001.gt.json"
    path.write_text(gt.model_dump_json())

    reloaded = load_event_ground_truth(path)

    assert reloaded == gt


def test_is_event_ground_truth_true_for_event_gt_file(tmp_path):
    gt = EventGroundTruth(source="soccernet-ball", fps=25.0, events=[])
    path = tmp_path / "event.gt.json"
    path.write_text(gt.model_dump_json())

    assert is_event_ground_truth(path) is True


def test_is_event_ground_truth_false_for_track_gt_file(tmp_path):
    track_gt = GroundTruth(source="soccernet-tracking", sequence="SNMOT-116", fps=25.0)
    path = tmp_path / "track.gt.json"
    path.write_text(track_gt.model_dump_json())

    assert is_event_ground_truth(path) is False


def test_is_event_ground_truth_false_for_absent_file_does_not_raise(tmp_path):
    assert is_event_ground_truth(tmp_path / "nope.gt.json") is False


def test_is_event_ground_truth_false_for_nonsense_json_does_not_raise(tmp_path):
    path = tmp_path / "garbage.gt.json"
    path.write_text("not json at all {{{")

    assert is_event_ground_truth(path) is False


def test_is_event_ground_truth_false_for_unrelated_json_does_not_raise(tmp_path):
    path = tmp_path / "unrelated.json"
    path.write_text(json.dumps({"hello": "world"}))

    assert is_event_ground_truth(path) is False
