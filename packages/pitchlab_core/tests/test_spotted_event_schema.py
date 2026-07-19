"""TDD for the SpottedEvent schema (SPO-46): the contract requires the JSON
key be literally "class", not "class_" — the Python-side field name forced
by "class" being a reserved word."""

from __future__ import annotations

import json

from pitchlab_core.schemas.spotting import SpottedEvent


def test_model_dump_json_emits_class_key():
    event = SpottedEvent(class_="PASS", frame_idx=42, t=1.68, confidence=0.9, half=1)

    dumped = json.loads(event.model_dump_json())

    assert dumped["class"] == "PASS"
    assert "class_" not in dumped


def test_model_dump_emits_class_key_even_without_by_alias():
    event = SpottedEvent(class_="SHOT", frame_idx=0, t=0.0, confidence=0.5, half=None)

    dumped = event.model_dump()

    assert dumped == {
        "class": "SHOT",
        "frame_idx": 0,
        "t": 0.0,
        "confidence": 0.5,
        "half": None,
    }


def test_model_validate_accepts_class_key_from_json():
    raw = {"class": "DRIVE", "frame_idx": 7, "t": 0.28, "confidence": 0.75, "half": None}

    event = SpottedEvent.model_validate(raw)

    assert event.class_ == "DRIVE"


def test_populate_by_name_allows_constructing_with_class_underscore():
    event = SpottedEvent(class_="TACKLE", frame_idx=1, t=0.04, confidence=0.6, half=2)

    assert event.class_ == "TACKLE"
