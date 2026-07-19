"""Ground-truth timestamped action-event labels (SPO-47).

Deliberately a sibling of `pitchlab_core.gt`, not an addition to it:
`gt.GroundTruth` is track/box-centric (a sequence of frame boxes per
`track_id`) while this models purely temporal action-spotting ground truth
(a class + a time + a half, no boxes/tracks at all) -- different enough in
shape and in what it is scored against (avg-mAP over event times, a later
task, vs. IDF1/HOTA/purity over boxes) that folding it into `GroundTruth`
would mean most of that model's fields never apply. Keeping `gt.py`'s
existing track models untouched avoids disturbing their current callers
(`load_soccernet_sequence` et al., the worker's box-GT auto-scoring path).

A video's `videos.gt_path` can point at either kind of JSON file today (a
track `GroundTruth` or this `EventGroundTruth`) -- `EventGroundTruth.kind`
is a required discriminator ("action_events") so a caller that only has the
path (e.g. the worker deciding which evaluator to run) can tell them apart
without heuristics; `is_event_ground_truth` does exactly that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GroundTruthEvent(BaseModel):
    """One ground-truth action event. `class` is a Python keyword, so the
    field is named `class_` with `alias="class"` -- the same convention as
    `pitchlab_core.schemas.spotting.SpottedEvent` (Task 2's spotting
    exchange contract), so ground truth and predictions share one on-disk
    vocabulary for the "class" key. `populate_by_name` lets Python callers
    construct with `class_=...`; `serialize_by_alias` guarantees
    `model_dump`/`model_dump_json` always emit the literal JSON key "class",
    regardless of the `by_alias` flag a caller passes (or omits)."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    class_: str = Field(alias="class")
    frame_idx: int
    t: float
    half: int | None = None


class EventGroundTruth(BaseModel):
    """Ground truth for one video's worth of timestamped action events.

    `kind` is a required discriminator, always `"action_events"` -- see the
    module docstring for why it exists. A track `GroundTruth` file has no
    `kind` field at all (and has `tracks`, not `events`), so `kind`'s
    presence/value is a reliable, cheap way to distinguish the two without
    fully validating either schema first; see `is_event_ground_truth`.
    """

    source: str
    sequence: str | None = None
    fps: float
    kind: Literal["action_events"] = "action_events"
    events: list[GroundTruthEvent] = []


def load_event_ground_truth(path: str | Path) -> EventGroundTruth:
    """Load and validate an `EventGroundTruth` JSON file.

    Assumes the caller already knows `path` is event ground truth (e.g. it
    passed `is_event_ground_truth`) -- raises via pydantic if the content
    isn't valid JSON or doesn't match the schema, it does not degrade
    gracefully the way `is_event_ground_truth` does.
    """
    return EventGroundTruth.model_validate_json(Path(path).read_text())


def is_event_ground_truth(path_or_obj: str | Path | dict[str, Any]) -> bool:
    """True if `path_or_obj` looks like an `EventGroundTruth`: a `kind` field
    equal to `"action_events"` and an `events` list. Accepts either a path to
    a JSON file or an already-parsed dict. False (never raises) for a track
    `GroundTruth` file (no `kind`/`events` in that shape), a missing file,
    unreadable or non-JSON content, or any other unexpected shape -- this is
    a cheap discriminator check, not a full-schema validation, so the
    worker can use it to pick which loader/evaluator to run without a
    dataset-format error taking down the whole scoring path."""
    try:
        if isinstance(path_or_obj, dict):
            obj: Any = path_or_obj
        else:
            path = Path(path_or_obj)
            if not path.is_file():
                return False
            obj = json.loads(path.read_text())
        return (
            isinstance(obj, dict)
            and obj.get("kind") == "action_events"
            and isinstance(obj.get("events"), list)
        )
    except (OSError, ValueError):
        return False
