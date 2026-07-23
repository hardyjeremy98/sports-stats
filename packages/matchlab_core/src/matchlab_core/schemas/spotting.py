"""SpottedEvent: the pipeline-side model for one element of the spotting
exchange contract's output array
(docs/reference/spotting-exchange-contract.md).

Deliberately separate from Event/EventType (schemas/events.py): a spotter's
native taxonomy class name (e.g. "PASS", "DRIVE", "SHOT") is written verbatim
and is never mapped to EventType here — that mapping, if and when consumed,
is a documented table applied downstream, never enforced in this schema or
inside a spotter CLI (see the contract doc's "class" field note).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SpottedEvent(BaseModel):
    """One event as written by a contract-conforming spotter CLI. `class` is a
    Python keyword, so the field is named `class_` with `alias="class"`;
    `populate_by_name` lets Python callers construct with `class_=...` while
    `serialize_by_alias` guarantees `model_dump`/`model_dump_json` always emit
    the literal JSON key `"class"` the contract requires, regardless of the
    `by_alias` flag a caller passes (or omits)."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    class_: str = Field(alias="class")
    frame_idx: int
    t: float
    confidence: float
    half: int | None = None
