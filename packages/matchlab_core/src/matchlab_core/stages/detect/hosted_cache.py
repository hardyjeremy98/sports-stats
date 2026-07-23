"""Hosted-detection response cache (SPO-10 part 2).

Freezes hosted-API detector responses to disk, keyed by model id +
confidence threshold + frame pixel content, so hosted detections become
frozen, replayable inputs: re-running over a warm cache produces identical
`Detection` lists without any network call. See
`.superpowers/sdd/task-2-brief.md` for the design brief this implements.

Cached JSON schema (one file per key, `"hosted-detections/v1"`)::

    {
      "schema": "hosted-detections/v1",
      "model_id": "football-players-detection-3zvbc/11",
      "confidence": 0.3,                 # request threshold (header field)
      "cached_at": "2026-07-16T12:00:00+00:00",
      "xyxy": [[x1, y1, x2, y2], ...],   # post-conversion arrays -- what the
      "scores": [0.91, ...],             # pipeline actually consumes, not
      "class_id": [2, ...]               # the raw `inference` response object
    }

Note: the brief names both the header's request-confidence field and each
detection's own confidence score "confidence". Both can't share one JSON
key, so this module keeps the header field literally named "confidence"
(paired with "model_id", matching the brief's header description) and puts
per-detection confidence scores under "scores" instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from matchlab_core.provenance import sha256_file

SCHEMA = "hosted-detections/v1"


def cache_key(model_id: str, confidence: float, image: np.ndarray) -> str:
    """Pure, process-stable cache key: sha256 over (model_id,
    repr(confidence), sha256-of-raw-pixel-bytes, shape, dtype).

    Frame index is deliberately NOT part of the key: identical pixels can
    recur across frames (static portions of a clip) and across runs/strides,
    and keying by content -- not position -- is what makes the cache
    replayable across differently-sampled runs.
    """
    contiguous = np.ascontiguousarray(image)
    image_hash = hashlib.sha256(contiguous.tobytes()).hexdigest()
    payload = "|".join(
        [
            model_id,
            repr(confidence),
            image_hash,
            str(tuple(contiguous.shape)),
            str(contiguous.dtype),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CachedDetections:
    """One cache entry's post-conversion payload, as read back from disk."""

    xyxy: list[list[float]]
    scores: list[float]
    class_id: list[int]
    model_id: str
    confidence: float
    schema: str = SCHEMA
    cached_at: str = ""
    extra: dict = field(default_factory=dict)


class HostedDetectionCache:
    """One JSON file per key under `dir`. `get`/`put` are the only mutation
    points; `content_hash()` folds the whole directory into one digest so a
    run's provenance can record exactly which frozen responses backed it.
    """

    def __init__(self, dir: str | Path):
        self.dir = Path(dir)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> CachedDetections | None:
        path = self._path(key)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        known = {"schema", "model_id", "confidence", "cached_at", "xyxy", "scores", "class_id"}
        return CachedDetections(
            xyxy=raw["xyxy"],
            scores=raw["scores"],
            class_id=raw["class_id"],
            model_id=raw["model_id"],
            confidence=raw["confidence"],
            schema=raw.get("schema", SCHEMA),
            cached_at=raw.get("cached_at", ""),
            extra={k: v for k, v in raw.items() if k not in known},
        )

    def put(self, key: str, payload: dict) -> None:
        """`payload` must have `model_id`, `confidence` (the request
        threshold), `xyxy`, `scores`, `class_id` -- the post-conversion
        detection arrays plus the header fields identifying what produced
        them. `cached_at` and `schema` are stamped here, not by the caller.

        Writes atomically (temp file in the same directory, then
        `os.replace`) so a run killed mid-write can never leave a truncated
        JSON file on disk that would poison a later replay.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": SCHEMA,
            "model_id": payload["model_id"],
            "confidence": payload["confidence"],
            "cached_at": datetime.now(UTC).isoformat(),
            "xyxy": payload["xyxy"],
            "scores": payload["scores"],
            "class_id": payload["class_id"],
        }
        final_path = self._path(key)
        tmp_path = final_path.with_suffix(f"{final_path.suffix}.tmp-{os.getpid()}")
        tmp_path.write_text(json.dumps(record))
        os.replace(tmp_path, final_path)

    def content_hash(self) -> str:
        """sha256 over the sorted set of (key, file-sha256) pairs -- the
        cache's content fingerprint for provenance. Order-independent (only
        the set of pairs matters, not insertion order); a fixed deterministic
        value when the cache directory holds no entries yet.
        """
        pairs: list[tuple[str, str]] = []
        if self.dir.exists():
            for path in sorted(self.dir.glob("*.json")):
                pairs.append((path.stem, sha256_file(path)))
        pairs.sort()
        h = hashlib.sha256()
        for key, file_hash in pairs:
            h.update(f"{key}:{file_hash}\n".encode())
        return h.hexdigest()
