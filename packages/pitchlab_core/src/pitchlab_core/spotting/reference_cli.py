"""Reference action-spotting CLI (permissive, deterministic, no real model).

Implements the spotting exchange contract
(docs/reference/spotting-exchange-contract.md) without any real model: it reads a job
manifest, resolves the ordered list of source-video frame indices it covers, and emits a
fixed synthetic event pattern -- one event every EVENT_INTERVAL_FRAMES frames, cycling
through a small fixed native-taxonomy class list, at a constant confidence. It exists so
the subprocess bridge and the `tdeed` pipeline stage (built in later tasks) can be
developed and tested without the real, GPL-licensed, non-commercially-weighted T-DEED
model.

Run as:
    python -m pitchlab_core.spotting.reference_cli --job <manifest.json>

Depends only on the Python stdlib (json, argparse, pathlib, re, sys) -- no torch, no
heavy deps -- so it runs anywhere, including CI, with no model or GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Deterministic synthetic pattern: emit one event every N frames (by position in the
# resolved frame sequence, not raw frame_idx), cycling through this fixed list of
# native (spotter-taxonomy) class names. Purely a reference-CLI implementation detail --
# not part of the exchange contract, which only fixes the *shape* of each event.
EVENT_INTERVAL_FRAMES = 10
REFERENCE_CLASSES = ("PASS", "DRIVE", "SHOT")
REFERENCE_CONFIDENCE = 0.75

_FRAME_FILENAME_RE = re.compile(r"^(\d+)\.\w+$")


class ManifestError(ValueError):
    """The job manifest is missing, unreadable, not valid JSON, or missing/invalid
    required fields per the exchange contract."""


def _load_manifest(job_path: Path) -> dict:
    try:
        raw = job_path.read_text()
    except OSError as exc:
        raise ManifestError(f"cannot read job manifest {job_path}: {exc}") from exc
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"job manifest {job_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError(f"job manifest {job_path} must be a JSON object")
    return manifest


def _require(obj: dict, key: str, *, where: str = "job manifest") -> object:
    if key not in obj or obj[key] is None:
        raise ManifestError(f"{where} missing required field {key!r}")
    return obj[key]


def _frame_indices(manifest: dict) -> list[int]:
    """Resolve the ordered list of source-video frame indices this job covers, from
    either a frames_dir listing or an explicit frame_count alongside clip_path."""
    frames_dir = manifest.get("frames_dir")
    clip_path = manifest.get("clip_path")
    if bool(frames_dir) == bool(clip_path):
        raise ManifestError(
            "job manifest must set exactly one of 'frames_dir' or 'clip_path'"
        )

    if frames_dir:
        frames_dir_path = Path(frames_dir)
        if not frames_dir_path.is_dir():
            raise ManifestError(f"frames_dir does not exist or is not a directory: {frames_dir}")
        indices = [
            int(match.group(1))
            for entry in frames_dir_path.iterdir()
            if (match := _FRAME_FILENAME_RE.match(entry.name))
        ]
        return sorted(indices)

    # clip_path given: no per-frame listing available without decoding the clip, so
    # fall back to an explicit frame_count (see the exchange contract).
    frame_count = manifest.get("frame_count")
    if frame_count is None:
        raise ManifestError(
            "job manifest sets 'clip_path' but no 'frame_count'; the reference CLI "
            "never decodes video, so it cannot count frames without one"
        )
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
        raise ManifestError("job manifest 'frame_count' must be a non-negative integer")
    return list(range(frame_count))


def _build_events(frame_indices: list[int], fps: float) -> list[dict]:
    events = []
    for position in range(0, len(frame_indices), EVENT_INTERVAL_FRAMES):
        frame_idx = frame_indices[position]
        event_number = position // EVENT_INTERVAL_FRAMES
        events.append(
            {
                "class": REFERENCE_CLASSES[event_number % len(REFERENCE_CLASSES)],
                "frame_idx": frame_idx,
                "t": frame_idx / fps,
                "confidence": REFERENCE_CONFIDENCE,
                "half": None,
            }
        )
    return events


def run(job_path: Path) -> None:
    """Read the manifest at job_path, validate it, and write contract-valid events
    JSON to its declared out_path. Raises ManifestError on any contract violation
    without writing anything to out_path."""
    manifest = _load_manifest(job_path)

    out_path = Path(_require(manifest, "out_path"))

    fps = _require(manifest, "fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ManifestError("job manifest 'fps' must be a positive number")

    params = _require(manifest, "params")
    if not isinstance(params, dict):
        raise ManifestError("job manifest 'params' must be an object")
    # The reference spotter does not need these values (it emits a fixed synthetic
    # pattern), but the contract fixes 'params' shape regardless of which spotter is
    # reading it, so their presence is still required.
    for required_param in ("weights", "confidence", "merge_window_s", "device"):
        _require(params, required_param, where="job manifest 'params'")

    frame_indices = _frame_indices(manifest)
    events = _build_events(frame_indices, float(fps))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(events))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pitchlab-reference-spotter",
        description="Permissive, deterministic reference action-spotter CLI "
        "(no real model) implementing the spotting exchange contract.",
    )
    parser.add_argument("--job", required=True, help="Path to the job manifest JSON")
    args = parser.parse_args(argv)

    try:
        run(Path(args.job))
    except ManifestError as exc:
        print(f"reference_cli: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
