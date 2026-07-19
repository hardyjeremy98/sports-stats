"""Subprocess bridge (SPO-46): the pipeline-side of the spotting exchange
contract (docs/reference/spotting-exchange-contract.md).

This module is the single GPL-isolation seam — it imports nothing from
T-DEED (or any other spotter) and knows nothing model-specific, only the
job-manifest/events-JSON shapes the contract fixes. The real, GPL-3.0,
non-commercially-weighted T-DEED CLI and the permissive in-repo reference
spotter (`pitchlab_core.spotting.reference_cli`) are interchangeable behind
this call — swapping `command` is the only thing that changes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from pitchlab_core.schemas.spotting import SpottedEvent


class SpottingBridgeError(RuntimeError):
    """The external spotter subprocess violated the exchange contract: a
    non-zero exit (message carries the stderr diagnostic), or an exit-0 that
    did not leave valid contract JSON at `out_path`. Never silently treated
    as an empty result — a genuinely empty match is `[]` written by the
    spotter itself, not the bridge inventing one."""


@dataclass(frozen=True)
class SpottingParams:
    """The contract's `params` object — opaque, model-facing values the
    bridge only carries through to the job manifest."""

    weights: str
    confidence: float
    merge_window_s: float
    device: str

    def as_dict(self) -> dict:
        return {
            "weights": self.weights,
            "confidence": self.confidence,
            "merge_window_s": self.merge_window_s,
            "device": self.device,
        }


def run_spotter(
    command: list[str],
    *,
    manifest_path: str | Path,
    out_path: str | Path,
    fps: float,
    params: SpottingParams,
    frames_dir: str | Path | None = None,
    clip_path: str | Path | None = None,
    frame_count: int | None = None,
) -> list[SpottedEvent]:
    """Write the job manifest, run `command --job <manifest_path>` as a
    subprocess, and return the parsed+validated `out_path` contents on
    success. Raises `SpottingBridgeError` on any contract violation.

    Exactly one of `frames_dir`/`clip_path` must be given, matching the
    contract's job-manifest requirement (checked here rather than deferred
    to the subprocess, so a caller mistake fails fast with a plain
    `ValueError` instead of a subprocess round trip).
    """
    if bool(frames_dir) == bool(clip_path):
        raise ValueError(
            "run_spotter requires exactly one of frames_dir or clip_path "
            f"(got frames_dir={frames_dir!r}, clip_path={clip_path!r})"
        )

    manifest_path = Path(manifest_path)
    out_path = Path(out_path)

    manifest: dict = {
        "frames_dir": str(frames_dir) if frames_dir is not None else None,
        "clip_path": str(clip_path) if clip_path is not None else None,
        "fps": fps,
        "out_path": str(out_path),
        "params": params.as_dict(),
    }
    if frame_count is not None:
        manifest["frame_count"] = frame_count

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    # Clear any stale out_path from a prior invocation so an exit-0 that
    # writes nothing is detected as "missing out_path" below, rather than
    # silently returning the previous run's contents.
    out_path.unlink(missing_ok=True)

    try:
        result = subprocess.run(
            [*command, "--job", str(manifest_path)],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SpottingBridgeError(
            f"failed to launch spotter command {command!r}: {exc}"
        ) from exc

    if result.returncode != 0:
        raise SpottingBridgeError(
            f"spotter command {command!r} exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )

    if not out_path.exists():
        raise SpottingBridgeError(
            f"spotter command {command!r} exited 0 but did not write "
            f"out_path {out_path}"
        )

    try:
        raw = json.loads(out_path.read_text())
    except json.JSONDecodeError as exc:
        raise SpottingBridgeError(
            f"spotter command {command!r} wrote invalid JSON to {out_path}: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise SpottingBridgeError(
            f"spotter command {command!r} wrote {out_path}, but it is not a "
            "JSON array as the spotting exchange contract requires"
        )

    try:
        return [SpottedEvent.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise SpottingBridgeError(
            f"spotter command {command!r} wrote {out_path}, but its contents "
            f"do not validate against the spotting exchange contract: {exc}"
        ) from exc
