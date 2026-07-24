"""Subprocess bridge for external camera calibrators: the pipeline-side of the
calibration exchange contract.

Mirrors matchlab_core.spotting.bridge. This module is the isolation seam — it
imports nothing model-specific (no torch, no HRNet, no external calibrator),
only the job-manifest / homographies-JSON shapes the contract fixes. The real
external calibrator CLI (its own isolated environment, GPU) and the permissive
in-repo reference calibrator (`matchlab_core.calib.reference_cli`) are
interchangeable behind this call — swapping `command` is the only thing that
changes.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from matchlab_core.schemas.calibration import ExternalHomography

_FRAME_FILENAME_RE = re.compile(r"^(\d+)\.\w+$")


class CalibrationBridgeError(RuntimeError):
    """The external calibrator subprocess violated the exchange contract: a
    non-zero exit (message carries the stderr diagnostic), or an exit-0 that did
    not leave valid contract JSON at `out_path` (missing, unparseable, not an
    array, schema-invalid records, a frame_idx set disagreeing with the manifest,
    or a singular/non-invertible homography). Never silently treated as an empty
    result — a frame the model genuinely could not calibrate is a record with
    `homography: null`, written by the calibrator itself, not the bridge
    inventing one."""


@dataclass(frozen=True)
class CalibrationParams:
    """The contract's `params` object — opaque, model-facing values the bridge
    only carries through to the job manifest. `weights_*` are empty for the
    reference calibrator, which needs no model."""

    weights_kp: str
    weights_line: str
    kp_threshold: float
    line_threshold: float
    pnl_refine: bool
    device: str

    def as_dict(self) -> dict:
        return {
            "weights_kp": self.weights_kp,
            "weights_line": self.weights_line,
            "kp_threshold": self.kp_threshold,
            "line_threshold": self.line_threshold,
            "pnl_refine": self.pnl_refine,
            "device": self.device,
        }


def _expected_frame_indices(frames_dir: Path) -> list[int]:
    return sorted(
        int(match.group(1))
        for entry in frames_dir.iterdir()
        if (match := _FRAME_FILENAME_RE.match(entry.name))
    )


def run_calibrator(
    command: list[str],
    *,
    manifest_path: str | Path,
    out_path: str | Path,
    fps: float,
    params: CalibrationParams,
    frames_dir: str | Path,
    timeout_s: float = 3600.0,
) -> list[ExternalHomography]:
    """Write the job manifest, run `command --job <manifest_path>` as a
    subprocess, and return the parsed+validated `out_path` contents on success.
    Raises `CalibrationBridgeError` on any contract violation, including the
    subprocess exceeding `timeout_s` (default 60 minutes — a hung external
    calibrator must not block the worker indefinitely)."""
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    frames_dir = Path(frames_dir)

    manifest = {
        "frames_dir": str(frames_dir),
        "fps": fps,
        "out_path": str(out_path),
        "params": params.as_dict(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    # Clear any stale out_path so an exit-0 that writes nothing is detected as
    # "missing out_path" below, not silently returning a prior run's contents.
    out_path.unlink(missing_ok=True)

    try:
        result = subprocess.run(
            [*command, "--job", str(manifest_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except OSError as exc:
        raise CalibrationBridgeError(
            f"failed to launch calibrator command {command!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CalibrationBridgeError(
            f"calibrator command {command!r} timed out after {timeout_s}s"
        ) from exc

    if result.returncode != 0:
        raise CalibrationBridgeError(
            f"calibrator command {command!r} exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )

    if not out_path.exists():
        raise CalibrationBridgeError(
            f"calibrator command {command!r} exited 0 but did not write "
            f"out_path {out_path}"
        )

    try:
        raw = json.loads(out_path.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationBridgeError(
            f"calibrator command {command!r} wrote invalid JSON to {out_path}: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise CalibrationBridgeError(
            f"calibrator command {command!r} wrote {out_path}, but it is not a "
            "JSON array as the calibration exchange contract requires"
        )

    try:
        records = [ExternalHomography.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise CalibrationBridgeError(
            f"calibrator command {command!r} wrote {out_path}, but its contents "
            f"do not validate against the calibration exchange contract: {exc}"
        ) from exc

    expected = _expected_frame_indices(frames_dir)
    got = sorted(r.frame_idx for r in records)
    if got != expected:
        raise CalibrationBridgeError(
            f"calibrator command {command!r} returned frame indices {got} but the "
            f"job manifest requested {expected}"
        )

    for r in records:
        if r.homography is None:
            continue
        matrix = np.array(r.homography, dtype=np.float64)
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
            raise CalibrationBridgeError(
                f"calibrator command {command!r} returned a singular (non-invertible) "
                f"homography for frame {r.frame_idx}"
            )

    return records
