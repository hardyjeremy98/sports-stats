"""Reference camera-calibration CLI (permissive, deterministic, no real model).

Implements the calibration exchange contract without any real model: it reads a
job manifest, resolves the ordered list of source-video frame indices from its
frames_dir listing, and emits one image→pitch-cm homography record per frame — a
single fixed, invertible matrix at constant confidence. It exists so the
subprocess bridge and the `pnlcalib` pipeline stage can be developed and tested
without a real external calibrator (its own isolated environment, GPU).

Run as:
    python -m matchlab_core.calib.reference_cli --job <manifest.json>

Depends only on the Python stdlib — no torch, no OpenCV — so it runs anywhere,
including CI, with no model or GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# A fixed, invertible image→pitch-cm homography (pure 10× scale). Not a real
# calibration — just a well-formed, affine matrix so the bridge/stage round-trip
# produces non-null FrameCalibrations whose reprojected pitch vertices land in a
# normal-sized frame. Purely a reference-CLI implementation detail; the contract
# only fixes the *shape* of each record.
REFERENCE_HOMOGRAPHY = [
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 1.0],
]
REFERENCE_CONFIDENCE = 0.5
REFERENCE_N_POINTS = 32

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
    frames_dir = _require(manifest, "frames_dir")
    frames_dir_path = Path(str(frames_dir))
    if not frames_dir_path.is_dir():
        raise ManifestError(f"frames_dir does not exist or is not a directory: {frames_dir}")
    indices = [
        int(match.group(1))
        for entry in frames_dir_path.iterdir()
        if (match := _FRAME_FILENAME_RE.match(entry.name))
    ]
    return sorted(indices)


def run(job_path: Path) -> None:
    """Read the manifest at job_path, validate it, and write contract-valid
    homographies JSON to its declared out_path. Raises ManifestError on any
    contract violation without writing anything to out_path."""
    manifest = _load_manifest(job_path)

    out_path = Path(str(_require(manifest, "out_path")))

    fps = _require(manifest, "fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ManifestError("job manifest 'fps' must be a positive number")

    params = _require(manifest, "params")
    if not isinstance(params, dict):
        raise ManifestError("job manifest 'params' must be an object")
    # The reference calibrator does not need these (it emits a fixed matrix), but
    # the contract fixes 'params' shape regardless of which calibrator reads it.
    for required_param in (
        "weights_kp",
        "weights_line",
        "kp_threshold",
        "line_threshold",
        "pnl_refine",
        "device",
    ):
        _require(params, required_param, where="job manifest 'params'")

    records = [
        {
            "frame_idx": frame_idx,
            "homography": REFERENCE_HOMOGRAPHY,
            "confidence": REFERENCE_CONFIDENCE,
            "n_points": REFERENCE_N_POINTS,
        }
        for frame_idx in _frame_indices(manifest)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="matchlab-reference-calibrator",
        description="Permissive, deterministic reference camera-calibration CLI "
        "(no real model) implementing the calibration exchange contract.",
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
