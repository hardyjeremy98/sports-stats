"""Frozen-detections export (SPO-18 part 1): turn a run's detections.jsonl
into a per-sequence frozen package for external MOT research trackers -- a
standard MOT-format `det.txt` plus a JSON provenance sidecar carrying the
detector's provenance and the emitted file's hash, so every external
candidate tracker consumes identical, provenance-stamped input.

`det.txt` columns are the standard MOT detections format:
`frame,id,x,y,w,h,conf,-1,-1,-1`. MOT frame numbers are **1-based**
(`frame = frame_idx + 1`, where `frame_idx` is the source-video frame index
carried by `FrameDetections`); `id` is always `-1` because raw detections
carry no identity; `x,y` is the box top-left and `w,h` its width/height
(xywh, converted from the `Detection.box` xyxy representation). Rows are
ordered by (frame, then input order within that frame).

This module is pure conversion: no CLI, no DB, no network.

`import_mot_tracklets` is the reverse direction (SPO-18 part 2): a MOT-format
tracklet file produced by an external tracker (fed the frozen `det.txt` above,
or run on its own detections) is parsed into `tracklets.json` plus a
`manifest.json` carrying a real `RunProvenance` for the external system -- the
raw online-tracklet layer only, never entity or identity artifacts. Imports
are refused loudly when the required provenance sidecar (`ExternalProvenance`)
is missing or incomplete: an external system with no stated identity, license
axes, or reference-only status cannot be compared against shipping results.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from pitchlab_core.provenance import (
    LicenseAxes,
    ModelProvenance,
    RunProvenance,
    StageProvenance,
    check_evaluation_set,
    sha256_file,
)
from pitchlab_core.schemas.detections import DetectionClass, FrameDetections
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.schemas.tracks import Tracklet, TrackletFrame

# External MOT trackers track persons; ball trajectories are handled by a
# dedicated ball-tracking stage, so ball detections are excluded by default.
# Pass `include_classes=(..., "ball")` to include them.
DEFAULT_INCLUDE_CLASSES: tuple[str, ...] = ("player", "goalkeeper", "referee")


def export_frozen_detections(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    include_classes: tuple[str, ...] = DEFAULT_INCLUDE_CLASSES,
) -> Path:
    """Read `<run_dir>/detections.jsonl` + `<run_dir>/manifest.json` and
    write a frozen detections package into `out_dir`:

    - `det.txt`: one MOT-format row per included detection (see module
      docstring for the column layout). Coordinates are formatted `%.2f`,
      confidence `%.6f` -- fixed formatting so the file (and its hash) is
      byte-identical across repeated exports of the same run.
    - `detections_provenance.json`: a sidecar recording the source run,
      the det.txt hash, video/detector provenance, and the class filter
      applied, serialized with `sort_keys=True` for determinism. Unknown
      provenance is always recorded as the literal string `"unknown"`,
      never omitted.

    Returns `out_dir`.

    Raises `FileNotFoundError` (naming the missing path) if either input
    file is absent, and `ValueError` (naming the file and line number) if a
    detections.jsonl row fails `FrameDetections` validation.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)

    detections_path = run_dir / "detections.jsonl"
    manifest_path = run_dir / "manifest.json"
    if not detections_path.exists():
        raise FileNotFoundError(f"No detections.jsonl found at {detections_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    include_set = set(include_classes)
    # (mot_frame, x, y, w, h, conf) in input order; sorted by mot_frame below.
    rows: list[tuple[int, float, float, float, float, float]] = []
    frames_with_rows: set[int] = set()

    with open(detections_path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                frame = FrameDetections.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid FrameDetections row in {detections_path}:{lineno}: {exc}"
                ) from exc

            mot_frame = frame.frame_idx + 1
            for det in frame.detections:
                if det.cls not in include_set:
                    continue
                box = det.box
                rows.append(
                    (mot_frame, box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1, det.confidence)
                )
                frames_with_rows.add(frame.frame_idx)

    rows.sort(key=lambda r: r[0])  # stable: preserves input order within a frame

    out_dir.mkdir(parents=True, exist_ok=True)
    det_txt_path = out_dir / "det.txt"
    with open(det_txt_path, "w") as f:
        for mot_frame, x, y, w, h, conf in rows:
            f.write(f"{mot_frame},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")

    det_txt_sha256 = sha256_file(det_txt_path)

    video = manifest.get("video", {})
    stages = manifest.get("provenance", {}).get("stages", {})
    detect_provenance = stages.get("detect", "unknown")

    sidecar = {
        "schema": "frozen-detections/v1",
        "source_run_id": manifest.get("run_id", "unknown"),
        "source_run_dir": str(run_dir),
        "det_txt_sha256": det_txt_sha256,
        "frame_count": video.get("frame_count", "unknown"),
        "sample_stride": video.get("sample_stride", "unknown"),
        "fps": video.get("fps", "unknown"),
        "include_classes": list(include_classes),
        "n_rows": len(rows),
        "n_frames_with_detections": len(frames_with_rows),
        "detect_provenance": detect_provenance,
        "class_map_note": (
            "det.txt rows are xywh MOT format with id=-1 (raw detections carry "
            f"no identity); classes were filtered to {sorted(include_set)} at "
            "export time, so classes outside that set (e.g. 'ball', unless "
            "explicitly included) produce no rows here."
        ),
    }
    (out_dir / "detections_provenance.json").write_text(
        json.dumps(sidecar, sort_keys=True, indent=2)
    )

    return out_dir


class ExternalProvenance(BaseModel):
    """The provenance sidecar an external tracker's output must ship with to
    be importable at all (SPO-18 part 2). `license` and `reference_only` are
    required, not defaulted -- a caller must state them explicitly (even
    "unknown" per axis) so a provenance-less external result can never enter
    a run directory, and `reference_only` is the flag that keeps reference
    systems out of shipping comparisons (SPO-17, out of scope here)."""

    system: str  # e.g. "TDLP" -- required, no default.
    variant: str = "unknown"  # e.g. "bbox-only"
    repo_url: str = "unknown"
    commit: str = "unknown"
    weights: str = "unknown"  # identifier/URL
    weights_sha256: str | None = None
    license: LicenseAxes  # required -- caller must state it, even if all axes "unknown".
    reference_only: bool  # required -- no safe default for a reference-vs-shipping flag.
    # The det.txt hash this tracker consumed, when known (cross-checked
    # against a frozen-detections export's det_txt_sha256 at import time).
    frozen_detections_sha256: str | None = None
    notes: str = ""


def _load_external_provenance(sidecar_path: Path) -> ExternalProvenance:
    if not sidecar_path.exists():
        raise FileNotFoundError(f"No provenance sidecar found at {sidecar_path}")
    try:
        raw = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Provenance sidecar {sidecar_path} is not valid JSON: {exc}") from exc
    try:
        return ExternalProvenance.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"Invalid provenance sidecar {sidecar_path} (provenance-less input is "
            f"refused): {exc}"
        ) from exc


def _parse_mot_tracklets(mot_path: Path) -> list[Tracklet]:
    """Parse MOT rows `frame,track_id,x,y,w,h,conf[,...]` (1-based frame ->
    0-based frame_idx; xywh -> xyxy Box; conf defaults to 1.0 if the column
    is absent) into `Tracklet`s, grouped by track_id and sorted by
    (track_id, frame_idx). A malformed row or a duplicate (track_id,
    frame_idx) pair -- an external tracker emitting two boxes for one track
    in one frame is malformed -- refuses loudly, naming the file and row.
    """
    if not mot_path.exists():
        raise FileNotFoundError(f"No MOT tracklet file found at {mot_path}")

    frames_by_track: dict[int, list[TrackletFrame]] = {}
    seen: set[tuple[int, int]] = set()
    for lineno, raw_line in enumerate(mot_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 6:
            raise ValueError(
                f"Malformed MOT row in {mot_path}, row {lineno}: {line!r} "
                "(expected at least frame,track_id,x,y,w,h)"
            )
        try:
            frame = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 and parts[6] != "" else 1.0
        except ValueError as exc:
            raise ValueError(
                f"Malformed MOT row in {mot_path}, row {lineno}: {line!r}"
            ) from exc

        frame_idx = frame - 1
        key = (track_id, frame_idx)
        if key in seen:
            raise ValueError(
                f"Duplicate row for track_id={track_id}, frame_idx={frame_idx} "
                f"in {mot_path}, row {lineno}: {line!r}"
            )
        seen.add(key)
        frames_by_track.setdefault(track_id, []).append(
            TrackletFrame(
                frame_idx=frame_idx,
                box=Box(x1=x, y1=y, x2=x + w, y2=y + h),
                confidence=conf,
                source="observed",
            )
        )

    tracklets = []
    for track_id in sorted(frames_by_track):
        frames = sorted(frames_by_track[track_id], key=lambda f: f.frame_idx)
        tracklets.append(
            Tracklet(tracklet_id=track_id, cls=DetectionClass.PLAYER, frames=frames)
        )
    return tracklets


def import_mot_tracklets(
    mot_path: str | Path,
    sidecar_path: str | Path,
    out_run_dir: str | Path,
    *,
    fps: float,
    frame_count: int,
    sample_stride: int = 1,
    frozen_detections_dir: str | Path | None = None,
) -> Path:
    """Import an external tracker's MOT-format output as a standard,
    scoreable run directory: `tracklets.json` in the native schema (the raw
    online tracklet layer only -- no entity or identity artifacts) plus a
    `manifest.json` carrying a real `RunProvenance` for the external system,
    and the raw sidecar copied verbatim to `external_provenance.json`.

    Refuses loudly (naming the offending path/field/row) on: a missing
    sidecar or MOT file, a sidecar missing required fields (`system`,
    `license`, `reference_only`), a malformed or duplicate MOT row, or -- when
    `frozen_detections_dir` is given and the sidecar states
    `frozen_detections_sha256` -- a mismatch against that export's
    `det_txt_sha256` (reuses `check_evaluation_set`, naming both hashes).

    Returns `out_run_dir`.
    """
    mot_path = Path(mot_path)
    sidecar_path = Path(sidecar_path)
    out_run_dir = Path(out_run_dir)

    sidecar = _load_external_provenance(sidecar_path)
    tracklets = _parse_mot_tracklets(mot_path)

    if frozen_detections_dir is not None:
        frozen_detections_dir = Path(frozen_detections_dir)
        frozen_prov_path = frozen_detections_dir / "detections_provenance.json"
        if not frozen_prov_path.exists():
            raise FileNotFoundError(
                f"No detections_provenance.json found at {frozen_prov_path}"
            )
        frozen_prov = json.loads(frozen_prov_path.read_text())
        if sidecar.frozen_detections_sha256 is not None:
            check_evaluation_set(
                sidecar.frozen_detections_sha256,
                frozen_prov.get("det_txt_sha256", "unknown"),
                context=f"frozen-detections check for import of {mot_path}",
            )

    out_run_dir.mkdir(parents=True, exist_ok=True)

    (out_run_dir / "tracklets.json").write_text(
        json.dumps([t.model_dump(mode="json") for t in tracklets])
    )

    model_provenance = ModelProvenance(
        architecture=sidecar.system,
        revision=f"{sidecar.variant}/{sidecar.commit}",
        weights_path=None,
        weights_sha256=sidecar.weights_sha256,
        lineage=f"external ({sidecar.repo_url})",
        license=sidecar.license,
    )
    stage_provenance = StageProvenance(
        impl=f"external:{sidecar.system}",
        params=sidecar.model_dump(mode="json"),
        models=[model_provenance],
    )
    run_provenance = RunProvenance(
        stages={"track": stage_provenance}, evaluation_set_hash="unknown"
    )

    manifest = {
        "video": {"fps": fps, "frame_count": frame_count, "sample_stride": sample_stride},
        "config_name": "external-import",
        "config": {
            "stages": {
                "track": {
                    "impl": "external",
                    "params": {
                        "mot_path": str(mot_path),
                        "sidecar_path": str(sidecar_path),
                        "fps": fps,
                        "frame_count": frame_count,
                        "sample_stride": sample_stride,
                        "frozen_detections_dir": (
                            str(frozen_detections_dir) if frozen_detections_dir else None
                        ),
                    },
                }
            }
        },
        "provenance": run_provenance.model_dump(mode="json"),
    }
    (out_run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Copied verbatim -- the sidecar artifact named by the acceptance criteria.
    (out_run_dir / "external_provenance.json").write_text(sidecar_path.read_text())

    return out_run_dir
