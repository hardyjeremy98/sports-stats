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

This module is pure conversion: no CLI, no DB, no network. The importer
(consuming an external tracker's output back into a run) is a separate,
later task -- not built here.
"""

from __future__ import annotations

import json
from pathlib import Path

from pitchlab_core.provenance import sha256_file
from pitchlab_core.schemas.detections import FrameDetections

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
