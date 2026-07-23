"""The single process boundary between matchlab and the external TDLP-full
stack. Pure stdlib + our own schemas — imports nothing from TDLP/CAMELTrack.

Responsibilities, in the order the stage calls them:

1. `stage_sequence` — turn the run's frames + detections into the benchmark
   MOT-sequence layout the external scripts already consume
   (`<data_root>/<split>/<seq>/{img1/*.jpg, det/det.txt, seqinfo.ini}`), and
   return the local↔source frame-index mapping needed to translate the result
   back. This is what makes the stage work on *any* video: the "arbitrary clip
   → known layout" step lives here, in permissive in-repo code, so the external
   scripts need no per-dataset special-casing.
2. `run_external` — invoke one external venv's python on one script, failing
   loudly (naming the command + a stderr tail) on non-zero exit or timeout.
3. `parse_tracker_output` — parse the tracker's MOT file (reusing the validated
   `exchange._parse_mot_tracklets`) and remap its contiguous 1-based frames back
   to source-video `frame_idx`, honouring `sample_stride`.

Frame-index bookkeeping (the subtle part): the external stack wants a
contiguous 1..N MOT sequence, but matchlab artifacts index by *source* video
`frame_idx`, which is stride-independent and need not start at 0. So we assign
each decoded frame a contiguous local index (1-based for MOT, 0-based for the
feature pkls) and keep `local_to_source` to invert it on the way out.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from matchlab_core.exchange import _parse_mot_tracklets
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.video import Frame

# Persons only; the ball has its own trajectory stage and TDLP tracks people.
DEFAULT_INCLUDE_CLASSES: frozenset[str] = frozenset(("player", "goalkeeper", "referee"))


@dataclass
class SequenceLayout:
    """Where the fabricated MOT sequence lives, plus what's needed to invert
    the frame indexing and to fill `seqinfo`/normalisation downstream."""

    data_root: Path  # pass as --data-root to the external tracker
    split: str  # the single split dir under data_root
    seq_name: str  # the single sequence under data_root/split
    img_dir: Path  # <seq>/img1
    det_file: Path  # <seq>/det/det.txt
    local_to_source: list[int]  # local 0-based frame index -> source frame_idx
    width: int
    height: int
    n_detections: int


def write_det_file(
    detections: list[FrameDetections],
    source_to_local: dict[int, int],
    path: Path,
    include_classes: frozenset[str] = DEFAULT_INCLUDE_CLASSES,
) -> int:
    """Write person detections as 1-based MOT det rows
    (`frame,-1,x,y,w,h,conf,-1,-1,-1`, xywh top-left), using the fabricated
    contiguous local frame numbers. Fixed `%.2f`/`%.6f` formatting so the file
    is byte-stable. Detections on a frame not in `source_to_local` (i.e. not
    decoded under the run's sampling) are skipped. Returns the row count."""
    rows: list[tuple[int, float, float, float, float, float]] = []
    for fd in detections:
        local = source_to_local.get(fd.frame_idx)
        if local is None:
            continue
        mot_frame = local + 1  # MOT frames are 1-based
        for det in fd.detections:
            if det.cls not in include_classes:
                continue
            b = det.box
            rows.append((mot_frame, b.x1, b.y1, b.width, b.height, det.confidence))
    rows.sort(key=lambda r: r[0])  # stable: preserves input order within a frame
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for mot_frame, x, y, w, h, conf in rows:
            f.write(f"{mot_frame},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.6f},-1,-1,-1\n")
    return len(rows)


def write_seqinfo(
    path: Path,
    *,
    name: str,
    seq_length: int,
    width: int,
    height: int,
    fps: float,
    im_dir: str = "img1",
    im_ext: str = ".jpg",
) -> None:
    """Write a MOTChallenge `seqinfo.ini` — the file the external tracker's
    `mot` dataset index reads for image size and sequence length."""
    path.write_text(
        "[Sequence]\n"
        f"name={name}\n"
        f"imDir={im_dir}\n"
        f"frameRate={max(1, int(round(fps)))}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        f"imExt={im_ext}\n"
    )


def remap_tracklets_to_source(
    tracklets: list[Tracklet], local_to_source: list[int]
) -> list[Tracklet]:
    """Translate each tracklet frame's local (contiguous, 0-based) index back to
    the source-video `frame_idx`. A local index outside the fabricated sequence
    means the tracker emitted a frame we never gave it — a bug, refused loudly."""
    n = len(local_to_source)
    out: list[Tracklet] = []
    for t in tracklets:
        frames = []
        for tf in t.frames:
            if tf.frame_idx < 0 or tf.frame_idx >= n:
                raise ValueError(
                    f"Tracker emitted local frame index {tf.frame_idx} for "
                    f"tracklet {t.tracklet_id}, outside the fabricated sequence "
                    f"of {n} frames — the MOT output does not match the input layout."
                )
            frames.append(tf.model_copy(update={"frame_idx": local_to_source[tf.frame_idx]}))
        out.append(t.model_copy(update={"frames": frames}))
    return out


def stage_sequence(
    frames: list[Frame] | object,
    detections: list[FrameDetections],
    work_dir: Path,
    *,
    seq_name: str,
    fps: float,
    split: str = "test",
    include_classes: frozenset[str] = DEFAULT_INCLUDE_CLASSES,
) -> SequenceLayout:
    """Decode `frames` to a MOT sequence under `work_dir` and write the
    matching `det.txt` + `seqinfo.ini`. `frames` is any iterable of `Frame`
    (typically `ctx.frames()`); iteration order defines the contiguous local
    index (1-based image filenames, so `000001.jpg` is local index 0). Image
    dimensions are taken from the decoded frames — they are the same
    (possibly resized) pixels the detector saw, so boxes stay consistent."""
    seq_dir = work_dir / split / seq_name
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)

    source_frames: list[int] = []
    width = height = 0
    for fr in frames:
        local = len(source_frames)
        cv2.imwrite(str(img_dir / f"{local + 1:06d}.jpg"), fr.image)
        source_frames.append(fr.frame_idx)
        height, width = fr.image.shape[:2]

    if not source_frames:
        raise RuntimeError(
            f"No frames decoded for sequence {seq_name!r}; cannot run TDLP-full "
            "on an empty clip."
        )

    source_to_local = {src: i for i, src in enumerate(source_frames)}
    det_file = seq_dir / "det" / "det.txt"
    n_det = write_det_file(detections, source_to_local, det_file, include_classes)
    write_seqinfo(
        seq_dir / "seqinfo.ini",
        name=seq_name,
        seq_length=len(source_frames),
        width=width,
        height=height,
        fps=fps,
    )
    return SequenceLayout(
        data_root=work_dir,
        split=split,
        seq_name=seq_name,
        img_dir=img_dir,
        det_file=det_file,
        local_to_source=source_frames,
        width=width,
        height=height,
        n_detections=n_det,
    )


def run_external(
    python_exe: Path,
    script: Path,
    args: list[str],
    *,
    cwd: Path,
    timeout_s: float | None,
    label: str,
    extra_pythonpath: Path | None = None,
    progress=None,
) -> str:
    """Run `<python_exe> <script> <args...>` in `cwd`, capturing output. Raise a
    RuntimeError (naming the command and a stderr tail) on non-zero exit or
    timeout — every external failure mode collapses here into one clear error.

    `extra_pythonpath` is prepended to PYTHONPATH for the child — used to make
    the TDLP repo's own `tdlp` package importable without relying on a
    (move-fragile) editable install in the venv."""
    cmd = [str(python_exe), str(script), *args]
    env = None
    if extra_pythonpath is not None:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{extra_pythonpath}{os.pathsep}{existing}" if existing else str(extra_pythonpath)
        )
    if progress is not None:
        progress(f"TDLP-full: {label} …")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"TDLP-full {label} timed out after {timeout_s}s. Command:\n  "
            + " ".join(cmd)
        ) from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        raise RuntimeError(
            f"TDLP-full {label} failed (exit {proc.returncode}). Command:\n  "
            + " ".join(cmd)
            + f"\n--- stderr (last 25 lines) ---\n{tail}"
        )
    return proc.stdout or ""


def parse_tracker_output(mot_path: Path, local_to_source: list[int]) -> list[Tracklet]:
    """Parse the tracker's MOT output (validated by `_parse_mot_tracklets`) and
    remap its local frame indices to source `frame_idx`."""
    local_tracklets = _parse_mot_tracklets(mot_path)
    return remap_tracklets_to_source(local_tracklets, local_to_source)
