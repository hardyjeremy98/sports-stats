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
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from matchlab_core.exchange import _parse_mot_tracklets
from matchlab_core.frame_features import FrameFeatures
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


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box `a` against (M, 4) xyxy boxes `b`."""
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def join_features_to_tracklets(
    tracklets: list[Tracklet],
    feat_dir: Path,
    local_to_source: list[int],
    *,
    width: int,
    height: int,
    min_iou: float = 0.5,
) -> FrameFeatures:
    """Join the gen_features per-frame pkls (one `{local:06d}.pkl` per decoded
    frame, each a list of per-detection dicts with W/H-normalised `bbox_xywh`)
    to the remapped tracklets, producing the FrameFeatures artifact.

    The tracker links detections but its MOT output only carries boxes, so the
    join is by geometry: within each frame, tracklet boxes claim detections
    greedily by IoU (best match first, each detection consumed once, matches
    below `min_iou` dropped). Tracklet frames with no surviving match simply
    have no feature row — downstream consumers treat missing evidence as
    neutral. Frames index by *source* frame_idx in the result."""
    source_to_local = {src: i for i, src in enumerate(local_to_source)}

    # Per local frame: (M, 4) pixel xyxy boxes + the raw per-detection dicts.
    cache: dict[int, tuple[np.ndarray, list[dict]]] = {}

    def frame_features(local: int) -> tuple[np.ndarray, list[dict]] | None:
        if local in cache:
            return cache[local]
        path = feat_dir / f"{local:06d}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            dets = pickle.load(f)
        if not dets:
            cache[local] = (np.empty((0, 4)), [])
            return cache[local]
        ltwh = np.array([d["bbox_xywh"] for d in dets], dtype=np.float64)
        ltwh *= np.array([width, height, width, height])
        xyxy = np.column_stack(
            [ltwh[:, 0], ltwh[:, 1], ltwh[:, 0] + ltwh[:, 2], ltwh[:, 1] + ltwh[:, 3]]
        )
        cache[local] = (xyxy, dets)
        return cache[local]

    # Gather all candidate (iou, tid, source frame, det dict) matches per frame,
    # then resolve greedily so each detection feeds at most one tracklet frame.
    by_frame: dict[int, list[tuple[float, int, int, int]]] = {}
    det_lookup: dict[int, list[dict]] = {}
    for t in tracklets:
        for tf in t.frames:
            local = source_to_local.get(tf.frame_idx)
            if local is None:
                continue
            feats = frame_features(local)
            if feats is None or not feats[1]:
                continue
            boxes, dets = feats
            det_lookup[local] = dets
            box = np.array([tf.box.x1, tf.box.y1, tf.box.x2, tf.box.y2])
            ious = _iou(box, boxes)
            j = int(np.argmax(ious))
            if ious[j] >= min_iou:
                by_frame.setdefault(local, []).append(
                    (float(ious[j]), t.tracklet_id, tf.frame_idx, j)
                )

    tids: list[int] = []
    fidxs: list[int] = []
    rows: list[dict] = []
    for local, candidates in by_frame.items():
        used: set[int] = set()
        for _iou_val, tid, fidx, j in sorted(candidates, reverse=True):
            if j in used:
                continue  # detection already claimed by a better-matching box
            used.add(j)
            tids.append(tid)
            fidxs.append(fidx)
            rows.append(det_lookup[local][j])

    n = len(rows)
    if n:
        emb = np.array([r["appearance_embeddings"] for r in rows], dtype=np.float32)
        vis = np.array([r["appearance_visibility"] for r in rows], dtype=np.float32)
        kpts = np.array([r["keypoints_xyc"] for r in rows], dtype=np.float32)
        kconf = np.array([r["keypoints_conf"] for r in rows], dtype=np.float32)
    else:
        emb = np.empty((0, 0, 0), dtype=np.float32)
        vis = np.empty((0, 0), dtype=np.float32)
        kpts = np.empty((0, 0, 3), dtype=np.float32)
        kconf = np.empty((0,), dtype=np.float32)
    return FrameFeatures(
        tracklet_ids=np.array(tids, dtype=np.int64),
        frame_idxs=np.array(fidxs, dtype=np.int64),
        embeddings=emb,
        visibility=vis,
        keypoints_xyc=kpts,
        keypoints_conf=kconf,
        meta={"width": width, "height": height, "source": "tdlp-full"},
    )


def parse_tracker_output(mot_path: Path, local_to_source: list[int]) -> list[Tracklet]:
    """Parse the tracker's MOT output (validated by `_parse_mot_tracklets`) and
    remap its local frame indices to source `frame_idx`."""
    local_tracklets = _parse_mot_tracklets(mot_path)
    return remap_tracklets_to_source(local_tracklets, local_to_source)
