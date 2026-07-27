"""Body-ID embeddings for FOOTPASS fragments.

FOOTPASS ships GT boxes (`ROI_*`, full-HD pixel scale) alongside identity and
position, so appearance and position can finally be measured on the SAME
fragments -- the join that has never existed in this programme.

Pipeline, reusing the existing external bridge rather than a second embedder:

    video -> sampled frames -> MOT det.txt of GT boxes -> bridge/gen_features.py
    (--reid-model prtreid --no-pose) -> per-frame pickles -> per-fragment prototype

Only a handful of frames per fragment are sampled: a tracklet prototype needs
crops, not every frame, and a 45-minute half is ~72,000 frames. Sampling is
deterministic (evenly spaced within the fragment's observable span) so a rerun
reproduces the same embeddings.

Pose is skipped (`--no-pose`): nothing under `matchlab_core/reid/` reads
keypoints, and RTMPose is the slowest stage in the bridge.
"""

from __future__ import annotations

import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matchlab_train.datasets.footpass import COL, FootpassHalf

CAMEL_PY = Path("../external-trackers/CAMELTrack/.venv/bin/python")
BRIDGE = Path("../external-trackers/bridge/gen_features.py")


@dataclass
class AppearanceConfig:
    weights: Path
    device: str = "cuda:0"
    samples_per_fragment: int = 8
    reid_model: str = "prtreid"


def sample_frames(frags, *, stride: int = 25) -> list[int]:
    """Frames to decode: a fixed global stride, not per-fragment seeks.

    Random seeking in a 3.7 GB H.264 file is far slower than one sequential
    pass, and a global grid lets every visible player in a decoded frame be
    embedded at once rather than decoding the same frame once per fragment.
    With `stride` below the minimum fragment length (50 frames), every fragment
    is guaranteed at least one sample.
    """
    if not frags:
        return []
    lo = min(f.start for f in frags)
    hi = max(f.end for f in frags)
    return list(range(int(lo), int(hi) + 1, stride))


def write_det_file(
    half: FootpassHalf, frags, want: list[int], out: Path
) -> dict[tuple[int, int], int]:
    """MOT det.txt of GT boxes for the sampled (frame, fragment) pairs.

    Returns a map from (frame_idx, row_order) -> fragment index so the bridge's
    per-frame outputs can be joined back. Rows with a NaN box are skipped: those
    are frames where the player is not visible, and a crop there would be a
    different player or empty pitch.
    """
    rows = half.rows
    # fragment lookup: (player_id, frame) -> fragment index
    owner: dict[tuple[int, int], int] = {}
    for fi, f in enumerate(frags):
        for fr in range(int(f.start), int(f.end) + 1):
            owner[(f.player_id, fr)] = fi

    want_set = set(want)
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    sel_rows = rows[np.isin(rows[:, COL.FRAME].astype(int), list(want_set))]
    sel_rows = sel_rows[~np.isnan(sel_rows[:, COL.ROI_X])]
    for r in sel_rows:
        fr = int(r[COL.FRAME])
        fi = owner.get((int(r[COL.PLAYER_ID]), fr))
        if fi is None:
            continue  # visible, but not inside any kept fragment
        by_frame.setdefault(fr, []).append(
            (fi, r[[COL.ROI_X, COL.ROI_Y, COL.ROI_W, COL.ROI_H]].astype(float))
        )

    index: dict[tuple[int, int], int] = {}
    lines = []
    for fr in sorted(by_frame):
        for order, (fi, box) in enumerate(by_frame[fr]):
            index[(fr, order)] = fi
            x, y, w, h = box
            # MOT det.txt is 1-based frame numbering
            lines.append(f"{fr + 1},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,-1,-1,-1")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return index


def run_bridge(img_dir: Path, det_file: Path, out_dir: Path, cfg: AppearanceConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(CAMEL_PY), str(BRIDGE),
        "--img-dir", str(img_dir),
        "--det-file", str(det_file),
        "--out-dir", str(out_dir),
        "--weights", str(cfg.weights),
        "--device", cfg.device,
        "--reid-model", cfg.reid_model,
        "--no-pose",
    ]
    subprocess.run(cmd, check=True)


def aggregate(
    out_dir: Path, index: dict[tuple[int, int], int], n_frags: int
) -> dict[int, np.ndarray]:
    """Mean-pool each fragment's sampled embeddings into one prototype.

    Mean pooling deliberately: multi-prototype representation already exists in
    `reid/representation.py` and belongs in the shipped path, not in a channel
    whose job here is to be a comparable INPUT alongside the others.
    """
    acc: dict[int, list[np.ndarray]] = {}
    for pkl in sorted(out_dir.glob("*.pkl")):
        frame_idx = int(pkl.stem)
        dets = pickle.loads(pkl.read_bytes())
        for order, det in enumerate(dets):
            fi = index.get((frame_idx, order))
            if fi is None:
                continue
            emb = np.asarray(det["appearance_embeddings"], dtype=np.float64).ravel()
            acc.setdefault(fi, []).append(emb)
    out = {}
    for fi, vs in acc.items():
        if fi < n_frags:
            v = np.mean(vs, axis=0)
            out[fi] = v / max(float(np.linalg.norm(v)), 1e-9)
    return out
