"""Run the trained action head over whole halves, writing the `(9,26,T)` contract.

Inference differs from training in the one way that matters: M is 26 SLOTS, not 5
sampled players. Every tactical role on both sides gets a row whether or not anyone
is currently visible in it, so the sequence stage always sees the same shape and can
learn what an empty slot looks like.

A slot's ROI at a frame comes from whichever row has that `(left_to_right, role_id)`
and an observable box. Where a substitution briefly puts two players in one slot, the
first row wins -- matching the reference, and a fact worth remembering when Phase 3
reads slot occupancy from MatchDay tracking instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from matchlab_core.pcbas.logits import (
    WindowAccumulator,
    logits_filename,
    save_logits,
    window_starts,
)
from matchlab_core.pcbas.schema import (
    FRAME,
    LEFT_TO_RIGHT,
    N_ROLES,
    N_SLOTS,
    ROI_HEIGHT,
    ROI_WIDTH,
    ROI_X,
    ROI_Y,
    ROLE_ID,
    slot_index,
)
from pydantic import BaseModel

from matchlab_train.datasets.footpass_clips import (
    NORM_MEAN,
    NORM_STD,
    ROI_COEFF,
    scale_box,
)
from matchlab_train.datasets.footpass_pcbas import list_halves, load_half
from matchlab_train.datasets.footpass_video import MatchVideo, match_video_path
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register

# Masked slots still need an in-bounds, non-degenerate box for roi_align.
INFERENCE_DUMMY_BOX = (50.0, 50.0, 72.5, 99.0)


class Params(BaseModel):
    h5_path: str = "data/footpass/tactical/val_tactical_data.h5"
    video_root: str = "data/footpass/videos_352x640"
    checkpoint: str = "data/weights/pcbas/action_head_selected.pt"
    out_dir: str = "data/footpass/logits/val"
    clip_length: int = 50
    stride: int = 25
    batch_size: int = 1
    device: str = "cuda"
    halves: list[str] | None = None  # default: every half in the file
    max_frames: int | None = None  # cap for smoke runs


def slot_rois_masks(
    rows: np.ndarray, frames: list[int], coeff: float = ROI_COEFF
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(26, T, 5)` ROIs and `(26, T)` masks for one window.

    `rows` is the tactical block for the half; only the requested frames are used.
    Column 0 of each ROI is the CLIP-LOCAL index, not the video frame.
    """
    t = len(frames)
    rois = np.zeros((N_SLOTS, t, 5), dtype=np.float32)
    masks = np.zeros((N_SLOTS, t), dtype=np.float32)
    rois[:, :, 0] = np.arange(t, dtype=np.float32)[None, :]
    rois[:, :, 1:] = np.array(INFERENCE_DUMMY_BOX, dtype=np.float32)

    frame_pos = {f: i for i, f in enumerate(frames)}
    observable = rows[~np.isnan(rows[:, ROI_X])]
    seen: set[tuple[int, int]] = set()
    for row in observable:
        pos = frame_pos.get(int(row[FRAME]))
        if pos is None:
            continue
        ltr, role = row[LEFT_TO_RIGHT], row[ROLE_ID]
        if np.isnan(ltr) or np.isnan(role) or not 1 <= int(role) <= N_ROLES:
            continue
        slot = slot_index(int(ltr), int(role))
        if (slot, pos) in seen:
            continue  # first row wins, as in the reference
        seen.add((slot, pos))
        x1, y1, x2, y2 = scale_box(
            row[ROI_X], row[ROI_Y], row[ROI_WIDTH], row[ROI_HEIGHT], coeff
        )
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        rois[slot, pos, 1:] = (x1, y1, x2, y2)
        masks[slot, pos] = 1.0
    return rois, masks


@register("pcbas-infer-logits")
class PCBASInferLogitsExperiment(Experiment):
    """Writes one `avg_logits_<half>.npy` per half, plus a coverage summary."""

    def run(self) -> dict:
        import torch
        from matchlab_core.pcbas.action_head import action_head_from_checkpoint

        p = Params(**self.config.params)
        workdir = self.workdir()
        device = torch.device(p.device if torch.cuda.is_available() else "cpu")

        # Built as the checkpoint was TRAINED. `ActionHead(pretrained=False)` is
        # always the conv arm, so this path could not load a transformer-arm
        # checkpoint at all -- the arm trained fine and was unscoreable.
        state = torch.load(p.checkpoint, map_location=device, weights_only=False)
        model = action_head_from_checkpoint(p.checkpoint, map_location=device)
        model = model.to(device).eval()
        temporal = (state.get("params") or {}).get("temporal", "conv")
        print(f"loaded {p.checkpoint} (epoch {state.get('epoch')}, temporal={temporal})")

        out_dir = Path(p.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        halves = p.halves or list_halves(p.h5_path)
        summary: list[dict] = []

        for key in halves:
            t0 = time.time()
            rows = load_half(p.h5_path, key)
            frames = rows[:, FRAME]
            first, last = int(np.nanmin(frames)), int(np.nanmax(frames))
            if p.max_frames is not None:
                last = min(last, first + p.max_frames - 1)

            video = MatchVideo(match_video_path(p.video_root, key))
            last = min(last, video.frame_count - 1)
            acc = WindowAccumulator(last - first + 1, offset=first)
            starts = window_starts(first, last, p.clip_length, p.stride)

            # Sort once and slice per window. Scanning the half's ~1.6M rows for
            # every one of ~3,000 windows dominated the wall clock -- 0.28s of a
            # 0.45s window against a 0.13s model forward.
            observable = rows[~np.isnan(rows[:, ROI_X])]
            observable = observable[np.argsort(observable[:, FRAME], kind="stable")]
            sorted_frames = observable[:, FRAME]

            for i, start in enumerate(starts):
                clip = video.read_clip(start, p.clip_length)
                window_frames = list(range(start, start + p.clip_length))
                lo = int(np.searchsorted(sorted_frames, start, side="left"))
                hi = int(np.searchsorted(sorted_frames, start + p.clip_length, side="left"))
                rois, masks = slot_rois_masks(observable[lo:hi], window_frames)

                # Normalise on the GPU, not the CPU. The float32 conversion of a
                # (50,352,640,3) clip is 135 MB through several numpy passes plus a
                # non-contiguous permute; uploading uint8 moves 34 MB instead and
                # does the arithmetic where it is free. Measured: 0.32s -> 0.15s per
                # window, which halves a full VAL inference pass.
                video_t = (
                    torch.from_numpy(clip)
                    .to(device, non_blocking=True)
                    .permute(3, 0, 1, 2)[None]
                    .float()
                    .div_(255.0)
                    .sub_(NORM_MEAN)
                    .div_(NORM_STD)
                )
                rois_t = torch.from_numpy(rois)[None].to(device)
                masks_t = torch.from_numpy(masks)[None].to(device)

                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    pred = model(video_t, rois_t, masks_t)  # (1, 9, 26, T)
                acc.add(pred[0].float().cpu().numpy(), start)

                if i % 200 == 0:
                    print(f"{key} [{i + 1}/{len(starts)}]", flush=True)

            video.close()
            # Pad back to absolute frame indexing: frame f lives at index f.
            full = np.zeros((9, N_SLOTS, last + 1), dtype=np.float16)
            full[:, :, first : last + 1] = acc.result()
            save_logits(full, out_dir / logits_filename(key))

            coverage = acc.coverage
            summary.append(
                {
                    "half": key,
                    "frames": int(last - first + 1),
                    "first_frame": first,
                    "last_frame": last,
                    "windows": len(starts),
                    "uncovered_frames": int((coverage == 0).sum()),
                    "seconds": time.time() - t0,
                }
            )
            print(f"{key}: {summary[-1]}", flush=True)
            (workdir / "summary.json").write_text(json.dumps(summary, indent=2))

        result = {"halves": summary, "out_dir": str(out_dir)}
        self.write_result(workdir, result)
        return result
