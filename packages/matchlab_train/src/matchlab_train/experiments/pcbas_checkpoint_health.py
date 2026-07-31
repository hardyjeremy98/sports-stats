"""Fast health check on an action-head checkpoint, and the checkpoint SELECTOR (SPO-96).

Guards the failure this task is most exposed to: **a falling loss is not evidence of
learning here.** Background is ~99% of (player, frame) cells and is weighted 0.05, so
a model that predicts background everywhere reaches a low loss and scores zero F1.
Nothing in the training log distinguishes that from real progress.

This samples a handful of clips and reports what the model actually emits — the
predicted class distribution over observed cells, which action classes it has learned
to produce at all, and how often it emits the anchor's own class on the anchor player.
It needs no full-half inference, so it can run against a mid-training checkpoint while
training continues.

**It also computes the real metric on the sampled clips**, which matters more than it
sounds. Validation loss is a bad selection criterion for this task: it is dominated by
background cells. Measured on the live run, val loss got WORSE from epoch 4 to 6
(0.0385 -> 0.0399) while anchor recovery improved 17/40 -> 21/40 and the model went
from emitting 7 of 8 action classes to all 8. Selecting on val loss would have frozen
`best` at a demonstrably worse model.

The clip-level micro-F1 here reuses `decode_logits` and `score_events` -- the same
NMS and the same matching rule the reference-validated gate uses -- so it is a small,
cheap version of the real thing rather than a different measure. It is scored over a
handful of clips, so it is a SELECTOR, not the gate. `pcbas-score` on full halves is
the gate.
"""

from __future__ import annotations

from collections import Counter

from matchlab_core.pcbas.decode import DEFAULT_NMS_WINDOW, decode_logits
from matchlab_core.pcbas.eval import DEFAULT_DELTA, score_events
from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.schema import ACTION_CLASSES, CLASS_NAMES, N_SLOTS
from pydantic import BaseModel

from matchlab_train.datasets.footpass_clips import FootpassClipDataset, load_anchors
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register


class Params(BaseModel):
    checkpoint: str = "data/weights/pcbas/action_head_last.pt"
    h5_path: str = "data/footpass/tactical/val_tactical_data.h5"
    video_root: str = "data/footpass/videos_352x640"
    anchors_cache: str = "data/footpass/anchors_val.json"
    clips: int = 40
    samples_per_class: int = 6
    nms_window: int = DEFAULT_NMS_WINDOW
    delta: int = DEFAULT_DELTA
    device: str = "cuda"
    seed: int = 345


def clip_events(
    logits, masks, *, nms_window: int = DEFAULT_NMS_WINDOW
) -> list[PCBASEvent]:
    """Decode one clip's `(9, M, T)` output, treating the player index as the slot.

    Valid because the sampler's M is at most `N_SLOTS`; the indices are clip-local
    player numbers, not real tactical slots, but the metric only needs them to be
    consistent between prediction and ground truth.
    """
    import numpy as np

    padded = np.zeros((logits.shape[0], N_SLOTS, logits.shape[2]), dtype=np.float32)
    padded[:, : logits.shape[1]] = logits
    padded[0, logits.shape[1] :] = 1e4  # unused slots are confidently background
    return decode_logits(padded, nms_window=nms_window)


def label_events(sharp) -> list[PCBASEvent]:
    """Ground-truth events for one clip, from its sharp (undilated) labels."""
    import numpy as np

    out: list[PCBASEvent] = []
    arr = sharp.numpy() if hasattr(sharp, "numpy") else np.asarray(sharp)
    for player, frame in zip(*np.nonzero(arr)):
        out.append(
            PCBASEvent(
                frame_idx=int(frame),
                left_to_right=int(player) // 13,
                slot=int(player),
                class_id=int(arr[player, frame]),
            )
        )
    return out


def summarise(cell_counts: Counter, anchor_hits: int, n_clips: int) -> dict:
    """Turn raw counts into the three numbers worth reading."""
    total = sum(cell_counts.values())
    action_cells = sum(v for k, v in cell_counts.items() if k != "background")
    return {
        "cells": total,
        "background_fraction": cell_counts["background"] / total if total else 0.0,
        "action_fraction": action_cells / total if total else 0.0,
        "classes_emitted": sorted(k for k in cell_counts if k != "background"),
        "classes_never_emitted": sorted(
            CLASS_NAMES[c] for c in ACTION_CLASSES if cell_counts[CLASS_NAMES[c]] == 0
        ),
        "anchor_class_recovered": anchor_hits,
        "clips": n_clips,
        "collapsed_to_background": action_cells == 0,
    }


@register("pcbas-checkpoint-health")
class PCBASCheckpointHealthExperiment(Experiment):
    def run(self) -> dict:
        import torch
        from matchlab_core.pcbas.action_head import action_head_from_checkpoint

        p = Params(**self.config.params)
        workdir = self.workdir()
        device = torch.device(p.device if torch.cuda.is_available() else "cpu")

        state = torch.load(p.checkpoint, map_location=device, weights_only=False)
        # Built as the checkpoint was TRAINED -- a transformer-arm checkpoint cannot
        # load into a conv-arm model, and constructing ActionHead() at its default
        # made A1's checkpoints unscoreable.
        model = action_head_from_checkpoint(p.checkpoint, map_location=device)
        model = model.to(device).eval()

        ds = FootpassClipDataset(
            p.h5_path,
            p.video_root,
            load_anchors(p.anchors_cache),
            max_samples_per_class=p.samples_per_class,
            train=False,
            seed=p.seed,
        )
        cells: Counter = Counter()
        gt_events: list[PCBASEvent] = []
        pred_events: list[PCBASEvent] = []
        hits = 0
        n = min(len(ds), p.clips)
        for i in range(n):
            video, rois, masks, sharp, _dil = ds[i]
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                out = model(video[None].to(device), rois[None].to(device), masks[None].to(device))
            raw = out[0].float().cpu()
            predicted = raw.argmax(0)  # (M, T)
            for c in predicted[masks.bool()].numpy():
                cells[CLASS_NAMES[int(c)]] += 1

            # Offset each clip's frames so clips never match across one another.
            offset = i * 10_000
            for e in clip_events(raw.numpy(), masks, nms_window=p.nms_window):
                pred_events.append(e.model_copy(update={"frame_idx": e.frame_idx + offset}))
            for e in label_events(sharp):
                gt_events.append(e.model_copy(update={"frame_idx": e.frame_idx + offset}))
            # The anchor player is the LAST row by construction in the sampler.
            if (predicted[-1] == ds.sampled[i].class_id).any():
                hits += 1
        ds.close()

        report = score_events(gt_events, pred_events, delta=p.delta, identity="slot")
        result = summarise(cells, hits, n)
        result["micro_f1"] = report.micro_f1
        result["macro_f1"] = report.macro_f1
        result["tp"], result["fp"], result["fn"] = report.tp, report.fp, report.fn
        result["per_class_f1"] = {
            m.class_name: round(m.f1, 4) for m in report.per_class.values()
        }
        result["epoch"] = state.get("epoch")
        result["checkpoint"] = p.checkpoint
        result["cell_distribution"] = dict(cells.most_common())
        self.write_result(workdir, result)
        print(
            f"epoch {result['epoch']}: micro-F1 {report.micro_f1:.4f} "
            f"macro-F1 {report.macro_f1:.4f} (TP {report.tp} FP {report.fp} FN {report.fn}) | "
            f"{result['background_fraction']:.1%} background, "
            f"anchor recovered {hits}/{n}, "
            f"never emitted: {result['classes_never_emitted'] or 'none'}"
        )
        return result
