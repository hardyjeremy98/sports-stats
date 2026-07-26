"""Fast health check on an action-head checkpoint (SPO-96).

Guards the failure this task is most exposed to: **a falling loss is not evidence of
learning here.** Background is ~99% of (player, frame) cells and is weighted 0.05, so
a model that predicts background everywhere reaches a low loss and scores zero F1.
Nothing in the training log distinguishes that from real progress.

This samples a handful of clips and reports what the model actually emits — the
predicted class distribution over observed cells, which action classes it has learned
to produce at all, and how often it emits the anchor's own class on the anchor player.
It needs no full-half inference, so it can run against a mid-training checkpoint while
training continues.

It is a smoke check, not a metric. `pcbas-score` is the gate.
"""

from __future__ import annotations

from collections import Counter

from matchlab_core.pcbas.schema import ACTION_CLASSES, CLASS_NAMES
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
    device: str = "cuda"
    seed: int = 345


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
        from matchlab_core.pcbas.action_head import ActionHead

        p = Params(**self.config.params)
        workdir = self.workdir()
        device = torch.device(p.device if torch.cuda.is_available() else "cpu")

        state = torch.load(p.checkpoint, map_location=device, weights_only=False)
        model = ActionHead(pretrained=False).to(device).eval()
        model.load_state_dict(state["model"])

        ds = FootpassClipDataset(
            p.h5_path,
            p.video_root,
            load_anchors(p.anchors_cache),
            max_samples_per_class=p.samples_per_class,
            train=False,
            seed=p.seed,
        )
        cells: Counter = Counter()
        hits = 0
        n = min(len(ds), p.clips)
        for i in range(n):
            video, rois, masks, _sharp, _dil = ds[i]
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                out = model(video[None].to(device), rois[None].to(device), masks[None].to(device))
            predicted = out[0].float().cpu().argmax(0)  # (M, T)
            for c in predicted[masks.bool()].numpy():
                cells[CLASS_NAMES[int(c)]] += 1
            # The anchor player is the LAST row by construction in the sampler.
            if (predicted[-1] == ds.sampled[i].class_id).any():
                hits += 1
        ds.close()

        result = summarise(cells, hits, n)
        result["epoch"] = state.get("epoch")
        result["checkpoint"] = p.checkpoint
        result["cell_distribution"] = dict(cells.most_common())
        self.write_result(workdir, result)
        print(
            f"epoch {result['epoch']}: {result['background_fraction']:.1%} background, "
            f"anchor class recovered {hits}/{n}, "
            f"never emitted: {result['classes_never_emitted'] or 'none'}"
        )
        return result
