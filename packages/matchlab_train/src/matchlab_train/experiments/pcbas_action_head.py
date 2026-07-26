"""Train the TAAD action head on FOOTPASS (SPO-96, Phase 1).

Reference recipe, matched unless a deviation is recorded here:
clip_length 50, nb_tracklets 4 (so M=5), max_nb_samples_per_class 500 **resampled
every epoch**, 20 epochs, AdamW with lr 5e-5 on the backbone / 1e-3 on the head,
weight decay 1e-4 on weights and 0 on norms/biases, 50-step linear warmup, AMP fp16.

**Recorded deviation -- batch size.** The reference uses batch 6 with 8 accumulation
steps (effective 48). This machine is a 16 GiB RTX 4060 Ti sharing ~3.7 GiB with the
desktop, and a single (1,3,50,352,640) clip peaks at 6.2 GiB; batch 2 OOMs. So the
micro-batch is 1 and `accum_steps` rises to keep the effective batch identical.
The only real consequence is BatchNorm statistics, which are now computed over one
clip -- still 50x44x80 spatial-temporal samples per channel for the 3D norms and
M*T = 250 for the 1D norm, so the estimates stay well conditioned.

The loss is cross-entropy over the DILATED labels (a single-frame target in a
50-frame clip is too sparse to learn from), multiplied by the observability mask and
by the reference's flat class weights [0.05 background, 0.95 x 8 actions]. Class
balance is the SAMPLER's job, not the loss's -- see `class_weights`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from matchlab_core.pcbas.schema import N_CLASSES
from pydantic import BaseModel

from matchlab_train.datasets.footpass_clips import (
    FootpassClipDataset,
    build_anchors,
    class_histogram,
    load_anchors,
    save_anchors,
)
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register


class Params(BaseModel):
    h5_path: str = "data/footpass/tactical/train_tactical_data.h5"
    video_root: str = "data/footpass/videos_352x640"
    val_h5_path: str = "data/footpass/tactical/val_tactical_data.h5"
    val_video_root: str = "data/footpass/videos_352x640"
    anchors_cache: str = "data/footpass/anchors_train.json"
    val_anchors_cache: str = "data/footpass/anchors_val.json"
    checkpoint_dir: str = "data/weights/pcbas"

    clip_length: int = 50
    nb_tracklets: int = 4
    max_samples_per_class: int = 500
    label_dilation: int = 1

    epochs: int = 20
    batch_size: int = 1
    accum_steps: int = 48  # effective batch 48, matching the reference's 6 x 8
    num_workers: int = 6
    lr_backbone: float = 5e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 50
    background_weight: float = 0.05
    action_weight: float = 0.95

    # Validation is expensive (a full pass over the balanced VAL sample), so it runs
    # every `val_every` epochs and on the last one.
    val_every: int = 2
    max_val_samples_per_class: int = 60

    device: str = "cuda"
    seed: int = 345
    # Wall-clock budget. Training stops cleanly at the end of the first epoch that
    # would exceed it, checkpointing as usual -- an unattended run that overruns is
    # worse than a shorter one that reports honestly how far it got.
    max_hours: float | None = None


def _param_groups(model, params: Params) -> list[dict]:
    """Four groups: backbone vs head, decayed vs not.

    Norm and bias parameters are excluded from weight decay -- decaying a BatchNorm
    scale slowly erases the pretrained feature calibration.
    """
    backbone_names = ("stem", "block2", "block3", "block4")
    groups: dict[str, list] = {
        "backbone_decay": [],
        "backbone_nodecay": [],
        "head_decay": [],
        "head_nodecay": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        where = "backbone" if name.startswith(backbone_names) else "head"
        decay = "nodecay" if (p.ndim <= 1 or "bn" in name or "norm" in name) else "decay"
        groups[f"{where}_{decay}"].append(p)
    return [
        {
            "params": groups["backbone_decay"],
            "lr": params.lr_backbone,
            "weight_decay": params.weight_decay,
            "name": "backbone_decay",
        },
        {
            "params": groups["backbone_nodecay"],
            "lr": params.lr_backbone,
            "weight_decay": 0.0,
            "name": "backbone_nodecay",
        },
        {
            "params": groups["head_decay"],
            "lr": params.lr_head,
            "weight_decay": params.weight_decay,
            "name": "head_decay",
        },
        {
            "params": groups["head_nodecay"],
            "lr": params.lr_head,
            "weight_decay": 0.0,
            "name": "head_nodecay",
        },
    ]


def class_weights(background_weight: float, action_weight: float):
    """The reference's flat weighting: background down, all 8 actions equal.

    Deliberately NOT inverse-frequency. Class balance is handled by the SAMPLER
    (`max_samples_per_class`, resampled every epoch); adding a second frequency
    correction on top double-counts it. A first attempt here used softened inverse
    frequency and produced a tackle:pass ratio of 10:1 on top of an already-balanced
    epoch -- a hyperparameter change dressed up as a bug fix, which would have made
    a reproduction miss unattributable.

    Background is weighted 0.05 rather than dropped because it is the correct answer
    for ~99% of (player, frame) cells and the model must still learn to say it.
    """
    import torch

    return torch.tensor(
        [background_weight, *([action_weight] * (N_CLASSES - 1))], dtype=torch.float32
    )


def masked_weighted_ce(logits, dilated, masks, weights):
    """Per-cell cross-entropy, zeroed where the player is not observed.

    The mask multiply is not optional. A masked cell's pooled feature is forced to
    exactly zero, so including it would train the classifier to map the zero vector
    to background -- on roughly 60% of cells, since that is how often a player is
    off-screen. The reference divides by the full cell count, not by the observed
    count, so the effective learning rate scales with observability; reproduced.
    """
    import torch

    per_cell = torch.nn.functional.cross_entropy(logits, dilated, reduction="none")
    return (per_cell * masks * weights[dilated]).mean()


def _anchors(h5_path: str, cache: str):
    path = Path(cache)
    if path.is_file():
        return load_anchors(path)
    anchors = build_anchors(h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_anchors(anchors, path)
    return anchors


@register("pcbas-action-head")
class PCBASActionHeadExperiment(Experiment):
    """Trains `ActionHead` and writes a checkpoint plus a per-epoch history."""

    def run(self) -> dict:
        import torch
        from matchlab_core.pcbas.action_head import ActionHead
        from torch.utils.data import DataLoader

        p = Params(**self.config.params)
        workdir = self.workdir()
        torch.manual_seed(self.config.seed)

        train_anchors = _anchors(p.h5_path, p.anchors_cache)
        val_anchors = _anchors(p.val_h5_path, p.val_anchors_cache)
        hist = class_histogram(train_anchors)
        print(f"train anchors: {len(train_anchors):,} {hist}")

        train_ds = FootpassClipDataset(
            p.h5_path,
            p.video_root,
            train_anchors,
            clip_length=p.clip_length,
            nb_tracklets=p.nb_tracklets,
            max_samples_per_class=p.max_samples_per_class,
            label_dilation=p.label_dilation,
            train=True,
            seed=p.seed,
        )
        val_ds = FootpassClipDataset(
            p.val_h5_path,
            p.val_video_root,
            val_anchors,
            clip_length=p.clip_length,
            nb_tracklets=p.nb_tracklets,
            max_samples_per_class=p.max_val_samples_per_class,
            label_dilation=p.label_dilation,
            train=False,
            seed=p.seed,
        )

        device = torch.device(p.device if torch.cuda.is_available() else "cpu")
        model = ActionHead().to(device)
        opt = torch.optim.AdamW(_param_groups(model, p))
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        weights = class_weights(p.background_weight, p.action_weight).to(device)

        step = 0

        def warmup(_: int) -> float:
            return min(1.0, (step + 1) / max(p.warmup_steps, 1))

        ckpt_dir = Path(p.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        started = time.time()
        best_val = float("inf")

        for epoch in range(1, p.epochs + 1):
            if epoch > 1:
                train_ds.resample()
            loader = DataLoader(
                train_ds,
                batch_size=p.batch_size,
                shuffle=True,
                num_workers=p.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=False,
            )
            model.train()
            epoch_loss, n_batches = 0.0, 0
            opt.zero_grad(set_to_none=True)
            t0 = time.time()

            for i, (video, rois, masks, _sharp, dilated) in enumerate(loader):
                video = video.to(device, non_blocking=True)
                rois = rois.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                dilated = dilated.to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(video, rois, masks)  # (B, 9, M, T)
                    loss = masked_weighted_ce(logits, dilated, masks, weights)

                scaler.scale(loss / p.accum_steps).backward()
                epoch_loss += float(loss.detach())
                n_batches += 1

                if (i + 1) % p.accum_steps == 0:
                    for g in opt.param_groups:
                        base = p.lr_backbone if g["name"].startswith("backbone") else p.lr_head
                        g["lr"] = base * warmup(step)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1

                if i % 200 == 0:
                    rate = (i + 1) / max(time.time() - t0, 1e-6)
                    print(
                        f"epoch {epoch} [{i + 1}/{len(loader)}] "
                        f"loss {epoch_loss / n_batches:.4f} {rate:.1f} clip/s",
                        flush=True,
                    )

            record = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(n_batches, 1),
                "clips": n_batches,
                "seconds": time.time() - t0,
            }

            if epoch % p.val_every == 0 or epoch == p.epochs:
                record["val_loss"] = self._validate(model, val_ds, p, device, weights)
                if record["val_loss"] < best_val:
                    best_val = record["val_loss"]
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch, "params": p.model_dump()},
                        ckpt_dir / "action_head_best.pt",
                    )

            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "params": p.model_dump()},
                ckpt_dir / "action_head_last.pt",
            )
            history.append(record)
            (workdir / "history.json").write_text(json.dumps(history, indent=2))
            print(f"epoch {epoch}: {record}", flush=True)

            elapsed_h = (time.time() - started) / 3600
            if p.max_hours is not None and elapsed_h >= p.max_hours:
                print(
                    f"stopping after epoch {epoch}: {elapsed_h:.2f}h reached the "
                    f"{p.max_hours}h budget ({p.epochs - epoch} epochs not run)",
                    flush=True,
                )
                break

        train_ds.close()
        val_ds.close()
        result = {
            "history": history,
            "epochs_run": len(history),
            "epochs_planned": p.epochs,
            "best_val_loss": best_val if best_val < float("inf") else None,
            "checkpoint": str(ckpt_dir / "action_head_best.pt"),
            "train_anchor_histogram": hist,
            "hours": (time.time() - started) / 3600,
        }
        self.write_result(workdir, result)
        return result

    def _validate(self, model, val_ds, p: Params, device, weights) -> float:
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            val_ds, batch_size=p.batch_size, shuffle=False, num_workers=p.num_workers
        )
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for video, rois, masks, _sharp, dilated in loader:
                video, rois = video.to(device), rois.to(device)
                masks, dilated = masks.to(device), dilated.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(video, rois, masks)
                    loss = masked_weighted_ce(logits, dilated, masks, weights)
                total += float(loss)
                n += 1
        return total / max(n, 1)
