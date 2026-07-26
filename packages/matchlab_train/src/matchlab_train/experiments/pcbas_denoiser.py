"""Train the DST sequence stage on stage-1 logits (SPO-96, Phase 1).

Reference recipe: framespan 750, d_model 512, 6+6 layers, 8 heads, dropout 0.1,
AdamW lr 2.5e-4 with no weight decay, 1000-step warmup, 15 epochs, batch 96.

**This stage cannot be trained until stage-1 logits exist for TRAIN**, which means
running the action head over all 96 TRAIN halves. That is the expensive prerequisite,
not the training itself, and it is why this experiment fails loudly with a useful
message rather than quietly producing an untrained model when the logits are absent.

The loss is three cross-entropies -- action, role slot, and timestamp -- summed with
configurable weights. They are reported separately as well as summed, because they
fail differently: a model that gets every action right and every player wrong scores
the same total as the reverse, and only one of those is a player-centric spotter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from matchlab_core.pcbas.denoiser import (
    ACTION_VOCAB,
    FRAMESPAN,
    OUTPUT_DIM,
    ROLE_VOCAB,
    DSTDenoiser,
)
from pydantic import BaseModel

from matchlab_train.datasets.footpass_windows import (
    FootpassWindowDataset,
    one_hot_frames,
)
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register


class Params(BaseModel):
    h5_path: str = "data/footpass/tactical/train_tactical_data.h5"
    logits_dir: str = "data/footpass/logits/train"
    val_h5_path: str = "data/footpass/tactical/val_tactical_data.h5"
    val_logits_dir: str = "data/footpass/logits/val"
    checkpoint_dir: str = "data/weights/pcbas"

    framespan: int = FRAMESPAN
    stride: int | None = None
    hidden_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    encoder: str = "flat"  # "flat" (reference) or "attn" (PAVE-style)

    epochs: int = 15
    batch_size: int = 8
    accum_steps: int = 12  # effective 96, matching the reference
    num_workers: int = 4
    lr: float = 2.5e-4
    warmup_steps: int = 1000

    action_loss_weight: float = 1.0
    role_loss_weight: float = 1.0
    timestamp_loss_weight: float = 1.0

    device: str = "cuda"
    seed: int = 345
    max_hours: float | None = None


def collate(batch, framespan: int):
    """Pad variable-length target sequences and build both key-padding masks.

    Encoder sources are all `framespan` long so they need no padding; targets are
    one token per event plus SOS/EOS, which varies per window.
    """
    import numpy as np
    import torch

    features, frames, actions, roles, event_frames = zip(*batch)
    max_tgt = max(a.shape[0] for a in actions)

    src = torch.stack(
        [
            torch.cat(
                [f, torch.from_numpy(one_hot_frames(np.arange(f.shape[0]), framespan))],
                dim=-1,
            )
            for f in features
        ]
    )
    src_frames = torch.stack(list(frames))

    tgt = torch.zeros(len(batch), max_tgt, OUTPUT_DIM + framespan + 2)
    tgt_frames = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    tgt_pad = torch.ones(len(batch), max_tgt, dtype=torch.bool)
    for i, (a, r, ef) in enumerate(zip(actions, roles, event_frames)):
        n = a.shape[0]
        tgt[i, :n, :ACTION_VOCAB] = a
        tgt[i, :n, ACTION_VOCAB:OUTPUT_DIM] = r
        tgt[i, :n, OUTPUT_DIM:] = torch.from_numpy(
            one_hot_frames(ef.numpy(), framespan)
        )
        tgt_frames[i, :n] = ef
        tgt_pad[i, :n] = False
    return src, src_frames, tgt, tgt_frames, tgt_pad


@register("pcbas-denoiser")
class PCBASDenoiserExperiment(Experiment):
    def run(self) -> dict:
        import torch
        from torch.utils.data import DataLoader

        p = Params(**self.config.params)
        workdir = self.workdir()
        torch.manual_seed(self.config.seed)

        if not any(Path(p.logits_dir).glob("avg_logits_*.npy")):
            raise RuntimeError(
                f"no stage-1 logits under {p.logits_dir}. The sequence stage trains "
                f"on what the action head ACTUALLY predicts, so run "
                f"`pcbas-infer-logits` over the TRAIN halves first."
            )

        train_ds = FootpassWindowDataset(
            p.h5_path,
            p.logits_dir,
            framespan=p.framespan,
            stride=p.stride,
            train=True,
            seed=p.seed,
        )
        val_ds = FootpassWindowDataset(
            p.val_h5_path,
            p.val_logits_dir,
            framespan=p.framespan,
            stride=p.framespan,
            train=False,
            seed=p.seed,
        )
        print(f"{len(train_ds):,} train windows, {len(val_ds):,} val windows")

        device = torch.device(p.device if torch.cuda.is_available() else "cpu")
        model = DSTDenoiser(
            framespan=p.framespan,
            hidden_dim=p.hidden_dim,
            n_heads=p.n_heads,
            n_enc_layers=p.n_layers,
            n_dec_layers=p.n_layers,
            dropout=p.dropout,
            encoder=p.encoder,  # type: ignore[arg-type]
        ).to(device)
        # The reference applies no weight decay anywhere in this stage.
        opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=0.0)

        ckpt_dir = Path(p.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        started = time.time()
        step = 0

        def loader(ds, shuffle):
            return DataLoader(
                ds,
                batch_size=p.batch_size,
                shuffle=shuffle,
                num_workers=p.num_workers,
                collate_fn=lambda b: collate(b, p.framespan),
            )

        for epoch in range(1, p.epochs + 1):
            model.train()
            totals = {"action": 0.0, "role": 0.0, "timestamp": 0.0, "total": 0.0}
            n = 0
            t0 = time.time()
            opt.zero_grad(set_to_none=True)

            for i, batch in enumerate(loader(train_ds, True)):
                losses = self._step(model, batch, device, p)
                (losses["total"] / p.accum_steps).backward()
                for k in totals:
                    totals[k] += float(losses[k].detach())
                n += 1
                if (i + 1) % p.accum_steps == 0:
                    for g in opt.param_groups:
                        g["lr"] = p.lr * min(1.0, (step + 1) / max(p.warmup_steps, 1))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                if i % 200 == 0:
                    print(
                        f"epoch {epoch} [{i + 1}] "
                        + " ".join(f"{k} {v / max(n, 1):.4f}" for k, v in totals.items()),
                        flush=True,
                    )

            record = {"epoch": epoch, "seconds": time.time() - t0, "windows": n}
            record.update({f"train_{k}": v / max(n, 1) for k, v in totals.items()})
            record.update(self._validate(model, loader(val_ds, False), device, p))
            history.append(record)
            (workdir / "history.json").write_text(json.dumps(history, indent=2))
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "params": p.model_dump()},
                ckpt_dir / f"denoiser_{p.encoder}_last.pt",
            )
            print(f"epoch {epoch}: {record}", flush=True)

            if p.max_hours is not None and (time.time() - started) / 3600 >= p.max_hours:
                print(f"stopping after epoch {epoch}: budget reached", flush=True)
                break

        result = {
            "history": history,
            "epochs_run": len(history),
            "epochs_planned": p.epochs,
            "encoder": p.encoder,
            "checkpoint": str(ckpt_dir / f"denoiser_{p.encoder}_last.pt"),
            "hours": (time.time() - started) / 3600,
        }
        self.write_result(workdir, result)
        return result

    @staticmethod
    def _step(model, batch, device, p: Params) -> dict:
        """Three cross-entropies, reported separately.

        A model with every action right and every player wrong scores the same total
        as the reverse, and only one of those is a player-centric spotter.
        """
        import torch

        src, src_frames, tgt, tgt_frames, tgt_pad = (x.to(device) for x in batch)
        # Teacher forcing: predict token t+1 from tokens 0..t.
        out = model(
            src,
            tgt[:, :-1],
            src_frames,
            tgt_frames[:, :-1],
            tgt_key_padding_mask=tgt_pad[:, :-1],
        )  # (B, 37 + framespan + 2, T-1)

        target = tgt[:, 1:]
        keep = ~tgt_pad[:, 1:]
        action_t = target[..., :ACTION_VOCAB].argmax(-1)
        role_t = target[..., ACTION_VOCAB:OUTPUT_DIM].argmax(-1)
        stamp_t = target[..., OUTPUT_DIM:].argmax(-1)

        def ce(logits, targets):
            per = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            return (per * keep).sum() / keep.sum().clamp(min=1)

        action = ce(out[:, :ACTION_VOCAB], action_t)
        role = ce(out[:, ACTION_VOCAB:OUTPUT_DIM], role_t)
        stamp = ce(out[:, OUTPUT_DIM:], stamp_t)
        total = (
            p.action_loss_weight * action
            + p.role_loss_weight * role
            + p.timestamp_loss_weight * stamp
        )
        return {"action": action, "role": role, "timestamp": stamp, "total": total}

    def _validate(self, model, loader, device, p: Params) -> dict:
        import torch

        model.eval()
        totals = {"action": 0.0, "role": 0.0, "timestamp": 0.0, "total": 0.0}
        n = 0
        with torch.no_grad():
            for batch in loader:
                losses = self._step(model, batch, device, p)
                for k in totals:
                    totals[k] += float(losses[k])
                n += 1
        model.train()
        return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}


assert ROLE_VOCAB + ACTION_VOCAB == OUTPUT_DIM
