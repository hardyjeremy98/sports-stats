"""Run the trained DST over VAL and score the result — the TAAD→DST arm (SPO-96).

Stage 2 does not emit per-frame logits; it autoregressively emits an ordered EVENT
LIST per 30-second window. Turning that into a scoreable half needs three things the
stage-1 path never had to do:

1. **Window-local timestamps become absolute frames.** Token timestamp index 0 is
   reserved for SOS, so window frame `f` is encoded as `f + 1`; the absolute frame is
   `window_start + (timestamp - 1)`. An off-by-one here shifts every event by one
   frame, which is invisible at delta=12 and would quietly corrupt any later
   tighter-tolerance analysis.
2. **Overlapping windows are deduplicated.** Windows advance by `stride < framespan`,
   so the same real action is emitted by several windows. Without dedup those become
   false positives against a ground-truth event that can only match once.
3. **Action indices shift back by one.** The decoder's action vocabulary is 0-7 for
   classes 1-8, with 8 = SOS and 9 = EOS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matchlab_core.pcbas.decode import DEFAULT_NMS_WINDOW, suppress
from matchlab_core.pcbas.denoiser import EOS_ACTION, FRAMESPAN, NEUTRAL_ROLE, DSTDenoiser
from matchlab_core.pcbas.eval import (
    DEFAULT_CONF_THRESH,
    DEFAULT_DELTA,
    score_halves,
)
from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.schema import N_CLASSES, N_SLOTS, slot_to_role
from pydantic import BaseModel

from matchlab_train.datasets.footpass_pcbas import (
    assign_shirts,
    half_to_events,
    load_half,
    slot_shirt_table,
)
from matchlab_train.datasets.footpass_windows import (
    FootpassWindowDataset,
    one_hot_frames,
)
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register


class Params(BaseModel):
    h5_path: str = "data/footpass/tactical/val_tactical_data.h5"
    logits_dir: str = "data/footpass/logits/val"
    checkpoint: str = "data/weights/pcbas/denoiser_flat_last.pt"
    framespan: int = FRAMESPAN
    stride: int = 375
    batch_size: int = 8
    max_events: int = 64
    dedup_window: int = DEFAULT_NMS_WINDOW
    delta: int = DEFAULT_DELTA
    conf_thresh: float = DEFAULT_CONF_THRESH
    device: str = "cuda"
    export_json: str | None = None


def tokens_to_events(
    tokens: list[tuple[int, int, int, float]], window_start: int
) -> list[PCBASEvent]:
    """Decoder tokens -> `PCBASEvent`s at absolute video frames.

    Drops EOS, the neutral role, and any timestamp of 0 (which is SOS's slot, not a
    real frame).
    """
    events: list[PCBASEvent] = []
    for action, role, timestamp, score in tokens:
        if action >= EOS_ACTION or role >= NEUTRAL_ROLE or timestamp == 0:
            continue
        class_id = action + 1
        if not 1 <= class_id < N_CLASSES or not 0 <= role < N_SLOTS:
            continue
        left_to_right, role_id = slot_to_role(role)
        events.append(
            PCBASEvent(
                frame_idx=window_start + timestamp - 1,
                left_to_right=left_to_right,
                role_id=role_id,
                slot=role,
                class_id=class_id,
                score=score,
            )
        )
    return events


def dedupe(events: list[PCBASEvent], window: int) -> list[PCBASEvent]:
    """Collapse the same action re-emitted by overlapping windows.

    Grouped by (slot, class) and suppressed by frame proximity, exactly as stage-1 NMS
    does. Without this, a stride of 375 against a framespan of 750 means every action
    is emitted about twice, and the duplicate is a guaranteed false positive because a
    ground-truth event can only be matched once.
    """
    by_group: dict[tuple[int, int], list[PCBASEvent]] = {}
    for e in events:
        by_group.setdefault((e.slot, e.class_id), []).append(e)

    kept: list[PCBASEvent] = []
    for group in by_group.values():
        frames = np.array([e.frame_idx for e in group])
        scores = np.array([e.score for e in group])
        by_frame = {e.frame_idx: e for e in sorted(group, key=lambda e: -e.score)}
        for frame in suppress(frames, scores, window):
            kept.append(by_frame[frame])
    kept.sort(key=lambda e: (e.frame_idx, e.slot))
    return kept


@register("pcbas-denoise-infer")
class PCBASDenoiseInferExperiment(Experiment):
    """Decode VAL with the trained DST and score it against ground truth."""

    def run(self) -> dict:
        import torch

        p = Params(**self.config.params)
        workdir = self.workdir()
        device = torch.device(p.device if torch.cuda.is_available() else "cpu")

        state = torch.load(p.checkpoint, map_location=device, weights_only=False)
        saved = state.get("params", {})
        model = DSTDenoiser(
            framespan=saved.get("framespan", p.framespan),
            hidden_dim=saved.get("hidden_dim", 512),
            n_heads=saved.get("n_heads", 8),
            n_enc_layers=saved.get("n_layers", 6),
            n_dec_layers=saved.get("n_layers", 6),
            dropout=0.0,
            encoder=saved.get("encoder", "flat"),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"loaded {p.checkpoint} (epoch {state.get('epoch')}, "
              f"encoder {saved.get('encoder', 'flat')})")

        ds = FootpassWindowDataset(
            p.h5_path, p.logits_dir, framespan=p.framespan, stride=p.stride, train=False
        )
        print(f"{len(ds):,} windows over {len(ds.keys)} halves")

        raw: dict[str, list[PCBASEvent]] = {k: [] for k in ds.keys}
        for start in range(0, len(ds), p.batch_size):
            batch = [ds.sample(i) for i in range(start, min(start + p.batch_size, len(ds)))]
            src = torch.stack(
                [
                    torch.cat(
                        [
                            torch.from_numpy(s.features),
                            # Same values as the positional encoding -- see collate().
                            torch.from_numpy(one_hot_frames(s.frames, p.framespan)),
                        ],
                        dim=-1,
                    )
                    for s in batch
                ]
            ).to(device)
            src_frames = torch.stack(
                [torch.from_numpy(s.frames) for s in batch]
            ).to(device)
            decoded = model.generate(src, src_frames, max_events=p.max_events)
            for sample, tokens in zip(batch, decoded):
                raw[sample.key].extend(tokens_to_events(tokens, sample.start_frame))
            if start % (p.batch_size * 25) == 0:
                print(f"  window {start}/{len(ds)}", flush=True)

        gt: dict[str, list[PCBASEvent]] = {}
        pred_slot: dict[str, list[PCBASEvent]] = {}
        pred_shirt: dict[str, list[PCBASEvent]] = {}
        per_half = []
        for key in ds.keys:
            rows = load_half(p.h5_path, key)
            deduped = dedupe(raw[key], p.dedup_window)
            n_frames = int(rows[:, 0].max()) + 1
            gt[key] = half_to_events(rows, key).events
            pred_slot[key] = deduped
            pred_shirt[key] = assign_shirts(deduped, slot_shirt_table(rows, n_frames))
            per_half.append(
                {
                    "half": key,
                    "gt_events": len(gt[key]),
                    "raw_events": len(raw[key]),
                    "after_dedup": len(deduped),
                }
            )
            print(f"  {key}: {len(raw[key])} raw -> {len(deduped)} deduped "
                  f"vs {len(gt[key])} GT", flush=True)

        kw = dict(delta=p.delta, conf_thresh=p.conf_thresh)
        by_slot = score_halves(gt, pred_slot, identity="slot", **kw)
        by_shirt = score_halves(gt, pred_shirt, identity="shirt", **kw)

        if p.export_json:
            payload = {"keys": list(pred_shirt), "events": {}}
            for key, events in pred_shirt.items():
                payload["events"][key] = [
                    [e.frame_idx, e.left_to_right, e.shirt_number, e.class_id, e.score]
                    for e in events
                ]
            Path(p.export_json).parent.mkdir(parents=True, exist_ok=True)
            Path(p.export_json).write_text(json.dumps(payload, indent=2))

        result = {
            "per_half": per_half,
            "by_slot": by_slot.model_dump(mode="json"),
            "by_shirt": by_shirt.model_dump(mode="json"),
            "reference_taad_dst_micro_f1": 0.7186,
            "our_stage1_micro_f1": 0.3274,
        }
        self.write_result(workdir, result)
        for name, report in (("slot", by_slot), ("shirt", by_shirt)):
            print(
                f"[{name}] micro-F1 {report.micro_f1:.4f}  macro-F1 {report.macro_f1:.4f}  "
                f"TP {report.tp} FP {report.fp} FN {report.fn} / GT {report.n_gt}"
            )
        return result
