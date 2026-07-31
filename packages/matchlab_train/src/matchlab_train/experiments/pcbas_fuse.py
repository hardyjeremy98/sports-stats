"""Fuse several DST models' exported event lists and score the ensemble (PAVE §4).

Consumes the `export_json` payloads `pcbas-denoise-infer` already writes, so it needs
no model, no GPU and no logits -- the four ensemble members can be trained and
inferred on different days and fused afterwards.

Those exports are SHIRT-keyed (`[frame, left_to_right, shirt, class_id, score]`), so
fusion runs at `identity="shirt"`, which is also the grouping PAVE describes. That
means `by_shirt` is the ensemble's headline and there is no `by_slot` figure here: the
role slot is not recoverable from the export, and re-deriving it would need the
half's slot->shirt table inverted, which is not injective when a slot is unoccupied.
"""

from __future__ import annotations

import json
from pathlib import Path

from matchlab_core.pcbas.eval import DEFAULT_CONF_THRESH, DEFAULT_DELTA, score_halves
from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.fusion import (
    DEFAULT_FUSION_DELTA,
    DEFAULT_MIN_MODELS,
    DEFAULT_SOLO_CLASSES,
    fuse_model_events,
)
from pydantic import BaseModel

from matchlab_train.datasets.footpass_pcbas import half_to_events, load_half
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register


class Params(BaseModel):
    h5_path: str = "data/footpass/tactical/val_tactical_data.h5"
    # One `pcbas-denoise-infer --export_json` payload per ensemble member.
    exports: list[str] = []
    delta: int = DEFAULT_DELTA
    conf_thresh: float = DEFAULT_CONF_THRESH
    fusion_delta: int = DEFAULT_FUSION_DELTA
    min_models: int = DEFAULT_MIN_MODELS
    solo_classes: list[str] = list(DEFAULT_SOLO_CLASSES)
    export_json: str | None = None


def load_export(path: str) -> dict[str, list[PCBASEvent]]:
    """One `pcbas-denoise-infer` export -> per-half shirt-keyed events."""
    payload = json.loads(Path(path).read_text())
    out: dict[str, list[PCBASEvent]] = {}
    for key, rows in payload["events"].items():
        out[key] = [
            PCBASEvent(
                frame_idx=int(frame),
                left_to_right=int(ltr),
                shirt_number=None if shirt is None else int(shirt),
                class_id=int(class_id),
                score=float(score),
            )
            for frame, ltr, shirt, class_id, score in rows
        ]
    return out


@register("pcbas-fuse")
class PCBASFuseExperiment(Experiment):
    """N model exports -> Weighted Event Fusion -> the ensemble's VAL score."""

    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()
        if len(p.exports) < 2:
            raise ValueError(
                f"fusion needs at least 2 model exports, got {len(p.exports)}. "
                "A one-model 'ensemble' would only apply the (n/N)^0.5 penalty."
            )

        per_model = [load_export(path) for path in p.exports]
        for path, events in zip(p.exports, per_model):
            print(f"loaded {path}: {len(events)} halves, "
                  f"{sum(len(v) for v in events.values()):,} events")

        keys = sorted(set().union(*(set(m) for m in per_model)))
        gt: dict[str, list[PCBASEvent]] = {}
        fused: dict[str, list[PCBASEvent]] = {}
        per_half = []
        for key in keys:
            gt[key] = half_to_events(load_half(p.h5_path, key), key).events
            lists = [model.get(key, []) for model in per_model]
            fused[key] = fuse_model_events(
                lists,
                delta=p.fusion_delta,
                identity="shirt",
                min_models=p.min_models,
                solo_classes=tuple(p.solo_classes),
            )
            per_half.append(
                {
                    "half": key,
                    "gt_events": len(gt[key]),
                    "per_model_events": [len(x) for x in lists],
                    "fused_events": len(fused[key]),
                }
            )
            print(f"  {key}: {[len(x) for x in lists]} -> {len(fused[key])} fused "
                  f"vs {len(gt[key])} GT", flush=True)

        report = score_halves(
            gt, fused, identity="shirt", delta=p.delta, conf_thresh=p.conf_thresh
        )

        if p.export_json:
            payload = {"keys": keys, "events": {}}
            for key, events in fused.items():
                payload["events"][key] = [
                    [e.frame_idx, e.left_to_right, e.shirt_number, e.class_id, e.score]
                    for e in events
                ]
            Path(p.export_json).parent.mkdir(parents=True, exist_ok=True)
            Path(p.export_json).write_text(json.dumps(payload, indent=2))

        result = {
            "n_models": len(p.exports),
            "exports": p.exports,
            "min_models": p.min_models,
            "solo_classes": p.solo_classes,
            "per_half": per_half,
            "by_shirt": report.model_dump(mode="json"),
            "reference_taad_dst_micro_f1": 0.7186,
            "pave_ensemble_val_macro_f1": 0.609,
        }
        self.write_result(workdir, result)
        print(
            f"[ensemble of {len(p.exports)}] micro-F1 {report.micro_f1:.4f}  "
            f"macro-F1 {report.macro_f1:.4f}  TP {report.tp} FP {report.fp} "
            f"FN {report.fn} / GT {report.n_gt}"
        )
        return result
