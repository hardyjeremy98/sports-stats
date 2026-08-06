"""Score `(9,26,T)` logits against FOOTPASS ground truth (SPO-96, Phase 1 gate).

Closes the loop: logits -> NMS-decoded slot events -> optional export-time roster
remap -> the PCBAS metric.

Both identity modes are always reported, because they answer different questions and
only one of them is comparable to the reference:

* `slot` -- did the model attribute the action to the right tactical role? This is
  what the model actually predicts and what Phase 3 must preserve.
* `shirt` -- the reference's on-disk exchange identity, reachable only through the
  ADR 008 per-frame roster remap. **This is the number comparable to the reference's
  micro-F1 0.4100 (TAAD) and 0.7186 (TAAD+DST).**

A gap between the two is informative rather than a bug: it is exactly the cost of the
roster remap, and it is measured here rather than assumed to be zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matchlab_core.pcbas.decode import DEFAULT_NMS_WINDOW, decode_logits
from matchlab_core.pcbas.eval import (
    DEFAULT_CONF_THRESH,
    DEFAULT_DELTA,
    score_halves,
)
from matchlab_core.pcbas.logits import load_logits, logits_filename
from pydantic import BaseModel

from matchlab_train.datasets.footpass_pcbas import (
    assign_shirts,
    half_to_events,
    list_halves,
    load_half,
    slot_shirt_table,
)
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register


class Params(BaseModel):
    h5_path: str = "data/footpass/tactical/val_tactical_data.h5"
    logits_dir: str = "data/footpass/logits/val"
    halves: list[str] | None = None
    nms_window: int = DEFAULT_NMS_WINDOW
    delta: int = DEFAULT_DELTA
    conf_thresh: float = DEFAULT_CONF_THRESH
    export_json: str | None = None  # write predictions in the reference's format


@register("pcbas-score")
class PCBASScoreExperiment(Experiment):
    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()

        gt: dict[str, list] = {}
        pred_slot: dict[str, list] = {}
        pred_shirt: dict[str, list] = {}
        per_half: list[dict] = []

        for key in p.halves or list_halves(p.h5_path):
            path = Path(p.logits_dir) / logits_filename(key)
            if not path.is_file():
                print(f"skipping {key}: no logits at {path}")
                continue
            logits = load_logits(path)
            rows = load_half(p.h5_path, key)

            events = decode_logits(logits, nms_window=p.nms_window)
            table = slot_shirt_table(rows, logits.shape[2])

            gt[key] = half_to_events(rows, key).events
            uncovered_gt = sum(1 for e in gt[key] if e.frame_idx >= logits.shape[2])
            if uncovered_gt:
                # Every one of these is counted as a false negative, which is right
                # for a full run and badly misleading for a truncated one.
                print(
                    f"WARNING {key}: {uncovered_gt} of {len(gt[key])} GT events lie "
                    f"beyond the logits' {logits.shape[2]} frames and will all count "
                    f"as false negatives"
                )
            pred_slot[key] = events
            pred_shirt[key] = assign_shirts(events, table)
            per_half.append(
                {
                    "half": key,
                    "gt_events": len(gt[key]),
                    "predicted_events": len(events),
                    "frames": int(logits.shape[2]),
                    "unknown_shirt": sum(
                        1 for e in pred_shirt[key] if e.shirt_number == -1
                    ),
                    "gt_beyond_logits": uncovered_gt,
                }
            )

        if not gt:
            raise RuntimeError(f"no logits found under {p.logits_dir}")

        kw = dict(delta=p.delta, conf_thresh=p.conf_thresh)
        by_slot = score_halves(gt, pred_slot, identity="slot", **kw)
        by_shirt = score_halves(gt, pred_shirt, identity="shirt", **kw)

        if p.export_json:
            self._export(pred_shirt, p.export_json)

        result = {
            "per_half": per_half,
            "by_slot": by_slot.model_dump(mode="json"),
            "by_shirt": by_shirt.model_dump(mode="json"),
            "reference_taad_micro_f1": 0.4100,
            "reference_taad_dst_micro_f1": 0.7186,
        }
        self.write_result(workdir, result)

        for name, report in (("slot", by_slot), ("shirt", by_shirt)):
            print(
                f"[{name}] micro-F1 {report.micro_f1:.4f}  macro-F1 {report.macro_f1:.4f}  "
                f"TP {report.tp} FP {report.fp} FN {report.fn} / GT {report.n_gt}"
            )
            for metrics in report.per_class.values():
                print(
                    f"    {metrics.class_name:<10} F1 {metrics.f1:.3f} "
                    f"P {metrics.precision:.3f} R {metrics.recall:.3f} "
                    f"(GT {metrics.n_gt})"
                )
        return result

    @staticmethod
    def _export(pred: dict[str, list], path: str) -> None:
        """Write the reference's playbyplay JSON so its own evaluation.py can read
        our predictions -- an independent check on our scorer, not just our model."""
        payload = {"keys": list(pred), "events": {}}
        for key, events in pred.items():
            payload["events"][key] = [
                [e.frame_idx, e.left_to_right, e.shirt_number, e.class_id, e.score]
                for e in events
            ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))


def logits_health(logits: np.ndarray) -> dict:
    """Cheap sanity summary of a logits array, for smoke runs."""
    return {
        "shape": list(logits.shape),
        "uncovered_frames": int((np.abs(logits).sum(axis=(0, 1)) == 0).sum()),
        "background_fraction": float((logits.argmax(axis=0) == 0).mean()),
        "max_abs": float(np.abs(logits).max()),
    }
