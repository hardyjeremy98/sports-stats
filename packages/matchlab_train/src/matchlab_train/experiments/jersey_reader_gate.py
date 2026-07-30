"""Gate 1: reproduce the reference jersey metric on the reference data.

Pre-registered bar: tracklet accuracy on the SoccerNet jersey TEST split
within 3 points of the published 87.45%. Below that, the fault is local wiring
and no downstream measurement is attributable, so gates 2-4 do not run.

Reported as an INTEGER COUNT TABLE (correct / wrong / abstained), not just an
accuracy: a single ratio cannot distinguish "reads wrongly" from "declines to
read", and those have opposite implications for a channel whose safety rests
on abstention.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from matchlab_core.crops import ScoredCrop
from matchlab_core.ocr.parseq import JerseyReader
from matchlab_core.reid.jersey import N_NUMBERS, tracklet_likelihood
from pydantic import BaseModel

from matchlab_train.datasets.soccernet_jersey import load_jersey_tracklets
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register

PUBLISHED_TRACKLET_ACCURACY = 0.8745
TOLERANCE = 0.03
OUT_DIR = Path("data/experiments/jersey-reader-gate")


class Params(BaseModel):
    root: str = "data/soccernet/jersey"
    split: str = "test"
    checkpoint: str = "data/weights/parseq-jersey.ckpt"
    device: str = "cuda"
    # 100 crops, evenly strided across the tracklet, was measured best: 12 ->
    # 0.3608 coverage, 40 -> 0.4467, 100 -> 0.8178 overall accuracy / 0.9031
    # legible precision / 0.8866 coverage, 250 -> 0.8133 (plateau). See the
    # task-4b jersey-ocr merge-channel report.
    max_crops_per_tracklet: int = 100
    min_confidence: float = 0.5  # tracklet likelihood below this abstains
    min_legibility: float = 0.9  # legibility score below this gates the crop
    max_tracklets: int | None = None


def _even_stride(paths: list, max_crops: int) -> list:
    """Up to `max_crops` items spanning the whole list, not just its prefix.

    A first-N sample is biased toward one part of the tracklet's lifespan;
    the reference reads the whole tracklet, and an even stride is what makes
    a `max_crops`-sized sample representative of it.
    """
    if len(paths) <= max_crops:
        return list(paths)
    stride = max(1, len(paths) // max_crops)
    return paths[::stride][:max_crops]


def _legible_stats(counts: dict) -> dict:
    """Precision/coverage over tracklets with a numeric GT (excludes
    `illegible_gt`): precision = correct / (correct + wrong); coverage =
    (correct + wrong) / (correct + wrong + abstained)."""
    decided = counts["correct"] + counts["wrong"]
    attempted = decided + counts["abstained"]
    return {
        "legible_precision": (counts["correct"] / decided) if decided else 0.0,
        "legible_coverage": (decided / attempted) if attempted else 0.0,
    }


@register("jersey-reader-gate")
class JerseyReaderGate(Experiment):
    def run(self) -> dict:
        params = Params(**self.config.params)
        data = load_jersey_tracklets(Path(params.root), params.split)
        items = sorted(data.items())[: params.max_tracklets]
        reader = JerseyReader(checkpoint=params.checkpoint, min_legibility=params.min_legibility)
        reader.prepare(device=params.device)

        counts = {"correct": 0, "wrong": 0, "abstained": 0, "illegible_gt": 0}
        rows = []
        total = len(items)
        for i, (tid, (paths, gt_number)) in enumerate(items):
            if i % 25 == 0:
                print(f"[jersey-reader-gate] {i}/{total} tracklets, counts so far: {counts}", flush=True)
            crops = []
            for p in _even_stride(paths, params.max_crops_per_tracklet):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                crops.append(
                    ScoredCrop(
                        image=img,
                        quality=1.0,
                        frame_idx=0,
                        box_height=float(img.shape[0]),
                        isolation_iou=0.0,
                    )
                )
            reads = reader.read(crops)
            likelihood = tracklet_likelihood(
                _logprobs(reads), np.array([r.legibility for r in reads])
            )
            best = int(np.argmax(likelihood))
            confidence = float(likelihood[best])
            predicted = best if confidence >= params.min_confidence else None

            if gt_number is None:
                counts["illegible_gt"] += 1
            elif predicted is None:
                counts["abstained"] += 1
            elif predicted == gt_number:
                counts["correct"] += 1
            else:
                counts["wrong"] += 1
            rows.append(
                {
                    "tracklet": tid,
                    "gt": gt_number,
                    "pred": predicted,
                    "confidence": confidence,
                    "n_reads": len(reads),
                }
            )

        decided = counts["correct"] + counts["wrong"] + counts["abstained"]
        accuracy = counts["correct"] / decided if decided else 0.0
        report = {
            "counts": counts,
            "tracklet_accuracy": accuracy,
            **_legible_stats(counts),
            "published": PUBLISHED_TRACKLET_ACCURACY,
            "tolerance": TOLERANCE,
            "passed": accuracy >= PUBLISHED_TRACKLET_ACCURACY - TOLERANCE,
            "train_adjacency": (
                "The checkpoint was fine-tuned on SoccerNet jersey data, so this "
                "figure is a reproduction check, NOT independent accuracy."
            ),
            "rows": rows,
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
        self.write_result(self.workdir(), report)
        return report


def _logprobs(reads) -> np.ndarray:
    """(n_reads, 100) per-crop number log-likelihoods."""
    from matchlab_core.reid.jersey import crop_number_logprobs

    if not reads:
        return np.zeros((0, N_NUMBERS))
    return np.array([crop_number_logprobs(r.char_probs) for r in reads])
