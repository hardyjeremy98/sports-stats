"""Detector fine-tuning on RF-DETR (Apache-2.0).

Chosen over ultralytics YOLO because YOLO code+weights are AGPL (see
technology/10-libraries.md): fine-tuning the roboflow/sports datasets on
RF-DETR keeps the shipped detector permissive. Datasets download in COCO
format, which RF-DETR consumes natively.
"""

from __future__ import annotations

from pydantic import BaseModel

from pitchlab_train.datasets import resolve_dataset
from pitchlab_train.experiments.base import Experiment
from pitchlab_train.registry import register


class Params(BaseModel):
    model_size: str = "base"  # base | large
    epochs: int = 30
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 1e-4
    resolution: int = 1280  # roboflow/sports player model was trained at 1280


@register("detector-rfdetr")
class RfDetrDetectorExperiment(Experiment):
    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()
        dataset_dir = resolve_dataset(self.config.dataset, workdir.parent.parent)

        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError as exc:
            raise RuntimeError(
                "detector-rfdetr needs the rfdetr package "
                "(pip install 'pitchlab-train[detector]')"
            ) from exc

        model_cls = RFDETRLarge if p.model_size == "large" else RFDETRBase
        model = model_cls(resolution=p.resolution)
        model.train(
            dataset_dir=str(dataset_dir),
            epochs=p.epochs,
            batch_size=p.batch_size,
            grad_accum_steps=p.grad_accum_steps,
            lr=p.lr,
            output_dir=str(workdir / "checkpoints"),
        )

        result = {
            "dataset_dir": str(dataset_dir),
            "checkpoints_dir": str(workdir / "checkpoints"),
            "params": p.model_dump(),
        }
        self.write_result(workdir, result)
        return result
