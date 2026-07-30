"""Legibility gate for jersey-number crops (CC BY-NC 3.0 fine-tuned weights).

Measured (450-tracklet band+vote run, see the task-4b jersey-ocr merge-channel
report): reading every crop produces 73 correct / 377 wrong -- confident,
plausible-looking, WRONG numbers dominate. Gating crops on this classifier's
score at >=0.9 before reading them drops that to 235 correct / 29 wrong, a
13x fall in wrong reads. The gate is on legibility, not on pose reliability
(`ocr/parseq.py`'s RTMPose torso localisation was measured actively harmful
here and removed) -- a crop can have a perfectly locatable torso and still
carry no readable digits (motion blur, occlusion, viewing angle), and that is
exactly the case this classifier exists to catch.

Same lazy-import/loud-failure convention as `ocr/parseq.py` and
`pose/rtmpose.py`: torch and torchvision are not declared dependencies --
supply them per invocation with `uv run --with torch --with torchvision ...`.
"""

from __future__ import annotations

import numpy as np

from matchlab_core.provenance import LicenseAxes, ModelProvenance

_DEFAULT_WEIGHTS = "data/weights/legibility-resnet34-soccer.pth"
_PREFIX = "model_ft."
_INPUT_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_BATCH_SIZE = 96


def _strip_prefix(state_dict: dict) -> dict:
    """Drop one leading `model_ft.` from every key that has it.

    The checkpoint was saved from a wrapper module that names the backbone
    `model_ft`; the bare `torchvision.models.resnet34()` this module loads
    into has no such wrapper, so the prefix must come off before
    `load_state_dict(strict=True)`.
    """
    return {
        (key[len(_PREFIX):] if key.startswith(_PREFIX) else key): value
        for key, value in state_dict.items()
    }


class LegibilityClassifier:
    """Crops in, legibility scores in [0, 1] out. Higher = more legible."""

    def __init__(self, weights: str = _DEFAULT_WEIGHTS):
        self.weights = weights
        self._model = None
        self._device = "cpu"

    def prepare(self, device: str = "cpu") -> None:
        try:
            import torch
            import torchvision
        except ImportError as exc:
            raise RuntimeError(
                "The legibility classifier needs torch and torchvision (not "
                "declared dependencies): supply them per invocation with "
                "`uv run --with torch --with torchvision ...`."
            ) from exc
        model = torchvision.models.resnet34()
        model.fc = torch.nn.Linear(512, 1)
        # Trusted because the checkpoint is a locally-provisioned research
        # artifact (data/weights/), never a network-fetched file.
        state = torch.load(self.weights, map_location="cpu", weights_only=False)
        state_dict = _strip_prefix(state)
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)
        self._model = model
        self._device = device

    def provenance(self) -> ModelProvenance:
        return ModelProvenance(
            architecture="resnet34",
            revision="legibility-finetune",
            weights_path=self.weights,
            lineage=(
                "ResNet34, ImageNet-pretrained, fine-tuned on SoccerNet jersey "
                "crops as a binary legibility classifier."
            ),
            license=LicenseAxes(
                code="unknown",
                weights="CC-BY-NonCommercial-3.0",
                training_data="SoccerNet (research)",
            ),
        )

    def score(self, crops: list[np.ndarray]) -> list[float]:
        """Legibility score per crop, in the same order as `crops`. Batched
        (~`_BATCH_SIZE` at a time) since a tracklet can carry up to 100."""
        if self._model is None:
            raise RuntimeError("LegibilityClassifier.score called before prepare()")
        if not crops:
            return []
        import cv2
        import torch

        scores: list[float] = []
        for start in range(0, len(crops), _BATCH_SIZE):
            batch = crops[start:start + _BATCH_SIZE]
            tensors = [self._preprocess(cv2, torch, img) for img in batch]
            x = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                logits = self._model(x).squeeze(-1)
                probs = torch.sigmoid(logits)
            scores.extend(float(p) for p in probs.cpu().numpy())
        return scores

    def _preprocess(self, cv2, torch, image: np.ndarray):
        rgb = cv2.cvtColor(
            cv2.resize(image, (_INPUT_SIZE, _INPUT_SIZE)), cv2.COLOR_BGR2RGB
        )
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        return x.sub_(mean).div_(std)
