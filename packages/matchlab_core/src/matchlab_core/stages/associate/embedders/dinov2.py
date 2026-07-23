"""DINOv2 (ViT-S/14) global body-appearance embedder — the *shippable*
appearance cue (SPO-38), replacing the research-only OSNet/KPR embedders.

Unlike OSNet (part-based re-ID feature), this produces a single GLOBAL vector
per crop — the ViT's pooled/CLS output (`num_classes=0`), `dim=384`. That is
intentional: DINOv2 is a general self-supervised visual backbone, not a re-ID
head, so the appearance signal is one holistic descriptor, not per-part limbs.

Loaded via `timm` (`vit_small_patch14_dinov2.lvd142m`); weights download lazily
on first `prepare()` from the HF hub (`timm/vit_small_patch14_dinov2.lvd142m`).
The small variant fits the 8GB dev GPU. Input is 518x518 (DINOv2's patch-14
native size), ImageNet mean/std normalization, taken from timm's own data
config so we track the checkpoint's expected preprocessing.

Licensing — recorded per axis (feeds the SPO-41 certification gate; be honest
about residual risk, do not overstate):
  * code:          Apache-2.0 — DINOv2 (facebookresearch/dinov2) and timm are
                   both Apache-2.0. (Note: the *earliest* DINOv2 release was
                   CC-BY-NC-4.0; it was relicensed to Apache-2.0 — this cue
                   depends on the Apache-2.0 line, not the NC one.)
  * weights:       Apache-2.0 — CONFIRMED for the exact checkpoint we load,
                   `timm/vit_small_patch14_dinov2.lvd142m`: its HF model card
                   and timm's embedded `pretrained_cfg['license']` both read
                   "apache-2.0" (verified 2026-07-19). Redistributable /
                   commercial-use OK.
  * training_data: LVD-142M — Meta's curated web-image dataset, self-supervised
                   (NO human labels). RESIDUAL RISK: LVD-142M is not publicly
                   released and the licenses of its underlying scraped web
                   images are unspecified/mixed. We ship only the trained
                   weights (Apache-2.0), never the data, so this does not gate
                   redistribution of MatchLab — but it is not a clean
                   permissive dataset and is flagged as such for certification.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from matchlab_core.provenance import LicenseAxes
from matchlab_core.stages.associate.embedders.base import BodyEmbedder, register_embedder

_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
_HF_HUB_ID = "timm/vit_small_patch14_dinov2.lvd142m"
_DIM = 384
# DINOv2 patch-14 native input; ImageNet mean/std. These mirror timm's data
# config for this checkpoint and are re-resolved from the model in prepare();
# kept here as constants so the pure `_preprocess` helper is testable offline.
_INPUT_SIZE = 518
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# Per-axis license facts for this exact checkpoint (see module docstring). A
# provenance()-style note: constants a future associator wire-up can record
# honestly, matching the LicenseAxes shape used in osnet/rfdetr provenance.
LICENSE = LicenseAxes(
    code="Apache-2.0 (DINOv2 facebookresearch/dinov2 + timm)",
    weights="Apache-2.0 (timm/vit_small_patch14_dinov2.lvd142m; HF card + timm cfg, verified)",
    training_data=(
        "LVD-142M (Meta curated web images, self-supervised/no labels; "
        "not publicly released, underlying image licenses unspecified)"
    ),
)
LINEAGE = "pretrained (LVD-142M, self-supervised DINOv2), no fine-tuning"


def _preprocess(
    crops: Sequence[np.ndarray],
    size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Pure, model-free mapping: BGR HxWx3 uint8 crops -> (N, 3, size, size)
    float32 batch, RGB-ordered, `[0,1]`-scaled, then `(x - mean) / std` per
    channel. Bicubic resize matches DINOv2's timm data config. No network, no
    torch — unit-testable with hand-made arrays."""
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    if not crops:
        return np.zeros((0, 3, size, size), dtype=np.float32)

    tensors: list[np.ndarray] = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
        normed = (resized.astype(np.float32) / 255.0 - mean_arr) / std_arr
        tensors.append(normed.transpose(2, 0, 1))
    return np.stack(tensors).astype(np.float32)


@register_embedder("dinov2")
class DinoV2Embedder(BodyEmbedder):
    dim = _DIM
    # Per-axis license for the assembled-stack certification gate (SPO-41).
    license = LICENSE
    lineage = LINEAGE

    def __init__(self, batch_size: int = 32, model_name: str = _MODEL_NAME):
        self.batch_size = batch_size
        self.model_name = model_name
        self._model = None
        self._device = "cpu"
        # Resolved from timm's data config in prepare(); default to the module
        # constants so shape/preprocessing is known before download.
        self._input_size = _INPUT_SIZE
        self._mean: tuple[float, ...] = _MEAN
        self._std: tuple[float, ...] = _STD

    def prepare(self, device: str) -> None:
        try:
            import timm
            import torch  # noqa: F401 — imported for its side effect + embed()
            from timm.data import resolve_data_config
        except ImportError as exc:
            raise RuntimeError(
                "The 'dinov2' body embedder needs torch + timm "
                "(pip install 'matchlab-core[cv]')."
            ) from exc

        # pretrained=True downloads the Apache-2.0 checkpoint from the HF hub
        # (timm/vit_small_patch14_dinov2.lvd142m) on first use, then caches it.
        # num_classes=0 drops the classifier head -> forward() returns the
        # pooled global feature vector (dim=384).
        model = timm.create_model(self.model_name, pretrained=True, num_classes=0)

        if model.num_features != self.dim:
            # Explicit raise, not assert (asserts vanish under `python -O`): a
            # wrong variant would silently produce mis-dimensioned embeddings.
            raise RuntimeError(
                f"DINOv2 model '{self.model_name}' has feature dim "
                f"{model.num_features}, expected {self.dim} — wrong variant?"
            )

        # Track the checkpoint's own expected preprocessing rather than assuming.
        cfg = resolve_data_config({}, model=model)
        self._input_size = int(cfg["input_size"][-1])
        self._mean = tuple(float(x) for x in cfg["mean"])
        self._std = tuple(float(x) for x in cfg["std"])

        self._model = model.eval().to(device)
        self._device = device

    def embed(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32), None

        import torch

        batches: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(crops), self.batch_size):
                batch = crops[i : i + self.batch_size]
                arr = _preprocess(batch, self._input_size, self._mean, self._std)
                x = torch.from_numpy(arr).to(self._device)
                feats = self._model(x).cpu().numpy()
                batches.append(feats)

        emb = np.concatenate(batches).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        return emb, None
