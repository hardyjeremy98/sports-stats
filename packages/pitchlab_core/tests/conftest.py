"""Shared test fixtures for pitchlab_core.

`FakeEmbedder` is a deterministic body-re-ID embedder (no torch/network) used
by every test that needs a `global-reid` associator without a real model:
embedding = normalize([mean_B, mean_G, mean_R, 1.0]), so same-colour crops
match and different-colour crops don't, and the aggregation math stays
hand-computable on solid-colour synthetic frames. Registered once here (as
"fake-reid") so `test_associate_reid.py` and any pipeline-level test can both
reference it by name without a duplicate-registration error.
"""

from __future__ import annotations

import numpy as np
from pitchlab_core.stages.associate.embedders import BodyEmbedder, register_embedder


@register_embedder("fake-reid")
class FakeEmbedder(BodyEmbedder):
    """embedding = normalize([mean_B, mean_G, mean_R, 1.0]); optional
    model-native quality = mean_B / 255 (deterministic, hand-computable)."""

    dim = 4

    def __init__(self, use_model_quality: bool = False):
        self.use_model_quality = use_model_quality
        self.prepared_device: str | None = None

    def prepare(self, device: str) -> None:
        self.prepared_device = device

    def embed(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray | None]:
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32), None
        embs, quals = [], []
        for crop in crops:
            b, g, r = crop.reshape(-1, 3).mean(axis=0)
            v = np.array([b, g, r, 1.0], dtype=np.float32)
            embs.append(v / np.linalg.norm(v))
            quals.append(b / 255.0)
        quality = np.array(quals, dtype=np.float32) if self.use_model_quality else None
        return np.stack(embs).astype(np.float32), quality
