"""Shared test fixtures for matchlab_core.

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
import pytest
from matchlab_core.stages.associate.embedders import BodyEmbedder, register_embedder


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


@pytest.fixture(autouse=True)
def _contract_free_default_fusion_model(request, monkeypatch, tmp_path):
    """Point the engine's DEFAULT fusion model at a contract-free copy of v1.

    The shipped artefact now carries a feature contract (embedding_dim=256,
    occupancy_coords) and `validate_serving` hard-fails on mismatch -- which is
    the production guarantee, but these unit tests exercise merge MECHANICS on
    2-4-dim toy embeddings that no fitted artefact could ever match. Stripping
    the contract keeps them tests of the engine rather than of the fixture.
    The contract/validation behaviour itself has dedicated tests in
    test_reid_twopass.py, which build models directly and are unaffected.
    """
    import json as _json
    from pathlib import Path as _Path

    from matchlab_core.reid.twopass import FusionModel as _FM

    def _load_nocontract(path):
        d = _json.loads(_Path(path).read_text())
        d.pop("contract", None)
        return _FM.from_dict(d)

    monkeypatch.setattr(_FM, "load", _load_nocontract)
    yield
