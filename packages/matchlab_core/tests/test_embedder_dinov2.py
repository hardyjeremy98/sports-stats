"""Contract tests for the permissive DINOv2 body-appearance embedder
(`stages/associate/embedders/dinov2.py`, SPO-38). Offline by default: no torch
or network use in the standard suite — the real-weights download + forward pass
is proved once by the `MATCHLAB_SLOW_TESTS`-gated smoke test below, which also
skips gracefully if the HF download is unreachable.

DINOv2 is the *shippable* appearance cue (Apache-2.0 code + weights),
contrasted with research-only OSNet/KPR."""

from __future__ import annotations

import os

import numpy as np
import pytest
from matchlab_core.stages.associate.embedders import get_embedder


def test_get_embedder_returns_dinov2_instance():
    emb = get_embedder("dinov2")
    assert emb.__class__.__name__ == "DinoV2Embedder"
    assert emb.name == "dinov2"
    assert emb.dim == 384


def test_dinov2_registered_alongside_osnet():
    from matchlab_core.stages.associate.embedders import EMBEDDERS

    assert "dinov2" in EMBEDDERS
    assert "osnet" in EMBEDDERS  # registering dinov2 must not clobber osnet


def test_dinov2_embed_empty_crops_is_offline_safe():
    """The empty-crops short-circuit must run before any torch/timm import or
    model access, so callers can probe an empty tracklet without prepare()."""
    emb = get_embedder("dinov2")
    out, quality = emb.embed([])
    assert out.shape == (0, 384)
    assert out.dtype == np.float32
    assert quality is None


def test_dinov2_module_import_does_not_eagerly_import_torch_or_timm():
    """dinov2.py may only import torch/timm lazily inside prepare()/embed() —
    never at package import time. Both ARE installed in the dev env (`cv`
    extra), so we can't assert absence from sys.modules globally; instead check
    the module's own namespace never bound those names."""
    from matchlab_core.stages.associate.embedders import dinov2 as dinov2_module

    assert "torch" not in dinov2_module.__dict__
    assert "timm" not in dinov2_module.__dict__


# --- Pure preprocessing helper (no model, no network) ------------------------


def test_preprocess_empty_returns_zero_batch():
    from matchlab_core.stages.associate.embedders.dinov2 import _preprocess

    out = _preprocess([], size=4, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    assert out.shape == (0, 3, 4, 4)
    assert out.dtype == np.float32


def test_preprocess_swaps_bgr_to_rgb_and_normalizes():
    """A 1x1 BGR pixel [B=10, G=20, R=30] must become CHW RGB [30, 20, 10]/255
    with identity mean/std, proving the BGR->RGB swap, /255 scale, and
    HWC->CHW transpose all happen in the pure helper."""
    from matchlab_core.stages.associate.embedders.dinov2 import _preprocess

    crop = np.array([[[10, 20, 30]]], dtype=np.uint8)  # (1,1,3) BGR
    out = _preprocess([crop], size=1, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

    assert out.shape == (1, 3, 1, 1)
    assert out.dtype == np.float32
    expected = np.array([30, 20, 10], dtype=np.float32) / 255.0  # R,G,B order
    np.testing.assert_allclose(out[0, :, 0, 0], expected, atol=1e-6)


def test_preprocess_applies_mean_std():
    """(x/255 - mean) / std applied per channel after the RGB swap."""
    from matchlab_core.stages.associate.embedders.dinov2 import _preprocess

    crop = np.array([[[10, 20, 30]]], dtype=np.uint8)  # BGR -> RGB [30,20,10]
    mean = (0.1, 0.2, 0.3)
    std = (0.5, 0.6, 0.7)
    out = _preprocess([crop], size=1, mean=mean, std=std)

    rgb = np.array([30, 20, 10], dtype=np.float32) / 255.0
    expected = (rgb - np.array(mean, np.float32)) / np.array(std, np.float32)
    np.testing.assert_allclose(out[0, :, 0, 0], expected, atol=1e-6)


def test_preprocess_resizes_to_square():
    from matchlab_core.stages.associate.embedders.dinov2 import _preprocess

    crop = np.zeros((37, 11, 3), dtype=np.uint8)
    out = _preprocess([crop], size=14, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    assert out.shape == (1, 3, 14, 14)


# --- Real-weights smoke test (gated + graceful skip) -------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("MATCHLAB_SLOW_TESTS"),
    reason="downloads real DINOv2 weights from the HF hub; set MATCHLAB_SLOW_TESTS=1 to run",
)
def test_dinov2_real_prepare_and_embed_smoke():
    emb = get_embedder("dinov2")
    try:
        emb.prepare("cpu")
    except Exception as exc:  # noqa: BLE001 — network/download failures must skip, not fail
        msg = str(exc).lower()
        if any(t in msg for t in ("connection", "download", "http", "offline", "resolve", "url")):
            pytest.skip(f"DINOv2 weights unreachable (offline?): {exc}")
        raise

    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, size=(180, 80, 3), dtype=np.uint8) for _ in range(2)]
    out, quality = emb.embed(crops)

    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    assert quality is None
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
