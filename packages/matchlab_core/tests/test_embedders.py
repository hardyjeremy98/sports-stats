"""Contract tests for the body re-ID embedder registry (`stages/associate/
embedders/`). Offline: no torch or network use in the default suite — the
real OSNet download + forward pass is proved once by the
`MATCHLAB_SLOW_TESTS`-gated smoke test below (also run manually, see
task-4-report.md)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from matchlab_core.stages.associate.embedders import (
    EMBEDDERS,
    BodyEmbedder,
    get_embedder,
    register_embedder,
)


def test_get_embedder_returns_osnet_instance():
    emb = get_embedder("osnet")
    assert emb.__class__.__name__ == "OsnetEmbedder"
    assert emb.name == "osnet"
    assert emb.dim == 512


def test_get_embedder_returns_solider_instance():
    emb = get_embedder("solider")
    assert emb.__class__.__name__ == "SoliderEmbedder"
    assert emb.name == "solider"
    assert emb.dim == 1024


def test_get_embedder_unknown_name_raises_keyerror_listing_known_names():
    with pytest.raises(KeyError) as exc_info:
        get_embedder("does-not-exist")
    message = str(exc_info.value)
    assert "does-not-exist" in message
    assert "osnet" in message
    assert "solider" in message


def test_register_embedder_registers_and_builds_with_params():
    class _FakeParamEmbedder(BodyEmbedder):
        dim = 4

        def __init__(self, scale: int = 1):
            self.scale = scale

        def prepare(self, device: str) -> None:
            pass

        def embed(self, crops):
            if not crops:
                return np.zeros((0, self.dim), dtype=np.float32), None
            return np.full((len(crops), self.dim), self.scale, dtype=np.float32), None

    decorated = register_embedder("fake-for-test")(_FakeParamEmbedder)
    assert decorated is _FakeParamEmbedder
    assert EMBEDDERS["fake-for-test"] is _FakeParamEmbedder
    try:
        built = get_embedder("fake-for-test", scale=3)
        assert isinstance(built, _FakeParamEmbedder)
        assert built.scale == 3
        assert built.name == "fake-for-test"
    finally:
        del EMBEDDERS["fake-for-test"]


def test_registering_duplicate_name_raises():
    with pytest.raises(ValueError):
        register_embedder("osnet")(object)


def test_embedders_package_import_does_not_eagerly_import_torch():
    """osnet.py may only import torch (and _osnet_arch, which itself imports
    torch at module level) lazily inside prepare() — never at package import
    time. torch IS installed in this dev env (it's a `cv` extra), so we can't
    assert "torch" not in sys.modules globally; instead check that osnet.py's
    own module namespace never bound a `torch` name, and that the vendored
    arch module — which nothing but prepare() ever imports — was never loaded.
    """
    from matchlab_core.stages.associate.embedders import osnet as osnet_module

    assert "torch" not in osnet_module.__dict__
    assert "huggingface_hub" not in osnet_module.__dict__
    assert "matchlab_core.stages.associate.embedders._osnet_arch" not in sys.modules


def test_osnet_embed_empty_crops_is_offline_safe():
    """The empty-crops short-circuit in OsnetEmbedder.embed() must run before
    any torch import or model access, so callers can probe an empty tracklet
    without ever calling prepare() (or having torch installed)."""
    emb = get_embedder("osnet")
    out, quality = emb.embed([])
    assert out.shape == (0, 512)
    assert out.dtype == np.float32
    assert quality is None


def test_embedders_package_import_does_not_eagerly_import_torch_for_solider():
    """Same laziness contract as OSNet: solider.py and its vendored arch module
    must only ever be imported/touched from prepare(), never at package import
    time."""
    from matchlab_core.stages.associate.embedders import solider as solider_module

    assert "torch" not in solider_module.__dict__
    assert "matchlab_core.stages.associate.embedders._solider_arch" not in sys.modules


def test_solider_embed_empty_crops_is_offline_safe():
    emb = get_embedder("solider")
    out, quality = emb.embed([])
    assert out.shape == (0, 1024)
    assert out.dtype == np.float32
    assert quality is None


def test_solider_prepare_missing_weights_raises_with_download_url():
    emb = get_embedder("solider", weights="/nonexistent/path/solider.pth")
    with pytest.raises(RuntimeError) as exc_info:
        emb.prepare("cpu")
    message = str(exc_info.value)
    assert "/nonexistent/path/solider.pth" in message
    assert "drive.google.com" in message
    assert "data/weights/reid/solider_swin_base_msmt17.pth" in message


class _FakeEmbedder(BodyEmbedder):
    name = "fake-contract"
    dim = 8

    def prepare(self, device: str) -> None:
        pass

    def embed(self, crops):
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32), None
        return np.ones((len(crops), self.dim), dtype=np.float32), None


def test_fake_embedder_empty_crops_contract():
    emb = _FakeEmbedder()
    out, quality = emb.embed([])
    assert out.shape == (0, 8)
    assert out.dtype == np.float32
    assert quality is None


def test_fake_embedder_nonempty_crops_shape():
    emb = _FakeEmbedder()
    out, quality = emb.embed([np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)])
    assert out.shape == (3, 8)
    assert quality is None


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("MATCHLAB_SLOW_TESTS"),
    reason="downloads real OSNet weights from the HF hub; set MATCHLAB_SLOW_TESTS=1 to run",
)
def test_osnet_real_prepare_and_embed_smoke():
    emb = get_embedder("osnet")
    emb.prepare("cpu")

    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, size=(180, 80, 3), dtype=np.uint8) for _ in range(3)]
    out, quality = emb.embed(crops)

    assert out.shape == (3, 512)
    assert out.dtype == np.float32
    assert quality is None
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("MATCHLAB_SLOW_TESTS"),
    reason="needs the downloaded SOLIDER-REID checkpoint; set MATCHLAB_SLOW_TESTS=1 to run",
)
def test_solider_real_prepare_and_embed_smoke():
    emb = get_embedder("solider")
    emb.prepare("cpu")

    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, size=(180, 80, 3), dtype=np.uint8) for _ in range(3)]
    out, quality = emb.embed(crops)

    assert out.shape == (3, 1024)
    assert out.dtype == np.float32
    assert quality is None
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("MATCHLAB_SLOW_TESTS"),
    reason="needs the downloaded SOLIDER-REID checkpoint; set MATCHLAB_SLOW_TESTS=1 to run",
)
def test_solider_embeddings_are_not_anisotropically_collapsed():
    """Regression test for the BN-neck bug: embedding the RAW Swin pooled
    feature (instead of the checkpoint's bottleneck/BN-neck output) produces
    anisotropic features whose pairwise cosine distances all collapse below
    ~0.13, silently defeating absolute-distance gating in the associator.
    With the BN-neck applied, visually distinct crops must separate well past
    that ceiling (measured 0.41-0.56 for these solid-colour crops; distinct
    real players from SNMOT-116 spread to median 0.387 / max 0.545)."""
    emb = get_embedder("solider")
    emb.prepare("cpu")

    red = np.zeros((180, 80, 3), np.uint8)
    red[:, :, 2] = 220
    blue = np.zeros((180, 80, 3), np.uint8)
    blue[:, :, 0] = 220
    green = np.zeros((180, 80, 3), np.uint8)
    green[:, :, 1] = 180
    out, _ = emb.embed([red, blue, green])

    sim = out @ out.T
    pair_dists = [1 - sim[0, 1], 1 - sim[0, 2], 1 - sim[1, 2]]
    assert min(pair_dists) > 0.2, (
        f"pairwise cosine distances {pair_dists} are collapsed — is embed() "
        "returning raw backbone features instead of the BN-neck output?"
    )
