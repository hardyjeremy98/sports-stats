"""Regression test for the `siglip-kmeans` team stage.

The stage converted BGR crops to RGB with a bare `[:, :, ::-1]`, which produces
a NEGATIVE-STRIDE view. `transformers`' image processor calls
`torch.from_numpy(...)` on whatever it is handed, and that rejects negative
strides outright, so the stage raised on the first crop of every run and had
never executed end to end despite being registered and referenced by two
configs.

The test drives the real `classify()` with the model and processor stubbed out,
and asserts on what the processor actually receives — the crops must survive
`torch.from_numpy`. Asserting `np.ascontiguousarray` was called somewhere would
just re-state the fix; this asserts the property the fix exists to guarantee.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from matchlab_core.schemas import DetectionClass, Team
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import Tracklet, TrackletFrame
from matchlab_core.stages.team import siglip as siglip_mod


class _StubProcessor:
    """Stands in for transformers' AutoProcessor, applying the one constraint
    that broke the real thing."""

    def __init__(self):
        self.seen: list[np.ndarray] = []

    def __call__(self, images, return_tensors=None):
        self.seen.extend(images)
        for img in images:
            # The exact call transformers makes; raises on negative strides.
            torch.from_numpy(np.asarray(img)).contiguous()
        return _StubBatch(len(images))


class _StubBatch(dict):
    def __init__(self, n: int):
        super().__init__(pixel_values=torch.zeros(n, 3, 8, 8))

    def to(self, device):
        return self


class _StubModel:
    def __call__(self, **kwargs):
        n = kwargs["pixel_values"].shape[0]
        # Two separable clusters so KMeans has something to find.
        half = n // 2
        vals = torch.cat([torch.zeros(half, 4, 6), torch.ones(n - half, 4, 6)])
        return type("Out", (), {"last_hidden_state": vals})()

    def to(self, device):
        return self


def _tracklet(tid: int) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(frame_idx=f, box=Box(x1=0, y1=0, x2=10, y2=20), confidence=0.9)
            for f in range(4)
        ],
    )


def test_crops_reach_the_processor_without_negative_strides(monkeypatch):
    tracklets = [_tracklet(i) for i in range(1, 7)]
    # BGR crops, exactly as sample_tracklet_crops returns them.
    crops = {
        t.tracklet_id: [np.random.randint(0, 255, (20, 10, 3), dtype=np.uint8) for _ in range(4)]
        for t in tracklets
    }
    monkeypatch.setattr(siglip_mod, "sample_tracklet_crops", lambda *a, **k: crops)

    stage = siglip_mod.SiglipTeamClassifier()
    processor = _StubProcessor()
    stage._processor = processor
    stage._model = _StubModel()

    ctx = type("Ctx", (), {"device": "cpu"})()
    # Before the fix this raised ValueError on the first batch.
    out = stage.classify(ctx, tracklets)

    assert len(processor.seen) == 24, "every crop should reach the processor"
    assert all(np.asarray(img).strides[-1] > 0 for img in processor.seen)
    assert {a.tracklet_id for a in out} == {t.tracklet_id for t in tracklets}
    assert all(a.team in (Team.HOME, Team.AWAY, Team.UNKNOWN) for a in out)


def test_channels_are_actually_reversed(monkeypatch):
    """Contiguity must not be bought by dropping the BGR->RGB swap itself."""
    # Same crop count as the test above: UMAP needs more samples than a
    # handful, and this test is about what the processor receives, not about
    # exercising the clustering with a degenerate input.
    tracklets = [_tracklet(i) for i in range(1, 7)]
    bgr = np.zeros((20, 10, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # pure blue in BGR
    monkeypatch.setattr(
        siglip_mod,
        "sample_tracklet_crops",
        lambda *a, **k: {t.tracklet_id: [bgr] * 4 for t in tracklets},
    )

    stage = siglip_mod.SiglipTeamClassifier()
    processor = _StubProcessor()
    stage._processor = processor
    stage._model = _StubModel()
    stage.classify(type("Ctx", (), {"device": "cpu"})(), tracklets)

    first = np.asarray(processor.seen[0])
    assert first[0, 0, 2] == 255 and first[0, 0, 0] == 0, "BGR->RGB swap was lost"


def test_too_few_crops_abstains(monkeypatch):
    tracklets = [_tracklet(1)]
    monkeypatch.setattr(siglip_mod, "sample_tracklet_crops", lambda *a, **k: {1: []})
    stage = siglip_mod.SiglipTeamClassifier()
    out = stage.classify(type("Ctx", (), {"device": "cpu"})(), tracklets)
    assert [a.team for a in out] == [Team.UNKNOWN]
    assert [a.confidence for a in out] == [0.0]


@pytest.mark.parametrize("arr", [np.zeros((4, 4, 3), dtype=np.uint8)[:, :, ::-1]])
def test_bare_reversed_view_would_have_failed(arr):
    """Pins the underlying constraint, so this test file explains itself if the
    transformers behaviour ever changes."""
    with pytest.raises(ValueError, match="negative"):
        torch.from_numpy(arr)
