"""SPO-31 Task 4: the `botsort-reid` track stage — online OSNet-style embedding
of quality-gated per-frame crops, wired into the vendored appearance BoT-SORT.
Uses a deterministic fake embedder (Testing Decisions: fakes over real weights)."""

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("trackers")

from pitchlab_core.registry import build  # noqa: E402
from pitchlab_core.schemas import (  # noqa: E402
    Box,
    Detection,
    DetectionClass,
    FrameDetections,
    VideoMeta,
)
from pitchlab_core.schemas.run import StageKind  # noqa: E402
from pitchlab_core.stages.associate.embedders.base import (  # noqa: E402
    BodyEmbedder,
    get_embedder,
    register_embedder,
)


@register_embedder("fake-reid-test")
class _FakeEmbedder(BodyEmbedder):
    name = "fake-reid-test"
    dim = 4

    def prepare(self, device: str) -> None:
        pass

    def embed(self, crops):
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            v = int(np.mean(c)) if getattr(c, "size", 0) else 0
            out[i, v % self.dim] = 1.0
        return out, None


def _det(x1, y1, x2, y2, conf=0.9, cls=DetectionClass.PLAYER):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf, cls=cls)


# --- embed_detections helper: crop + quality gate + embed ---


def test_embed_detections_gates_short_and_low_conf():
    from pitchlab_core.stages.track.botsort_reid import embed_detections

    model = get_embedder("fake-reid-test")
    model.prepare("cpu")
    img = np.full((200, 200, 3), 100, np.uint8)
    dets = [
        _det(0, 0, 30, 90, conf=0.9),    # tall + high conf -> embedded
        _det(0, 0, 30, 10, conf=0.9),    # too short -> gated out
        _det(50, 50, 80, 140, conf=0.1),  # low conf -> gated out
    ]
    emb, ok = embed_detections(img, dets, model, min_box_height_px=60, min_crop_confidence=0.3)
    assert ok.tolist() == [True, False, False]
    assert emb.shape == (3, 4)
    assert np.any(emb[0] != 0.0)   # embedded
    assert np.all(emb[1] == 0.0)   # gated -> zero row
    assert np.all(emb[2] == 0.0)


def test_embed_detections_none_image_returns_none():
    from pitchlab_core.stages.track.botsort_reid import embed_detections

    model = get_embedder("fake-reid-test")
    emb, ok = embed_detections(None, [_det(0, 0, 30, 90)], model, 60, 0.3)
    assert emb is None and ok is None


# --- stage wiring smoke test ---


@dataclass
class _Frame:
    frame_idx: int
    t: float
    image: np.ndarray


class _CtxWithFrames:
    def __init__(self, n, fps=25.0, stride=1):
        self.video = VideoMeta(path="x.mp4", fps=fps, frame_count=n, width=400, height=200,
                               duration_s=n / fps)
        self.config = type("C", (), {"video": type("V", (), {
            "sample_stride": stride, "max_frames": None})()})()
        self.device = "cpu"
        self._n = n

    def frames(self):
        for i in range(self._n):
            img = np.full((200, 400, 3), 50 + i, np.uint8)
            yield _Frame(i, i / 25.0, img)


def test_botsort_reid_stage_builds_and_tracks():
    stage = build(StageKind.TRACK, "botsort-reid",
                  {"embedder": "fake-reid-test", "appearance_weight": 0.3, "enable_cmc": False})
    ctx = _CtxWithFrames(5)
    stage.prepare(ctx)
    # two well-separated objects moving right -> two stable tracklets.
    dets = []
    for f in range(5):
        x = 40 + f * 4
        dets.append(FrameDetections(frame_idx=f, t=f / 25.0, detections=[
            _det(x, 60, x + 24, 150), _det(x + 200, 60, x + 224, 150)]))
    tracklets = stage.track(ctx, dets)
    assert len(tracklets) == 2
    assert all(len(t.frames) >= 1 for t in tracklets)
