"""Integration tests for the assembled `tdlp-shippable` track stage (SPO-42).

Proves the stage runs end-to-end on an ARBITRARY video (fake frames), emitting
standard Tracklet artifacts — the SPO-42 acceptance criterion. Correctness of
the wiring (frame walk -> feature assembly -> loop) is checked by swapping in a
deterministic proximity head after prepare(); the untrained-head path only needs
to run and produce valid artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from matchlab_core.registry import build
from matchlab_core.schemas import (
    Box,
    Detection,
    DetectionClass,
    FrameDetections,
    Tracklet,
    VideoMeta,
)
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.associate.embedders.base import BodyEmbedder, register_embedder


@register_embedder("fake-dino-test")
class _FakeDino(BodyEmbedder):
    name = "fake-dino-test"
    dim = 8

    def prepare(self, device: str) -> None:
        pass

    def embed(self, crops):
        out = np.zeros((len(crops), self.dim), np.float32)
        for i, c in enumerate(crops):
            out[i, (int(np.mean(c)) if getattr(c, "size", 0) else 0) % self.dim] = 1.0
        return out, None


class _FakeProximityModel:
    """MultiModalTDSP-shaped: high logit when a track's last bbox centre is near
    a detection centre (normalized coords)."""

    feature_names = {"bbox"}

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, obs_feat, obs_mask, unobs_feat, unobs_mask):
        bbox_obs, mask, det = obs_feat["bbox"][0], obs_mask[0], unobs_feat["bbox"][0]
        n, m = bbox_obs.shape[0], det.shape[0]
        logits = torch.full((n, m), -20.0)
        for i in range(n):
            valid = (~mask[i]).nonzero().flatten()
            if len(valid) == 0:
                continue
            tc = bbox_obs[i, valid[-1].item(), :2]
            for j in range(m):
                logits[i, j] = 10.0 - 60.0 * torch.norm(tc - det[j, :2])
        return logits.unsqueeze(0), {}


@dataclass
class _Frame:
    frame_idx: int
    t: float
    image: np.ndarray


class _Ctx:
    def __init__(self, n, w=400, h=200, fps=25.0, stride=1):
        self.video = VideoMeta(path="x.mp4", fps=fps, frame_count=n, width=w, height=h,
                               duration_s=n / fps)
        self.config = type("C", (), {"video": type("V", (), {
            "sample_stride": stride, "max_frames": None})()})()
        self.device = "cpu"
        self.pitch = None
        self.progress = lambda *a, **k: None
        self._n, self._w, self._h = n, w, h

    def frames(self):
        for i in range(self._n):
            yield _Frame(i, i / 25.0, np.full((self._h, self._w, 3), 50 + i, np.uint8))


def _det(x1, y1, x2, y2, conf=0.95):
    return Detection(box=Box(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf, cls=DetectionClass.PLAYER)


def _two_object_dets(n, w=400, h=200):
    out = []
    for f in range(n):
        x = 40 + f * 4
        out.append(FrameDetections(frame_idx=f, t=f / 25.0, detections=[
            _det(x, 60, x + 24, 150), _det(x + 200, 60, x + 224, 150)]))
    return out


def test_bbox_only_untrained_head_runs_end_to_end():
    """The core SPO-42 acceptance: runs on arbitrary video, emits valid
    Tracklets, even with a randomly-initialized head (plumbing proof)."""
    torch.manual_seed(0)
    stage = build(StageKind.TRACK, "tdlp-shippable",
                  {"use_appearance": False, "use_keypoints": False, "min_length": 1})
    ctx = _Ctx(5)
    stage.prepare(ctx)
    tracklets = stage.track(ctx, _two_object_dets(5))
    assert isinstance(tracklets, list)
    assert all(isinstance(t, Tracklet) for t in tracklets)
    for t in tracklets:
        idxs = [f.frame_idx for f in t.frames]
        assert idxs == sorted(idxs)  # frame-ordered, no dupes
        assert all(f.source == "observed" for f in t.frames)


def test_deterministic_head_yields_two_clean_tracklets():
    """Full wiring (frame walk -> assembly -> loop) with a deterministic head:
    two well-separated objects -> two stable tracklets, correct identity."""
    stage = build(StageKind.TRACK, "tdlp-shippable",
                  {"use_appearance": False, "use_keypoints": False, "min_length": 1,
                   "new_tracklet_detection_threshold": 0.9})
    ctx = _Ctx(5)
    stage.prepare(ctx)
    stage._model = _FakeProximityModel()  # deterministic association
    tracklets = stage.track(ctx, _two_object_dets(5))
    assert len(tracklets) == 2
    for t in tracklets:
        assert len(t.frames) == 5
    left = min(tracklets, key=lambda t: t.frames[0].box.x1)
    right = max(tracklets, key=lambda t: t.frames[0].box.x1)
    assert all(f.box.x1 < 200 for f in left.frames)
    assert all(f.box.x1 >= 200 for f in right.frames)


def test_multicue_appearance_path_runs_with_fake_embedder():
    """Appearance cue on (fake DINOv2 embedder), keypoints off: the multi-cue
    assembly + loop run end-to-end and emit valid artifacts."""
    torch.manual_seed(0)
    stage = build(StageKind.TRACK, "tdlp-shippable",
                  {"use_appearance": True, "embedder": "fake-dino-test",
                   "use_keypoints": False, "min_length": 1, "min_box_height_px": 10})
    ctx = _Ctx(4)
    stage.prepare(ctx)
    assert stage._modality.appearance_dim == 8  # picked up from the embedder
    tracklets = stage.track(ctx, _two_object_dets(4))
    assert all(isinstance(t, Tracklet) for t in tracklets)
