"""Tests for IdentityEvidence crop geometry + raw-crop persistence.

Covers the `box` / `raw_crop_artifact` fields added to IdentityEvidence and the
face identity resolver's crop-rect computation and raw-crop saving. face.py
imports insightface/realesrgan lazily inside prepare()/_load_realesrgan(), so
these tests construct the resolver and stub its `_app` / `_upscaler`
attributes directly rather than calling prepare() — no heavy CV deps needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pitchlab_core.artifacts import ArtifactStore
from pitchlab_core.schemas import Box, IdentityEvidence, PlayerEntity, PlayerIdentity
from pitchlab_core.schemas.detections import DetectionClass
from pitchlab_core.schemas.tracks import Tracklet, TrackletFrame
from pitchlab_core.stages.identity.face import FaceIdentityResolver
from pitchlab_core.video import Frame


def test_identity_evidence_roundtrip_with_new_fields():
    ev = IdentityEvidence(
        tracklet_id=1,
        frame_idx=10,
        score=0.9,
        crop_artifact="crops/face_p1_f10.jpg",
        upscaled=True,
        box=Box(x1=1, y1=2, x2=3, y2=4),
        raw_crop_artifact="crops/face_p1_f10_raw.jpg",
    )
    restored = IdentityEvidence.model_validate(ev.model_dump())
    assert restored == ev


def test_identity_evidence_roundtrip_without_new_fields():
    """Old-style players.json (written before this task) must still validate."""
    old_style = {
        "tracklet_id": 1,
        "frame_idx": 10,
        "score": 0.9,
        "crop_artifact": "crops/face_p1_f10.jpg",
        "upscaled": False,
    }
    ev = IdentityEvidence.model_validate(old_style)
    assert ev.box is None
    assert ev.raw_crop_artifact is None


def test_head_crop_rect_matches_crop_and_stays_in_frame():
    resolver = FaceIdentityResolver()
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    box = Box(x1=100, y1=50, x2=140, y2=150)

    crop, rect, upscaled, raw_crop = resolver._head_crop(image, box)

    assert crop is not None
    assert rect is not None
    x1, y1, x2, y2 = rect
    assert 0 <= x1 < x2 <= image.shape[1]
    assert 0 <= y1 < y2 <= image.shape[0]
    assert crop.shape[0] == y2 - y1
    assert crop.shape[1] == x2 - x1
    assert upscaled is False
    assert raw_crop is None


def test_head_crop_rect_clamped_to_frame_bounds_near_edge():
    resolver = FaceIdentityResolver()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = Box(x1=0, y1=0, x2=20, y2=40)  # near top-left corner, padding would go negative

    crop, rect, upscaled, raw_crop = resolver._head_crop(image, box)

    x1, y1, x2, y2 = rect
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100
    assert crop.shape[0] == y2 - y1
    assert crop.shape[1] == x2 - x1


class _FakeUpscaler:
    """Stand-in for the RealESRGANer instance _load_realesrgan() would build."""

    def enhance(self, crop, outscale):
        return crop.repeat(2, axis=0).repeat(2, axis=1), None


def test_head_crop_returns_raw_crop_when_upscaler_fires():
    resolver = FaceIdentityResolver(upscaler="realesrgan", upscale_below_px=1000)
    resolver._upscaler = _FakeUpscaler()
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    box = Box(x1=100, y1=50, x2=140, y2=150)

    crop, rect, upscaled, raw_crop = resolver._head_crop(image, box)

    assert upscaled is True
    assert raw_crop is not None
    x1, y1, x2, y2 = rect
    assert raw_crop.shape[0] == y2 - y1
    assert raw_crop.shape[1] == x2 - x1
    assert crop.shape[0] == raw_crop.shape[0] * 2  # actually upscaled


def test_head_crop_no_raw_crop_when_upscaler_absent():
    resolver = FaceIdentityResolver()  # upscaler="none" by default -> self._upscaler stays None
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    box = Box(x1=100, y1=50, x2=140, y2=150)

    crop, rect, upscaled, raw_crop = resolver._head_crop(image, box)

    assert upscaled is False
    assert raw_crop is None


class _FakeFace:
    def __init__(self, det_score: float):
        self.det_score = det_score
        self.normed_embedding = np.array([1.0, 0.0], dtype=np.float32)


class _FakeApp:
    def get(self, crop):
        return [_FakeFace(0.9)]


@dataclass
class _FakeCtx:
    """Minimal stand-in for StageContext: resolve() only touches .store and
    .frames(), so we don't need a real VideoMeta/PipelineConfig/decoded video."""

    store: ArtifactStore
    _frame: Frame

    def frames(self):
        yield self._frame


def _make_players_and_tracklets():
    box = Box(x1=100, y1=50, x2=140, y2=150)
    tracklet = Tracklet(
        tracklet_id=1,
        cls=DetectionClass.PLAYER,
        frames=[TrackletFrame(frame_idx=0, box=box, confidence=0.9)],
    )
    entity = PlayerEntity(player_id=1, tracklet_ids=[1], identity=PlayerIdentity())
    return [entity], [tracklet]


def test_resolve_sets_box_and_raw_crop_artifact_when_upscaled(tmp_path):
    resolver = FaceIdentityResolver(upscaler="realesrgan", upscale_below_px=1000)
    resolver._app = _FakeApp()
    resolver._upscaler = _FakeUpscaler()

    store = ArtifactStore(tmp_path / "run")
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    ctx = _FakeCtx(store=store, _frame=Frame(frame_idx=0, t=0.0, image=image))
    players, tracklets = _make_players_and_tracklets()

    out = resolver.resolve(ctx, players, tracklets)

    evidence = out[0].identity.evidence
    assert len(evidence) == 1
    ev = evidence[0]
    assert ev.box is not None
    assert ev.box.x1 >= 0 and ev.box.y1 >= 0
    assert ev.upscaled is True
    assert ev.raw_crop_artifact == "crops/face_p1_f0_raw.jpg"
    assert (store.run_dir / ev.crop_artifact).exists()
    assert (store.run_dir / ev.raw_crop_artifact).exists()


def test_resolve_leaves_raw_crop_artifact_none_when_not_upscaled(tmp_path):
    resolver = FaceIdentityResolver()  # upscaler="none"
    resolver._app = _FakeApp()

    store = ArtifactStore(tmp_path / "run")
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    ctx = _FakeCtx(store=store, _frame=Frame(frame_idx=0, t=0.0, image=image))
    players, tracklets = _make_players_and_tracklets()

    out = resolver.resolve(ctx, players, tracklets)

    ev = out[0].identity.evidence[0]
    assert ev.upscaled is False
    assert ev.raw_crop_artifact is None
    assert (store.run_dir / ev.crop_artifact).exists()
    # no duplicate raw crop written
    assert not (store.run_dir / "crops" / "face_p1_f0_raw.jpg").exists()
