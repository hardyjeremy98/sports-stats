"""Hosted-detection response cache tests (SPO-10 part 2). No network
anywhere: a fake model object stands in for the roboflow `inference`
package's `get_model(...).infer(...)`, returning a canned
inference-package-shaped dict (`sv.Detections.from_inference` accepts a
plain dict directly, no `inference` SDK object needed)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from pitchlab_core.schemas.detections import DetectionClass
from pitchlab_core.schemas.run import VideoMeta
from pitchlab_core.stages.detect.hosted_cache import SCHEMA, HostedDetectionCache, cache_key
from pitchlab_core.stages.detect.roboflow import RoboflowDetector
from pitchlab_core.video import Frame

# --- shared fixtures --------------------------------------------------------


def _image(seed: int, shape=(16, 16, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=shape, dtype=np.uint8)


class _FakePlayerModel:
    """Stands in for `inference.get_model(...)`: `infer()` returns a canned
    dict matching what `sv.Detections.from_inference` accepts directly."""

    def __init__(self, boxes: list[tuple[float, float, float, float, float, int]]):
        self.calls = 0
        self._boxes = boxes

    def infer(self, image, confidence=0.3):
        self.calls += 1
        predictions = [
            {
                "x": x, "y": y, "width": w, "height": h,
                "confidence": conf, "class_id": cls_id, "class": "player",
            }
            for (x, y, w, h, conf, cls_id) in self._boxes
        ]
        result = {
            "predictions": predictions,
            "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        }
        return [result]


@dataclass
class _FakeVideoConfig:
    sample_stride: int = 1


@dataclass
class _FakeConfig:
    video: _FakeVideoConfig = field(default_factory=_FakeVideoConfig)


@dataclass
class _FakeCtx:
    video: VideoMeta
    config: _FakeConfig
    _frames: list

    def frames(self):
        return iter(self._frames)

    def progress(self, kind, frac, msg):
        pass


def _ctx(frames: list[Frame]) -> _FakeCtx:
    meta = VideoMeta(
        path="fake.mp4", fps=25.0, frame_count=len(frames), width=16, height=16,
        duration_s=len(frames) / 25.0,
    )
    return _FakeCtx(video=meta, config=_FakeConfig(), _frames=frames)


def _three_frames() -> list[Frame]:
    return [Frame(frame_idx=i, t=i / 25.0, image=_image(seed=i)) for i in range(3)]


# --- cache_key: pure, content-addressed, process-stable ---------------------


def test_cache_key_stable_for_identical_inputs():
    image = _image(seed=1)
    assert cache_key("model-a/1", 0.3, image) == cache_key("model-a/1", 0.3, image.copy())


def test_cache_key_changes_with_model_id():
    image = _image(seed=1)
    assert cache_key("model-a/1", 0.3, image) != cache_key("model-b/1", 0.3, image)


def test_cache_key_changes_with_confidence():
    image = _image(seed=1)
    assert cache_key("model-a/1", 0.3, image) != cache_key("model-a/1", 0.4, image)


def test_cache_key_changes_with_pixels():
    assert cache_key("model-a/1", 0.3, _image(seed=1)) != cache_key("model-a/1", 0.3, _image(seed=2))


def test_cache_key_ignores_frame_index_by_construction():
    # cache_key has no frame_idx parameter at all -- two "different frames"
    # with identical pixels necessarily produce the same key.
    image = _image(seed=7)
    assert cache_key("model-a/1", 0.3, image) == cache_key("model-a/1", 0.3, image)


# --- HostedDetectionCache: get/put roundtrip + schema ------------------------


def test_get_on_empty_cache_is_none(tmp_path):
    cache = HostedDetectionCache(tmp_path / "cache")
    assert cache.get("nonexistent-key") is None


def test_put_then_get_roundtrips_arrays(tmp_path):
    cache = HostedDetectionCache(tmp_path / "cache")
    payload = {
        "model_id": "model-a/1",
        "confidence": 0.3,
        "xyxy": [[1.0, 2.0, 3.0, 4.0]],
        "scores": [0.9],
        "class_id": [2],
    }
    cache.put("k1", payload)

    got = cache.get("k1")
    assert got is not None
    assert got.xyxy == [[1.0, 2.0, 3.0, 4.0]]
    assert got.scores == [0.9]
    assert got.class_id == [2]
    assert got.model_id == "model-a/1"
    assert got.confidence == 0.3


def test_cached_file_schema_header_fields_present(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = HostedDetectionCache(cache_dir)
    cache.put(
        "k1",
        {
            "model_id": "model-a/1", "confidence": 0.3,
            "xyxy": [], "scores": [], "class_id": [],
        },
    )

    raw = (cache_dir / "k1.json").read_text()
    import json

    data = json.loads(raw)
    assert data["schema"] == SCHEMA
    assert data["model_id"] == "model-a/1"
    assert data["confidence"] == 0.3
    assert "cached_at" in data and data["cached_at"]


# --- content_hash: changes on write, order-independent -----------------------


def test_content_hash_changes_when_entry_added(tmp_path):
    cache = HostedDetectionCache(tmp_path / "cache")
    before = cache.content_hash()
    cache.put("k1", {"model_id": "m", "confidence": 0.3, "xyxy": [], "scores": [], "class_id": []})
    after = cache.content_hash()
    assert before != after


def test_content_hash_order_independent(tmp_path):
    # Write raw files directly (fixed cached_at) rather than via put(), so
    # wall-clock timestamp jitter between calls can't contaminate this check
    # -- what's under test is content_hash()'s own (sorted-pairs) ordering,
    # not put()'s cached_at stamping.
    def _write(cache_dir: Path, key: str) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": SCHEMA, "model_id": "m", "confidence": 0.3,
            "cached_at": "2026-01-01T00:00:00+00:00",
            "xyxy": [[0, 0, 1, 1]], "scores": [0.5], "class_id": [2],
        }
        (cache_dir / f"{key}.json").write_text(json.dumps(record))

    dir_a = tmp_path / "cache_a"
    dir_b = tmp_path / "cache_b"
    _write(dir_a, "k1")
    _write(dir_a, "k2")
    _write(dir_b, "k2")
    _write(dir_b, "k1")

    assert HostedDetectionCache(dir_a).content_hash() == HostedDetectionCache(dir_b).content_hash()


def test_content_hash_empty_cache_is_deterministic(tmp_path):
    a = HostedDetectionCache(tmp_path / "does-not-exist-a")
    b = HostedDetectionCache(tmp_path / "does-not-exist-b")
    assert a.content_hash() == b.content_hash()


# --- RoboflowDetector wiring: readwrite mode ---------------------------------


def test_readwrite_second_run_zero_network_calls_identical_detections(tmp_path):
    cache_dir = tmp_path / "cache"
    frames = _three_frames()

    model_1 = _FakePlayerModel(boxes=[(50.0, 50.0, 20.0, 40.0, 0.9, 2)])
    stage_1 = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="readwrite")
    stage_1._player_model = model_1
    out_1 = stage_1.detect(_ctx(frames))

    assert model_1.calls == 3  # one call per distinct frame, cache was cold
    assert len(list(cache_dir.glob("*.json"))) == 3

    model_2 = _FakePlayerModel(boxes=[(0.0, 0.0, 1.0, 1.0, 0.1, 0)])  # would differ if called
    stage_2 = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="readwrite")
    stage_2._player_model = model_2
    out_2 = stage_2.detect(_ctx(frames))

    assert model_2.calls == 0  # every frame was a cache hit
    assert [fd.detections for fd in out_1.frames] == [fd.detections for fd in out_2.frames]
    assert all(fd.detections for fd in out_1.frames)  # sanity: not vacuously equal


# --- RoboflowDetector wiring: replay mode ------------------------------------


def test_replay_prepare_requires_no_api_key_and_builds_no_model(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    cache_dir = tmp_path / "cache"
    stage = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="replay")

    stage.prepare(_ctx([]))  # must not raise despite no API key in env

    assert stage._player_model is None
    assert stage._ball_model is None


def test_replay_warm_cache_no_network_and_matches_readwrite_output(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    cache_dir = tmp_path / "cache"
    frames = _three_frames()

    warm_model = _FakePlayerModel(boxes=[(50.0, 50.0, 20.0, 40.0, 0.9, 2)])
    warm_stage = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="readwrite")
    warm_stage._player_model = warm_model
    warm_out = warm_stage.detect(_ctx(frames))

    replay_stage = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="replay")
    replay_stage.prepare(_ctx([]))  # no API key, no model constructed
    replay_out = replay_stage.detect(_ctx(frames))

    assert replay_stage._player_model is None  # never touched the network
    assert [fd.detections for fd in warm_out.frames] == [fd.detections for fd in replay_out.frames]


def test_replay_cold_cache_raises_naming_key_frame_and_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    cache_dir = tmp_path / "cache"
    frames = _three_frames()

    stage = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="replay")
    stage.prepare(_ctx([]))

    expected_key = cache_key(stage.params.player_model_id, stage.params.confidence, frames[0].image)

    with pytest.raises(RuntimeError) as exc_info:
        stage.detect(_ctx(frames))

    msg = str(exc_info.value)
    assert expected_key in msg
    assert "0" in msg  # frame_idx of the first (missed) frame
    assert str(cache_dir) in msg


# --- RoboflowDetector wiring: off mode is unchanged (no cache touched) ------


def test_off_mode_never_touches_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    frames = _three_frames()
    model = _FakePlayerModel(boxes=[(50.0, 50.0, 20.0, 40.0, 0.9, 2)])

    stage = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="off")
    stage._player_model = model
    stage.detect(_ctx(frames))

    assert model.calls == 3  # every frame hit the network, no caching
    assert not cache_dir.exists()


# --- Ball path: tiled detections also go through the cache -----------------


def test_ball_path_readwrite_second_run_zero_calls(tmp_path):
    cache_dir = tmp_path / "cache"
    frames = [Frame(frame_idx=0, t=0.0, image=_image(seed=99, shape=(128, 128, 3)))]

    player_model_1 = _FakePlayerModel(boxes=[])
    ball_model_1 = _FakePlayerModel(boxes=[(64.0, 64.0, 5.0, 5.0, 0.8, 0)])
    stage_1 = RoboflowDetector(
        cache_dir=str(cache_dir), cache_mode="readwrite", use_ball_model=True,
    )
    stage_1._player_model = player_model_1
    stage_1._ball_model = ball_model_1
    out_1 = stage_1.detect(_ctx(frames))
    assert ball_model_1.calls >= 1

    player_model_2 = _FakePlayerModel(boxes=[])
    ball_model_2 = _FakePlayerModel(boxes=[])  # would differ if called
    stage_2 = RoboflowDetector(
        cache_dir=str(cache_dir), cache_mode="readwrite", use_ball_model=True,
    )
    stage_2._player_model = player_model_2
    stage_2._ball_model = ball_model_2
    out_2 = stage_2.detect(_ctx(frames))

    assert ball_model_2.calls == 0
    balls_1 = [d for d in out_1.frames[0].detections if d.cls == DetectionClass.BALL]
    balls_2 = [d for d in out_2.frames[0].detections if d.cls == DetectionClass.BALL]
    assert balls_1 == balls_2
    assert len(balls_1) >= 1


# --- provenance() carries the cache's content hash ---------------------------
#
# cache_dir/cache_mode are NOT re-asserted on ModelProvenance here: they
# already appear in StageProvenance.params (the resolved-params snapshot),
# so detections_cache_hash is the only cache fact that needs a home on the
# model entry itself (see provenance.py's ModelProvenance docstring).


def test_provenance_records_content_hash_in_dedicated_field(tmp_path):
    cache_dir = tmp_path / "cache"
    frames = _three_frames()
    model = _FakePlayerModel(boxes=[(50.0, 50.0, 20.0, 40.0, 0.9, 2)])

    stage = RoboflowDetector(cache_dir=str(cache_dir), cache_mode="readwrite")
    stage._player_model = model
    cache = HostedDetectionCache(cache_dir)
    empty_hash = stage.provenance()[0].detections_cache_hash
    assert empty_hash == cache.content_hash()  # empty-cache hash, not None

    stage.detect(_ctx(frames))
    warm_hash = stage.provenance()[0].detections_cache_hash

    assert empty_hash != warm_hash
    assert warm_hash == cache.content_hash()


def test_provenance_off_mode_leaves_cache_hash_null():
    stage = RoboflowDetector(cache_mode="off")
    m = stage.provenance()[0]
    assert m.detections_cache_hash is None
    # dataset_split_manifest fields are unrelated to caching and stay at
    # their own defaults regardless of cache_mode.
    assert m.dataset_split_manifest is None
    assert m.dataset_split_manifest_sha256 is None


# --- runner integration: manifest records the post-detect() cache hash ------


def test_runner_records_post_detect_cache_hash_not_pre_run_empty_hash(tmp_path, monkeypatch):
    """Regression test for the ordering bug: provenance() called only before
    detect() would record the empty-cache hash for a cold readwrite run. The
    runner must refresh provenance again after the stage executes, so the
    manifest reflects the cache actually used by this run."""
    import inference
    from pitchlab_core.config import PipelineConfig, StageConfig, VideoConfig
    from pitchlab_core.demo import render_demo_video
    from pitchlab_core.runner import PipelineRunner
    from pitchlab_core.schemas.run import StageKind, StageStatus

    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    fake_model = _FakePlayerModel(boxes=[(30.0, 30.0, 10.0, 20.0, 0.9, 2)])
    # prepare() does `from inference import get_model` locally at call time,
    # so patching the real `inference` module attribute is what's actually
    # invoked -- no need to reach into pitchlab_core.stages.detect.roboflow.
    monkeypatch.setattr(inference, "get_model", lambda model_id, api_key: fake_model)

    video = render_demo_video(tmp_path / "clip.mp4", duration_s=0.4, fps=10, width=64, height=64)
    cache_dir = tmp_path / "cache"
    config = PipelineConfig(
        name="roboflow-cache-runner-test",
        video=VideoConfig(sample_stride=1, max_frames=3),
        stages={
            StageKind.DETECT: StageConfig(
                impl="roboflow",
                params={"cache_dir": str(cache_dir), "cache_mode": "readwrite"},
            ),
            StageKind.TRACK: StageConfig(impl="iou", params={}),
        },
    )
    runner = PipelineRunner(
        run_id="test", video_path=video, config=config, run_dir=tmp_path / "run", device="cpu"
    )
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error

    recorded_hash = manifest.provenance.stages["detect"].models[0].detections_cache_hash
    warm_cache = HostedDetectionCache(cache_dir)
    assert recorded_hash is not None
    assert recorded_hash == warm_cache.content_hash()
    empty_cache_hash = HostedDetectionCache(tmp_path / "never-written").content_hash()
    assert recorded_hash != empty_cache_hash
