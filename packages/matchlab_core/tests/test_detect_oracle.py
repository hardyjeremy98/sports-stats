"""Tests for the oracle detect stage: GT boxes emitted as detections, with
optional dropout/jitter knobs for tracker-ceiling sensitivity analysis.

Unit-level tests (a-c) build a `_FakeCtx` directly -- the oracle only reads
`ctx.video` (VideoMeta) and `ctx.config.video` (stride/max_frames), never
decoded pixels, so no real video file is needed. The integration test (d)
exercises the real PipelineRunner end to end with a rendered demo video and a
sibling `.gt.json`, per the repo convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from matchlab_core.config import PipelineConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import StageStatus, VideoMeta
from matchlab_core.stages.detect.oracle import OracleDetector
from matchlab_core.video import probe

FPS = 10.0
N_FRAMES = 6
ORACLE_CONFIG_PATH = Path(__file__).parents[3] / "configs" / "pipeline.oracle-eval.yaml"

_REFEREE_BOX = Box(x1=200.0, y1=20.0, x2=230.0, y2=80.0)
_BALL_BOX = Box(x1=100.0, y1=100.0, x2=110.0, y2=110.0)


def _player_box(i: int) -> Box:
    return Box(x1=10.0 + i, y1=20.0, x2=50.0 + i, y2=80.0)


def _make_gt(n_frames: int = N_FRAMES) -> GroundTruth:
    player = GroundTruthTrack(
        track_id=1,
        role="player",
        frames=[GroundTruthFrame(frame_idx=i, box=_player_box(i)) for i in range(n_frames)],
    )
    referee = GroundTruthTrack(
        track_id=2,
        role="referee",
        frames=[GroundTruthFrame(frame_idx=i, box=_REFEREE_BOX) for i in range(n_frames)],
    )
    ball = GroundTruthTrack(
        track_id=3,
        role="ball",
        frames=[GroundTruthFrame(frame_idx=i, box=_BALL_BOX) for i in range(n_frames)],
    )
    other = GroundTruthTrack(
        track_id=4,
        role="other",
        frames=[GroundTruthFrame(frame_idx=0, box=Box(x1=0, y1=0, x2=5, y2=5))],
    )
    return GroundTruth(
        source="test", sequence="tiny", fps=FPS, width=320, height=240,
        seq_length=n_frames, tracks=[player, referee, ball, other],
    )


def _write_gt(path: Path, gt: GroundTruth) -> Path:
    path.write_text(gt.model_dump_json())
    return path


@dataclass
class _FakeVideoConfig:
    sample_stride: int = 1
    max_frames: int | None = None


@dataclass
class _FakeConfig:
    video: _FakeVideoConfig


@dataclass
class _FakeCtx:
    video: VideoMeta
    config: _FakeConfig


def _ctx(video_path: Path, frame_count: int = N_FRAMES, stride: int = 1, max_frames=None) -> _FakeCtx:
    meta = VideoMeta(
        path=str(video_path), fps=FPS, frame_count=frame_count, width=320, height=240,
        duration_s=frame_count / FPS,
    )
    return _FakeCtx(video=meta, config=_FakeConfig(video=_FakeVideoConfig(stride, max_frames)))


# -- (a) fixture: no knobs -> exact match; knobs -> configured degradation ----


def test_no_knobs_matches_gt_exactly(tmp_path):
    gt_path = _write_gt(tmp_path / "gt.json", _make_gt())
    ctx = _ctx(tmp_path / "clip.mp4")
    detector = OracleDetector(gt_path=str(gt_path))

    out = detector.detect(ctx)

    assert [fd.frame_idx for fd in out.frames] == list(range(N_FRAMES))
    for i, fd in enumerate(out.frames):
        assert len(fd.detections) == 3  # player + referee + ball; "other" is skipped
        by_cls = {d.cls: d for d in fd.detections}
        assert by_cls[DetectionClass.PLAYER].box == _player_box(i)
        assert by_cls[DetectionClass.PLAYER].confidence == 1.0
        assert by_cls[DetectionClass.REFEREE].box == _REFEREE_BOX
        assert by_cls[DetectionClass.BALL].box == _BALL_BOX
    assert len(out.ball) == N_FRAMES  # every frame has a ball GT box


def test_dropout_rate_one_drops_every_detection(tmp_path):
    gt_path = _write_gt(tmp_path / "gt.json", _make_gt())
    ctx = _ctx(tmp_path / "clip.mp4")
    detector = OracleDetector(gt_path=str(gt_path), dropout_rate=1.0)

    out = detector.detect(ctx)

    assert len(out.frames) == N_FRAMES
    assert all(fd.detections == [] for fd in out.frames)
    assert out.ball == []


def test_jitter_is_bounded_and_seed_reproducible(tmp_path):
    gt_path = _write_gt(tmp_path / "gt.json", _make_gt())
    ctx = _ctx(tmp_path / "clip.mp4")
    jitter_px = 5.0

    def _run() -> list:
        detector = OracleDetector(gt_path=str(gt_path), jitter_px=jitter_px, seed=42)
        return detector.detect(ctx).frames

    frames_a = _run()
    frames_b = _run()
    assert frames_a == frames_b  # same seed -> identical output

    differed = False
    for i, fd in enumerate(frames_a):
        player_det = next(d for d in fd.detections if d.cls == DetectionClass.PLAYER)
        gt_box = _player_box(i)
        for got, want in (
            (player_det.box.x1, gt_box.x1),
            (player_det.box.y1, gt_box.y1),
            (player_det.box.x2, gt_box.x2),
            (player_det.box.y2, gt_box.y2),
        ):
            assert abs(got - want) <= jitter_px + 1e-9
            if got != want:
                differed = True
    assert differed, "jitter_px > 0 should perturb at least one coordinate"


# -- (b) missing GT: loud error naming both paths tried ----------------------


def test_missing_gt_raises_loudly(tmp_path):
    video_path = tmp_path / "clip.mp4"
    ctx = _ctx(video_path)
    detector = OracleDetector()  # no gt_path param, no sibling clip.gt.json

    with pytest.raises(FileNotFoundError) as exc_info:
        detector.detect(ctx)

    msg = str(exc_info.value)
    assert "gt_path" in msg
    assert str(video_path.parent / "clip.gt.json") in msg


def test_missing_explicit_gt_path_raises_loudly(tmp_path):
    ctx = _ctx(tmp_path / "clip.mp4")
    missing = tmp_path / "does-not-exist.json"
    detector = OracleDetector(gt_path=str(missing))

    with pytest.raises(FileNotFoundError) as exc_info:
        detector.detect(ctx)

    assert str(missing) in str(exc_info.value)


# -- (c) stride / max_frames respect the same iteration as synthetic ---------


def test_sample_stride_emits_only_strided_frame_indices(tmp_path):
    gt_path = _write_gt(tmp_path / "gt.json", _make_gt())
    ctx = _ctx(tmp_path / "clip.mp4", stride=2)
    detector = OracleDetector(gt_path=str(gt_path))

    out = detector.detect(ctx)

    assert [fd.frame_idx for fd in out.frames] == [0, 2, 4]


def test_max_frames_caps_the_emitted_frames(tmp_path):
    gt_path = _write_gt(tmp_path / "gt.json", _make_gt())
    ctx = _ctx(tmp_path / "clip.mp4", max_frames=2)
    detector = OracleDetector(gt_path=str(gt_path))

    out = detector.detect(ctx)

    assert [fd.frame_idx for fd in out.frames] == [0, 1]


# -- (d) integration: real PipelineRunner, oracle config, sibling GT ---------


def test_pipeline_run_with_oracle_detect(tmp_path):
    video = render_demo_video(tmp_path / "clip.mp4", duration_s=0.6, fps=FPS, width=320, height=240)
    meta = probe(video)
    gt = _make_gt(n_frames=meta.frame_count)
    (video.parent / f"{video.stem}.gt.json").write_text(gt.model_dump_json())

    config = PipelineConfig.from_yaml(ORACLE_CONFIG_PATH)
    run_dir = tmp_path / "run"
    runner = PipelineRunner(run_id="test", video_path=video, config=config, run_dir=run_dir)
    manifest = runner.run()
    assert manifest.status == StageStatus.COMPLETED, manifest.error

    assert (run_dir / "tracklets.json").exists()
    tracklets = json.loads((run_dir / "tracklets.json").read_text())
    assert len(tracklets) > 0

    rows = [json.loads(line) for line in (run_dir / "detections.jsonl").read_text().splitlines()]
    expected_idx = list(range(0, meta.frame_count, config.video.sample_stride))
    assert [r["frame_idx"] for r in rows] == expected_idx
    for r in rows:
        i = r["frame_idx"]
        assert len(r["detections"]) == 3
        by_cls = {d["cls"]: d for d in r["detections"]}
        assert by_cls["player"]["box"] == _player_box(i).model_dump()
        assert by_cls["player"]["confidence"] == 1.0
        assert by_cls["referee"]["box"] == _REFEREE_BOX.model_dump()
        assert by_cls["ball"]["box"] == _BALL_BOX.model_dump()
