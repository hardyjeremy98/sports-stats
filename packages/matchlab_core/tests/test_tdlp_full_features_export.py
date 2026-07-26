"""tdlp-full stage exports frame_features.npz (SPO-51): fake-external test in
the bridge pattern — `bridge.run_external` is monkeypatched so the "feature
generation" step writes gen_features-schema pkls and the "offline tracking"
step writes a MOT file, then the stage's own join/export runs for real.

The fake feature-gen writes each detection's embedding filled with its LOCAL
frame index, so the assertions prove the local↔source frame mapping is
honoured across stride settings, not just that arrays round-trip.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.config import VideoConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.interfaces import StageContext
from matchlab_core.registry import build
from matchlab_core.schemas import FrameDetections
from matchlab_core.schemas.detections import Detection, DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.run import ArtifactName, StageKind
from matchlab_core.stages.track.tdlp_full import bridge
from matchlab_core.stages.track.tdlp_full import stage as tdlp_full_stage

BOX = Box(x1=40.0, y1=30.0, x2=80.0, y2=120.0)


class _Config:
    def __init__(self, video: VideoConfig):
        self.video = video


def _make_ctx(tmp_path: Path, *, stride: int) -> StageContext:
    video_path = render_demo_video(
        tmp_path / "clip.mp4", duration_s=1.0, fps=10.0, width=320, height=180
    )
    from matchlab_core.video import probe

    meta = probe(video_path, sample_stride=stride)
    store = ArtifactStore(tmp_path / "run")
    return StageContext(
        video=meta, config=_Config(VideoConfig(sample_stride=stride, max_frames=4)), store=store
    )


def _fake_external_root(tmp_path: Path) -> Path:
    """A fake external-trackers checkout: every path prepare() checks exists."""
    root = tmp_path / "external"
    for rel in [
        "CAMELTrack/.venv/bin/python",
        "TDLP/.venv/bin/python",
        "bridge/gen_features.py",
        "run_tdlp_frozen.py",
        "weights/tdlp_sportsmot",
        "TDLP/history/SportsMOT/tdlp.yaml",
        "CAMELTrack/pretrained_models/reid/"
        "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar",
        "CAMELTrack/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar",
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    return root


def _fake_run_external(python_exe, script, args, *, cwd, timeout_s, label,
                       extra_pythonpath=None, progress=None):
    opts = dict(zip(args[::2], args[1::2]))
    # The real subprocess runs with cwd=external_root; every path argument
    # must be absolute or it silently resolves against the wrong directory
    # (regression: relative run dirs from the matchlab-run CLI).
    for key, value in opts.items():
        if key in ("--img-dir", "--det-file", "--out-dir", "--data-root"):
            assert Path(value).is_absolute(), f"{key} must be absolute, got {value}"
    reid_model = opts.get("--reid-model", "kpr")
    parts, dim = (1, 256) if reid_model == "prtreid" else (6, 128)
    if "feature generation" in label:
        img_dir = Path(opts["--img-dir"])
        out_dir = Path(opts["--out-dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted(img_dir.glob("*.jpg"))
        import cv2

        h, w = cv2.imread(str(imgs[0])).shape[:2]
        det_rows = np.loadtxt(opts["--det-file"], delimiter=",", ndmin=2)
        by_frame: dict[int, list] = {}
        for r in det_rows:
            by_frame.setdefault(int(r[0]), []).append(r)
        for i in range(len(imgs)):
            dets = []
            for r in by_frame.get(i + 1, []):
                dets.append({
                    "bbox_xywh": [r[2] / w, r[3] / h, r[4] / w, r[5] / h],
                    "bbox_conf": float(r[6]),
                    "keypoints_xyc": [[0.5, 0.5, 0.9]] * 17,
                    "keypoints_conf": 0.9,
                    # embedding fill = LOCAL frame idx: proves index mapping
                    "appearance_embeddings": np.full((parts, dim), float(i)).tolist(),
                    "appearance_visibility": [1.0] * parts,
                })
            with open(out_dir / f"{i:06d}.pkl", "wb") as f:
                pickle.dump(dets, f)
    elif "offline tracking" in label:
        out_dir = Path(opts["--out-dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        seq = opts["--seqs"]
        det_file = Path(opts["--data-root"]) / opts["--split"] / seq / "det" / "det.txt"
        det_rows = np.loadtxt(det_file, delimiter=",", ndmin=2)
        with open(out_dir / f"{seq}.txt", "w") as f:
            for r in det_rows:  # every detection joins track 1 (one box/frame)
                f.write(f"{int(r[0])},1,{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},-1,-1,-1\n")
    else:
        raise AssertionError(f"unexpected external call: {label}")
    return ""


@pytest.mark.parametrize("stride", [1, 2])
def test_stage_exports_frame_features(tmp_path, monkeypatch, stride):
    ctx = _make_ctx(tmp_path, stride=stride)
    monkeypatch.setattr(bridge, "run_external", _fake_run_external)
    monkeypatch.setattr(
        tdlp_full_stage, "_default_external_root", lambda: _fake_external_root(tmp_path)
    )
    detections = [
        FrameDetections(
            frame_idx=k * stride,
            t=k * stride / 10.0,
            detections=[Detection(box=BOX, confidence=0.9, cls=DetectionClass.PLAYER)],
        )
        for k in range(4)
    ]

    stage = build(StageKind.TRACK, "tdlp-full", {"device": "cpu", "reid_model": "kpr"})
    tracklets = stage.track(ctx, detections)

    assert len(tracklets) == 1
    assert [f.frame_idx for f in tracklets[0].frames] == [0, stride, 2 * stride, 3 * stride]

    path = ctx.store.path(ArtifactName.FRAME_FEATURES)
    assert path.exists()
    ff = FrameFeatures.load(path)
    assert len(ff) == 4
    tid = tracklets[0].tracklet_id
    for local, source in enumerate([0, stride, 2 * stride, 3 * stride]):
        row = ff.get(tid, source)
        assert row is not None
        assert row.embedding[0, 0] == float(local)  # local↔source mapping held
    # default keep_work=False: scratch removed, artifact survives
    assert not (ctx.store.run_dir / "_tdlp_work").exists()


def test_stage_exports_features_with_keep_work(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, stride=1)
    monkeypatch.setattr(bridge, "run_external", _fake_run_external)
    monkeypatch.setattr(
        tdlp_full_stage, "_default_external_root", lambda: _fake_external_root(tmp_path)
    )
    detections = [
        FrameDetections(
            frame_idx=k,
            t=k / 10.0,
            detections=[Detection(box=BOX, confidence=0.9, cls=DetectionClass.PLAYER)],
        )
        for k in range(4)
    ]
    stage = build(StageKind.TRACK, "tdlp-full", {"device": "cpu", "keep_work": True, "reid_model": "kpr"})
    stage.track(ctx, detections)
    assert ctx.store.path(ArtifactName.FRAME_FEATURES).exists()
    assert (ctx.store.run_dir / "_tdlp_work").exists()


def _dets(stride: int = 1) -> list[FrameDetections]:
    return [
        FrameDetections(
            frame_idx=k * stride,
            t=k * stride / 10.0,
            detections=[Detection(box=BOX, confidence=0.9, cls=DetectionClass.PLAYER)],
        )
        for k in range(4)
    ]


def _recorder(calls: list[str]):
    def _run(python_exe, script, args, *, cwd, timeout_s, label,
             extra_pythonpath=None, progress=None):
        calls.append(label)
        return _fake_run_external(
            python_exe, script, args, cwd=cwd, timeout_s=timeout_s, label=label,
            extra_pythonpath=extra_pythonpath, progress=progress,
        )

    return _run


def test_reid_model_prtreid_sources_frame_features_from_a_second_pass(tmp_path, monkeypatch):
    """The tracker keeps KPR internally (the released TDLP head consumes its
    6-part shape), but frame_features — the ASSOCIATE layer's input — must come
    from the PRTreID pass. The fake emits 1x256 for prtreid and 6x128 for kpr,
    so the written artifact's shape identifies which pass produced it."""
    calls: list[str] = []
    monkeypatch.setattr(bridge, "run_external", _recorder(calls))
    root = _fake_external_root(tmp_path)
    (root / "CAMELTrack/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar").touch()
    monkeypatch.setattr(tdlp_full_stage, "_default_external_root", lambda: root)

    ctx = _make_ctx(tmp_path, stride=1)
    stage = build(StageKind.TRACK, "tdlp-full", {"device": "cpu", "reid_model": "prtreid"})
    stage.track(ctx, _dets())

    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert feats.embeddings.shape[1:] == (1, 256), "frame_features must come from PRTreID"
    assert feats.meta["reid_model"] == "prtreid"
    assert sum("re-ID feature generation" in c for c in calls) == 1
    assert sum("feature generation (RTMPose + KPR)" in c for c in calls) == 1


def test_reid_model_defaults_to_prtreid_and_runs_the_second_pass(tmp_path, monkeypatch):
    """PRTreID is the adopted backbone (2026-07-26), so the default path must
    source frame_features from the second pass. The fake emits 1x256 for
    prtreid and 6x128 for kpr, so the artifact's shape identifies the source."""
    calls: list[str] = []
    monkeypatch.setattr(bridge, "run_external", _recorder(calls))
    root = _fake_external_root(tmp_path)
    monkeypatch.setattr(tdlp_full_stage, "_default_external_root", lambda: root)

    ctx = _make_ctx(tmp_path, stride=1)
    stage = build(StageKind.TRACK, "tdlp-full", {"device": "cpu"})
    stage.track(ctx, _dets())

    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert feats.embeddings.shape[1:] == (1, 256)
    assert feats.meta["reid_model"] == "prtreid"
    assert sum("re-ID feature generation" in c for c in calls) == 1


def test_reid_model_kpr_opts_out_of_the_second_pass(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(bridge, "run_external", _recorder(calls))
    root = _fake_external_root(tmp_path)
    monkeypatch.setattr(tdlp_full_stage, "_default_external_root", lambda: root)

    ctx = _make_ctx(tmp_path, stride=1)
    stage = build(StageKind.TRACK, "tdlp-full", {"device": "cpu", "reid_model": "kpr"})
    stage.track(ctx, _dets())

    feats = FrameFeatures.load(ctx.store.path(ArtifactName.FRAME_FEATURES))
    assert feats.embeddings.shape[1:] == (6, 128)
    assert not any("re-ID feature generation" in c for c in calls)
