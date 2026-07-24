"""FrameFeatures artifact (frame_features.npz): exact save/load round-trips,
(tracklet, source-frame) keyed lookup, and the bridge-side join of per-frame
feature pkls to tracklets across stride settings. Hand-built fixtures."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import TrackletFrame
from matchlab_core.stages.track.tdlp_full import bridge


def _small() -> FrameFeatures:
    rng = np.random.default_rng(7)
    return FrameFeatures(
        tracklet_ids=np.array([1, 1, 2], dtype=np.int64),
        frame_idxs=np.array([0, 5, 0], dtype=np.int64),
        embeddings=rng.standard_normal((3, 6, 128)).astype(np.float32),
        visibility=rng.random((3, 6)).astype(np.float32),
        keypoints_xyc=rng.random((3, 17, 3)).astype(np.float32),
        keypoints_conf=rng.random(3).astype(np.float32),
        meta={"width": 1920, "height": 1080, "source": "tdlp-full"},
    )


def test_round_trip_identical_arrays_and_keys(tmp_path: Path):
    ff = _small()
    path = tmp_path / "frame_features.npz"
    ff.save(path)
    back = FrameFeatures.load(path)

    np.testing.assert_array_equal(back.tracklet_ids, ff.tracklet_ids)
    np.testing.assert_array_equal(back.frame_idxs, ff.frame_idxs)
    np.testing.assert_array_equal(back.embeddings, ff.embeddings)
    np.testing.assert_array_equal(back.visibility, ff.visibility)
    np.testing.assert_array_equal(back.keypoints_xyc, ff.keypoints_xyc)
    np.testing.assert_array_equal(back.keypoints_conf, ff.keypoints_conf)
    assert back.meta == ff.meta
    assert back.embeddings.dtype == np.float32
    assert back.tracklet_ids.dtype == np.int64


def test_lookup_by_tracklet_and_frame():
    ff = _small()

    # (tid, frame) -> row index; missing keys -> None
    row = ff.get(1, 5)
    assert row is not None
    np.testing.assert_array_equal(row.embedding, ff.embeddings[1])
    np.testing.assert_array_equal(row.visibility, ff.visibility[1])
    np.testing.assert_array_equal(row.keypoints_xyc, ff.keypoints_xyc[1])
    assert row.keypoints_conf == ff.keypoints_conf[1]
    assert ff.get(1, 99) is None
    assert ff.get(42, 0) is None

    # per-tracklet slice, ordered by frame
    rows = ff.for_tracklet(1)
    assert [r.frame_idx for r in rows] == [0, 5]
    assert ff.for_tracklet(2)[0].frame_idx == 0
    assert ff.for_tracklet(42) == []


# --- bridge: pkl collection + join to tracklets ---------------------------

W, H = 100.0, 50.0  # image size the pkl boxes are normalised by


def _pkl_det(x, y, w, h, fill: float):
    """One gen_features-schema detection dict with a recognisable embedding."""
    return {
        "bbox_xywh": [x / W, y / H, w / W, h / H],
        "bbox_conf": 0.9,
        "keypoints_xyc": [[0.5, 0.5, 0.8]] * 17,
        "keypoints_conf": 0.8,
        "appearance_embeddings": np.full((6, 128), fill, dtype=np.float32).tolist(),
        "appearance_visibility": [1.0] * 6,
    }


def _write_pkls(feat_dir: Path, per_frame: dict[int, list]):
    feat_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, dets in per_frame.items():
        with open(feat_dir / f"{local_idx:06d}.pkl", "wb") as f:
            pickle.dump(dets, f)


def _tracklet(tid, frames: list[tuple[int, Box]]) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[TrackletFrame(frame_idx=fi, box=b, confidence=1.0) for fi, b in frames],
    )


def test_join_features_to_tracklets_across_stride(tmp_path: Path):
    # Stride-5 sampling: local frames 0,1,2 <-> source frames 0,5,10.
    local_to_source = [0, 5, 10]
    # Frame local-0 has two detections; frames local-1/2 have one each.
    _write_pkls(
        tmp_path,
        {
            0: [_pkl_det(10, 10, 10, 20, fill=1.0), _pkl_det(60, 10, 10, 20, fill=2.0)],
            1: [_pkl_det(12, 10, 10, 20, fill=3.0)],
            2: [],
        },
    )
    # Tracklets are in SOURCE frame space (post-remap), boxes in pixels (xyxy).
    t1 = _tracklet(1, [(0, Box(x1=10, y1=10, x2=20, y2=30)), (5, Box(x1=12, y1=10, x2=22, y2=30))])
    t2 = _tracklet(2, [(0, Box(x1=60, y1=10, x2=70, y2=30))])

    ff = bridge.join_features_to_tracklets(
        [t1, t2], tmp_path, local_to_source, width=int(W), height=int(H)
    )

    assert len(ff) == 3
    # Each tracklet frame picked up the co-located detection's embedding.
    assert ff.get(1, 0).embedding[0, 0] == 1.0
    assert ff.get(1, 5).embedding[0, 0] == 3.0
    assert ff.get(2, 0).embedding[0, 0] == 2.0
    assert ff.get(2, 5) is None  # tracklet 2 absent on that frame
    assert ff.meta["width"] == int(W) and ff.meta["height"] == int(H)


def test_join_skips_unmatched_tracklet_frames(tmp_path: Path):
    # One detection far away from the tracklet's box -> no join row.
    _write_pkls(tmp_path, {0: [_pkl_det(80, 30, 10, 15, fill=1.0)]})
    t = _tracklet(1, [(0, Box(x1=0, y1=0, x2=10, y2=20))])
    ff = bridge.join_features_to_tracklets([t], tmp_path, [0], width=int(W), height=int(H))
    assert len(ff) == 0


def test_join_assigns_each_detection_at_most_once(tmp_path: Path):
    # Two overlapping tracklet boxes, one detection: best IoU wins, the other
    # tracklet frame is left featureless rather than sharing the row.
    _write_pkls(tmp_path, {0: [_pkl_det(10, 10, 10, 20, fill=1.0)]})
    exact = _tracklet(1, [(0, Box(x1=10, y1=10, x2=20, y2=30))])
    near = _tracklet(2, [(0, Box(x1=13, y1=10, x2=23, y2=30))])
    ff = bridge.join_features_to_tracklets(
        [near, exact], tmp_path, [0], width=int(W), height=int(H)
    )
    assert len(ff) == 1
    assert ff.get(1, 0) is not None
    assert ff.get(2, 0) is None
