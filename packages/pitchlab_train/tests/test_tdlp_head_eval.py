"""Integration test for the cached-feature sweep scoring path
(`tdlp_head_eval.sweep`): synthetic cache + GT -> loop -> run dir ->
`evaluate_run` -> headline metrics, exercising the real scoring integration
(the risky part) without GPU or DINOv2.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from pitchlab_core.stages.track.tdlp.model import ModalityConfig, build_head
from pitchlab_train.tdlp_head_eval import sweep

APP_DIM = 8


def _write_synthetic_case(tmp_path):
    n_frames = 12
    app_by_id = {1: np.eye(APP_DIM, dtype=np.float32)[0], 2: np.eye(APP_DIM, dtype=np.float32)[3]}
    frame_ids, boxes, confs, apps = [], [], [], []
    gt_tracks = {1: [], 2: []}
    for f in range(n_frames):
        for tid, base_x in ((1, 200.0), (2, 1400.0)):
            x = base_x + 2.0 * f
            box = [x, 500.0, x + 60.0, 700.0]
            frame_ids.append(f)
            boxes.append(box)
            confs.append(0.95)
            apps.append(app_by_id[tid] + np.random.default_rng(f * 10 + tid).normal(0, 0.01, APP_DIM).astype(np.float32))
            gt_tracks[tid].append({"frame_idx": f, "box": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}})

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.savez_compressed(
        cache_dir / "SEQ1.npz",
        frame_ids=np.asarray(frame_ids, np.int64),
        boxes=np.asarray(boxes, np.float32),
        confs=np.asarray(confs, np.float32),
        apps=np.stack(apps),
        width=1920, height=1080, appearance_dim=APP_DIM,
        frame_count=n_frames, fps=25.0,
    )
    gt = {
        "source": "test", "sequence": "SEQ1", "fps": 25.0, "width": 1920, "height": 1080,
        "seq_length": n_frames,
        "tracks": [{"track_id": tid, "role": "player", "frames": fr} for tid, fr in gt_tracks.items()],
    }
    gt_path = tmp_path / "SEQ1.gt.json"
    gt_path.write_text(json.dumps(gt))
    manifest = {"sequences": [{"name": "SEQ1", "role": "held_out",
                               "gt": str(gt_path), "video": "unused.mp4"}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # tiny random head checkpoint matching the cache's appearance dim
    modality = ModalityConfig(use_keypoints=False, use_appearance=True, appearance_dim=APP_DIM)
    model = build_head(modality, hidden_dim=16, mm_dim=16)
    ckpt = tmp_path / "head.pt"
    torch.save({"model": model.state_dict(),
                "config": {"use_keypoints": False, "use_appearance": True,
                           "appearance_dim": APP_DIM, "hidden_dim": 16, "remember": 8}}, ckpt)
    return str(cache_dir), str(manifest_path), str(ckpt)


def test_sweep_scoring_path_produces_metrics(tmp_path):
    cache_dir, manifest_path, ckpt = _write_synthetic_case(tmp_path)
    grid = {
        "sim_threshold": [0.7, 0.95],
        "remember_threshold": [8],
        "new_tracklet_detection_threshold": [0.5],
        "min_length": [1],
    }
    results = sweep(cache_dir, manifest_path, ckpt, str(tmp_path / "out"),
                    device="cpu", iou_threshold=0.5, grid=grid)
    assert len(results) == 2
    for _params, agg in results:
        # the scoring integration returned real numbers
        assert "idsw_tracklet" in agg
        assert "hota_tracklet" in agg
        assert isinstance(agg["idsw_tracklet"], (int, float))
    # results are sorted by IDsw ascending
    idsws = [agg["idsw_tracklet"] for _p, agg in results]
    assert idsws == sorted(idsws)
    assert (tmp_path / "out" / "sweep_results.json").exists()
