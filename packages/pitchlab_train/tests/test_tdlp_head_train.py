"""Tests for the preliminary TDLP head trainer (corrupt-and-recover).

Uses tiny synthetic tracker-states (no video, no real embedder) with two
appearance-separable identities, so the clip sampler's link targets are
hand-verifiable and a short training run must drive the loss down.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from pitchlab_train.tdlp_head_train import load_states, sample_clip, save_states, train

APP_DIM = 8


def _fake_states(n_frames: int = 40, seed: int = 0) -> dict:
    """Two ids on opposite sides, each with a distinct one-hot appearance +
    slowly drifting bbox — trivially separable by appearance and position."""
    rng = np.random.default_rng(seed)
    frames = {}
    app_by_id = {1: np.eye(APP_DIM, dtype=np.float32)[0], 2: np.eye(APP_DIM, dtype=np.float32)[3]}
    for f in range(n_frames):
        recs = []
        for gid, base_x in ((1, 0.2), (2, 0.7)):
            x = base_x + 0.001 * f
            noise = rng.normal(0, 0.01, APP_DIM).astype(np.float32)
            recs.append({
                "id": gid,
                "bbox": np.array([x, 0.5, 0.05, 0.1, 1.0], np.float32),
                "app": (app_by_id[gid] + noise),
            })
        frames[f] = recs
    return {"width": 1920, "height": 1080, "appearance_dim": APP_DIM, "frames": frames}


def test_sample_clip_targets_match_same_id():
    states = _fake_states(20)
    fidx = sorted(states["frames"])
    clip = sample_clip(
        states["frames"], fidx, pos=10, remember=8, appearance_dim=APP_DIM, rng=random.Random(0)
    )
    assert clip is not None
    target = clip["target"]
    assert target.shape == (2, 2)
    # exactly one positive per track row (its own next detection)
    assert target.sum().item() == 2
    assert set(target.argmax(dim=1).tolist()) == {0, 1}
    # observed history tail is populated, earlier slots masked
    assert clip["track_mask"][:, -1].sum().item() == 0  # last slot present for both


def test_sample_clip_degenerate_returns_none():
    states = _fake_states(20)
    fidx = sorted(states["frames"])
    # pos 0 has no preceding window
    assert sample_clip(states["frames"], fidx, 0, remember=8, appearance_dim=APP_DIM,
                       rng=random.Random(0)) is None


def test_save_load_states_roundtrip(tmp_path):
    states = _fake_states(5)
    p = tmp_path / "seq.npz"
    save_states(states, str(p))
    loaded = load_states(str(p))
    assert loaded["appearance_dim"] == APP_DIM
    assert len(loaded["frames"]) == 5
    assert len(loaded["frames"][0]) == 2


def test_training_drives_loss_down_and_checkpoint_loads(tmp_path):
    # two synthetic sequences
    paths = []
    for i in range(2):
        p = tmp_path / f"seq{i}.npz"
        save_states(_fake_states(40, seed=i), str(p))
        paths.append(str(p))

    ckpt = tmp_path / "head.pt"
    cfg = train(
        [str(p) for p in paths], str(ckpt), device="cpu", epochs=6,
        clips_per_epoch=60, remember=8, hidden_dim=32, seed=0,
    )
    assert cfg["appearance_dim"] == APP_DIM

    payload = torch.load(str(ckpt), map_location="cpu")
    hist = payload["loss_history"]
    # learning: last epoch clearly below the first (separable data)
    assert hist[-1] < hist[0] * 0.7, f"loss did not fall enough: {hist}"

    # checkpoint loads into a freshly built head of the recorded config
    from pitchlab_core.stages.track.tdlp.model import ModalityConfig, build_head

    model = build_head(
        ModalityConfig(use_keypoints=False, use_appearance=True, appearance_dim=APP_DIM),
        hidden_dim=32, mm_dim=32,
    )
    model.load_state_dict(payload["model"])
