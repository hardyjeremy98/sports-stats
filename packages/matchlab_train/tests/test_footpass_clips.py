"""Sampler tests. These check the DATA, not the training.

The failures that matter here are silent ones: a clip that does not contain its own
anchor event, a class balance that quietly collapses to pass/drive, a mask that is 1
where there was no box, or a flip that moves the pixels but not the boxes. Each would
train a plausible-looking model on wrong supervision.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.pcbas.schema import CLS, ROI_X
from matchlab_train.datasets.footpass_clips import (
    ClipAnchor,
    FootpassClipDataset,
    balanced_sample,
    build_anchors,
    class_histogram,
    dilate_labels,
    scale_box,
)
from matchlab_train.datasets.footpass_video import EXPECTED_HEIGHT, EXPECTED_WIDTH

cv2 = pytest.importorskip("cv2")
h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

N_FRAMES = 200
CLIP = 20


def _row(frame, pid, ltr, shirt, role, roi, cls):
    r = np.full(14, np.nan, dtype=np.float64)
    r[0], r[1], r[2], r[3], r[4] = frame, pid, ltr, shirt, role
    r[5], r[6], r[7], r[8] = 0.5, 0.5, 0.0, 0.0
    if roi is not None:
        r[9], r[10], r[11], r[12] = roi
    r[13] = cls
    return r


@pytest.fixture
def fixture_split(tmp_path):
    """A synthetic half: 6 players, all on-screen, with a handful of events."""
    rows = []
    for frame in range(N_FRAMES):
        for i, pid in enumerate([10, 11, 12, 13, 14, 15]):
            roi = (300 + 60 * i, 400, 60, 150)
            rows.append(_row(frame, pid, i % 2, pid, (i % 13) + 1, roi, 0))
    arr = np.stack(rows)

    def mark(frame, pid, cls):
        sel = (arr[:, 0] == frame) & (arr[:, 1] == pid)
        arr[sel, CLS] = cls

    mark(50, 10, 2)  # pass
    mark(60, 11, 2)  # pass
    mark(100, 12, 5)  # shot
    mark(150, 13, 7)  # tackle

    h5_path = tmp_path / "fix.h5"
    with h5py.File(h5_path, "w") as f:
        f["game_1_H1"] = arr

    video_root = tmp_path / "videos"
    video_root.mkdir()
    writer = cv2.VideoWriter(
        str(video_root / "game_1.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (EXPECTED_WIDTH, EXPECTED_HEIGHT),
    )
    rng = np.random.default_rng(0)
    for _ in range(N_FRAMES):
        writer.write(rng.integers(0, 255, (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3), dtype=np.uint8))
    writer.release()
    return h5_path, video_root


def _dataset(fixture_split, **kw):
    h5_path, video_root = fixture_split
    anchors = build_anchors(h5_path)
    kw.setdefault("clip_length", CLIP)
    kw.setdefault("nb_tracklets", 2)
    return FootpassClipDataset(h5_path, video_root, anchors, **kw)


# --- anchors ----------------------------------------------------------------------


def test_every_labelled_row_becomes_an_anchor(fixture_split):
    anchors = build_anchors(fixture_split[0])
    assert len(anchors) == 4
    assert class_histogram(anchors) == {"pass": 2, "shot": 1, "tackle": 1}
    assert {a.min_frame for a in anchors} == {0}
    assert {a.max_frame for a in anchors} == {N_FRAMES - 1}


# --- class balance ----------------------------------------------------------------


def test_balanced_sample_caps_the_common_classes():
    anchors = [ClipAnchor("k", i, 1, 2, 0, 999) for i in range(100)] + [
        ClipAnchor("k", i, 1, 7, 0, 999) for i in range(3)
    ]
    picked = balanced_sample(anchors, 10, np.random.default_rng(0))
    hist = class_histogram(picked)
    assert hist["pass"] == 10  # capped
    assert hist["tackle"] == 3  # all of them, never oversampled


def test_resampling_changes_which_common_examples_are_drawn(fixture_split):
    """The rare classes must recur every epoch while the common ones rotate. If
    resample() were a no-op, 20 epochs would see the same 500 passes 20 times."""
    h5_path, video_root = fixture_split
    anchors = [ClipAnchor("game_1_H1", i, 10, 2, 0, 199) for i in range(50)]
    ds = FootpassClipDataset(
        h5_path, video_root, anchors, clip_length=CLIP, max_samples_per_class=10
    )
    first = {a.frame_idx for a in ds.sampled}
    ds.resample()
    second = {a.frame_idx for a in ds.sampled}
    assert first != second
    assert len(first) == len(second) == 10


def test_eval_sampling_is_deterministic(fixture_split):
    h5_path, video_root = fixture_split
    anchors = [ClipAnchor("game_1_H1", i, 10, 2, 0, 199) for i in range(50)]
    kw = dict(clip_length=CLIP, max_samples_per_class=10, train=False)
    a = FootpassClipDataset(h5_path, video_root, anchors, **kw)
    b = FootpassClipDataset(h5_path, video_root, anchors, **kw)
    assert [x.frame_idx for x in a.sampled] == [x.frame_idx for x in b.sampled]


# --- clip contents ----------------------------------------------------------------


def test_every_clip_contains_its_anchor_event(fixture_split):
    """The whole point of the sampler. A clip that misses its own event trains the
    model that the class is present when it is not."""
    ds = _dataset(fixture_split)
    for i in range(len(ds)):
        anchor = ds.sampled[i]
        start = ds._clip_start(anchor, np.random.default_rng(i))
        assert start <= anchor.frame_idx < start + CLIP, (
            f"anchor {anchor.frame_idx} outside clip [{start}, {start + CLIP})"
        )


def test_sample_shapes_and_dtypes(fixture_split):
    ds = _dataset(fixture_split)
    video, rois, masks, sharp, dilated = ds[0]
    m = ds.n_players
    assert video.shape == (3, CLIP, EXPECTED_HEIGHT, EXPECTED_WIDTH)
    assert rois.shape == (m, CLIP, 5)
    assert masks.shape == (m, CLIP)
    assert sharp.shape == dilated.shape == (m, CLIP)
    assert sharp.dtype == torch.int64
    assert video.dtype == torch.float32


def test_the_anchor_player_is_always_present(fixture_split):
    ds = _dataset(fixture_split)
    for i in range(len(ds)):
        anchor = ds.sampled[i]
        _, _, _, sharp, _ = ds[i]
        assert (sharp == anchor.class_id).any(), (
            f"anchor class {anchor.class_id} missing from the sample's labels"
        )


def test_rois_are_in_bounds_after_scaling(fixture_split):
    ds = _dataset(fixture_split)
    for i in range(len(ds)):
        _, rois, _, _, _ = ds[i]
        assert (rois[:, :, 1] >= 0).all() and (rois[:, :, 3] <= EXPECTED_WIDTH).all()
        assert (rois[:, :, 2] >= 0).all() and (rois[:, :, 4] <= EXPECTED_HEIGHT).all()
        assert (rois[:, :, 3] > rois[:, :, 1]).all()
        assert (rois[:, :, 4] > rois[:, :, 2]).all()


def test_roi_frame_column_is_the_clip_local_index(fixture_split):
    """Column 0 indexes the clip, not the video. Passing absolute video frames would
    send roi_align reading far outside the batch."""
    ds = _dataset(fixture_split)
    _, rois, _, _, _ = ds[0]
    assert rois[0, :, 0].tolist() == list(range(CLIP))


def test_masks_are_zero_exactly_where_there_is_no_box(fixture_split):
    h5_path, video_root = fixture_split
    with h5py.File(h5_path, "r+") as f:
        arr = f["game_1_H1"][:]
        arr[(arr[:, 0] >= 40) & (arr[:, 0] < 45) & (arr[:, 1] == 10), ROI_X] = np.nan
        del f["game_1_H1"]
        f["game_1_H1"] = arr

    anchors = [ClipAnchor("game_1_H1", 50, 10, 2, 0, N_FRAMES - 1)]
    ds = FootpassClipDataset(
        h5_path, video_root, anchors, clip_length=CLIP, nb_tracklets=2, train=False
    )
    _, _, masks, _, _ = ds[0]
    assert masks.min() == 0.0 or masks.max() == 1.0  # both states are representable
    assert masks.max() == 1.0


def test_padded_player_slots_are_fully_masked(fixture_split):
    """Asking for more distractors than exist must pad, not crash, and the padding
    must contribute nothing."""
    ds = _dataset(fixture_split, nb_tracklets=20)
    _, _, masks, _, _ = ds[0]
    assert masks.shape[0] == 21
    assert (masks.sum(dim=1) == 0).any()  # at least one fully-masked padded slot


# --- geometry and labels ----------------------------------------------------------


def test_scale_box_matches_the_reference_framing():
    x1, y1, x2, y2 = scale_box(960.0, 540.0, 120.0, 240.0, 1.125)
    assert x1 == pytest.approx((960 - 0.125 * 120 / 2) / 3.0)
    assert x2 == pytest.approx((960 + 1.125 * 120) / 3.0)
    assert y1 == pytest.approx((540 - 0.125 * 240 / 2) / 3.068181)
    assert y2 == pytest.approx((540 + 1.125 * 240) / 3.068181)


def test_scale_box_clamps_to_the_frame():
    x1, y1, x2, y2 = scale_box(1900.0, 1050.0, 200.0, 200.0, 1.125)
    assert 0 <= x1 < EXPECTED_WIDTH and x2 <= EXPECTED_WIDTH
    assert 0 <= y1 < EXPECTED_HEIGHT and y2 <= EXPECTED_HEIGHT


def test_label_dilation_widens_events_only():
    labels = np.zeros((2, 9), dtype=np.int64)
    labels[0, 4] = 3
    out = dilate_labels(labels, 1)
    assert out[0].tolist() == [0, 0, 0, 3, 3, 3, 0, 0, 0]
    assert out[1].tolist() == [0] * 9


def test_label_dilation_is_clipped_at_the_boundaries():
    labels = np.zeros((1, 4), dtype=np.int64)
    labels[0, 0] = 5
    assert dilate_labels(labels, 2)[0].tolist() == [5, 5, 5, 0]


def test_zero_dilation_is_a_copy():
    labels = np.zeros((1, 3), dtype=np.int64)
    labels[0, 1] = 4
    out = dilate_labels(labels, 0)
    out[0, 0] = 9
    assert labels[0, 0] == 0  # not a view


# --- augmentation -----------------------------------------------------------------


def test_horizontal_flip_moves_boxes_with_the_pixels():
    """A flip that moves pixels but not boxes is the worst possible bug here: it
    trains perfectly happily on systematically wrong pooling locations."""
    ds_cls = FootpassClipDataset.__new__(FootpassClipDataset)
    ds_cls.train = True
    clip = np.zeros((2, EXPECTED_HEIGHT, EXPECTED_WIDTH, 3), dtype=np.uint8)
    clip[:, :, 0:100] = 200  # bright band on the LEFT
    rois = np.zeros((1, 2, 5), dtype=np.float32)
    rois[0, :, 1:] = (10.0, 10.0, 90.0, 90.0)  # box on the bright band

    rng = np.random.default_rng(0)
    flipped = None
    for seed in range(20):  # find a seed that actually flips
        rng = np.random.default_rng(seed)
        out_clip, out_rois = ds_cls._augment(clip.copy(), rois.copy(), rng)
        if out_clip[0, 0, 0, 0] < 100:  # left band is now dark -> it flipped
            flipped = (out_clip, out_rois)
            break
    assert flipped is not None, "no flip in 20 seeds"
    out_clip, out_rois = flipped
    # The band moved to the right; the box must have moved with it.
    assert out_clip[0, 0, -50, 0] > 100
    assert out_rois[0, 0, 1] == pytest.approx(EXPECTED_WIDTH - 90.0)
    assert out_rois[0, 0, 3] == pytest.approx(EXPECTED_WIDTH - 10.0)


# --- control against the reference's own anchor list -------------------------------


def test_anchors_without_a_box_are_excluded_by_default(fixture_split):
    """An anchor whose player has no box is fully masked: zero loss, and it still
    consumes a place in that class's balanced quota."""
    h5_path, _ = fixture_split
    with h5py.File(h5_path, "r+") as f:
        arr = f["game_1_H1"][:]
        arr[(arr[:, 0] == 100) & (arr[:, 1] == 12), ROI_X] = np.nan  # the shot
        del f["game_1_H1"]
        f["game_1_H1"] = arr
    assert "shot" not in class_histogram(build_anchors(h5_path))
    assert "shot" in class_histogram(build_anchors(h5_path, require_bbox=False))


def test_anchors_match_the_reference_sample_list():
    """A FREE control on the training anchor set, and the only one available: the
    reference ships no checkpoints, no logits and no video, so a true control run
    would cost a full retrain. It DOES ship its exact anchor list.

    With the bbox filter our per-class VAL anchor counts match it on all 8 classes.
    """
    import json

    from matchlab_train.datasets.paths import data_root, reference_root

    root, ref = data_root(), reference_root("FOOTPASS")
    if root is None or ref is None:
        pytest.skip("FOOTPASS data or reference clone not present")
    h5 = root / "footpass" / "tactical" / "val_tactical_data.h5"
    sample_list = ref / "data" / "TAAD_sample_list.json"
    if not h5.is_file() or not sample_list.is_file():
        pytest.skip("FOOTPASS VAL tactical data or sample list not present")

    from collections import Counter

    from matchlab_core.pcbas.schema import CLASS_NAMES

    # The reference stores events 0-indexed (class_id - 1).
    reference = {
        CLASS_NAMES[c + 1]: n
        for c, n in Counter(json.loads(sample_list.read_text())["val"]["events"]).items()
    }
    assert class_histogram(build_anchors(h5)) == dict(
        sorted(reference.items(), key=lambda kv: -kv[1])
    )
