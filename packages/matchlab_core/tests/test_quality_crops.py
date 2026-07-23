"""Tests for the quality-gated crop sampler (`matchlab_core.crops`).

The isolation gate is the behavior that matters most: overlapping-track crops
(a duel, a tackle) silently mix two players' pixels into one "full-body" crop,
which is exactly the kind of contamination that produced confident wrong
identity matches in the July 2026 face-harvesting experiments. Every other
gate (height, confidence, sharpness) is standard hygiene and gets a narrower
test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from matchlab_core.crops import sample_quality_crops
from matchlab_core.schemas import Box, DetectionClass, Tracklet, TrackletFrame
from matchlab_core.video import Frame

IMG_H, IMG_W = 210, 300


def _img(fill: int = 60) -> np.ndarray:
    return np.full((IMG_H, IMG_W, 3), fill, dtype=np.uint8)


def _checkerboard(h: int = IMG_H, w: int = IMG_W, square: int = 20) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, square):
        for x in range(0, w, square):
            if ((y // square) + (x // square)) % 2 == 0:
                img[y : y + square, x : x + square] = 255
    return img


@dataclass
class _FakeCtx:
    """Only `.frames()` is exercised by the sampler; no need for the full
    StageContext (store/video/config) machinery."""

    _frames: list[Frame]

    def frames(self):
        return iter(self._frames)


def _tf(frame_idx: int, box: Box, confidence: float = 0.9) -> TrackletFrame:
    return TrackletFrame(frame_idx=frame_idx, box=box, confidence=confidence)


def _tr(
    tid: int, frames: list[TrackletFrame], cls: DetectionClass = DetectionClass.PLAYER
) -> Tracklet:
    return Tracklet(tracklet_id=tid, cls=cls, frames=frames)


# 1. Isolation gate -----------------------------------------------------------


def test_isolation_gate_keeps_only_the_separated_frame():
    overlap_a = Box(x1=10, y1=10, x2=100, y2=200)  # heavy overlap with overlap_b
    overlap_b = Box(x1=30, y1=10, x2=120, y2=200)
    apart_a = Box(x1=10, y1=10, x2=100, y2=200)
    apart_b = Box(x1=220, y1=10, x2=290, y2=200)  # far away -> isolated

    tr1 = _tr(1, [_tf(0, overlap_a), _tf(1, apart_a)])
    tr2 = _tr(2, [_tf(0, overlap_b), _tf(1, apart_b)])

    ctx = _FakeCtx([Frame(frame_idx=0, t=0.0, image=_img()), Frame(frame_idx=1, t=0.1, image=_img())])
    result = sample_quality_crops(ctx, [tr1, tr2])

    assert [c.frame_idx for c in result[1]] == [1]
    assert [c.frame_idx for c in result[2]] == [1]
    assert result[1][0].image.size > 0
    assert result[2][0].image.size > 0


# 2. Height / confidence hard gates -------------------------------------------


def test_height_and_confidence_gates_reject_bad_frames():
    tiny = Box(x1=10, y1=10, x2=100, y2=50)  # height 40 < min_box_height_px
    low_conf = Box(x1=10, y1=10, x2=100, y2=200)  # height 190, but conf too low
    good = Box(x1=10, y1=10, x2=100, y2=200)

    tr = _tr(
        1,
        [
            _tf(0, tiny, confidence=0.9),
            _tf(1, low_conf, confidence=0.1),
            _tf(2, good, confidence=0.9),
        ],
    )
    ctx = _FakeCtx([Frame(frame_idx=i, t=i * 0.1, image=_img()) for i in range(3)])
    result = sample_quality_crops(ctx, [tr])

    assert [c.frame_idx for c in result[1]] == [2]


# 3. Temporal bucketing --------------------------------------------------------


def test_temporal_bucketing_spreads_picks_across_the_span():
    per_tracklet = 4
    frame_idxs = list(range(0, 190, 20))  # 10 good frames, span 0..180
    box = Box(x1=10, y1=10, x2=100, y2=200)
    tr = _tr(1, [_tf(i, box, confidence=0.9) for i in frame_idxs])
    ctx = _FakeCtx([Frame(frame_idx=i, t=i * 0.1, image=_img()) for i in frame_idxs])

    result = sample_quality_crops(ctx, [tr], per_tracklet=per_tracklet)
    picked = sorted(c.frame_idx for c in result[1])

    assert len(picked) <= per_tracklet
    assert len(picked) > 1
    # picks must not all clump near the start of the tracklet's span.
    span = frame_idxs[-1] - frame_idxs[0]
    assert (picked[-1] - picked[0]) > 0.5 * span


# 4. Zero-survivor tracklet -----------------------------------------------------


def test_zero_survivor_tracklet_maps_to_empty_list_not_missing_key():
    tiny = Box(x1=10, y1=10, x2=100, y2=50)  # height 40, always rejected
    tr = _tr(1, [_tf(0, tiny), _tf(1, tiny)])
    ctx = _FakeCtx([Frame(frame_idx=i, t=i * 0.1, image=_img()) for i in range(2)])

    result = sample_quality_crops(ctx, [tr])

    assert 1 in result
    assert result[1] == []


# 5. Quality score ordering -----------------------------------------------------


def test_quality_orders_taller_and_isolated_frames_higher():
    tall_isolated = Box(x1=10, y1=10, x2=100, y2=200)  # height 190, alone at frame 0
    short_isolated = Box(x1=10, y1=10, x2=100, y2=75)  # height 65, alone at frame 1
    tall_contaminated_a = Box(x1=0, y1=10, x2=100, y2=200)  # height 190
    tall_contaminated_b = Box(x1=80, y1=10, x2=180, y2=200)  # overlaps a, iou ~0.11

    tr_tall = _tr(100, [_tf(0, tall_isolated)])
    tr_short = _tr(101, [_tf(1, short_isolated)])
    tr_contaminated = _tr(102, [_tf(2, tall_contaminated_a)])
    tr_contaminant = _tr(103, [_tf(2, tall_contaminated_b)])

    ctx = _FakeCtx([Frame(frame_idx=i, t=i * 0.1, image=_img()) for i in range(3)])
    result = sample_quality_crops(ctx, [tr_tall, tr_short, tr_contaminated, tr_contaminant])

    q_tall = result[100][0].quality
    q_short = result[101][0].quality
    q_contaminated = result[102][0].quality

    assert q_tall > q_short
    assert q_tall > q_contaminated


# 6. Padding + clamping ---------------------------------------------------------


def test_padding_clamps_to_image_bounds_and_stays_non_empty():
    edge_box = Box(x1=0, y1=0, x2=90, y2=IMG_H - 5)  # near top-left and bottom edges
    tr = _tr(1, [_tf(0, edge_box)])
    ctx = _FakeCtx([Frame(frame_idx=0, t=0.0, image=_img())])

    result = sample_quality_crops(ctx, [tr])

    assert len(result[1]) == 1
    crop = result[1][0].image
    assert crop.size > 0
    assert crop.shape[0] <= IMG_H
    assert crop.shape[1] <= IMG_W


# 7. Sharpness gate --------------------------------------------------------------


def test_sharpness_gate_drops_blurred_keeps_textured_crop():
    box = Box(x1=10, y1=10, x2=150, y2=180)
    tr_blur = _tr(1, [_tf(0, box)])
    tr_textured = _tr(2, [_tf(1, box)])

    flat_frame = Frame(frame_idx=0, t=0.0, image=_img(fill=128))
    textured_frame = Frame(frame_idx=1, t=0.1, image=_checkerboard())
    ctx = _FakeCtx([flat_frame, textured_frame])

    result = sample_quality_crops(ctx, [tr_blur, tr_textured], min_sharpness=100.0)

    assert result[1] == []
    assert len(result[2]) == 1


# 8. Off-frame box yields no crop (never a wrong region) --------------------------


def test_box_entirely_outside_frame_yields_no_crop():
    # Passes every metadata gate (tall, confident, isolated) but lies fully off
    # the left edge — a negative slice stop must not return a wrong region.
    off_frame = Box(x1=-120, y1=10, x2=-20, y2=200)
    tr = _tr(1, [_tf(0, off_frame)])
    ctx = _FakeCtx([Frame(frame_idx=0, t=0.0, image=_img())])

    result = sample_quality_crops(ctx, [tr])

    assert result[1] == []


# 9. h95 normalization is PLAYER-only ---------------------------------------------


def test_h95_ignores_non_player_tracklets():
    player_box = Box(x1=10, y1=10, x2=100, y2=200)  # height 190, alone at frame 0
    giant_gk_box = Box(x1=150, y1=0, x2=250, y2=400)  # height 400, alone at frame 1
    tr_player = _tr(1, [_tf(0, player_box, confidence=0.9)])
    tr_gk = _tr(2, [_tf(1, giant_gk_box, confidence=0.9)], cls=DetectionClass.GOALKEEPER)
    ctx = _FakeCtx([Frame(frame_idx=i, t=i * 0.1, image=_img()) for i in range(2)])

    result = sample_quality_crops(ctx, [tr_player, tr_gk])

    # h95 comes from PLAYER heights only (= 190), so the player frame gets
    # h_norm = 1.0 and quality = confidence exactly. If the goalkeeper's 400px
    # box leaked into h95, quality would drop to roughly 0.44.
    assert result[1][0].quality == pytest.approx(0.9)


def test_all_non_player_tracklets_use_h_norm_guard_without_crashing():
    box = Box(x1=10, y1=10, x2=100, y2=200)
    tr = _tr(1, [_tf(0, box, confidence=0.8)], cls=DetectionClass.REFEREE)
    ctx = _FakeCtx([Frame(frame_idx=0, t=0.0, image=_img())])

    result = sample_quality_crops(ctx, [tr])

    # No PLAYER heights -> h95 = 0 -> h_norm guard kicks in (1.0).
    assert len(result[1]) == 1
    assert result[1][0].quality == pytest.approx(0.8)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
