"""Action-head tests.

The ROI pooling tests are the load-bearing ones and they need no backbone: the
batch-index arithmetic that maps a (B, M, T, 5) ROI tensor onto a flattened (B*T)
feature batch is the single most likely thing to be silently wrong, and a shape test
cannot catch it -- every wrong permutation still produces the right shape. So they
plant a distinctive value at a known (batch, frame, region) and assert it comes back
out at the matching (b, m, t).
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from matchlab_core.pcbas.action_head import ActionHead, pool_player_features  # noqa: E402


def _roi(x1, y1, x2, y2, frame):
    return [float(frame), float(x1), float(y1), float(x2), float(y2)]


def test_pooled_features_come_from_where_the_roi_points():
    """Two players, same frame, different halves of the image. spatial_scale=0.125
    means input coords are 8x feature coords, so a feature map that is 5.0 on its
    left half is 5.0 for input x < 32."""
    b, c, t, h, w = 1, 2, 1, 8, 8
    feats = torch.zeros(b, c, t, h, w)
    feats[0, :, 0, :, 0:4] = 5.0  # left half only

    rois = torch.zeros(b, 2, t, 5)
    rois[0, 0, 0] = torch.tensor(_roi(8, 8, 24, 56, 0))  # well inside the left half
    rois[0, 1, 0] = torch.tensor(_roi(40, 8, 56, 56, 0))  # well inside the right half
    out = pool_player_features(feats, rois, torch.ones(b, 2, t))

    assert out.shape == (b, 2, c, t)
    assert out[0, 0, 0, 0] == pytest.approx(5.0)
    assert out[0, 1, 0, 0] == pytest.approx(0.0)


def test_frame_index_selects_the_right_feature_map():
    """A ROI tagged frame 2 must read frame 2's features, not frame 0's."""
    feats = torch.zeros(1, 1, 3, 8, 8)
    feats[0, 0, 2] = 7.0
    rois = torch.tensor(
        [[[_roi(0, 0, 64, 64, 0), _roi(0, 0, 64, 64, 1), _roi(0, 0, 64, 64, 2)]]]
    )
    out = pool_player_features(feats, rois, torch.ones(1, 1, 3))
    assert out[0, 0, 0].tolist() == pytest.approx([0.0, 0.0, 7.0])


def test_mismatched_roi_time_dimension_is_rejected():
    with pytest.raises(ValueError, match="must be"):
        pool_player_features(
            torch.zeros(1, 1, 3, 8, 8), torch.zeros(1, 1, 2, 5), torch.ones(1, 1, 2)
        )


def test_batch_index_does_not_leak_across_clips():
    """Clip 1's ROI must never pool clip 0's features. With B-major flattening the
    batch offset is `frame + b*T`; getting it wrong reads a neighbouring clip and
    still returns the correct shape."""
    b, t = 2, 4
    feats = torch.zeros(b, 1, t, 8, 8)
    feats[0] = 1.0  # clip 0 is all ones
    feats[1] = 9.0  # clip 1 is all nines
    rois = torch.zeros(b, 1, t, 5)
    for bi in range(b):
        for ti in range(t):
            rois[bi, 0, ti] = torch.tensor(_roi(0, 0, 64, 64, ti))
    out = pool_player_features(feats, rois, torch.ones(b, 1, t))
    assert torch.allclose(out[0], torch.full((1, 1, t), 1.0))
    assert torch.allclose(out[1], torch.full((1, 1, t), 9.0))


def test_masked_players_produce_zero_features():
    """A player with mask 0 must not leak features from the shared map."""
    feats = torch.ones(1, 2, 3, 8, 8)
    rois = torch.zeros(1, 2, 3, 5)
    for m in range(2):
        for ti in range(3):
            rois[0, m, ti] = torch.tensor(_roi(0, 0, 64, 64, ti))
    masks = torch.ones(1, 2, 3)
    masks[0, 1] = 0.0  # player 1 is never observed
    out = pool_player_features(feats, rois, masks)
    assert out[0, 0].abs().sum() > 0
    assert torch.count_nonzero(out[0, 1]) == 0


def test_input_rois_are_not_mutated():
    """The reference writes the batch offset back into the caller's tensor in place.
    Ours must not -- the same ROI tensor is reused when a clip is scored twice."""
    rois = torch.zeros(1, 1, 2, 5)
    rois[0, 0, 1, 0] = 1.0
    before = rois.clone()
    pool_player_features(torch.zeros(1, 1, 2, 8, 8), rois, torch.ones(1, 1, 2))
    assert torch.equal(rois, before)


slow = pytest.mark.skipif(
    not os.environ.get("MATCHLAB_SLOW_TESTS"),
    reason="downloads real X3D-S weights via torch.hub; set MATCHLAB_SLOW_TESTS=1 to run",
)


@slow
def test_output_shape_is_b_classes_m_t():
    m = ActionHead()
    out = m(torch.zeros(1, 3, 8, 352, 640), torch.zeros(1, 4, 8, 5), torch.ones(1, 4, 8))
    assert out.shape == (1, 9, 4, 8)


@slow
def test_accepts_full_26_slots_at_inference():
    out = ActionHead()(
        torch.zeros(1, 3, 4, 352, 640), torch.zeros(1, 26, 4, 5), torch.ones(1, 26, 4)
    )
    assert out.shape == (1, 9, 26, 4)


@slow
def test_feature_map_is_stride_8():
    """roi_align's spatial_scale=0.125 is only correct if the merged map is 1/8 of
    the input. A backbone change that moved it would silently misplace every ROI."""
    m = ActionHead()
    feats = m.backbone_features(torch.zeros(1, 3, 4, 352, 640))
    assert feats.shape == (1, 192, 4, 44, 80)
