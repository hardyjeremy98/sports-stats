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


def test_temporal_transformer_shape():
    from matchlab_core.pcbas.action_head import TemporalTransformer

    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10)
    x = torch.randn(3, 8, 10)
    mask = torch.ones(3, 10)
    assert block(x, mask).shape == (3, 12, 10)


def test_temporal_transformer_sees_the_whole_clip():
    """The point of the change: a Conv1d(k=3) cannot, and this must.

    Perturbing frame 0 has to alter the output at frame 9. Under the old 3-frame
    receptive field it provably could not.
    """
    from matchlab_core.pcbas.action_head import TemporalTransformer

    torch.manual_seed(0)
    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10).eval()
    x = torch.randn(1, 8, 10)
    mask = torch.ones(1, 10)
    a = block(x, mask)
    x2 = x.clone()
    x2[0, :, 0] += 5.0
    b = block(x2, mask)
    assert not torch.allclose(a[0, :, 9], b[0, :, 9], atol=1e-6)


def test_temporal_transformer_ignores_unobserved_frames():
    """Pooled features are zeroed where the mask is 0 (~60% of cells).

    Attention over those zeros would be attention over frames that were never
    observed. Changing a masked frame's input must not change any output.
    """
    from matchlab_core.pcbas.action_head import TemporalTransformer

    torch.manual_seed(0)
    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10).eval()
    x = torch.randn(1, 8, 10)
    mask = torch.ones(1, 10)
    mask[0, 4:7] = 0.0
    a = block(x, mask)
    x2 = x.clone()
    x2[0, :, 4:7] += 9.0
    b = block(x2, mask)
    torch.testing.assert_close(a, b)


def test_temporal_transformer_survives_a_fully_absent_player():
    """An all-masked sequence makes PyTorch attention return NaN.

    A player observed in zero frames is ordinary -- players leave frame. The NaN
    would propagate through the whole batch's gradient and present as an
    unexplained training collapse, not as a masking bug.
    """
    from matchlab_core.pcbas.action_head import TemporalTransformer

    block = TemporalTransformer(in_channels=8, d_model=16, out_channels=12,
                                n_layers=1, n_heads=2, ff_dim=32, max_frames=10).eval()
    x = torch.zeros(2, 8, 10)
    mask = torch.ones(2, 10)
    mask[1] = 0.0
    out = block(x, mask)
    assert torch.isfinite(out).all()


def test_action_head_from_checkpoint_honours_the_recorded_temporal_kind(tmp_path):
    """A transformer-arm checkpoint must not be loaded as a conv-arm model.

    Both inference paths built `ActionHead()` at its "conv" default and then called
    load_state_dict, which raises on every temporal_transformer.* key. That made an
    A1 checkpoint unscoreable -- the arm trained for 4.5 h and could not be measured.
    The checkpoint records `temporal` in its params; the loader reads it.
    """
    from matchlab_core.pcbas.action_head import action_head_from_checkpoint

    for kind in ("conv", "transformer"):
        model = ActionHead(pretrained=False, temporal=kind)
        path = tmp_path / f"{kind}.pt"
        torch.save({"model": model.state_dict(), "epoch": 7, "params": {"temporal": kind}}, path)
        loaded = action_head_from_checkpoint(path, pretrained=False)
        assert loaded.temporal_kind == kind
        for (ka, va), (kb, vb) in zip(
            model.state_dict().items(), loaded.state_dict().items(), strict=True
        ):
            assert ka == kb
            torch.testing.assert_close(va, vb)


def test_action_head_from_checkpoint_defaults_to_conv_for_older_checkpoints(tmp_path):
    """Checkpoints written before the selector existed carry no `temporal` key."""
    from matchlab_core.pcbas.action_head import action_head_from_checkpoint

    model = ActionHead(pretrained=False)
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "epoch": 3}, path)
    assert action_head_from_checkpoint(path, pretrained=False).temporal_kind == "conv"
