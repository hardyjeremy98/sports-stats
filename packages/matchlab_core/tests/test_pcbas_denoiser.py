"""Denoiser tests.

The model is small enough to instantiate in tests, so these check the things that
would otherwise only surface as a bad F1 after hours of training: the token
vocabulary, that the decoder cannot see its own future, that padding is honoured,
and that autoregressive decoding actually stops at EOS.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from matchlab_core.pcbas.denoiser import (  # noqa: E402
    ABSENT_FILL,
    ACTION_VOCAB,
    ENCODER_FEATURE_DIM,
    EOS_ACTION,
    FEATURES_PER_SLOT,
    NEUTRAL_ROLE,
    OUTPUT_DIM,
    ROLE_VOCAB,
    SOS_ACTION,
    DSTDenoiser,
    FlatEncoderEmbedding,
    sinusoidal_positional_encoding,
)
from matchlab_core.pcbas.schema import N_CLASSES, N_SLOTS  # noqa: E402

FRAMESPAN = 40  # small stand-in for the real 750, so tests stay fast


def _model(encoder="flat", **kw):
    return DSTDenoiser(
        framespan=FRAMESPAN,
        hidden_dim=32,
        n_heads=4,
        n_enc_layers=1,
        n_dec_layers=1,
        dropout=0.0,
        encoder=encoder,
        **kw,
    )


def _src(b=2, t=6):
    return torch.randn(b, t, ENCODER_FEATURE_DIM + FRAMESPAN + 2)


def _tgt(b=2, n=4):
    return torch.randn(b, n, OUTPUT_DIM + FRAMESPAN + 2)


# --- vocabulary -------------------------------------------------------------------


def test_vocabulary_matches_the_reference():
    """Frozen by every checkpoint trained against it. Action has no background class:
    background is expressed by NOT emitting a token."""
    assert ACTION_VOCAB == 10
    assert (SOS_ACTION, EOS_ACTION) == (8, 9)
    assert ROLE_VOCAB == N_SLOTS + 1 == 27
    assert NEUTRAL_ROLE == 26
    assert OUTPUT_DIM == 37


def test_encoder_feature_width_is_26_slots_by_14():
    assert FEATURES_PER_SLOT == 5 + N_CLASSES == 14
    assert ENCODER_FEATURE_DIM == N_SLOTS * FEATURES_PER_SLOT == 364


def test_absent_fill_is_outside_the_coordinate_range():
    """0 is an ordinary normalised pitch coordinate. Filling absent slots with it
    would make 'at the origin' and 'not there' indistinguishable."""
    assert ABSENT_FILL < -1.0


def test_flat_embedding_input_width_is_1116_at_the_real_framespan():
    embedding = FlatEncoderEmbedding(512)
    assert embedding.linear.in_features == 1116


# --- shapes -----------------------------------------------------------------------


@pytest.mark.parametrize("encoder", ["flat", "attn"])
def test_forward_shape(encoder):
    model = _model(encoder)
    out = model(
        _src(), _tgt(), torch.zeros(2, 6, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)
    )
    assert out.shape == (2, OUTPUT_DIM + FRAMESPAN + 2, 4)


def test_positional_encoding_shape_and_range():
    pe = sinusoidal_positional_encoding(torch.arange(6).unsqueeze(0), 32)
    assert pe.shape == (1, 6, 32)
    assert pe.abs().max() <= 1.0


def test_positional_encoding_distinguishes_frames():
    pe = sinusoidal_positional_encoding(torch.tensor([[0, 1, 500]]), 32)
    assert not torch.allclose(pe[0, 0], pe[0, 1])
    assert not torch.allclose(pe[0, 1], pe[0, 2])


# --- the attention encoder --------------------------------------------------------


def test_spatial_attention_lets_slots_see_each_other():
    """If a slot's token ignored the other slots this would be a per-slot MLP, and
    the whole point is that a pass has a receiver.

    Asserted on the SPATIAL STAGE's output rather than the branch's, because the
    branch mean-pools over slots: perturbing slot 1 moves that mean whether or not
    any cross-slot mixing happened, so a branch-level assertion would pass on a
    per-slot MLP. Slot 0's own token responding to slot 1 is the real property.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    torch.manual_seed(0)
    branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                      n_heads=2, n_layers=1).eval()
    captured = []
    branch.spatial.register_forward_hook(lambda _m, _i, o: captured.append(o.detach().clone()))

    a = torch.zeros(1, 1, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    b = a.clone()
    b[0, 0, FEATURES_PER_SLOT : 2 * FEATURES_PER_SLOT] = 5.0  # change slot 1 only
    with torch.no_grad():
        branch(a)
        branch(b)
    assert not torch.allclose(captured[0][:, 0], captured[1][:, 0])


def test_flat_encoder_scales_with_slots_where_the_attention_branch_shares_weights():
    """The structural contrast that motivated per-player attention: the flat encoder
    dedicates a fixed span of its input to each slot, so it must learn each slot's
    role separately. The attention's slot projection is shared across all of them.

    Only `slot_proj` and the two encoder stacks are slot-independent -- `out_proj`
    deliberately is not, because the concat step re-introduces `n_slots * 9` TAAD
    logit channels.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    shared = lambda m: sum(  # noqa: E731
        p.numel()
        for k, p in m.named_parameters()
        if k.startswith(("slot_proj", "spatial", "temporal"))
    )
    small = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                     n_heads=2, n_layers=1, n_slots=13)
    large = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                     n_heads=2, n_layers=1, n_slots=26)
    assert shared(small) == shared(large)
    assert FlatEncoderEmbedding(32, FRAMESPAN).linear.in_features > FRAMESPAN + 2


# --- masking ----------------------------------------------------------------------


def test_decoder_cannot_see_its_own_future():
    """Causal masking. Without it the model trains beautifully and generates
    nonsense, because at inference the future tokens do not exist."""
    torch.manual_seed(0)
    model = _model().eval()
    src, src_frames = _src(1, 6), torch.zeros(1, 6, dtype=torch.long)
    tgt = _tgt(1, 5)
    tgt_frames = torch.zeros(1, 5, dtype=torch.long)

    with torch.no_grad():
        base = model(src, tgt, src_frames, tgt_frames)
        altered = tgt.clone()
        altered[0, 4] = torch.randn(OUTPUT_DIM + FRAMESPAN + 2)  # change the LAST token
        changed = model(src, altered, src_frames, tgt_frames)

    # Positions 0..3 precede the altered token and must be untouched.
    assert torch.allclose(base[0, :, :4], changed[0, :, :4], atol=1e-5)
    assert not torch.allclose(base[0, :, 4], changed[0, :, 4], atol=1e-5)


def test_source_padding_mask_excludes_padded_frames():
    torch.manual_seed(0)
    model = _model().eval()
    src = _src(1, 6)
    frames = torch.zeros(1, 6, dtype=torch.long)
    tgt, tgt_frames = _tgt(1, 3), torch.zeros(1, 3, dtype=torch.long)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[0, 4:] = True  # last two frames are padding

    with torch.no_grad():
        base = model(src, tgt, frames, tgt_frames, mask)
        src2 = src.clone()
        src2[0, 4:] = torch.randn(2, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
        other = model(src2, tgt, frames, tgt_frames, mask)
    assert torch.allclose(base, other, atol=1e-5)


def test_unmasked_source_changes_do_reach_the_output():
    """Guards the test above: if the model ignored src entirely, that test would
    pass for the wrong reason."""
    torch.manual_seed(0)
    model = _model().eval()
    src = _src(1, 6)
    frames = torch.zeros(1, 6, dtype=torch.long)
    tgt, tgt_frames = _tgt(1, 3), torch.zeros(1, 3, dtype=torch.long)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[0, 4:] = True

    with torch.no_grad():
        base = model(src, tgt, frames, tgt_frames, mask)
        src2 = src.clone()
        src2[0, 0] = torch.randn(ENCODER_FEATURE_DIM + FRAMESPAN + 2)  # UNmasked
        other = model(src2, tgt, frames, tgt_frames, mask)
    assert not torch.allclose(base, other, atol=1e-5)


# --- autoregressive decoding ------------------------------------------------------


def test_generate_returns_one_list_per_batch_item():
    torch.manual_seed(0)
    model = _model().eval()
    out = model.generate(_src(3, 6), torch.zeros(3, 6, dtype=torch.long), max_events=5)
    assert len(out) == 3
    assert all(isinstance(seq, list) for seq in out)


def test_generate_respects_max_events():
    torch.manual_seed(0)
    model = _model().eval()
    out = model.generate(_src(1, 6), torch.zeros(1, 6, dtype=torch.long), max_events=4)
    assert len(out[0]) <= 4


def test_generate_stops_at_eos():
    """The reference decodes a fixed number of steps regardless, so everything past
    the real end of the event list is emitted as a prediction."""
    torch.manual_seed(0)
    model = _model().eval()
    with torch.no_grad():
        # Force EOS: bias the action head so class 9 always wins.
        model.token_projection.bias.zero_()
        model.token_projection.bias[EOS_ACTION] = 1e4
        model.token_projection.weight.zero_()
    out = model.generate(_src(2, 6), torch.zeros(2, 6, dtype=torch.long), max_events=20)
    assert out == [[], []]


def test_generated_tokens_are_in_vocabulary():
    torch.manual_seed(0)
    model = _model().eval()
    for seq in model.generate(_src(2, 6), torch.zeros(2, 6, dtype=torch.long), max_events=6):
        for action, role, frame, score in seq:
            assert 0 <= action < ACTION_VOCAB and action != EOS_ACTION
            assert 0 <= role < ROLE_VOCAB
            assert 0 <= frame < FRAMESPAN + 2
            assert 0.0 <= score <= 1.0


def test_generate_returns_a_score_the_metric_can_use():
    """A decode returning only argmax indices is unscoreable: the metric needs a
    confidence to order greedy matching and to apply its threshold."""
    torch.manual_seed(0)
    model = _model().eval()
    seqs = model.generate(_src(1, 6), torch.zeros(1, 6, dtype=torch.long), max_events=4)
    assert all(len(tok) == 4 for tok in seqs[0])


@pytest.mark.parametrize("encoder", ["flat", "attn"])
def test_both_encoders_are_trainable_end_to_end(encoder):
    """A backward pass through the whole graph -- catches a detached branch, which
    would otherwise show up only as a model that never improves."""
    torch.manual_seed(0)
    model = _model(encoder)
    out = model(
        _src(), _tgt(), torch.zeros(2, 6, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long)
    )
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


# --- the PAVE per-player attention branch -----------------------------------------


def test_attn_arm_reduces_to_flat_when_the_branch_is_zeroed():
    """The ablation must be a pure ADDITION, not a substitution.

    If this fails, an attn-vs-flat comparison is measuring two changes at once
    and no result from it is attributable.
    """
    torch.manual_seed(0)
    flat = _model(encoder="flat")
    torch.manual_seed(0)
    attn = _model(encoder="attn")
    attn.attention_branch.out_proj.weight.data.zero_()
    attn.attention_branch.out_proj.bias.data.zero_()
    flat.eval()
    attn.eval()

    src = torch.randn(2, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    frames = torch.arange(1, FRAMESPAN + 1).expand(2, FRAMESPAN)
    torch.testing.assert_close(
        flat.encode(src, frames), attn.encode(src, frames), rtol=0, atol=0
    )


def test_attention_uses_game_state_channels_only_by_default():
    """PAVE measured that EXCLUDING the TAAD logits from the attention wins.

    The logits re-enter at the concat step, so they still reach the encoder --
    they just do not drive the cross-player attention.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    torch.manual_seed(0)
    branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                      n_heads=2, n_layers=1).eval()
    assert branch.slot_proj.in_features == 5

    src = torch.randn(1, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    out = branch(src)
    assert out.shape == (1, FRAMESPAN, 32)


def test_attention_orderings_differ():
    """Spatial-first vs temporal-first is PAVE's only internal ablation of this
    block (+1.87% macro-F1). If the two orderings are identical, our module is
    not doing what theirs does.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    src = torch.randn(1, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    outs = {}
    for order in ("spatial_first", "temporal_first", "parallel"):
        torch.manual_seed(0)
        branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                          n_heads=2, n_layers=1, order=order).eval()
        outs[order] = branch(src)
    assert not torch.allclose(outs["spatial_first"], outs["temporal_first"])
    assert not torch.allclose(outs["spatial_first"], outs["parallel"])


def test_absent_slots_reach_the_projection_unchanged():
    """ABSENT_FILL = -15.0 IS the signal -- deliberately out of range so the model
    can learn 'absent', where 0 is an ordinary normalised coordinate. The reshape
    into per-slot tokens must not silently zero it.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                      n_heads=2, n_layers=1).eval()
    src = torch.zeros(1, FRAMESPAN, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    src[..., :ENCODER_FEATURE_DIM] = ABSENT_FILL
    captured = {}

    # A forward hook that RETURNS a value replaces the module's output, and
    # `dict.setdefault` returns one -- as a lambda this silently substituted
    # slot_proj's 5-wide input for its 8-wide output. Must return None.
    def _capture(_module, inputs, _output) -> None:
        captured.setdefault("x", inputs[0])

    branch.slot_proj.register_forward_hook(_capture)
    branch(src)
    assert torch.equal(
        captured["x"], torch.full_like(captured["x"], ABSENT_FILL)
    )


def test_temporal_attention_uses_window_local_one_based_frames():
    """The convention whose violation cost 3.4x. `build_tokens` encodes window
    frame f as f+1, so the positional encoding here must be 1-based and
    window-local -- never absolute video frames.
    """
    from matchlab_core.pcbas.denoiser import PerPlayerAttentionBranch

    branch = PerPlayerAttentionBranch(hidden_dim=32, framespan=FRAMESPAN, d_p=8,
                                      n_heads=2, n_layers=1)
    frames = branch.temporal_frames(4, torch.device("cpu"))
    assert frames.tolist() == [1, 2, 3, 4]
