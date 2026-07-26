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
    SlotAttentionEncoderEmbedding,
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


def test_attention_encoder_parameter_count_is_independent_of_slot_count():
    """The structural difference from the flat encoder, and why PAVE-style encoding
    can generalise to a formation it never saw: one shared embedding is applied to
    every slot, so nothing grows with the number of slots."""
    small = SlotAttentionEncoderEmbedding(32, FRAMESPAN, n_slots=13)
    large = SlotAttentionEncoderEmbedding(32, FRAMESPAN, n_slots=26)
    exclude = "slot_embedding"
    n = lambda m: sum(  # noqa: E731
        p.numel() for k, p in m.named_parameters() if exclude not in k
    )
    assert n(small) == n(large)
    # The flat encoder, by contrast, scales with slots.
    assert FlatEncoderEmbedding(32, FRAMESPAN).linear.in_features > FRAMESPAN + 2


def test_attention_encoder_lets_slots_see_each_other():
    """If a slot's output ignored the other slots this would be a per-slot MLP, and
    the whole point is that a pass has a receiver."""
    torch.manual_seed(0)
    enc = SlotAttentionEncoderEmbedding(32, FRAMESPAN).eval()
    a = torch.zeros(1, 1, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    b = a.clone()
    b[0, 0, FEATURES_PER_SLOT : 2 * FEATURES_PER_SLOT] = 5.0  # change slot 1 only
    with torch.no_grad():
        assert not torch.allclose(enc(a), enc(b))


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
        for action, role, frame in seq:
            assert 0 <= action < ACTION_VOCAB and action != EOS_ACTION
            assert 0 <= role < ROLE_VOCAB
            assert 0 <= frame < FRAMESPAN + 2


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
