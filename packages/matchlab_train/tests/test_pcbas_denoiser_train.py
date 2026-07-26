from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.pcbas.denoiser import (
    ACTION_VOCAB,
    ENCODER_FEATURE_DIM,
    OUTPUT_DIM,
)
from matchlab_train.experiments.pcbas_denoiser import Params, collate
from matchlab_train.registry import available

torch = pytest.importorskip("torch")

FRAMESPAN = 20


def _sample(n_events: int, t: int = 6):
    from matchlab_train.datasets.footpass_windows import build_tokens

    events = np.array(
        [[i * 2, i % 26, (i % 8) + 1] for i in range(n_events)], dtype=np.int64
    ).reshape(-1, 3)
    actions, roles, frames = build_tokens(events, FRAMESPAN)
    return (
        torch.zeros(t, ENCODER_FEATURE_DIM),
        torch.arange(t),
        torch.from_numpy(actions),
        torch.from_numpy(roles),
        torch.from_numpy(frames),
    )


def test_experiment_is_registered():
    assert "pcbas-denoiser" in available()


def test_effective_batch_matches_the_reference():
    p = Params()
    assert p.batch_size * p.accum_steps == 96


def test_denoiser_hyperparameters_match_the_reference():
    p = Params()
    assert (p.lr, p.warmup_steps, p.epochs) == (2.5e-4, 1000, 15)
    assert (p.hidden_dim, p.n_heads, p.n_layers, p.dropout) == (512, 8, 6, 0.1)
    assert p.framespan == 750
    assert p.encoder == "flat"  # reference default; "attn" is the PAVE arm


def test_optimizer_applies_no_weight_decay():
    """The reference decays nothing in this stage, unlike the action head which
    decays its Conv/Linear weights. Silently inheriting AdamW's 0.01 default would
    be an unrecorded hyperparameter change."""
    import inspect

    from matchlab_train.experiments.pcbas_denoiser import PCBASDenoiserExperiment

    source = inspect.getsource(PCBASDenoiserExperiment.run)
    assert "weight_decay=0.0" in source


def test_collate_pads_targets_to_the_longest_sequence():
    """Windows contain different numbers of events, so the decoder target is
    ragged. Padding without a mask would train the model to emit padding."""
    batch = [_sample(1), _sample(4)]
    src, src_frames, tgt, tgt_frames, tgt_pad = collate(batch, FRAMESPAN)
    assert src.shape == (2, 6, ENCODER_FEATURE_DIM + FRAMESPAN + 2)
    assert tgt.shape == (2, 6, OUTPUT_DIM + FRAMESPAN + 2)  # 4 events + SOS + EOS
    assert tgt_pad[0].tolist() == [False, False, False, True, True, True]
    assert tgt_pad[1].tolist() == [False] * 6


def test_collate_one_hot_streams_are_exclusive():
    """Each of the three streams must be exactly one-hot per real token; two hot
    bits in the action stream would make the argmax target arbitrary."""
    src, _, tgt, _, tgt_pad = collate([_sample(3)], FRAMESPAN)
    real = ~tgt_pad[0]
    assert tgt[0, real, :ACTION_VOCAB].sum(-1).tolist() == [1.0] * int(real.sum())
    assert tgt[0, real, ACTION_VOCAB:OUTPUT_DIM].sum(-1).tolist() == [1.0] * int(real.sum())
    assert tgt[0, real, OUTPUT_DIM:].sum(-1).tolist() == [1.0] * int(real.sum())


def test_collate_source_carries_the_frame_one_hot():
    src, _, _, _, _ = collate([_sample(1)], FRAMESPAN)
    assert src[0, 3, ENCODER_FEATURE_DIM:].argmax() == 3


def test_training_step_produces_three_finite_losses():
    from matchlab_core.pcbas.denoiser import DSTDenoiser
    from matchlab_train.experiments.pcbas_denoiser import PCBASDenoiserExperiment

    torch.manual_seed(0)
    model = DSTDenoiser(
        framespan=FRAMESPAN, hidden_dim=32, n_heads=4, n_enc_layers=1,
        n_dec_layers=1, dropout=0.0,
    )
    batch = collate([_sample(2), _sample(3)], FRAMESPAN)
    losses = PCBASDenoiserExperiment._step(model, batch, torch.device("cpu"), Params())
    assert set(losses) == {"action", "role", "timestamp", "total"}
    for value in losses.values():
        assert torch.isfinite(value)
    losses["total"].backward()


def test_padded_target_positions_do_not_contribute_to_the_loss():
    """Otherwise the model spends its capacity learning to predict padding."""
    from matchlab_core.pcbas.denoiser import DSTDenoiser
    from matchlab_train.experiments.pcbas_denoiser import PCBASDenoiserExperiment

    torch.manual_seed(0)
    model = DSTDenoiser(
        framespan=FRAMESPAN, hidden_dim=32, n_heads=4, n_enc_layers=1,
        n_dec_layers=1, dropout=0.0,
    ).eval()
    device = torch.device("cpu")
    with torch.no_grad():
        short = PCBASDenoiserExperiment._step(
            model, collate([_sample(2)], FRAMESPAN), device, Params()
        )
        # Same window, batched with a longer one -> more padding, same real tokens.
        padded = PCBASDenoiserExperiment._step(
            model, collate([_sample(2), _sample(2)], FRAMESPAN), device, Params()
        )
    assert float(short["action"]) == pytest.approx(float(padded["action"]), abs=1e-4)
