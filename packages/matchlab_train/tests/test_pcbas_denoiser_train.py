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


def test_effective_batch_is_deliberately_smaller_than_the_reference():
    """We trade batch size for optimiser steps. The reference gets ~2,000 steps per
    epoch from 192,000 windows; we can afford 19,200 windows, so at its effective
    batch of 96 we would take only 200 steps -- and step count, not batch size, is
    what DST is short of (~3,000 vs their ~30,000)."""
    p = Params()
    assert p.batch_size * p.accum_steps == 24


def test_denoiser_hyperparameters_match_the_reference():
    p = Params()
    # warmup_steps is scaled to OUR step count, not copied: the reference's 1000
    # steps is <=1 epoch for it (2,000 optimiser steps in epoch 1) but would be half
    # our 10-epoch run. Scaled to ~1 epoch at our 200 steps/epoch.
    assert (p.lr, p.warmup_steps, p.epochs) == (2.5e-4, 200, 15)
    assert (p.hidden_dim, p.n_heads, p.n_layers, p.dropout) == (512, 8, 6, 0.1)
    assert p.framespan == 750
    # `encoder` and the attention params are asserted by
    # test_attention_defaults_match_the_paper.


def test_optimizer_applies_the_reference_weight_decay():
    """CORRECTED. An earlier version of this test (and the docstring it guarded)
    asserted the reference applies NO weight decay in this stage. It does: 1e-4 on
    every non-bias parameter. The wrong assertion locked in the wrong behaviour."""
    import inspect

    from matchlab_train.experiments.pcbas_denoiser import PCBASDenoiserExperiment

    source = inspect.getsource(PCBASDenoiserExperiment.run)
    assert "weight_decay=p.weight_decay" in source
    assert Params().weight_decay == 1e-4


def test_lr_decay_is_disabled_until_the_step_budget_supports_it():
    """Copying the reference's epoch numbers (3/6/8) was wrong: it takes ~2,000
    optimiser steps per epoch to our ~200, so the same epochs are 10x earlier in
    steps. It annealed us to 2.5e-7 before convergence -- the run flatlined from
    epoch 8 and scored micro-F1 0.048 against 0.119 undecayed."""
    p = Params()
    assert p.lr_decay == 1.0 and p.lr_decay_epochs == ()


def test_label_smoothing_matches_the_reference():
    """0.05 on action and role; deliberately NOT on the frame head."""
    assert Params().label_smoothing == 0.05


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


def test_every_shipped_pcbas_config_parses_and_resolves():
    """A config with a typo'd task name fails only when someone tries to run it,
    typically hours into a session."""
    from pathlib import Path

    from matchlab_train.config import ExperimentConfig
    from matchlab_train.registry import available

    root = Path(__file__).resolve().parents[1] / "src/matchlab_train/experiments"
    configs = sorted(root.glob("pcbas_*.yaml"))
    # action_head (Phase 0 control), action_head_pave (arms A1/A2), denoiser,
    # denoise_infer, score. The count is asserted so a config added without a task
    # name cannot slip through the glob unchecked.
    assert len(configs) == 5
    for path in configs:
        cfg = ExperimentConfig.from_yaml(path)
        assert cfg.task in available(), f"{path.name} -> unknown task {cfg.task}"


def test_collate_one_hot_encodes_the_same_frames_as_the_positional_encoding():
    """The encoder gets frame information twice -- a one-hot channel concatenated
    onto the features, and a sinusoidal positional encoding. They must encode the
    SAME values. The reference one-hots `enc_abs_frame` itself (1-based window-local);
    one-hotting arange(T) instead makes them disagree by one, a quieter cousin of the
    absolute/window-local bug that made DST's first run score 0.035."""
    import numpy as np
    from matchlab_core.pcbas.denoiser import ENCODER_FEATURE_DIM

    frames = torch.arange(1, 7)  # 1-based window-local, as the dataset now emits
    sample = (
        torch.zeros(6, ENCODER_FEATURE_DIM),
        frames,
        *_sample(1, t=6)[2:],
    )
    src, src_frames, _, _, _ = collate([sample], FRAMESPAN)
    one_hot = src[0, :, ENCODER_FEATURE_DIM:]
    assert np.array_equal(one_hot.argmax(-1).numpy(), src_frames[0].numpy())
    assert one_hot[0].argmax() == 1  # NOT 0


def test_attention_defaults_match_the_paper():
    from matchlab_train.experiments.pcbas_denoiser import Params

    p = Params()
    assert p.encoder == "flat"          # control
    assert p.attn_order == "spatial_first"
    assert p.attn_dim == 64             # PAVE Model A
    assert p.attn_layers == 1
    assert p.attn_use_logits is False   # game-state channels only


def test_dst_lr_decay_stays_disabled():
    """Copying PAVE's epoch 3/6/8 decay annealed us to 2.5e-7 before convergence
    and flatlined the run: micro-F1 0.048 vs 0.119. Their epoch 3 is 6,000
    optimiser steps in; ours is 600. Do not re-run this hypothesis.
    """
    from matchlab_train.experiments.pcbas_denoiser import Params

    p = Params()
    assert p.lr_decay == 1.0
    assert p.lr_decay_epochs == ()


def test_fuse_experiment_is_registered():
    assert "pcbas-fuse" in available()


def test_fuse_round_trips_the_denoise_infer_export_format(tmp_path):
    """The fusion driver consumes exactly what pcbas-denoise-infer writes.

    That export is shirt-keyed `[frame, left_to_right, shirt, class_id, score]`, so a
    format drift between the two would only surface when an ensemble is finally
    assembled -- after four models have been trained.
    """
    import json

    from matchlab_train.experiments.pcbas_fuse import load_export

    payload = {
        "keys": ["game_18_H1"],
        "events": {"game_18_H1": [[100, 1, 7, 2, 0.8], [250, 0, 11, 7, 0.4]]},
    }
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload))
    loaded = load_export(str(path))
    assert list(loaded) == ["game_18_H1"]
    first, second = loaded["game_18_H1"]
    assert (first.frame_idx, first.left_to_right, first.shirt_number, first.class_id) == (
        100, 1, 7, 2
    )
    assert first.score == pytest.approx(0.8)
    assert (second.frame_idx, second.left_to_right, second.shirt_number) == (250, 0, 11)


def test_fuse_refuses_a_single_model(tmp_path):
    """A one-model 'ensemble' would only apply the (n/N)^0.5 penalty to itself, which
    silently reports a WORSE score under an ensemble's name."""
    from matchlab_train.config import ExperimentConfig
    from matchlab_train.experiments.pcbas_fuse import PCBASFuseExperiment

    cfg = ExperimentConfig(
        name="fuse-one", task="pcbas-fuse", description="one model",
        output_dir=str(tmp_path), seed=345, params={"exports": ["only.json"]},
    )
    with pytest.raises(ValueError, match="at least 2"):
        PCBASFuseExperiment(cfg).run()


def test_denoise_infer_rebuilds_the_attention_the_checkpoint_was_trained_with():
    """spatial_first and temporal_first share an IDENTICAL state_dict, so a B2
    (temporal_first) checkpoint loads cleanly into a spatial_first model and scores
    the wrong architecture with no error raised. The rebuild must read the attn
    params the trainer saves, not construct at the defaults.
    """
    from matchlab_train.experiments.pcbas_denoise_infer import denoiser_from_state

    p = Params(encoder="attn", attn_order="temporal_first", attn_dim=8, attn_layers=1,
               framespan=FRAMESPAN, hidden_dim=32, n_heads=4, n_layers=1)
    from matchlab_core.pcbas.denoiser import DSTDenoiser

    torch.manual_seed(0)
    trained = DSTDenoiser(
        framespan=FRAMESPAN, hidden_dim=32, n_heads=4, n_enc_layers=1, n_dec_layers=1,
        encoder="attn", attn_order="temporal_first", attn_dim=8, attn_layers=1,
    )
    state = {"model": trained.state_dict(), "epoch": 3, "params": p.model_dump()}
    rebuilt = denoiser_from_state(state, torch.device("cpu"))
    assert rebuilt.attention_branch is not None
    assert rebuilt.attention_branch.order == "temporal_first"
    assert rebuilt.attention_branch.d_p == 8


def test_denoise_infer_rebuild_defaults_to_flat_for_legacy_checkpoints():
    from matchlab_core.pcbas.denoiser import DSTDenoiser
    from matchlab_train.experiments.pcbas_denoise_infer import denoiser_from_state

    torch.manual_seed(0)
    trained = DSTDenoiser(framespan=FRAMESPAN, hidden_dim=32, n_heads=4,
                          n_enc_layers=1, n_dec_layers=1)
    state = {"model": trained.state_dict(), "epoch": 3,
             "params": {"framespan": FRAMESPAN, "hidden_dim": 32, "n_heads": 4,
                        "n_layers": 1}}
    rebuilt = denoiser_from_state(state, torch.device("cpu"))
    assert rebuilt.attention_branch is None
