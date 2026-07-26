from __future__ import annotations

import pytest
from matchlab_core.pcbas.schema import CLASS_NAMES
from matchlab_train.datasets.footpass_clips import ClipAnchor
from matchlab_train.experiments.pcbas_action_head import (
    Params,
    _param_groups,
    class_weights,
    masked_weighted_ce,
)
from matchlab_train.registry import available

torch = pytest.importorskip("torch")


def test_experiment_is_registered():
    assert "pcbas-action-head" in available()


def test_effective_batch_matches_the_reference():
    """The reference trains at 6 x 8 = 48. This machine cannot fit batch 2, so the
    micro-batch drops to 1 and accumulation must rise to keep the SAME effective
    batch -- otherwise the deviation is a hyperparameter change, not a memory
    workaround, and a reproduction miss becomes unattributable."""
    p = Params()
    assert p.batch_size * p.accum_steps == 48


def test_class_weights_match_the_reference_vector():
    """The reference's exact weighting. Not inverse-frequency: class balance is the
    sampler's job, and a second frequency correction in the loss double-counts it."""
    w = class_weights(0.05, 0.95)
    assert w.tolist() == pytest.approx([0.05] + [0.95] * 8)
    assert len(w) == len(CLASS_NAMES)


def test_masked_cells_contribute_no_loss():
    """A masked cell's pooled feature is forced to exactly zero. Including it would
    train the classifier to map the zero vector to background, on ~60% of cells."""
    b, m, t = 1, 2, 4
    logits = torch.zeros(b, 9, m, t)
    logits[:, 5] = 10.0  # confidently, wrongly predicts class 5 everywhere
    dilated = torch.zeros(b, m, t, dtype=torch.long)
    weights = class_weights(0.05, 0.95)

    all_masked = masked_weighted_ce(logits, dilated, torch.zeros(b, m, t), weights)
    none_masked = masked_weighted_ce(logits, dilated, torch.ones(b, m, t), weights)
    assert float(all_masked) == 0.0
    assert float(none_masked) > 0.0


def test_loss_denominator_counts_every_cell_not_just_observed_ones():
    """The reference divides by the full cell count, so the effective learning rate
    scales with observability. Reproduced deliberately -- normalising by the observed
    count instead would silently change the schedule."""
    b, m, t = 1, 2, 4
    logits = torch.zeros(b, 9, m, t)
    logits[:, 5] = 10.0
    dilated = torch.zeros(b, m, t, dtype=torch.long)
    weights = class_weights(0.05, 0.95)

    half = torch.zeros(b, m, t)
    half[0, 0] = 1.0  # one of the two players observed
    full = torch.ones(b, m, t)
    assert float(masked_weighted_ce(logits, dilated, half, weights)) == pytest.approx(
        float(masked_weighted_ce(logits, dilated, full, weights)) / 2
    )


def test_background_is_weighted_far_below_actions():
    """Background is ~99% of cells. Weighting it equally would drown the actions."""
    b, m, t = 1, 1, 2
    logits = torch.zeros(b, 9, m, t)
    weights = class_weights(0.05, 0.95)
    masks = torch.ones(b, m, t)

    bg = masked_weighted_ce(logits, torch.zeros(b, m, t, dtype=torch.long), masks, weights)
    action = masked_weighted_ce(
        logits, torch.full((b, m, t), 2, dtype=torch.long), masks, weights
    )
    assert float(action) == pytest.approx(float(bg) * (0.95 / 0.05))


def test_param_groups_split_backbone_from_head_and_exclude_norms_from_decay():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv3d(3, 4, 1)
            self.block2 = torch.nn.BatchNorm3d(4)
            self.classifier = torch.nn.Linear(4, 9)
            self.temporal_bn = torch.nn.BatchNorm1d(4)

    groups = {g["name"]: g for g in _param_groups(Fake(), Params())}
    assert {g["name"] for g in groups.values()} == {
        "backbone_decay",
        "backbone_nodecay",
        "head_decay",
        "head_nodecay",
    }
    assert groups["backbone_decay"]["lr"] == 5e-5
    assert groups["head_decay"]["lr"] == 1e-3
    assert groups["backbone_nodecay"]["weight_decay"] == 0.0
    assert groups["head_nodecay"]["weight_decay"] == 0.0
    # The Conv3d weight decays; its bias and every norm parameter do not.
    assert len(groups["backbone_decay"]["params"]) == 1
    assert len(groups["head_decay"]["params"]) == 1


def test_anchor_is_json_round_trippable(tmp_path):
    from matchlab_train.datasets.footpass_clips import load_anchors, save_anchors

    anchors = [ClipAnchor("game_1_H1", 50, 10, 2, 0, 999)]
    save_anchors(anchors, tmp_path / "a.json")
    assert load_anchors(tmp_path / "a.json") == anchors
