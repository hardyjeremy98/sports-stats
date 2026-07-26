from __future__ import annotations

from collections import Counter

from matchlab_train.experiments.pcbas_checkpoint_health import summarise
from matchlab_train.registry import available


def test_experiment_is_registered():
    assert "pcbas-checkpoint-health" in available()


def test_all_background_is_flagged_as_collapsed():
    """The exact failure this exists for: background is ~99% of cells and weighted
    0.05, so a model that predicts it everywhere reaches a low loss and zero F1.
    Nothing in the training log distinguishes that from real progress."""
    s = summarise(Counter({"background": 1000}), 0, 40)
    assert s["collapsed_to_background"] is True
    assert s["action_fraction"] == 0.0
    assert len(s["classes_never_emitted"]) == 8


def test_a_learning_model_is_not_flagged():
    s = summarise(Counter({"background": 900, "pass": 60, "drive": 40}), 15, 40)
    assert s["collapsed_to_background"] is False
    assert s["background_fraction"] == 0.9
    assert s["classes_emitted"] == ["drive", "pass"]
    assert "tackle" in s["classes_never_emitted"]
    assert s["anchor_class_recovered"] == 15


def test_empty_counts_do_not_divide_by_zero():
    s = summarise(Counter(), 0, 0)
    assert s["background_fraction"] == 0.0
    assert s["cells"] == 0
