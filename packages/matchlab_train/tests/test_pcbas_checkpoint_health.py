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


def test_clip_events_pads_unused_slots_as_background():
    """The sampler's M is 5, the decoder expects 26. Padding must be CONFIDENT
    background, not zeros -- zeros would tie with every class and emit phantom
    events in slots that hold no player at all."""
    import numpy as np
    from matchlab_train.experiments.pcbas_checkpoint_health import clip_events

    logits = np.zeros((9, 3, 40), dtype=np.float32)
    logits[0] = 5.0
    logits[2, 1, 20] = 9.0  # player 1 passes at frame 20
    events = clip_events(logits, None)
    assert len(events) == 1
    assert (events[0].slot, events[0].class_id, events[0].frame_idx) == (1, 2, 20)


def test_label_events_reads_sharp_labels():
    import numpy as np
    from matchlab_train.experiments.pcbas_checkpoint_health import label_events

    sharp = np.zeros((3, 10), dtype=np.int64)
    sharp[2, 7] = 5
    events = label_events(sharp)
    assert len(events) == 1
    assert (events[0].slot, events[0].class_id, events[0].frame_idx) == (2, 5, 7)


def test_label_events_ignores_background():
    import numpy as np
    from matchlab_train.experiments.pcbas_checkpoint_health import label_events

    assert label_events(np.zeros((3, 10), dtype=np.int64)) == []
