"""THE metric gate: our scorer must reproduce the FOOTPASS reference exactly.

The cheapest correctness check in the whole programme -- no video, no GPU, no
training. The reference ships its own VAL ground truth and the final event lists its
two published arms produced. If our matching rule differs from theirs by even a
tie-break, every number we later report about a model would actually be measuring
our metric bug.

Targets were produced by running the reference's own `evaluation.py` unmodified on
its own artifacts (delta=12, conf_thresh=0.15) on 2026-07-27:

    uv run python evaluation.py --predictions_file playbyplay_PRED/playbyplay_TAAD_val.json

Exact integer TP/FP/GT counts are asserted rather than F1 alone: two different
matching rules can land on the same rounded F1, but not on the same 8x3 count table.
"""

from __future__ import annotations

import pytest
from matchlab_core.pcbas.eval import score_halves
from matchlab_train.datasets.footpass_pcbas import load_playbyplay
from matchlab_train.datasets.paths import reference_root

FOOTPASS = reference_root("FOOTPASS")
requires_reference = pytest.mark.skipif(
    FOOTPASS is None or not (FOOTPASS / "playbyplay_GT" / "playbyplay_val.json").is_file(),
    reason="FOOTPASS reference clone not present at data/reference/FOOTPASS",
)

# class_id -> (TP, FP, GT), from the reference's aggregated per-class table.
TAAD_PER_CLASS = {
    1: (1556, 3595, 2470),  # drive
    2: (1982, 1901, 3059),  # pass
    3: (66, 385, 111),  # cross
    4: (43, 188, 97),  # throw-in
    5: (43, 261, 67),  # shot
    6: (86, 926, 162),  # header
    7: (9, 578, 26),  # tackle
    8: (37, 920, 78),  # block
}
DST_PER_CLASS = {
    1: (1763, 635, 2470),
    2: (2273, 662, 3059),
    3: (64, 35, 111),
    4: (65, 39, 97),
    5: (40, 26, 67),
    6: (45, 100, 162),
    7: (1, 6, 26),
    8: (16, 36, 78),
}


def _score(pred_name: str):
    gt = load_playbyplay(FOOTPASS / "playbyplay_GT" / "playbyplay_val.json", "gt")
    pred = load_playbyplay(FOOTPASS / "playbyplay_PRED" / pred_name, "pred")
    return score_halves(gt, pred, delta=12, conf_thresh=0.15, identity="shirt")


@requires_reference
def test_val_ground_truth_matches_the_published_event_count():
    gt = load_playbyplay(FOOTPASS / "playbyplay_GT" / "playbyplay_val.json", "gt")
    assert sorted(gt) == [
        "game_18_H1",
        "game_18_H2",
        "game_24_H1",
        "game_24_H2",
        "game_47_H1",
        "game_47_H2",
    ]
    assert sum(len(v) for v in gt.values()) == 6070  # README


@requires_reference
@pytest.mark.parametrize(
    "pred_name,per_class,tp,fp,micro,macro,tp_nobb",
    [
        ("playbyplay_TAAD_val.json", TAAD_PER_CLASS, 3822, 8754, 0.4100, 0.2445, 33),
        ("playbyplay_DST_val.json", DST_PER_CLASS, 4267, 1539, 0.7186, 0.4926, 390),
    ],
    ids=["TAAD", "TAAD+DST"],
)
def test_reproduces_reference_counts(
    pred_name, per_class, tp, fp, micro, macro, tp_nobb
):
    r = _score(pred_name)

    measured = {c: (m.tp, m.fp, m.n_gt) for c, m in r.per_class.items()}
    assert measured == per_class

    assert (r.tp, r.fp, r.n_gt) == (tp, fp, 6070)
    assert r.micro_f1 == pytest.approx(micro, abs=5e-5)
    # macro-F1 is our addition, not the reference's; these targets were computed
    # independently from the reference's own per-class precision/recall table.
    assert r.macro_f1 == pytest.approx(macro, abs=5e-5)

    # The observability split the two-stage argument rests on: DST recovers 390 of
    # the 1,062 off-screen actions, TAAD only 33.
    assert r.tp_without_bbox == tp_nobb
    assert r.gt_without_bbox == 1062


@requires_reference
def test_dst_beats_taad_on_every_aggregate():
    taad, dst = _score("playbyplay_TAAD_val.json"), _score("playbyplay_DST_val.json")
    assert dst.micro_f1 > taad.micro_f1
    assert dst.macro_f1 > taad.macro_f1


@requires_reference
def test_macro_f1_is_far_below_micro_on_val():
    """Recorded so nobody quotes micro as if it were the per-class picture: tackle
    recall collapses to 0.038 in the arm with micro-F1 0.72."""
    dst = _score("playbyplay_DST_val.json")
    assert dst.macro_f1 < dst.micro_f1 - 0.15
    assert dst.per_class[7].recall < 0.05
