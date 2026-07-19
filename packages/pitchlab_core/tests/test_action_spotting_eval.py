"""Hand-computed tests for the avg-mAP action-spotting metric (SPO-49).

Every expected number here is worked out by hand from the two documented
conventions in `action_spotting_eval.average_map`:

  * AP = area under the precision envelope, all-points / VOC2010+
    interpolation (the standard detection-mAP definition).
  * A class with **no GT events** is excluded from the per-class mean; a
    class with GT but zero correct predictions contributes AP 0.

If either convention changes, these numbers change -- that is the point.
"""

from __future__ import annotations

import json

from pitchlab_core.action_spotting_eval import (
    average_map,
    evaluate_spotting_run,
    spotting_headline_metrics,
)
from pitchlab_core.artifacts import ArtifactStore
from pitchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
from pitchlab_core.schemas.run import ArtifactName
from pitchlab_core.schemas.spotting import SpottedEvent


def _pred(class_: str, t: float, confidence: float) -> SpottedEvent:
    return SpottedEvent(class_=class_, frame_idx=int(t * 25), t=t, confidence=confidence)


def _gt(class_: str, t: float) -> GroundTruthEvent:
    return GroundTruthEvent(class_=class_, frame_idx=int(t * 25), t=t)


TOL = [1.0]


def test_perfect_match_single_class() -> None:
    gts = [_gt("PASS", 1.0), _gt("PASS", 2.0), _gt("PASS", 3.0)]
    preds = [_pred("PASS", 1.0, 0.9), _pred("PASS", 2.0, 0.8), _pred("PASS", 3.0, 0.7)]

    result = average_map(preds, gts, TOL)

    assert result["kind"] == "action_spotting"
    assert result["avg_map"] == 1.0
    assert result["per_tolerance"]["1.0"]["map"] == 1.0
    assert result["per_tolerance"]["1.0"]["per_class_ap"] == {"PASS": 1.0}
    assert result["counts"] == {
        "predictions": 3,
        "gt_events": 3,
        "tp": 3,
        "fp": 0,
        "fn": 0,
    }


def test_no_predictions_with_gt_is_zero() -> None:
    gts = [_gt("PASS", 1.0)]
    result = average_map([], gts, TOL)

    assert result["avg_map"] == 0.0
    assert result["per_tolerance"]["1.0"]["map"] == 0.0
    assert result["per_tolerance"]["1.0"]["per_class_ap"] == {"PASS": 0.0}
    assert result["counts"]["tp"] == 0
    assert result["counts"]["fp"] == 0
    assert result["counts"]["fn"] == 1


def test_class_with_no_gt_excluded_from_mean() -> None:
    # Class "X" has predictions but no GT anywhere -> excluded from the mean.
    # With no class carrying GT, map is defined as 0.0.
    preds = [_pred("X", 1.0, 0.9), _pred("X", 2.0, 0.8)]
    result = average_map(preds, [], TOL)

    assert result["per_tolerance"]["1.0"]["per_class_ap"] == {}
    assert result["per_tolerance"]["1.0"]["map"] == 0.0
    assert result["avg_map"] == 0.0
    # Both predictions are false positives (no GT of that class to match).
    assert result["counts"]["fp"] == 2
    assert result["counts"]["tp"] == 0
    assert result["counts"]["fn"] == 0


def test_all_empty() -> None:
    result = average_map([], [], TOL)
    assert result["avg_map"] == 0.0
    assert result["per_tolerance"]["1.0"]["per_class_ap"] == {}
    assert result["counts"] == {
        "predictions": 0,
        "gt_events": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def test_confidence_tie_break_is_deterministic() -> None:
    # Two preds tie at conf 0.5: a TP at t=1.0 and a FP at t=5.0. The
    # documented tie-break (t ascending, then original index) orders the
    # t=1.0 TP BEFORE the t=5.0 FP. GT also at t=10.0 matched by a lower-conf
    # pred. Hand-computed AP with this order = 5/6.
    gts = [_gt("PASS", 1.0), _gt("PASS", 10.0)]
    preds = [
        _pred("PASS", 5.0, 0.5),  # FP, ties with the t=1.0 TP
        _pred("PASS", 1.0, 0.5),  # TP
        _pred("PASS", 10.0, 0.3),  # TP
    ]
    result = average_map(preds, gts, TOL)
    ap = result["per_tolerance"]["1.0"]["per_class_ap"]["PASS"]
    assert abs(ap - 5.0 / 6.0) < 1e-9
    assert result["counts"]["tp"] == 2
    assert result["counts"]["fp"] == 1
    assert result["counts"]["fn"] == 0


def test_just_inside_tolerance_is_tp() -> None:
    gts = [_gt("PASS", 10.0)]
    preds = [_pred("PASS", 10.0 + 0.999, 0.9)]  # |diff| = 0.999 <= 1.0 -> TP
    result = average_map(preds, gts, TOL)
    assert result["avg_map"] == 1.0
    assert result["counts"]["tp"] == 1
    assert result["counts"]["fp"] == 0
    assert result["counts"]["fn"] == 0


def test_just_outside_tolerance_is_fp_and_fn() -> None:
    gts = [_gt("PASS", 10.0)]
    preds = [_pred("PASS", 10.0 + 1.001, 0.9)]  # |diff| = 1.001 > 1.0 -> FP
    result = average_map(preds, gts, TOL)
    assert result["avg_map"] == 0.0
    assert result["per_tolerance"]["1.0"]["per_class_ap"] == {"PASS": 0.0}
    assert result["counts"]["tp"] == 0
    assert result["counts"]["fp"] == 1
    assert result["counts"]["fn"] == 1


def test_one_gt_not_double_matched() -> None:
    # Two same-class preds near one GT -> exactly one TP, one FP.
    gts = [_gt("PASS", 10.0)]
    preds = [_pred("PASS", 10.0, 0.9), _pred("PASS", 10.1, 0.8)]
    result = average_map(preds, gts, TOL)
    assert result["counts"]["tp"] == 1
    assert result["counts"]["fp"] == 1
    assert result["counts"]["fn"] == 0
    # FP arrives only after full recall, so the VOC envelope keeps AP at 1.0.
    assert result["per_tolerance"]["1.0"]["per_class_ap"]["PASS"] == 1.0


def test_multi_class_averaging() -> None:
    # Class A: perfect -> AP 1.0. Class B: a high-conf FP precedes the only TP
    # -> AP 0.5. map = mean(1.0, 0.5) = 0.75.
    preds = [
        _pred("A", 1.0, 0.9),  # A TP
        _pred("B", 50.0, 0.9),  # B FP (higher conf, ranked first)
        _pred("B", 1.0, 0.5),  # B TP
    ]
    gts = [_gt("A", 1.0), _gt("B", 1.0)]
    result = average_map(preds, gts, TOL)
    per_class = result["per_tolerance"]["1.0"]["per_class_ap"]
    assert per_class["A"] == 1.0
    assert abs(per_class["B"] - 0.5) < 1e-9
    assert abs(result["per_tolerance"]["1.0"]["map"] - 0.75) < 1e-9
    assert abs(result["avg_map"] - 0.75) < 1e-9


def test_spotting_headline_metrics() -> None:
    gts = [_gt("PASS", 1.0)]
    preds = [_pred("PASS", 1.0, 0.9)]
    result = average_map(preds, gts, TOL)
    heads = spotting_headline_metrics(result)
    assert heads["spotting_map_at_1"] == 1.0


def test_evaluate_spotting_run_happy(tmp_path) -> None:
    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir)
    store.write_json(
        ArtifactName.SPOTTING,
        [_pred("PASS", 1.0, 0.9), _pred("PASS", 2.0, 0.8)],
    )
    event_gt = EventGroundTruth(
        source="test",
        fps=25.0,
        events=[_gt("PASS", 1.0), _gt("PASS", 2.0)],
    )
    result = evaluate_spotting_run(run_dir, event_gt)
    assert result is not None
    assert result["avg_map"] == 1.0
    assert result["counts"]["tp"] == 2
    assert result["counts"]["predictions"] == 2
    assert result["counts"]["gt_events"] == 2


def test_evaluate_spotting_run_absent_artifact_returns_none(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    event_gt = EventGroundTruth(source="test", fps=25.0, events=[_gt("PASS", 1.0)])
    assert evaluate_spotting_run(run_dir, event_gt) is None


def test_evaluate_spotting_run_reads_written_json_shape(tmp_path) -> None:
    # Guards the on-disk contract: spotting.json is a JSON array of objects
    # with the literal "class" key, and evaluate_spotting_run reads it back.
    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir)
    store.write_json(ArtifactName.SPOTTING, [_pred("PASS", 1.0, 0.9)])
    on_disk = json.loads((run_dir / "spotting.json").read_text())
    assert on_disk[0]["class"] == "PASS"
    event_gt = EventGroundTruth(source="test", fps=25.0, events=[_gt("PASS", 1.0)])
    result = evaluate_spotting_run(run_dir, event_gt)
    assert result["avg_map"] == 1.0
