"""avg-mAP action-spotting metric (SPO-49): score timestamped event
predictions against timestamped event ground truth.

This is the ONE unit-tested number the action-spotting feature is judged by,
so the two conventions that fix the number are stated up front and pinned by
hand-computed tests in `tests/test_action_spotting_eval.py`:

**AP definition.** For each (class, tolerance), predictions of that class are
ranked by descending confidence and walked in that order; each is matched to
the nearest still-unmatched GT event of the same class within +/- tolerance
seconds (a true positive that consumes that GT), otherwise it is a false
positive. From the resulting TP/FP sequence we build the precision-recall
curve and take AP as the **area under the precision envelope, all-points /
VOC2010+ interpolation** -- the standard detection-mAP definition (precision
made monotonically non-increasing from the right, then integrated over
recall, with recall running to num_gt so unrecalled GT depresses AP). This is
`_average_precision` below.

**Zero-GT-class rule.** The per-tolerance mAP is the mean of per-class AP over
**only the classes that have at least one GT event**. A class that appears in
the predictions but has no GT anywhere is a pure-false-positive class: it is
**excluded from the mean** (its predictions still count as false positives in
`counts`, but it contributes no AP term). A class that has GT but zero correct
predictions contributes **AP 0** to the mean. When no class has any GT, mAP is
defined as **0.0**.

**avg-mAP** is the mean of the per-tolerance mAP over `tolerances_s`; for the
ball-action single tolerance `[1.0]`, avg-mAP == mAP@1.

**Tie-break.** Predictions that tie on confidence are ordered by ascending
`t`, then by their original index in the input -- fully deterministic.

**Tolerance boundary.** A prediction at exactly +/- tolerance seconds from a
GT event matches (the window is inclusive, `abs(dt) <= tolerance`).

Pure and I/O-free (stdlib only): matches on `t` seconds so the metric is
independent of frame rate. `evaluate_spotting_run` is the thin file-loading
wrapper used by the server.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from matchlab_core.event_gt import EventGroundTruth, GroundTruthEvent
from matchlab_core.schemas.spotting import SpottedEvent


def _average_precision(tp_flags: Sequence[bool], num_gt: int) -> float:
    """All-points (VOC2010+) AP from a confidence-ranked TP/FP sequence.

    `tp_flags[i]` is True iff the i-th prediction (in descending-confidence
    order) is a true positive. `num_gt` is the number of GT events for the
    class; recall is TP/num_gt, so GT that no prediction recalls holds AP
    below 1. Caller guarantees `num_gt > 0` (a zero-GT class is excluded from
    the mean, never scored here). With no predictions AP is 0.0.
    """
    if num_gt <= 0:
        raise ValueError("AP is undefined for a class with no ground truth")
    if not tp_flags:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    cum_tp = 0
    cum_fp = 0
    for is_tp in tp_flags:
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / num_gt)

    # VOC all-points: pad the curve, make precision monotonically
    # non-increasing from the right, then sum the area over recall steps.
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _match_class(
    preds: list[tuple[float, float, int]],
    gt_times: list[float],
    tolerance: float,
) -> list[bool]:
    """Greedy nearest-unmatched matching for one class at one tolerance.

    `preds` is a list of (confidence, t, original_index); `gt_times` the GT
    event times for the class. Returns TP flags in descending-confidence
    order (tie-break: ascending t, then original index). Each prediction
    consumes the nearest still-unmatched GT within `abs(dt) <= tolerance`;
    equidistant GT are broken by earliest t (list order after sorting).
    """
    order = sorted(preds, key=lambda p: (-p[0], p[1], p[2]))
    gts_sorted = sorted(gt_times)
    matched = [False] * len(gts_sorted)
    flags: list[bool] = []
    for _conf, t, _idx in order:
        best_j = -1
        best_dt = tolerance
        for j, gt_t in enumerate(gts_sorted):
            if matched[j]:
                continue
            dt = abs(gt_t - t)
            if dt <= best_dt:
                # Strictly-closer wins; equidistant keeps the earlier GT
                # (gts_sorted is ascending, so the first seen at this dt).
                if best_j == -1 or dt < best_dt:
                    best_dt = dt
                    best_j = j
        if best_j >= 0:
            matched[best_j] = True
            flags.append(True)
        else:
            flags.append(False)
    return flags


def _class_of(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("class_", item["class"])
    return item.class_


def _t_of(item: Any) -> float:
    return item["t"] if isinstance(item, dict) else item.t


def _conf_of(item: Any) -> float:
    if isinstance(item, dict):
        return item.get("confidence", 0.0)
    return item.confidence


def average_map(
    predictions: Iterable[Any],
    ground_truth: Iterable[Any],
    tolerances_s: Sequence[float],
) -> dict[str, Any]:
    """avg-mAP of `predictions` against `ground_truth` at each tolerance.

    Accepts `SpottedEvent`/`GroundTruthEvent` objects (or plain dicts with a
    `class`/`class_`, `t`, and -- for predictions -- `confidence` key). See
    the module docstring for the AP definition, the zero-GT-class rule, the
    tie-break, and the inclusive tolerance boundary; all are load-bearing on
    the returned number.

    Returns a JSON-ready dict::

        {"kind": "action_spotting",
         "avg_map": float,
         "tolerances_s": [float, ...],
         "per_tolerance": {"<tol>": {"map": float,
                                     "per_class_ap": {cls: ap}}},
         "counts": {"predictions", "gt_events", "tp", "fp", "fn"}}

    `counts.tp/fp/fn` are totals summed across all classes and all tolerances
    (for the single ball-action tolerance this is the natural per-class sum);
    a no-GT class's predictions count as `fp`.
    """
    preds = list(predictions)
    gts = list(ground_truth)

    preds_by_class: dict[str, list[tuple[float, float, int]]] = {}
    for idx, p in enumerate(preds):
        preds_by_class.setdefault(_class_of(p), []).append((_conf_of(p), _t_of(p), idx))

    gt_by_class: dict[str, list[float]] = {}
    for g in gts:
        gt_by_class.setdefault(_class_of(g), []).append(_t_of(g))

    all_classes = set(preds_by_class) | set(gt_by_class)
    gt_classes = set(gt_by_class)

    tolerances = [float(tol) for tol in tolerances_s]
    per_tolerance: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0

    for tol in tolerances:
        per_class_ap: dict[str, float] = {}
        for cls in sorted(all_classes):
            cls_preds = preds_by_class.get(cls, [])
            cls_gts = gt_by_class.get(cls, [])
            flags = _match_class(cls_preds, cls_gts, tol)
            n_tp = sum(flags)
            total_tp += n_tp
            total_fp += len(flags) - n_tp
            total_fn += len(cls_gts) - n_tp
            if cls in gt_classes:
                per_class_ap[cls] = _average_precision(flags, len(cls_gts))
        map_at_tol = (
            sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0
        )
        per_tolerance[_tol_key(tol)] = {"map": map_at_tol, "per_class_ap": per_class_ap}

    maps = [entry["map"] for entry in per_tolerance.values()]
    avg_map = sum(maps) / len(maps) if maps else 0.0

    return {
        "kind": "action_spotting",
        "avg_map": avg_map,
        "tolerances_s": tolerances,
        "per_tolerance": per_tolerance,
        "counts": {
            "predictions": len(preds),
            "gt_events": len(gts),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
    }


def _tol_key(tol: float) -> str:
    """Stable JSON string key for a tolerance (e.g. 1.0 -> "1.0")."""
    return str(float(tol))


def evaluate_spotting_run(
    run_dir: str | Path,
    event_gt: EventGroundTruth,
    tolerances_s: Sequence[float] = (1.0,),
) -> dict[str, Any] | None:
    """Score a run's `spotting.json` against event GT; None if absent.

    Loads `list[SpottedEvent]` from the run dir via `ArtifactStore`; if the
    spotting artifact is not present (a run that produced no spotting output)
    returns None -- there is nothing to score. Otherwise delegates to
    `average_map` and returns its dict.
    """
    from matchlab_core.artifacts import ArtifactStore
    from matchlab_core.schemas.run import ArtifactName

    store = ArtifactStore(run_dir)
    if not store.exists(ArtifactName.SPOTTING):
        return None
    predictions = store.read_json_list(ArtifactName.SPOTTING, SpottedEvent)
    gt_events: list[GroundTruthEvent] = list(event_gt.events)
    return average_map(predictions, gt_events, tolerances_s)


def class_ap(result: dict[str, Any], class_name: str, tolerance: float = 1.0) -> float:
    """AP for a single class at one tolerance, pulled from an `average_map`
    result. This is how the possession track (SPO-80) reports an honest
    **pass** number: the full `avg_map` averages over every GT class, so a
    pass-only predictor evaluated against multi-class SoccerNet-ball GT is
    diluted by the classes it never attempts -- `class_ap(result, "PASS")` is
    the number that actually reflects pass detection. Returns 0.0 for a class
    with no AP term (no GT for it, or the tolerance absent)."""
    per_tol = result.get("per_tolerance", {}).get(_tol_key(tolerance), {})
    return per_tol.get("per_class_ap", {}).get(class_name, 0.0)


def spotting_headline_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Headline metric(s) for a spotting eval result, analogous to
    `matchlab_core.evaluation.headline_metrics`. `spotting_map_at_1` is the
    dashboard/benchmark number (avg-mAP; == mAP@1 for the single ball-action
    tolerance), rounded to 4 places."""
    return {"spotting_map_at_1": round(result["avg_map"], 4)}
