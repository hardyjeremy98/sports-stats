"""Player-centric action-spotting metric: class + time + IDENTITY must all match.

Mirrors `utils/metric_utils.py::evaluate_events_from_json` in the FOOTPASS reference
(Apache-2.0), whose matching rule this reproduces exactly -- see
`test_pcbas_eval_reproduces_reference.py`, which pins our counts to the reference's
own shipped VAL predictions.

**This is not `matchlab_core.action_spotting_eval.average_map`.** That one is
class+time avg-mAP for SoccerNet-ball and knows nothing about who did the action.
Both metrics stay; quoting one for the other's task is a governance error, not a
rounding difference.

Two aggregations are always reported, because either alone misleads:

* `micro_f1` -- one pooled P/R over all events. The reference's headline number, and
  dominated by `pass` and `drive` (5,529 of VAL's 6,070 events).
* `macro_f1` -- mean per-class F1 over the action classes PRESENT IN GT. A class the
  model never gets right (VAL has 26 tackles) can vanish entirely from micro.

A class with `n_gt == 0` is excluded from macro rather than scored 1.0.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.schema import ACTION_CLASSES, CLASS_NAMES

Identity = Literal["slot", "shirt"]

DEFAULT_DELTA = 12
DEFAULT_CONF_THRESH = 0.15


class ClassMetrics(BaseModel):
    class_id: int
    class_name: str
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class PCBASReport(BaseModel):
    delta: int
    conf_thresh: float
    identity: Identity
    per_class: dict[int, ClassMetrics]
    tp: int
    fp: int
    fn: int
    n_gt: int
    precision: float
    recall: float
    micro_f1: float
    macro_f1: float
    # Observability split of the true positives, populated only when the ground
    # truth carries `has_bbox`. `tp_without_bbox` is the headline evidence that the
    # sequence stage recovers actions no visual model can see.
    tp_with_bbox: int = 0
    tp_without_bbox: int = 0
    gt_without_bbox: int = 0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _identity_key(ev: PCBASEvent, identity: Identity) -> tuple[int, ...]:
    if identity == "slot":
        if ev.slot is None:
            raise ValueError(
                f"identity='slot' but event at frame {ev.frame_idx} has no slot "
                f"(and no role_id to derive one from)"
            )
        return (ev.slot,)
    if ev.shirt_number is None:
        raise ValueError(
            f"identity='shirt' but event at frame {ev.frame_idx} has no shirt_number"
        )
    return (ev.left_to_right, ev.shirt_number)


class _Counts:
    """TP/FP/GT accumulated per class across any number of halves."""

    def __init__(self) -> None:
        self.tp: defaultdict[int, int] = defaultdict(int)
        self.fp: defaultdict[int, int] = defaultdict(int)
        self.gt: defaultdict[int, int] = defaultdict(int)
        self.n_pred: defaultdict[int, int] = defaultdict(int)
        # Observability split, counted only where GT carries the flag.
        self.tp_with_bbox = 0
        self.tp_without_bbox = 0
        self.gt_without_bbox = 0


def _match_one_group(
    gt: Sequence[PCBASEvent],
    pred: Sequence[PCBASEvent],
    counts: _Counts,
    *,
    delta: int,
    conf_thresh: float,
    identity: Identity,
) -> None:
    """Greedy match within one half, accumulating into `counts`.

    Faithful to the reference in three details that change the numbers:
    1. Predictions below `conf_thresh` are dropped BEFORE matching, so they cannot
       even become false positives.
    2. All surviving predictions are sorted by DESCENDING SCORE -- globally, not per
       class. A confident wrong-class prediction therefore claims its GT first.
    3. Within a group, the chosen GT is the unmatched one with the smallest
       |dframe| <= delta, with ties going to the earliest GT (`dt < best_dt`).
    """
    buckets: defaultdict[tuple[int, ...], list[PCBASEvent]] = defaultdict(list)
    for ev in gt:
        if ev.class_id not in ACTION_CLASSES:
            continue
        buckets[(*_identity_key(ev, identity), ev.class_id)].append(ev)
        counts.gt[ev.class_id] += 1
        if ev.has_bbox is False:
            counts.gt_without_bbox += 1
    for events in buckets.values():
        events.sort(key=lambda e: e.frame_idx)

    dets = [
        ev
        for ev in pred
        if ev.class_id in ACTION_CLASSES and ev.score >= conf_thresh
    ]
    for ev in dets:
        counts.n_pred[ev.class_id] += 1
    dets.sort(key=lambda e: e.score, reverse=True)

    used: dict[tuple[int, ...], list[bool]] = {
        k: [False] * len(v) for k, v in buckets.items()
    }
    for det in dets:
        group = (*_identity_key(det, identity), det.class_id)
        candidates = buckets.get(group)
        best_i, best_dt = -1, delta + 1
        if candidates is not None:
            for i, gt_ev in enumerate(candidates):
                if used[group][i]:
                    continue
                dt = abs(gt_ev.frame_idx - det.frame_idx)
                if dt <= delta and dt < best_dt:
                    best_i, best_dt = i, dt
        if best_i >= 0:
            used[group][best_i] = True
            counts.tp[det.class_id] += 1
            matched = buckets[group][best_i]
            if matched.has_bbox is True:
                counts.tp_with_bbox += 1
            elif matched.has_bbox is False:
                counts.tp_without_bbox += 1
        else:
            counts.fp[det.class_id] += 1


def _report(
    counts: _Counts, *, delta: int, conf_thresh: float, identity: Identity
) -> PCBASReport:
    per_class: dict[int, ClassMetrics] = {}
    for c in ACTION_CLASSES:
        n_gt, tp, fp = counts.gt[c], counts.tp[c], counts.fp[c]
        if n_gt == 0 and counts.n_pred[c] == 0:
            continue
        fn = n_gt - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_gt if n_gt else 0.0
        per_class[c] = ClassMetrics(
            class_id=c,
            class_name=CLASS_NAMES[c],
            n_gt=n_gt,
            n_pred=counts.n_pred[c],
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
        )

    total_tp = sum(counts.tp.values())
    total_fp = sum(counts.fp.values())
    total_gt = sum(counts.gt.values())
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    scored = [m.f1 for m in per_class.values() if m.n_gt > 0]
    return PCBASReport(
        delta=delta,
        conf_thresh=conf_thresh,
        identity=identity,
        per_class=per_class,
        tp=total_tp,
        fp=total_fp,
        fn=total_gt - total_tp,
        n_gt=total_gt,
        precision=precision,
        recall=recall,
        micro_f1=_f1(precision, recall),
        macro_f1=sum(scored) / len(scored) if scored else 0.0,
        tp_with_bbox=counts.tp_with_bbox,
        tp_without_bbox=counts.tp_without_bbox,
        gt_without_bbox=counts.gt_without_bbox,
    )


def score_events(
    gt: Iterable[PCBASEvent],
    pred: Iterable[PCBASEvent],
    *,
    delta: int = DEFAULT_DELTA,
    conf_thresh: float = DEFAULT_CONF_THRESH,
    identity: Identity = "slot",
) -> PCBASReport:
    """Score one half's worth of events.

    Use `score_halves` for a whole split: frame indices run continuously across the
    halves of a match but overlap between matches, so pooling them would let one
    match's prediction satisfy another's ground truth.
    """
    counts = _Counts()
    _match_one_group(
        list(gt),
        list(pred),
        counts,
        delta=delta,
        conf_thresh=conf_thresh,
        identity=identity,
    )
    return _report(counts, delta=delta, conf_thresh=conf_thresh, identity=identity)


def score_halves(
    gt: Mapping[str, Sequence[PCBASEvent]],
    pred: Mapping[str, Sequence[PCBASEvent]],
    *,
    delta: int = DEFAULT_DELTA,
    conf_thresh: float = DEFAULT_CONF_THRESH,
    identity: Identity = "slot",
) -> PCBASReport:
    """Match within each half key independently, then aggregate the counts.

    A GT half with no predictions contributes all its events as false negatives, and
    a predicted half with no GT contributes all its detections as false positives.
    (The reference silently drops both by intersecting the key sets; its own files
    have identical key sets, so this cannot change the reproduction gate, but
    scoring a partial prediction set honestly matters for us.)
    """
    counts = _Counts()
    for key in sorted(set(gt) | set(pred)):
        _match_one_group(
            gt.get(key, []),
            pred.get(key, []),
            counts,
            delta=delta,
            conf_thresh=conf_thresh,
            identity=identity,
        )
    return _report(counts, delta=delta, conf_thresh=conf_thresh, identity=identity)
