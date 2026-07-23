"""HOTA/DetA/AssA/LocA via the vendored TrackEval metric math
(`matchlab_core._vendor.trackeval`), computed from the SAME per-frame
GT/prediction structures `evaluation.py` already builds for its motmetrics
IDF1/MOTA accumulators (`gt_by_frame`, `pred_tracklet`/`pred_entity`: dict of
frame_idx -> list[(id, xywh)]) -- both backends see identical input, and
each is authoritative for its own metrics (this module never reconciles
HOTA against IDF1/MOTA).

Requires the `eval` extra (scipy, pulled in transitively via motmetrics).
Imported lazily from `evaluate_run`, mirroring the motmetrics idiom, so lean
installs degrade the same way they do today.
"""

from __future__ import annotations

import numpy as np

from matchlab_core._vendor.trackeval.hota import HOTA


def compute_hota(
    gt_by_frame: dict[int, list[tuple[int, list[float]]]],
    pred_by_frame: dict[int, list[tuple[int, list[float]]]],
) -> dict[str, float]:
    """HOTA, DetA, AssA, LocA for one sequence, alpha-averaged (TrackEval's
    standard scalar summary: the mean over its built-in 19-point alpha grid,
    0.05..0.95 step 0.05 -- this IS the metric normally reported as "HOTA" in
    papers/leaderboards, not a value at one threshold).

    Deliberately no `iou_threshold` parameter: unlike this file's sibling
    helpers in `evaluation.py` (`merge_quality`, `tracklet_purity`,
    `_evaluate_identity`), HOTA's alpha grid already sweeps every threshold
    from 0.05 to 0.95 (including 0.5) internally and reports the mean -- an
    external single threshold would be a no-op at best and a silent,
    config-that-does-nothing trap at worst (exactly the failure mode this
    program exists to police), or would diverge from both the standard HOTA
    definition and the golden test's independently generated reference
    values (which were computed on raw/unclamped IoU). A future caller that
    genuinely needs a similarity floor should apply it before calling this
    function -- that's a benchmark-runner concern, not this adapter's.

    `gt_by_frame` / `pred_by_frame` boxes are xywh, matching
    `evaluation.py::_xywh`'s convention. IDs are arbitrary hashable ints (raw
    tracklet/entity/GT-track ids); this function remaps each side to
    contiguous 0..N-1 indices (sorted, deterministic) since TrackEval's data
    dict uses ids as array indices.
    """
    frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    gt_ids_all = sorted({tid for f in frames for tid, _ in gt_by_frame.get(f, [])})
    pred_ids_all = sorted({tid for f in frames for tid, _ in pred_by_frame.get(f, [])})
    gt_index = {tid: i for i, tid in enumerate(gt_ids_all)}
    pred_index = {tid: i for i, tid in enumerate(pred_ids_all)}

    gt_ids_per_ts: list[np.ndarray] = []
    tracker_ids_per_ts: list[np.ndarray] = []
    similarity_per_ts: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0
    for f in frames:
        gts = gt_by_frame.get(f, [])
        preds = pred_by_frame.get(f, [])
        gt_ids_per_ts.append(np.array([gt_index[tid] for tid, _ in gts], dtype=int))
        tracker_ids_per_ts.append(np.array([pred_index[tid] for tid, _ in preds], dtype=int))
        similarity_per_ts.append(_iou_similarity([g[1] for g in gts], [p[1] for p in preds]))
        num_gt_dets += len(gts)
        num_tracker_dets += len(preds)

    data = {
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "num_gt_ids": len(gt_ids_all),
        "num_tracker_ids": len(pred_ids_all),
        "gt_ids": gt_ids_per_ts,
        "tracker_ids": tracker_ids_per_ts,
        "similarity_scores": similarity_per_ts,
    }

    res = HOTA().eval_sequence(data)
    return {
        "hota": round(float(np.mean(res["HOTA"])), 4),
        "deta": round(float(np.mean(res["DetA"])), 4),
        "assa": round(float(np.mean(res["AssA"])), 4),
        "loca": round(float(np.mean(res["LocA"])), 4),
    }


def _iou_similarity(objs: list[list[float]], hyps: list[list[float]]) -> np.ndarray:
    """Pairwise IoU for xywh boxes, raw (no thresholding to NaN) -- unlike
    `evaluation._iou_distance`, which NaNs out pairs below a match threshold
    for motmetrics' contract. TrackEval's HOTA needs the full similarity
    matrix: it sweeps its own alpha thresholds internally (see
    `compute_hota`'s docstring), and non-overlapping boxes already score
    exactly 0.0 IoU, so no external floor is needed for correctness."""
    if not objs or not hyps:
        return np.zeros((len(objs), len(hyps)))
    a = np.asarray(objs, dtype=float)
    b = np.asarray(hyps, dtype=float)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.maximum(
        0.0, np.minimum(ax2[:, None], bx2[None, :]) - np.maximum(ax1[:, None], bx1[None, :])
    )
    ih = np.maximum(
        0.0, np.minimum(ay2[:, None], by2[None, :]) - np.maximum(ay1[:, None], by1[None, :])
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)
