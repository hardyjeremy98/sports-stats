"""Detection-quality layer (SPO-9): the metric suite that tells the program
whether tracking failures come from the DETECTOR or the TRACKER, computed
ONCE per run -- level-independent, never conflated with the tracklet/entity
layers in `evaluation.py`. Structural precedent is the HOTA adapter
(`pitchlab_core.hota`): a standalone, pure-function module folded into
`eval.json` under one top-level key by `evaluate_run`, not the inline style
`tracklet_purity` uses.

Pure numpy only -- no motmetrics/scipy/trackeval. `evaluate_run` lazily
imports this module and passes it the same eval_frames-restricted
`gt_by_frame`-shaped structure it already builds for HOTA/motmetrics
(`dict[frame_idx -> list[(id, xywh)]]`); `det_by_frame` is the analogous
per-frame structure for raw detections.jsonl.

Matching (shared by every metric below): per frame, confidence-descending
greedy matching -- each detection matches the highest-IoU unmatched GT box
with IoU >= `iou_threshold` (VOC-style single-pass assignment, not a global
Hungarian solve). Ties are broken deterministically: equal-IoU GT
candidates for the same detection resolve to the LOWER gt_track_id; equal
confidences between detections resolve by input order (Python's stable
sort keeps original relative order for equal keys). Unmatched detections
are FPs, unmatched GT boxes are FNs.

The function evaluates exactly the frames present as keys in `gt_by_frame`
(document this explicitly since it is easy to get backwards): a frame with
GT but no `det_by_frame` entry counts every GT box as a miss (FN); a frame
absent from `gt_by_frame` is never visited at all, even if `det_by_frame`
has an entry for it (the caller is expected to pass both dicts already
restricted to the same evaluated-frame set -- see `evaluate_run`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Box = Sequence[float]  # (x, y, w, h)


def evaluate_detections(
    det_by_frame: dict[int, list[tuple[float, Box]]],
    gt_by_frame: dict[int, list[tuple[int, Box]]],
    *,
    iou_threshold: float = 0.5,
    height_bin_edges: tuple[float, ...] = (25.0, 50.0, 100.0),
    stride: int = 1,
    fps: float = 25.0,
) -> dict:
    """Detection-layer metrics for one sequence: operating-point
    precision/recall, average precision, per-height-bin breakdown, miss-burst
    distribution, duplicate-detection rate, and temporal box jitter.

    `det_by_frame`: frame_idx -> list[(confidence, xywh)].
    `gt_by_frame`: frame_idx -> list[(gt_track_id, xywh)], already
    role-filtered by the caller (e.g. `_SCORED_ROLES` in `evaluation.py`).

    Every declared output key is always present; values that are
    mathematically undefined (e.g. precision with zero detections, AP with
    zero GT boxes in a bin) are `None`, never NaN, so the result is always
    JSON-clean.

    The pipeline applies a confidence floor upstream of `detections.jsonl`,
    so `ap` here is computed over the emitted operating range only, not a
    full precision-recall curve down to confidence zero.
    """
    frames = sorted(gt_by_frame)

    all_det_records: list[dict] = []
    all_gt_records: list[dict] = []
    frame_matched_gt_ids: dict[int, set[int]] = {}
    frame_gt_to_det_box: dict[int, dict[int, Box]] = {}
    n_duplicates = 0

    for f in frames:
        gts = gt_by_frame[f]
        dets = det_by_frame.get(f, [])
        det_gt, is_dup = _match_frame(dets, gts, iou_threshold)

        matched_ids: set[int] = set()
        gt_to_det: dict[int, Box] = {}
        for di, gi in enumerate(det_gt):
            conf, dbox = dets[di]
            if gi is not None:
                gid, gbox = gts[gi]
                matched_ids.add(gid)
                gt_to_det[gid] = dbox
                all_det_records.append(
                    {
                        "confidence": conf,
                        "is_tp": True,
                        "det_height": dbox[3],
                        "matched_gt_height": gbox[3],
                    }
                )
            else:
                if is_dup[di]:
                    n_duplicates += 1
                all_det_records.append(
                    {
                        "confidence": conf,
                        "is_tp": False,
                        "det_height": dbox[3],
                        "matched_gt_height": None,
                    }
                )
        for _gid, gbox in gts:
            all_gt_records.append({"height": gbox[3]})

        frame_matched_gt_ids[f] = matched_ids
        frame_gt_to_det_box[f] = gt_to_det

    n_detections = len(all_det_records)
    n_gt_boxes = len(all_gt_records)
    n_tp = sum(1 for r in all_det_records if r["is_tp"])

    precision = (n_tp / n_detections) if n_detections else None
    recall = (n_tp / n_gt_boxes) if n_gt_boxes else None
    ap = _compute_ap(
        [r["confidence"] for r in all_det_records],
        [r["is_tp"] for r in all_det_records],
        n_gt_boxes,
    )

    return {
        "iou_threshold": iou_threshold,
        "height_bin_edges": list(height_bin_edges),
        "stride": stride,
        "fps": fps,
        "n_frames_evaluated": len(frames),
        "n_detections": n_detections,
        "n_gt_boxes": n_gt_boxes,
        "precision": _round(precision),
        "recall": _round(recall),
        "ap": ap,
        "by_height_bin": _height_bins(all_det_records, all_gt_records, height_bin_edges),
        "miss_bursts": _miss_bursts(gt_by_frame, frame_matched_gt_ids, frames, stride, fps),
        "duplicates": {
            "n_duplicates": n_duplicates,
            "duplicate_rate": _round(n_duplicates / n_detections) if n_detections else None,
        },
        "jitter": _jitter(gt_by_frame, frame_gt_to_det_box, frame_matched_gt_ids, frames),
    }


def _iou_xywh(a: Box, b: Box) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_frame(
    dets: list[tuple[float, Box]],
    gts: list[tuple[int, Box]],
    iou_threshold: float,
) -> tuple[list[int | None], list[bool]]:
    """Confidence-descending greedy match. Returns, per detection index:
    the matched GT index into `gts` (or None), and whether an unmatched
    detection is a "duplicate" (overlaps, at >= iou_threshold, a GT box
    some OTHER detection in this frame already claimed)."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][0])
    used: set[int] = set()
    det_gt: list[int | None] = [None] * len(dets)
    for i in order:
        best_g: int | None = None
        best_iou = -1.0
        for g, (gid, gbox) in enumerate(gts):
            if g in used:
                continue
            iou = _iou_xywh(dets[i][1], gbox)
            if iou < iou_threshold:
                continue
            if best_g is None or iou > best_iou or (iou == best_iou and gid < gts[best_g][0]):
                best_g, best_iou = g, iou
        if best_g is not None:
            det_gt[i] = best_g
            used.add(best_g)

    is_dup = [False] * len(dets)
    for i in range(len(dets)):
        if det_gt[i] is not None:
            continue
        for g in used:
            if _iou_xywh(dets[i][1], gts[g][1]) >= iou_threshold:
                is_dup[i] = True
                break
    return det_gt, is_dup


def _compute_ap(confidences: list[float], is_tp: list[bool], n_gt: int) -> float | None:
    """Average precision via all-point (VOC2010+) interpolation: the
    precision-recall envelope is the running max of precision scanned from
    the high-recall end backwards, and AP is the area under that envelope
    (sum of delta-recall * envelope-precision at each recall step).

    `None` (never NaN) when `n_gt == 0` -- AP is undefined with no GT to
    recall, not zero. `0.0` when there are GT boxes but zero detections --
    the precision-recall curve never leaves the origin, so its area is
    genuinely zero, not undefined.
    """
    if n_gt == 0:
        return None
    if not confidences:
        return 0.0

    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    tp = np.array([1.0 if is_tp[i] else 0.0 for i in order])
    fp = np.array([0.0 if is_tp[i] else 1.0 for i in order])
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return round(ap, 4)


def _bin_index(height: float, edges: tuple[float, ...]) -> int:
    idx = 0
    for e in edges:
        if height >= e:
            idx += 1
        else:
            break
    return idx


def _bin_label(lo: float | None, hi: float | None) -> str:
    if lo is None:
        return f"h<{hi:g}"
    if hi is None:
        return f"h>={lo:g}"
    return f"{lo:g}<=h<{hi:g}"


def _height_bins(
    det_records: list[dict], gt_records: list[dict], edges: tuple[float, ...]
) -> list[dict]:
    """Per-height-bin precision/recall/AP. Binning rule: TP pairs and FNs
    bin by the GT box's pixel height (`gt_records`, and a TP det record's
    `matched_gt_height`); FPs bin by their OWN detected height
    (`det_height`) since they have no GT to inherit a height from. Fixed
    pixel edges (not percentile) so bin numbers are comparable across runs
    of the same sequence/resolution."""
    n_bins = len(edges) + 1
    gt_counts = [0] * n_bins
    for r in gt_records:
        gt_counts[_bin_index(r["height"], edges)] += 1

    bin_dets: list[list[dict]] = [[] for _ in range(n_bins)]
    for r in det_records:
        h = r["matched_gt_height"] if r["is_tp"] else r["det_height"]
        bin_dets[_bin_index(h, edges)].append(r)

    records = []
    for b in range(n_bins):
        lo = edges[b - 1] if b > 0 else None
        hi = edges[b] if b < len(edges) else None
        dets_b = bin_dets[b]
        n_det_b = len(dets_b)
        n_gt_b = gt_counts[b]
        tp_b = sum(1 for r in dets_b if r["is_tp"])
        precision_b = (tp_b / n_det_b) if n_det_b else None
        recall_b = (tp_b / n_gt_b) if n_gt_b else None
        ap_b = _compute_ap([r["confidence"] for r in dets_b], [r["is_tp"] for r in dets_b], n_gt_b)
        records.append(
            {
                "bin": _bin_label(lo, hi),
                "edges": [lo, hi],
                "n_gt": n_gt_b,
                "n_det": n_det_b,
                "precision": _round(precision_b),
                "recall": _round(recall_b),
                "ap": ap_b,
            }
        )
    return records


def _consecutive_false_runs(flags: list[bool]) -> list[int]:
    runs: list[int] = []
    cur = 0
    for flag in flags:
        if not flag:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _distribution_summary(values: list[int]) -> dict[str, float | int | None]:
    """min/median/p95/max/mean via numpy's linear-interpolation percentile.
    A local twin of `evaluation._distribution_summary` -- deliberately NOT
    imported from there, keeping this module's purity/independence from
    evaluation.py per the layer design (see module docstring) -- with a
    different key set: `p95` added, since burst-length analysis and the
    `detection_miss_burst_p95` headline need it, and no `p25`/`p75`, which
    the tracklet-purity summary this mirrors has but detection bursts do
    not need."""
    arr = np.asarray(values, dtype=float)
    return {
        "min": min(values),
        "median": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": max(values),
        "mean": round(float(arr.mean()), 4),
    }


def _null_summary() -> dict[str, None]:
    return {"min": None, "median": None, "p95": None, "max": None, "mean": None}


def _miss_bursts(
    gt_by_frame: dict[int, list[tuple[int, Box]]],
    frame_matched_gt_ids: dict[int, set[int]],
    frames: list[int],
    stride: int,
    fps: float,
) -> dict:
    """Per-GT-track consecutive-miss-burst lengths, in evaluated-frame
    units. For each GT track, walk ITS OWN frames (the evaluated frames in
    which it has a GT box, in order) and find maximal runs of consecutive
    unmatched frames -- a frame where the track has no GT box at all
    (e.g. off-screen) is simply absent from that track's own frame list, so
    it neither starts nor continues a burst (mirrors
    `evaluation.tracklet_purity`'s per-tracklet independence, not a global
    timeline)."""
    track_frames: dict[int, list[int]] = {}
    for f in frames:
        for gid, _ in gt_by_frame[f]:
            track_frames.setdefault(gid, []).append(f)

    per_track: dict[int, dict] = {}
    all_lengths: list[int] = []
    n_tracks_with_bursts = 0
    for gid in sorted(track_frames):
        fl = track_frames[gid]
        matched_flags = [gid in frame_matched_gt_ids.get(f, set()) for f in fl]
        bursts = _consecutive_false_runs(matched_flags)
        if bursts:
            n_tracks_with_bursts += 1
        all_lengths.extend(bursts)
        per_track[gid] = {
            "n_bursts": len(bursts),
            "max_burst": max(bursts) if bursts else None,
            "burst_lengths_summary": _distribution_summary(bursts) if bursts else None,
        }

    overall: dict = dict(_distribution_summary(all_lengths) if all_lengths else _null_summary())
    p95 = overall["p95"]
    overall["burst_seconds_p95"] = round(p95 * stride / fps, 4) if p95 is not None and fps else None

    return {
        "stride": stride,
        "fps": fps,
        "per_track": per_track,
        "overall": overall,
        "n_tracks_with_bursts": n_tracks_with_bursts,
    }


def _jitter(
    gt_by_frame: dict[int, list[tuple[int, Box]]],
    frame_gt_to_det_box: dict[int, dict[int, Box]],
    frame_matched_gt_ids: dict[int, set[int]],
    frames: list[int],
) -> dict:
    """Temporal box jitter measured through GT association: for each GT
    track and each pair of ADJACENT entries in its own evaluated-frame
    presence list where BOTH are matched, the residual offset
    `r(t) = det_center - gt_center` isolates detector box instability from
    real player motion (subtracting the GT's own position cancels out
    genuine movement). Jitter contribution is `|r(t+1) - r(t)|` (Euclidean
    pixels) plus a size term `|Δ(det_h - gt_h)|`.

    "Adjacent" means adjacent in the track's OWN frame-presence list, not
    raw video-frame adjacency -- a track's absence from an evaluated frame,
    or an unmatched frame, breaks the chain the same way: no pair is ever
    formed across a gap, so a real detector dropout never gets miscounted
    as a big jitter spike.
    """
    track_frames: dict[int, list[int]] = {}
    gt_box_at: dict[tuple[int, int], Box] = {}
    for f in frames:
        for gid, box in gt_by_frame[f]:
            track_frames.setdefault(gid, []).append(f)
            gt_box_at[(f, gid)] = box

    center_deltas: list[float] = []
    height_deltas: list[float] = []
    for gid, fl in track_frames.items():
        matched_flags = [gid in frame_matched_gt_ids.get(f, set()) for f in fl]
        for i in range(len(fl) - 1):
            if not (matched_flags[i] and matched_flags[i + 1]):
                continue
            f_a, f_b = fl[i], fl[i + 1]
            gbox_a, gbox_b = gt_box_at[(f_a, gid)], gt_box_at[(f_b, gid)]
            dbox_a, dbox_b = frame_gt_to_det_box[f_a][gid], frame_gt_to_det_box[f_b][gid]
            r_a = _residual(dbox_a, gbox_a)
            r_b = _residual(dbox_b, gbox_b)
            center_deltas.append(((r_b[0] - r_a[0]) ** 2 + (r_b[1] - r_a[1]) ** 2) ** 0.5)
            height_deltas.append(abs((dbox_b[3] - gbox_b[3]) - (dbox_a[3] - gbox_a[3])))

    n_pairs = len(center_deltas)
    return {
        "center_jitter_mean": round(float(np.mean(center_deltas)), 4) if n_pairs else None,
        "center_jitter_p95": round(float(np.percentile(center_deltas, 95)), 4) if n_pairs else None,
        "height_jitter_mean": round(float(np.mean(height_deltas)), 4) if n_pairs else None,
        "height_jitter_p95": round(float(np.percentile(height_deltas, 95)), 4) if n_pairs else None,
        "n_pairs": n_pairs,
    }


def _residual(det_box: Box, gt_box: Box) -> tuple[float, float]:
    dx, dy, dw, dh = det_box
    gx, gy, gw, gh = gt_box
    return (dx + dw / 2 - (gx + gw / 2), dy + dh / 2 - (gy + gh / 2))


def _round(v: float | None) -> float | None:
    return round(v, 4) if v is not None else None
