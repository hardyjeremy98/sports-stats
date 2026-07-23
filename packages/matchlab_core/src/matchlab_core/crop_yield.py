"""Quality-approved crop-yield guardrail (SPO-30).

A tracker that wins on purity by fragmenting players into short/small
tracklets starves downstream identity evidence. This measures approved crops
per GT player from the tracker's OUTPUT boxes, so a winning stack cannot
quietly starve the body/face evidence later stages depend on.

Geometry proxy by design: every Phase 3 candidate consumes identical frozen
detections, so per-box confidence and sharpness are constant across trackers —
the box HEIGHT gate (the offline associator's ``min_box_height_px``, see
``stages/associate/global_reid.py``) is the tracker-attributable component of
crop yield. The offline associator stays frozen; only its threshold *values*
are reused here.
"""

from __future__ import annotations


def _majority_gt(frames, gt_by_frame, iou_threshold: float) -> int | None:
    """The GT track id a tracklet overlaps most across its frames, or None."""
    from matchlab_core.evaluation import _iou_distance

    votes: dict[int, int] = {}
    for frame_idx, xywh in frames:
        gts = gt_by_frame.get(frame_idx, [])
        if not gts:
            continue
        dist = _iou_distance([g[1] for g in gts], [xywh], max_dist=1 - iou_threshold)
        best_i, best_d = None, None
        for i in range(len(gts)):
            d = float(dist[i][0])
            if d != d:  # NaN -> no overlap within threshold
                continue
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            gid = gts[best_i][0]
            votes[gid] = votes.get(gid, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


def crop_yield(
    tracklets_by_id,
    gt_by_frame,
    eval_frames,
    iou_threshold: float,
    min_box_height_px: float = 60.0,
    min_crops_per_tracklet: int = 2,
) -> dict:
    """Approved-crop yield from tracker output boxes.

    ``tracklets_by_id``: tid -> [(frame_idx, [x, y, w, h])].
    A box is approved if it lands on a scored frame and its height >=
    ``min_box_height_px``. A tracklet with fewer than
    ``min_crops_per_tracklet`` approved boxes is "starved" (mirrors the
    associator's abstention threshold).
    """
    eval_set = set(eval_frames)
    per_tracklet_counts: list[int] = []
    per_player: dict[int, int] = {}
    starved = 0
    for _tid, frames in tracklets_by_id.items():
        scored = [(f, xywh) for f, xywh in frames if f in eval_set]
        approved = sum(1 for _f, xywh in scored if xywh[3] >= min_box_height_px)
        per_tracklet_counts.append(approved)
        if approved < min_crops_per_tracklet:
            starved += 1
        gid = _majority_gt(scored, gt_by_frame, iou_threshold)
        if gid is not None:
            per_player[gid] = per_player.get(gid, 0) + approved
    n_tracklets = len(per_tracklet_counts) or 1
    player_vals = list(per_player.values())
    return {
        "approved_total": sum(per_tracklet_counts),
        "starved_tracklet_fraction": round(starved / n_tracklets, 4),
        "approved_per_gt_player_mean": (
            round(sum(player_vals) / len(player_vals), 4) if player_vals else 0.0
        ),
        "per_tracklet": {
            "mean": round(sum(per_tracklet_counts) / n_tracklets, 4),
            "min": min(per_tracklet_counts) if per_tracklet_counts else 0,
            "max": max(per_tracklet_counts) if per_tracklet_counts else 0,
        },
        "params": {
            "min_box_height_px": min_box_height_px,
            "min_crops_per_tracklet": min_crops_per_tracklet,
        },
    }
