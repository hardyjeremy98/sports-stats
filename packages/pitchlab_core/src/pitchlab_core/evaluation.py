"""Score a run's tracking output against ground truth (MOT metrics).

Two levels are scored so the association stage's contribution is measurable:
  tracklet — raw tracker output (tracklets.json ids)
  entity   — post-association identities (players.json player_ids)
entity IDF1 minus tracklet IDF1 is the association gain: positive means the
associator repaired identity fragmentation, negative means it merged wrongly.

Requires the `eval` extra (motmetrics). Imported lazily so the lean server
install works without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pitchlab_core.gt import GroundTruth

# GT roles scored against the tracker. Ball is a separate pipeline stream, and
# "other" (staff, photographers) is not something we ask the tracker to hold.
_SCORED_ROLES = ("player", "goalkeeper", "referee")

_METRICS = [
    "idf1",
    "idp",
    "idr",
    "mota",
    "num_switches",
    "num_fragmentations",
    "num_false_positives",
    "num_misses",
    "num_objects",
    "num_unique_objects",
    "mostly_tracked",
    "mostly_lost",
    "precision",
    "recall",
]


def evaluate_run(run_dir: str | Path, gt: GroundTruth, iou_threshold: float = 0.5) -> dict:
    """Compute MOT metrics + per-instance error list for one run directory.

    Returns a JSON-ready dict (the eval.json artifact)."""
    import motmetrics as mm

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stride = int(manifest["video"].get("sample_stride", 1) or 1)
    frame_count = int(manifest["video"]["frame_count"])
    fps = float(manifest["video"]["fps"] or gt.fps)

    tracklets = json.loads((run_dir / "tracklets.json").read_text())
    entity_of: dict[int, int] = {}
    players_path = run_dir / "players.json"
    if players_path.exists():
        for p in json.loads(players_path.read_text()):
            for tid in p["tracklet_ids"]:
                entity_of[tid] = p["player_id"]

    # Predictions per frame, per level: frame_idx -> list[(id, xywh)].
    pred_tracklet: dict[int, list[tuple[int, list[float]]]] = {}
    pred_entity: dict[int, list[tuple[int, list[float]]]] = {}
    for tr in tracklets:
        tid = tr["tracklet_id"]
        # Tracklets the associator never saw (e.g. referees) stay their own
        # identity; offset avoids colliding with player_ids.
        eid = entity_of.get(tid, 100000 + tid)
        for f in tr["frames"]:
            xywh = _xywh(f["box"])
            pred_tracklet.setdefault(f["frame_idx"], []).append((tid, xywh))
            pred_entity.setdefault(f["frame_idx"], []).append((eid, xywh))

    # GT per frame, restricted to scored roles.
    scored_tracks = [t for t in gt.tracks if t.role in _SCORED_ROLES]
    gt_by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    for t in scored_tracks:
        for f in t.frames:
            gt_by_frame.setdefault(f.frame_idx, []).append((t.track_id, _xywh(f.box.model_dump())))
    gt_label = {
        t.track_id: (t.role if t.role != "player" else f"#{t.jersey}" if t.jersey else "player")
        + (f" ({t.team})" if t.team else "")
        for t in scored_tracks
    }

    # Score only frames the pipeline actually sampled; GT is dense per-frame.
    eval_frames = [f for f in range(0, frame_count, stride) if f in gt_by_frame]

    levels: dict[str, Any] = {}
    instances: list[dict] = []
    for level, preds in (("tracklet", pred_tracklet), ("entity", pred_entity)):
        acc = mm.MOTAccumulator(auto_id=False)
        for f in eval_frames:
            gts = gt_by_frame.get(f, [])
            hyps = preds.get(f, [])
            # Not mm.distances.iou_matrix: it uses np.asfarray, removed in numpy 2.
            dist = _iou_distance(
                [g[1] for g in gts], [h[1] for h in hyps], max_dist=1 - iou_threshold
            )
            acc.update([g[0] for g in gts], [h[0] for h in hyps], dist, frameid=f)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=_METRICS, name=level)
        row = summary.loc[level]
        levels[level] = {k: _num(row[k]) for k in _METRICS}
        instances.extend(_switch_instances(acc, level, fps, gt_label))

    result = {
        "source": gt.source,
        "sequence": gt.sequence,
        "iou_threshold": iou_threshold,
        "sample_stride": stride,
        "n_frames_evaluated": len(eval_frames),
        "n_gt_tracks": len(scored_tracks),
        "n_gt_tracks_excluded": len(gt.tracks) - len(scored_tracks),
        "levels": levels,
        "association": {
            "idf1_gain": round(levels["entity"]["idf1"] - levels["tracklet"]["idf1"], 4),
            "idsw_delta": levels["entity"]["num_switches"] - levels["tracklet"]["num_switches"],
        },
        "instances": sorted(instances, key=lambda i: (i["frame_idx"], i["level"])),
    }
    return result


def headline_metrics(result: dict) -> dict[str, float | int]:
    """The few numbers worth a dashboard column / diff delta."""
    lv = result["levels"]
    return {
        "idf1_tracklet": round(lv["tracklet"]["idf1"], 3),
        "idf1_entity": round(lv["entity"]["idf1"], 3),
        "mota_entity": round(lv["entity"]["mota"], 3),
        "idsw_tracklet": int(lv["tracklet"]["num_switches"]),
        "idsw_entity": int(lv["entity"]["num_switches"]),
        "assoc_idf1_gain": result["association"]["idf1_gain"],
    }


def _xywh(box: dict) -> list[float]:
    return [box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"]]


def _iou_distance(objs: list[list[float]], hyps: list[list[float]], max_dist: float):
    """Pairwise 1-IoU for xywh boxes; entries above max_dist become NaN
    (unmatchable), matching motmetrics' iou_matrix contract."""
    import numpy as np

    if not objs or not hyps:
        return np.empty((len(objs), len(hyps)))
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
        dist = 1.0 - np.where(union > 0, inter / union, 0.0)
    return np.where(dist > max_dist, np.nan, dist)


def _num(v) -> float | int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if f != f:  # NaN (e.g. no matches at all)
        return 0
    return int(f) if f.is_integer() else round(f, 4)


def _switch_instances(acc, level: str, fps: float, gt_label: dict[int, str]) -> list[dict]:
    """Extract ID-switch events with enough context to seek to them: which GT
    track, which predicted id it left and joined, and when."""
    events = acc.mot_events
    last_hyp: dict[Any, Any] = {}
    out: list[dict] = []
    for (frame_idx, _), ev in events.iterrows():
        if ev["Type"] in ("MATCH", "SWITCH"):
            oid, hid = ev["OId"], ev["HId"]
            if ev["Type"] == "SWITCH":
                out.append(
                    {
                        "level": level,
                        "kind": "id_switch",
                        "frame_idx": int(frame_idx),
                        "t": round(int(frame_idx) / fps, 2),
                        "gt_track_id": int(oid),
                        "gt_label": gt_label.get(int(oid), "?"),
                        "prev_id": _maybe_int(last_hyp.get(oid)),
                        "new_id": _maybe_int(hid),
                    }
                )
            last_hyp[oid] = hid
    return out


def _maybe_int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
