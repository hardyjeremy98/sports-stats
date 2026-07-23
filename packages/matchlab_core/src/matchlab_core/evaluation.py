"""Score a run's tracking output against ground truth (MOT metrics).

Two levels are scored so the association stage's contribution is measurable:
  tracklet — raw tracker output (tracklets.json ids)
  entity   — post-association identities (players.json player_ids)
entity IDF1 minus tracklet IDF1 is the association gain: positive means the
associator repaired identity fragmentation, negative means it merged wrongly.

A third layer (ADR 004) scores the identity stage itself: cluster
purity/completeness of `PlayerIdentity.label` against GT tracks, plus
coverage/abstention so abstaining on everyone can't score well. See
`_evaluate_identity`.

A fourth, orthogonal metric (SPO-6 / tracklet-modernization) measures
per-tracklet GT contamination directly: `merge_quality` above collapses each
tracklet to one majority GT id and discards everything else, which is
precisely the failure this program cares about -- a tracklet that silently
switches between two players. See `tracklet_purity`, computed at both the
tracklet and entity level under `result["purity"]`.

A fifth, independent backend (SPO-7 / tracklet-modernization) scores the
same per-frame GT/prediction structures with HOTA/DetA/AssA/LocA via vendored
TrackEval metric math (`matchlab_core.hota.compute_hota`,
`matchlab_core._vendor.trackeval`), computed at both levels under
`result["hota"]`. This is a second, authoritative-for-its-own-metrics
backend alongside the motmetrics IDF1/MOTA suite above -- the two are never
reconciled or cross-corrected.

A sixth, LEVEL-INDEPENDENT layer (SPO-9) scores the detector itself, not the
tracker: precision/recall/AP per player-height bin, consecutive miss-burst
lengths, duplicate-detection rate, and temporal box jitter, computed once
(not per tracklet/entity level) from `<run_dir>/detections.jsonl` against the
same scored GT, via `matchlab_core.detection_eval.evaluate_detections` --
see `_load_detections` and `result["detection"]` below. `detections.jsonl`
is absent for imported external runs (`exchange.py::import_mot_tracklets`
writes none), in which case `result["detection"] is None` -- never a crash
or a fabricated score.

A seventh annotation pass (SPO-19) attributes every per-instance ID switch
to a pipeline layer -- detection, online association, refinement (reserved),
offline association -- or an explicit `ambiguous` tag, with the evidence
basis recorded per instance; see `matchlab_core.attribution`.

Requires the `eval` extra (motmetrics; scipy comes in transitively and is
what the HOTA backend actually needs). Imported lazily so the lean server
install works without it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from matchlab_core.gt import GroundTruth

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


def evaluate_run(
    run_dir: str | Path,
    gt: GroundTruth,
    iou_threshold: float = 0.5,
    min_track_length: int | None = None,
) -> dict:
    """Compute MOT metrics + per-instance error list for one run directory.

    Returns a JSON-ready dict (the eval.json artifact).

    `min_track_length` gates the purity block's pre/post-filter split (see
    `tracklet_purity`); if omitted it is discovered from the manifest's
    resolved track-stage config, falling back to the registered track impl's
    `Params.min_length` pydantic default, falling back to `None` (not
    discoverable) if neither is available.
    """
    import motmetrics as mm

    from matchlab_core.hota import compute_hota

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stride = int(manifest["video"].get("sample_stride", 1) or 1)
    frame_count = int(manifest["video"]["frame_count"])
    fps = float(manifest["video"]["fps"] or gt.fps)
    if min_track_length is None:
        min_track_length = _discover_min_track_length(manifest)

    tracklets = json.loads((run_dir / "tracklets.json").read_text())
    entity_of: dict[int, int] = {}
    players_path = run_dir / "players.json"
    players_data: list[dict] = json.loads(players_path.read_text()) if players_path.exists() else []
    for p in players_data:
        for tid in p["tracklet_ids"]:
            entity_of[tid] = p["player_id"]

    # Predictions per frame, per level: frame_idx -> list[(id, xywh)].
    pred_tracklet: dict[int, list[tuple[int, list[float]]]] = {}
    pred_entity: dict[int, list[tuple[int, list[float]]]] = {}
    # Same boxes, indexed per tracklet instead of per frame, for merge_quality
    # and tracklet_purity. entities_by_id is the same shape one level up: all
    # frames of every tracklet an entity absorbed, concatenated.
    tracklets_by_id: dict[int, list[tuple[int, list[float]]]] = {}
    entities_by_id: dict[int, list[tuple[int, list[float]]]] = {}
    for tr in tracklets:
        tid = tr["tracklet_id"]
        # Tracklets the associator never saw (e.g. referees) stay their own
        # identity; offset avoids colliding with player_ids.
        eid = entity_of.get(tid, 100000 + tid)
        frames_xywh: list[tuple[int, list[float]]] = []
        for f in tr["frames"]:
            xywh = _xywh(f["box"])
            pred_tracklet.setdefault(f["frame_idx"], []).append((tid, xywh))
            pred_entity.setdefault(f["frame_idx"], []).append((eid, xywh))
            frames_xywh.append((f["frame_idx"], xywh))
        tracklets_by_id[tid] = frames_xywh
        entities_by_id.setdefault(eid, []).extend(frames_xywh)

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

    # Per-track GT boxes ({source frame -> xywh}) + image dims feed the
    # persistent-switch frame-exit exemption; width/height of 0 (unknown)
    # disables exemption rather than guessing.
    gt_track_boxes: dict[int, dict[int, list[float]]] = {
        t.track_id: {f.frame_idx: _xywh(f.box.model_dump()) for f in t.frames}
        for t in scored_tracks
    }
    gt_frame_size = (gt.width, gt.height)

    # Score only frames the pipeline actually sampled; GT is dense per-frame.
    eval_frames = [f for f in range(0, frame_count, stride) if f in gt_by_frame]

    # Same eval_frames-restricted GT/prediction structures feed both
    # backends (motmetrics below, TrackEval's HOTA in the loop) so IDF1/MOTA
    # and HOTA/DetA/AssA/LocA score identical input -- each is authoritative
    # for its own metrics, this file never reconciles one against the other.
    gt_scored = {f: gt_by_frame.get(f, []) for f in eval_frames}

    levels: dict[str, Any] = {}
    hota_levels: dict[str, Any] = {}
    persistent_levels: dict[str, Any] = {}
    instances: list[dict] = []
    seconds_per_frame = (stride / fps) if fps else 0.0
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
        persistent_levels[level] = persistent_switch_counts(
            _matched_id_sequences(acc),
            seconds_per_frame,
            stride=stride,
            gt_boxes=gt_track_boxes,
            frame_size=gt_frame_size,
        )

        pred_scored = {f: preds.get(f, []) for f in eval_frames}
        hota_levels[level] = compute_hota(gt_scored, pred_scored)

    result = {
        "source": gt.source,
        "sequence": gt.sequence,
        "iou_threshold": iou_threshold,
        "sample_stride": stride,
        "n_frames_evaluated": len(eval_frames),
        "n_gt_tracks": len(scored_tracks),
        "n_gt_tracks_excluded": len(gt.tracks) - len(scored_tracks),
        "levels": levels,
        "hota": hota_levels,
        "persistent_switches": {
            "threshold_headline_s": _PERSISTENCE_HEADLINE_S,
            "tracklet": persistent_levels["tracklet"],
            "entity": persistent_levels["entity"],
        },
        "association": {
            "idf1_gain": round(levels["entity"]["idf1"] - levels["tracklet"]["idf1"], 4),
            "idsw_delta": levels["entity"]["num_switches"] - levels["tracklet"]["num_switches"],
        },
        "instances": sorted(instances, key=lambda i: (i["frame_idx"], i["level"])),
        "identity": _evaluate_identity(players_data, gt_by_frame, pred_entity, eval_frames, iou_threshold),
    }

    from matchlab_core.attribution import attribute_switches, detect_context

    # SPO-19: every switch instance carries a layer attribution or an explicit
    # ambiguous tag. Single-run evidence only here; callers holding an
    # oracle-counterpart eval payload re-invoke attribute_switches to upgrade
    # ambiguous tracklet-level switches (see matchlab_core.attribution).
    result["attribution"] = detect_context(manifest)
    attribute_switches(result)

    mq = merge_quality(tracklets_by_id, players_data, gt_by_frame, iou_threshold)
    result["association"].update({k: v for k, v in mq.items() if k != "merged_pairs"})
    result["association"]["merged_pairs"] = mq["merged_pairs"]
    result["purity"] = {
        "tracklet": tracklet_purity(
            tracklets_by_id, gt_by_frame, fps, stride, iou_threshold, min_track_length
        ),
        "entity": tracklet_purity(
            entities_by_id, gt_by_frame, fps, stride, iou_threshold, min_track_length
        ),
    }

    from matchlab_core.crop_yield import crop_yield

    result["crop_yield"] = crop_yield(
        tracklets_by_id, gt_by_frame, eval_frames, iou_threshold
    )

    det_by_frame = _load_detections(run_dir, eval_frames)
    if det_by_frame is None:
        result["detection"] = None
    else:
        from matchlab_core.detection_eval import evaluate_detections

        result["detection"] = evaluate_detections(
            det_by_frame, gt_scored, iou_threshold=iou_threshold, stride=stride, fps=fps
        )
    return result


def _load_detections(
    run_dir: Path, eval_frames: list[int]
) -> dict[int, list[tuple[float, list[float]]]] | None:
    """Read `<run_dir>/detections.jsonl` into `det_by_frame` (frame_idx ->
    list[(confidence, xywh)]) for the detection-quality layer (SPO-9).

    Keeps only person classes -- player, goalkeeper, referee -- mirroring
    `_SCORED_ROLES` (ball is a separate pipeline stream/GT role, excluded on
    both sides symmetrically); restricted to `eval_frames` so it lines up
    frame-for-frame with the GT structure `evaluate_run` already scores
    against.

    Returns `None` if the file doesn't exist at all -- imported external
    runs (`exchange.py::import_mot_tracklets`) write no detections.jsonl,
    and `evaluate_run` must report `result["detection"] = None` rather than
    crash or fabricate a score. A malformed row raises `ValueError` naming
    the file and line, mirroring `exchange.py`'s convention for the same
    file format.
    """
    path = run_dir / "detections.jsonl"
    if not path.exists():
        return None

    from matchlab_core.schemas.detections import FrameDetections

    eval_frame_set = set(eval_frames)
    det_by_frame: dict[int, list[tuple[float, list[float]]]] = {}
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                frame = FrameDetections.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid FrameDetections row in {path}:{lineno}: {exc}") from exc
            if frame.frame_idx not in eval_frame_set:
                continue
            for det in frame.detections:
                if det.cls not in _SCORED_ROLES:
                    continue
                det_by_frame.setdefault(frame.frame_idx, []).append(
                    (det.confidence, _xywh(det.box.model_dump()))
                )
    return det_by_frame


def _discover_min_track_length(manifest: dict) -> int | None:
    """Where minimum-track-length filtering actually happens: the track
    stage (`stages/track/{iou,botsort}.py`'s `min_length` param), BEFORE
    tracklets.json is written -- tracklets shorter than it never reach this
    evaluator at all, they simply don't exist in tracklets.json. So this can
    only read the resolved threshold back out of the manifest for reporting
    (see `tracklet_purity`'s `note` field); it cannot recover what was
    already dropped.

    Two fallbacks, in order:
    1. `manifest.config.stages.track.params.min_length`, if the resolved
       config snapshot set it explicitly.
    2. Configs written before SPO-15 and hand-built manifests may omit
       `min_length`; shipped configs now state it explicitly. Falling
       back straight to 0 here would silently report a wrong threshold
       with no signal that anything was lost, exactly the failure mode this
       program exists to police. Instead resolve the registered track
       impl's `Params.min_length` default via
       `_resolve_track_min_length_default`.

    Returns `None` (not `0`) if neither works, e.g. the manifest has no
    track-stage config at all (hand-written test fixtures) or names an
    unknown/unresolvable impl -- callers must treat `None` as "not
    discoverable", never coerce it to 0.
    """
    track_cfg = manifest.get("config", {}).get("stages", {}).get("track", {})
    track_params = track_cfg.get("params", {})
    value = track_params.get("min_length")
    if isinstance(value, (int, float)):
        return int(value)
    impl = track_cfg.get("impl")
    return _resolve_track_min_length_default(impl) if impl else None


def _resolve_track_min_length_default(impl: str) -> int | None:
    """Best-effort recovery of a track impl's `min_length` pydantic default
    by instantiating it with no params via the stage registry (the same
    (kind, impl_name) -> class resolution the pipeline itself uses, rather
    than guessing a module path from the impl name).

    Triggers `matchlab_core.stages`' registration import, which pulls in
    every stage module across every kind. Verified safe in a lean/cv-less
    env: every stage module in this repo (detect/team/calibrate/associate/
    identity/track, including the OSNet embedder whose vendored arch module
    has a module-level `import torch`) imports its heavy CV deps (torch,
    trackers, inference, insightface, ultralytics, transformers) lazily
    inside `prepare()`/method bodies, never at import time -- there's even an
    existing test (`test_embedders.py`) asserting torch isn't eagerly
    imported. Still wrapped in try/except: an unregistered impl name, a
    Params model with no `min_length` field, or a future stage module that
    breaks that lazy-import convention must degrade to "not discoverable",
    never crash evaluation.
    """
    try:
        from matchlab_core.registry import build
        from matchlab_core.schemas.run import StageKind

        stage = build(StageKind.TRACK, impl, {})
        value = getattr(getattr(stage, "params", None), "min_length", None)
    except Exception:
        return None
    return int(value) if isinstance(value, (int, float)) else None


def merge_quality(
    tracklets_by_id: dict[int, list[tuple[int, list[float]]]],
    entities: list[dict],
    gt_by_frame: dict[int, list[tuple[int, list[float]]]],
    iou_threshold: float = 0.5,
) -> dict:
    """Judge association's tracklet-to-tracklet merges for correctness, not
    just tracking-count deltas (July-2026 finding: colour merges looked fine
    on IDF1 but were 76% wrong against GT identity).

    Each tracklet is first assigned a majority-vote GT track id: over its
    frames, match its box to GT boxes in that frame by IoU (>= iou_threshold,
    same rule as the MOT levels); each matched frame casts ONE vote, for the
    single best-IoU qualifying GT box, and the tracklet's gt_id is the argmax
    (ties -> lowest gt id, deterministic). Tracklets with zero matched frames
    get gt_id None.

    Then every entity (players.json record) with more than one tracklet
    contributes one "was this merge right?" judgment per unordered tracklet
    pair. A pair is correct iff both sides have a (non-None) gt_id and they
    match. A pair where either side never matched any GT box is an
    unverifiable merge -- counted incorrect for precision (silent wrong
    merges are worse than unmerged tracklets, so "unknown" must not read as
    "correct") but tallied separately as `n_pairs_unmatched` so a run heavy in
    off-pitch/occluded fragments doesn't get misread as badly wrong.

    Synthetic referee/unseen-tracklet entities (100000+tracklet_id) are never
    passed in here -- `entities` is players.json's real records -- so they
    can't manufacture pairs; entities always have >1 real tracklet or none.

    numpy-only (via `_iou_distance`); no motmetrics. Called from `evaluate_run`
    above — the reid-ablation sweep consumes it indirectly through that, not
    standalone.
    """
    gt_id_of_tracklet: dict[int, int | None] = {}
    for tid, frames in tracklets_by_id.items():
        votes = _gt_composition_of_tracklet(frames, gt_by_frame, iou_threshold)
        gt_id_of_tracklet[tid] = min(votes, key=lambda gid: (-votes[gid], gid)) if votes else None

    n_entities_merged = 0
    n_pairs = 0
    n_pairs_correct = 0
    n_pairs_unmatched = 0
    merged_pairs: list[dict] = []
    for e in entities:
        tids = e["tracklet_ids"]
        if len(tids) < 2:
            continue
        n_entities_merged += 1
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                gt_a = gt_id_of_tracklet.get(a)
                gt_b = gt_id_of_tracklet.get(b)
                unmatched = gt_a is None or gt_b is None
                correct = not unmatched and gt_a == gt_b
                n_pairs += 1
                if unmatched:
                    n_pairs_unmatched += 1
                if correct:
                    n_pairs_correct += 1
                merged_pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "player_id": e["player_id"],
                        "gt_a": gt_a,
                        "gt_b": gt_b,
                        "correct": correct,
                    }
                )

    merge_precision = (n_pairs_correct / n_pairs) if n_pairs else None
    return {
        "n_entities_merged": n_entities_merged,
        "n_pairs": n_pairs,
        "n_pairs_correct": n_pairs_correct,
        "n_pairs_unmatched": n_pairs_unmatched,
        "merge_precision": merge_precision,
        "merged_pairs": merged_pairs,
    }


def _gt_composition_of_tracklet(
    frames: list[tuple[int, list[float]]],
    gt_by_frame: dict[int, list[tuple[int, list[float]]]],
    iou_threshold: float,
) -> dict[int, int]:
    """Per-frame single-best-IoU vote against qualifying (>= iou_threshold)
    GT boxes: one vote per frame, for the single best match, not one per
    qualifying box -- a brushing neighbour must not be able to split the
    tally in crowded scenes. Shared by `merge_quality` (majority-vote gt_id)
    and `tracklet_purity` (full composition, not just the argmax)."""
    votes: dict[int, int] = {}
    for frame_idx, box in frames:
        gts = gt_by_frame.get(frame_idx, [])
        if not gts:
            continue
        dist = _iou_distance([g[1] for g in gts], [box], max_dist=1 - iou_threshold)
        best_gi: int | None = None
        for gi in range(len(gts)):
            d = dist[gi, 0]
            if d == d and (best_gi is None or d < dist[best_gi, 0]):  # not NaN
                best_gi = gi
        if best_gi is not None:
            gid = gts[best_gi][0]
            votes[gid] = votes.get(gid, 0) + 1
    return votes


def tracklet_purity(
    tracklets_by_id: dict[int, list[tuple[int, list[float]]]],
    gt_by_frame: dict[int, list[tuple[int, list[float]]]],
    fps: float,
    stride: int,
    iou_threshold: float = 0.5,
    min_track_length: int | None = 0,
) -> dict:
    """Per-tracklet GT composition and contamination -- the metric
    `merge_quality`'s majority-vote collapse silently discards (it only ever
    surfaces one gt_id per tracklet). This is the program's primary-objective
    metric: a tracklet that silently switches between two GT identities is
    worse than a fragmented one, and today's IDF1/MOTA can't see it.

    Matching mirrors `merge_quality` / `_gt_composition_of_tracklet`: per
    frame, single best-IoU match against GT boxes clearing `iou_threshold`,
    independent per tracklet (no cross-tracklet frame assignment) -- the same
    simple, already-in-use matching philosophy, not a fresh Hungarian/global
    assignment.

    Per tracklet: `length` (all frames it has, matched or not),
    `matched_frames`, `gt_composition` (gt_track_id -> matched frame count),
    `majority_gt_track_id` (argmax composition, ties -> lowest gt id; None if
    never matched), `purity` (majority frames / matched frames, None if never
    matched), `mixed_frames`/`mixed_seconds` (matched frames NOT on the
    majority id; seconds accounts for `stride` -- each sampled frame stands in
    for `stride/fps` real seconds).

    Aggregates (frame-weighted, so one long tracklet outweighs many short
    ones) are computed twice: `pre_filter` over every tracklet in
    `tracklets_by_id`, `post_filter` restricted to `length >= min_track_length`
    -- see the `note` field for why that pre/post split can only see
    filtering applied here, not upstream. `min_track_length=None` means the
    threshold could not be determined at all (see `_discover_min_track_length`):
    `post_filter` is then identical to `pre_filter` rather than fabricating a
    filter at an assumed 0 -- abstention over a wrong, silent answer.
    """
    seconds_per_frame = (stride / fps) if fps else 0.0
    records: list[dict] = []
    for tid in sorted(tracklets_by_id):
        frames = tracklets_by_id[tid]
        length = len(frames)
        composition = _gt_composition_of_tracklet(frames, gt_by_frame, iou_threshold)
        matched_frames = sum(composition.values())
        if composition:
            majority_gt = min(composition, key=lambda gid: (-composition[gid], gid))
            majority_frames = composition[majority_gt]
            purity: float | None = majority_frames / matched_frames
            mixed_frames = matched_frames - majority_frames
        else:
            majority_gt = None
            purity = None
            mixed_frames = 0
        records.append(
            {
                "tracklet_id": tid,
                "length": length,
                "matched_frames": matched_frames,
                "unmatched_frames": length - matched_frames,
                "gt_composition": dict(sorted(composition.items())),
                "majority_gt_track_id": majority_gt,
                "purity": round(purity, 4) if purity is not None else None,
                "mixed_frames": mixed_frames,
                "mixed_seconds": round(mixed_frames * seconds_per_frame, 4),
            }
        )

    note = (
        "min-length filtering happens upstream in the track stage "
        "(stages/track/{iou,botsort}.py's min_length param) before "
        "tracklets.json is written -- tracklets dropped there never "
        "reach this evaluator, so pre_filter/post_filter below only "
        "re-apply min_track_length to what's already in tracklets.json; "
        "they cannot recover what was already discarded upstream."
    )
    if min_track_length is None:
        post_records = records
        note += (
            " min_track_length was not discoverable for this run (absent from "
            "the manifest's resolved track-stage config and the registered "
            "track impl's default could not be resolved either); post_filter "
            "is therefore identical to pre_filter -- treat pre_filter as the "
            "only trustworthy view rather than assuming no filtering (0) was "
            "applied upstream."
        )
    else:
        post_records = [r for r in records if r["length"] >= min_track_length]

    return {
        "min_track_length": min_track_length,
        "note": note,
        "tracklets": records,
        "pre_filter": _aggregate_purity_records(records),
        "post_filter": _aggregate_purity_records(post_records),
    }


def _aggregate_purity_records(records: list[dict]) -> dict:
    """Frame-weighted purity summary + tracklets-per-GT-player and
    track-length distributions over one set of `tracklet_purity` records."""
    matched = [r for r in records if r["purity"] is not None]
    total_matched_frames = sum(r["matched_frames"] for r in matched)
    mean_purity = (
        sum(r["purity"] * r["matched_frames"] for r in matched) / total_matched_frames
        if total_matched_frames
        else None
    )
    frac_impure = (sum(1 for r in matched if r["purity"] < 1.0) / len(matched)) if matched else None
    total_mixed_seconds = round(sum(r["mixed_seconds"] for r in records), 4)

    tracklets_per_gt: dict[int, int] = {}
    for r in matched:
        gid = r["majority_gt_track_id"]
        tracklets_per_gt[gid] = tracklets_per_gt.get(gid, 0) + 1
    tpg_counts = sorted(tracklets_per_gt.values())
    tpg_full = _distribution_summary(tpg_counts) if tpg_counts else None
    tpg_summary = (
        {"mean": tpg_full["mean"], "median": tpg_full["median"], "max": tpg_full["max"]}
        if tpg_full is not None
        else None
    )

    lengths = [r["length"] for r in records]
    track_length = _distribution_summary(lengths) if lengths else None

    return {
        "n_tracklets": len(records),
        "n_tracklets_matched": len(matched),
        "mean_purity": round(mean_purity, 4) if mean_purity is not None else None,
        "frac_impure": round(frac_impure, 4) if frac_impure is not None else None,
        "total_mixed_seconds": total_mixed_seconds,
        "tracklets_per_gt_player": {
            "counts": dict(sorted(tracklets_per_gt.items())),
            "summary": tpg_summary,
        },
        "track_length": track_length,
    }


def _distribution_summary(values: list[int]) -> dict:
    """min/p25/median/p75/max/mean via numpy's standard linear-interpolation
    percentile (same convention `np.percentile` uses) -- not a custom
    nearest-rank scheme, so hand-computed test values match a well-known
    formula."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {
        "min": min(values),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "max": max(values),
        "mean": round(float(arr.mean()), 4),
    }


def _persistent_headline(result: dict, level: str) -> int | None:
    """Headline-threshold persistent-switch count, None for eval payloads
    written before the metric existed (re-evaluate to backfill)."""
    ps = result.get("persistent_switches")
    if ps is None:
        return None
    return int(ps[level][_threshold_key(_PERSISTENCE_HEADLINE_S)])


def headline_metrics(result: dict) -> dict[str, float | int | None]:
    """The few numbers worth a dashboard column / diff delta."""
    lv = result["levels"]
    heads: dict[str, float | int | None] = {
        "idf1_tracklet": round(lv["tracklet"]["idf1"], 3),
        "idf1_entity": round(lv["entity"]["idf1"], 3),
        "mota_entity": round(lv["entity"]["mota"], 3),
        "idsw_tracklet": int(lv["tracklet"]["num_switches"]),
        "idsw_entity": int(lv["entity"]["num_switches"]),
        "idsw_persistent_tracklet": _persistent_headline(result, "tracklet"),
        "idsw_persistent_entity": _persistent_headline(result, "entity"),
        "hota_tracklet": result["hota"]["tracklet"]["hota"],
        "hota_entity": result["hota"]["entity"]["hota"],
        "assoc_idf1_gain": result["association"]["idf1_gain"],
        "merge_precision": (
            round(result["association"]["merge_precision"], 3)
            if result["association"]["merge_precision"] is not None
            else None
        ),
    }
    identity = result.get("identity")
    if identity is not None:
        heads["identity_coverage"] = round(identity["coverage"], 3)
        purity = identity["cluster_purity"]
        heads["cluster_purity"] = round(purity, 3) if purity is not None else None
    tracklet_purity_post = result["purity"]["tracklet"]["post_filter"]
    heads["tracklet_purity"] = tracklet_purity_post["mean_purity"]
    heads["mixed_track_seconds"] = tracklet_purity_post["total_mixed_seconds"]
    detection = result.get("detection")
    if detection is not None:
        heads["detection_ap"] = detection["ap"]
        heads["detection_recall"] = detection["recall"]
        heads["detection_miss_burst_p95"] = detection["miss_bursts"]["overall"]["p95"]
    cy = result.get("crop_yield")
    if cy is not None:
        heads["crop_yield_per_player"] = cy["approved_per_gt_player_mean"]
    return heads


def _evaluate_identity(
    players_data: list[dict],
    gt_by_frame: dict[int, list[tuple[int, list[float]]]],
    pred_entity: dict[int, list[tuple[int, list[float]]]],
    eval_frames: list[int],
    iou_threshold: float,
) -> dict | None:
    """Third eval layer (ADR 004): does `identity.label` correspond to the
    right person, judged against GT tracks — cluster purity/completeness plus
    coverage so abstaining on everyone can't score well.

    Synthetic entity ids (100000+tracklet_id, assigned to tracklets the
    associator never grouped) never carry identity output and are excluded;
    only real `player_id`s from players.json are in scope.

    Cluster mass assignment: an entity is one physical person with exactly
    one true identity, so before aggregating purity/completeness each
    matched, labeled entity's FULL overlap mass (every qualifying frame,
    summed across all GT tracks it touched) is attributed to the single GT
    track it overlaps most — argmax by frame count, ties broken by the
    lowest gt_track_id for determinism. Incidental IoU overlap with a second
    GT track (e.g. players crossing paths in a dense scene) is
    detection/bbox noise, not a labeling error, and must not be smeared
    across tracks — that would make a correctly-labeled cluster look impure.
    """
    real_pids = {p["player_id"] for p in players_data}
    if not real_pids:
        return None

    label_of: dict[int, str | None] = {}
    kind_of: dict[int, str] = {}
    for p in players_data:
        identity = p.get("identity") or {}
        kind_of[p["player_id"]] = identity.get("kind") or "none"
        label_of[p["player_id"]] = identity.get("label")

    if all(k == "none" for k in kind_of.values()):
        return None

    # (entity_player_id, gt_track_id) -> n_frames of qualifying (IoU >= threshold) overlap.
    contingency: dict[tuple[int, int], int] = {}
    for f in eval_frames:
        gts = gt_by_frame.get(f, [])
        hyps = [(eid, box) for eid, box in pred_entity.get(f, []) if eid in real_pids]
        if not gts or not hyps:
            continue
        dist = _iou_distance([g[1] for g in gts], [h[1] for h in hyps], max_dist=1 - iou_threshold)
        for gi, (gid, _) in enumerate(gts):
            for hi, (eid, _) in enumerate(hyps):
                if dist[gi, hi] == dist[gi, hi]:  # not NaN => qualifying overlap
                    key = (eid, gid)
                    contingency[key] = contingency.get(key, 0) + 1

    # Collapse to entity -> {gt_track_id: n_frames}. An entity with an empty
    # row has zero qualifying overlap with any GT track and is not matched.
    entity_rows: dict[int, dict[int, int]] = {}
    for (eid, gid), n in contingency.items():
        entity_rows.setdefault(eid, {})[gid] = n

    matched_pids = sorted(entity_rows)  # deterministic order
    n_entities_matched = len(matched_pids)
    labeled_matched = [pid for pid in matched_pids if label_of.get(pid)]
    n_labeled = len(labeled_matched)

    coverage = (n_labeled / n_entities_matched) if n_entities_matched else 0.0
    abstention_rate = 1.0 - coverage

    # Cluster (identity.label) x GT track mass, over labeled+matched entities
    # only: unlabeled entities never pollute purity/completeness. Each
    # entity's full overlap mass goes to its single argmax GT track (see
    # docstring) rather than being spread across every track it overlapped.
    cluster_gt_mass: dict[str, dict[int, int]] = {}
    for pid in labeled_matched:
        label = label_of[pid]
        row = entity_rows[pid]
        assigned_gid = min(row, key=lambda gid: (-row[gid], gid))
        total_mass = sum(row.values())
        bucket = cluster_gt_mass.setdefault(label, {})
        bucket[assigned_gid] = bucket.get(assigned_gid, 0) + total_mass

    n_clusters = len(cluster_gt_mass)

    purity: float | None = None
    completeness: float | None = None
    if n_labeled > 0:
        purity_num = sum(max(bucket.values()) for bucket in cluster_gt_mass.values())
        purity_den = sum(sum(bucket.values()) for bucket in cluster_gt_mass.values())
        purity = purity_num / purity_den if purity_den else None

        gt_cluster_mass: dict[int, dict[str, int]] = {}
        for label, bucket in cluster_gt_mass.items():
            for gid, n in bucket.items():
                gt_cluster_mass.setdefault(gid, {})[label] = n
        completeness_num = sum(max(bucket.values()) for bucket in gt_cluster_mass.values())
        completeness_den = sum(sum(bucket.values()) for bucket in gt_cluster_mass.values())
        completeness = completeness_num / completeness_den if completeness_den else None

    return {
        "n_entities_matched": n_entities_matched,
        "n_labeled": n_labeled,
        "coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "n_clusters": n_clusters,
        "cluster_purity": round(purity, 4) if purity is not None else None,
        "cluster_completeness": round(completeness, 4) if completeness is not None else None,
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


# Persistence thresholds for the flicker-insensitive switch count; 1 s is the
# headline (see docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md).
_PERSISTENCE_THRESHOLDS_S = (0.5, 1.0, 2.0)
_PERSISTENCE_HEADLINE_S = 1.0
# Frame-exit exemption: a switch across an absence where the player left the
# image does not tally (reported under "frame_exit" instead). "Left the image"
# requires border-touching GT boxes on both sides of a >= 0.2 s gap with no GT
# boxes inside it -- full occlusion mid-pitch and losing a visible player are
# NOT exempt (silent swaps are the product's worst failure; abstain from
# excusing, never from charging).
_FRAME_EXIT_BORDER_FRAC = 0.02
_FRAME_EXIT_MIN_ABSENCE_S = 0.2


def persistent_switch_counts(
    id_sequences: dict[int, list[tuple[int, int]]],
    seconds_per_frame: float,
    thresholds_s: tuple[float, ...] = _PERSISTENCE_THRESHOLDS_S,
    *,
    stride: int = 1,
    gt_boxes: dict[int, dict[int, list[float]]] | None = None,
    frame_size: tuple[int, int] | None = None,
) -> dict:
    """Flicker-insensitive ID-switch count. Raw motmetrics IDsw charges every
    matched-ID change, so a 3-frame occlusion flicker (A->B->A) costs 2 -- the
    same as a permanent handoff. Here each GT track's matched-ID sequence is
    segmented into runs of constant ID, runs shorter than the threshold are
    dropped as flicker, and only transitions between surviving runs with
    *different* IDs count -- so A->B->A with a short B is 0 (the reversion
    vanishes with the flicker), while A->B->C with a short B is 1 (identity
    genuinely moved, via a brief intermediary).

    `id_sequences` holds, per GT track, `(source_frame_idx, matched hyp id)`
    at each evaluated frame where a match existed, in frame order. Unmatched
    frames are simply absent: runs are compared across occlusion gaps, because
    a handoff across an occlusion is precisely the real failure.

    A run's duration is `frames x seconds_per_frame` (`stride / fps`), so
    counts are comparable across sampling strides. A boundary-length run
    (exactly the threshold) survives. `seconds_per_frame == 0` (unknown fps)
    drops every run: the result is 0 everywhere rather than a fabricated
    count -- raw IDsw remains the number to read in that case.

    Frame-exit exemption (`gt_boxes` = GT track id -> {source_frame_idx:
    [x, y, w, h]}, `frame_size` = (width, height)): a transition whose gap has
    no GT boxes strictly inside it, border-touching GT boxes at both edges,
    and an absence of at least `_FRAME_EXIT_MIN_ABSENCE_S` is tallied under
    the returned dict's "frame_exit" sub-dict instead of the `t_*` counts.
    When `gt_boxes`/`frame_size` are missing or dimensions are 0 the exemption
    is never applied -- unverifiable switches still count. Note the exemption
    itself is stride-sensitive in the conservative direction: gap edges are
    matched sampled frames, so at coarser strides (or late tracker
    reacquisition) a genuine exit may fail verification and still count --
    it never exempts what it cannot verify.
    """
    counts = {_threshold_key(t): 0 for t in thresholds_s}
    exits = {_threshold_key(t): 0 for t in thresholds_s}
    for gt_id, seq in id_sequences.items():
        runs: list[list] = []  # [hyp_id, n_frames, first_frame, last_frame]
        for frame_idx, hid in seq:
            if runs and runs[-1][0] == hid:
                runs[-1][1] += 1
                runs[-1][3] = frame_idx
            else:
                runs.append([hid, 1, frame_idx, frame_idx])
        track_boxes = (gt_boxes or {}).get(gt_id)
        for t in thresholds_s:
            surviving = [r for r in runs if r[1] * seconds_per_frame >= t]
            for a, b in zip(surviving, surviving[1:]):
                if a[0] == b[0]:
                    continue
                if _is_frame_exit_gap(
                    track_boxes, a[3], b[2], frame_size, seconds_per_frame, stride
                ):
                    exits[_threshold_key(t)] += 1
                else:
                    counts[_threshold_key(t)] += 1
    return {**counts, "frame_exit": exits}


def _is_frame_exit_gap(
    track_boxes: dict[int, list[float]] | None,
    gap_start: int,
    gap_end: int,
    frame_size: tuple[int, int] | None,
    seconds_per_frame: float,
    stride: int,
) -> bool:
    """True only when the gap between two surviving runs is positively a
    frame exit; anything unverifiable is False (the switch then counts)."""
    if not track_boxes or not frame_size or not frame_size[0] or not frame_size[1]:
        return False
    if stride <= 0 or seconds_per_frame <= 0:
        return False
    absence_s = (gap_end - gap_start) * seconds_per_frame / stride
    if absence_s < _FRAME_EXIT_MIN_ABSENCE_S:
        return False
    if any(gap_start < f < gap_end for f in track_boxes):
        return False  # the player was annotated (visible) during the gap
    exit_box = track_boxes.get(gap_start)
    entry_box = track_boxes.get(gap_end)
    if exit_box is None or entry_box is None:
        return False
    return _touches_border(exit_box, frame_size) and _touches_border(entry_box, frame_size)


def _touches_border(box: list[float], frame_size: tuple[int, int]) -> bool:
    x, y, w, h = box
    width, height = frame_size
    mx, my = _FRAME_EXIT_BORDER_FRAC * width, _FRAME_EXIT_BORDER_FRAC * height
    return x <= mx or x + w >= width - mx or y <= my or y + h >= height - my


def _threshold_key(t: float) -> str:
    return f"t_{t:g}s"


def _matched_id_sequences(acc) -> dict[int, list[tuple[int, int]]]:
    """Per GT track, `(source_frame_idx, matched hyp id)` at each evaluated
    frame with a match, in frame order -- from the same motmetrics event
    stream raw IDsw is computed from, so the two never disagree about
    matching."""
    seqs: dict[int, list[tuple[int, int]]] = {}
    for (frame_idx, _), ev in acc.mot_events.iterrows():
        if ev["Type"] in ("MATCH", "SWITCH"):
            seqs.setdefault(int(ev["OId"]), []).append((int(frame_idx), int(ev["HId"])))
    return seqs


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
