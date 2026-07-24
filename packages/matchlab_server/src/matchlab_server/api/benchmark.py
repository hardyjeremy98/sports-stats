"""GET /api/benchmark — config x GT-video matrix aggregating repeat completed runs.

Per ADR 004, batch experiments aggregate GT metrics (not artifact counts): this
groups completed, GT-scored runs by (config_name, normalized-config-yaml hash) and
computes mean/range per (group, video) cell. Read-only; no schema changes."""

from __future__ import annotations

import hashlib

import yaml
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from matchlab_server.api.schemas import BenchmarkOut
from matchlab_server.db import get_db
from matchlab_server.models import Run, RunStatus, Video

router = APIRouter(prefix="/api", tags=["benchmark"])

# Exact key list (brief): the run must have at least one of these, numeric and
# non-null, to be eligible; aggregation below is per-key over just these.
BENCHMARK_METRIC_KEYS = (
    "idf1_tracklet",
    "idf1_entity",
    "mota_entity",
    "idsw_tracklet",
    "idsw_entity",
    # Flicker-insensitive (>=1s persistence) switch counts; spec:
    # docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md.
    "idsw_persistent_tracklet",
    "idsw_persistent_entity",
    "assoc_idf1_gain",
    "merge_precision",
    "identity_coverage",
    "cluster_purity",
    # SPO-52: naming-vs-GT-jersey layer (roster precision reported jointly
    # with abstention so abstain-everywhere can't masquerade as precise).
    "roster_precision",
    "naming_abstention",
    # SPO-59: entity-level purity, the do-no-harm gate's contamination metric.
    "entity_purity",
    # SPO-20: the decision metrics the tracklet-modernization program is
    # steered by. `tracklet_purity`/`mixed_track_seconds` are the tracklet
    # layer (SPO-6) -- not to be confused with `cluster_purity` above, which is
    # the semantic identity layer (ADR 004).
    "hota_tracklet",
    "hota_entity",
    "tracklet_purity",
    "mixed_track_seconds",
    "detection_ap",
    "detection_recall",
    # SPO-49: action-spotting avg-mAP@1. A pure spotting run carries none of
    # the tracking keys above, so without this it would be ineligible for the
    # benchmark matrix entirely.
    "spotting_map_at_1",
)


def _is_numeric(value: object) -> bool:
    # bool is an int subclass but was never a benchmark metric value; keep the
    # numeric check honest.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _config_hash(config_yaml: str) -> str:
    """First 12 hex chars of sha1(normalized YAML). Normalizing by
    parse-then-sorted-redump means formatting/key-order differences don't split
    groups; unparsable YAML falls back to hashing the raw string."""
    try:
        normalized = yaml.safe_dump(yaml.safe_load(config_yaml), sort_keys=True)
    except yaml.YAMLError:
        normalized = config_yaml
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


@router.get("/benchmark", response_model=BenchmarkOut)
def get_benchmark(db: Session = Depends(get_db)):
    runs = db.scalars(
        select(Run).where(Run.status == RunStatus.COMPLETED).order_by(Run.created_at.desc())
    ).all()
    if not runs:
        return {"videos": [], "groups": []}

    video_ids = {r.video_id for r in runs}
    videos_by_id = {v.id: v for v in db.scalars(select(Video).where(Video.id.in_(video_ids)))}

    # groups keyed by (config_name, config_hash) -> video_id -> [Run] (query order
    # is already newest-first, so appending preserves that within each cell).
    groups: dict[tuple[str, str], dict[int, list[Run]]] = {}
    used_video_ids: set[int] = set()

    for run in runs:
        video = videos_by_id.get(run.video_id)
        if video is None or not video.gt_path:
            continue
        metrics = run.metrics or {}
        if not any(k in metrics and metrics[k] is not None for k in BENCHMARK_METRIC_KEYS):
            continue
        key = (run.config_name, _config_hash(run.config_yaml))
        cells = groups.setdefault(key, {})
        cells.setdefault(run.video_id, []).append(run)
        used_video_ids.add(run.video_id)

    group_outs = []
    for (config_name, config_hash), cells_by_video in groups.items():
        cells_out = {}
        n_runs_total = 0
        for video_id, cell_runs in cells_by_video.items():
            n_runs_total += len(cell_runs)
            metrics_mean: dict[str, float] = {}
            metrics_range: dict[str, list[float]] = {}
            for metric_key in BENCHMARK_METRIC_KEYS:
                values = [
                    run.metrics[metric_key]
                    for run in cell_runs
                    if isinstance(run.metrics, dict) and _is_numeric(run.metrics.get(metric_key))
                ]
                if not values:
                    continue
                metrics_mean[metric_key] = sum(values) / len(values)
                if len(cell_runs) > 1 and len(values) > 1:
                    metrics_range[metric_key] = [min(values), max(values)]
            cells_out[str(video_id)] = {
                "n_runs": len(cell_runs),
                "run_ids": [run.id for run in cell_runs],
                "metrics_mean": metrics_mean,
                "metrics_range": metrics_range,
            }
        group_outs.append(
            {
                "config_name": config_name,
                "config_hash": config_hash,
                "n_runs": n_runs_total,
                "cells": cells_out,
            }
        )

    group_outs.sort(key=lambda g: (g["config_name"], g["config_hash"]))

    videos_out = [
        {"video_id": vid, "filename": videos_by_id[vid].filename, "sequence": None}
        for vid in sorted(used_video_ids)
    ]

    return {"videos": videos_out, "groups": group_outs}
