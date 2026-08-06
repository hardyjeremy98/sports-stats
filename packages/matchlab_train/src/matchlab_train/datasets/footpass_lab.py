"""Publish a PCBAS half into the Lab: run directory + registered video + run row.

The two-stage spotter is not a `PipelineRunner` pipeline -- it consumes tracking as
an INPUT rather than producing it (see docs/reference/pcbas-inference-recipe.md), so
there is no config whose stages could write a run directory. This module produces the
same run-directory contract by hand so the existing Lab plumbing (artifact endpoint,
run viewer, timeline) works on it unchanged.

One run = one HALF, deliberately. `left_to_right` is a pitch side that rebinds to
clubs at half time, and frame indices only mean anything within the match they were
cut from, so a "whole match" run would be a category error in both.

The video registered is the whole match mp4 -- FOOTPASS ships one file per match with
frame indices continuous across both halves -- and the run's events simply occupy the
half's frame range inside it. No re-encode, no frame offset arithmetic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.pcbas.eval import DEFAULT_CONF_THRESH, DEFAULT_DELTA
from matchlab_core.pcbas.lab_view import build_lab_events
from matchlab_core.pcbas.schema import (
    FRAME,
    LEFT_TO_RIGHT,
    ROI_HEIGHT,
    ROI_WIDTH,
    ROI_X,
    ROI_Y,
    SHIRT_NUMBER,
)
from matchlab_core.schemas.run import ArtifactName, RunManifest, StageStatus
from matchlab_core.schemas.spotting import SpottedEvent

from matchlab_train.datasets.footpass_clips import scale_box
from matchlab_train.datasets.footpass_pcbas import (
    half_to_events,
    load_half,
    load_playbyplay,
    parse_key,
)

# The tactical ROI columns are full-HD; coeff=1.0 is the true, unexpanded box mapped
# into the 352x640 release's pixel space. (The model's own ROI uses ROI_COEFF to
# expand it -- that framing is for pooling, not for showing a human who acted.)
DISPLAY_COEFF = 1.0


def build_box_index(arr: np.ndarray) -> dict[tuple[int, int, int], tuple[float, float, float, float]]:
    """`(frame, left_to_right, shirt) -> display box` for every observed player.

    Rows whose player has no box (off-screen: 17.5% of VAL actions) are absent, so a
    lookup miss is the normal, meaningful "not visible here" -- not an error.
    """
    out: dict[tuple[int, int, int], tuple[float, float, float, float]] = {}
    observed = arr[~np.isnan(arr[:, ROI_X]) & ~np.isnan(arr[:, SHIRT_NUMBER])]
    for row in observed:
        ltr = row[LEFT_TO_RIGHT]
        if np.isnan(ltr):
            continue
        key = (int(row[FRAME]), int(ltr), int(row[SHIRT_NUMBER]))
        if key in out:
            continue  # first row wins, as everywhere else in this pipeline
        out[key] = scale_box(
            row[ROI_X], row[ROI_Y], row[ROI_WIDTH], row[ROI_HEIGHT], DISPLAY_COEFF
        )
    return out


def publish_half(
    *,
    h5_path: Path,
    playbyplay_path: Path,
    video_path: Path,
    run_dir: Path,
    key: str,
    run_id: str,
    fps: float = 25.0,
    delta: int = DEFAULT_DELTA,
    conf_thresh: float = DEFAULT_CONF_THRESH,
    label: str | None = None,
    register: bool = True,
) -> dict:
    """Score one half against its tactical ground truth and write a Lab run."""
    game_id, half = parse_key(key)

    arr = load_half(h5_path, key)
    gt_events = half_to_events(arr, key, fps=fps)
    boxes = build_box_index(arr)

    pred_by_half = load_playbyplay(playbyplay_path, "pred")
    if key not in pred_by_half:
        raise KeyError(
            f"{playbyplay_path} has no predictions for {key} "
            f"(it has {sorted(pred_by_half)}) -- scoring it would silently report "
            f"every ground-truth event as a miss"
        )
    pred = pred_by_half[key]

    # identity="shirt": the predictions come from the reference's shirt-native
    # playbyplay exchange and carry no role, so "slot" is not even expressible here.
    lab = build_lab_events(
        key=key,
        game_id=game_id,
        half=half,
        fps=fps,
        gt=gt_events.events,
        pred=pred,
        boxes=boxes,
        identity="shirt",
        delta=delta,
        conf_thresh=conf_thresh,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(run_dir)
    store.path(ArtifactName.PCBAS_EVENTS).write_text(lab.model_dump_json())

    # Downcast for the existing timeline strip, which knows nothing about players.
    # Predictions only -- a strip that mixed in ground truth would read as detections.
    spotted = [
        SpottedEvent(
            class_=e.class_name,
            frame_idx=e.frame_idx,
            t=e.t,
            confidence=e.score if e.score is not None else 1.0,
            half=half,
        )
        for e in lab.events
        if e.verdict in ("tp", "fp")
    ]
    store.path(ArtifactName.SPOTTING).write_text(
        json.dumps([s.model_dump() for s in spotted])
    )

    from matchlab_core.video import probe

    meta = probe(video_path)
    r = lab.report
    metrics: dict[str, float | int | str] = {
        "pcbas_micro_f1": round(r.micro_f1, 4),
        "pcbas_macro_f1": round(r.macro_f1, 4),
        "pcbas_precision": round(r.precision, 4),
        "pcbas_recall": round(r.recall, 4),
        "pcbas_tp": r.tp,
        "pcbas_fp": r.fp,
        "pcbas_fn": r.fn,
        "pcbas_n_gt": r.n_gt,
        # The headline capability claim: events recovered whose player is off-screen.
        "pcbas_tp_without_bbox": r.tp_without_bbox,
        "pcbas_gt_without_bbox": r.gt_without_bbox,
    }

    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        video=meta,
        config={
            "pcbas": {
                "key": key,
                "h5_path": str(h5_path),
                "playbyplay": str(playbyplay_path),
                "identity": "shirt",
                "delta": delta,
                "conf_thresh": conf_thresh,
            }
        },
        config_name="pcbas-v1-a1-b1",
        artifacts={
            ArtifactName.PCBAS_EVENTS.value: store.relpath(ArtifactName.PCBAS_EVENTS),
            ArtifactName.SPOTTING.value: store.relpath(ArtifactName.SPOTTING),
        },
        metrics=metrics,
        status=StageStatus.COMPLETED,
    )
    (run_dir / "manifest.json").write_text(manifest.model_dump_json())

    if register:
        _register(video_path, run_dir, run_id, key, label, metrics)
    return metrics


def _register(
    video_path: Path,
    run_dir: Path,
    run_id: str,
    key: str,
    label: str | None,
    metrics: dict,
) -> None:
    """Add the video and run rows the Lab lists from.

    The video's `gt_path` is deliberately left unset: that column means TRACKING
    ground truth, and the worker auto-scores any run on such a video with motmetrics.
    This run has no tracklets, so that would write an eval.json of zeros next to a
    perfectly good spotting result.
    """
    from matchlab_core.video import probe
    from matchlab_server.api.videos import register_video_file
    from matchlab_server.db import init_db, session
    from matchlab_server.models import Run, RunStatus

    init_db()
    with session() as db:
        video = register_video_file(db, video_path.name, video_path, probe(video_path))
        db.flush()
        existing = db.get(Run, run_id)
        if existing is not None:
            db.delete(existing)
            db.flush()
        db.add(
            Run(
                id=run_id,
                video_id=video.id,
                config_name="pcbas-v1-a1-b1",
                config_yaml=f"# PCBAS two-stage spotter (A1 + B1) on {key}\n",
                label=label or f"PCBAS {key}",
                status=RunStatus.COMPLETED,
                run_dir=str(run_dir),
                progress_stage="spotting",
                progress_frac=1.0,
                metrics=metrics,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
        )
        db.commit()
        print(f"registered run {run_id} on video {video.id} ({video_path.name})")
