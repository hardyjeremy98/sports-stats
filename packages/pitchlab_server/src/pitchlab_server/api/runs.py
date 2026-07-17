from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pitchlab_core.artifacts import ARTIFACT_FILES
from pitchlab_core.config import PipelineConfig
from pitchlab_core.schemas import ArtifactName
from sqlalchemy import select
from sqlalchemy.orm import Session

from pitchlab_server.api.schemas import (
    RunCreate,
    RunDetailOut,
    RunDiffOut,
    RunOut,
    VideoOut,
)
from pitchlab_server.db import get_db
from pitchlab_server.evaluation import diff_switch_instances
from pitchlab_server.models import Job, Run, Video
from pitchlab_server.settings import get_settings

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", response_model=RunOut)
def create_run(body: RunCreate, db: Session = Depends(get_db)):
    settings = get_settings()
    video = db.get(Video, body.video_id)
    if video is None:
        raise HTTPException(404, "Video not found")

    if body.config_yaml:
        try:
            config = PipelineConfig.model_validate(__import__("yaml").safe_load(body.config_yaml))
        except Exception as exc:  # pydantic/yaml errors -> 422 with detail
            raise HTTPException(422, f"Invalid pipeline config: {exc}") from exc
    elif body.config_name:
        path = settings.config_dir / f"pipeline.{body.config_name}.yaml"
        candidates = list(settings.config_dir.glob("pipeline.*.yaml"))
        config = None
        if path.exists():
            config = PipelineConfig.from_yaml(path)
        else:
            for c in candidates:
                cfg = PipelineConfig.from_yaml(c)
                if cfg.name == body.config_name:
                    config = cfg
                    break
        if config is None:
            raise HTTPException(404, f"No config named '{body.config_name}'")
    else:
        raise HTTPException(422, "Provide config_name or config_yaml")

    run_id = uuid.uuid4().hex[:12]
    run = Run(
        id=run_id,
        video_id=video.id,
        config_name=config.name,
        config_yaml=config.to_yaml(),
        label=body.label,
        run_dir=str(settings.runs_dir / run_id),
    )
    db.add(run)
    db.add(Job(run_id=run_id))
    db.commit()
    return run


@router.get("", response_model=list[RunOut])
def list_runs(video_id: int | None = None, db: Session = Depends(get_db)):
    q = select(Run).order_by(Run.created_at.desc())
    if video_id is not None:
        q = q.where(Run.video_id == video_id)
    return db.scalars(q).all()


def _detail(run: Run, db: Session) -> RunDetailOut:
    manifest = None
    manifest_path = Path(run.run_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    video = db.get(Video, run.video_id)
    return RunDetailOut(
        **RunOut.model_validate(run).model_dump(),
        config_yaml=run.config_yaml,
        manifest=manifest,
        video=VideoOut.model_validate(video) if video else None,
    )


@router.get("/{run_id}", response_model=RunDetailOut)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return _detail(run, db)


@router.post("/{run_id}/evaluate", response_model=RunDetailOut)
def evaluate_run(run_id: str, oracle_run_id: str | None = None, db: Session = Depends(get_db)):
    """(Re-)score a completed run against its video's ground truth. Writes the
    eval.json artifact and folds headline metrics into run.metrics.

    `oracle_run_id` (optional, SPO-19): a scored run of the SAME video that
    consumed pristine oracle detections; its eval.json upgrades this run's
    ambiguous tracklet-level switch attributions via oracle comparison. The
    oracle run must already carry an eval.json -- evaluate it first."""
    from pitchlab_server.evaluation import evaluate_run_against_gt, merged_metrics

    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    video = db.get(Video, run.video_id)
    if video is None or not video.gt_path:
        raise HTTPException(422, "Video has no ground truth to evaluate against")

    oracle_eval = None
    if oracle_run_id is not None:
        oracle = db.get(Run, oracle_run_id)
        if oracle is None:
            raise HTTPException(404, "Oracle run not found")
        if oracle.video_id != run.video_id:
            raise HTTPException(422, "Oracle run must be a run of the same video")
        oracle_eval_path = Path(oracle.run_dir) / ARTIFACT_FILES[ArtifactName.EVAL]
        if not oracle_eval_path.exists():
            raise HTTPException(
                422, f"Oracle run '{oracle_run_id}' has no eval.json -- evaluate it first"
            )
        oracle_eval = json.loads(oracle_eval_path.read_text())

    try:
        result = evaluate_run_against_gt(
            run, video, oracle_eval=oracle_eval, oracle_run_id=oracle_run_id
        )
    except ImportError as exc:
        raise HTTPException(501, "motmetrics not installed (uv sync --group eval)") from exc
    except ValueError as exc:  # attribution refusals (pitchlab_core.attribution)
        raise HTTPException(422, str(exc)) from exc
    if result is None:
        raise HTTPException(422, "Run has no tracklets artifact to score")
    run.metrics = merged_metrics(run, result)
    db.commit()
    return _detail(run, db)


@router.get("/{run_id}/artifacts/{artifact}")
def get_artifact(run_id: str, artifact: str, db: Session = Depends(get_db)):
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    try:
        name = ArtifactName(artifact)
    except ValueError as exc:
        raise HTTPException(404, f"Unknown artifact '{artifact}'") from exc
    path = Path(run.run_dir) / ARTIFACT_FILES[name]
    if not path.exists():
        raise HTTPException(404, f"Artifact '{artifact}' not produced by this run")
    media = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".mp4": "video/mp4",
    }.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media)


@router.get("/{run_id}/files/{rel_path:path}")
def get_run_file(run_id: str, rel_path: str, db: Session = Depends(get_db)):
    """Generic run-dir file access (identity evidence crops, etc.), jailed to
    the run directory."""
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    root = Path(run.run_dir).resolve()
    target = (root / rel_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target)


@router.get("/{run_a}/diff/{run_b}", response_model=RunDiffOut)
def diff_runs(run_a: str, run_b: str, db: Session = Depends(get_db)):
    a = db.get(Run, run_a)
    b = db.get(Run, run_b)
    if a is None or b is None:
        raise HTTPException(404, "Run not found")
    if a.video_id != b.video_id:
        raise HTTPException(status_code=422, detail="Run diff requires two runs on the same video")

    import yaml

    cfg_a = yaml.safe_load(a.config_yaml)
    cfg_b = yaml.safe_load(b.config_yaml)

    eval_a = _load(a, ArtifactName.EVAL)
    eval_b = _load(b, ArtifactName.EVAL)

    return RunDiffOut(
        run_a=_detail(a, db),
        run_b=_detail(b, db),
        config_changes=_dict_diff(cfg_a, cfg_b),
        metric_deltas=_metric_deltas(a.metrics or {}, b.metrics or {}),
        stats_a=_load(a, ArtifactName.STATS),
        stats_b=_load(b, ArtifactName.STATS),
        timeline_a=_load(a, ArtifactName.TIMELINE),
        timeline_b=_load(b, ArtifactName.TIMELINE),
        eval_a=eval_a,
        eval_b=eval_b,
        switch_diff=diff_switch_instances(eval_a, eval_b),
    )


def _load(run: Run, name: ArtifactName):
    path = Path(run.run_dir) / ARTIFACT_FILES[name]
    if path.exists():
        return json.loads(path.read_text())
    return None


def _dict_diff(a, b, path="") -> list[dict]:
    """Flat list of leaf-level differences between two config dicts."""
    diffs = []
    keys = sorted(set(a or {}) | set(b or {})) if isinstance(a, dict) or isinstance(b, dict) else []
    if keys:
        for k in keys:
            va = (a or {}).get(k)
            vb = (b or {}).get(k)
            sub = f"{path}.{k}" if path else str(k)
            if isinstance(va, dict) or isinstance(vb, dict):
                diffs.extend(_dict_diff(va, vb, sub))
            elif va != vb:
                diffs.append({"path": sub, "a": va, "b": vb})
    elif a != b:
        diffs.append({"path": path, "a": a, "b": b})
    return diffs


def _metric_deltas(ma: dict, mb: dict) -> dict:
    out = {}
    for k in sorted(set(ma) | set(mb)):
        va, vb = ma.get(k), mb.get(k)
        delta = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
        out[k] = {"a": va, "b": vb, "delta": delta}
    return out
