"""The pipeline worker: polls the job table, claims a job, executes the run.

Deliberately vendor-free (locked decision #6): any box that can reach the DB
and the data volume can be a worker — locally that's `make worker`, in
production a GPU cloud instance running the same container.
"""

from __future__ import annotations

import json
import socket
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pitchlab_core.config import PipelineConfig
from pitchlab_core.runner import PipelineRunner
from pitchlab_core.schemas.run import StageStatus
from sqlalchemy import select

from pitchlab_server.db import init_db, session
from pitchlab_server.models import Job, JobStatus, QARecord, Run, RunStatus, Video
from pitchlab_server.settings import get_settings

WORKER_ID = f"{socket.gethostname()}-{int(time.time())}"


def claim_next_job() -> int | None:
    """Claim the oldest queued job. `with_for_update(skip_locked=True)` gives
    safe multi-worker claiming on postgres; sqlite (single local worker) just
    ignores the hint."""
    with session() as db:
        job = db.scalars(
            select(Job)
            .where(Job.status == JobStatus.QUEUED)
            .order_by(Job.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).first()
        if job is None:
            return None
        job.status = JobStatus.RUNNING
        job.claimed_by = WORKER_ID
        job.claimed_at = datetime.now(UTC)
        job.attempts += 1
        db.commit()
        return job.id


def execute_job(job_id: int) -> None:
    settings = get_settings()
    with session() as db:
        job = db.get(Job, job_id)
        run = db.get(Run, job.run_id)
        video = db.get(Video, run.video_id)
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(UTC)
        db.commit()
        run_id, run_dir, video_path = run.id, run.run_dir, video.path
        config = PipelineConfig.model_validate(yaml.safe_load(run.config_yaml))

    def progress(kind, frac, msg):
        with session() as db:
            r = db.get(Run, run_id)
            r.progress_stage = kind.value
            r.progress_frac = frac
            r.progress_msg = msg
            db.commit()

    try:
        runner = PipelineRunner(
            run_id=run_id,
            video_path=video_path,
            config=config,
            run_dir=run_dir,
            device=settings.device,
            progress=progress,
        )
        manifest = runner.run()
        ok = manifest.status == StageStatus.COMPLETED
        error = manifest.error
        metrics = manifest.metrics
    except Exception:
        ok, error, metrics = False, traceback.format_exc(), None

    with session() as db:
        job = db.get(Job, job_id)
        run = db.get(Run, job.run_id)
        run.finished_at = datetime.now(UTC)
        run.metrics = metrics
        if ok:
            job.status = JobStatus.DONE
            run.status = RunStatus.COMPLETED
            _import_qa_items(db, run)
            _evaluate_against_gt(db, run)
        else:
            job.status = JobStatus.FAILED
            run.status = RunStatus.FAILED
            run.error = error
        db.commit()


def _evaluate_against_gt(db, run: Run) -> None:
    """Score the run against the video's ground truth, when it has any. Never
    fails the run: eval is diagnostics, not pipeline."""
    from pitchlab_server.evaluation import evaluate_run_against_gt, merged_metrics

    video = db.get(Video, run.video_id)
    try:
        result = evaluate_run_against_gt(run, video)
    except ImportError:
        print(f"[worker {WORKER_ID}] gt eval skipped: motmetrics not installed", flush=True)
        return
    except Exception:
        print(f"[worker {WORKER_ID}] gt eval failed:\n{traceback.format_exc()}", flush=True)
        return
    if result is not None:
        run.metrics = merged_metrics(run, result)
        if result.get("kind") == "action_spotting":
            metric = run.metrics.get("spotting_map_at_1")
            print(f"[worker {WORKER_ID}] gt eval: spotting_map_at_1={metric}", flush=True)
        else:
            print(f"[worker {WORKER_ID}] gt eval: idf1_entity={run.metrics.get('idf1_entity')}", flush=True)


def _import_qa_items(db, run: Run) -> None:
    qa_path = Path(run.run_dir) / "qa_items.json"
    if not qa_path.exists():
        return
    for item in json.loads(qa_path.read_text()):
        db.add(QARecord(run_id=run.id, qa_id=item["qa_id"], payload=item))


def main() -> None:
    settings = get_settings()
    init_db()
    print(f"[worker {WORKER_ID}] polling every {settings.worker_poll_seconds}s", flush=True)
    while True:
        job_id = claim_next_job()
        if job_id is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        print(f"[worker {WORKER_ID}] executing job {job_id}", flush=True)
        execute_job(job_id)
        print(f"[worker {WORKER_ID}] job {job_id} finished", flush=True)


if __name__ == "__main__":
    main()
