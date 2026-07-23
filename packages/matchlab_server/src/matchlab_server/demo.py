"""`matchlab-demo` — seed the local environment with a working demo:

1. render a synthetic match video (matchlab_core.demo),
2. register it as an uploaded Video,
3. execute TWO runs inline (stub config + a tracker-variant of it) so the Lab's
   run-diff view has something real to show,
4. import QA items.

No API keys, no GPU, ~a minute of CPU. Run via `make demo`.
"""

from __future__ import annotations

import argparse
import uuid

import yaml
from matchlab_core.config import PipelineConfig
from matchlab_core.demo import render_demo_video
from matchlab_core.video import probe

from matchlab_server.db import init_db, session
from matchlab_server.models import Job, JobStatus, Run, RunStatus, Video
from matchlab_server.settings import get_settings
from matchlab_server.worker import execute_job


def main() -> None:
    parser = argparse.ArgumentParser(prog="matchlab-demo")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--skip-runs", action="store_true", help="only create the video")
    args = parser.parse_args()

    settings = get_settings()
    init_db()

    print("rendering synthetic match video...", flush=True)
    path = settings.videos_dir / "demo_match.mp4"
    render_demo_video(path, duration_s=args.duration, fps=25)
    meta = probe(path)

    with session() as db:
        existing = db.query(Video).filter(Video.filename == "demo_match.mp4").first()
        if existing:
            video = existing
            video.path = str(path)
        else:
            video = Video(filename="demo_match.mp4", path=str(path))
            db.add(video)
        video.size_bytes = path.stat().st_size
        video.duration_s = meta.duration_s
        video.fps = meta.fps
        video.width = meta.width
        video.height = meta.height
        db.commit()
        video_id = video.id
    print(f"registered video #{video_id} ({meta.duration_s:.0f}s)", flush=True)

    if args.skip_runs:
        return

    base = PipelineConfig.from_yaml(settings.config_dir / "pipeline.stub.yaml")

    # Variant with laxer tracking — a genuinely different tracker config so the
    # diff view shows real deltas (more fragmentation, different stats).
    variant_dict = yaml.safe_load(base.to_yaml())
    variant_dict["name"] = "stub-loose-tracking"
    variant_dict["description"] = "Stub recipe with a stricter, fragmentation-prone tracker setup."
    variant_dict["stages"]["track"]["params"] = {"iou_threshold": 0.45, "max_age_frames": 4}
    variant = PipelineConfig.model_validate(variant_dict)

    for config, label in [(base, "demo: baseline"), (variant, "demo: loose tracker")]:
        run_id = uuid.uuid4().hex[:12]
        with session() as db:
            run = Run(
                id=run_id,
                video_id=video_id,
                config_name=config.name,
                config_yaml=config.to_yaml(),
                label=label,
                status=RunStatus.QUEUED,
                run_dir=str(settings.runs_dir / run_id),
            )
            db.add(run)
            db.add(Job(run_id=run_id, status=JobStatus.QUEUED))
            db.commit()
            job_id = db.query(Job).filter(Job.run_id == run_id).one().id
        print(f"executing run {run_id} ({config.name})...", flush=True)
        execute_job(job_id)
        with session() as db:
            print(f"  -> {db.get(Run, run_id).status.value}", flush=True)

    print("demo ready: open the Lab at http://localhost:5173/lab", flush=True)


if __name__ == "__main__":
    main()
