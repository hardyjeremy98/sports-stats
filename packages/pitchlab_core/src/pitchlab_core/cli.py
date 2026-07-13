"""`pitchlab-run` — run a pipeline config against a video from the shell.

The server/worker is the production path; this CLI is for Lab-style local
iteration and CI smoke tests.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from pitchlab_core.config import PipelineConfig
from pitchlab_core.runner import PipelineRunner
from pitchlab_core.schemas.run import StageStatus


def main() -> int:
    parser = argparse.ArgumentParser(prog="pitchlab-run")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--out", default="data/runs", help="Runs directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:12]
    config = PipelineConfig.from_yaml(args.config)
    run_dir = Path(args.out) / run_id

    def progress(kind, frac, msg):
        print(f"[{run_id}] {msg}", flush=True)

    runner = PipelineRunner(
        run_id=run_id,
        video_path=args.video,
        config=config,
        run_dir=run_dir,
        device=args.device,
        progress=progress,
    )
    manifest = runner.run()
    print(f"run {run_id}: {manifest.status} -> {run_dir}")
    if manifest.status == StageStatus.FAILED:
        print(manifest.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
