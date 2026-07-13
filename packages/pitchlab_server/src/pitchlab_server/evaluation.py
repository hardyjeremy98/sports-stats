"""Score a completed run against its video's ground truth (when it has any).

Wraps pitchlab_core.evaluation: loads the GT file pointed at by
videos.gt_path, writes the eval.json artifact into the run dir, and returns
the headline metrics to merge into runs.metrics. Requires the `eval`
dependency group (motmetrics); callers treat ImportError as "not installed".
"""

from __future__ import annotations

import json
from pathlib import Path

from pitchlab_server.models import Run, Video


def evaluate_run_against_gt(run: Run, video: Video) -> dict | None:
    """Returns the full eval result dict, or None when the video has no usable
    ground truth or the run produced no tracklets. Raises ImportError when
    motmetrics is missing."""
    from pitchlab_core.evaluation import evaluate_run
    from pitchlab_core.gt import GroundTruth

    if not video.gt_path or not Path(video.gt_path).exists():
        return None
    run_dir = Path(run.run_dir)
    if not (run_dir / "tracklets.json").exists():
        return None

    gt = GroundTruth.model_validate_json(Path(video.gt_path).read_text())
    result = evaluate_run(run_dir, gt)
    (run_dir / "eval.json").write_text(json.dumps(result))
    return result


def merged_metrics(run: Run, result: dict) -> dict:
    from pitchlab_core.evaluation import headline_metrics

    return {**(run.metrics or {}), **headline_metrics(result)}
