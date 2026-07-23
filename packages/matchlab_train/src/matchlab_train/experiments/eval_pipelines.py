"""Pipeline A/B evaluation: run two pipeline configs over a set of clips and
report comparative metrics. This is the CLI counterpart of the Lab's diff view
— useful for batch regression checks when a stage implementation changes.

Dependency-free (uses matchlab-core directly), so it doubles as the smoke-test
experiment for the training package itself.
"""

from __future__ import annotations

import statistics
import uuid
from pathlib import Path

from matchlab_core.config import PipelineConfig
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas.run import StageStatus
from pydantic import BaseModel

from matchlab_train.experiments.base import Experiment
from matchlab_train.gt_clips import load_sibling_gt
from matchlab_train.registry import register


class Params(BaseModel):
    config_a: str  # paths to pipeline YAMLs
    config_b: str
    clips_dir: str
    max_clips: int = 5
    device: str = "cpu"


@register("eval-pipelines")
class EvalPipelinesExperiment(Experiment):
    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()
        cfg_a = PipelineConfig.from_yaml(p.config_a)
        cfg_b = PipelineConfig.from_yaml(p.config_b)

        clips = sorted(Path(p.clips_dir).glob("*.mp4"))[: p.max_clips]
        if not clips:
            raise FileNotFoundError(f"No .mp4 clips in {p.clips_dir}")

        per_clip = []
        for clip in clips:
            row = {"clip": clip.name}
            # Not every clip is labelled; when it is, fold GT metrics in per
            # ADR 004 (measure against ground truth, not just artifact counts).
            gt = load_sibling_gt(clip)
            for tag, cfg in (("a", cfg_a), ("b", cfg_b)):
                run_id = f"{tag}-{uuid.uuid4().hex[:8]}"
                runner = PipelineRunner(
                    run_id=run_id,
                    video_path=clip,
                    config=cfg,
                    run_dir=workdir / "runs" / f"{clip.stem}-{tag}",
                    device=p.device,
                )
                manifest = runner.run()
                row[tag] = {
                    "status": manifest.status.value,
                    "metrics": manifest.metrics,
                    "stage_durations": {
                        s.kind.value: s.duration_s for s in manifest.stages
                    },
                }
                if manifest.status != StageStatus.COMPLETED:
                    row[tag]["error"] = (manifest.error or "")[:500]
                elif gt is not None:
                    from matchlab_core.evaluation import evaluate_run, headline_metrics

                    result = evaluate_run(runner.store.run_dir, gt)
                    row[tag]["gt_metrics"] = headline_metrics(result)
            per_clip.append(row)

        summary = _summarize(per_clip)
        result = {
            "config_a": cfg_a.name,
            "config_b": cfg_b.name,
            "clips": per_clip,
            "summary": summary,
        }
        self.write_result(workdir, result)
        return result


def _summarize(per_clip: list[dict]) -> dict:
    keys = ("n_tracklets", "n_players", "n_events", "n_qa_items")
    sums: dict[str, dict[str, float]] = {t: dict.fromkeys(keys, 0.0) for t in ("a", "b")}
    # Per-tag, per-metric-name list of values collected across clips that had
    # GT (a subset of the completed clips; only present when a sibling
    # <clip stem>.gt.json existed for that clip).
    gt_values: dict[str, dict[str, list[float]]] = {"a": {}, "b": {}}
    n = 0
    for row in per_clip:
        if row["a"]["status"] != "completed" or row["b"]["status"] != "completed":
            continue
        n += 1
        for tag in ("a", "b"):
            for k in keys:
                sums[tag][k] += row[tag]["metrics"].get(k, 0)
            for mk, mv in (row[tag].get("gt_metrics") or {}).items():
                if isinstance(mv, (int, float)):
                    gt_values[tag].setdefault(mk, []).append(mv)
    if n == 0:
        return {"completed_pairs": 0}
    summary: dict = {
        "completed_pairs": n,
        "mean": {
            tag: {k: round(v / n, 2) for k, v in vals.items()}
            for tag, vals in sums.items()
        },
    }
    gt_summary = {
        tag: {
            mk: {"mean": round(statistics.mean(vs), 4), "median": round(statistics.median(vs), 4)}
            for mk, vs in metrics.items()
        }
        for tag, metrics in gt_values.items()
        if metrics
    }
    if gt_summary:
        summary["gt_metrics"] = gt_summary
    return summary
