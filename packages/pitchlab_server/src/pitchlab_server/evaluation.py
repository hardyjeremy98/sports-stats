"""Score a completed run against its video's ground truth (when it has any).

Wraps pitchlab_core.evaluation: loads the GT file pointed at by
videos.gt_path, writes the eval.json artifact into the run dir, and returns
the headline metrics to merge into runs.metrics. Requires the `eval`
dependency group (motmetrics); callers treat ImportError as "not installed".
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pitchlab_core.attribution import match_instances

from pitchlab_server.models import Run, Video


def evaluate_run_against_gt(
    run: Run,
    video: Video,
    oracle_eval: dict | None = None,
    oracle_run_id: str | None = None,
) -> dict | None:
    """Returns the full eval result dict, or None when the video has no usable
    ground truth or the run produced no tracklets. Raises ImportError when
    motmetrics is missing.

    Before scoring, records the evaluation-set's identity (SPO-10) into the
    run's manifest.json provenance block: every GT-scored run must carry a
    hash of the exact ground truth it was measured against, so a benchmark
    runner comparing runs months apart can tell whether they used the same
    evaluation set.

    `oracle_eval`/`oracle_run_id` (SPO-19): an already-scored eval.json
    payload from a pristine oracle-detections run of the same video, used to
    upgrade ambiguous tracklet-level switch attributions via oracle
    comparison; refusals (`pitchlab_core.attribution`) raise ValueError."""
    from pitchlab_core.evaluation import evaluate_run
    from pitchlab_core.gt import GroundTruth
    from pitchlab_core.provenance import hash_evaluation_set
    from pitchlab_core.schemas.run import RunManifest

    if not video.gt_path or not Path(video.gt_path).exists():
        return None
    run_dir = Path(run.run_dir)
    if not (run_dir / "tracklets.json").exists():
        return None

    gt_path = Path(video.gt_path)
    gt_text = gt_path.read_text()
    gt = GroundTruth.model_validate_json(gt_text)

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
        manifest.provenance.evaluation_set_hash = hash_evaluation_set(gt_text)
        manifest.provenance.evaluation_set_source = str(gt_path)
        manifest_path.write_text(manifest.model_dump_json(indent=2))

    result = evaluate_run(run_dir, gt)
    if oracle_eval is not None:
        from pitchlab_core.attribution import attribute_switches

        attribute_switches(result, oracle_eval=oracle_eval, oracle_run_id=oracle_run_id)
    (run_dir / "eval.json").write_text(json.dumps(result))
    return result


def merged_metrics(run: Run, result: dict) -> dict:
    from pitchlab_core.evaluation import headline_metrics

    return {**(run.metrics or {}), **headline_metrics(result)}


def diff_switch_instances(
    eval_a: dict | None, eval_b: dict | None, tol_s: float = 1.0
) -> dict | None:
    """Diff two eval.json payloads' `instances` (ID-switch records) for the
    Lab's diff view: which switches were fixed, introduced, or persisted
    between run A and run B.

    Instances are grouped by (level, gt_track_id); within each group, A- and
    B-instances are greedily matched by nearest `t` via
    `pitchlab_core.attribution.match_instances` (the same matcher the SPO-19
    layer-attribution pass uses), closest pairs first, each instance matched
    at most once. Matches farther apart than `tol_s` don't count. Returns
    None when either eval payload is missing or has no `instances` key.
    """
    if eval_a is None or eval_b is None:
        return None
    if "instances" not in eval_a or "instances" not in eval_b:
        return None

    def _grouped(instances: list[dict]) -> dict[tuple, list[dict]]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for inst in instances:
            groups[(inst.get("level"), inst.get("gt_track_id"))].append(inst)
        return groups

    groups_a = _grouped(eval_a["instances"])
    groups_b = _grouped(eval_b["instances"])

    fixed: list[dict] = []
    introduced: list[dict] = []
    persisted: list[dict] = []

    for key in dict.fromkeys([*groups_a, *groups_b]):
        a_list = groups_a.get(key, [])
        b_list = groups_b.get(key, [])

        pairs = match_instances(a_list, b_list, tol_s)
        matched_a = {i for i, _ in pairs}
        matched_b = {j for _, j in pairs}
        persisted.extend({"a": a_list[i], "b": b_list[j]} for i, j in pairs)

        fixed.extend(a_inst for i, a_inst in enumerate(a_list) if i not in matched_a)
        introduced.extend(b_inst for j, b_inst in enumerate(b_list) if j not in matched_b)

    return {
        "fixed": fixed,
        "introduced": introduced,
        "persisted": persisted,
        "counts": {
            "fixed": len(fixed),
            "introduced": len(introduced),
            "persisted": len(persisted),
        },
    }
