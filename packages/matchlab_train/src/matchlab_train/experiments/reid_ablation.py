"""Ablate cross-tracklet association strategies against GT-scored SoccerNet
clips — the measurement harness ADR 004 requires before any association
change ships (GT metrics, not artifact counts).

For each variant (an associate-stage impl + params) this runs the full
pipeline once per clip, scores it against the clip's GT (merge precision,
IDF1 gain, ID-switch delta — see `matchlab_core.evaluation`), then, for
`global-reid` variants, sweeps `max_embed_distance` cheaply by re-running only
the association math against the run's saved `reid_embeddings.npz` (no repeat
detect/track/team/calibrate) and re-scoring against a scratch run dir.

Calibration picks a threshold via `MERGE_PRECISION_GATE`: silent wrong merges
are worse than temporarily unknown identity (locked product invariant), so a
threshold only qualifies if it is provably not making any clip worse (gain
>= 0 everywhere), clears the merge-precision floor pooled across all clips —
never mean-of-ratios, which would let a bad clip hide behind a good one —
*and* has sweep rows for every discovered clip, so a starved clip (no
reid_embeddings.npz) can't silently shrink the evidence base.
"""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

import numpy as np
from matchlab_core.config import PipelineConfig, StageConfig
from matchlab_core.evaluation import evaluate_run, headline_metrics
from matchlab_core.gt import GroundTruth
from matchlab_core.runner import PipelineRunner
from matchlab_core.schemas import PlayerEntity, TeamAssignment, Tracklet
from matchlab_core.schemas.run import StageKind, StageStatus
from matchlab_core.stages.associate.global_reid import GlobalReidAssociator
from pydantic import BaseModel

from matchlab_train.experiments.base import Experiment
from matchlab_train.gt_clips import discover_clips_with_gt
from matchlab_train.registry import register

# Product invariant (see merge_quality's docstring in matchlab_core.evaluation):
# a silent wrong merge is worse than an unmerged tracklet, so calibration must
# gate on pooled merge precision clearing this floor, not just be non-negative
# on average.
MERGE_PRECISION_GATE = 0.90


class Params(BaseModel):
    base_config: str  # pipeline YAML to clone per variant (associate slot swapped)
    clips_dir: str  # dir of .mp4 clips with sibling <stem>.gt.json
    max_clips: int = 8
    device: str = "cuda"
    variants: list[dict]  # [{"name": str, "impl": str, "params": dict}, ...]
    threshold_sweep: list[float] = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    iou_threshold: float = 0.5


@register("reid-ablation")
class ReidAblationExperiment(Experiment):
    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()
        base_cfg = PipelineConfig.from_yaml(p.base_config)

        clip_gt_pairs = discover_clips_with_gt(p.clips_dir, p.max_clips)
        if not clip_gt_pairs:
            raise FileNotFoundError(f"No clips with sibling GT under {p.clips_dir}")

        variant_runs: dict[str, list[dict]] = {}
        # variant name -> threshold -> list of per-clip sweep rows.
        sweep_runs: dict[str, dict[float, list[dict]]] = {}

        for variant in p.variants:
            name, impl, v_params = variant["name"], variant["impl"], variant.get("params", {})
            cfg = _with_associate(base_cfg, impl, v_params)
            rows: list[dict] = []
            sweep_by_threshold: dict[float, list[dict]] = {t: [] for t in p.threshold_sweep}

            for clip, gt in clip_gt_pairs:
                run_id = f"{name}-{clip.stem}"
                run_dir = workdir / "runs" / run_id
                print(f"[reid-ablation] running {name} x {clip.name}", flush=True)
                runner = PipelineRunner(
                    run_id=run_id,
                    video_path=clip,
                    config=cfg,
                    run_dir=run_dir,
                    device=p.device,
                )
                manifest = runner.run()
                row: dict = {"clip": clip.name, "status": manifest.status.value}
                if manifest.status == StageStatus.COMPLETED:
                    result = evaluate_run(run_dir, gt, p.iou_threshold)
                    row.update(_metric_row(result))
                    row["n_tracklets"] = manifest.metrics.get("n_tracklets")
                    row["n_players"] = manifest.metrics.get("n_players")
                    row["stage_durations"] = {
                        s.kind.value: s.duration_s for s in manifest.stages
                    }
                else:
                    row["error"] = (manifest.error or "")[:500]
                rows.append(row)

                npz_path = run_dir / "reid_embeddings.npz"
                if impl == "global-reid" and manifest.status == StageStatus.COMPLETED and npz_path.exists():
                    for t in p.threshold_sweep:
                        print(
                            f"[reid-ablation] sweeping {name} x {clip.name} @ t={t}",
                            flush=True,
                        )
                        sweep_row = _sweep_one(
                            run_dir=run_dir,
                            npz_path=npz_path,
                            base_params=v_params,
                            threshold=t,
                            gt=gt,
                            iou_threshold=p.iou_threshold,
                            scratch_dir=workdir / "sweep" / name / clip.stem / f"t{t}",
                        )
                        sweep_row["clip"] = clip.name
                        sweep_row["threshold"] = t
                        sweep_by_threshold[t].append(sweep_row)

            variant_runs[name] = rows
            if any(sweep_by_threshold.values()):
                sweep_runs[name] = sweep_by_threshold

        summary = _summarize(variant_runs, sweep_runs, n_clips_total=len(clip_gt_pairs))
        result = {
            "base_config": base_cfg.name,
            "clips": [clip.name for clip, _ in clip_gt_pairs],
            "variants": variant_runs,
            "sweep": sweep_runs,
            "summary": summary,
        }
        self.write_result(workdir, result)
        return result


def _with_associate(base_cfg: PipelineConfig, impl: str, params: dict) -> PipelineConfig:
    """A deep copy of `base_cfg` with only the associate slot swapped to
    `impl`/`params` — every other stage's config is untouched."""
    cfg = base_cfg.model_copy(deep=True)
    cfg.stages = {**cfg.stages, StageKind.ASSOCIATE: StageConfig(impl=impl, params=params)}
    return cfg


def _metric_row(result: dict) -> dict:
    """The subset of an evaluate_run() result the ablation tracks per (variant,
    clip[, threshold]) row: headline IDF1/merge numbers plus the raw pair
    counts pooling needs (headline_metrics rounds merge_precision and drops
    the counts entirely)."""
    heads = headline_metrics(result)
    assoc = result["association"]
    return {
        "idf1_tracklet": heads["idf1_tracklet"],
        "idf1_entity": heads["idf1_entity"],
        "assoc_idf1_gain": assoc["idf1_gain"],
        "idsw_delta": assoc["idsw_delta"],
        "merge_precision": assoc["merge_precision"],
        "n_pairs": assoc["n_pairs"],
        "n_pairs_correct": assoc["n_pairs_correct"],
        "n_pairs_unmatched": assoc["n_pairs_unmatched"],
    }


def _sweep_one(
    run_dir: Path,
    npz_path: Path,
    base_params: dict,
    threshold: float,
    gt: GroundTruth,
    iou_threshold: float,
    scratch_dir: Path,
) -> dict:
    """Re-associate one run's saved tracklets/teams/embeddings at a different
    `max_embed_distance`, without repeating detect/track/team/calibrate, and
    score the result. `write_report=False` — the sweep only needs the
    resulting entities, not a full association.json decision trail."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fps = float(manifest["video"]["fps"])

    tracklets = [Tracklet.model_validate(t) for t in json.loads((run_dir / "tracklets.json").read_text())]
    teams = [TeamAssignment.model_validate(t) for t in json.loads((run_dir / "teams.json").read_text())]

    associator = GlobalReidAssociator(**{**base_params, "max_embed_distance": threshold, "save_embeddings": False})

    npz = np.load(npz_path)
    # Provenance gate: the sweep scores the npz's embeddings as if this
    # variant produced them. If the npz was written by a different embedder,
    # the run dir isn't what the sweep thinks it is — refuse to score it.
    meta = json.loads(npz["meta"].item()) if "meta" in npz else {}
    npz_embedder = meta.get("embedder")
    if npz_embedder is not None and npz_embedder != associator.params.embedder:
        raise RuntimeError(
            f"Embedder provenance mismatch in {npz_path}: npz was written by "
            f"'{npz_embedder}' but the variant is configured with "
            f"'{associator.params.embedder}'."
        )
    feats = {int(tid): emb for tid, emb in zip(npz["tracklet_ids"], npz["embeddings"], strict=True)}

    entities = associator._associate_with_features(
        None, tracklets, teams, feats, fps=fps, write_report=False
    )

    _rescore_with_players(run_dir, scratch_dir, entities)
    result = evaluate_run(scratch_dir, gt, iou_threshold)
    row = _metric_row(result)
    row["embedder"] = associator.params.embedder
    return row


def _rescore_with_players(run_dir: Path, scratch_dir: Path, entities: list[PlayerEntity]) -> Path:
    """Build a minimal run dir `evaluate_run` can score without a full
    pipeline re-run: copy the shared manifest.json + tracklets.json (detection
    and tracking are identical across the sweep — only association changed)
    and write the new entities as players.json in the exact format
    ArtifactStore.write_json uses, so evaluate_run reads it the same way it
    would a real run's output."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "manifest.json", scratch_dir / "manifest.json")
    shutil.copy(run_dir / "tracklets.json", scratch_dir / "tracklets.json")
    (scratch_dir / "players.json").write_text(
        json.dumps([e.model_dump(mode="json") for e in entities])
    )
    return scratch_dir


def _aggregate(rows: list[dict]) -> dict:
    """Mean/median assoc_idf1_gain, total idsw_delta, POOLED merge precision
    (sum of correct pairs / sum of all pairs — never a mean of per-clip
    ratios, which would hide a badly-wrong clip behind others), and the
    per-clip table, over one variant's (or one variant+threshold's) rows."""
    gains = [r["assoc_idf1_gain"] for r in rows]
    total_pairs = sum(r["n_pairs"] for r in rows)
    total_correct = sum(r["n_pairs_correct"] for r in rows)
    return {
        "n_clips": len(rows),
        "mean_assoc_idf1_gain": round(statistics.mean(gains), 4) if gains else None,
        "median_assoc_idf1_gain": round(statistics.median(gains), 4) if gains else None,
        "total_idsw_delta": sum(r["idsw_delta"] for r in rows),
        "pooled_merge_precision": round(total_correct / total_pairs, 4) if total_pairs else None,
        "total_n_pairs": total_pairs,
        "per_clip": rows,
    }


def _calibrate(sweep_by_threshold: dict[float, list[dict]], n_clips_total: int) -> dict:
    """Largest threshold whose pooled merge precision clears
    MERGE_PRECISION_GATE, has non-negative assoc_idf1_gain on every clip
    (never makes any clip worse), *and* covers ALL discovered clips — a clip
    whose reid_embeddings.npz never materialized (e.g. every tracklet starved
    on crops) drops out of the sweep rows, and a threshold judged on that
    shrunken subset must not qualify.

    Returns {"threshold": float|None, "n_clips_covered": int,
    "n_clips_total": int}; threshold is None when nothing qualifies, with
    coverage still reported from the best-covered threshold so a starved
    variant is diagnosable at a glance."""
    best: float | None = None
    best_covered = 0
    for t in sorted(sweep_by_threshold):
        rows = sweep_by_threshold[t]
        if not rows:
            continue
        best_covered = max(best_covered, len(rows))
        if len(rows) != n_clips_total:
            continue
        total_pairs = sum(r["n_pairs"] for r in rows)
        total_correct = sum(r["n_pairs_correct"] for r in rows)
        pooled = (total_correct / total_pairs) if total_pairs else None
        gate_ok = pooled is not None and pooled >= MERGE_PRECISION_GATE
        gain_ok = all(r["assoc_idf1_gain"] >= 0 for r in rows)
        if gate_ok and gain_ok:
            best = t
    return {
        "threshold": best,
        "n_clips_covered": n_clips_total if best is not None else best_covered,
        "n_clips_total": n_clips_total,
    }


def _summarize(
    variant_runs: dict[str, list[dict]],
    sweep_runs: dict[str, dict[float, list[dict]]],
    n_clips_total: int,
) -> dict:
    return {
        "variants": {
            name: _aggregate(rows) for name, rows in variant_runs.items() if rows
        },
        "sweep": {
            name: {t: _aggregate(rows) for t, rows in sweep.items() if rows}
            for name, sweep in sweep_runs.items()
        },
        "calibration": {
            name: _calibrate(sweep, n_clips_total) for name, sweep in sweep_runs.items()
        },
    }
