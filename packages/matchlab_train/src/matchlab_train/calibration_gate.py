"""Gate 1 harness: reproduce PnLCalib's published SoccerNet-Calibration
test-split accuracy under the official challenge protocol (SPO-60 Gate 1,
authored under SPO-67).

This module owns the *in-repo, testable* half of the gate:

* ``aggregate_camera_metrics`` -- the split-level aggregation into the official
  metrics (completeness x mean per-image Jaccard = final score), matching
  ``sn-calibration/src/evaluate_camera.py``'s aggregation exactly;
* ``build_gate_record`` -- compare the measured metrics against PnLCalib's
  published numbers with a tolerance and produce a gate-record dict;
* ``run_gate1_calibration_eval`` -- orchestrate the two isolated subprocesses
  (predictor + official scorer) and write the gate record + markdown summary.

Design decision (vendor vs. subprocess): the official SoccerNet-Calibration
evaluator (github.com/SoccerNet/sn-calibration) ships **no LICENSE file**, so it
is reached only as a subprocess from a documented sibling checkout -- the same
dependency-isolation posture used for the PnLCalib predictor and the T-DEED
spotter -- rather than copied into this tree. Its geometry-exact per-image
scoring (project the 3D pitch template through predicted camera parameters,
count polyline TP/FP/FN at a pixel threshold) stays behind a thin scoring
adapter (``docs/reference/adapters/sn_calibration_eval_cli.py``); what this
module does with its output -- the split aggregation and the published-number
comparison -- is the official metric formula and is unit-tested here.

Exchange contracts (both documented in
``docs/reference/external-calibrators-setup.md``):

* predictor: ``<predictor_command> --job <manifest.json>`` where the manifest is
  ``{"frames_dir", "fps", "out_dir", "params": {"mode": "camera", ...}}`` and
  the predictor writes one ``camera_<frame_id>.json`` (SoccerNet camera-param
  schema) per image into ``out_dir``. This is the calibration job-manifest
  contract from ``matchlab_core.calib.bridge`` extended with a ``camera`` output
  mode (the official evaluator consumes camera parameters, not a homography, so
  a camera-output mode is required -- see the setup doc).
* scorer: ``<scorer_command> --gt-dir <dir> --pred-dir <dir>
  --thresholds 5,10,20 --width W --height H --out <confusions.json>`` where the
  scorer writes a JSON array of per-image confusion records
  ``{"image", "has_prediction", "per_threshold": {"5": {"tp","fp","fn"}, ...}}``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# PnLCalib's published SoccerNet-Calibration test-split (SN23-test) numbers.
# Source: Gutiérrez-Pérez & Agudo, "PnLCalib: Sports Field Registration via
# Points and Lines Optimization", arXiv:2404.08401v5, Table I (row
# "Ours_MV + PnL" on SN23-test) and the P/L ablation row (P✓ L✓). JaC_gamma is
# the per-image Jaccard at gamma pixels; CR is completeness; FS is final score.
#
# Provenance caveat: this SN23-test headline uses the MULTI-VIEW weights
# (MV_kp/MV_lines). The single-view base pair (SV_kp/SV_lines) that the
# `pnlcalib` calibrate stage defaults to corresponds to the paper's single-view
# rows (evaluated on WC14-style data, completeness ~99%), NOT this number. To
# reproduce 78.7 JaC@5 / 61.8 FS on SN23-test, use the MV weights.
PNLCALIB_SN23_TEST_PUBLISHED: dict[str, Any] = {
    "method": "PnLCalib (multi-view MV_kp/MV_lines weights + PnL refinement)",
    "dataset": "SoccerNet-Calibration SN23 test split",
    "source": (
        "Gutiérrez-Pérez & Agudo, arXiv:2404.08401v5, Table I (SN23-test, "
        "Ours_MV + PnL) and the points+lines ablation row (P✓ L✓)"
    ),
    "jac": {5: 0.787, 10: 0.896, 20: 0.919},
    "completeness": 0.784,
    "final_score": 0.618,
    "weights_note": (
        "SN23-test headline uses multi-view MV weights; the pnlcalib stage's "
        "default single-view SV weights map to the paper's WC14-style single-view "
        "rows, not this number."
    ),
}


def _confusion_at(record: dict, threshold: int) -> dict:
    per_threshold = record.get("per_threshold", {})
    # JSON object keys are strings; tolerate int keys too.
    if str(threshold) in per_threshold:
        return per_threshold[str(threshold)]
    if threshold in per_threshold:
        return per_threshold[threshold]
    raise KeyError(f"per-image confusion record missing threshold {threshold}: {record!r}")


def aggregate_camera_metrics(
    per_image: list[dict],
    total_images: int,
    thresholds: list[int],
) -> dict[int, dict[str, float]]:
    """Aggregate per-image confusion records into the official camera-calibration
    metrics, per threshold.

    Matches ``sn-calibration/src/evaluate_camera.py``:

    * per-image accuracy (JaC@t) = ``tp / (tp + fp + fn)`` (0.0 when the
      denominator is 0), averaged over images that HAVE a prediction;
    * completeness = ``n_predicted / total_images`` (threshold-independent, but
      reported at each threshold for convenience);
    * final score = completeness x mean accuracy.

    Images without a prediction are excluded from the accuracy mean but count
    against completeness -- the official loop skips a missing prediction file
    without appending an accuracy.
    """
    if total_images <= 0:
        raise ValueError(f"total_images must be positive, got {total_images}")

    n_predicted = sum(1 for r in per_image if r.get("has_prediction"))
    completeness = n_predicted / total_images

    out: dict[int, dict[str, float]] = {}
    for threshold in thresholds:
        accuracies: list[float] = []
        for record in per_image:
            if not record.get("has_prediction"):
                continue
            conf = _confusion_at(record, threshold)
            tp = float(conf["tp"])
            denom = tp + float(conf["fp"]) + float(conf["fn"])
            accuracies.append(tp / denom if denom > 0 else 0.0)
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        out[threshold] = {
            "jac": mean_accuracy,
            "completeness": completeness,
            "final_score": completeness * mean_accuracy,
            "n_evaluated": len(accuracies),
            "n_total": total_images,
        }
    return out


def build_gate_record(
    measured: dict[int, dict[str, float]],
    *,
    total_images: int,
    split: str,
    thresholds: list[int],
    tolerance: float = 0.03,
    published: dict[str, Any] = PNLCALIB_SN23_TEST_PUBLISHED,
    predictor_command: list[str] | None = None,
    scorer_command: list[str] | None = None,
) -> dict[str, Any]:
    """Compare measured metrics against the published targets and assemble the
    gate-record dict.

    A metric is ``within_tolerance`` when it meets or exceeds the published value
    minus ``tolerance`` (reproducing or beating the paper is a pass). The gate
    ``passed`` when JaC@5 and final-score@5 are both within tolerance -- JaC@5
    and FS are the challenge's headline metrics.
    """
    comparison: dict[str, dict[str, Any]] = {}
    published_jac = published.get("jac", {})
    for threshold in thresholds:
        m = measured[threshold]
        entry: dict[str, Any] = {
            "measured_jac": m["jac"],
            "measured_completeness": m["completeness"],
            "measured_final_score": m["final_score"],
        }
        pub_jac = published_jac.get(threshold)
        if pub_jac is not None:
            entry["published_jac"] = pub_jac
            entry["jac_delta"] = m["jac"] - pub_jac
            entry["jac_within_tolerance"] = m["jac"] >= pub_jac - tolerance
        comparison[str(threshold)] = entry

    # Final-score comparison is a threshold-5 concept (JaC@5 x completeness).
    fs_within = None
    if 5 in measured and published.get("final_score") is not None:
        pub_fs = published["final_score"]
        comparison.setdefault("5", {})
        comparison["5"]["published_final_score"] = pub_fs
        comparison["5"]["final_score_delta"] = measured[5]["final_score"] - pub_fs
        fs_within = measured[5]["final_score"] >= pub_fs - tolerance
        comparison["5"]["final_score_within_tolerance"] = fs_within

    jac5_within = comparison.get("5", {}).get("jac_within_tolerance")
    passed = bool(jac5_within) and bool(fs_within)

    return {
        "gate": "SPO-60 Gate 1: reproduce PnLCalib SoccerNet-Calibration accuracy",
        "created": datetime.now(UTC).isoformat(),
        "split": split,
        "n_images": total_images,
        "thresholds": list(thresholds),
        "tolerance": tolerance,
        "measured": {str(t): measured[t] for t in thresholds},
        "published": published,
        "comparison": comparison,
        "passed": passed,
        "predictor_command": predictor_command,
        "scorer_command": scorer_command,
    }


def _render_markdown(record: dict[str, Any]) -> str:
    published = record["published"]
    lines = [
        "# Gate 1: SoccerNet-Calibration eval",
        "",
        f"**Gate:** {record['gate']}  ",
        f"**Created:** {record['created']}  ",
        f"**Split:** {record['split']}  ",
        f"**Images:** {record['n_images']}  ",
        f"**Result:** {'PASSED' if record['passed'] else 'NOT PASSED'}  ",
        f"**Tolerance:** {record['tolerance']:.3f} (absolute, on the [0,1] metric)",
        "",
        f"Published reference: **{published['method']}** on {published['dataset']}.  ",
        f"Source: {published['source']}.",
        "",
        f"> {published['weights_note']}",
        "",
        "| Threshold (px) | Measured JaC | Published JaC | Δ JaC | Within tol |",
        "| --- | --- | --- | --- | --- |",
    ]
    for threshold in record["thresholds"]:
        cmp = record["comparison"].get(str(threshold), {})
        pub = cmp.get("published_jac")
        delta = cmp.get("jac_delta")
        within = cmp.get("jac_within_tolerance")
        measured_jac = cmp.get("measured_jac", float("nan"))
        pub_cell = f"{pub * 100:.1f}%" if pub is not None else "n/a"
        delta_cell = f"{delta * 100:+.1f} pts" if delta is not None else "n/a"
        within_cell = "n/a" if within is None else ("yes" if within else "no")
        lines.append(
            f"| {threshold} | {measured_jac * 100:.1f}% | {pub_cell} "
            f"| {delta_cell} | {within_cell} |"
        )
    m5 = record["measured"].get("5")
    if m5 is not None:
        lines += [
            "",
            f"Completeness: **{m5['completeness'] * 100:.1f}%** "
            f"(published {published['completeness'] * 100:.1f}%)  ",
            f"Final score (JaC@5 × completeness): **{m5['final_score'] * 100:.1f}%** "
            f"(published {published['final_score'] * 100:.1f}%)",
        ]
    lines += [
        "",
        "Published targets: "
        f"JaC@5 = {published['jac'][5] * 100:.1f}%, "
        f"completeness = {published['completeness'] * 100:.1f}%, "
        f"final score = {published['final_score'] * 100:.1f}%.",
        "",
    ]
    return "\n".join(lines)


def _run(command: list[str], *, args: list[str], timeout_s: float) -> None:
    try:
        result = subprocess.run(
            [*command, *args], capture_output=True, text=True, timeout=timeout_s
        )
    except OSError as exc:
        raise RuntimeError(f"failed to launch command {command!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command {command!r} timed out after {timeout_s}s") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"command {command!r} exited {result.returncode}: {result.stderr.strip()}"
        )


def run_gate1_calibration_eval(
    *,
    soccernet_dir: str | Path,
    split: str,
    thresholds: list[int],
    out_dir: str | Path,
    scorer_command: list[str],
    predictor_command: list[str] | None = None,
    prediction_dir: str | Path | None = None,
    tolerance: float = 0.03,
    fps: float = 25.0,
    resolution_width: int = 960,
    resolution_height: int = 540,
    predictor_params: dict[str, Any] | None = None,
    timeout_s: float = 24 * 3600.0,
) -> dict[str, Any]:
    """Run the Gate 1 SoccerNet-Calibration evaluation end to end and write the
    gate record (JSON + markdown) under ``out_dir``.

    Either ``predictor_command`` (drive the predictor over the split to produce
    ``camera_<id>.json`` predictions) or ``prediction_dir`` (use precomputed
    predictions) must be given. The official scorer is always invoked via
    ``scorer_command``. Returns the gate-record dict.
    """
    soccernet_dir = Path(soccernet_dir)
    out_dir = Path(out_dir)
    split_dir = soccernet_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"SoccerNet-Calibration split directory not found: {split_dir}"
        )

    # Per-image GT is <id>.json with a numeric id; splits also ship metadata
    # files (match_info.json, per_match_info.json) that must not enter the
    # completeness denominator.
    gt_files = sorted(p for p in split_dir.glob("*.json") if p.stem.isdigit())
    if not gt_files:
        raise FileNotFoundError(
            f"no ground-truth annotation JSON files in {split_dir}"
        )
    total_images = len(gt_files)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Predictions: drive the predictor subprocess, or use a precomputed dir.
    if predictor_command is not None:
        pred_dir = out_dir / "predictions" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "frames_dir": str(split_dir),
            "fps": fps,
            "out_dir": str(pred_dir),
            "params": {"mode": "camera", "split": split, **(predictor_params or {})},
        }
        manifest_path = out_dir / "predict_job.json"
        manifest_path.write_text(json.dumps(manifest))
        _run(predictor_command, args=["--job", str(manifest_path)], timeout_s=timeout_s)
    elif prediction_dir is not None:
        pred_dir = Path(prediction_dir)
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"prediction_dir does not exist: {pred_dir}")
    else:
        raise ValueError("one of predictor_command or prediction_dir is required")

    # 2. Official scoring (isolated subprocess) -> per-image confusions JSON.
    confusions_path = out_dir / "per_image_confusions.json"
    confusions_path.unlink(missing_ok=True)
    _run(
        scorer_command,
        args=[
            "--gt-dir", str(split_dir),
            "--pred-dir", str(pred_dir),
            "--thresholds", ",".join(str(t) for t in thresholds),
            "--width", str(resolution_width),
            "--height", str(resolution_height),
            "--out", str(confusions_path),
        ],
        timeout_s=timeout_s,
    )
    if not confusions_path.exists():
        raise RuntimeError(
            f"scorer command {scorer_command!r} exited 0 but did not write "
            f"{confusions_path}"
        )
    per_image = json.loads(confusions_path.read_text())
    if not isinstance(per_image, list):
        raise RuntimeError(
            f"scorer command {scorer_command!r} wrote {confusions_path}, but it is "
            "not a JSON array of per-image confusion records"
        )

    # 3. Aggregate into official metrics and compare against published numbers.
    measured = aggregate_camera_metrics(per_image, total_images, thresholds)
    record = build_gate_record(
        measured,
        total_images=total_images,
        split=split,
        thresholds=thresholds,
        tolerance=tolerance,
        predictor_command=predictor_command,
        scorer_command=scorer_command,
    )

    (out_dir / "gate1_calibration.json").write_text(json.dumps(record, indent=2))
    (out_dir / "gate1_calibration.md").write_text(_render_markdown(record))
    return record
