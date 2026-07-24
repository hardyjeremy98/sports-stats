#!/usr/bin/env python3
"""SoccerNet-Calibration -> MatchDay Gate 1 scoring adapter.

Copy this file into the sibling ``external-calibrators/`` directory beside a
clone of the official evaluator (``sn-calibration/``) and run it with an
environment that has the evaluator's dependencies installed. It is the ONLY
code that executes inside that checkout on the lab's behalf; nothing under
``lab/packages/`` ever imports it. The lab side
(``matchlab_train.calibration_gate.run_gate1_calibration_eval``) invokes it as a
subprocess::

    <python> sn_calibration_eval_cli.py \
        --gt-dir <soccernet>/test --pred-dir <predictions> \
        --thresholds 5,10,20 --width 960 --height 540 --out confusions.json

Why a subprocess and not vendored: the official evaluator
(github.com/SoccerNet/sn-calibration) ships **no LICENSE file**, so it is
reached only from an isolated sibling checkout -- the same dependency-isolation
posture as PnLCalib and T-DEED -- rather than copied into the lab tree.

What it does, per the Gate 1 scoring contract:

* For every ``<frame_id>.json`` ground-truth annotation in ``--gt-dir``, load the
  matching predicted ``camera_<frame_id>.json`` from ``--pred-dir`` (absent =>
  ``has_prediction: false``).
* Reproject the 3D pitch model through the predicted camera parameters
  (official ``get_polylines``) and count per-line true/false
  positives/negatives at each pixel threshold (official
  ``evaluate_camera_prediction``), evaluating against both the annotation and
  its centre-mirror and keeping the higher-accuracy assignment -- exactly what
  ``src/evaluate_camera.py`` does per image.
* Emit a JSON array of per-image records the lab aggregates into the official
  metrics (completeness x mean per-image Jaccard = final score)::

      [{"image": "00001", "has_prediction": true,
        "per_threshold": {"5": {"tp": 12, "fp": 3, "fn": 2}, ...}}, ...]

Locate the evaluator clone at ``./sn-calibration`` (repo root, so its ``src``
package imports) or set ``SN_CALIBRATION_ROOT``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SNCAL_ROOT = Path(
    os.environ.get("SN_CALIBRATION_ROOT", Path(__file__).resolve().parent / "sn-calibration")
).resolve()
if not (_SNCAL_ROOT / "src").is_dir():
    raise SystemExit(
        f"sn_calibration_eval_cli: sn-calibration clone not found at {_SNCAL_ROOT}. "
        "Clone github.com/SoccerNet/sn-calibration there or set SN_CALIBRATION_ROOT "
        "(see docs/reference/external-calibrators-setup.md)."
    )
sys.path.insert(0, str(_SNCAL_ROOT))

from src.evaluate_camera import evaluate_camera_prediction, get_polylines  # noqa: E402
from src.evaluate_extremities import mirror_labels, scale_points  # noqa: E402


def _best_confusion(projected, gt_scaled, threshold):
    """Per-image 2x2 confusion, taking the better of the annotation and its
    centre-mirror (as the official evaluator does to absorb the left/right
    labelling ambiguity)."""
    conf1, _, _ = evaluate_camera_prediction(projected, gt_scaled, threshold)
    conf2, _, _ = evaluate_camera_prediction(projected, mirror_labels(gt_scaled), threshold)
    acc1 = conf1[0, 0] / conf1.sum() if conf1.sum() > 0 else 0.0
    acc2 = conf2[0, 0] / conf2.sum() if conf2.sum() > 0 else 0.0
    return conf1 if acc1 >= acc2 else conf2


def run(args) -> None:
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    thresholds = [int(t) for t in args.thresholds.split(",")]

    records = []
    for gt_path in sorted(gt_dir.glob("*.json")):
        frame_id = gt_path.stem
        pred_path = pred_dir / f"camera_{frame_id}.json"
        if not pred_path.exists():
            records.append({"image": frame_id, "has_prediction": False, "per_threshold": {}})
            continue

        gt_scaled = scale_points(json.loads(gt_path.read_text()), args.width, args.height)
        camera_json = json.loads(pred_path.read_text())
        projected = get_polylines(camera_json, args.width, args.height, sampling_factor=0.9)

        per_threshold = {}
        for threshold in thresholds:
            confusion = _best_confusion(projected, gt_scaled, threshold)
            per_threshold[str(threshold)] = {
                "tp": float(confusion[0, 0]),
                "fp": float(confusion[0, 1]),
                "fn": float(confusion[1, 0]),
            }
        records.append(
            {"image": frame_id, "has_prediction": True, "per_threshold": per_threshold}
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sn_calibration_eval_cli",
        description="SoccerNet-Calibration per-image scoring for MatchDay Gate 1.",
    )
    parser.add_argument("--gt-dir", required=True, help="SoccerNet split dir (<id>.json + <id>.jpg)")
    parser.add_argument("--pred-dir", required=True, help="Dir of camera_<id>.json predictions")
    parser.add_argument("--thresholds", default="5,10,20", help="Comma-separated pixel thresholds")
    parser.add_argument("--width", type=int, default=960, help="Evaluation resolution width")
    parser.add_argument("--height", type=int, default=540, help="Evaluation resolution height")
    parser.add_argument("--out", required=True, help="Output per-image confusions JSON path")
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 -- surface every failure on stderr, exit non-zero
        print(f"sn_calibration_eval_cli: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
