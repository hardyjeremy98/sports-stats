"""TDD for the Gate 1 SoccerNet-Calibration eval harness (SPO-67).

SPO-60's Gate 1 requires reproducing PnLCalib's published SoccerNet-Calibration
test-split accuracy under the *official* challenge protocol before the pipeline
integration is trusted. The heavy, geometry-exact per-image scoring (project the
3D pitch template through predicted camera parameters, count polyline
true/false positives/negatives at a pixel threshold) is the official
`sn-calibration` evaluator's job and runs in an isolated sibling checkout,
reached only as a subprocess (no upstream LICENSE => not vendored; same
isolation posture as the PnLCalib predictor). What lives *in-repo* and is tested
here is (a) the split-level aggregation into the official metrics
(completeness x mean per-image Jaccard = final score) and (b) the orchestration
that drives the predictor + scorer subprocesses and writes the gate record.

Metric definitions (github.com/SoccerNet/sn-calibration, evaluate_camera.py):

* per-image accuracy (a.k.a. JaC@t, Jaccard at t pixels) = TP / (TP + FP + FN)
  over the image's line classes, at pixel threshold t;
* completeness = (# images with a prediction) / (# images total);
* final score = completeness x mean-over-predicted-images(accuracy@t).

Missed images (no prediction) are excluded from the accuracy mean but still
count against completeness -- exactly as the official loop does (it `continue`s
past a missing prediction file without appending an accuracy).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from matchlab_train.calibration_gate import (
    PNLCALIB_SN23_TEST_PUBLISHED,
    aggregate_camera_metrics,
    build_gate_record,
    run_gate1_calibration_eval,
)


def _img(has_prediction: bool, per_threshold: dict[int, tuple[int, int, int]]) -> dict:
    """Build a per-image confusion record (tp, fp, fn) at each threshold."""
    return {
        "image": "x",
        "has_prediction": has_prediction,
        "per_threshold": {
            str(t): {"tp": tp, "fp": fp, "fn": fn}
            for t, (tp, fp, fn) in per_threshold.items()
        },
    }


# --- metric math (hand-computed fixtures) ------------------------------------


def test_aggregate_matches_hand_computed_official_metrics():
    # 3 images total: two with predictions, one missed.
    #   A@5: tp=2 fp=1 fn=1 -> acc = 2/4 = 0.5
    #   B@5: tp=3 fp=0 fn=1 -> acc = 3/4 = 0.75
    #   C   : no prediction (missed)
    # completeness = 2/3 ; mean acc@5 = (0.5 + 0.75)/2 = 0.625
    # final score@5 = 2/3 * 0.625 = 0.416666...
    per_image = [
        _img(True, {5: (2, 1, 1)}),
        _img(True, {5: (3, 0, 1)}),
        _img(False, {}),
    ]
    out = aggregate_camera_metrics(per_image, total_images=3, thresholds=[5])
    m = out[5]
    assert m["completeness"] == pytest.approx(2 / 3)
    assert m["jac"] == pytest.approx(0.625)
    assert m["final_score"] == pytest.approx(2 / 3 * 0.625)
    assert m["n_evaluated"] == 2
    assert m["n_total"] == 3


def test_aggregate_zero_confusion_image_scores_zero_not_dropped():
    # An image WITH a prediction but no matched/annotated lines (tp+fp+fn == 0)
    # contributes accuracy 0.0 to the mean (it is not silently dropped), matching
    # the official evaluator appending 0.0 for such a frame.
    per_image = [
        _img(True, {5: (0, 0, 0)}),
        _img(True, {5: (4, 0, 0)}),
    ]
    out = aggregate_camera_metrics(per_image, total_images=2, thresholds=[5])
    assert out[5]["jac"] == pytest.approx(0.5)  # (0.0 + 1.0)/2
    assert out[5]["completeness"] == pytest.approx(1.0)
    assert out[5]["n_evaluated"] == 2


def test_aggregate_multiple_thresholds_independent():
    per_image = [
        _img(True, {5: (1, 1, 0), 10: (2, 0, 0)}),
        _img(False, {}),
    ]
    out = aggregate_camera_metrics(per_image, total_images=2, thresholds=[5, 10])
    assert out[5]["jac"] == pytest.approx(0.5)
    assert out[10]["jac"] == pytest.approx(1.0)
    # completeness is threshold-independent (0.5) but reported at every threshold
    assert out[5]["completeness"] == pytest.approx(0.5)
    assert out[10]["completeness"] == pytest.approx(0.5)


def test_aggregate_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        aggregate_camera_metrics([], total_images=0, thresholds=[5])


# --- gate record / published-number comparison -------------------------------


def test_published_targets_are_the_paper_sn23_mv_numbers():
    # arXiv:2404.08401v5 Table I, SN23-test, Ours_MV + PnL (P and L refinement).
    assert PNLCALIB_SN23_TEST_PUBLISHED["jac"][5] == pytest.approx(0.787)
    assert PNLCALIB_SN23_TEST_PUBLISHED["final_score"] == pytest.approx(0.618)
    assert PNLCALIB_SN23_TEST_PUBLISHED["completeness"] == pytest.approx(0.784)


def test_gate_record_flags_reproduction_below_tolerance_as_not_passed():
    measured = {
        5: {"jac": 0.50, "completeness": 1.0, "final_score": 0.50, "n_evaluated": 3, "n_total": 3},
    }
    rec = build_gate_record(
        measured, total_images=3, split="test", thresholds=[5], tolerance=0.03
    )
    assert rec["passed"] is False
    cmp5 = rec["comparison"]["5"]
    assert cmp5["published_jac"] == pytest.approx(0.787)
    assert cmp5["measured_jac"] == pytest.approx(0.50)
    assert cmp5["jac_delta"] == pytest.approx(0.50 - 0.787)
    assert cmp5["jac_within_tolerance"] is False


def test_gate_record_passes_when_measured_meets_published_within_tolerance():
    # Meeting or exceeding published (minus tolerance) is a pass.
    measured = {
        5: {"jac": 0.79, "completeness": 0.78, "final_score": 0.62, "n_evaluated": 9, "n_total": 10},
    }
    rec = build_gate_record(
        measured, total_images=10, split="test", thresholds=[5], tolerance=0.03
    )
    assert rec["passed"] is True
    assert rec["comparison"]["5"]["jac_within_tolerance"] is True


# --- orchestration plumbing (stubbed predictor + scorer subprocesses) --------

# A stub predictor: reads the job manifest, writes one camera_<id>.json into the
# out dir per <id>.jpg in the frames dir. No model, no torch.
_STUB_PREDICT = """
import json, sys
from pathlib import Path
job = None
for i, a in enumerate(sys.argv):
    if a == "--job":
        job = sys.argv[i + 1]
m = json.loads(Path(job).read_text())
frames = Path(m["frames_dir"]); out = Path(m["out_dir"]); out.mkdir(parents=True, exist_ok=True)
for jpg in sorted(frames.glob("*.jpg")):
    fid = jpg.stem
    (out / f"camera_{fid}.json").write_text(json.dumps({
        "pan_degrees": 0.0, "tilt_degrees": 0.0, "roll_degrees": 0.0,
        "position_meters": [0.0, 0.0, 10.0],
        "x_focal_length": 1.0, "y_focal_length": 1.0, "principal_point": [480.0, 270.0],
        "radial_distortion": [0.0] * 6, "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0] * 4,
    }))
"""

# A stub scorer: emits a fixed per-image confusion for every GT image, marking an
# image predicted iff camera_<id>.json exists. Deterministic so the aggregate is
# hand-checkable in the test below.
_STUB_SCORE = """
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--gt-dir", required=True); p.add_argument("--pred-dir", required=True)
p.add_argument("--thresholds", required=True); p.add_argument("--out", required=True)
p.add_argument("--width", type=int, default=960); p.add_argument("--height", type=int, default=540)
a = p.parse_args()
ts = [int(x) for x in a.thresholds.split(",")]
gt = Path(a.gt_dir); pred = Path(a.pred_dir)
recs = []
for j in sorted(gt.glob("*.json")):
    fid = j.stem
    has = (pred / f"camera_{fid}.json").exists()
    recs.append({
        "image": fid, "has_prediction": has,
        "per_threshold": ({str(t): {"tp": 2, "fp": 1, "fn": 1} for t in ts} if has else {}),
    })
Path(a.out).write_text(json.dumps(recs))
"""


def _make_soccernet_split(root: Path, ids: list[str], *, with_images: bool) -> None:
    split = root / "test"
    split.mkdir(parents=True, exist_ok=True)
    for fid in ids:
        (split / f"{fid}.json").write_text(json.dumps({"Side line top": [{"x": 0.1, "y": 0.1}]}))
        if with_images:
            (split / f"{fid}.jpg").write_bytes(b"")


def test_run_gate1_end_to_end_with_stubbed_predictor_and_scorer(tmp_path):
    sn = tmp_path / "soccernet" / "calibration"
    _make_soccernet_split(sn, ["00001", "00002", "00003"], with_images=True)
    out_dir = tmp_path / "reports" / "gate1"

    rec = run_gate1_calibration_eval(
        soccernet_dir=sn,
        split="test",
        thresholds=[5],
        out_dir=out_dir,
        predictor_command=[sys.executable, "-c", _STUB_PREDICT],
        scorer_command=[sys.executable, "-c", _STUB_SCORE],
        tolerance=0.03,
    )

    # All three images predicted (stub predictor wrote a camera json for each),
    # each scored tp=2 fp=1 fn=1 => acc = 0.5 at t=5.
    # completeness = 3/3 = 1.0 ; jac@5 = 0.5 ; final = 0.5.
    assert rec["n_images"] == 3
    assert rec["measured"]["5"]["completeness"] == pytest.approx(1.0)
    assert rec["measured"]["5"]["jac"] == pytest.approx(0.5)
    assert rec["measured"]["5"]["final_score"] == pytest.approx(0.5)
    # 0.5 is well below the published 0.787 target -> gate not passed.
    assert rec["passed"] is False

    # Gate record + human-readable summary both land under out_dir.
    record_json = out_dir / "gate1_calibration.json"
    summary_md = out_dir / "gate1_calibration.md"
    assert record_json.exists() and summary_md.exists()
    on_disk = json.loads(record_json.read_text())
    assert on_disk["measured"]["5"]["jac"] == pytest.approx(0.5)
    text = summary_md.read_text()
    assert "78.7" in text  # published JaC@5 cited in the summary
    assert "Gate 1" in text


def test_run_gate1_uses_precomputed_predictions_when_no_predictor(tmp_path):
    sn = tmp_path / "soccernet" / "calibration"
    _make_soccernet_split(sn, ["00001", "00002"], with_images=False)
    # Pre-supply predictions for only one of the two images.
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    (pred_dir / "camera_00001.json").write_text(json.dumps({"principal_point": [480.0, 270.0]}))
    out_dir = tmp_path / "reports"

    rec = run_gate1_calibration_eval(
        soccernet_dir=sn,
        split="test",
        thresholds=[5],
        out_dir=out_dir,
        prediction_dir=pred_dir,
        scorer_command=[sys.executable, "-c", _STUB_SCORE],
        tolerance=0.03,
    )
    # 1 of 2 images predicted -> completeness 0.5.
    assert rec["measured"]["5"]["completeness"] == pytest.approx(0.5)
    assert rec["n_images"] == 2


def test_run_gate1_raises_when_split_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_gate1_calibration_eval(
            soccernet_dir=tmp_path / "nope",
            split="test",
            thresholds=[5],
            out_dir=tmp_path / "out",
            prediction_dir=tmp_path,
            scorer_command=[sys.executable, "-c", _STUB_SCORE],
        )


def test_run_gate1_raises_when_scorer_fails(tmp_path):
    sn = tmp_path / "soccernet" / "calibration"
    _make_soccernet_split(sn, ["00001"], with_images=False)
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    with pytest.raises(RuntimeError):
        run_gate1_calibration_eval(
            soccernet_dir=sn,
            split="test",
            thresholds=[5],
            out_dir=tmp_path / "out",
            prediction_dir=pred_dir,
            scorer_command=[sys.executable, "-c", "import sys; sys.exit(3)"],
        )


def test_cli_passes_predictor_params_through_to_job_manifest(tmp_path, monkeypatch):
    sn = tmp_path / "soccernet" / "calibration"
    _make_soccernet_split(sn, ["00001"], with_images=True)
    out_dir = tmp_path / "reports" / "gate1"
    sentinel = tmp_path / "params_seen.json"

    predict_py = tmp_path / "stub_predict.py"
    predict_py.write_text(
        _STUB_PREDICT + f"\nPath({str(sentinel)!r}).write_text(json.dumps(m['params']))\n"
    )
    score_py = tmp_path / "stub_score.py"
    score_py.write_text(_STUB_SCORE)

    from matchlab_train.cli import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matchlab-train",
            "gate1-calibration-eval",
            "--soccernet-dir", str(sn),
            "--split", "test",
            "--thresholds", "5",
            "--out", str(out_dir),
            "--predictor-cmd", f"{sys.executable} {predict_py}",
            "--scorer-cmd", f"{sys.executable} {score_py}",
            "--predictor-params", '{"weights_kp": "W_KP_MARKER", "device": "cuda:0"}',
        ],
    )
    rc = main()
    assert rc in (0, 1)  # the gate verdict is irrelevant here; the stub scores low
    seen = json.loads(sentinel.read_text())
    assert seen["weights_kp"] == "W_KP_MARKER"
    assert seen["device"] == "cuda:0"
    assert seen["mode"] == "camera"
