"""RTMPose pose front-end (SPO-37): output mapping, provenance/license axes,
fail-loud prepare. rtmlib + the ONNX weights are not exercised here (that is the
spot-check script); these tests pin the pure mapping and the *honest* licensing
posture — stock RTMPose weights are Apache code + Apache weight-files but
trained on body7 (incl. AI Challenger, non-commercial), so the training-data
axis is non-shippable and the SPO-41 gate must refuse the stock checkpoint."""

from __future__ import annotations

import pytest
from pitchlab_core.pose.rtmpose import DetectionPose, RTMPoseEstimator, _to_poses
from pitchlab_core.schemas.geometry import Box


def test_to_poses_maps_parallel_keypoint_and_score_arrays():
    # 2 detections, 3 keypoints each (toy K); (x, y) + score.
    kpts = [[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], [[30.0, 40.0], [31.0, 41.0], [32.0, 42.0]]]
    scores = [[0.9, 0.8, 0.1], [0.7, 0.6, 0.5]]
    poses = _to_poses(kpts, scores)
    assert len(poses) == 2
    assert isinstance(poses[0], DetectionPose)
    assert poses[0].keypoints[0] == (10.0, 20.0, 0.9)
    assert poses[0].keypoints[2] == (12.0, 22.0, 0.1)
    assert poses[1].keypoints[1] == (31.0, 41.0, 0.6)


def test_to_poses_empty():
    assert _to_poses([], []) == []


def test_prepare_missing_rtmlib_fails_loudly():
    est = RTMPoseEstimator()
    with pytest.raises(RuntimeError, match="rtmlib"):
        est.prepare(device="cpu")


def test_estimate_before_prepare_fails_loudly():
    est = RTMPoseEstimator()
    with pytest.raises(RuntimeError, match="prepare"):
        est.estimate(image=None, boxes=[Box(x1=0, y1=0, x2=10, y2=20)])


def test_provenance_code_and_weights_permissive():
    prov = RTMPoseEstimator().provenance()
    assert "Apache" in prov.license.code
    assert "Apache" in prov.license.weights
    assert prov.architecture.startswith("rtmpose")


def test_provenance_training_data_flags_non_commercial_stock_weights():
    """The decisive honesty: stock RTMPose is body7-trained, so training_data
    carries the non-commercial AI Challenger taint -> non-shippable until the
    head is retrained. The SPO-41 gate must be able to see this."""
    prov = RTMPoseEstimator().provenance()
    td = prov.license.training_data.lower()
    assert "ai challenger" in td or "aic" in td
    assert "non-commercial" in td or "non-shippable" in td
