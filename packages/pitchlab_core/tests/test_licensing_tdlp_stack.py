"""SPO-41 x SPO-42/44: the per-axis licensing gate applied to the ACTUAL
assembled tdlp-shippable stack's provenance.

Demonstrates the gate walks every component's real `provenance()` output and
correctly refuses the current stack — surfacing exactly the non-permissive
axes that block shipping today (RTMPose stock weights' NC training data, the
DINOv2 unspecified training-data axis, and the untrained/NC-preliminary head) —
so a non-shippable component can never reach the shipping path unnoticed.
"""

from __future__ import annotations

from pitchlab_core.licensing import (
    AxisVerdict,
    LicenseCertificationError,
    assert_stack_shippable,
    certify_stack,
    classify_axis,
)
from pitchlab_core.pose.rtmpose import RTMPoseEstimator
from pitchlab_core.provenance import RunProvenance, StageProvenance
from pitchlab_core.registry import build
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.detect.rfdetr import RfDetrDetector


def _assembled_stack_provenance() -> RunProvenance:
    """Build the assembled stack's provenance from each component's real
    provenance() (all cheap/static — no weights loaded)."""
    detect_models = RfDetrDetector(model_size="base").provenance()
    track = build(StageKind.TRACK, "tdlp-shippable",
                  {"use_appearance": True, "use_keypoints": False, "checkpoint": ""})
    track_models = list(track.provenance())
    track_models.append(RTMPoseEstimator().provenance())  # pose cue (pre-prepare)
    return RunProvenance(
        stages={
            "detect": StageProvenance(impl="rfdetr-local", models=detect_models),
            "track": StageProvenance(impl="tdlp-shippable", models=track_models),
        }
    )


def test_rfdetr_axes_all_permissive():
    (m,) = RfDetrDetector(model_size="base").provenance()
    assert classify_axis(m.license.code) == AxisVerdict.PERMISSIVE
    assert classify_axis(m.license.weights) == AxisVerdict.PERMISSIVE
    # COCO annotations are CC BY (attribution-only) -> permissive
    assert classify_axis(m.license.training_data) == AxisVerdict.PERMISSIVE


def test_rtmpose_training_data_is_non_permissive():
    m = RTMPoseEstimator().provenance()
    assert classify_axis(m.license.code) == AxisVerdict.PERMISSIVE  # Apache
    # stock body7 weights include AI Challenger (non-commercial)
    assert classify_axis(m.license.training_data) == AxisVerdict.NON_PERMISSIVE


def test_dinov2_training_data_not_certifiable():
    from pitchlab_core.stages.associate.embedders.dinov2 import LICENSE

    assert classify_axis(LICENSE.code) == AxisVerdict.PERMISSIVE
    assert classify_axis(LICENSE.weights) == AxisVerdict.PERMISSIVE
    # LVD-142M underlying image licenses unspecified -> UNKNOWN (fails closed)
    assert classify_axis(LICENSE.training_data) == AxisVerdict.UNKNOWN


def test_assembled_stack_is_refused_with_diagnostic_findings():
    prov = _assembled_stack_provenance()
    cert = certify_stack(prov)
    assert not cert.passed
    flagged = {(f.stage, f.axis) for f in cert.findings}
    # the shipping blockers we expect to be surfaced today
    assert ("track", "training_data") in flagged  # rtmpose NC + dinov2 unspecified + head
    assert ("track", "weights") in flagged  # untrained/NC-preliminary head
    # RF-DETR must contribute NO findings (fully permissive)
    assert all(f.stage != "detect" for f in cert.findings)

    try:
        assert_stack_shippable(prov, context="tdlp-shippable Bar A")
        raise AssertionError("expected LicenseCertificationError")
    except LicenseCertificationError as exc:
        assert "tdlp-shippable Bar A" in str(exc)
        assert "rtmpose" in str(exc).lower() or "training_data" in str(exc)
