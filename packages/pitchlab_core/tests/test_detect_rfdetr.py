"""rfdetr-local stage (SPO-36): permissive RF-DETR detector — output mapping,
provenance/license axes, fail-loud prepare. The model itself (rfdetr package +
weights) is not exercised here; inference is measured end-to-end via the
frozen-detection export + benchmark runner. These tests pin the pure mapping
and the licensing posture the certification gate (SPO-41) reads."""

import pytest
from pitchlab_core.registry import available
from pitchlab_core.schemas.detections import DetectionClass
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.stages.detect.rfdetr import RfDetrDetector, _to_detections


def test_registered():
    assert "rfdetr-local" in available()[StageKind.DETECT.value]


def test_prepare_missing_package_fails_loudly():
    """rfdetr is not a declared dependency (supplied per-invocation with
    `uv run --with rfdetr`); prepare must refuse loudly, not import-error."""
    det = RfDetrDetector()
    with pytest.raises(RuntimeError, match="rfdetr"):
        det.prepare(ctx=None)


def test_to_detections_filters_to_person_and_maps_to_player():
    # Three boxes: person, person, and a non-person class that must be dropped.
    xyxy = [[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 5.0, 5.0], [1.0, 1.0, 2.0, 2.0]]
    conf = [0.9, 0.8, 0.95]
    class_id = [1, 3, 1]  # person_class_id = 1
    dets = _to_detections(xyxy, conf, class_id, person_class_id=1)
    assert len(dets) == 2
    assert all(d.cls == DetectionClass.PLAYER for d in dets)
    assert dets[0].box.x1 == pytest.approx(10.0)
    assert dets[0].box.y2 == pytest.approx(60.0)
    assert dets[0].confidence == pytest.approx(0.9)
    assert dets[1].confidence == pytest.approx(0.95)


def test_to_detections_empty():
    assert _to_detections([], [], [], person_class_id=1) == []


def test_provenance_permissive_axes():
    det = RfDetrDetector(model_size="base")
    (prov,) = det.provenance()
    assert prov.architecture == "rf-detr-base"
    # Code + weights permissive (Apache / Roboflow commercial grant).
    assert "Apache-2.0" in prov.license.code
    assert "Apache-2.0" in prov.license.weights
    # Training data records the COCO/Objects365 residual honestly, but as a
    # commercial-permissive CC-BY basis (ship weights, not data).
    assert "CC BY" in prov.license.training_data
    # Must NOT carry any non-permissive marker (this stage IS the shipping
    # path; the SPO-41 gate must certify it). Checked as strings here to keep
    # this stage independent of the SPO-41 branch; SPO-41 covers the gate math.
    markers = ("non-shippable", "by-nc", "non-commercial", "agpl", "research-only")
    for axis in (prov.license.code, prov.license.weights, prov.license.training_data):
        low = axis.lower()
        assert not any(m in low for m in markers), f"non-permissive marker in {axis!r}"
