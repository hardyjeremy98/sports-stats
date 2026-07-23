"""yolox-local stage: preproc math, output mapping, provenance, fail-loud prepare."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matchlab_core.provenance import sha256_file  # noqa: E402
from matchlab_core.registry import available  # noqa: E402
from matchlab_core.schemas.detections import DetectionClass  # noqa: E402
from matchlab_core.schemas.run import StageKind  # noqa: E402
from matchlab_core.stages.detect.yolox_local import (  # noqa: E402
    YoloxLocalDetector,
    _preproc,
    _to_detections,
)


def test_registered():
    assert "yolox-local" in available()[StageKind.DETECT.value]


def test_prepare_missing_weights_fails_loudly():
    det = YoloxLocalDetector(weights="data/weights/does-not-exist.pth.tar")
    with pytest.raises(RuntimeError, match="does-not-exist"):
        det.prepare(ctx=None)


def test_preproc_letterbox_and_normalization():
    # 100x200 BGR image, all pixels BGR=(114, 114, 114) -> after /255 and
    # ImageNet norm, channel 0 (R) = (114/255 - 0.485) / 0.229
    img = np.full((100, 200, 3), 114, dtype=np.uint8)
    out, ratio = _preproc(img, (800, 1440))
    assert out.shape == (3, 800, 1440)
    assert out.dtype == np.float32
    assert ratio == pytest.approx(min(800 / 100, 1440 / 200))  # 7.2
    expected_r = (114 / 255 - 0.485) / 0.229
    assert out[0, 0, 0] == pytest.approx(expected_r, abs=1e-4)
    # padding region has the same value (pad fill is 114.0 pre-normalization)
    assert out[0, 799, 1439] == pytest.approx(expected_r, abs=1e-4)


def test_to_detections_scales_and_maps():
    # one postprocess row: box (72, 72, 144, 144) in input space, obj 0.9, cls 0.8
    rows = torch.tensor([[72.0, 72.0, 144.0, 144.0, 0.9, 0.8, 0.0]])
    dets = _to_detections(rows, ratio=7.2)
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == DetectionClass.PLAYER
    assert d.confidence == pytest.approx(0.72)
    assert d.box.x1 == pytest.approx(10.0)
    assert d.box.y2 == pytest.approx(20.0)


def test_to_detections_none_is_empty():
    assert _to_detections(None, ratio=1.0) == []


def test_provenance_license_axes(tmp_path):
    w = tmp_path / "w.pth.tar"
    w.write_bytes(b"fake-weights")
    det = YoloxLocalDetector(weights=str(w))
    (prov,) = det.provenance()
    assert prov.weights_sha256 == sha256_file(w)
    assert prov.architecture == "yolox-x"
    assert "CC BY-NC 4.0" in prov.license.training_data
    assert "non-shippable" in prov.license.training_data
    assert "Apache-2.0" in prov.license.code
