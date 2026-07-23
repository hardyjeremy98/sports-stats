"""Vendored MixSort YOLOX: model builds and loads the frozen checkpoint."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CKPT = Path("data/weights/mixsort/yolox_x_sports_train.pth.tar")


def test_build_yolox_x_shape():
    from matchlab_core.vendor.mixsort_yolox import build_yolox

    model = build_yolox(1.33, 1.25, 1)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 96, 160))
    # 1-class YOLOX output: [batch, n_anchors, 4 box + 1 obj + 1 cls]
    assert out.shape[0] == 1 and out.shape[2] == 6


@pytest.mark.skipif(not CKPT.exists(), reason="frozen checkpoint not downloaded")
def test_checkpoint_loads_strict():
    from matchlab_core.vendor.mixsort_yolox import build_yolox

    model = build_yolox(1.33, 1.25, 1)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
