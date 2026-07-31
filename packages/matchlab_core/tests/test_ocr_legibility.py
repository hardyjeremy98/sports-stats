import numpy as np
import pytest
from matchlab_core.ocr.legibility import LegibilityClassifier


def test_score_before_prepare_is_loud():
    with pytest.raises(RuntimeError, match="prepare"):
        LegibilityClassifier().score([np.zeros((60, 40, 3), dtype=np.uint8)])


def test_provenance_records_the_noncommercial_finetune():
    prov = LegibilityClassifier().provenance()
    assert prov.architecture == "resnet34"
    assert "NonCommercial" in prov.license.weights
    assert "soccer" in prov.lineage.lower()


def test_strip_model_ft_prefix():
    from matchlab_core.ocr.legibility import _strip_prefix

    raw = {"model_ft.fc.weight": 1, "model_ft.conv1.weight": 2}
    stripped = _strip_prefix(raw)
    assert stripped == {"fc.weight": 1, "conv1.weight": 2}


def test_strip_prefix_leaves_unprefixed_keys_alone():
    from matchlab_core.ocr.legibility import _strip_prefix

    raw = {"fc.weight": 1}
    assert _strip_prefix(raw) == {"fc.weight": 1}
