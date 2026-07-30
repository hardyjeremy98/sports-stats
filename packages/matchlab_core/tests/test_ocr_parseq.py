import numpy as np
import pytest
from matchlab_core.ocr.parseq import CropRead, JerseyReader, torso_box
from matchlab_core.pose.rtmpose import DetectionPose

# COCO-17 indices used by torso_box.
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _pose(**kv: tuple[float, float, float]) -> DetectionPose:
    kpts = [(0.0, 0.0, 0.0)] * 17
    idx = {"ls": L_SHOULDER, "rs": R_SHOULDER, "lh": L_HIP, "rh": R_HIP}
    for name, value in kv.items():
        kpts[idx[name]] = value
    return DetectionPose(keypoints=kpts)


def test_torso_box_spans_shoulders_and_hips():
    pose = _pose(ls=(30.0, 40.0, 0.9), rs=(70.0, 40.0, 0.9),
                 lh=(35.0, 100.0, 0.9), rh=(65.0, 100.0, 0.9))
    box = torso_box(pose, (200, 100), pad_frac=0.0)
    assert box is not None
    assert box.x1 == pytest.approx(30.0)
    assert box.x2 == pytest.approx(70.0)
    assert box.y1 == pytest.approx(40.0)
    assert box.y2 == pytest.approx(100.0)


def test_torso_box_is_clamped_to_the_crop():
    pose = _pose(ls=(-50.0, -20.0, 0.9), rs=(500.0, -20.0, 0.9),
                 lh=(-50.0, 400.0, 0.9), rh=(500.0, 400.0, 0.9))
    box = torso_box(pose, (200, 100), pad_frac=0.5)
    assert box is not None
    assert box.x1 >= 0.0 and box.y1 >= 0.0
    assert box.x2 <= 100.0 and box.y2 <= 200.0


def test_low_confidence_keypoints_abstain():
    """No torso means no read -- never a guessed region."""
    pose = _pose(ls=(30.0, 40.0, 0.05), rs=(70.0, 40.0, 0.05),
                 lh=(35.0, 100.0, 0.05), rh=(65.0, 100.0, 0.05))
    assert torso_box(pose, (200, 100)) is None


def test_degenerate_torso_abstains():
    """All four keypoints collapsed onto a point is not a readable region."""
    pose = _pose(ls=(50.0, 50.0, 0.9), rs=(50.0, 50.0, 0.9),
                 lh=(50.0, 50.0, 0.9), rh=(50.0, 50.0, 0.9))
    assert torso_box(pose, (200, 100), pad_frac=0.0) is None


def test_read_before_prepare_is_loud():
    with pytest.raises(RuntimeError, match="prepare"):
        JerseyReader().read([])


def test_crop_read_char_probs_rows_are_distributions():
    r = CropRead(char_probs=np.full((3, 11), 1.0 / 11), legibility=0.5, frame_idx=7)
    assert np.allclose(r.char_probs.sum(axis=1), 1.0)


def test_provenance_records_the_noncommercial_finetune():
    """Train-adjacency and the NC weights licence must both be recorded."""
    prov = JerseyReader().provenance()
    assert prov.license.code == "Apache-2.0"
    assert "NonCommercial" in prov.license.weights
    assert "hockey" in prov.lineage.lower()


class _FakeTokenizer:
    def __init__(self, itos, eos_id=0):
        self._itos = itos
        self.eos_id = eos_id


class _FakeModel:
    def __init__(self, itos, eos_id=0):
        self.tokenizer = _FakeTokenizer(itos, eos_id=eos_id)

    def eval(self):
        return self

    def to(self, device):
        return self

    def load_state_dict(self, state, strict=True):
        pass


def test_bad_tokenizer_mapping_raises_loudly(monkeypatch):
    """A missing digit or a digit/EOS collision must fail loudly at prepare()
    time rather than silently producing wrong digit/EOS columns."""
    reader = JerseyReader()

    # itos is missing digit "7" entirely.
    bad_itos = ["[E]"] + [str(d) for d in range(10) if d != 7]

    class _FakeHub:
        @staticmethod
        def load(*a, **k):
            return _FakeModel(bad_itos)

    class _FakeTorch:
        hub = _FakeHub()

        @staticmethod
        def load(*a, **k):
            return {}

    monkeypatch.setitem(__import__("sys").modules, "torch", _FakeTorch())

    with pytest.raises(RuntimeError, match="(?i)digit|tokenizer|mapping"):
        reader.prepare()


def test_lightning_state_dict_gains_the_model_prefix():
    """The checkpoint's `state_dict` keys are bare (`pos_queries`), but the
    hub-loaded LightningModule's own state_dict keys are `model.*` (it wraps
    the net as `self.model`) -- loading the checkpoint unprefixed matches
    zero keys, so every bare key must gain the `model.` prefix."""
    from matchlab_core.ocr.parseq import _add_lightning_prefix

    raw = {"pos_queries": 1, "encoder.pos_embed": 2, "model.already_prefixed": 3}
    prefixed = _add_lightning_prefix(raw)
    assert prefixed == {
        "model.pos_queries": 1,
        "model.encoder.pos_embed": 2,
        "model.already_prefixed": 3,
    }
