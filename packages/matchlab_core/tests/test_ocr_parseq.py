import numpy as np
import pytest
from matchlab_core.ocr.parseq import CropRead, JerseyReader, number_band


def test_number_band_slices_the_named_fraction():
    image = np.zeros((100, 40, 3), dtype=np.uint8)
    band = number_band(image, lo=0.12, hi=0.50)
    assert band.shape == (38, 40, 3)  # rows 12..49


def test_number_band_defaults_match_the_measured_sweep():
    image = np.zeros((200, 60, 3), dtype=np.uint8)
    band = number_band(image)
    assert band.shape == (int(0.50 * 200) - int(0.12 * 200), 60, 3)


def test_number_band_degenerate_zero_height_returns_original():
    image = np.zeros((0, 40, 3), dtype=np.uint8)
    band = number_band(image)
    assert band.shape == image.shape


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


class _FakeLegibility:
    def __init__(self, scores):
        self._scores = scores

    def score(self, crops):
        return list(self._scores[: len(crops)])


def _reader_with_fakes(scores, min_legibility=None):
    kwargs = {} if min_legibility is None else {"min_legibility": min_legibility}
    reader = JerseyReader(**kwargs)
    reader._model = object()  # bypass "prepare() not called" guard
    reader._legibility = _FakeLegibility(scores)
    reader._char_probs = lambda patch: (np.full((3, 11), 1.0 / 11), 1.0)  # stub
    return reader


class _Crop:
    def __init__(self, frame_idx):
        self.image = np.zeros((100, 40, 3), dtype=np.uint8)
        self.frame_idx = frame_idx
        self.quality = 0.5


def test_crops_below_min_legibility_produce_no_read():
    reader = _reader_with_fakes(scores=[0.2, 0.95], min_legibility=0.9)
    reads = reader.read([_Crop(0), _Crop(1)])
    assert len(reads) == 1
    assert reads[0].frame_idx == 1


def test_legibility_score_becomes_crop_read_legibility_not_crop_quality():
    reader = _reader_with_fakes(scores=[0.97])
    reads = reader.read([_Crop(0)])
    assert reads[0].legibility == pytest.approx(0.97)


def test_all_crops_gated_out_returns_empty_list_not_an_error():
    reader = _reader_with_fakes(scores=[0.01, 0.05], min_legibility=0.1)
    assert reader.read([_Crop(0), _Crop(1)]) == []


def test_default_min_legibility_is_a_floor_not_a_hard_gate():
    """Offline sweep (2026-07-31): the hard legibility>=0.9 gate is replaced
    by a soft weight; min_legibility now only floors out attractor-biased
    sub-0.1 reads, so a mid-range score (e.g. 0.5) must be READ, not gated."""
    reader = _reader_with_fakes(scores=[0.5, 0.05])
    reads = reader.read([_Crop(0), _Crop(1)])
    assert len(reads) == 1
    assert reads[0].frame_idx == 0
    assert reads[0].legibility == pytest.approx(0.5)


def test_crop_read_carries_decode_confidence():
    reader = _reader_with_fakes(scores=[0.5])
    reads = reader.read([_Crop(0)])
    assert reads[0].confidence == pytest.approx(1.0)


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
