"""The oracle tracker's external feature backend (SPO-85).

The subprocess itself is not exercised here (it needs the CAMELTrack venv and
GPU weights); what is tested is the contract that keeps a benchmark honest --
an unavailable checkpoint refuses loudly instead of quietly running a
different model -- plus the fragment -> det.txt conversion the bridge consumes.
"""

from __future__ import annotations

import pytest
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.tracks import TrackletFrame
from matchlab_core.stages.track.oracle_external import (
    _resolve_weights,
    tracklets_to_detections,
)

BOX = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0}


def _tracklet(tid: int, frame_idxs: list[int]) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[TrackletFrame(frame_idx=i, box=BOX, confidence=1.0) for i in frame_idxs],
    )


def test_unknown_model_is_a_loud_error():
    with pytest.raises(ValueError, match="Unknown external re-ID model"):
        _resolve_weights("not-a-model", external_root="/tmp")


def test_missing_weights_names_the_acquisition_step(tmp_path):
    with pytest.raises(RuntimeError, match="zenodo"):
        _resolve_weights("prtreid", external_root=str(tmp_path))


def test_missing_weights_refuses_rather_than_falling_back(tmp_path):
    # A silent fallback would mislabel a benchmark row, which is worse than a crash.
    with pytest.raises(RuntimeError, match="Refusing to run this arm"):
        _resolve_weights("kpr", external_root=str(tmp_path))


def test_present_weights_resolve(tmp_path):
    path = tmp_path / "CAMELTrack/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar"
    path.parent.mkdir(parents=True)
    path.touch()
    assert _resolve_weights("prtreid", external_root=str(tmp_path)) == path


def test_fragments_become_per_frame_detections():
    dets = tracklets_to_detections([_tracklet(1, [0, 1]), _tracklet(2, [1, 2])], fps=25.0)
    assert [d.frame_idx for d in dets] == [0, 1, 2]
    assert [len(d.detections) for d in dets] == [1, 2, 1]
    assert dets[1].t == pytest.approx(1 / 25.0)


def test_every_fragment_frame_survives_the_conversion():
    # Row loss here would silently drop evidence from the measurement.
    tracklets = [_tracklet(1, [0, 5, 9]), _tracklet(2, [3])]
    dets = tracklets_to_detections(tracklets, fps=25.0)
    total = sum(len(d.detections) for d in dets)
    assert total == sum(len(t.frames) for t in tracklets)
