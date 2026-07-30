"""Recovering a tracklet's detection class after a MOT round trip.

`tdlp-full` hands detections to an external tracker as a MOT `det.txt` and reads
back a MOT result file. MOT has no class column, so the class is structurally
lost: on SNMOT-118 the detector produced 641 goalkeeper and 749 referee boxes and
every one of the 25 returned tracklets came back `player`.

That silently disabled two deliberate mechanisms. `reid_engine.eligible()`
excludes referees from merging by testing `cls == REFEREE`, and both team
classifiers halve a goalkeeper's confidence by testing `cls == GOALKEEPER` --
neither can fire when nothing is ever anything but a player. The measured cost on
SNMOT-118 was a referee merged into a player's identity thread, which was that
run's ONLY wrong merge.

The class is still recoverable because the input detections carry it: match each
tracklet frame back to the detection it came from by box overlap, then take the
majority, exactly as `_assembly.py` already does for in-repo trackers (which get
the detection index for free and so need no matching).
"""

from __future__ import annotations

from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.detections import Detection, DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import TrackletFrame
from matchlab_core.stages.track._assembly import assign_classes_by_overlap


def _box(x: float) -> Box:
    return Box(x1=x, y1=0.0, x2=x + 10.0, y2=20.0)


def _dets(frame_idx: int, *pairs) -> FrameDetections:
    return FrameDetections(
        frame_idx=frame_idx,
        t=frame_idx / 25.0,
        detections=[
            Detection(box=_box(x), confidence=0.9, cls=c) for x, c in pairs
        ],
    )


def _tracklet(tid: int, xs, frames=None) -> Tracklet:
    frames = frames or list(range(len(xs)))
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(frame_idx=f, box=_box(x), confidence=0.9, source="observed")
            for f, x in zip(frames, xs, strict=True)
        ],
    )


def test_a_referee_tracklet_recovers_its_class():
    """The case that actually cost a wrong merge on SNMOT-118."""
    dets = [_dets(0, (0.0, DetectionClass.REFEREE)), _dets(1, (1.0, DetectionClass.REFEREE))]
    out = assign_classes_by_overlap([_tracklet(1, [0.0, 1.0])], dets)
    assert out[0].cls == DetectionClass.REFEREE


def test_the_majority_class_wins_over_a_dissenting_frame():
    """Per-frame detector class is noisy; one stray label must not flip a
    tracklet, which is why this votes rather than taking the first match."""
    dets = [
        _dets(0, (0.0, DetectionClass.GOALKEEPER)),
        _dets(1, (1.0, DetectionClass.GOALKEEPER)),
        _dets(2, (2.0, DetectionClass.PLAYER)),
    ]
    out = assign_classes_by_overlap([_tracklet(1, [0.0, 1.0, 2.0])], dets)
    assert out[0].cls == DetectionClass.GOALKEEPER


def test_the_right_detection_is_chosen_when_several_share_a_frame():
    """A frame holds every detection, so matching must be by overlap and not by
    position in the list -- otherwise a tracklet inherits a neighbour's class."""
    dets = [
        _dets(0, (0.0, DetectionClass.PLAYER), (100.0, DetectionClass.REFEREE)),
        _dets(1, (1.0, DetectionClass.PLAYER), (101.0, DetectionClass.REFEREE)),
    ]
    out = assign_classes_by_overlap([_tracklet(7, [100.0, 101.0])], dets)
    assert out[0].cls == DetectionClass.REFEREE


def test_an_unmatched_tracklet_keeps_the_class_it_arrived_with():
    """No detection overlaps, so there is no evidence to act on. Inventing a
    class here would be worse than the default it already carries."""
    dets = [_dets(0, (500.0, DetectionClass.REFEREE))]
    out = assign_classes_by_overlap([_tracklet(1, [0.0])], dets)
    assert out[0].cls == DetectionClass.PLAYER


def test_frames_are_matched_by_index_not_by_position():
    """Tracklet frames are source `frame_idx` and need not start at 0 or be
    contiguous; a positional lookup would silently read the wrong frame."""
    dets = [_dets(0, (0.0, DetectionClass.PLAYER)), _dets(9, (5.0, DetectionClass.GOALKEEPER))]
    out = assign_classes_by_overlap([_tracklet(1, [5.0], frames=[9])], dets)
    assert out[0].cls == DetectionClass.GOALKEEPER


def test_the_input_tracklets_are_not_mutated():
    dets = [_dets(0, (0.0, DetectionClass.REFEREE))]
    original = _tracklet(1, [0.0])
    out = assign_classes_by_overlap([original], dets)
    assert original.cls == DetectionClass.PLAYER
    assert out[0].cls == DetectionClass.REFEREE
