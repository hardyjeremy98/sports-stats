from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.gt_fragments import fragment_tracks
from matchlab_core.schemas.detections import DetectionClass


def _track(track_id: int, frame_idxs: list[int], **kw) -> GroundTruthTrack:
    return GroundTruthTrack(
        track_id=track_id,
        frames=[
            GroundTruthFrame(frame_idx=i, box={"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 20.0})
            for i in frame_idxs
        ],
        **kw,
    )


def _gt(tracks: list[GroundTruthTrack]) -> GroundTruth:
    return GroundTruth(fps=25.0, width=100, height=100, seq_length=100, tracks=tracks)


def test_contiguous_track_yields_one_fragment():
    res = fragment_tracks(_gt([_track(7, [0, 1, 2, 3])]))
    assert len(res.tracklets) == 1
    assert [f.frame_idx for f in res.tracklets[0].frames] == [0, 1, 2, 3]
    assert res.gt_track_by_fragment == {res.tracklets[0].tracklet_id: 7}


def test_gap_larger_than_threshold_splits():
    # frames 0,1 then 10,11 -- a 9-frame step, well over gap_frames=2
    res = fragment_tracks(_gt([_track(7, [0, 1, 10, 11])]), gap_frames=2)
    assert len(res.tracklets) == 2
    assert [f.frame_idx for f in res.tracklets[0].frames] == [0, 1]
    assert [f.frame_idx for f in res.tracklets[1].frames] == [10, 11]
    assert set(res.gt_track_by_fragment.values()) == {7}


def test_gap_exactly_at_threshold_does_not_split():
    # 0 -> 2 is a step of 2; gap_frames=2 means "split when the step exceeds 2"
    res = fragment_tracks(_gt([_track(7, [0, 2, 4])]), gap_frames=2)
    assert len(res.tracklets) == 1


def test_gap_one_over_threshold_splits():
    res = fragment_tracks(_gt([_track(7, [0, 3])]), gap_frames=2)
    assert len(res.tracklets) == 2


def test_min_fragment_frames_drops_slivers():
    res = fragment_tracks(_gt([_track(7, [0, 1, 2, 20])]), gap_frames=2, min_fragment_frames=2)
    assert len(res.tracklets) == 1
    assert [f.frame_idx for f in res.tracklets[0].frames] == [0, 1, 2]


def test_roles_map_to_detection_classes_and_other_is_excluded():
    gt = _gt(
        [
            _track(1, [0, 1], role="player"),
            _track(2, [0, 1], role="goalkeeper"),
            _track(3, [0, 1], role="referee"),
            _track(4, [0, 1], role="other"),
            _track(5, [0, 1], role="ball"),
        ]
    )
    res = fragment_tracks(gt)
    classes = sorted(t.cls.value for t in res.tracklets)
    assert classes == sorted(
        [
            DetectionClass.PLAYER.value,
            DetectionClass.GOALKEEPER.value,
            DetectionClass.REFEREE.value,
        ]
    )


def test_fragment_ids_are_unique_and_stable_across_calls():
    gt = _gt([_track(7, [0, 1, 10, 11]), _track(8, [0, 1])])
    a = fragment_tracks(gt)
    b = fragment_tracks(gt)
    ids = [t.tracklet_id for t in a.tracklets]
    assert len(set(ids)) == len(ids)
    assert ids == [t.tracklet_id for t in b.tracklets]


def test_jersey_and_team_are_carried_per_fragment():
    gt = _gt([_track(7, [0, 1, 10, 11], jersey="9", team="left")])
    res = fragment_tracks(gt)
    assert set(res.jersey_by_fragment.values()) == {"9"}
    assert set(res.team_by_fragment.values()) == {"left"}


def test_unsorted_gt_frames_are_ordered_before_splitting():
    # A GT file need not list frames in order; fragmentation must not depend on it.
    res = fragment_tracks(_gt([_track(7, [11, 0, 10, 1])]), gap_frames=2)
    assert [[f.frame_idx for f in t.frames] for t in res.tracklets] == [[0, 1], [10, 11]]


def test_empty_ground_truth_yields_no_fragments():
    res = fragment_tracks(_gt([]))
    assert res.tracklets == []
    assert res.gt_track_by_fragment == {}
