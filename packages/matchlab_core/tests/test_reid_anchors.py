"""Anchor layer (SPO-56): roster built from GT jersey identities, the oracle
jersey anchor source with coverage/noise/box-height/seed knobs, and the face
stub. GT jersey identities are consumed here and nowhere else."""

from __future__ import annotations

from matchlab_core.gt import GroundTruth, GroundTruthFrame, GroundTruthTrack
from matchlab_core.reid.anchors import (
    FaceAnchorSource,
    OracleJerseyAnchorSource,
    Roster,
    match_tracklets_to_gt,
)
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.tracks import TrackletFrame


def _gt_track(tid, team, jersey, frames, role="player", x=0.0, height=20.0) -> GroundTruthTrack:
    return GroundTruthTrack(
        track_id=tid,
        role=role,
        team=team,
        jersey=jersey,
        frames=[
            GroundTruthFrame(frame_idx=f, box=Box(x1=x, y1=0, x2=x + 10, y2=height))
            for f in frames
        ],
    )


def _gt(tracks) -> GroundTruth:
    return GroundTruth(source="test", tracks=tracks)


def _tracklet(tid, frames, x=0.0, height=20.0) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(frame_idx=f, box=Box(x1=x, y1=0, x2=x + 10, y2=height), confidence=1.0)
            for f in frames
        ],
    )


# --- roster ---------------------------------------------------------------


def test_roster_from_gt_uses_identified_jerseys_with_team_disambiguation():
    gt = _gt(
        [
            _gt_track(1, "left", "7", [0]),
            _gt_track(2, "right", "7", [0]),  # same number, other team
            _gt_track(3, "left", "GK1", [0]),  # letters = unidentified
            _gt_track(4, "left", None, [0]),  # no jersey
            _gt_track(5, None, None, [0], role="referee"),
            _gt_track(6, "left", "7", [5]),  # duplicate identity, second tracklet
        ]
    )
    roster = Roster.from_ground_truth(gt)
    assert roster.candidates == ["left:7", "right:7"]


# --- tracklet -> GT matching ----------------------------------------------


def test_match_tracklets_to_gt_by_overlap():
    gt = _gt(
        [
            _gt_track(1, "left", "7", [0, 1, 2], x=0.0),
            _gt_track(2, "right", "9", [0, 1, 2], x=100.0),
        ]
    )
    t1 = _tracklet(10, [0, 1, 2], x=0.0)
    t2 = _tracklet(20, [0, 1], x=100.0)
    t3 = _tracklet(30, [0, 1], x=500.0)  # overlaps nothing
    matched = match_tracklets_to_gt([t1, t2, t3], gt)
    assert matched[10].track_id == 1
    assert matched[20].track_id == 2
    assert 30 not in matched


# --- oracle jersey anchors ------------------------------------------------


def _setup(n=4):
    """n tracklets, each cleanly overlapping its own GT track with a distinct
    identity left:1..n."""
    gt = _gt(
        [_gt_track(i, "left", str(i), [0, 1, 2], x=100.0 * i) for i in range(1, n + 1)]
    )
    tracklets = [_tracklet(10 * i, [0, 1, 2], x=100.0 * i) for i in range(1, n + 1)]
    return gt, tracklets


def test_oracle_full_coverage_no_noise_anchors_every_tracklet_correctly():
    gt, tracklets = _setup()
    roster = Roster.from_ground_truth(gt)
    src = OracleJerseyAnchorSource(gt, coverage=1.0, noise=0.0, seed=0)
    anchors = src.anchors(tracklets, roster)
    assert {(a.tracklet_id, a.candidate) for a in anchors} == {
        (10, "left:1"), (20, "left:2"), (30, "left:3"), (40, "left:4")
    }
    assert all(a.source == "oracle-jersey" for a in anchors)
    assert all(a.log_lr > 0 for a in anchors)


def test_oracle_coverage_fraction_is_exact_and_seeded():
    gt, tracklets = _setup(n=4)
    roster = Roster.from_ground_truth(gt)
    src = OracleJerseyAnchorSource(gt, coverage=0.5, noise=0.0, seed=7)
    anchors = src.anchors(tracklets, roster)
    assert len(anchors) == 2  # round(0.5 * 4)
    again = OracleJerseyAnchorSource(gt, coverage=0.5, noise=0.0, seed=7).anchors(
        tracklets, roster
    )
    assert [(a.tracklet_id, a.candidate) for a in anchors] == [
        (a.tracklet_id, a.candidate) for a in again
    ]
    different_seed = OracleJerseyAnchorSource(gt, coverage=0.5, noise=0.0, seed=8).anchors(
        tracklets, roster
    )
    assert [(a.tracklet_id, a.candidate) for a in anchors] != [
        (a.tracklet_id, a.candidate) for a in different_seed
    ]


def test_oracle_noise_one_makes_every_anchor_wrong():
    gt, tracklets = _setup()
    roster = Roster.from_ground_truth(gt)
    truth = {10: "left:1", 20: "left:2", 30: "left:3", 40: "left:4"}
    anchors = OracleJerseyAnchorSource(gt, coverage=1.0, noise=1.0, seed=3).anchors(
        tracklets, roster
    )
    assert len(anchors) == 4
    for a in anchors:
        assert a.candidate != truth[a.tracklet_id]
        assert a.candidate in roster.candidates


def test_oracle_min_box_height_excludes_small_tracklets():
    gt = _gt(
        [
            _gt_track(1, "left", "1", [0, 1, 2], x=0.0, height=80.0),
            _gt_track(2, "left", "2", [0, 1, 2], x=100.0),
        ]
    )
    tall = _tracklet(10, [0, 1, 2], x=0.0, height=80.0)
    short = _tracklet(20, [0, 1, 2], x=100.0, height=20.0)
    roster = Roster.from_ground_truth(gt)
    anchors = OracleJerseyAnchorSource(
        gt, coverage=1.0, noise=0.0, min_box_height=50.0, seed=0
    ).anchors([tall, short], roster)
    assert [a.tracklet_id for a in anchors] == [10]


def test_face_stub_emits_nothing():
    gt, tracklets = _setup()
    roster = Roster.from_ground_truth(gt)
    assert FaceAnchorSource().anchors(tracklets, roster) == []
