"""Quality-approved crop-yield guardrail metric (SPO-30)."""

from matchlab_core.crop_yield import crop_yield


def test_height_gate_and_per_player_mean():
    # tracklet 1: two tall boxes (h=80) -> approved; tracklet 2: one short box (h=10) -> starved
    tracklets = {
        1: [(0, [0, 0, 30, 80]), (1, [0, 0, 30, 80])],
        2: [(0, [200, 200, 30, 10])],
    }
    # GT track 7 overlaps tracklet 1; GT track 8 overlaps tracklet 2
    gt = {
        0: [(7, [0, 0, 30, 80]), (8, [200, 200, 30, 10])],
        1: [(7, [0, 0, 30, 80])],
    }
    out = crop_yield(
        tracklets, gt, eval_frames=[0, 1], iou_threshold=0.5,
        min_box_height_px=60, min_crops_per_tracklet=2,
    )
    assert out["approved_total"] == 2  # only tracklet 1's two tall boxes
    assert out["starved_tracklet_fraction"] == 0.5  # tracklet 2 has < 2 approved
    # player 7: 2 approved crops; player 8: 0 -> mean over the 2 assigned players = 1.0
    assert out["approved_per_gt_player_mean"] == 1.0
    assert out["per_tracklet"]["max"] == 2


def test_eval_frames_restrict_counting():
    # A box outside eval_frames must not count.
    tracklets = {1: [(0, [0, 0, 30, 80]), (5, [0, 0, 30, 80])]}
    gt = {0: [(7, [0, 0, 30, 80])]}
    out = crop_yield(tracklets, gt, eval_frames=[0], iou_threshold=0.5)
    assert out["approved_total"] == 1  # frame 5 not in eval_frames


def test_empty_tracklets_is_zero_not_crash():
    out = crop_yield({}, {}, eval_frames=[0], iou_threshold=0.5)
    assert out["approved_total"] == 0
    assert out["approved_per_gt_player_mean"] == 0.0
    assert out["starved_tracklet_fraction"] == 0.0
