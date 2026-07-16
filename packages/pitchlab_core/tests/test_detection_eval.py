"""Detection-quality evaluator tests (SPO-9).

`evaluate_detections` needs only numpy (no motmetrics/scipy/trackeval), so
unlike test_gt_eval.py's integration tests there is no importorskip here --
mirrors test_hota.py's analytic-docstring style: every expected number is
hand-derivable from the algorithm itself (see each test's docstring).
"""

from __future__ import annotations

import pytest
from pitchlab_core.detection_eval import evaluate_detections


def test_precision_recall_operating_point_threshold_binds():
    """Two frames, two GT tracks, two detections.

    Frame 0: GT track 1 box (0,0,10,10). Detection at the identical box
    (IoU=1.0) -> matches, contributing 1 TP.

    Frame 1: GT track 2 box (100,100,10,10). Detection shifted +4px in x,
    box (104,100,10,10): intersection width = 10-4=6, height=10 ->
    inter=60, union=100+100-60=140, IoU=60/140=3/7~=0.4286 < 0.5 -- strictly
    below the 0.5 threshold, proving it does not bind loosely: this pair
    must NOT match (GT track 2 -> FN, the detection -> FP).

    Totals: n_detections=2, n_gt_boxes=2, TP=1, FP=1, FN=1.
    precision = TP/n_detections = 1/2 = 0.5
    recall = TP/n_gt_boxes = 1/2 = 0.5
    """
    det_by_frame = {
        0: [(0.9, (0.0, 0.0, 10.0, 10.0))],
        1: [(0.8, (104.0, 100.0, 10.0, 10.0))],
    }
    gt_by_frame = {
        0: [(1, (0.0, 0.0, 10.0, 10.0))],
        1: [(2, (100.0, 100.0, 10.0, 10.0))],
    }

    result = evaluate_detections(det_by_frame, gt_by_frame)

    assert result["n_detections"] == 2
    assert result["n_gt_boxes"] == 2
    assert result["n_frames_evaluated"] == 2
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_average_precision_hand_computed_envelope():
    """One frame, 3 GT tracks far apart (each 10x10), 4 detections: 3 exact
    matches (IoU=1.0, TP) and 1 far-away false positive, ranked by
    confidence descending: TP(0.9), FP(0.8), TP(0.7), TP(0.6).

    cumulative TP: 1,1,2,3   cumulative FP: 0,1,1,1
    precision:     1, .5, 2/3, 3/4
    recall:        1/3,1/3,2/3,1

    All-point interpolation (VOC2010+): the precision envelope is the
    running max scanned from the high-recall end backwards:
      raw precision list [1, .5, 2/3, 3/4] (+0 sentinel at the end)
      -> envelope [1, 3/4, 3/4, 3/4]  (envelope[0] stays 1 since 1 > 3/4;
         everything from index 1 on collapses to the trailing max 3/4)

    AP = sum over each *change* in recall of (delta_recall * envelope at
    the new recall level):
      (1/3 - 0)   * 1    = 1/3
      (1/3 - 1/3) * ...  = 0        (no recall change between det 1 and 2)
      (2/3 - 1/3) * 3/4  = 1/4
      (1   - 2/3) * 3/4  = 1/4
    AP = 1/3 + 1/4 + 1/4 = 5/6 ~= 0.8333
    """
    gts = [
        (1, (0.0, 0.0, 10.0, 10.0)),
        (2, (1000.0, 0.0, 10.0, 10.0)),
        (3, (2000.0, 0.0, 10.0, 10.0)),
    ]
    dets = [
        (0.9, (0.0, 0.0, 10.0, 10.0)),  # TP -> gt 1
        (0.8, (5000.0, 5000.0, 10.0, 10.0)),  # FP, far from everything
        (0.7, (1000.0, 0.0, 10.0, 10.0)),  # TP -> gt 2
        (0.6, (2000.0, 0.0, 10.0, 10.0)),  # TP -> gt 3
    ]

    result = evaluate_detections({0: dets}, {0: gts})

    assert result["ap"] == round(5 / 6, 4)


def test_height_bins_straddle_edges_and_fp_bins_by_detected_height():
    """Default edges (25, 50, 100) -> bins h<25, 25<=h<50, 50<=h<100, h>=100.

    GT heights 20 (bin0), 30 (bin1), 120 (bin3) -- bin2 (50<=h<100) has NO
    GT at all. Each GT is exactly matched (TP), so TP/FN binning by GT
    height is trivially exercised. One extra FP at height 60 (far from
    every GT box) lands in bin2 purely by its OWN (detected) height --
    proving FP binning is independent of GT height, and exercising the
    "empty GT bin -> ap null" rule (bin2 has n_gt=0 despite having a
    detection assigned to it).
    """
    gts = [
        (1, (0.0, 0.0, 40.0, 20.0)),
        (2, (200.0, 0.0, 40.0, 30.0)),
        (3, (400.0, 0.0, 40.0, 120.0)),
    ]
    dets = [
        (0.9, (0.0, 0.0, 40.0, 20.0)),
        (0.9, (200.0, 0.0, 40.0, 30.0)),
        (0.9, (400.0, 0.0, 40.0, 120.0)),
        (0.5, (9000.0, 9000.0, 40.0, 60.0)),  # FP, height 60 -> bin2, no GT there
    ]

    result = evaluate_detections({0: dets}, {0: gts})

    by_bin = {b["bin"]: b for b in result["by_height_bin"]}
    assert set(by_bin) == {"h<25", "25<=h<50", "50<=h<100", "h>=100"}

    b0 = by_bin["h<25"]
    assert b0["n_gt"] == 1 and b0["n_det"] == 1
    assert b0["precision"] == 1.0 and b0["recall"] == 1.0 and b0["ap"] == 1.0
    assert b0["edges"] == [None, 25.0]

    b1 = by_bin["25<=h<50"]
    assert b1["n_gt"] == 1 and b1["n_det"] == 1 and b1["ap"] == 1.0
    assert b1["edges"] == [25.0, 50.0]

    b3 = by_bin["h>=100"]
    assert b3["n_gt"] == 1 and b3["n_det"] == 1 and b3["ap"] == 1.0
    assert b3["edges"] == [100.0, None]

    b2 = by_bin["50<=h<100"]
    assert b2["n_gt"] == 0
    assert b2["n_det"] == 1  # the FP, binned by its OWN height
    assert b2["precision"] == 0.0  # 0 TP / 1 det
    assert b2["recall"] is None  # no GT to recall
    assert b2["ap"] is None  # zero GT in a bin -> ap null, never NaN


def test_miss_bursts_hand_derived_lengths_and_summary():
    """Track 1 present at evaluated frames 0,2,4,6,8 (stride=2), matched at
    0 and 6 only -> misses at 2,4 (consecutive -> one burst of length 2)
    and 8 (isolated -> one burst of length 1): bursts = [2, 1].

    Track 2 present at 0,2,4 and matched every time -> zero bursts.

    Track 3 present at 0,2,4 and never matched -> one burst spanning its
    whole presence, length 3.

    Pooled burst lengths across all tracks: [2, 1, 3].
    mean = 2.0, median = 2.0, max = 3 (numpy linear-interpolation
    percentiles: p95 of [1,2,3] -> index 0.95*(3-1)=1.9 -> 2 + 0.9*(3-2)
    = 2.9).
    burst_seconds_p95 = p95 * stride / fps = 2.9 * 2 / 25 = 0.232.
    """
    t1, t2, t3 = (0.0, 0.0, 10.0, 10.0), (100.0, 0.0, 10.0, 10.0), (200.0, 0.0, 10.0, 10.0)
    gt_by_frame = {
        0: [(1, t1), (2, t2), (3, t3)],
        2: [(1, t1), (2, t2), (3, t3)],
        4: [(1, t1), (2, t2), (3, t3)],
        6: [(1, t1)],
        8: [(1, t1)],
    }
    det_by_frame = {
        0: [(0.9, t1), (0.9, t2)],  # matches T1, T2; T3 untouched
        2: [(0.9, t2)],  # matches T2 only -- T1, T3 missed
        4: [(0.9, t2)],  # matches T2 only -- T1, T3 missed
        6: [(0.9, t1)],  # matches T1
        8: [],  # T1 missed
    }

    result = evaluate_detections(det_by_frame, gt_by_frame, stride=2, fps=25.0)
    mb = result["miss_bursts"]

    t1_rec = mb["per_track"][1]
    assert t1_rec["n_bursts"] == 2
    assert t1_rec["max_burst"] == 2
    assert t1_rec["burst_lengths_summary"]["mean"] == 1.5
    assert t1_rec["burst_lengths_summary"]["max"] == 2

    t2_rec = mb["per_track"][2]
    assert t2_rec["n_bursts"] == 0
    assert t2_rec["max_burst"] is None
    assert t2_rec["burst_lengths_summary"] is None

    t3_rec = mb["per_track"][3]
    assert t3_rec["n_bursts"] == 1
    assert t3_rec["max_burst"] == 3

    assert mb["n_tracks_with_bursts"] == 2  # T1, T3 (T2 has none)
    overall = mb["overall"]
    assert overall["mean"] == pytest.approx(2.0)
    assert overall["median"] == pytest.approx(2.0)
    assert overall["max"] == 3
    assert overall["p95"] == pytest.approx(2.9)
    assert overall["burst_seconds_p95"] == pytest.approx(2.9 * 2 / 25)
    assert mb["stride"] == 2 and mb["fps"] == 25.0


def test_miss_bursts_empty_when_no_gt_tracks():
    result = evaluate_detections({}, {})
    mb = result["miss_bursts"]
    assert mb["per_track"] == {}
    assert mb["n_tracks_with_bursts"] == 0
    assert mb["overall"] == {
        "min": None,
        "median": None,
        "p95": None,
        "max": None,
        "mean": None,
        "burst_seconds_p95": None,
    }


def test_duplicate_detection_on_matched_gt():
    """One GT box, two detections both exactly on it (IoU=1.0 each). The
    higher-confidence detection is processed first and wins the match; the
    second, lower-confidence detection overlaps an ALREADY-matched GT at
    IoU >= threshold -> counted as a duplicate, not a fresh FN/TP.

    n_detections=2, n_duplicates=1, duplicate_rate=1/2=0.5.
    """
    box = (0.0, 0.0, 10.0, 10.0)
    det_by_frame = {0: [(0.9, box), (0.5, box)]}
    gt_by_frame = {0: [(1, box)]}

    result = evaluate_detections(det_by_frame, gt_by_frame)

    assert result["n_detections"] == 2
    assert result["duplicates"]["n_duplicates"] == 1
    assert result["duplicates"]["duplicate_rate"] == 0.5


def test_duplicates_null_rate_with_zero_detections():
    result = evaluate_detections({}, {0: [(1, (0.0, 0.0, 10.0, 10.0))]})
    assert result["duplicates"] == {"n_duplicates": 0, "duplicate_rate": None}
    # GT present, zero detections: AP is 0.0 (curve never leaves the origin),
    # never None -- None is reserved for n_gt == 0 (undefined, not zero).
    assert result["ap"] == 0.0


def test_jitter_constant_offset_is_zero_and_gap_breaks_pairing():
    """A GT track moves at constant velocity (x += 10px per frame, 40x40
    box) across evaluated frames 0..4. The detector tracks it with a
    constant (dx=2, dy=3) offset for frames 0-1 (residual r(t) identical
    -> zero jitter, proving GT motion is subtracted out via the residual),
    is entirely missing at frame 2 (breaking the chain), resumes with the
    SAME (2, 3) offset at frame 3, then jumps to (dx=6, dy=3) at frame 4.

    Valid adjacent matched pairs: (0,1) and (3,4) only -- (1,2) and (2,3)
    require frame 2 to be matched, but it has no detection at all, so
    those pairs must not exist and must not smuggle the gap's real motion
    into a jitter number (no pair spans the gap).

    pair(0,1): r0=r1=(2,3) -> center_delta=|Δr|=0.
    pair(3,4): r3=(2,3), r4=(6,3) -> Δr=(4,0) -> center_delta=4.0.
    Heights never change here -> height_delta=0 for both pairs.

    n_pairs=2. center_jitter values=[0,4]: mean=2.0,
    p95=np.percentile([0,4],95)=3.8. height_jitter values=[0,0]: mean=0,
    p95=0.
    """

    def gt_box(f: int) -> tuple[float, float, float, float]:
        return (10.0 * f, 0.0, 40.0, 40.0)

    def det_box(f: int, dx: float, dy: float) -> tuple[float, float, float, float]:
        gx, gy, gw, gh = gt_box(f)
        return (gx + dx, gy + dy, gw, gh)

    gt_by_frame = {f: [(1, gt_box(f))] for f in range(5)}
    det_by_frame = {
        0: [(0.9, det_box(0, 2, 3))],
        1: [(0.9, det_box(1, 2, 3))],
        2: [],  # missed entirely -- breaks the chain
        3: [(0.9, det_box(3, 2, 3))],
        4: [(0.9, det_box(4, 6, 3))],
    }

    result = evaluate_detections(det_by_frame, gt_by_frame)
    jitter = result["jitter"]

    assert jitter["n_pairs"] == 2
    assert jitter["center_jitter_mean"] == pytest.approx(2.0)
    assert jitter["center_jitter_p95"] == pytest.approx(3.8)
    assert jitter["height_jitter_mean"] == pytest.approx(0.0)
    assert jitter["height_jitter_p95"] == pytest.approx(0.0)


def test_jitter_height_only_change_isolated_from_center():
    """Frame 0: detection exactly on the (static) GT box (r=(0,0), height
    residual 0). Frame 1: detection's height grows by 4px, but its y is
    shifted up by half that (2px) so its CENTER stays exactly on the GT
    center (r stays (0,0)) -- isolating a pure height-jitter signal with
    zero center jitter.

    pair(0,1): center_delta=0; height_delta=|4-0|=4.
    n_pairs=1 -> mean/p95 both equal the single value.
    """
    gt = (0.0, 0.0, 40.0, 40.0)
    det_by_frame = {
        0: [(0.9, (0.0, 0.0, 40.0, 40.0))],
        1: [(0.9, (0.0, -2.0, 40.0, 44.0))],
    }
    gt_by_frame = {0: [(1, gt)], 1: [(1, gt)]}

    result = evaluate_detections(det_by_frame, gt_by_frame)
    jitter = result["jitter"]

    assert jitter["n_pairs"] == 1
    assert jitter["center_jitter_mean"] == pytest.approx(0.0)
    assert jitter["height_jitter_mean"] == pytest.approx(4.0)
    assert jitter["height_jitter_p95"] == pytest.approx(4.0)


def test_jitter_null_with_zero_pairs():
    result = evaluate_detections({}, {})
    assert result["jitter"] == {
        "center_jitter_mean": None,
        "center_jitter_p95": None,
        "height_jitter_mean": None,
        "height_jitter_p95": None,
        "n_pairs": 0,
    }


def test_gt_id_tie_break_lower_id_wins_on_equal_iou():
    """One detection, two GT tracks with IDENTICAL boxes (so IoU against
    the detection is exactly equal for both) but different track ids (5
    and 2, deliberately listed with the higher id first). The tie-break
    rule (lower GT id wins) must pick track 2, not 5 -- observed indirectly
    through miss_bursts: the WINNING track is "matched" (zero bursts over
    its one frame), the LOSING track is left unmatched (one burst of
    length 1, the size of its entire presence).
    """
    box = (0.0, 0.0, 10.0, 10.0)
    det_by_frame = {0: [(0.9, box)]}
    gt_by_frame = {0: [(5, box), (2, box)]}

    result = evaluate_detections(det_by_frame, gt_by_frame)

    per_track = result["miss_bursts"]["per_track"]
    assert per_track[2]["n_bursts"] == 0  # matched -> lower id wins the tie
    assert per_track[5]["n_bursts"] == 1
    assert per_track[5]["max_burst"] == 1


def test_empty_input_no_crash_no_nan():
    result = evaluate_detections({}, {})

    assert result["n_frames_evaluated"] == 0
    assert result["n_detections"] == 0
    assert result["n_gt_boxes"] == 0
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["ap"] is None
    assert result["duplicates"] == {"n_duplicates": 0, "duplicate_rate": None}
    assert result["jitter"]["n_pairs"] == 0
    assert result["miss_bursts"]["n_tracks_with_bursts"] == 0
    for b in result["by_height_bin"]:
        assert b["ap"] is None
        assert b["n_gt"] == 0
        assert b["n_det"] == 0
        assert b["precision"] is None
        assert b["recall"] is None


def test_frames_evaluated_are_exactly_gt_by_frame_keys():
    """A detection frame outside gt_by_frame's keys must be silently
    ignored -- the function evaluates exactly the frames present as keys
    in gt_by_frame, per its documented contract."""
    det_by_frame = {
        0: [(0.9, (0.0, 0.0, 10.0, 10.0))],
        99: [(0.9, (0.0, 0.0, 10.0, 10.0))],  # not an evaluated frame
    }
    gt_by_frame = {0: [(1, (0.0, 0.0, 10.0, 10.0))]}

    result = evaluate_detections(det_by_frame, gt_by_frame)

    assert result["n_frames_evaluated"] == 1
    assert result["n_detections"] == 1  # frame 99's detection never counted
