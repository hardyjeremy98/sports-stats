"""HOTA/DetA/AssA/LocA adapter tests (SPO-7, tracklet-modernization).

Two kinds of coverage, per the task brief:
  1. Analytic cases whose expected values are hand-derivable from the HOTA
     algorithm itself (see docstrings on each test for the arithmetic).
  2. One golden-reference case validated against real, unpatched upstream
     TrackEval (github.com/JonathonLuiten/TrackEval @
     12c8791b303e0a0b50f753af204249e622d0281a) run independently -- see
     `test_hota_matches_upstream_trackeval_golden` for the exact
     command/environment used to generate the hardcoded values.

`compute_hota` needs scipy (vendored TrackEval's Hungarian matching), which
comes in transitively via the `eval` extra's motmetrics -- importorskip on
scipy directly since that's the adapter's actual dependency.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from pitchlab_core.hota import compute_hota  # noqa: E402


def _dense(track_id: int, box: list[float], frames: range) -> dict[int, list[tuple[int, list[float]]]]:
    return {f: [(track_id, box)] for f in frames}


def test_perfect_tracking_one_track_is_all_ones():
    """A single GT track perfectly covered by a single predicted track
    (same box every frame, no gaps, no extra detections) is the textbook
    HOTA=1.0 case: every frame is a true positive at every alpha threshold,
    there is exactly one possible GT<->tracker pairing (so association is
    trivially perfect), and matched boxes are IoU=1.0 (perfect localization).
    HOTA = sqrt(DetA * AssA) = sqrt(1*1) = 1.0.
    """
    box = [100.0, 100.0, 40.0, 120.0]
    gt = _dense(1, box, range(10))
    pred = _dense(5, box, range(10))  # different id on purpose: ids need not match

    result = compute_hota(gt, pred)

    assert result == {"hota": 1.0, "deta": 1.0, "assa": 1.0, "loca": 1.0}


def test_partial_coverage_no_false_positives():
    """Tracker id 5 reproduces GT track 1's exact box for only the first 5 of
    10 frames (frames 5-9 it simply isn't there -- no false positives, just
    missed detections). Hand-derived arithmetic (HOTA's algorithm, see
    vendored hota.py `eval_sequence`):

      TP = 5 (matched frames), FN = 5 (GT frames with no tracker det), FP = 0
      DetRe = TP/(TP+FN) = 5/10 = 0.5      (the brief's headline number)
      DetA  = TP/(TP+FN+FP) = 5/10 = 0.5

      Association: single GT id, single tracker id, matched every frame the
      tracker is present (IoU=1.0 >= every alpha in the grid).
        gt_id_count = 10 (GT detections across the whole sequence)
        tracker_id_count = 5
        matches_count = 5
        AssA = matches_count / (gt_id_count + tracker_id_count - matches_count)
             = 5 / (10 + 5 - 5) = 5/10 = 0.5
        (this holds identically at every alpha bin, since IoU=1.0 clears
        every threshold up to 0.95 -- so the alpha-average equals this too)

      HOTA = sqrt(DetA * AssA) = sqrt(0.5 * 0.5) = sqrt(0.25) = 0.5
      LocA = 1.0 (every matched pair has IoU exactly 1.0)
    """
    box = [100.0, 100.0, 40.0, 120.0]
    gt = _dense(1, box, range(10))
    pred = _dense(5, box, range(5))  # only frames 0-4

    result = compute_hota(gt, pred)

    assert result == {"hota": 0.5, "deta": 0.5, "assa": 0.5, "loca": 1.0}


def test_id_swap_between_two_tracks_hurts_assa_not_deta():
    """Two GT tracks (1 at a fixed position PA, 2 at a fixed, non-overlapping
    position PB) are both perfectly detected every frame -- one tracker
    detection lands exactly on PA and one exactly on PB in every frame, so
    detection is flawless (DetA = 1.0, no FPs/FNs, every GT frame matched at
    IoU=1.0). But the tracker's ID assignment swaps at frame 5: tracker id 10
    tracks GT 1 for frames 0-4 then GT 2 for frames 5-9; tracker id 20 does
    the reverse. This is a pure identity swap with no detection error, so it
    must show up in AssA, not DetA.

    Hand-derived: gt_id_count = tracker_id_count = 10 for every id (present
    every frame). matches_count[gt, tracker] = 5 for all 4 (gt, tracker)
    pairs (each pair overlaps for exactly half the sequence). At every alpha:
      ass_a = 5 / (10 + 10 - 5) = 5/15 = 1/3 for all 4 cells
      AssA = sum(matches_count * ass_a) / TP = (4 * 5 * (1/3)) / 20 = 1/3
    HOTA = sqrt(DetA * AssA) = sqrt(1 * 1/3) = sqrt(1/3) ~= 0.5774
    LocA = 1.0 (every matched pair is an exact box match)

    The brief's ask is qualitative (AssA strictly below DetA); the exact
    fraction is asserted too since it was hand-derivable.
    """
    pa = [0.0, 0.0, 10.0, 10.0]
    pb = [500.0, 500.0, 10.0, 10.0]
    gt: dict[int, list[tuple[int, list[float]]]] = {}
    pred: dict[int, list[tuple[int, list[float]]]] = {}
    for f in range(10):
        gt[f] = [(1, pa), (2, pb)]
        if f < 5:
            pred[f] = [(10, pa), (20, pb)]
        else:
            pred[f] = [(10, pb), (20, pa)]  # swapped

    result = compute_hota(gt, pred)

    assert result["deta"] == 1.0
    assert result["assa"] < result["deta"]
    assert result["assa"] == round(1 / 3, 4)
    assert result["hota"] == round((1 / 3) ** 0.5, 4)
    assert result["loca"] == 1.0


def test_hota_matches_upstream_trackeval_golden():
    """Independent reference values from real, unpatched upstream TrackEval
    (github.com/JonathonLuiten/TrackEval @
    12c8791b303e0a0b50f753af204249e622d0281a, MIT, main branch) -- NOT the
    vendored/patched copy under test, run in a separate throwaway venv so the
    two code paths are genuinely independent.

    Fixture: 6 frames, 2 GT tracks (id 1 static at [0,0,10,10], id 2 static
    at [100,100,10,10]). Tracker id 11 follows GT 1 shifted +2px in x every
    frame (partial IoU 0.6667 -- intersection 8*10=80, union 100+100-80=120)
    except frame 5, which it misses entirely. Tracker id 12 follows GT 2
    exactly (IoU 1.0) every frame. Tracker id 13 is a false positive present
    only at frame 2, far from both GT tracks.

    Upstream cannot run under numpy>=1.24 (`dtype=np.float`, removed there --
    verified empirically that numpy 1.23.5 only warns while 1.26.4 already
    raises; not a numpy-2-specific issue as originally assumed). This host's
    system Python (3.14) has no numpy<1.24 wheel available, so the reference
    run used a throwaway Python-3.11 venv instead:

      uv venv --python 3.11 trackeval_golden_venv
      uv pip install --python trackeval_golden_venv/bin/python "numpy<1.24" scipy \\
          "git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a"
      trackeval_golden_venv/bin/python gen_hota_golden.py   # builds the same
        # data-dict construction compute_hota uses (sorted frames, sorted-
        # unique-id remap, raw/unclamped xywh IoU), then calls upstream
        # HOTA().eval_sequence(data) directly.

    Recorded versions: Python 3.11.15, numpy 1.23.5, scipy 1.15.3,
    trackeval @ 12c8791b303e0a0b50f753af204249e622d0281a.
    Script output: {'hota': 0.7874, 'deta': 0.6842, 'assa': 0.9482, 'loca': 0.8963}
    """
    gt_by_frame = {
        f: [(1, [0.0, 0.0, 10.0, 10.0]), (2, [100.0, 100.0, 10.0, 10.0])] for f in range(6)
    }
    pred_by_frame = {
        0: [(11, [2.0, 0.0, 10.0, 10.0]), (12, [100.0, 100.0, 10.0, 10.0])],
        1: [(11, [2.0, 0.0, 10.0, 10.0]), (12, [100.0, 100.0, 10.0, 10.0])],
        2: [
            (11, [2.0, 0.0, 10.0, 10.0]),
            (12, [100.0, 100.0, 10.0, 10.0]),
            (13, [50.0, 50.0, 5.0, 5.0]),
        ],
        3: [(11, [2.0, 0.0, 10.0, 10.0]), (12, [100.0, 100.0, 10.0, 10.0])],
        4: [(11, [2.0, 0.0, 10.0, 10.0]), (12, [100.0, 100.0, 10.0, 10.0])],
        5: [(12, [100.0, 100.0, 10.0, 10.0])],
    }

    result = compute_hota(gt_by_frame, pred_by_frame)

    assert result == {"hota": 0.7874, "deta": 0.6842, "assa": 0.9482, "loca": 0.8963}


def test_empty_sequence_no_gt_no_pred():
    """No detections on either side at all -- degenerate but must not crash;
    TrackEval's own early-return path handles num_tracker_dets==0 before
    num_gt_ids/num_tracker_ids would matter."""
    result = compute_hota({}, {})

    assert result == {"hota": 0.0, "deta": 0.0, "assa": 0.0, "loca": 1.0}
