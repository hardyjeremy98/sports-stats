"""SPO-31 Tasks 2-3: the vendored BoT-SORT's appearance extension — per-track
feature EMA and the quality-gated, boost-only first-association blend."""

import numpy as np
import pytest

pytest.importorskip("trackers")

from pitchlab_core.vendor.botsort_reid.tracklet import BoTSORTReidTracklet  # noqa: E402
from trackers.utils.state_representations import XCYCWHStateEstimator  # noqa: E402


def _bbox():
    return np.array([0.0, 0.0, 10.0, 20.0])


# --- Task 2: per-track appearance EMA ---


def test_tracklet_smooth_feat_starts_none_and_sets_on_first():
    t = BoTSORTReidTracklet(_bbox(), XCYCWHStateEstimator)
    assert t.smooth_feat is None
    f1 = np.array([1.0, 0.0], np.float32)
    t.update(_bbox(), feat=f1)
    assert np.allclose(t.smooth_feat, f1)


def test_tracklet_smooth_feat_ema_renormalized():
    t = BoTSORTReidTracklet(_bbox(), XCYCWHStateEstimator)
    f1 = np.array([1.0, 0.0], np.float32)
    f2 = np.array([0.0, 1.0], np.float32)
    t.update(_bbox(), feat=f1)
    t.update(_bbox(), feat=f2)  # default momentum 0.9
    exp = 0.9 * f1 + 0.1 * f2
    exp = exp / np.linalg.norm(exp)
    assert np.allclose(t.smooth_feat, exp, atol=1e-6)
    assert abs(np.linalg.norm(t.smooth_feat) - 1.0) < 1e-6


def test_tracklet_bbox_only_update_leaves_appearance_untouched():
    t = BoTSORTReidTracklet(_bbox(), XCYCWHStateEstimator)
    t.update(_bbox())  # no feat -> pure bbox-only path
    assert t.smooth_feat is None


# --- Task 3: quality-gated, boost-only first-association blend ---


def _mk_tracker(**kw):
    from pitchlab_core.vendor.botsort_reid.tracker import BoTSORTReidTracker

    base = dict(frame_rate=25.0, lost_track_buffer=25, enable_cmc=False)
    base.update(kw)
    return BoTSORTReidTracker(**base)


class _FakeTrack:
    def __init__(self, feat):
        self.smooth_feat = feat


def test_blend_appearance_weight_zero_is_identity():
    trk = _mk_tracker(appearance_weight=0.0)
    iou = np.array([[0.5, 0.1], [0.1, 0.5]], np.float32)
    tracks = [_FakeTrack(np.array([1, 0], np.float32)), _FakeTrack(np.array([0, 1], np.float32))]
    emb = np.array([[1, 0], [0, 1]], np.float32)
    out = trk._blend_appearance(iou.copy(), tracks, np.array([0, 1]), emb, np.array([True, True]))
    assert np.allclose(out, iou)  # weight 0 -> exact twin of the fused IoU matrix


def test_blend_boosts_feasible_matching_pair_only():
    trk = _mk_tracker(appearance_weight=0.3, minimum_iou_threshold_first_assoc=0.2)
    iou = np.array([[0.5, 0.5], [0.5, 0.5]], np.float32)  # all feasible, IoU-tied
    tracks = [_FakeTrack(np.array([1, 0], np.float32)), _FakeTrack(np.array([0, 1], np.float32))]
    emb = np.array([[1, 0], [0, 1]], np.float32)  # det0~track0, det1~track1
    out = trk._blend_appearance(iou.copy(), tracks, np.array([0, 1]), emb, np.array([True, True]))
    assert out[0, 0] == pytest.approx(0.8)  # 0.5 + 0.3*cos(1.0)
    assert out[1, 1] == pytest.approx(0.8)
    assert out[0, 1] == pytest.approx(0.5)  # cos 0 -> no boost (tie-break, not force)
    assert out[1, 0] == pytest.approx(0.5)


def test_blend_never_lifts_infeasible_pair():
    # perfect appearance but IoU below the gate must stay below the gate (no force).
    trk = _mk_tracker(appearance_weight=0.9, minimum_iou_threshold_first_assoc=0.2)
    iou = np.array([[0.1]], np.float32)
    tracks = [_FakeTrack(np.array([1, 0], np.float32))]
    out = trk._blend_appearance(iou.copy(), tracks, np.array([0]), np.array([[1, 0]], np.float32),
                                np.array([True]))
    assert out[0, 0] == pytest.approx(0.1)


def test_blend_gated_by_embed_ok_and_max_distance():
    trk = _mk_tracker(appearance_weight=0.5, max_embed_distance=0.25,
                      minimum_iou_threshold_first_assoc=0.2)
    iou = np.array([[0.5, 0.5]], np.float32)
    tracks = [_FakeTrack(np.array([1, 0], np.float32))]
    # det0 matches but embed_ok False; det1 embed_ok True but cos 0 (dist 1.0 > 0.25).
    emb = np.array([[1, 0], [0, 1]], np.float32)
    out = trk._blend_appearance(iou.copy(), tracks, np.array([0, 1]), emb, np.array([False, True]))
    assert np.allclose(out, iou)  # neither boosted


def test_appearance_off_equals_bbox_only_end_to_end():
    import supervision as sv
    from pitchlab_core.vendor.botsort_reid.tracker import BoTSORTReidTracker

    kw = dict(frame_rate=25.0, lost_track_buffer=25, enable_cmc=False)
    plain = BoTSORTReidTracker(**kw)  # no embeddings passed
    withemb = BoTSORTReidTracker(appearance_weight=0.0, **kw)  # embeddings present, weight 0
    ids_p, ids_w = [], []
    for f in range(6):
        x = 100 + f * 5
        xyxy = np.array([[x, 100, x + 20, 160], [x + 200, 100, x + 220, 160]], np.float32)
        conf = np.array([0.9, 0.9], np.float32)
        rp = plain.update(sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(2, int)))
        rw = withemb.update(sv.Detections(
            xyxy=xyxy, confidence=conf, class_id=np.zeros(2, int),
            data={"embedding": np.array([[1, 0], [0, 1]], np.float32),
                  "embed_ok": np.array([True, True])}))
        ids_p.append(list(rp.tracker_id))
        ids_w.append(list(rw.tracker_id))

    # The two trackers share the class-level id counter, so absolute ids differ
    # by an offset; the invariant is an identical assignment PATTERN.
    def _canon(streams):
        m: dict[int, int] = {}
        return [[m.setdefault(int(i), len(m)) for i in row] for row in streams]

    assert _canon(ids_p) == _canon(ids_w)


def test_matched_track_absorbs_detection_feature():
    import supervision as sv

    trk = _mk_tracker(appearance_weight=0.3)
    feat = np.array([0.6, 0.8], np.float32)  # will be L2-normalized inside
    trk.update(sv.Detections(
        xyxy=np.array([[0, 0, 10, 20]], np.float32), confidence=np.array([0.9], np.float32),
        class_id=np.zeros(1, int),
        data={"embedding": feat[None, :], "embed_ok": np.array([True])}))
    assert len(trk.tracks) == 1
    assert np.allclose(trk.tracks[0].smooth_feat, feat / np.linalg.norm(feat))
