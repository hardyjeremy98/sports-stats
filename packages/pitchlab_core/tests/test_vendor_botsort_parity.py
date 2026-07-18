"""SPO-31 Task 1: the vendored BoT-SORT (before any appearance change, and with
appearance off) must reproduce the installed `trackers` BoT-SORT exactly. This
is the regression guard that keeps the bbox-only twin airtight."""

import numpy as np
import pytest


def _frames():
    """A 6-frame two-object sequence: both move steadily rightward, far apart."""

    out = []
    for f in range(6):
        x = 100 + f * 5
        out.append(
            dict(
                xyxy=np.array([[x, 100, x + 20, 160], [x + 200, 100, x + 220, 160]], np.float32),
                confidence=np.array([0.9, 0.9], np.float32),
                class_id=np.zeros(2, int),
            )
        )
    return out


def _canonical(id_lists):
    """Relabel ids to first-occurrence order so two trackers whose class-level id
    counters start at different offsets still compare equal on assignment pattern."""
    mapping: dict[int, int] = {}
    out = []
    for ids in id_lists:
        row = []
        for i in ids:
            row.append(mapping.setdefault(int(i), len(mapping)))
        out.append(row)
    return out


def test_vendored_matches_installed_bbox_only():
    trackers = pytest.importorskip("trackers")
    import supervision as sv
    from pitchlab_core.vendor.botsort_reid.tracker import BoTSORTReidTracker

    kw = dict(frame_rate=25.0, lost_track_buffer=25, enable_cmc=False)
    up = trackers.BoTSORTTracker(**kw)
    vd = BoTSORTReidTracker(**kw)

    up_ids, vd_ids, up_boxes, vd_boxes = [], [], [], []
    for fr in _frames():
        ru = up.update(sv.Detections(**fr))
        rv = vd.update(sv.Detections(**fr))
        up_ids.append(list(ru.tracker_id))
        vd_ids.append(list(rv.tracker_id))
        up_boxes.append(ru.xyxy)
        vd_boxes.append(rv.xyxy)

    assert _canonical(up_ids) == _canonical(vd_ids)
    for a, b in zip(up_boxes, vd_boxes):
        assert np.allclose(a, b)
