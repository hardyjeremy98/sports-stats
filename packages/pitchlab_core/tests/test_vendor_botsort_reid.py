"""SPO-31 Tasks 2-3: the vendored BoT-SORT's appearance extension — per-track
feature EMA and the quality-gated, boost-only first-association blend."""

import numpy as np
import pytest

pytest.importorskip("trackers")

from trackers.utils.state_representations import XCYCWHStateEstimator  # noqa: E402

from pitchlab_core.vendor.botsort_reid.tracklet import BoTSORTReidTracklet  # noqa: E402


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
