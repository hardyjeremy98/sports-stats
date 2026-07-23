"""Tracklet representation (SPO-53 tracer level): single quality-weighted
prototype per tracklet with embedding norm as the quality proxy, and cosine
pair similarity over prototype pairs. Hand-computed expectations."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.reid.representation import build_representations, pair_similarity


def _ff(tids: list[int], fidxs: list[int], embs: list[list[list[float]]]) -> FrameFeatures:
    n = len(tids)
    e = np.array(embs, dtype=np.float32)  # (N, P, D)
    return FrameFeatures(
        tracklet_ids=np.array(tids, dtype=np.int64),
        frame_idxs=np.array(fidxs, dtype=np.int64),
        embeddings=e,
        visibility=np.ones((n, e.shape[1]), dtype=np.float32),
        keypoints_xyc=np.zeros((n, 17, 3), dtype=np.float32),
        keypoints_conf=np.ones(n, dtype=np.float32),
    )


def test_single_prototype_is_norm_weighted_mean():
    # Tracklet 1, one part (P=1, D=2): frames [2,0] (norm 2) and [0,1] (norm 1)
    # -> prototype = (2*[2,0] + 1*[0,1]) / 3 = [4/3, 1/3].
    ff = _ff([1, 1], [0, 5], [[[2.0, 0.0]], [[0.0, 1.0]]])
    reps = build_representations(ff)
    assert set(reps) == {1}
    proto = reps[1].prototypes
    assert proto.shape == (1, 1, 2)
    np.testing.assert_allclose(proto[0, 0], [4 / 3, 1 / 3], rtol=1e-6)


def test_pair_similarity_is_cosine_over_prototypes():
    ff = _ff([1, 2], [0, 10], [[[1.0, 0.0]], [[1.0, 1.0]]])
    reps = build_representations(ff)
    # cos([1,0],[1,1]) = 1/sqrt(2)
    assert pair_similarity(reps[1], reps[2]) == pytest.approx(1 / np.sqrt(2))


def test_zero_embedding_tracklet_has_no_representation():
    ff = _ff([1, 2], [0, 10], [[[0.0, 0.0]], [[1.0, 0.0]]])
    reps = build_representations(ff)
    assert set(reps) == {2}  # all-zero evidence -> abstain, not a NaN prototype
