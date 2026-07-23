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


def test_capped_single_prototype_is_norm_weighted_mean():
    # Tracklet 1, one part (P=1, D=2): frames [2,0] (norm 2) and [0,1] (norm 1)
    # with the prototype cap at 1 -> prototype = (2*[2,0] + 1*[0,1]) / 3.
    ff = _ff([1, 1], [0, 5], [[[2.0, 0.0]], [[0.0, 1.0]]])
    reps = build_representations(ff, max_prototypes=1)
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


# --- multi-prototype view clustering (SPO-54) ------------------------------


def test_two_view_clusters_yield_two_prototypes_with_weighted_centroids():
    # Four frames of tracklet 1 in two orthogonal view directions (P=1, D=2):
    # "front" frames [2,0] and [4,0] (norms 2 and 4), "back" frames [0,1] and
    # [0,3] (norms 1 and 3). With a view threshold below their in-cluster
    # cosine (1.0) and above the cross-cluster cosine (0.0), leader clustering
    # must produce exactly two prototypes, each the norm-weighted mean:
    #   front: (2*[2,0] + 4*[4,0]) / 6 = [10/3, 0]
    #   back:  (1*[0,1] + 3*[0,3]) / 4 = [0, 10/4]
    ff = _ff(
        [1, 1, 1, 1],
        [0, 1, 2, 3],
        [[[2.0, 0.0]], [[0.0, 1.0]], [[4.0, 0.0]], [[0.0, 3.0]]],
    )
    reps = build_representations(ff, max_prototypes=4, view_threshold=0.5)
    protos = reps[1].prototypes
    assert protos.shape == (2, 1, 2)
    # Highest-quality frame ([4,0], norm 4) seeds the first cluster.
    np.testing.assert_allclose(protos[0, 0], [10 / 3, 0.0], rtol=1e-6)
    np.testing.assert_allclose(protos[1, 0], [0.0, 10 / 4], rtol=1e-6)


def test_max_prototypes_caps_cluster_count():
    # Three mutually orthogonal directions but max_prototypes=2: the third
    # direction is absorbed into its nearest cluster instead of crashing.
    ff = _ff(
        [1, 1, 1],
        [0, 1, 2],
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]],
    )
    reps = build_representations(ff, max_prototypes=2, view_threshold=0.5)
    assert reps[1].prototypes.shape[0] == 2


def test_short_tracklets_fall_back_and_are_marked_starved():
    ff = _ff([1], [0], [[[1.0, 0.0]]])
    reps = build_representations(ff, max_prototypes=4, starved_max_frames=2)
    assert reps[1].prototypes.shape[0] == 1
    assert reps[1].starved is True

    ff3 = _ff([2, 2, 2], [0, 1, 2], [[[1.0, 0.0]]] * 3)
    reps3 = build_representations(ff3, starved_max_frames=2)
    assert reps3[2].starved is False


def test_two_view_player_beats_different_player_unlike_mean_pooling():
    # Player A's tracklet 1 holds two orthogonal views F=[1,0,0], B=[0,1,0].
    # Tracklet 2 is player A again from the front (F). Tracklet 3 is a
    # different player whose single view M=[1,1,0]/sqrt(2) happens to resemble
    # tracklet 1's naive mean. Prototype similarity must prefer the true match
    # (cos(F,F)=1 > cos(.,M)=0.707); naive mean pooling prefers the imposter
    # (cos(mean,F)=0.707 < cos(mean,M)=1.0) — the regression this module fixes.
    s = 1 / np.sqrt(2)
    ff = _ff(
        [1, 1, 2, 3],
        [0, 1, 10, 10],
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]], [[s, s, 0.0]]],
    )
    reps = build_representations(ff, view_threshold=0.5)
    same_player = pair_similarity(reps[1], reps[2])
    imposter = pair_similarity(reps[1], reps[3])
    assert same_player == pytest.approx(1.0)
    assert imposter == pytest.approx(s)
    assert same_player > imposter

    # The naive single-mean comparison, computed inline, ranks them wrongly.
    mean1 = np.array([0.5, 0.5, 0.0])
    mean1 /= np.linalg.norm(mean1)
    naive_same = float(mean1 @ np.array([1.0, 0.0, 0.0]))
    naive_imposter = float(mean1 @ np.array([s, s, 0.0]))
    assert naive_same < naive_imposter


def test_part_aware_scoring_ignores_parts_invisible_on_either_side():
    def _rep(embs, vis):
        return build_representations(
            FrameFeatures(
                tracklet_ids=np.array([1], dtype=np.int64),
                frame_idxs=np.array([0], dtype=np.int64),
                embeddings=np.array([embs], dtype=np.float32),  # (1, P, D)
                visibility=np.array([vis], dtype=np.float32),
                keypoints_xyc=np.zeros((1, 17, 3), dtype=np.float32),
                keypoints_conf=np.ones(1, dtype=np.float32),
            )
        )[1]

    # Part 0 matches, part 1 is orthogonal. With part 1 invisible on side a,
    # only part 0 is compared -> perfect score. Making part 1 visible on both
    # sides pulls the score down to the visibility-weighted mean (0.5).
    a_hidden = _rep([[1.0, 0.0], [1.0, 0.0]], [1.0, 0.1])
    a_seen = _rep([[1.0, 0.0], [1.0, 0.0]], [1.0, 1.0])
    b = _rep([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0])

    assert pair_similarity(a_hidden, b, min_part_visibility=0.3) == pytest.approx(1.0)
    assert pair_similarity(a_seen, b, min_part_visibility=0.3) == pytest.approx(0.5)


def test_no_shared_visible_parts_returns_none():
    def _rep(vis):
        return build_representations(
            FrameFeatures(
                tracklet_ids=np.array([1], dtype=np.int64),
                frame_idxs=np.array([0], dtype=np.int64),
                embeddings=np.ones((1, 2, 2), dtype=np.float32),
                visibility=np.array([vis], dtype=np.float32),
                keypoints_xyc=np.zeros((1, 17, 3), dtype=np.float32),
                keypoints_conf=np.ones(1, dtype=np.float32),
            )
        )[1]

    a = _rep([1.0, 0.0])  # only part 0 visible
    b = _rep([0.0, 1.0])  # only part 1 visible
    assert pair_similarity(a, b, min_part_visibility=0.3) is None
