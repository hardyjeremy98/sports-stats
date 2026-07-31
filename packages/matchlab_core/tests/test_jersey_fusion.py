"""Unit tests for the pure parts of the jersey-OCR fusion ablation (task 7):
alignment verification and the fused-degrades-to-body property. No GPU, no
cache file, no npz -- everything here is synthetic dict/array plumbing."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.jersey_fusion import (
    assert_do_no_harm,
    fuse_sum,
    fuse_weighted,
    pooled_pairs,
    verify_fragment_alignment,
    veto_impact,
)


def test_alignment_passes_when_cache_and_npz_agree():
    cache_rows = [
        {"clip": "SNMOT-116.mp4", "frag": 1, "gt_track": 1, "gt": 4},
        {"clip": "SNMOT-116.mp4", "frag": 2, "gt_track": 3, "gt": 7},
    ]
    npz_meta_by_clip = {
        "SNMOT-116": {"gt_track_by_fragment": {"1": 1, "2": 3}},
    }
    verify_fragment_alignment(cache_rows, npz_meta_by_clip)  # must not raise


def test_alignment_fails_loudly_on_mismatch():
    cache_rows = [{"clip": "SNMOT-116.mp4", "frag": 1, "gt_track": 1, "gt": 4}]
    npz_meta_by_clip = {"SNMOT-116": {"gt_track_by_fragment": {"1": 99}}}
    with pytest.raises(ValueError, match="Fragment alignment check FAILED"):
        verify_fragment_alignment(cache_rows, npz_meta_by_clip)


def test_alignment_fails_loudly_on_missing_clip():
    cache_rows = [{"clip": "SNMOT-999.mp4", "frag": 1, "gt_track": 1, "gt": 4}]
    with pytest.raises(ValueError, match="Fragment alignment check FAILED"):
        verify_fragment_alignment(cache_rows, {})


def test_pooled_pairs_never_crosses_clips():
    keys = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    labels = {k: k[0] for k in keys}
    pairs = pooled_pairs(keys, labels)
    assert set(pairs) == {(("A", 1), ("A", 2)), (("B", 1), ("B", 2))}


def test_fuse_sum_is_additive_with_zero_default():
    body = {("A", 1, 2): 1.5}
    jersey = {("A", 1, 2): -0.5, ("A", 2, 3): 3.0}  # second key has no body evidence
    fused = fuse_sum(body, jersey)
    assert fused[("A", 1, 2)] == pytest.approx(1.0)
    assert fused[("A", 2, 3)] == pytest.approx(3.0)  # 0.0 (missing body) + 3.0


def test_fused_degrades_to_body_when_jersey_abstains():
    """The spec's degeneration property: jersey abstentions are exactly 0, so
    fused must equal body alone on every such pair, for BOTH fusion arms."""
    body = {("A", 1, 2): 2.3, ("A", 1, 3): -1.1}
    jersey = {("A", 1, 2): 0.0, ("A", 1, 3): 0.0}
    fused_sum = fuse_sum(body, jersey)
    fused_w = fuse_weighted(body, jersey, weights=np.array([1.7, 0.4]))
    for k in body:
        assert fused_sum[k] == pytest.approx(body[k])
        assert fused_w[k] == pytest.approx(1.7 * body[k])  # weighted body scale, not raw


def test_assert_do_no_harm_passes_on_abstained_pairs():
    body = {("A", 1, 2): 2.3}
    jersey = {("A", 1, 2): 0.0}
    fused = fuse_sum(body, jersey)
    checked = assert_do_no_harm(body, jersey, fused)
    assert checked == 1


def test_assert_do_no_harm_passes_on_weighted_arm_with_body_scale():
    body = {("A", 1, 2): 2.3}
    jersey = {("A", 1, 2): 0.0}
    weights = np.array([1.7, 0.4])
    fused = fuse_weighted(body, jersey, weights)
    checked = assert_do_no_harm(body, jersey, fused, body_scale=float(weights[0]))
    assert checked == 1


def test_assert_do_no_harm_raises_on_a_genuine_violation():
    body = {("A", 1, 2): 2.3}
    jersey = {("A", 1, 2): 0.0}
    fused = {("A", 1, 2): 9.9}  # deliberately broken fusion
    with pytest.raises(AssertionError, match="do-no-harm violated"):
        assert_do_no_harm(body, jersey, fused)


def test_veto_impact_counts_correct_and_false_vetoes():
    labels = {1: "x", 2: "x", 3: "y"}
    body = {(1, 2): 1.0, (1, 3): 1.0}  # both would merge at threshold 0.5
    fused = {(1, 2): -5.0, (1, 3): 1.0}  # jersey vetoes the (1,2) pair only
    out = veto_impact(body, fused, labels, body_threshold=0.5)
    assert out["body_merges_at_threshold"] == 2
    assert out["still_merges_when_fused"] == 1
    assert out["false_vetoes"] == 1  # (1, 2) was actually correct (same label)
    assert out["correct_vetoes"] == 0
