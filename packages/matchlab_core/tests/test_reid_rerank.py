"""k-reciprocal re-ranking (SPO-85 amendment #2).

The load-bearing test is `test_reranking_demotes_a_lookalike_with_a_different
_neighbourhood`: it constructs the exact failure mode re-ranking is supposed to
fix — an impostor that is close in direct similarity but keeps different company
— and asserts the ranking flips. Without that, the other tests would only prove
the function returns numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.reid.rerank import k_reciprocal_rerank


def _sym(rows: list[list[float]]) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float64)
    return (a + a.T) / 2.0


def test_lambda_one_returns_the_original_affinities_exactly():
    # The control arm: any deviation here invalidates every comparison made
    # against the plain-affinity baseline.
    a = _sym([[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]])
    out = k_reciprocal_rerank(a, lambda_value=1.0)
    np.testing.assert_array_equal(out, a)


def test_output_is_square_symmetric_and_finite():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 1.0, (8, 8))
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 1.0)
    out = k_reciprocal_rerank(a, k1=3, k2=2, lambda_value=0.3)
    assert out.shape == a.shape
    np.testing.assert_allclose(out, out.T, atol=1e-12)
    assert np.isfinite(out).all()


def test_missing_evidence_stays_missing():
    a = _sym([[1.0, 0.9, np.nan], [0.9, 1.0, 0.4], [np.nan, 0.4, 1.0]])
    out = k_reciprocal_rerank(a, k1=2, k2=1, lambda_value=0.3)
    assert np.isnan(out[0, 2]) and np.isnan(out[2, 0])
    assert not np.isnan(out[0, 1])


def test_reranking_demotes_a_lookalike_with_a_different_neighbourhood():
    """Query 0's true partner is 1. Impostor 2 scores HIGHER directly (0.95 vs
    0.90) but belongs to a tight cluster {2,3,4} that excludes 0, while 1 shares
    0's neighbourhood. Re-ranking should reverse the order; plain affinity
    cannot."""
    n = 5
    a = np.full((n, n), 0.30)
    np.fill_diagonal(a, 1.0)

    def put(i, j, v):
        a[i, j] = a[j, i] = v

    put(0, 1, 0.90)  # true partner
    put(0, 2, 0.95)  # impostor: closer directly
    # 0 and 1 share company
    put(0, 3, 0.35)
    put(1, 3, 0.35)
    # 2 sits inside a tight cluster with 3 and 4 that 0 is far from
    put(2, 3, 0.97)
    put(2, 4, 0.97)
    put(3, 4, 0.97)
    put(0, 4, 0.30)
    put(1, 4, 0.30)
    put(1, 2, 0.30)

    assert a[0, 2] > a[0, 1], "fixture must start with the impostor winning"
    out = k_reciprocal_rerank(a, k1=2, k2=1, lambda_value=0.0)
    assert out[0, 1] > out[0, 2], "re-ranking should promote the true partner"


def test_plain_affinity_keeps_the_lookalike_on_top():
    """The disconfirming half: the same fixture under lambda=1.0 (no
    re-ranking) must still prefer the impostor, so the test above is measuring
    re-ranking rather than a fixture artefact."""
    n = 5
    a = np.full((n, n), 0.30)
    np.fill_diagonal(a, 1.0)
    a[0, 1] = a[1, 0] = 0.90
    a[0, 2] = a[2, 0] = 0.95
    out = k_reciprocal_rerank(a, lambda_value=1.0)
    assert out[0, 2] > out[0, 1]


def test_tiny_matrices_are_returned_unchanged():
    a = _sym([[1.0, 0.7], [0.7, 1.0]])
    np.testing.assert_array_equal(k_reciprocal_rerank(a, lambda_value=0.3), a)


def test_non_square_input_is_rejected():
    with pytest.raises(ValueError, match="square"):
        k_reciprocal_rerank(np.zeros((2, 3)))
