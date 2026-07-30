import numpy as np
import pytest
from matchlab_core.reid.evidence import LOG_CLAMP
from matchlab_core.reid.jersey import (
    DIGITS,
    EOS,
    N_NUMBERS,
    crop_number_logprobs,
    number_prior,
    pair_llr,
    tracklet_likelihood,
    uniform_prior,
)


def _probs(rows: list[dict[int, float]]) -> np.ndarray:
    """Build an (n_rows, 11) char-prob matrix from {column: prob} per row."""
    m = np.zeros((len(rows), DIGITS + 1), dtype=np.float64)
    for i, row in enumerate(rows):
        for col, p in row.items():
            m[i, col] = p
    return m


def test_confident_single_digit_puts_mass_on_that_number():
    # "7" then end-of-string.
    lp = crop_number_logprobs(_probs([{7: 1.0}, {EOS: 1.0}, {EOS: 1.0}]))
    assert lp.shape == (N_NUMBERS,)
    assert int(np.argmax(lp)) == 7


def test_confident_double_digit_puts_mass_on_that_number():
    # "2" "3" then end-of-string.
    lp = crop_number_logprobs(_probs([{2: 1.0}, {3: 1.0}, {EOS: 1.0}]))
    assert int(np.argmax(lp)) == 23


def test_leading_zero_is_not_a_number_string():
    # "0" "7" is impossible: 7 is written "7", never "07".
    lp = crop_number_logprobs(_probs([{0: 1.0}, {7: 1.0}, {EOS: 1.0}]))
    # Mass must not land on 7 via a two-digit route: numbers 10..99 are indexed
    # by a leading digit of 1..9, so "07" has no slot at all.
    assert int(np.argmax(lp)) == 0


def test_single_digit_prior_reweights_length_classes():
    # Ambiguous between "1" and "12": equal network mass on EOS and "2".
    rows = _probs([{1: 1.0}, {EOS: 0.5, 2: 0.5}, {EOS: 1.0}])
    heavy_single = crop_number_logprobs(rows, single_digit_prior=0.9)
    heavy_double = crop_number_logprobs(rows, single_digit_prior=0.1)
    assert int(np.argmax(heavy_single)) == 1
    assert int(np.argmax(heavy_double)) == 12


def test_prior_none_trusts_the_network_length_belief():
    rows = _probs([{1: 1.0}, {EOS: 0.9, 2: 0.1}, {EOS: 1.0}])
    lp = crop_number_logprobs(rows, single_digit_prior=None)
    assert int(np.argmax(lp)) == 1


def test_too_few_rows_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError):
        crop_number_logprobs(_probs([{7: 1.0}, {EOS: 1.0}]))


def _peaked(n: int, mass: float = 0.999) -> np.ndarray:
    """A tracklet likelihood concentrated on number `n`."""
    v = np.full(N_NUMBERS, (1.0 - mass) / (N_NUMBERS - 1))
    v[n] = mass
    return v


def test_no_crops_is_a_flat_likelihood():
    v = tracklet_likelihood(np.zeros((0, N_NUMBERS)), np.zeros(0))
    assert np.allclose(v, 1.0 / N_NUMBERS)


def test_zero_weight_crops_are_flat_not_confident():
    lp = crop_number_logprobs(_probs([{7: 1.0}, {EOS: 1.0}, {EOS: 1.0}]))
    v = tracklet_likelihood(lp[None, :], np.zeros(1))
    assert np.allclose(v, 1.0 / N_NUMBERS)


def test_agreeing_crops_sharpen_the_tracklet_likelihood():
    lp = crop_number_logprobs(_probs([{7: 0.6, 3: 0.4}, {EOS: 1.0}, {EOS: 1.0}]))
    one = tracklet_likelihood(lp[None, :], np.ones(1))
    five = tracklet_likelihood(np.repeat(lp[None, :], 5, axis=0), np.ones(5))
    assert five[7] > one[7]


def test_illegible_pair_is_exactly_neutral():
    """ADR 001/003 abstention, produced by the algebra rather than a gate."""
    flat = np.full(N_NUMBERS, 1.0 / N_NUMBERS)
    assert pair_llr(flat, flat, uniform_prior()) == pytest.approx(0.0, abs=1e-9)


def test_agreement_is_positive_evidence():
    assert pair_llr(_peaked(7), _peaked(7), uniform_prior()) > 3.0


def test_disagreement_is_strong_negative_evidence():
    """The property no other channel has: this one can veto a merge."""
    assert pair_llr(_peaked(7), _peaked(9), uniform_prior()) < -3.0


def test_agreement_on_a_common_number_is_weaker_than_on_a_rare_one():
    prior = number_prior([7] * 100 + [23])
    common = pair_llr(_peaked(7), _peaked(7), prior)
    rare = pair_llr(_peaked(23), _peaked(23), prior)
    assert rare > common


def test_one_sided_evidence_is_near_neutral():
    """A confident read against an unreadable partner must not merge them."""
    flat = np.full(N_NUMBERS, 1.0 / N_NUMBERS)
    assert abs(pair_llr(_peaked(7), flat, uniform_prior())) < 0.05


def test_llr_is_bounded_so_the_channel_cannot_veto_absolutely():
    extreme = pair_llr(_peaked(7, mass=1.0 - 1e-15), _peaked(9, mass=1.0 - 1e-15),
                       uniform_prior())
    assert extreme > -LOG_CLAMP - 1e-9


def test_number_prior_is_laplace_smoothed_and_normalised():
    prior = number_prior([7, 7, 23])
    assert prior.shape == (N_NUMBERS,)
    assert prior.sum() == pytest.approx(1.0)
    assert prior.min() > 0.0          # unobserved != impossible
    assert prior[7] > prior[23] > prior[50]


def test_temperature_damps_overconfident_crops():
    lp = crop_number_logprobs(_probs([{7: 0.6, 3: 0.4}, {EOS: 1.0}, {EOS: 1.0}]))
    crops = np.repeat(lp[None, :], 10, axis=0)
    hot = tracklet_likelihood(crops, np.ones(10), temperature=5.0)
    cold = tracklet_likelihood(crops, np.ones(10), temperature=1.0)
    assert cold[7] > hot[7]
