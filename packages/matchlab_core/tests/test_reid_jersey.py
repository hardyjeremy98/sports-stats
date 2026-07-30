import numpy as np
import pytest
from matchlab_core.reid.jersey import (
    DIGITS,
    EOS,
    N_NUMBERS,
    crop_number_logprobs,
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
