"""Attacking direction and epoch boundary estimation."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.formation import (
    estimate_direction,
    evaluate_probes,
    probe_fn_from_arrays,
)

FPS = 25.0


def synth(
    *,
    n_frames: int = 25 * 60 * 90,
    boundary: int | None = None,
    n_per_team: int = 7,
    noise: float = 0.10,
    seed: int = 0,
    flips: tuple[int, ...] = (),
):
    """A match where club 0's centroid sits left of club 1's, flipping at
    `boundary`. `noise` is the per-frame probability the sign is inverted --
    the real failure mode, where truncated visibility hides one team's deep
    players during a siege."""
    rng = np.random.default_rng(seed)
    cuts = list(flips) if flips else ([boundary] if boundary is not None else [])
    step = 25
    frames = np.arange(0, n_frames, step, dtype=np.int64)

    f_all, x_all, c_all = [], [], []
    for f in frames:
        polarity = -1.0
        for c in cuts:
            if f >= c:
                polarity *= -1.0
        if rng.random() < noise:
            polarity *= -1.0
        # club 0 centred at 0.5 + 0.05*polarity, club 1 mirrored.
        for club in (0, 1):
            centre = 0.5 + 0.05 * polarity * (1 if club == 0 else -1)
            x = np.clip(rng.normal(centre, 0.12, n_per_team), 0.0, 1.0)
            f_all.append(np.full(n_per_team, f, dtype=np.int64))
            x_all.append(x)
            c_all.append(np.full(n_per_team, club, dtype=np.int64))

    f_all = np.concatenate(f_all)
    x_all = np.concatenate(x_all)
    c_all = np.concatenate(c_all)
    return probe_fn_from_arrays(
        f_all, x_all, c_all, observable=np.ones(len(f_all), dtype=bool)
    )


def test_clean_flip_is_recovered():
    n = 25 * 60 * 90
    b = n // 2
    est = estimate_direction(synth(n_frames=n, boundary=b), total_frames=n, fps=FPS)
    assert est.ok, est.reason
    assert abs(est.boundary - b) < 30 * FPS


def test_boundary_away_from_the_midpoint_is_recovered_and_beats_the_midpoint():
    """THE centre-bias regression guard.

    score(t) peaks at n/2 under the null and `min_half` narrows the admissible
    window toward the centre, so on an equal-halves substrate a constant
    "boundary = n/2" estimator scores near-perfectly and nothing else in the
    suite would notice. This fixture puts the truth at 30% of the timeline,
    where the midpoint estimator is demonstrably wrong.
    """
    n = 25 * 60 * 110
    b = int(n * 0.30)
    est = estimate_direction(
        synth(n_frames=n, boundary=b), total_frames=n, fps=FPS,
        min_half_seconds=10 * 60.0,
    )
    assert est.ok, est.reason
    err = abs(est.boundary - b)
    midpoint_err = abs(n // 2 - b)
    assert err < 30 * FPS
    assert err < 0.1 * midpoint_err, "no better than guessing the midpoint"


def test_no_flip_abstains_rather_than_inventing_a_boundary():
    n = 25 * 60 * 90
    est = estimate_direction(synth(n_frames=n, boundary=None), total_frames=n, fps=FPS)
    assert not est.ok
    assert est.boundary is None


def test_min_half_refuses_a_boundary_near_the_ends():
    """A flip 2 minutes in is inadmissible, so the module must ABSTAIN.

    (The earlier form of this test accepted a boundary as long as it was far
    from the truth -- i.e. it asserted the exact failure it exists to forbid,
    and passed vacuously.)
    """
    n = 25 * 60 * 90
    est = estimate_direction(
        synth(n_frames=n, boundary=int(2 * 60 * FPS)), total_frames=n, fps=FPS,
        min_half_seconds=20 * 60.0,
    )
    assert not est.ok, f"returned a confident boundary at {est.boundary}"


def test_reported_ci_actually_covers_the_error():
    """`ci_frames` is what `epoch_of_fragment` abstains inside, so an
    understated band hands the consumer a confident mirror decision for
    exactly the fragments near half-time. The bisection interval is NOT an
    error bar -- it always ends at `tolerance` however wrong the answer is."""
    n = 25 * 60 * 90
    covered = 0
    trials = 8
    for seed in range(trials):
        b = int(n * (0.35 + 0.04 * seed))
        est = estimate_direction(
            synth(n_frames=n, boundary=b, seed=seed), total_frames=n, fps=FPS,
            min_half_seconds=10 * 60.0,
        )
        if est.ok and abs(est.boundary - b) <= est.ci_frames:
            covered += 1
    assert covered >= trials - 1, f"CI covered only {covered}/{trials}"


def test_three_epochs_abstain_rather_than_returning_one_wrong_boundary():
    """Extra-time shaped input violates the two-epoch model.

    Which guard catches it is not the point -- with an odd number of flips the
    opposite-sign check usually fires first. The safety property is that the
    module never returns one confident boundary for a match that has more.
    """
    n = 25 * 60 * 120
    est = estimate_direction(
        synth(n_frames=n, flips=(int(n * 0.35), int(n * 0.7))),
        total_frames=n, fps=FPS, min_half_seconds=10 * 60.0,
    )
    assert not est.ok
    assert est.reason in {"segments do not have opposite sign"} or (
        "sub-boundary" in est.reason or "agreement" in est.reason
    )


def test_four_epochs_trip_the_sub_boundary_guard_specifically():
    """-,+,-,+ passes the opposite-sign check, so the sub-boundary guard is
    the only thing standing between this and a confident wrong answer."""
    n = 25 * 60 * 130
    est = estimate_direction(
        synth(n_frames=n, flips=(int(n * 0.4), int(n * 0.6), int(n * 0.8))),
        total_frames=n, fps=FPS, min_half_seconds=8 * 60.0,
        # Sign agreement inside a segment containing extra flips is poor, so
        # that guard fires first and would mask this one. Relax it to isolate
        # the sub-boundary check as the thing under test.
        min_agreement=0.0,
    )
    assert not est.ok
    assert "sub-boundary" in est.reason, est.reason


def test_pure_label_noise_abstains():
    n = 25 * 60 * 90
    est = estimate_direction(
        synth(n_frames=n, boundary=n // 2, noise=0.5), total_frames=n, fps=FPS
    )
    assert not est.ok


def test_coarse_to_fine_probes_a_tiny_fraction_of_the_timeline():
    n = 25 * 60 * 90
    est = estimate_direction(
        synth(n_frames=n, boundary=n // 2), total_frames=n, fps=FPS
    )
    assert est.ok
    assert est.probes_used < 0.01 * n
    assert est.probes_used < 300


def test_swapping_club_labels_keeps_the_boundary_and_inverts_direction():
    n = 25 * 60 * 90
    b = n // 2
    base = estimate_direction(synth(n_frames=n, boundary=b), total_frames=n, fps=FPS)
    inner = synth(n_frames=n, boundary=b)

    def swapped(frames):
        return [(xb, xa) for xa, xb in inner(frames)]

    other = estimate_direction(swapped, total_frames=n, fps=FPS)
    assert base.ok and other.ok
    assert abs(base.boundary - other.boundary) <= 2 * base.ci_frames
    assert base.attacks_positive_x(0, 0) is not other.attacks_positive_x(0, 0)


def test_absolute_direction_is_consistent_between_clubs_and_epochs():
    n = 25 * 60 * 90
    est = estimate_direction(synth(n_frames=n, boundary=n // 2), total_frames=n, fps=FPS)
    assert est.ok
    for epoch in (0, 1):
        assert est.attacks_positive_x(0, epoch) is not est.attacks_positive_x(1, epoch)
    assert est.attacks_positive_x(0, 0) is not est.attacks_positive_x(0, 1)


def test_absolute_direction_matches_the_fixture_truth_not_just_itself():
    """Pins the SIGN against ground truth.

    The consistency assertions above are inversion-invariant: flipping the
    `s < 0` convention in `attacks_positive_x` leaves every one of them
    passing. `synth` places club 0 left of club 1 before the flip, so club 0
    defends the left goal and therefore attacks +x in epoch 0.
    """
    n = 25 * 60 * 90
    est = estimate_direction(synth(n_frames=n, boundary=n // 2), total_frames=n, fps=FPS)
    assert est.ok
    assert est.attacks_positive_x(0, 0) is True
    assert est.attacks_positive_x(1, 0) is False
    assert est.attacks_positive_x(0, 1) is False
    assert est.attacks_positive_x(1, 1) is True


def test_bursty_asymmetric_label_noise_does_not_fabricate_a_boundary():
    """The realistic team-stage failure: a kit/lighting confusion spell where
    one club's tracks are absorbed by the other for a contiguous window.

    Unlike symmetric flips (which shrink |d| but leave the sign unbiased, so
    the detector looks robust), this OFFSETS d and can invert it. There is no
    real flip here, so any boundary at all is a fabrication.
    """
    n = 25 * 60 * 90
    inner = synth(n_frames=n, boundary=None, seed=3)
    lo, hi = int(n * 0.30), int(n * 0.55)

    def bursty(frames):
        out = []
        for f, (xa, xb) in zip(frames, inner(frames)):
            if lo <= f < hi:  # club A absorbed into club B
                out.append((np.empty(0), np.concatenate([xa, xb])))
            else:
                out.append((xa, xb))
        return out

    est = estimate_direction(bursty, total_frames=n, fps=FPS)
    assert not est.ok, f"fabricated a boundary at {est.boundary} ({est.reason})"


def test_a_blackout_around_the_boundary_widens_the_ci_instead_of_guessing():
    """A stretch of abstaining probes straddling the flip is real ambiguity.

    Measured on FOOTPASS (2026-08-05): a 20-minute one-club team-label
    collapse around half-time made every probe there abstain, leaving a hole
    and a 361 s boundary error whose z, sign agreement and separation were all
    indistinguishable from a clean match -- no confidence statistic could flag
    it. The band must widen to the hole so the consumer gets "undecidable",
    not a confident wrong epoch.
    """
    n = 25 * 60 * 90
    b = n // 2
    inner = synth(n_frames=n, boundary=b)
    lo, hi = b - int(10 * 60 * FPS), b + int(10 * 60 * FPS)

    def blacked_out(frames):
        return [
            (np.empty(0), np.empty(0)) if lo <= f < hi else v
            for f, v in zip(frames, inner(frames))
        ]

    est = estimate_direction(blacked_out, total_frames=n, fps=FPS)
    if est.ok:
        assert est.ci_frames >= int(8 * 60 * FPS), (
            f"ci {est.ci_frames / FPS:.0f}s does not reflect a 20-minute hole"
        )
        assert est.epoch_of_fragment(b - 100, b + 100) is None


def test_partially_visible_formation_relative_input_abstains_and_says_why():
    """The exact-0.5 check only catches a fully visible roster.

    With the normal 6-8 of 11 visible the subset mean is not 0.5, and sampling
    noise alone puts |d| near 0.08 -- LARGER than the ~0.043 margin a real
    match shows. So no magnitude test can separate them and nothing raises.
    The outcome must still be safe (abstain, consumer keeps its existing
    behaviour) and diagnosable (the reason names the coordinate-frame
    mistake), not a silent "the cue is dead".
    """
    rng = np.random.default_rng(0)
    n = 25 * 60 * 90
    fr, xs, cl = [], [], []
    for f in range(0, n, 250):
        for club in (0, 1):
            full = rng.normal(0.5, 0.15, 11)
            rel = full - full.mean() + 0.5          # centroid-subtracted
            vis = rng.choice(11, 7, replace=False)  # only 7 of 11 observable
            fr.append(np.full(7, f))
            xs.append(rel[vis])
            cl.append(np.full(7, club))
    fr = np.concatenate(fr)
    xs = np.concatenate(xs)
    cl = np.concatenate(cl)
    fn = probe_fn_from_arrays(fr, xs, cl, observable=np.ones(len(xs), bool))
    est = estimate_direction(fn, total_frames=n, fps=FPS, min_half_seconds=10 * 60.0)
    assert not est.ok
    assert "FORMATION-RELATIVE" in est.reason, est.reason


def test_formation_relative_input_raises_instead_of_abstaining_forever():
    frames = np.repeat(np.arange(0, 1000, 25, dtype=np.int64), 8)
    clubs = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 40)
    # centroid-subtracted: each team's mean is exactly 0.5
    xs = np.tile([0.3, 0.5, 0.5, 0.7, 0.4, 0.5, 0.5, 0.6], 40)
    fn = probe_fn_from_arrays(frames, xs, clubs, observable=np.ones(len(xs), bool))
    with pytest.raises(ValueError, match="FORMATION-RELATIVE"):
        evaluate_probes(fn, np.arange(0, 1000, 25, dtype=np.int64))


def test_metre_scale_input_raises():
    frames = np.repeat(np.arange(0, 500, 25, dtype=np.int64), 8)
    clubs = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 20)
    xs = np.tile([10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0], 20)
    fn = probe_fn_from_arrays(frames, xs, clubs, observable=np.ones(len(xs), bool))
    with pytest.raises(ValueError, match="NORMALISED"):
        evaluate_probes(fn, np.arange(0, 500, 25, dtype=np.int64))


def test_probes_below_min_per_team_abstain_and_do_not_contribute():
    frames = np.repeat(np.arange(0, 500, 25, dtype=np.int64), 3)
    clubs = np.tile([0, 0, 1], 20)  # club 1 never reaches 3 observable
    xs = np.tile([0.2, 0.3, 0.8], 20)
    fn = probe_fn_from_arrays(frames, xs, clubs, observable=np.ones(len(xs), bool))
    p = evaluate_probes(fn, np.arange(0, 500, 25, dtype=np.int64), min_per_team=3)
    assert len(p.d) == 0
    assert p.n_evaluated == 20


def test_epoch_of_fragment_returns_none_when_straddling_the_boundary():
    n = 25 * 60 * 90
    b = n // 2
    est = estimate_direction(synth(n_frames=n, boundary=b), total_frames=n, fps=FPS)
    assert est.ok
    assert est.epoch_of_fragment(0, 1000) == 0
    assert est.epoch_of_fragment(n - 1000, n - 1) == 1
    assert est.epoch_of_fragment(est.boundary - 5000, est.boundary + 5000) is None
    assert est.same_epoch((0, 1000), (100, 2000)) is True
    assert est.same_epoch((0, 1000), (n - 1000, n - 1)) is False
    assert est.same_epoch((0, 1000), (est.boundary - 5000, est.boundary + 5000)) is None


def test_abstention_propagates_to_the_consumer_bit():
    n = 25 * 60 * 90
    est = estimate_direction(synth(n_frames=n, boundary=None), total_frames=n, fps=FPS)
    assert est.same_epoch((0, 100), (n - 100, n)) is None
    assert est.attacks_positive_x(0, 0) is None
