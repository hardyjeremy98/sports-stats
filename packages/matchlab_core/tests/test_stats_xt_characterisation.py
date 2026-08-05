"""Characterisation tests for xT against the real FOOTPASS split.

Separated from the structural tests by file, per a Tier 1 review finding: the
structural suite pins *behaviour that should hold for any correct model*, and it
is weak (Tier 1 measured 8/20 coefficient mutations caught). This file pins
**the actual numbers this code produces on the actual data**, which is what
catches a model that is quietly different rather than quietly broken.

Note honestly what a characterisation test is and is not: it catches **change**,
not **error**, and it cannot fail on the commit that creates it. Per the PRD it
is a regression tripwire; §12's real validation is the mutation run.

The headline finding pinned here
---------------------------------
`test_success_only_arm_is_degenerate_on_this_data` is the one test in this file
that is a *result* rather than a tripwire. On the train split the success-only
(Singh) arm produces a surface that is nearly **constant** across 12 of the 16
length bands, while the absorbing-failure (socceraction) arm is monotone from
0.006 at a team's own goal to 0.451 at the goal it attacks. That is the
predictable consequence of a transition matrix whose rows sum to 1: possession
never ends, so every zone inherits almost the same long-run scoring probability
and the surface carries very little positional information. It resolves the fork
on this data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TACTICAL = Path("data/footpass/tactical/train_tactical_data.h5")
PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_train.json")
VAL_TACTICAL = Path("data/footpass/tactical/val_tactical_data.h5")
VAL_PLAYBYPLAY = Path("data/reference/FOOTPASS/playbyplay_GT/playbyplay_val.json")

pytestmark = pytest.mark.skipif(
    not (TACTICAL.exists() and PLAYBYPLAY.exists()),
    reason="FOOTPASS train data not present (gitignored, double-gated acquisition)",
)


@pytest.fixture(scope="module")
def train_models():
    from matchlab_core.stats.xt import FailureModel
    from matchlab_train.datasets.footpass_xt import cached_fit

    return {fm: cached_fit("train", failure_model=fm) for fm in FailureModel}


@pytest.fixture(scope="module")
def val_half():
    from matchlab_core.stats.chains import build_chains
    from matchlab_train.datasets.footpass_events import load_half_events

    events, _ = load_half_events(
        VAL_TACTICAL, "game_18_H1", VAL_PLAYBYPLAY, with_offball=False
    )
    return build_chains(events)


def _centre_channel(model):
    g = model.grid
    return [model.xt[(g.ny // 2) * g.nx + ix] for ix in range(g.nx)]


# --------------------------------------------------------------------------
# The fit, pinned
# --------------------------------------------------------------------------


def test_train_fit_action_counts(train_models):
    """Live (replay-filtered) counts. These are the denominators every §12
    claim rests on, so a change in the adapter must surface here."""
    from matchlab_core.stats.xt import FailureModel

    d = train_models[FailureModel.SOCCERACTION].diagnostics
    assert d.n_moves == 72972
    assert d.n_shots == 1093
    assert d.n_unknown_excluded == 3738
    assert d.n_no_end_point == 0


def test_shots_are_the_sparse_quantity_not_moves(train_models):
    """The plan's first draft argued the opposite and was wrong.

    139 of 192 zones contain zero shots, while moves are ample everywhere. That
    is why `s(z)`/`g(z)` are handled differently from `T`, and why coarsening the
    grid does not fix it -- shots are spatially *concentrated*, not uniformly
    sparse.
    """
    from matchlab_core.stats.xt import FailureModel

    d = train_models[FailureModel.SOCCERACTION].diagnostics
    assert d.zones_with_no_shots == 139
    assert d.zones_with_no_actions == 0
    assert d.zones_shrunk == 0  # every zone clears the support floor for moves


def test_absorbing_failure_arm_leaks_in_every_zone_so_the_contraction_holds_here(
    train_models,
):
    """Measured, not assumed -- the withdrawn contraction claim, checked.

    A zero here would mean the contraction argument fails for some zone. It does
    not on this fit; that is an empirical property of these counts, and the
    diagnostic exists so a future refit cannot lose it silently.
    """
    from matchlab_core.stats.xt import FailureModel

    d = train_models[FailureModel.SOCCERACTION].diagnostics
    assert d.n_zones_zero_leakage == 0
    assert d.min_leakage == pytest.approx(0.0366, abs=5e-4)
    assert d.converged and d.iterations == 46


def test_success_only_arm_has_no_leakage_and_needs_far_more_iterations(train_models):
    from matchlab_core.stats.xt import FailureModel

    d = train_models[FailureModel.SINGH].diagnostics
    assert d.n_zones_zero_leakage == 191  # of 192
    assert d.min_leakage == 0.0
    assert d.converged and d.iterations == 160


# --------------------------------------------------------------------------
# The result
# --------------------------------------------------------------------------


def test_absorbing_failure_surface_is_monotone_toward_the_attacked_goal(train_models):
    from matchlab_core.stats.xt import FailureModel

    centre = _centre_channel(train_models[FailureModel.SOCCERACTION])
    assert all(a <= b for a, b in zip(centre, centre[1:], strict=False))
    assert centre[0] == pytest.approx(0.0058, abs=5e-4)
    assert centre[-1] == pytest.approx(0.4511, abs=5e-4)


def test_success_only_arm_is_degenerate_on_this_data(train_models):
    """The finding, pinned so it cannot quietly stop being true.

    With T's rows summing to 1 the chain has no absorbing state: possession
    never ends, so the value function collapses toward a constant and the
    surface stops carrying positional information. Concretely, the first 12
    length bands -- three quarters of the pitch -- span a range of 0.012 around
    ~0.118, against a full-pitch range of 0.445 for the absorbing-failure arm.
    """
    from matchlab_core.stats.xt import FailureModel

    centre = _centre_channel(train_models[FailureModel.SINGH])
    flat = centre[:12]
    assert max(flat) - min(flat) < 0.02
    assert min(flat) > 0.10
    # And it is not even monotone, unlike the other arm: the surface DIPS at
    # band 12 (0.091) below a team's own goal-mouth value (0.123).
    assert not all(a <= b for a, b in zip(centre, centre[1:], strict=False))
    assert centre[12] < centre[0]

    socc = _centre_channel(train_models[FailureModel.SOCCERACTION])
    assert (max(socc) - min(socc)) > 10 * (max(flat) - min(flat))


def test_the_two_arms_disagree_substantially_so_the_fork_is_resolvable_here(
    train_models,
):
    """Pre-registered criterion (PRD R6): the arms are declared equivalent only
    if per-zone agreement is tight. It is not -- so the choice matters and must
    be justified rather than defaulted."""
    from matchlab_core.stats.xt import FailureModel

    a = train_models[FailureModel.SOCCERACTION].xt
    b = train_models[FailureModel.SINGH].xt
    assert max(abs(x - y) for x, y in zip(a, b, strict=True)) > 0.1


def test_g_is_identical_across_arms_because_it_is_a_geometric_prior(train_models):
    from matchlab_core.stats.xt import FailureModel

    assert (
        train_models[FailureModel.SOCCERACTION].g == train_models[FailureModel.SINGH].g
    )
    assert "geometric prior" in train_models[FailureModel.SOCCERACTION].g_source


# --------------------------------------------------------------------------
# Applying the train grid to val (out-of-sample at the match level)
# --------------------------------------------------------------------------


def test_val_credit_coverage_is_reported_not_assumed(train_models, val_half):
    from matchlab_core.stats.xt import FailureModel, credit_actions

    model = train_models[FailureModel.SOCCERACTION]
    credits = credit_actions(model, val_half.events)
    assert len(credits) == 839
    rated = [c for c in credits if c.delta is not None]
    assert len(rated) == 720
    # 14.2% of move actions carry no rating. Visible, not silently dropped.
    assert 0.10 < 1 - len(rated) / len(credits) < 0.20


def test_the_credit_convention_changes_the_player_ranking(train_models, val_half):
    """Not a detail, and the reason both conventions ship.

    The top player by the non-negative convention is not the top player by the
    risk-adjusted one, and **17 of 23 players on this half have a negative
    risk-adjusted total** -- i.e. most players give up more xT through failed
    moves than they create through successful ones. A player card built on the
    non-negative total alone would show all 23 as net-positive contributors,
    which is an artefact of the convention, not a fact about the match.
    """
    from matchlab_core.stats.xt import (
        FailureModel,
        aggregate_by_player,
        credit_actions,
    )

    model = train_models[FailureModel.SOCCERACTION]
    lines = aggregate_by_player(credit_actions(model, val_half.events))
    by_total = sorted(lines.values(), key=lambda p: -p.xt_total)
    by_risk = sorted(lines.values(), key=lambda p: -p.xt_risk_adjusted)
    assert by_total[0].player_id == 203
    assert by_risk[0].player_id == 102
    assert by_total[0].player_id != by_risk[0].player_id
    # Note the successes-only total is NOT itself non-negative -- a completed
    # backward pass has a negative delta. What the convention omits is the debit
    # for *failures*, and that omission moves far more players than the
    # backward-pass effect does: 4 of 23 are negative before the debit, 17 after.
    assert sum(1 for p in lines.values() if p.xt_total < 0.0) == 4
    assert sum(1 for p in lines.values() if p.xt_risk_adjusted < 0.0) == 17


def test_player_lines_are_keyed_by_match_and_player(train_models, val_half):
    from matchlab_core.stats.xt import (
        FailureModel,
        aggregate_by_player,
        credit_actions,
    )

    model = train_models[FailureModel.SOCCERACTION]
    lines = aggregate_by_player(credit_actions(model, val_half.events))
    assert len(lines) == 23
    assert all(isinstance(k, tuple) and len(k) == 2 for k in lines)
    assert {k[0] for k in lines} == {"game_18"}


def test_location_only_xg_reproduces_xg_on_real_val_shots(val_half):
    """The claim `xt_shotvalue.py` makes, actually checked.

    That module asserts that on FOOTPASS `xg()` is already a function of
    location alone, so evaluating a bare synthetic shot at a point reproduces
    exactly what `xg()` returns for a real shot there. That is the whole
    justification for the Tier 3 guard's narrow signature -- if it were false,
    the guard would be silently discarding real signal rather than only
    discarding a leak.

    An earlier version of this suite left that claim in a docstring with no test
    behind it, which is the exact shape of unfalsifiable reassurance this branch
    exists to catch.
    """
    from matchlab_core.pitch import FIFA_PITCH
    from matchlab_core.stats.schema import StatEventType
    from matchlab_core.stats.xg import xg
    from matchlab_core.stats.xt_shotvalue import location_only_xg

    shots = [e for e in val_half.events if e.type is StatEventType.SHOT]
    assert shots, "no shots in the val half -- the test would be vacuous"
    for shot in shots:
        assert location_only_xg(shot.start, FIFA_PITCH) == pytest.approx(
            xg(shot, pitch=FIFA_PITCH)
        )


def test_off_ball_context_would_change_xg_which_is_why_the_guard_matters(val_half):
    """The disconfirming half: show the guard is not vacuous.

    If off-ball context never changed `xg()`, the Tier 3 guard would be
    protecting against nothing. Loading opponents into a real shot must move its
    value, or the previous test proves only that the feature is dead.
    """
    from matchlab_core.pitch import FIFA_PITCH
    from matchlab_core.stats.schema import PitchPoint, StatEventType
    from matchlab_core.stats.xg import xg

    shot = next(e for e in val_half.events if e.type is StatEventType.SHOT)
    crowded = shot.model_copy(
        update={
            "opponents": [
                PitchPoint(x=FIFA_PITCH.length - 200.0, y=FIFA_PITCH.width / 2 + d)
                for d in (-150.0, -50.0, 50.0, 150.0)
            ]
        }
    )
    assert xg(crowded, pitch=FIFA_PITCH) != pytest.approx(xg(shot, pitch=FIFA_PITCH))
