"""Formation-aware role assignment: prediction, joint assignment, round-trip."""

from __future__ import annotations

import numpy as np
import pytest
from matchlab_core.formation import (
    FormationModel,
    assign_team,
    assign_team_roles,
    fit_formation_model,
    outfield_inventory,
    predict_inventory,
)
from matchlab_core.formation.roles import GK_ROLE, N_ROLES

#: Two formations that differ by ONE role, as the two dominant FOOTPASS classes
#: do (attacking versus defensive midfielder). Roles are FOOTPASS ids.
#: 2 LB, 3 LCB, 5 RCB, 13 RB, 6 LM, 7 RM, 9 AM, 8 DM, 10 LW, 11 RW, 12 CF
ANCHORS = {
    2: (-1.2, -1.0), 3: (-1.3, -0.35), 5: (-1.3, 0.35), 13: (-1.2, 1.0),
    6: (-0.1, -0.9), 7: (-0.1, 0.9),
    9: (0.5, 0.0),    # AM, ahead of the midfield line
    8: (-0.7, 0.0),   # DM, behind it
    10: (1.1, -0.8), 11: (1.1, 0.8), 12: (1.3, 0.0),
    GK_ROLE: (-2.2, 0.0),
}
INV_AM = (2, 3, 5, 6, 7, 9, 10, 11, 12, 13)
INV_DM = (2, 3, 5, 6, 7, 8, 10, 11, 12, 13)


def _samples(inv, n_half, *, per_role=80, sigma=0.18, seed=0, with_gk=True):
    """Labelled per-sample training data for `n_half` team-halves of `inv`."""
    rng = np.random.default_rng(seed)
    feats, roles, half = [], [], []
    for h in range(n_half):
        for r in (*inv, *((GK_ROLE,) if with_gk else ())):
            feats.append(rng.normal(ANCHORS[r], sigma, (per_role, 2)))
            roles.append(np.full(per_role, r))
            half.append(np.full(per_role, f"{inv[5]}-{h}"))
    return (
        np.concatenate(feats), np.concatenate(roles), np.concatenate(half),
    )


def _model(*, n_am=8, n_dm=4, seed=0) -> FormationModel:
    fa, ra, ha = _samples(INV_AM, n_am, seed=seed)
    fb, rb, hb = _samples(INV_DM, n_dm, seed=seed + 100)
    return fit_formation_model(
        np.concatenate([fa, fb]), np.concatenate([ra, rb]),
        np.concatenate([ha, hb]), min_per_role=20,
    )


def _slot_means(inv, *, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([ANCHORS[r] for r in inv]) + rng.normal(0, jitter, (len(inv), 2))


def test_outfield_inventory_is_a_set_not_a_multiset():
    """Substitutes inherit a slot, so a 12-thread half still fields ten."""
    roles = [2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 9, 12, GK_ROLE]
    assert outfield_inventory(roles) == INV_AM


def test_predicts_the_formation_it_was_shown():
    m = _model()
    for inv in (INV_AM, INV_DM):
        got = predict_inventory(m, _slot_means(inv))
        assert got.ok and got.inventory == inv, got.reason


def test_the_prior_decides_when_the_geometry_does_not():
    """With an uninformative descriptor the prevalent class must win.

    This is the regression guard for the defect that made an earlier version of
    this work report a sub-baseline number: scored without a prior, the argmax
    selects on VARIANCE and prefers the class fitted from few samples.
    """
    m = _model(n_am=20, n_dm=2)
    # Built from the model's OWN fitted templates, so it is exactly equidistant
    # from the two candidates and the likelihood genuinely cannot choose. The
    # midpoint of the underlying ANCHORS is not: templates are fitted from
    # noisy samples, and the rare class's mean is the noisier of the two, so
    # that version tests a decision the geometry can still make.
    ambiguous = 0.5 * (np.stack([m.templates.means[r] for r in INV_AM])
                       + np.stack([m.templates.means[r] for r in INV_DM]))
    scores = predict_inventory(m, ambiguous).scores
    assert scores[INV_AM] > scores[INV_DM]
    assert predict_inventory(m, ambiguous).inventory == INV_AM


def test_a_rare_class_can_still_win_on_strong_geometry():
    """The prior must not be an override. If it were, the predictor could never
    beat a constant guess -- which is the only thing it is FOR."""
    m = _model(n_am=20, n_dm=2)
    assert predict_inventory(m, _slot_means(INV_DM)).inventory == INV_DM


def test_wrong_slot_count_falls_back_to_the_modal_formation():
    m = _model()
    got = predict_inventory(m, _slot_means(INV_AM)[:7])
    assert got.inventory == m.modal_inventory
    assert "7 roles" in got.reason or "fell back" in got.reason


def test_prediction_is_invariant_to_translation():
    """Slot means come from a different estimator than the templates, so a
    constant offset between the two must not choose the formation."""
    m = _model()
    shifted = _slot_means(INV_DM) + np.array([0.4, -0.3])
    assert predict_inventory(m, shifted).inventory == INV_DM


def test_assignment_respects_the_predicted_inventory():
    """A role outside the inventory must never be emitted."""
    m = _model()
    feats = np.stack([ANCHORS[r] for r in INV_DM])
    got = assign_team_roles(
        m, feats, np.full(len(feats), 50), left_to_right=True, inventory=INV_DM
    )
    assert all(g.ok for g in got)
    assert {g.role for g in got} <= set(INV_DM)
    assert 9 not in {g.role for g in got}  # AM is not fielded here


def test_capacity_two_lets_a_substitute_share_a_slot():
    """A substitution puts a second thread in the same tactical slot. Capacity
    1 would force one of them into a different, wrong role."""
    m = _model()
    feats = np.concatenate([
        np.stack([ANCHORS[r] for r in INV_AM]),
        ANCHORS[12] + np.zeros((1, 2)),   # a second centre forward
    ])
    got = assign_team_roles(
        m, feats, np.full(len(feats), 50), left_to_right=True,
        inventory=INV_AM, capacity=2,
    )
    assert [g.role for g in got].count(12) == 2


def test_capacity_one_is_a_strict_bijection():
    m = _model()
    feats = np.stack([ANCHORS[r] for r in INV_AM])
    got = assign_team_roles(
        m, feats, np.full(len(feats), 50), left_to_right=True,
        inventory=INV_AM, capacity=1,
    )
    assert sorted(g.role for g in got) == sorted(INV_AM)


def test_more_threads_than_slots_falls_back_instead_of_discarding():
    """A rectangular assignment would leave rows unmatched and score them
    wrong, penalising smaller inventories asymmetrically."""
    m = _model()
    feats = np.tile(np.stack([ANCHORS[r] for r in INV_AM]), (3, 1))  # 30 threads
    got = assign_team_roles(
        m, feats, np.full(len(feats), 50), left_to_right=True,
        inventory=INV_AM, capacity=2,
    )
    assert all(g.ok for g in got), "no thread may be silently dropped"


def test_short_threads_abstain_and_do_not_consume_a_slot():
    """A stub thread occupying a slot denies it to a thread that can use it."""
    m = _model()
    feats = np.stack([ANCHORS[r] for r in INV_AM])
    n = np.full(len(feats), 50)
    n[0] = 2
    got = assign_team_roles(
        m, feats, n, left_to_right=True, inventory=INV_AM, capacity=1
    )
    assert not got[0].ok and "too short" in got[0].reason
    assert got[0].role is None
    # The role that thread would have taken is still available to the others.
    assert len({g.role for g in got if g.ok}) == len(feats) - 1


def test_undefined_features_abstain():
    m = _model()
    feats = np.stack([ANCHORS[r] for r in INV_AM])
    feats[3] = np.nan
    got = assign_team_roles(
        m, feats, np.full(len(feats), 50), left_to_right=True, inventory=INV_AM
    )
    assert not got[3].ok and "undefined" in got[3].reason


def test_direction_flips_every_slot_but_no_role():
    """LEFT_TO_RIGHT is half the DST slot index; a wrong bit mirrors the team."""
    m = _model()
    feats = np.stack([ANCHORS[r] for r in INV_AM])
    n = np.full(len(feats), 50)
    a = assign_team_roles(m, feats, n, left_to_right=False, inventory=INV_AM)
    b = assign_team_roles(m, feats, n, left_to_right=True, inventory=INV_AM)
    assert [x.role for x in a] == [x.role for x in b]
    assert all(y.slot - x.slot == N_ROLES for x, y in zip(a, b))


def test_unrestricted_assignment_reproduces_per_thread_argmin():
    """The previous system must still be reachable, for ablation."""
    m = _model()
    feats = np.stack([ANCHORS[r] for r in INV_AM])
    n = np.full(len(feats), 50)
    joint = assign_team_roles(
        m, feats, n, left_to_right=True, inventory=None, capacity=0
    )
    for i, g in enumerate(joint):
        costs = {
            r: float((feats[i] - m.templates.means[r]) @ m.templates.inv_cov[r]
                     @ (feats[i] - m.templates.means[r]))
            for r in m.templates.means
        }
        assert g.role == min(costs, key=costs.get)


def test_assign_team_runs_prediction_and_assignment_together():
    m = _model()
    feats = np.stack([ANCHORS[r] for r in INV_DM])
    got, pred = assign_team(
        m, feats, np.full(len(feats), 50), _slot_means(INV_DM), left_to_right=True
    )
    assert pred.inventory == INV_DM
    assert {g.role for g in got if g.ok} <= set(INV_DM)


def test_fit_serve_frame_mismatch_is_refused():
    """Three fit/serve coordinate mismatches have shipped in this repo."""
    m = _model()
    with pytest.raises(ValueError, match="fitted in frame"):
        predict_inventory(m, _slot_means(INV_AM), frame="absolute")


def test_unnormalised_input_is_refused():
    m = _model()
    with pytest.raises(ValueError, match="SPREAD-NORMALISED"):
        predict_inventory(m, _slot_means(INV_AM) * 5250.0)


def test_model_round_trips_through_a_dict():
    m = _model()
    back = FormationModel.from_dict(m.to_dict())
    assert back.inventories == m.inventories
    assert back.frame == m.frame
    assert back.sigma2 == pytest.approx(m.sigma2)
    assert set(back.templates.means) == set(m.templates.means)
    got = _slot_means(INV_DM)
    assert predict_inventory(back, got).inventory == predict_inventory(m, got).inventory


def test_parallel_array_mismatch_raises():
    f, r, h = _samples(INV_AM, 2)
    with pytest.raises(ValueError, match="parallel"):
        fit_formation_model(f, r[:-1], h, min_per_role=20)
