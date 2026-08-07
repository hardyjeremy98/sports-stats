"""Formation-aware role assignment: the measured-best configuration, as one model.

This is `formation.roles` with two additions that were measured to pay on
FOOTPASS TRAIN (48 matches, 192 team-halves, 5 seeds x 5 folds held out BY
MATCH, merged GT threads, excluding keepers, 2026-08-06):

    argmax over all 13 roles, per thread          0.666   <- the previous system
    + capacity-2 assignment, no formation          0.677
    + constant guess of the modal formation        0.680
    + PREDICTED formation                          0.689   <- this module
    + oracle formation (ceiling)                   0.706

Decomposed, paired over match clusters:

    capacity-2 assignment rule    +0.010  [+0.003, +0.017]   needs no formation
    constant modal-formation guess +0.003 [-0.007, +0.013]   not resolvable
    actually predicting formation +0.009  [+0.002, +0.017]   21/19/8 per match
    ------------------------------------------------------
    total vs the previous system  +0.022  [+0.011, +0.033]   33/4/11 per match

Read that honestly: **nearly half the gain is the assignment rule and requires
no formation prediction at all**, and the whole effect is ~2 points. The
ceiling with a perfect formation predictor is +0.029 over the same assignment
rule, so even flawless formation knowledge caps out near 3 points. Two measured
reasons: 85% of team-halves share one shape (4 defenders, 3 midfielders, 3
attackers), so there is little for a formation to disambiguate; and the single
distinction separating the two dominant classes is attacking- versus
defensive-midfielder, which is the weakest role in this estimator to begin with.

SUPERSEDES a note in `formation.roles`
--------------------------------------
That module records "Hungarian one-to-one assignment: actively harmful". That
finding stands and is not contradicted: a STRICT bijection forces one confident
correct assignment into two wrong ones when only ~4 of 11 players are visible.
What pays here is **capacity 2** -- each role slot may take two threads,
because a substitution puts a second thread into the same slot within a half.
The earlier note was measured on 3 val matches; this on 48 train matches.

Its note that "off-camera interpolation does not help" also stands for the
per-thread FEATURE (the mean position is recovered about as well either way).
It is nonetheless required upstream here, for a different quantity: the role
representation needs frames showing all ten outfield threads at once, and with
observed rows only the fixed point gets a median of 104 such frames against
1811 with bracketed fill. Formation accuracy follows -- 0.623 observed against
0.721 filled. Use LINEAR fill: the concave TimeWarp coefficient was fitted for
the team centroid in different units and measured slightly WORSE here (0.697).

WHAT IS ORACLE IN THE MEASUREMENT, AND THEREFORE UNPROVEN IN DEPLOYMENT
-----------------------------------------------------------------------
Every figure above used GT threads (a perfect re-ID merge), GT identity through
off-camera gaps, and oracle attacking direction. Clubs also recur across the 48
matches with no global club key in the data, so a by-match split cannot
separate them and formation is largely a club property. All figures are
optimistic by an unmeasured amount. Role accuracy remains bounded by re-ID
merge quality (+0.112 perfect versus 16-piece), which is larger than everything
in this module combined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from matchlab_core.formation.roles import (
    FRAME,
    GK_ROLE,
    MIN_SAMPLES,
    N_ROLES,
    RoleAssignment,
    RoleTemplates,
    _assert_normalised_feature,
    fit_role_templates,
)

#: Threads a single role slot may hold within one half. TWO, not one: a
#: substitution puts a second thread into the same tactical slot, and a strict
#: bijection was measured to LOSE (see the module docstring).
DEFAULT_CAPACITY = 2

#: Likelihood temperature. Fitted per fold in the harness and landed on 1.0 for
#: this arm on every position arm, so 1.0 is the default rather than a knob
#: nobody set. It exists because an arm whose log-likelihood is scaled in larger
#: units silently drowns the prior and behaves as if prior-free -- the defect
#: that made an earlier version of this work report a sub-baseline number.
DEFAULT_TAU = 1.0

#: Laplace count added to every candidate inventory's prior, so a formation
#: seen only in the training fold's tail is unlikely rather than impossible.
PRIOR_SMOOTHING = 0.5


def outfield_inventory(roles) -> tuple[int, ...]:
    """The distinct OUTFIELD role slots in a team-half, ascending.

    A SET, not a multiset: substitutes inherit the outgoing player's slot, so a
    14-thread half still fields ten slots. Taking the multiset makes almost
    every team-half its own class.
    """
    return tuple(sorted({int(r) for r in roles if int(r) != GK_ROLE}))


@dataclass(frozen=True)
class InventoryPrediction:
    """A predicted formation, with the evidence that chose it."""

    inventory: tuple[int, ...]
    margin: float          # log-odds over the runner-up
    scores: dict[tuple[int, ...], float]
    reason: str

    @property
    def ok(self) -> bool:
        return bool(self.inventory)


@dataclass(frozen=True)
class FormationModel:
    """Role templates plus a formation prior, fitted together.

    `inventories` carries the TRAINING counts, not probabilities, so the prior
    can be re-smoothed and so a consumer can see how thin a class is before
    trusting a prediction of it.
    """

    templates: RoleTemplates
    inventories: dict[tuple[int, ...], int]
    sigma2: float
    tau: float = DEFAULT_TAU
    frame: str = FRAME

    @property
    def modal_inventory(self) -> tuple[int, ...]:
        """The fallback whenever the descriptor is unusable."""
        if not self.inventories:
            return ()
        return max(self.inventories.items(), key=lambda kv: kv[1])[0]

    def log_prior(self, inv: tuple[int, ...]) -> float:
        total = sum(self.inventories.values()) + PRIOR_SMOOTHING * len(self.inventories)
        return float(
            np.log((self.inventories.get(inv, 0) + PRIOR_SMOOTHING) / max(total, 1e-9))
        )

    def validate_serving(self, frame: str) -> None:
        self.templates.validate_serving(frame)
        if frame != self.frame:
            raise ValueError(
                f"formation model fitted in frame {self.frame!r}, served "
                f"{frame!r}"
            )

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "tau": float(self.tau),
            "sigma2": float(self.sigma2),
            "templates": self.templates.to_dict(),
            # A list of records, not a dict: tuple keys have no JSON form, and
            # stringifying them invites a parser that silently disagrees.
            "inventories": [
                {"roles": list(inv), "count": int(n)}
                for inv, n in sorted(self.inventories.items())
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> FormationModel:
        return cls(
            templates=RoleTemplates.from_dict(d["templates"]),
            inventories={
                tuple(int(r) for r in rec["roles"]): int(rec["count"])
                for rec in d["inventories"]
            },
            sigma2=float(d["sigma2"]),
            tau=float(d.get("tau", DEFAULT_TAU)),
            frame=d.get("frame", FRAME),
        )


def _match_cost(slot_means: np.ndarray, targets: np.ndarray) -> float:
    """Least-squares Hungarian cost of matching slot means onto role templates.

    Both point sets are mean-centred first. They come from different estimators
    -- the role representation's fixed point and a per-thread mean -- and a
    constant offset between them would otherwise be charged unequally to
    candidate inventories of different shape.
    """
    a = slot_means - slot_means.mean(axis=0)
    b = targets - targets.mean(axis=0)
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum())


def predict_inventory(
    model: FormationModel, slot_means: np.ndarray, *, frame: str = FRAME
) -> InventoryPrediction:
    """Which formation is this team fielding?

    `slot_means` is the (K, 2) role representation from
    `formation.soccercpd.assign_roles` -- role SLOTS, not threads, so it is
    invariant to which player occupies which position. Candidates are the
    inventories seen in training with exactly K roles; everything else is
    unreachable and the modal inventory is returned instead.

    Scores `tau * log P(slot means | inventory) + log P(inventory)`. The prior
    is not optional: without it the argmax selects on VARIANCE and prefers
    classes fitted from two samples over one fitted from a hundred.
    """
    model.validate_serving(frame)
    slot_means = np.asarray(slot_means, dtype=np.float64)
    fallback = model.modal_inventory
    if slot_means.ndim != 2 or slot_means.shape[0] < 1 or slot_means.shape[1] != 2:
        return InventoryPrediction(fallback, 0.0, {}, "no role representation")
    if not np.isfinite(slot_means).all():
        return InventoryPrediction(fallback, 0.0, {}, "slot means not finite")
    _assert_normalised_feature(slot_means)

    k = len(slot_means)
    scores: dict[tuple[int, ...], float] = {}
    for inv in model.inventories:
        if len(inv) != k or any(r not in model.templates.means for r in inv):
            continue
        targets = np.stack([model.templates.means[r] for r in inv])
        loglik = -_match_cost(slot_means, targets) / (2.0 * model.sigma2)
        scores[inv] = model.tau * loglik + model.log_prior(inv)
    if not scores:
        return InventoryPrediction(
            fallback, 0.0, {},
            f"no training inventory has {k} roles; fell back to the modal one",
        )
    order = sorted(scores, key=lambda i: -scores[i])
    margin = scores[order[0]] - scores[order[1]] if len(order) > 1 else float("inf")
    return InventoryPrediction(order[0], float(margin), scores, "ok")


def assign_team_roles(
    model: FormationModel,
    features: np.ndarray,
    n_samples,
    *,
    left_to_right: bool,
    inventory: tuple[int, ...] | None = None,
    capacity: int = DEFAULT_CAPACITY,
    min_samples: int = MIN_SAMPLES,
    min_margin: float | None = None,
    frame: str = FRAME,
) -> list[RoleAssignment]:
    """Assign every thread of ONE team-half a role, jointly.

    Joint, not per-thread: the whole point of knowing the formation is that the
    threads compete for a bounded set of slots. `capacity` threads may take each
    role (2 by default -- substitutions).

    `inventory=None` means unrestricted: every fitted role is a candidate, which
    reproduces the previous per-thread behaviour when `capacity <= 0`.

    `left_to_right` comes from 3a and is REQUIRED. It is half the DST slot
    index, so a wrong bit mirrors every role -- LB becomes RW.

    `min_margin=None` (the default) reproduces the measured configuration,
    which did NOT abstain. Setting a number enables the `formation.roles`
    abstention discipline on top of the joint assignment; that combination is
    UNMEASURED.

    NOTE the margin here is NOT the per-thread quantity `roles.assign_role`
    uses, and 0.0 is not a harmless value for it. Under a joint assignment a
    thread is routinely given a role that is not its own argmin -- that is what
    "joint" means -- so its margin against the best alternative is NEGATIVE by
    design. A 0.0 floor would abstain on exactly those threads and quietly
    collapse the assignment back to per-thread argmin.
    """
    model.validate_serving(frame)
    features = np.asarray(features, dtype=np.float64)
    n_samples = np.asarray(n_samples).reshape(-1)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(f"features must be (T, 2), got {features.shape}")
    if len(features) != len(n_samples):
        raise ValueError("features and n_samples must be parallel")
    if features.size:
        _assert_normalised_feature(features)

    roles = (
        sorted(model.templates.means)
        if inventory is None
        else [r for r in inventory if r in model.templates.means]
    )
    out: list[RoleAssignment] = [
        RoleAssignment(None, None, 0.0, "unassigned") for _ in features
    ]
    if not roles:
        for i in range(len(out)):
            out[i] = RoleAssignment(None, None, 0.0, "no fitted role available")
        return out

    # Threads that cannot be scored at all never enter the assignment; letting
    # them occupy a slot would deny it to a thread that can use it.
    usable = []
    for i in range(len(features)):
        if n_samples[i] < min_samples:
            out[i] = RoleAssignment(
                None, None, 0.0,
                f"thread too short ({int(n_samples[i])} < {min_samples} samples)",
            )
        elif not np.isfinite(features[i]).all():
            out[i] = RoleAssignment(None, None, 0.0, "feature undefined")
        else:
            usable.append(i)
    if not usable:
        return out

    cost = np.empty((len(usable), len(roles)))
    for a, i in enumerate(usable):
        for b, r in enumerate(roles):
            d = features[i] - model.templates.means[r]
            cost[a, b] = float(d @ model.templates.inv_cov[r] @ d)

    if capacity <= 0 or capacity * len(roles) < len(usable):
        # Per-thread argmin. Also the fallback when there are more threads than
        # slots x capacity: a rectangular assignment would leave rows unmatched
        # and score them wrong, penalising smaller inventories asymmetrically.
        pick = cost.argmin(axis=1)
        chosen = {a: int(pick[a]) for a in range(len(usable))}
    else:
        cols = [b for b in range(len(roles)) for _ in range(capacity)]
        r_i, c_i = linear_sum_assignment(cost[:, cols])
        chosen = {int(a): cols[int(c)] for a, c in zip(r_i, c_i)}

    for a, i in enumerate(usable):
        if a not in chosen:
            out[i] = RoleAssignment(None, None, 0.0, "no slot available")
            continue
        b = chosen[a]
        role = roles[b]
        alt = np.delete(cost[a], b)
        margin = float(alt.min() - cost[a, b]) if alt.size else float("inf")
        costs = {r: float(cost[a, j]) for j, r in enumerate(roles)}
        if min_margin is not None and margin < min_margin:
            out[i] = RoleAssignment(
                None, None, margin,
                f"ambiguous (margin {margin:.2f} < {min_margin})", costs,
            )
        else:
            out[i] = RoleAssignment(
                role=role,
                slot=int(left_to_right) * N_ROLES + (role - 1),
                margin=margin,
                reason="ok",
                costs=costs,
            )
    return out


def assign_team(
    model: FormationModel,
    features: np.ndarray,
    n_samples,
    slot_means: np.ndarray,
    *,
    left_to_right: bool,
    capacity: int = DEFAULT_CAPACITY,
    **kwargs,
) -> tuple[list[RoleAssignment], InventoryPrediction]:
    """Predict the formation, then assign roles under it. The whole system."""
    pred = predict_inventory(model, slot_means)
    got = assign_team_roles(
        model, features, n_samples, left_to_right=left_to_right,
        inventory=pred.inventory or None, capacity=capacity, **kwargs,
    )
    return got, pred


def fit_formation_model(
    features: np.ndarray,
    roles: np.ndarray,
    team_half: np.ndarray,
    slot_means_by_team_half: dict | None = None,
    *,
    min_per_role: int = 50,
    tau: float = DEFAULT_TAU,
) -> FormationModel:
    """Fit templates and the formation prior from labelled training data.

    `features`, `roles` and `team_half` are parallel per-sample arrays in the
    spread-normalised formation-relative frame. `team_half` groups samples into
    team-halves, which is what defines a formation.

    `slot_means_by_team_half` maps a team-half key to its (K, 2) role
    representation. It is used ONLY to fit `sigma2`, the residual scale that
    puts the likelihood and the log-prior in comparable units. Omit it and
    sigma2 falls back to the pooled feature variance -- workable, but the
    likelihood is then not calibrated against the estimator that will serve it.

    MUST be fitted on training data only. The formation prior is by far the
    strongest single signal here (a constant modal guess is 60% accurate), so
    fitting it on data that includes the evaluation set inflates everything
    downstream of it.
    """
    features = np.asarray(features, dtype=np.float64)
    roles = np.asarray(roles, dtype=np.int64)
    team_half = np.asarray(team_half)
    if not (len(features) == len(roles) == len(team_half)):
        raise ValueError("features, roles and team_half must be parallel arrays")

    templates = fit_role_templates(features, roles, min_per_role=min_per_role)

    inventories: dict[tuple[int, ...], int] = {}
    for key in np.unique(team_half):
        inv = outfield_inventory(roles[team_half == key])
        if inv:
            inventories[inv] = inventories.get(inv, 0) + 1
    if not inventories:
        raise ValueError("no team-half yielded an outfield role inventory")

    # sigma2 puts the log-likelihood and the log-prior in comparable units, so
    # it decides how much geometry it takes to overturn the prior. It MUST be a
    # residual of the same matching operation the serving path performs.
    #
    # An earlier version fell back to the pooled feature variance when no slot
    # means were supplied. That is roughly the squared radius of a whole
    # formation -- one to two orders of magnitude too large -- which flattens
    # the likelihood until the prior always wins. The model then silently
    # degrades to "always guess the most common formation": still ~60%
    # accurate, so nothing looks broken, while the entire measured benefit of
    # predicting (+0.009 downstream) is gone. Caught by
    # `test_a_rare_class_can_still_win_on_strong_geometry`.
    #
    # The fallback now matches each team-half's own PER-ROLE MEAN FEATURES onto
    # its own inventory's templates -- the same cost function, one estimator
    # removed from the role representation, and available from the labelled
    # data alone.
    resid: list[float] = []
    for key, means in (slot_means_by_team_half or {}).items():
        inv = outfield_inventory(roles[team_half == key])
        means = np.asarray(means, dtype=np.float64)
        if len(inv) != len(means) or any(r not in templates.means for r in inv):
            continue
        targets = np.stack([templates.means[r] for r in inv])
        resid.append(_match_cost(means, targets) / len(inv))
    if not resid:
        for key in np.unique(team_half):
            sel = team_half == key
            inv = outfield_inventory(roles[sel])
            if not inv or any(r not in templates.means for r in inv):
                continue
            obs = []
            for r in inv:
                rows = features[sel & (roles == r)]
                rows = rows[np.isfinite(rows).all(axis=1)]
                if not len(rows):
                    break
                obs.append(rows.mean(axis=0))
            if len(obs) != len(inv):
                continue
            targets = np.stack([templates.means[r] for r in inv])
            resid.append(_match_cost(np.stack(obs), targets) / len(inv))
    if not resid:
        raise ValueError(
            "could not fit sigma2: no team-half yielded a matchable set of "
            "role means. Supply `slot_means_by_team_half`, or check that the "
            "labelled roles reach `min_per_role`. Refusing to guess a scale -- "
            "too large a value silently collapses this model onto its prior."
        )
    sigma2 = float(np.mean(resid))
    return FormationModel(
        templates=templates,
        inventories=inventories,
        sigma2=max(sigma2, 1e-6),
        tau=tau,
    )
