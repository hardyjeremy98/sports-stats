"""Positional role assignment for merged threads, in the DST slot encoding.

Consumer
--------
Role feeds **DST action spotting**, not re-ID. DST's `teamvec` consumes
`slot = LEFT_TO_RIGHT * 13 + (ROLE_ID - 1)`, giving 26 slots
(`docs/reference/footpass-setup.md:159`). Re-ID uses the occupancy channel,
which is separate and already built. Two consequences that shape this module:

* The unit is a **merged thread** -- re-ID output, minutes of evidence -- not a
  raw tracklet. Measured 2026-08-05 on FOOTPASS val: 0.746 exact-role
  (excluding GK, 11 roles, chance ~0.10) on perfectly merged threads versus
  0.443 on raw observable spans. The unit was worth 30 points.
* **Identity through off-camera gaps is available**, because re-ID has already
  run. That is legitimate here and is not for a re-ID input.

`LEFT_TO_RIGHT` is half the slot index, so attacking direction is a HARD
dependency (3a, `formation.direction`). Get it wrong and every role maps to its
mirror -- LB becomes RW. There is deliberately no default.

What was measured NOT to help (2026-08-05, paired, match-clustered, n=3
matches, 6 halves) -- do not re-add these without new evidence:

* **Off-camera position interpolation: -0.008, 0/6 halves.** Not a method
  failure: giving the estimator the TRUE off-camera positions scores the same
  (-0.008, 0/6). Linear interpolation recovers a player's MEAN position about
  as well as the truth does (mean |mu| 0.7334 vs 0.7387) and the mean is what
  is consumed. NOTE the contrast: interpolation *halves* the error of the team
  CENTROID and is used there -- different quantity, different answer.
* **Concave (fast-then-settle) interpolation: -0.008, 0/6.**
* **Hungarian one-to-one assignment: actively harmful.** With ~4 of 11 players
  visible, a hard one-to-one constraint converts one confident correct
  assignment into two wrong ones -- the forced-choice failure already recorded
  in this repo. The literature's constraint (Bialkowski, SoccerCPD) assumes
  full visibility.

The only effect that survived: **merge quality**, +0.112 with a perfect merge
versus a 16-piece one, 6/6 halves, the one comparison whose match-clustered CI
excluded zero. Role accuracy is bounded by re-ID quality, not role modelling.

Coordinates
-----------
Spread-normalised formation-relative: position minus team centroid, divided by
the team's RMS radius at that frame, with attack direction canonicalised.
Normalisation matters -- team spread varies with phase of play, and EFPI's one
substantive finding is that unscaled matching against fixed templates yields
illogical labels (a centre back labelled DM). `formation_relative` applies no
scale normalisation, so it is done here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: FOOTPASS role ids. GK is 1; slots are ROLE_ID - 1 in the DST encoding.
N_ROLES = 13
GK_ROLE = 1
#: Shrinkage of each role's covariance toward the pooled covariance. Rare roles
#: are genuinely rare (MCB appears in 22 of 192 train team-halves), so an
#: unshrunk 2x2 estimate is ill-conditioned.
COV_SHRINKAGE = 0.3
#: Samples a thread needs before a role is claimed at all.
MIN_SAMPLES = 10
#: Minimum gap between the best and second-best role cost. Below this the
#: thread abstains (ADR 003) rather than guessing between adjacent roles.
MIN_MARGIN = 0.5
#: The serving frame. Recorded in the model and hard-checked, because this repo
#: has shipped three fit/serve coordinate mismatches.
FRAME = "relative-norm"


def _assert_normalised_feature(feat: np.ndarray) -> None:
    if feat.size and np.nanmax(np.abs(feat)) > 50.0:
        raise ValueError(
            f"role feature magnitude {float(np.nanmax(np.abs(feat))):.3g} is "
            "implausible for a SPREAD-NORMALISED coordinate (values are radii "
            "in team-RMS units). Raw pitch fractions or metres passed here?"
        )


def team_spread(rel_xy: np.ndarray, frames: np.ndarray) -> dict[int, float]:
    """frame -> RMS radius of the team's observed players about their centroid.

    Computed from ALL of a team's observations at that frame, so it must be
    built once per team-frame and shared by every thread of that team -- not
    recomputed per thread, which would make each thread's normaliser depend on
    itself.
    """
    rel_xy = np.asarray(rel_xy, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    ok = np.isfinite(rel_xy).all(axis=1)
    rel_xy, frames = rel_xy[ok], frames[ok]
    out: dict[int, float] = {}
    if not len(frames):
        return out
    order = np.argsort(frames, kind="stable")
    frames, rel_xy = frames[order], rel_xy[order]
    uniq, start = np.unique(frames, return_index=True)
    bounds = np.append(start, len(frames))
    for i, f in enumerate(uniq):
        block = rel_xy[bounds[i]:bounds[i + 1]]
        r = float(np.sqrt(np.mean((block * block).sum(axis=1))))
        if r > 1e-6:
            out[int(f)] = r
    return out


def thread_feature(
    rel_xy: np.ndarray, frames: np.ndarray, spread: dict[int, float]
) -> np.ndarray:
    """The 2-D feature for one merged thread: mean spread-normalised offset.

    Returns NaN when nothing is usable. The MEAN is deliberate -- richer set
    statistics were measured not to beat it, and interpolating the thread's
    off-camera gaps does not change it materially.
    """
    rel_xy = np.asarray(rel_xy, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    s = np.array([spread.get(int(f), np.nan) for f in frames])
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = rel_xy / s[:, None]
    ok = np.isfinite(norm).all(axis=1)
    if not ok.any():
        return np.array([np.nan, np.nan])
    return norm[ok].mean(axis=0)


@dataclass(frozen=True)
class RoleTemplates:
    """Per-role Gaussian in the spread-normalised formation-relative frame."""

    means: dict[int, np.ndarray]
    inv_cov: dict[int, np.ndarray]
    n_fitted: dict[int, int]
    frame: str = FRAME

    def validate_serving(self, frame: str) -> None:
        """Hard-fail on a fit/serve coordinate mismatch."""
        if frame != self.frame:
            raise ValueError(
                f"role templates were fitted in frame {self.frame!r} but are "
                f"being served {frame!r}. This repo has shipped three fit/serve "
                "coordinate mismatches; this one is refused."
            )

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "means": {str(r): v.tolist() for r, v in self.means.items()},
            "inv_cov": {str(r): v.tolist() for r, v in self.inv_cov.items()},
            "n_fitted": {str(r): int(n) for r, n in self.n_fitted.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> RoleTemplates:
        return cls(
            means={int(k): np.asarray(v, np.float64) for k, v in d["means"].items()},
            inv_cov={
                int(k): np.asarray(v, np.float64) for k, v in d["inv_cov"].items()
            },
            n_fitted={int(k): int(v) for k, v in d["n_fitted"].items()},
            frame=d.get("frame", FRAME),
        )


@dataclass(frozen=True)
class RoleAssignment:
    """A thread's role, or an abstention.

    `slot` is the DST `teamvec` index: `LEFT_TO_RIGHT * 13 + (ROLE_ID - 1)`.
    """

    role: int | None
    slot: int | None
    margin: float
    reason: str
    costs: dict[int, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.role is not None


def fit_role_templates(
    features: np.ndarray, roles: np.ndarray, *, min_per_role: int = 50
) -> RoleTemplates:
    """Fit per-role means and covariances from labelled per-sample features.

    Features must already be in the spread-normalised frame -- the SAME
    transform used at serving. Roles with fewer than `min_per_role` samples are
    dropped rather than fitted badly; a dropped role can never be predicted,
    which is the honest behaviour.
    """
    features = np.asarray(features, dtype=np.float64)
    roles = np.asarray(roles, dtype=np.int64)
    ok = np.isfinite(features).all(axis=1)
    features, roles = features[ok], roles[ok]
    if not len(features):
        raise ValueError("no finite features to fit role templates from")
    _assert_normalised_feature(features)

    pooled = np.cov(features.T) + 1e-6 * np.eye(2)
    means, inv_cov, n_fitted = {}, {}, {}
    for r in np.unique(roles):
        z = features[roles == r]
        if len(z) < min_per_role:
            continue
        cov = np.cov(z.T) + 1e-6 * np.eye(2)
        cov = (1.0 - COV_SHRINKAGE) * cov + COV_SHRINKAGE * pooled
        means[int(r)] = z.mean(axis=0)
        inv_cov[int(r)] = np.linalg.inv(cov)
        n_fitted[int(r)] = int(len(z))
    if not means:
        raise ValueError("no role reached min_per_role samples")
    return RoleTemplates(means=means, inv_cov=inv_cov, n_fitted=n_fitted)


def assign_role(
    feature: np.ndarray,
    templates: RoleTemplates,
    *,
    left_to_right: bool,
    n_samples: int,
    min_samples: int = MIN_SAMPLES,
    min_margin: float = MIN_MARGIN,
    frame: str = FRAME,
) -> RoleAssignment:
    """Assign one merged thread a role and its DST slot, or abstain.

    `left_to_right` comes from 3a (`formation.direction`) and is REQUIRED --
    it is half the slot index, so a wrong bit mirrors every role. There is no
    default and no inference from the feature.

    Abstains when the thread is too short, the feature is undefined, or the
    best two roles are within `min_margin` -- adjacent roles (LCB/RCB, LM/LW)
    are genuinely hard to separate and a confident wrong role is worse for a
    downstream spotter than a missing one.
    """
    templates.validate_serving(frame)
    feature = np.asarray(feature, dtype=np.float64)

    def bail(reason: str) -> RoleAssignment:
        return RoleAssignment(role=None, slot=None, margin=0.0, reason=reason)

    if n_samples < min_samples:
        return bail(f"thread too short ({n_samples} < {min_samples} samples)")
    if feature.shape != (2,) or not np.isfinite(feature).all():
        return bail("feature undefined (no usable spread-normalised samples)")
    _assert_normalised_feature(feature)

    costs = {
        r: float((feature - templates.means[r]) @ templates.inv_cov[r]
                 @ (feature - templates.means[r]))
        for r in templates.means
    }
    order = sorted(costs, key=costs.get)
    best = order[0]
    margin = costs[order[1]] - costs[best] if len(order) > 1 else np.inf
    if margin < min_margin:
        return RoleAssignment(
            role=None, slot=None, margin=float(margin),
            reason=f"ambiguous between {best} and {order[1]} "
                   f"(margin {margin:.2f} < {min_margin})",
            costs=costs,
        )
    return RoleAssignment(
        role=best,
        slot=int(left_to_right) * N_ROLES + (best - 1),
        margin=float(margin),
        reason="ok",
        costs=costs,
    )


def dst_slot(role: int, left_to_right: bool) -> int:
    """The DST `teamvec` slot index for a role and attacking direction.

    `slot = LEFT_TO_RIGHT * 13 + (ROLE_ID - 1)`, 26 slots
    (docs/reference/footpass-setup.md:159).
    """
    if not 1 <= role <= N_ROLES:
        raise ValueError(f"role {role} outside 1..{N_ROLES}")
    return int(left_to_right) * N_ROLES + (role - 1)
