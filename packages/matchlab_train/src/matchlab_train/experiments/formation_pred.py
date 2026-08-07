"""Formation prediction from broadcast-visible threads, four methods compared.

WHY THIS EXISTS, AND WHAT IT CAN AND CANNOT SHOW
------------------------------------------------
The hypothesis under test is that knowing a team's formation BEFORE assigning
tactical roles makes role assignment substantially more accurate. Two measured
facts frame the whole exercise:

* **The ceiling is small.** With the ORACLE inventory handed over for free,
  role accuracy on FOOTPASS TRAIN (48 matches, held out by match, merged GT
  threads, exGK) rises by +0.031 over the SAME capacity-2 assignment rule
  applied without any inventory. Roughly a third of that is available from
  guessing the single most common formation every time. So a PERFECT formation
  predictor buys ~3 points, and this harness measures how much of that a real
  predictor keeps. Exact figures come from the run, not from this docstring --
  an earlier version of this paragraph quoted a superseded set and disagreed
  with the harness's own output.
* **The FOOTPASS `ROLE` column is not an algorithmic function of position.** A
  label-free Bialkowski assignment reproduces it at mean 0.700 (max 0.830,
  none above 0.95) over 41 team-halves, so the labels are genuine annotation
  and the arms below are not reproducing their own label generator.

Two things this harness CANNOT show, both properties of the data:

* `ROLE` is assigned once per player per half and inherited by substitutes, so
  across 192 team-halves the fielded inventory changes mid-half exactly ZERO
  times. SoccerCPD's `FormCPD` and `RoleCPD` have no observable target here.
  They are implemented and unit-tested in `matchlab_core.formation.soccercpd`;
  they are not evaluated, and the absence of labelled changes is an annotation
  artefact, not evidence that teams hold shape for 48 minutes.
* VAL contains a single formation in all 12 team-halves, so evaluation runs on
  TRAIN and VAL is untouched.

CLASS STRUCTURE, STATED UP FRONT
--------------------------------
192 TRAIN team-halves carry 9 distinct outfield inventories: 116 / 47 / 12 / 4
/ 4 / 4 / 2 / 2 / 1. The tail classes live in one or two matches each, so under
the by-match split they are never in both train and test. Macro-F1 is therefore
undefined and is not reported, and no per-class or tail-class claim is
supportable. The honest headline is accuracy against the 0.604 majority
baseline (116/192).

The formation deltas are UNCORRECTED for multiplicity: three descriptor arms
times four position arms is twelve comparisons against the same baseline, and
the reported CIs are nominal. Read a single arm's win as suggestive, and check
its leave-one-match-out range -- one match moves the per-match mean by 1/48.

An uncontrolled leak remains: the tactical h5 has no global club key
(`PLAYER_ID // 100` is a WITHIN-match index), formation is largely a club
property, and the same clubs recur across these 48 matches. Grouping by match
cannot remove that. Every number below is optimistic by an unmeasured amount.

Every input here is ORACLE -- GT threads, GT identity through gaps, oracle
attacking direction from `COL.TEAM`. No arm is deployable as measured.

Usage:
  uv run python -m matchlab_train.experiments.formation_pred --arm interp-warp
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np
from matchlab_core.formation import (
    TimeWarp,
    assign_roles,
    formation_label,
    role_adjacency,
)
from scipy.optimize import linear_sum_assignment

from matchlab_train.datasets.footpass import COL, ROLE_NAMES, half_keys, load_half
from matchlab_train.experiments.position_evidence import TRAIN_H5
from matchlab_train.experiments.role_headroom import STRIDE, _centroid_lookup
from matchlab_train.experiments.role_threads import thread_positions

GK = 1
N_OUTFIELD = 10

#: Named for WHAT THEY COMPUTE, not for the papers they are drawn from.
#: A results row labelled "soccercpd" is a claim about SoccerCPD however many
#: caveats follow it, and none of these is the published method: `adjacency`
#: repurposes SoccerCPD's Delaunay descriptor from change DETECTION to
#: classification, under a Bernoulli likelihood the paper does not contain;
#: `line-signature` is the line partition Bialkowski's role clustering implies,
#: not his clustering; `template-match` is a Hungarian match to fitted role
#: templates, which is EFPI's idea but not EFPI's procedure.
DESCRIPTOR_ARMS = ("line-signature", "adjacency", "template-match")

WARP_LINEAR = TimeWarp()
WARP_CONCAVE = TimeWarp(k_x=2.0, k_y=2.0)
MIN_SAMPLES = 10


def match_of(key: str) -> str:
    return key.rpartition("_H")[0]


class TeamHalf:
    """One team's half: per-thread mean position, plus its true role inventory."""

    def __init__(self, key, team, pid, frames, feat, role):
        self.key, self.team = key, team
        self.pid, self.frames, self.feat, self.role = pid, frames, feat, role
        keep = [p for p in np.unique(pid) if (pid == p).sum() >= MIN_SAMPLES]
        self.threads = [int(p) for p in keep]
        self.truth = np.array([int(role[pid == p][0]) for p in keep])
        self.mu = np.stack([feat[pid == p].mean(axis=0) for p in keep]) if keep else \
            np.empty((0, 2))
        self.outfield = self.truth != GK

    @property
    def inventory(self) -> tuple[int, ...]:
        """The true fielded OUTFIELD role SLOTS, over the whole half.

        A SET of distinct roles, not a multiset over threads. Substitutes
        inherit the outgoing player's slot, so a 14-thread half has 14 role
        labels but still only ten slots; taking the multiset made almost every
        team-half its own class (a 96-class census of mostly n=1).

        Union across the half rather than one frame: a frame missing a row
        would silently drop a slot and manufacture a spurious class.
        """
        return tuple(sorted(set(self.truth[self.outfield].tolist())))


def team_halves(h5, key: str, arm: str, warp: TimeWarp) -> list[TeamHalf]:
    """Both teams of `key`, in the spread-normalised formation-relative frame.

    The spread denominator is computed from OBSERVED rows only in every arm.
    Computing it from the arm's own filled rows would give each arm a different
    scale -- interpolation pulls players onto a chord and compresses the RMS
    radius -- and the arms' accuracies would not be on a common footing.
    """
    rows = load_half(h5, key).rows
    rows = rows[rows[:, COL.FRAME].astype(np.int64) % STRIDE == 0]
    cent = _centroid_lookup(rows, "module")
    team_of = {
        int(p): int(rows[rows[:, COL.PLAYER_ID] == p][0, COL.TEAM])
        for p in np.unique(rows[:, COL.PLAYER_ID])
    }

    def relative(position_arm, w):
        pid, fr, xy, role = thread_positions(rows, position_arm, w)
        if not len(pid):
            return pid, fr, np.empty((0, 2)), role, []
        c = np.full_like(xy, np.nan)
        for i in range(len(xy)):
            got = cent.get((team_of[int(pid[i])], int(fr[i])))
            if got is not None:
                c[i] = got
        # A plain LIST of tuples, never an object array: np.array of tuples
        # builds a 2-D array whose rows are unhashable, and the scale dict is
        # keyed by (team, frame).
        keys = [(team_of[int(p)], int(f)) for p, f in zip(pid, fr)]
        return pid, fr, xy - c, role, keys

    # Scale from observed rows only -- shared by every arm.
    _, _, rel_obs, _, keys_obs = relative("observed", WARP_LINEAR)
    acc: dict[tuple[int, int], list[float]] = defaultdict(list)
    for i, k in enumerate(keys_obs):
        if np.isfinite(rel_obs[i]).all():
            acc[k].append(float(rel_obs[i] @ rel_obs[i]))
    scale = {k: float(np.sqrt(np.mean(v))) for k, v in acc.items() if v}

    pid, fr, rel, role, keys = relative(arm, warp)
    if not len(pid):
        return []
    s = np.array([scale.get(k, np.nan) for k in keys])
    with np.errstate(invalid="ignore", divide="ignore"):
        feat = rel / np.where(s[:, None] > 1e-6, s[:, None], np.nan)
    ok = np.isfinite(feat).all(axis=1)
    pid, fr, feat, role = pid[ok], fr[ok], feat[ok], role[ok]

    out = []
    for team in (0, 1):
        sel = np.array([team_of.get(int(p), -1) == team for p in pid], dtype=bool)
        if sel.sum():
            out.append(TeamHalf(key, team, pid[sel], fr[sel], feat[sel], role[sel]))
    return out


# ---------------------------------------------------------------- descriptors


#: The role-representation EM is the expensive step and each team-half's
#: descriptor is fold-independent (it reads no training data), so it is
#: computed once. Keyed by identity, not by (key, team), so a re-run with a
#: different position arm cannot collide with a stale entry.
_DESC_CACHE: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, ...]]] = {}

#: Set by --gk-proxy. Module-level because `slot_descriptors` is memoised and
#: the flag must be fixed for the whole run.
GK_PROXY = [False]

#: How many all-ten frames each team-half's role representation actually used.
#: "Coverage" only asks whether that count is > 0; an EM fitted on four frames
#: and one fitted on two thousand are not the same estimator, and under the
#: `observed` position arm (mean 3.95 of 11 visible) the difference is the
#: whole story.
_FRAMES_USED: list[int] = []


def slot_descriptors(th: TeamHalf) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    got = _DESC_CACHE.get(id(th))
    if got is None:
        got = _DESC_CACHE[id(th)] = _slot_descriptors(th)
    return got


def _slot_descriptors(th: TeamHalf) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Role-slot means, their Delaunay adjacency, and the line label.

    Runs on OUTFIELD threads only. The keeper is excluded from the role
    representation because it is a fixed point that every formation shares, so
    including it adds a slot that carries no discriminating information while
    costing one degree of freedom in the Hungarian bijection.

    NOTE the keeper is identified by its TRUE role label -- an oracle input,
    handed to every descriptor arm and to none of the baseline (which never
    looks at positions at all). It is the right direction for a NEGATIVE
    result (the rivals lose despite the help) and would contaminate a positive
    one; `--gk-proxy` swaps in a label-free stand-in to check.

    `already_relative=True`: `th.feat` is ALREADY expressed against the imputed
    eleven-player team centroid. Letting `assign_roles` re-centre each frame on
    the mean of the outfield subset present would shift the slot means by the
    keeper's share plus the imputation residual -- putting them in a different
    frame from the role templates that `efpi` matches them against.
    """
    if GK_PROXY[0]:
        # Label-free stand-in: coordinates are canonicalised so every team
        # attacks +x, so the keeper is the DEEPEST thread. Crude, but it is the
        # control that decides whether a positive result was bought with an
        # oracle input.
        if len(th.mu) < N_OUTFIELD + 1:
            return np.empty((0, 2)), np.empty((0, 0), bool), ()
        idx = [i for i in range(len(th.threads)) if i != int(th.mu[:, 0].argmin())]
    else:
        idx = [i for i, _ in enumerate(th.threads) if th.truth[i] != GK]
    if len(idx) < N_OUTFIELD:
        return np.empty((0, 2)), np.empty((0, 0), bool), ()
    keep_pid = {th.threads[i] for i in idx}
    m = np.array([int(p) in keep_pid for p in th.pid], dtype=bool)
    try:
        slots = assign_roles(
            th.feat[m], th.frames[m], th.pid[m],
            n_slots=N_OUTFIELD, already_relative=True,
        )
    except ValueError:
        return np.empty((0, 2)), np.empty((0, 0), bool), ()
    _FRAMES_USED.append(slots.n_frames_used)
    if not slots.is_role_representation:
        # No frame showed all ten outfield threads, so `means` is the seed --
        # per-entity means, not role slots. Reporting that as a descriptor
        # would measure a different method under this method's name.
        return np.empty((0, 2)), np.empty((0, 0), bool), ()
    means = slots.means[np.argsort(slots.means[:, 0])]
    return means, role_adjacency(means), formation_label(means)


# ---------------------------------------------------------------------- arms


def fit_arms(train: list[TeamHalf]):
    """Everything the arms need, fitted on the training fold ONLY."""
    inv_counts = Counter(t.inventory for t in train if t.inventory)
    majority = inv_counts.most_common(1)[0][0] if inv_counts else ()

    role_mu: dict[int, list[np.ndarray]] = defaultdict(list)
    for t in train:
        for i in range(len(t.threads)):
            if t.truth[i] != GK:
                role_mu[int(t.truth[i])].append(t.mu[i])
    templates = {r: np.mean(v, axis=0) for r, v in role_mu.items() if len(v) >= 20}

    lines: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    adjs: dict[tuple[int, ...], list[np.ndarray]] = defaultdict(list)
    for t in train:
        if not t.inventory:
            continue
        means, adj, line = slot_descriptors(t)
        if not len(means):
            continue
        lines[t.inventory][line] += 1
        adjs[t.inventory].append(adj.astype(np.float64))
    proto = {k: np.mean(v, axis=0) for k, v in adjs.items() if v}

    # Scale for efpi's Gaussian likelihood: the mean squared residual of a
    # training team-half against its OWN inventory's templates. Fitted, so the
    # likelihood and the log-prior are on a comparable scale rather than the
    # prior being swamped by an arbitrary cost unit.
    resid = []
    for t in train:
        means, _, _ = slot_descriptors(t)
        inv = t.inventory
        if not len(means) or len(inv) != len(means) or any(r not in templates
                                                           for r in inv):
            continue
        T = np.stack([templates[r] for r in inv])
        M = means - means.mean(axis=0)
        T = T - T.mean(axis=0)
        cost = ((M[:, None, :] - T[None, :, :]) ** 2).sum(-1)
        r, c = linear_sum_assignment(cost)
        resid.append(cost[r, c].sum() / len(inv))
    sigma2 = float(np.mean(resid)) if resid else 1.0

    fitted = {
        "counts": inv_counts, "majority": majority, "templates": templates,
        "lines": lines, "proto": proto, "tpl_sigma2": max(sigma2, 1e-6),
        "tau": {},
    }
    for arm in DESCRIPTOR_ARMS:
        fitted["tau"][arm] = fit_temperature(arm, train, fitted)
    return fitted


def descriptor_loglik(arm: str, th: TeamHalf, fitted) -> dict[tuple[int, ...], float]:
    """Unnormalised log P(descriptor | inventory) per candidate. No prior.

    Split out from `predict` so that the prior and the likelihood TEMPERATURE
    are applied identically to every arm. Getting that wrong is how the first
    version of this harness produced a bogus result: scored as
    `argmin |query - mean(prototype)|_1` with a 1e-6 prevalence tie-break, the
    adjacency arm chose the 116-count majority class 0.6% of the time and each
    n=2 class ~19% -- ANTI-correlated with the prior -- because the L1 distance
    to a near-boolean prototype fitted from two samples has far higher variance
    than the distance to one fitted from 116, and an argmin over nine
    candidates selects on variance. The tie-break was ~1e-4 against distances
    of order 1-10, i.e. numerically inert. That arm scored 0.266 against a
    0.604 baseline and would have been written up as "the descriptor fails"
    when it was the decision rule failing. A comparison in which the baseline
    is prior-only and the rivals are prior-free measures nothing.

    The same reasoning applies one level up, which is why the temperature is
    FITTED PER ARM by the identical procedure (`fit_temperature`): an arm whose
    log-likelihood happens to be scaled in larger units drowns the prior and
    behaves as if prior-free, so comparing a scale-fitted arm against unscaled
    ones would re-introduce the same bug in subtler form.
    """
    counts = fitted["counts"]
    means, adj, line = slot_descriptors(th)
    if not len(means):
        return {}

    if arm == "line-signature":
        out = {}
        for inv, c in fitted["lines"].items():
            n_inv = sum(c.values())
            # Laplace-smoothed P(line|inv): an unseen signature is unlikely
            # rather than impossible, so the prior still decides.
            out[inv] = float(np.log((c.get(line, 0) + 0.5) / (n_inv + 0.5 * 8)))
        return out

    if arm == "adjacency":
        # Per-edge Bernoulli over the UPPER TRIANGLE only: the matrix is
        # symmetric, so scoring all of it counts every edge as two independent
        # observations and doubles the log-likelihood against the prior.
        iu = np.triu_indices(len(adj), k=1)
        a = adj[iu].astype(np.float64)
        out = {}
        for inv, p in fitted["proto"].items():
            if p.shape != adj.shape:
                continue
            q = np.clip(p[iu], 1e-3, 1 - 1e-3)
            out[inv] = float(a @ np.log(q) + (1 - a) @ np.log(1 - q))
        return out

    if arm == "template-match":
        # Hungarian cost of matching the slot means onto each candidate
        # inventory's role templates, as a Gaussian log-likelihood. Both point
        # sets are mean-centred first: they come from different estimators, and
        # a constant offset would be charged unequally to inventories of
        # different shape.
        tpl, sigma2 = fitted["templates"], fitted["tpl_sigma2"]
        M = means - means.mean(axis=0)
        out = {}
        for inv in counts:
            if any(r not in tpl for r in inv) or len(inv) != len(means):
                continue
            T = np.stack([tpl[r] for r in inv])
            T = T - T.mean(axis=0)
            cost = ((M[:, None, :] - T[None, :, :]) ** 2).sum(-1)
            r, c = linear_sum_assignment(cost)
            out[inv] = -float(cost[r, c].sum()) / (2 * sigma2)
        return out

    raise ValueError(f"unknown arm {arm!r}")


def predict(arm: str, th: TeamHalf, fitted) -> tuple[int, ...]:
    """argmax over `tau * log P(descriptor|inv) + log P(inv)`."""
    counts, majority = fitted["counts"], fitted["majority"]
    if arm == "majority" or not counts:
        return majority
    ll = descriptor_loglik(arm, th, fitted)
    if not ll:
        return majority
    total_n = sum(counts.values())
    tau = fitted["tau"][arm]
    best, best_s = majority, -np.inf
    for inv, v in ll.items():
        s = tau * v + np.log(counts[inv] / total_n)
        if s > best_s:
            best, best_s = inv, s
    return best


#: Likelihood temperatures searched per arm, on the TRAINING fold only. 0.0 is
#: the prior-only degenerate case and is included deliberately: an arm whose
#: descriptor carries nothing should be able to fall back to the baseline
#: rather than being forced to spend its likelihood.
TAU_GRID = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def fit_temperature(arm: str, train: list[TeamHalf], fitted) -> float:
    """Pick tau by training-fold accuracy. Identical procedure for every arm."""
    counts, majority = fitted["counts"], fitted["majority"]
    total_n = sum(counts.values())
    cached = [(t.inventory, descriptor_loglik(arm, t, fitted))
              for t in train if t.inventory]
    best_tau, best_acc = 0.0, -1.0
    for tau in TAU_GRID:
        hit = 0
        for truth, ll in cached:
            if not ll:
                pred = majority
            else:
                pred = max(
                    ll, key=lambda inv: tau * ll[inv] + np.log(counts[inv] / total_n)
                )
            hit += pred == truth
        acc = hit / max(len(cached), 1)
        if acc > best_acc:
            best_tau, best_acc = tau, acc
    return best_tau


# ------------------------------------------------------------------ downstream


def fit_role_gaussians(train: list[TeamHalf]):
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for t in train:
        for i in range(len(t.threads)):
            acc[int(t.truth[i])].append(t.mu[i])
    pooled = np.concatenate([np.stack(v) for v in acc.values()])
    pcov = np.cov(pooled.T) + 1e-6 * np.eye(2)
    out = {}
    for r, v in acc.items():
        z = np.stack(v)
        if len(z) < 20:
            continue
        cov = 0.7 * (np.cov(z.T) + 1e-6 * np.eye(2)) + 0.3 * pcov
        out[r] = (z.mean(axis=0), np.linalg.inv(cov))
    return out


#: Times the capacity-2 assignment had fewer columns than threads. Counted
#: rather than asserted: it needs ~21 outfield threads in one half, so it
#: should never fire, and a silent zero is not the same as an unchecked one.
_OVERFLOW = [0]


def _maha(mu, tpl, roles):
    d = np.empty(len(roles))
    for i, r in enumerate(roles):
        v = mu - tpl[r][0]
        d[i] = v @ tpl[r][1] @ v
    return d


def role_accuracy(th: TeamHalf, tpl, inventory, *, capacity: int) -> tuple[int, int]:
    """Correct / total outfield role assignments under a given inventory.

    `inventory=None` means unrestricted argmax over every fitted role.
    `capacity` is how many threads may take each role -- 2, because a
    substitution puts a second thread into the same slot within a half, and a
    strict bijection has been measured to LOSE on this substrate.
    """
    mask = th.outfield
    if not mask.any():
        return 0, 0
    mus, truth = th.mu[mask], th.truth[mask]
    # GK is excluded from the UNRESTRICTED candidate set too. It is fitted (the
    # templates cover every role) but the scored threads are outfield-only, so
    # leaving it in gives the unrestricted arm a candidate it can only ever be
    # wrong about -- while the inventory-restricted arms, whose inventories are
    # outfield-only by construction, cannot make that error. That asymmetry
    # inflated every downstream delta in the first run.
    roles = (
        sorted(r for r in tpl if r != GK) if inventory is None
        else [r for r in inventory if r in tpl]
    )
    if not roles:
        return 0, 0
    D = np.stack([_maha(m, tpl, roles) for m in mus])
    if capacity <= 0:
        pred = np.array([roles[j] for j in D.argmin(axis=1)])
    else:
        cols, colrole = [], []
        for j, r in enumerate(roles):
            cols += [j] * capacity
            colrole += [r] * capacity
        if len(cols) < len(mus):
            # More threads than slots x capacity: the rectangular assignment
            # would leave rows unassigned and score them wrong, penalising
            # arms that predict SMALLER inventories. Fall back to argmin so
            # the comparison stays fair, and let the caller count it.
            pred = np.array([roles[j] for j in D.argmin(axis=1)])
            _OVERFLOW[0] += 1
        else:
            r_i, c_i = linear_sum_assignment(D[:, cols])
            pred = np.full(len(mus), -1)
            pred[r_i] = [colrole[c] for c in c_i]
    return int((pred == truth).sum()), int(len(truth))


# ----------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="interp-warp",
                    choices=["observed", "interp", "interp-warp", "gt"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5,
                    help="repeat the CV partition this many times")
    ap.add_argument("--gk-proxy", action="store_true",
                    help="identify the keeper label-free (deepest thread) "
                         "instead of by its true role -- the control for "
                         "whether a positive result was bought with an oracle")
    a = ap.parse_args()
    GK_PROXY[0] = a.gk_proxy
    warp = WARP_CONCAVE if a.arm == "interp-warp" else WARP_LINEAR

    keys = half_keys(TRAIN_H5)
    print(f"loading {len(keys)} halves (position arm: {a.arm}) ...", flush=True)
    cache: dict[str, list[TeamHalf]] = {}
    for k in keys:
        cache[k] = team_halves(TRAIN_H5, k, a.arm, warp)

    all_th = [t for k in keys for t in cache[k]]
    inv_counts = Counter(t.inventory for t in all_th if t.inventory)
    print(f"{len(all_th)} team-halves, {len(inv_counts)} distinct outfield "
          f"inventories; majority share {inv_counts.most_common(1)[0][1]/len(all_th):.3f}")

    matches = sorted({match_of(k) for k in keys})

    ARMS = ["majority", *DESCRIPTOR_ARMS]
    form_hit = {m: defaultdict(lambda: [0, 0]) for m in matches}
    role_hit = {m: defaultdict(lambda: [0, 0]) for m in matches}

    # Repeat the CV PARTITION over several seeds. A single partition leaves the
    # fold-to-fold variance of the fitted templates, temperatures and majority
    # class unmeasured, and that variance is the same size as the effects here.
    plan = [(s, f) for s in range(a.seeds)
            for f in np.array_split(np.random.default_rng(s).permutation(
                len(matches)), a.folds)]
    taus: dict[str, list[float]] = defaultdict(list)
    for fi, (seed, fold) in enumerate(plan):
        test_m = {matches[i] for i in fold}
        train = [t for k in keys if match_of(k) not in test_m for t in cache[k]]
        test = [(match_of(k), t) for k in keys if match_of(k) in test_m
                for t in cache[k]]
        fitted = fit_arms(train)
        for arm in DESCRIPTOR_ARMS:
            taus[arm].append(fitted["tau"][arm])
        tpl = fit_role_gaussians(train)
        for m, th in test:
            if not th.inventory:
                continue
            preds = {arm: predict(arm, th, fitted) for arm in ARMS}
            for arm, p in preds.items():
                form_hit[m][arm][0] += int(p == th.inventory)
                form_hit[m][arm][1] += 1
            # Downstream role accuracy. The two effects the formation is meant
            # to supply -- a smaller candidate set, and a capacity constraint
            # -- are separated: comparing unrestricted-argmax against
            # restricted-capacity-2 would credit the inventory with a gain
            # that the assignment rule produced.
            for name, inv, cap in [
                ("unrestricted", None, 0),
                ("unrestricted-cap2", None, 2),
                ("oracle-inv-argmax", th.inventory, 0),
                ("oracle-inv-cap2", th.inventory, 2),
                *[(f"pred:{arm}", preds[arm], 2) for arm in ARMS],
            ]:
                h, t = role_accuracy(th, tpl, inv, capacity=cap)
                role_hit[m][name][0] += h
                role_hit[m][name][1] += t
        print(f"  seed {seed} fold {fi % a.folds + 1}/{a.folds} done", flush=True)

    def report(store, names, title):
        print(f"\n== {title} ==")
        ms = [m for m in matches if store[m][names[0]][1] > 0]
        acc = {n: np.array([store[m][n][0] / max(store[m][n][1], 1) for m in ms])
               for n in names}
        print(f"{'arm':<16}{'acc':>8}{'per-match sd':>15}")
        for n in names:
            w = sum(store[m][n][0] for m in ms) / max(
                sum(store[m][n][1] for m in ms), 1)
            print(f"{n:<16}{w:>8.3f}{acc[n].std():>15.3f}")
        return ms, acc

    ms, facc = report(
        form_hit, ARMS,
        f"FORMATION accuracy ({len(matches)} matches, held out by match)",
    )
    ROLE_NAMES_ORDER = [
        "unrestricted", "unrestricted-cap2", "oracle-inv-argmax", "oracle-inv-cap2",
    ] + [f"pred:{a_}" for a_ in ARMS]
    ms2, racc = report(role_hit, ROLE_NAMES_ORDER,
                       "DOWNSTREAM role accuracy, exGK (what the formation is FOR)")

    n_desc = sum(1 for t in all_th if t.inventory and len(slot_descriptors(t)[0]))
    n_lab = sum(1 for t in all_th if t.inventory)
    fu = np.array(_FRAMES_USED)
    print(f"\ndescriptor coverage: {n_desc}/{n_lab} team-halves produced a role "
          f"representation; the rest fall back to the majority class in every "
          f"descriptor arm (so those arms are pulled toward the baseline).")
    if len(fu):
        print(f"all-ten frames the role EM actually used: median {int(np.median(fu))},"
              f" p10 {int(np.percentile(fu, 10))}, p90 {int(np.percentile(fu, 90))}."
              f"  Coverage only asks whether this is > 0.")
    ten = sum(1 for t in all_th if len(t.inventory) == N_OUTFIELD)
    print(f"template-match reachability: {ten}/{n_lab} team-halves have exactly "
          f"{N_OUTFIELD} distinct outfield roles; it cannot predict the rest.")
    print(f"capacity-overflow fallbacks: {_OVERFLOW[0]}")
    print("fitted likelihood temperature per arm (median over folds): " + ", ".join(
        f"{k}={np.median(v):g}" for k, v in taus.items()))
    print(f"GK identified by: {'LABEL-FREE PROXY' if GK_PROXY[0] else 'TRUE LABEL'}")

    boot = np.random.default_rng(1)

    def contrast(acc, a_name, b_name, cluster):
        """Paired per-match-cluster delta a - b, with a bootstrap CI, the
        leave-one-match-out range of the point estimate, and pos/tie/neg.

        The jackknife range is reported because with 48 clusters a single match
        moves the mean by up to 1/48 = 0.021 -- larger than several of the CI
        lower bounds here, so 'excludes zero' can mean 'one match wide'.
        """
        d = acc[a_name] - acc[b_name]
        n = len(cluster)
        idx = boot.integers(0, n, size=(10000, n))
        lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
        loo = [(d.sum() - d[i]) / (n - 1) for i in range(n)]
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        return (f"{d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
                f"  LOO [{min(loo):+.3f}, {max(loo):+.3f}]"
                f"  pos/tie/neg {pos}/{n - pos - neg}/{neg}")

    print("\nDOWNSTREAM deltas. Read against unrestricted-cap2, NOT unrestricted:")
    print("  the capacity-2 assignment rule needs no formation information at all,")
    print("  so crediting it to the inventory overstates every arm.")
    for n in ROLE_NAMES_ORDER[1:]:
        print(f"  {n:<20} vs unrestricted      {contrast(racc, n, 'unrestricted', ms2)}")
    print()
    for n in ROLE_NAMES_ORDER[2:]:
        print(f"  {n:<20} vs unrestricted-cap2 "
              f"{contrast(racc, n, 'unrestricted-cap2', ms2)}")

    print("\nTHE CRUX -- does the learned predictor beat a CONSTANT guess of the")
    print("most common formation? (paired; the two agree on ~60% of team-halves)")
    for arm in DESCRIPTOR_ARMS:
        print(f"  pred:{arm:<15} vs pred:majority     "
              f"{contrast(racc, f'pred:{arm}', 'pred:majority', ms2)}")

    print(f"\nFORMATION deltas vs majority baseline ({len(DESCRIPTOR_ARMS)} arms x 4 "
          f"position arms = {4 * len(DESCRIPTOR_ARMS)} comparisons; UNCORRECTED):")
    for n in DESCRIPTOR_ARMS:
        print(f"  {n:<16}{contrast(facc, n, 'majority', ms)}")

    print("\ninventory census (outfield roles):")
    for inv, c in inv_counts.most_common():
        print(f"  {c:4d}x  " + "-".join(ROLE_NAMES.get(r, str(r)) for r in inv))


if __name__ == "__main__":
    main()
