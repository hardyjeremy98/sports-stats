"""Attacking direction and epoch (half) boundary from sparse positional probes.

Why this exists
---------------
The occupancy re-ID channel compares formation-relative footprints. Attack
direction reverses at half-time, so a player's footprint rotates 180 degrees.
Measured 2026-08-02 on FOOTPASS VAL: cross-half raw same-player AUC 0.34-0.38
vs 0.74-0.80 once rotated; fused LOMO 0.855 without occupancy, 0.854 served
raw, 0.888 with an explicit per-half flip. The flip was adopted as direction
but deferred because it needs a boundary estimate. This module is that
estimate.

What the consumer actually needs is ONE BIT per candidate pair: are these two
fragments in the same attacking-direction epoch? It does not need a left/right
label. So the primary output is an epoch partition of the timeline, and the
absolute direction is a secondary, weaker-evidence output for role templates.

Prior art (2026-08-05): every sports-analytics library (kloppy, floodlight,
SoccerCPD, EFPI) takes direction and periods from PROVIDER METADATA and does
not estimate them. The one public CV implementation, the SoccerNet GSR
baseline (sn-gamestate, arXiv 2404.11335), compares "the average 2D positions
of each team on the pitch" -- the same centroid-ordering cue used here -- but
publishes no isolated accuracy, and its 30-second sequences never contain a
flip. The comparison bar for boundary detection is SoccerNet's game-clock OCR
(90% of half-starts within +-2 s), a cue that needs a broadcast clock overlay
and is unavailable on the target amateur footage.

Coordinates
-----------
ABSOLUTE pitch-normalised x in [0, 1]. NOT the formation-relative coordinate
occupancy consumes -- that is centroid-subtracted, so the statistic would be
identically zero. Both mistakes are checked for and raise.

Scope limit
-----------
Two epochs (one flip). Extra time adds two more end swaps; warm-up footage has
teams shooting at the wrong ends; a reverse-angle replay mirrors x wholesale
for seconds. A sub-boundary guard abstains when the two-epoch model is
violated, but this module does not segment more than two epochs.

Offline by construction: the coarse pass reads the whole timeline. The product
processes complete uploaded matches, so this is legitimate -- but do not wire
it into a streaming path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from matchlab_core.formation.centroid import _assert_normalised

#: Coarse probe spacing, seconds. Must exceed the error autocorrelation length
#: (attacking spells run 30-60 s), or a whole cluster of probes goes wrong
#: together at one place in the timeline -- exactly what an argmax is sensitive
#: to. Validated by the V3 sweep, not assumed.
DEFAULT_COARSE_SECONDS = 30.0
#: Refinement stops here. Below ~30 s the consumer cannot tell the difference,
#: because it only asks which side of the boundary a fragment falls on.
DEFAULT_TOLERANCE_SECONDS = 5.0
#: A half is never shorter than this. Defined on the TIME axis, not probe
#: index: spacing is non-uniform under coarse-to-fine and abstained probes are
#: dropped, so index is not proportional to time.
DEFAULT_MIN_HALF_SECONDS = 20 * 60.0
#: Observable players required on EACH side for a probe to count.
DEFAULT_MIN_PER_TEAM = 3
#: Permutation z below this -> abstain. The confidence must be scale-free;
#: raw separation is in pitch fractions and does not transfer across zoom or
#: camera style, so it is kept as a diagnostic only.
DEFAULT_MIN_Z = 4.0
#: Per-epoch sign agreement below this -> abstain.
DEFAULT_MIN_AGREEMENT = 0.65
#: A sub-boundary this strong inside a returned segment violates the two-epoch
#: model -> abstain rather than return one confident wrong boundary.
#: Calibrated 2026-08-05 on synthetic matches, not guessed: across 24
#: (seed x noise 0.05-0.20) clean two-epoch matches the largest per-segment
#: sub-z was 1.53, while a three-flip (extra-time shaped) match scored 4.1-4.7.
#: 3.0 sits in that gap. Re-measure if the coarse spacing changes.
DEFAULT_SUB_Z = 3.0
DEFAULT_N_PERMUTATIONS = 400

#: A probe function maps frame indices to, per frame, the absolute normalised
#: x of every OBSERVABLE player of club A and of club B.
ProbeFn = Callable[[np.ndarray], Sequence[tuple[np.ndarray, np.ndarray]]]


@dataclass(frozen=True)
class Probes:
    """Evaluated probes, abstentions already dropped."""

    frames: np.ndarray  # (n,) int64, ascending
    d: np.ndarray  # (n,) signed statistic mean_x(A) - mean_x(B)
    n_a: np.ndarray
    n_b: np.ndarray
    n_evaluated: int  # frames actually handed to the probe fn, incl. abstained

    @property
    def sign(self) -> np.ndarray:
        return np.sign(self.d)


@dataclass(frozen=True)
class DirectionEstimate:
    """Epoch partition of a match timeline, or an abstention.

    `boundary is None` means abstained -- the caller must fall back to its
    known-safe behaviour (for occupancy, `mirror="off"`, i.e. today's default).
    Never treat an abstention as "no flip happened".
    """

    boundary: int | None
    z: float
    reason: str
    probes_used: int
    #: Mean sign(d) within each epoch; None when abstained.
    epoch_signs: tuple[float, float] | None
    #: Half-width of the boundary's uncertainty, in frames.
    ci_frames: int
    #: Diagnostic only, in pitch fractions, NOT a transferable threshold.
    separation: float

    @property
    def ok(self) -> bool:
        return self.boundary is not None

    def epoch_of_fragment(self, start: int, end: int) -> int | None:
        """Epoch of a fragment spanning [start, end], or None if undecidable.

        Fragment-level, not frame-level, because the consumer pairs fragments.
        A fragment straddling the boundary (or its confidence band) belongs to
        neither epoch and returns None -> neutral evidence (ADR 003). The
        "straddling fragments are negligible" shortcut only holds when a
        half-time gap separates the epochs, which the no-gap evaluation
        protocol deliberately removes.
        """
        if self.boundary is None:
            return None
        lo = self.boundary - self.ci_frames
        hi = self.boundary + self.ci_frames
        if end < lo:
            return 0
        if start > hi:
            return 1
        return None

    def same_epoch(self, a: tuple[int, int], b: tuple[int, int]) -> bool | None:
        """The one bit occupancy needs.

        None = undecidable. **The caller must then treat occupancy as NEUTRAL
        for this pair -- contributing nothing -- not serve it unmirrored.**
        Undecidable pairs are concentrated around the boundary, i.e. they are
        disproportionately CROSS-half, and raw cross-half occupancy is
        anti-informative there (measured AUC 0.34-0.38, worse than chance),
        not merely uninformative. Falling back to `mirror="off"` is safe as a
        whole-match default because raw occupancy roughly cancels overall
        (fused 0.854 vs 0.855 without it); it is NOT safe on this subset,
        where the fallback is the actively harmful regime.
        """
        ea = self.epoch_of_fragment(*a)
        eb = self.epoch_of_fragment(*b)
        if ea is None or eb is None:
            return None
        return ea == eb

    def attacks_positive_x(self, club: int, epoch: int) -> bool | None:
        """Absolute direction: does `club` (0=A, 1=B) attack increasing x?

        SECONDARY OUTPUT, for role templates (3b). Not consumed by occupancy
        and not validated by the occupancy measurements -- 3b must establish
        its own evidence before relying on it.
        """
        if self.epoch_signs is None or club not in (0, 1) or epoch not in (0, 1):
            return None
        s = self.epoch_signs[epoch]
        if s == 0.0:
            return None
        # d > 0 means A's centroid is to the RIGHT of B's, i.e. A is pinned
        # back defending the right goal, so A attacks -x.
        a_attacks_positive = s < 0
        return a_attacks_positive if club == 0 else not a_attacks_positive


#: The smallest |d| a real match produces. Measured on FOOTPASS VAL: the mean
#: margin is ~0.043 pitch fractions and the per-probe distribution is far
#: wider. A whole match an order of magnitude below this is not a weak signal,
#: it is the wrong coordinate frame.
MIN_PLAUSIBLE_MARGIN = 0.004


def _assert_absolute_normalised(xs: np.ndarray, ys: np.ndarray | None = None) -> None:
    """Units guard, delegating to the shared check so the tolerance can't drift."""
    for arr, label in ((xs, "A"), (ys, "B")):
        if arr is None or arr.size == 0:
            continue
        try:
            _assert_normalised(arr, f"club {label} x")
        except ValueError as exc:  # re-raise with this module's context
            raise ValueError(f"direction: {exc}") from None


def evaluate_probes(
    probe_fn: ProbeFn,
    frames: np.ndarray,
    *,
    min_per_team: int = DEFAULT_MIN_PER_TEAM,
) -> Probes:
    """Run `probe_fn` at `frames`, keeping only probes with enough on each side."""
    frames = np.asarray(frames, dtype=np.int64)
    batches = probe_fn(frames)
    if len(batches) != len(frames):
        raise ValueError("probe_fn must return one (xs_a, xs_b) pair per frame")

    keep_f, keep_d, keep_a, keep_b = [], [], [], []
    rel_hits = 0
    for f, (xa, xb) in zip(frames, batches):
        xa = np.asarray(xa, dtype=np.float64).ravel()
        xb = np.asarray(xb, dtype=np.float64).ravel()
        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]
        if len(xa) < min_per_team or len(xb) < min_per_team:
            continue
        _assert_absolute_normalised(xa, xb)
        ma, mb = float(xa.mean()), float(xb.mean())
        if abs(ma - 0.5) < 1e-9 and abs(mb - 0.5) < 1e-9:
            rel_hits += 1
        keep_f.append(int(f))
        keep_d.append(ma - mb)
        keep_a.append(len(xa))
        keep_b.append(len(xb))

    # Formation-relative coordinates are centroid-subtracted, so d carries no
    # team-ordering signal. When the full roster is visible every team mean is
    # exactly 0.5 and we can say so outright.
    #
    # With the normal 6-8 of 11 visible we CANNOT: the observed subset mean is
    # not 0.5, and sampling noise alone puts |d| around 0.08 -- larger than the
    # ~0.043 margin a real match shows. There is no magnitude test that
    # separates them, so no exception is raised. What happens instead is safe
    # but must be diagnosable: the change-point test finds no structure and the
    # module abstains, which drops the consumer back to its existing behaviour.
    # `estimate_direction` names this coordinate-frame mistake in the
    # abstention reason so it does not read as "the cue is dead".
    if keep_f and rel_hits == len(keep_f):
        raise ValueError(
            "direction: every team mean is exactly 0.5 -- these are "
            "FORMATION-RELATIVE coordinates. This estimator needs ABSOLUTE "
            "pitch-normalised x."
        )
    return Probes(
        frames=np.asarray(keep_f, dtype=np.int64),
        d=np.asarray(keep_d, dtype=np.float64),
        n_a=np.asarray(keep_a, dtype=np.int64),
        n_b=np.asarray(keep_b, dtype=np.int64),
        n_evaluated=len(frames),
    )


def _split_scores(s: np.ndarray) -> np.ndarray:
    """score(k) = |mean(s[:k]) - mean(s[k:])| * sqrt(k(n-k)/n) for k=1..n-1.

    The sqrt factor is the textbook two-sample standardisation (Var of the
    difference of means is proportional to n/(k(n-k))), i.e. the standard
    CUSUM / binary-segmentation statistic. Computed on sign(d) rather than d:
    the raw magnitude is dominated by long sieges in one end, which are
    precisely the correlated-error episodes.
    """
    n = len(s)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    csum = np.cumsum(s)
    k = np.arange(1, n, dtype=np.float64)
    left = csum[:-1] / k
    right = (csum[-1] - csum[:-1]) / (n - k)
    return np.abs(left - right) * np.sqrt(k * (n - k) / n)


def _best_split(
    frames: np.ndarray, s: np.ndarray, *, min_half_frames: float
) -> tuple[int, float]:
    """(index, score) of the best admissible split. Admissibility is on TIME."""
    scores = _split_scores(s)
    if scores.size == 0:
        return -1, 0.0
    t0, t1 = float(frames[0]), float(frames[-1])
    # Split k puts frames[:k] left; the boundary time is between frames[k-1]
    # and frames[k].
    mid = (frames[:-1] + frames[1:]) / 2.0
    ok = (mid - t0 >= min_half_frames) & (t1 - mid >= min_half_frames)
    if not ok.any():
        return -1, 0.0
    masked = np.where(ok, scores, -np.inf)
    k = int(np.argmax(masked)) + 1
    return k, float(scores[k - 1])


def _permutation_z(
    frames: np.ndarray,
    s: np.ndarray,
    *,
    min_half_frames: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """z of the observed max split score against a shuffled-label null.

    Scale-free, unlike `separation`. Shuffling breaks the temporal structure
    while preserving the sign marginal, so the null answers "how big an argmax
    do I get from a series with this much imbalance but no change point" --
    which is exactly the centre-bias trap the raw score cannot see.
    """
    _, obs = _best_split(frames, s, min_half_frames=min_half_frames)
    if obs == 0.0:
        return 0.0
    null = np.empty(n_perm, dtype=np.float64)
    perm = s.copy()
    for i in range(n_perm):
        rng.shuffle(perm)
        null[i] = _best_split(frames, perm, min_half_frames=min_half_frames)[1]
    sd = float(null.std())
    if sd <= 0.0:
        # A degenerate null must never wave the estimate through: returning
        # +inf here would pass `min_z` unconditionally.
        return 0.0
    return (obs - float(null.mean())) / sd


def _local_sign(
    probe_fn: ProbeFn,
    centre: int,
    *,
    span: int,
    votes: int,
    min_per_team: int,
    budget: list[int],
) -> float:
    """Robust sign of d near `centre`, by majority over a few spread probes.

    The window is CENTRED on `centre` and clipped symmetrically. Shifting it
    rightwards near frame 0 (the naive `max(0, centre - span//2)` then
    `centre + span`) would make the "sign just left of the bracket" probe
    sample the wrong side of the bracket it is deciding.
    """
    half = max(1, span // 2)
    lo = max(0, centre - half)
    hi = max(lo + 1, centre + half)
    frames = np.unique(np.linspace(lo, hi, votes).astype(np.int64))
    p = evaluate_probes(probe_fn, frames, min_per_team=min_per_team)
    budget[0] += p.n_evaluated
    if len(p.d) == 0:
        return 0.0
    return float(np.sign(np.mean(np.sign(p.d))))


def estimate_direction(
    probe_fn: ProbeFn,
    *,
    total_frames: int,
    fps: float,
    coarse_seconds: float = DEFAULT_COARSE_SECONDS,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    min_half_seconds: float = DEFAULT_MIN_HALF_SECONDS,
    min_per_team: int = DEFAULT_MIN_PER_TEAM,
    min_z: float = DEFAULT_MIN_Z,
    min_agreement: float = DEFAULT_MIN_AGREEMENT,
    sub_z: float = DEFAULT_SUB_Z,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    refine: bool = True,
    seed: int = 0,
) -> DirectionEstimate:
    """Locate the single attacking-direction flip in a match timeline.

    Probes `total_frames` sparsely: a coarse pass every `coarse_seconds`, then
    bisection around the change point. For a 90-minute match at the defaults
    that is roughly 190 probed frames out of ~135,000.

    Abstains (`boundary is None`) rather than guessing when the permutation z
    is low, the two segments do not have opposite sign, per-segment agreement
    is poor, or a significant sub-boundary shows the two-epoch model is wrong.
    """
    if total_frames <= 0 or fps <= 0:
        raise ValueError("total_frames and fps must be positive")
    rng = np.random.default_rng(seed)
    min_half_frames = min_half_seconds * fps
    budget = [0]

    step = max(1, int(round(coarse_seconds * fps)))
    coarse = np.arange(0, total_frames, step, dtype=np.int64)
    probes = evaluate_probes(probe_fn, coarse, min_per_team=min_per_team)
    budget[0] += probes.n_evaluated

    def bail(reason: str, z: float = 0.0) -> DirectionEstimate:
        return DirectionEstimate(
            boundary=None, z=z, reason=reason, probes_used=budget[0],
            epoch_signs=None, ci_frames=0, separation=0.0,
        )

    if len(probes.d) < 8:
        return bail("too few usable probes")
    if 2 * min_half_frames > (probes.frames[-1] - probes.frames[0]):
        return bail("timeline shorter than two minimum halves")

    s = probes.sign
    k, _ = _best_split(probes.frames, s, min_half_frames=min_half_frames)
    if k < 0:
        return bail("no admissible split")
    z = _permutation_z(
        probes.frames, s, min_half_frames=min_half_frames,
        n_perm=n_permutations, rng=rng,
    )
    if z < min_z:
        hint = ""
        scale = float(np.mean(np.abs(probes.d)))
        # Ratio, not an absolute floor: |mean(d)| shrinks as 1/sqrt(n_probes),
        # so any fixed threshold is a probe-count threshold in disguise.
        if scale > MIN_PLAUSIBLE_MARGIN and abs(float(np.mean(probes.d))) < 0.25 * scale:
            # |d| is large per frame but averages to nothing and has no
            # temporal structure -- the signature of a centroid-subtracted
            # coordinate frame, not of a match with no flip.
            hint = (
                "; |d| is large per probe but unstructured -- check these are "
                "ABSOLUTE and not FORMATION-RELATIVE pitch coordinates"
            )
        return bail(f"weak change point (z={z:.2f} < {min_z}){hint}", z)

    left, right = s[:k], s[k:]
    m_left, m_right = float(left.mean()), float(right.mean())
    if np.sign(m_left) == np.sign(m_right) or m_left == 0.0 or m_right == 0.0:
        return bail("segments do not have opposite sign", z)
    agree_l = float((left == np.sign(m_left)).mean())
    agree_r = float((right == np.sign(m_right)).mean())
    if min(agree_l, agree_r) < min_agreement:
        return bail(
            f"low sign agreement ({min(agree_l, agree_r):.2f} < {min_agreement})", z
        )

    # Two-epoch guard: a strong further change point INSIDE either segment
    # means extra time, warm-up footage, or a mirrored replay -- return
    # nothing rather than one confident wrong boundary.
    for seg_f, seg_s in ((probes.frames[:k], left), (probes.frames[k:], right)):
        if len(seg_s) < 8:
            continue
        # Admissibility inside a segment is a quarter of that segment's own
        # span -- `min_half_frames` is a whole-match constant and would leave
        # no admissible window inside a segment barely longer than a half.
        seg_span = float(seg_f[-1] - seg_f[0])
        sub = _permutation_z(
            seg_f, seg_s, min_half_frames=min(min_half_frames, seg_span / 4.0),
            n_perm=n_permutations, rng=rng,
        )
        if sub > sub_z:
            return bail(f"sub-boundary detected (z={sub:.2f}) -- not a 2-epoch match", z)

    lo, hi = int(probes.frames[k - 1]), int(probes.frames[k])
    tol = max(1, int(round(tolerance_seconds * fps)))
    bracketed = False
    if refine:
        sign_lo = float(np.sign(m_left))
        sign_hi = float(np.sign(m_right))
        # The coarse argmax locates the flip only to within about one coarse
        # step, so the bracket it hands over may not actually CONTAIN the sign
        # change. Widen outward until the two ends disagree (and agree with
        # their own segments) before bisecting -- otherwise bisection converges
        # neatly onto a bracket endpoint and reports ~1 step of error as if it
        # were 5 s of precision.
        for _ in range(3):
            s_lo = _local_sign(
                probe_fn, lo - step // 2, span=step, votes=5,
                min_per_team=min_per_team, budget=budget,
            )
            s_hi = _local_sign(
                probe_fn, hi + step // 2, span=step, votes=5,
                min_per_team=min_per_team, budget=budget,
            )
            if s_lo == sign_lo and s_hi == sign_hi:
                bracketed = True
                break
            if s_lo != sign_lo:
                lo = max(0, lo - step)
            if s_hi != sign_hi:
                hi = min(total_frames - 1, hi + step)
        while hi - lo > tol:
            mid = (lo + hi) // 2
            # The vote window must shrink with the bracket. Held at the coarse
            # step, every vote below one step straddles the true boundary and
            # the last halvings are decided by whichever side of a 30 s window
            # happens to dominate -- manufacturing precision that isn't there.
            sm = _local_sign(
                probe_fn, mid, span=max(tol, hi - lo), votes=5,
                min_per_team=min_per_team, budget=budget,
            )
            if sm == 0.0:
                break
            if sm == sign_lo:
                lo = mid
            else:
                hi = mid

    boundary = (lo + hi) // 2
    # The bisection interval is NOT an error bar. It is the width of a
    # deterministic search over a NOISY oracle (per-probe sign accuracy is
    # 0.86-0.97 on FOOTPASS), so it always ends at `tol` regardless of how far
    # off the answer is -- measured errors of 20-40 s were being reported
    # alongside a 5 s CI, coverage about 50%. Floor the band at the coarse
    # resolution the search can actually resolve, and widen it when the
    # bracket was never confirmed to contain the sign change. This band is
    # what `epoch_of_fragment` abstains inside, so understating it hands
    # occupancy a confident mirror decision for exactly the fragments sitting
    # near half-time -- where getting it backwards is maximally damaging.
    ci = max(tol, step, (hi - lo) // 2)
    if not bracketed:
        ci = max(ci, 3 * step)
    # The flip can only be localised as tightly as the surrounding probes.
    # When a stretch of probes ABSTAINS -- a camera blackout, or a team-stage
    # failure that collapses one club so `min_per_team` is never met -- the
    # coarse bracket spanning that stretch is the real uncertainty. Measured
    # 2026-08-05: a 20-minute one-club corruption straddling half-time left a
    # 20-minute hole and a 361 s boundary error whose z (9.2), sign agreement
    # (0.96/0.89) and separation (0.078) were all indistinguishable from a
    # clean match -- no confidence statistic could flag it. Widening the band
    # to the hole turns that confident wrong answer into an honest
    # "undecidable here", which is what the consumer needs.
    ci = max(ci, int(probes.frames[k] - probes.frames[k - 1]) // 2)
    return DirectionEstimate(
        boundary=boundary,
        z=z,
        reason="ok",
        probes_used=budget[0],
        epoch_signs=(m_left, m_right),
        ci_frames=int(ci),
        separation=abs(float(probes.d[:k].mean())) + abs(float(probes.d[k:].mean())),
    )


def probe_fn_from_arrays(
    frames: np.ndarray,
    xs: np.ndarray,
    clubs: np.ndarray,
    *,
    observable: np.ndarray,
    max_snap: int | None = None,
) -> ProbeFn:
    """Build a `ProbeFn` over flat per-observation arrays.

    `frames`/`xs`/`clubs`/`observable` are parallel, one entry per player-frame.
    `clubs` is 0/1 and must be CLUB-consistent across the whole match (what an
    appearance-based team stage emits) -- not a pitch-side code, which by
    definition does not flip and would make the boundary undetectable.

    Requested frames SNAP to the nearest populated frame within `max_snap`
    (default: twice the median inter-frame spacing). Artifacts index by source
    video frame even when the pipeline samples at a stride, and the repo
    convention is that consumers snap; without this, refinement probes land
    between sampled frames, return nothing, and bisection silently gives up --
    reporting a boundary error of about one coarse step as though it were the
    refinement tolerance.
    """
    frames = np.asarray(frames, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.float64)
    clubs = np.asarray(clubs, dtype=np.int64)
    obs = np.asarray(observable, dtype=bool)
    keep = obs & np.isfinite(xs)
    frames, xs, clubs = frames[keep], xs[keep], clubs[keep]
    order = np.argsort(frames, kind="stable")
    frames, xs, clubs = frames[order], xs[order], clubs[order]

    uniq = np.unique(frames)
    if max_snap is None:
        spacing = np.diff(uniq)
        max_snap = int(2 * np.median(spacing)) if spacing.size else 1
    max_snap = max(1, int(max_snap))

    def fn(want: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        want = np.asarray(want, dtype=np.int64)
        if uniq.size == 0:
            return [(np.empty(0), np.empty(0))] * len(want)
        j = np.clip(np.searchsorted(uniq, want), 0, uniq.size - 1)
        jm = np.clip(j - 1, 0, uniq.size - 1)
        pick = np.where(
            np.abs(uniq[jm] - want) <= np.abs(uniq[j] - want), uniq[jm], uniq[j]
        )
        pick = np.where(np.abs(pick - want) <= max_snap, pick, -1)
        lo = np.searchsorted(frames, pick, side="left")
        hi = np.searchsorted(frames, pick, side="right")
        out = []
        for p, a, b in zip(pick, lo, hi):
            if p < 0:
                out.append((np.empty(0), np.empty(0)))
                continue
            c, x = clubs[a:b], xs[a:b]
            out.append((x[c == 0], x[c == 1]))
        return out

    return fn
