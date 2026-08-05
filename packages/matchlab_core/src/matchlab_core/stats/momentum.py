"""Tier 2 stat 13 -- the team momentum chart.

This is a **presentation choice, not a measurement**, and the module says so
because no amount of care in the smoothing changes that.

There is no canonical cross-provider definition. Two research passes converged
on that independently. Two fully specified published implementations exist and
they disagree on the value model:

* **Opta / Stats Perform "Match Momentum"**
  (https://theanalyst.com/articles/what-is-match-momentum), built on their
  *possession value* model -- "the likelihood of the team in possession scoring
  within the next 10 seconds". Aggregation, verbatim: "We look at the maximum
  possession value for each team in every minute of the game so far (capped
  between zero and 0.1)." Smoothing: "weighted for how recently they occurred...
  Only the most recent three to four minutes have significant impact here."
  Series: "the difference between these values for each team". **The kernel
  itself is not disclosed.**
* An open xT-based reimplementation (Kapich) uses a 4-minute window, exponential
  decay 0.25, clip to [0, 0.1], per-minute max, Gaussian post-smooth sigma 1. It
  explicitly derives from Opta's published approach, so it corroborates the
  structure only because it copied it -- not independent evidence, exactly as
  socceraction is not independent of Singh.

**Adopted from the sources** (their shared, stated shape): per-minute bins, the
**maximum** per club per bin rather than a sum, clipping to [0, 0.1], recency
weighting over a ~3-4 minute effective window, and a series that is the
difference between the clubs.

**Declared as ours**, because no source specifies them: the kernel family and its
half-life, and the use of xT in place of Opta's undisclosed possession-value
model. Both ride on the returned object so a reader never has to guess.

Two things that are bugs, not choices
--------------------------------------
1. **The time axis.** `MatchEvent.t = frame_idx / 25` and `frame_idx` is
   **per-half**, so a chart that concatenates H1 and H2 without an explicit
   offset silently overlays them. `half_offsets` is required, not optional.
2. **Kernel truncation.** A kernel truncated at the start of a half must be
   renormalised over its available support, or the opening minutes are damped
   toward zero and every chart shows a slow start that is a rendering artefact
   and not a fact about the match. The renormalisation applies at the **half
   boundary** too, not only at the series start -- otherwise H2's opening minutes
   borrow from H1's closing minutes, which is the same artefact wearing a
   disguise. `test_stats_momentum_structural.py` pins both with a constant-input
   test: constant in must give constant out, *including* at boundary bins.

Antisymmetry, declared per the PRD's rule R3
---------------------------------------------
The series is **antisymmetric by construction**: club B's series is exactly club
A's negated. So every both-club aggregate is identically 0, and standard
deviation, peak magnitude and zero-crossing count are one number reported twice.
Only one signed series is emitted, and `MomentumSeries.club_id` names whose sign
is positive. Tier 1 killed the ground-duel win rate for being 0.5 by
construction; this is the same shape and gets the same treatment.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from matchlab_core.stats.xt import ActionCredit

#: Opta, verbatim: "capped between zero and 0.1". Adopted from the source.
VALUE_CAP = 0.1

#: Bin width. Opta bins per minute; both published implementations do.
DEFAULT_BIN_S = 60.0

#: OURS. Opta says only that "the most recent three to four minutes have
#: significant impact"; the kernel is undisclosed. An exponential kernel with
#: this half-life puts ~94% of its weight inside 4 minutes, which matches that
#: description without pretending to reproduce an unpublished curve.
DEFAULT_HALF_LIFE_MIN = 1.0

#: Weights below this are dropped -- a support bound, not a modelling claim.
_WEIGHT_FLOOR = 1e-6


@dataclass(frozen=True)
class MomentumPoint:
    """One bin. `raw_*` are pre-smoothing, so a reader can see the smoother's
    contribution rather than having to trust it."""

    half: int
    minute: float
    value: float
    raw_club: float
    raw_opponent: float
    #: How much of the kernel's mass was available at this bin. 1.0 in the
    #: interior; below 1.0 where the kernel was truncated and renormalised.
    kernel_support: float


@dataclass
class MomentumSeries:
    """A signed momentum series, positive toward `club_id`.

    Per R3 the opposing club's series is this one negated and is NOT a second
    observation. `both_club_aggregate` deliberately does not exist.
    """

    club_id: int
    match_id: str
    points: list[MomentumPoint] = field(default_factory=list)
    #: Every parameter that is ours rather than sourced, carried explicitly.
    value_model: str = "xt"
    bin_seconds: float = DEFAULT_BIN_S
    half_life_min: float = DEFAULT_HALF_LIFE_MIN
    kernel: str = "exponential"
    cap: float = VALUE_CAP
    provenance: str = (
        "Structure (per-minute bins, per-club maximum, [0, 0.1] cap, recency "
        "weighting over ~3-4 minutes, inter-club difference) follows Opta's "
        "published Match Momentum. The kernel family, half-life and the use of "
        "xT in place of Opta's undisclosed possession-value model are OURS. "
        "This is a presentation choice, not a measurement."
    )


def _bin_index(t: float, half: int, offsets: Mapping[int, float], bin_s: float) -> int:
    return int((t + offsets.get(half, 0.0)) // bin_s)


def build_momentum(
    credits: Iterable[ActionCredit],
    halves: Sequence[int],
    times: Sequence[float],
    *,
    club_id: int,
    match_id: str,
    half_offsets: Mapping[int, float],
    bin_seconds: float = DEFAULT_BIN_S,
    half_life_min: float = DEFAULT_HALF_LIFE_MIN,
    cap: float = VALUE_CAP,
) -> MomentumSeries:
    """Build the signed momentum series for `club_id`.

    `credits`, `halves` and `times` are parallel sequences -- the credit objects
    carry no timestamp, and inventing one from `event_id` ordering would be a
    silent assumption about the stream's sort order.

    `half_offsets` maps half number to the seconds to add to that half's `t`.
    It is **required**: `frame_idx` is per-half, so without it H2 is drawn on
    top of H1. There is no sensible default, because the real elapsed gap
    between halves is not in the data.
    """
    per_bin_club: dict[tuple[int, int], float] = {}
    per_bin_opp: dict[tuple[int, int], float] = {}
    bins_seen: set[tuple[int, int]] = set()

    for credit, half, t in zip(credits, halves, times, strict=True):
        if credit.delta is None:
            # Unrated actions carry no value signal. Not zero -- absent.
            continue
        b = _bin_index(t, half, half_offsets, bin_seconds)
        key = (half, b)
        bins_seen.add(key)
        # Opta takes the MAXIMUM per team per minute, not the sum: momentum is
        # about the most threatening moment, not accumulated volume.
        value = min(max(credit.delta, 0.0), cap)
        target = per_bin_club if credit.club_id == club_id else per_bin_opp
        target[key] = max(target.get(key, 0.0), value)

    if not bins_seen:
        return MomentumSeries(
            club_id=club_id,
            match_id=match_id,
            bin_seconds=bin_seconds,
            half_life_min=half_life_min,
            cap=cap,
        )

    ordered = sorted(bins_seen)
    decay = math.log(2.0) / (half_life_min * 60.0 / bin_seconds)

    points: list[MomentumPoint] = []
    for half, b in ordered:
        num_c = num_o = 0.0
        support = 0.0
        # The kernel looks BACKWARD only, and never across the half boundary.
        for other_half, ob in ordered:
            if other_half != half or ob > b:
                continue
            w = math.exp(-decay * (b - ob))
            if w < _WEIGHT_FLOOR:
                continue
            support += w
            num_c += w * per_bin_club.get((other_half, ob), 0.0)
            num_o += w * per_bin_opp.get((other_half, ob), 0.0)
        if support <= 0.0:
            continue
        # Renormalise over available support. Without this the opening minutes
        # of every half are damped toward zero and the chart shows a slow start
        # that is an artefact of truncation.
        smoothed_c = num_c / support
        smoothed_o = num_o / support
        points.append(
            MomentumPoint(
                half=half,
                minute=(b * bin_seconds + half_offsets.get(half, 0.0)) / 60.0,
                value=smoothed_c - smoothed_o,
                raw_club=per_bin_club.get((half, b), 0.0),
                raw_opponent=per_bin_opp.get((half, b), 0.0),
                kernel_support=support,
            )
        )

    return MomentumSeries(
        club_id=club_id,
        match_id=match_id,
        points=points,
        bin_seconds=bin_seconds,
        half_life_min=half_life_min,
        cap=cap,
    )
