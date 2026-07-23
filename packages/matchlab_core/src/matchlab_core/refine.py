"""Purity policies (SPO-43): terminate-over-force + GTA-style offline
split-and-reconnect, producing a refined-tracklet layer over the immutable raw
tracklets. Folds in the core Phase-4 purity policies from the superseded
tracklet-modernization PRD (SAM2-class correction stays out of scope).

Two independent mechanisms:

1. **terminate-over-force** (online, margin-gated). When the tracker must choose
   between competing candidates for a detection and the top two are near-tied,
   *forcing* the assignment risks contaminating a tracklet with the wrong
   identity. The policy refuses the forced assignment (ends/does-not-extend the
   tracklet) below a confidence margin, trading a recoverable fragmentation for
   an unrecoverable contamination. `AssignmentMargin` is the first-class,
   loggable record of each such decision (best/runner-up ids + scores + margin +
   whether it terminated), and `summarize_terminations` makes the
   contamination-vs-fragmentation trade explicit.

2. **GTA-style split-and-reconnect** (offline). `refine_tracklets` splits a raw
   tracklet at an appearance/motion discontinuity (an ID switch mid-tracklet)
   into pure fragments, then conservatively reconnects fragments that are
   confidently the same identity across a temporal gap. It returns a NEW list of
   tracklets (the refined layer); the raw tracklets are never mutated, so the
   immutable raw comparator stands alongside.

The functions are pure and consume a `feature_at(frame) -> vector` accessor, so
the appearance/motion cue is supplied by the assembly (SPO-42) while the policy
math is testable in isolation. Wiring the margin gate into the online tracker
and scoring the refined layer as its own benchmark artifact land with SPO-42.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.tracks import Tracklet, TrackletFrame

FeatureAt = Callable[[TrackletFrame], Sequence[float]]


# ---------------------------------------------------------------------------
# 1. terminate-over-force
# ---------------------------------------------------------------------------


def terminate_over_force(
    best_score: float, runner_up_score: float | None, *, margin_threshold: float
) -> bool:
    """Return True when a forced assignment should be *refused* (terminate the
    tracklet) because the top two candidates are within `margin_threshold` of
    each other. A sole candidate (`runner_up_score is None`) is never a
    near-tie. The gate is strict-less-than, so a margin exactly equal to the
    threshold is confident enough to assign. Higher score == better match."""
    if runner_up_score is None:
        return False
    return (best_score - runner_up_score) < margin_threshold


@dataclass(frozen=True)
class AssignmentMargin:
    """First-class, loggable record of one competitive assignment decision."""

    best_track_id: int
    best_score: float
    runner_up_track_id: int | None
    runner_up_score: float | None
    margin: float | None
    terminated: bool

    @classmethod
    def from_scores(
        cls,
        best_track_id: int,
        best_score: float,
        runner_up_track_id: int | None,
        runner_up_score: float | None,
        *,
        margin_threshold: float,
    ) -> AssignmentMargin:
        margin = None if runner_up_score is None else best_score - runner_up_score
        return cls(
            best_track_id=best_track_id,
            best_score=best_score,
            runner_up_track_id=runner_up_track_id,
            runner_up_score=runner_up_score,
            margin=margin,
            terminated=terminate_over_force(
                best_score, runner_up_score, margin_threshold=margin_threshold
            ),
        )


def summarize_terminations(decisions: Sequence[AssignmentMargin]) -> dict:
    """The contamination-vs-fragmentation trade, made explicit: how many
    competitive decisions terminated (introduced a fragmentation) rather than
    forcing a possibly-contaminating assignment."""
    n = len(decisions)
    n_terminated = sum(1 for d in decisions if d.terminated)
    return {
        "n_decisions": n,
        "n_terminated": n_terminated,
        "terminate_rate": (n_terminated / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. GTA-style split-and-reconnect
# ---------------------------------------------------------------------------


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity. Two zero vectors are identical (distance 0); a
    zero vs non-zero is maximally distant (1.0)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 and nb == 0.0:
        return 0.0
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _mean_feature(frames: list[TrackletFrame], feature_at: FeatureAt) -> list[float]:
    vecs = [list(feature_at(f)) for f in frames]
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]


def detect_split_frames(
    tracklet: Tracklet, feature_at: FeatureAt, *, split_threshold: float
) -> list[int]:
    """Frame indices that begin a new pure fragment: a frame whose feature is
    more than `split_threshold` from its predecessor's (an appearance/motion
    discontinuity)."""
    splits: list[int] = []
    frames = tracklet.frames
    for prev, cur in zip(frames, frames[1:]):
        if _cosine_distance(feature_at(prev), feature_at(cur)) > split_threshold:
            splits.append(cur.frame_idx)
    return splits


def _split_tracklet(
    tracklet: Tracklet, feature_at: FeatureAt, *, split_threshold: float
) -> list[list[TrackletFrame]]:
    """Partition a tracklet's frames into contiguous runs broken at every
    detected discontinuity. Returns lists of (copied) frames."""
    boundaries = set(detect_split_frames(tracklet, feature_at, split_threshold=split_threshold))
    runs: list[list[TrackletFrame]] = []
    current: list[TrackletFrame] = []
    for f in tracklet.frames:
        if f.frame_idx in boundaries and current:
            runs.append(current)
            current = []
        current.append(f.model_copy(deep=True))
    if current:
        runs.append(current)
    return runs


@dataclass
class _Fragment:
    frames: list[TrackletFrame]

    @property
    def start(self) -> int:
        return self.frames[0].frame_idx

    @property
    def end(self) -> int:
        return self.frames[-1].frame_idx

    @property
    def frame_set(self) -> set[int]:
        return {f.frame_idx for f in self.frames}


def _can_reconnect(
    earlier: _Fragment,
    later: _Fragment,
    feature_at: FeatureAt,
    *,
    reconnect_threshold: float,
    max_reconnect_gap: int,
) -> bool:
    """Conservative reconnect test for two fragments: temporally disjoint,
    within `max_reconnect_gap` idle frames, and mean-appearance within
    `reconnect_threshold`."""
    if earlier.frame_set & later.frame_set:
        return False  # temporally overlapping -> cannot be the same player
    gap = later.start - earlier.end - 1
    if gap < 0 or gap > max_reconnect_gap:
        return False
    dist = _cosine_distance(
        _mean_feature(earlier.frames, feature_at), _mean_feature(later.frames, feature_at)
    )
    return dist <= reconnect_threshold


def refine_tracklets(
    tracklets: list[Tracklet],
    feature_at: FeatureAt,
    *,
    split_threshold: float,
    reconnect_threshold: float,
    max_reconnect_gap: int,
) -> list[Tracklet]:
    """Produce the refined-tracklet layer: split every raw tracklet at its
    appearance/motion discontinuities, then conservatively reconnect fragments
    that are confidently the same identity across a gap. Returns NEW Tracklet
    objects with fresh sequential ids; the input `tracklets` are never mutated
    (the immutable raw comparator stands alongside)."""
    # 1. Split.
    fragments: list[_Fragment] = []
    for tr in tracklets:
        for run in _split_tracklet(tr, feature_at, split_threshold=split_threshold):
            fragments.append(_Fragment(frames=run))

    # 2. Conservative reconnect via union-find over fragments, earliest-first.
    fragments.sort(key=lambda fr: fr.start)
    parent = list(range(len(fragments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            if find(i) == find(j):
                continue
            if _can_reconnect(
                fragments[i], fragments[j], feature_at,
                reconnect_threshold=reconnect_threshold,
                max_reconnect_gap=max_reconnect_gap,
            ):
                parent[find(j)] = find(i)

    # 3. Materialize one refined tracklet per union group.
    groups: dict[int, list[TrackletFrame]] = {}
    for idx, frag in enumerate(fragments):
        groups.setdefault(find(idx), []).extend(frag.frames)

    refined: list[Tracklet] = []
    for new_id, (_root, frames) in enumerate(
        sorted(groups.items(), key=lambda kv: min(f.frame_idx for f in kv[1]))
    ):
        frames_sorted = sorted(frames, key=lambda f: f.frame_idx)
        refined.append(
            Tracklet(
                tracklet_id=new_id,
                cls=_dominant_cls(tracklets),
                frames=frames_sorted,
            )
        )
    return refined


def _dominant_cls(tracklets: list[Tracklet]) -> DetectionClass:
    """The class carried onto refined tracklets: the raw tracklets' shared
    class (they are all PLAYER in the shipping pipeline)."""
    return tracklets[0].cls if tracklets else DetectionClass.PLAYER
