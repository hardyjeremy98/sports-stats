"""The anchor layer: roster model, the AnchorSource seam, and the oracle
jersey anchor source (SPO-56).

Every anchor modality emits the same currency — (tracklet, roster candidate,
calibrated log-likelihood ratio) — so fusion and decode never care where
evidence came from. The v1 benchmark source is the oracle jersey anchor:
ground-truth jersey identities under controlled degradation (coverage, label
noise, minimum box height, seed), the same epistemic trick as the oracle
detector — isolate the naming layer's own error with anchors of known
quantity and quality. GT jersey identities are consumed HERE and by the
roster builder only; they are never a pipeline perception input (ADR 001).
The face source is a registered stub emitting nothing until footage with
harvestable faces exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from matchlab_core.gt import GroundTruth, GroundTruthTrack
from matchlab_core.schemas import Tracklet
from matchlab_core.schemas.geometry import Box


@dataclass(frozen=True)
class Anchor:
    """The single anchor currency."""

    tracklet_id: int
    candidate: str  # roster candidate id, e.g. "left:7"
    log_lr: float  # calibrated log-likelihood ratio for candidate vs not
    source: str  # producing anchor-source name


@dataclass(frozen=True)
class Roster:
    """The N candidate identities threads are named against. The benchmark
    roster is built from the sequence GT's identified jersey set (team:number
    — numbers repeat across teams); a product enrollment roster can replace
    it without touching the engine."""

    candidates: list[str]

    @classmethod
    def from_ground_truth(cls, gt: GroundTruth) -> Roster:
        seen: dict[str, None] = {}
        for track in gt.tracks:
            cand = candidate_for_track(track)
            if cand is not None:
                seen.setdefault(cand, None)
        return cls(candidates=sorted(seen))


def candidate_for_track(track: GroundTruthTrack) -> str | None:
    """GT track -> roster candidate id, or None when the track has no
    identified jersey (letters mean unidentified) or no team side."""
    if track.role not in ("player", "goalkeeper"):
        return None
    if track.team is None or track.jersey is None or not track.jersey.isdigit():
        return None
    return f"{track.team}:{track.jersey}"


class AnchorSource(Protocol):
    name: str

    def anchors(self, tracklets: list[Tracklet], roster: Roster) -> list[Anchor]: ...


def _iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def match_tracklets_to_gt(
    tracklets: list[Tracklet],
    gt: GroundTruth,
    *,
    min_iou: float = 0.5,
    min_overlap_frames: int = 1,
) -> dict[int, GroundTruthTrack]:
    """Argmax-overlap assignment of tracklets to GT tracks: a tracklet matches
    the GT track it shares the most frames with at IoU >= `min_iou`. Tracklets
    matching nothing are simply absent (missing evidence is neutral)."""
    gt_frames: list[dict[int, Box]] = [
        {f.frame_idx: f.box for f in track.frames} for track in gt.tracks
    ]
    matched: dict[int, GroundTruthTrack] = {}
    for tr in tracklets:
        counts = np.zeros(len(gt.tracks), dtype=np.int64)
        for tf in tr.frames:
            for gi, frames in enumerate(gt_frames):
                box = frames.get(tf.frame_idx)
                if box is not None and _iou(tf.box, box) >= min_iou:
                    counts[gi] += 1
        if counts.size and counts.max() >= min_overlap_frames:
            matched[tr.tracklet_id] = gt.tracks[int(counts.argmax())]
    return matched


class OracleJerseyAnchorSource:
    """Anchors from GT jersey identities under controlled degradation.

    Eligible tracklets (matched to an identified-jersey GT track, tallest box
    >= `min_box_height`) are sampled to a `coverage` fraction (seeded,
    deterministic); each emitted anchor's candidate is flipped to a uniformly
    chosen wrong roster member with probability `noise`. The log-LR is
    calibrated to the configured noise rate (floored so noise=0 stays finite).
    """

    name = "oracle-jersey"

    def __init__(
        self,
        gt: GroundTruth,
        *,
        coverage: float = 1.0,
        noise: float = 0.0,
        min_box_height: float = 0.0,
        seed: int = 0,
        min_iou: float = 0.5,
    ):
        self.gt = gt
        self.coverage = coverage
        self.noise = noise
        self.min_box_height = min_box_height
        self.seed = seed
        self.min_iou = min_iou

    @property
    def log_lr(self) -> float:
        eps = min(max(self.noise, 1e-3), 1.0 - 1e-3)
        return math.log((1.0 - eps) / eps)

    def anchors(self, tracklets: list[Tracklet], roster: Roster) -> list[Anchor]:
        matched = match_tracklets_to_gt(tracklets, self.gt, min_iou=self.min_iou)
        eligible: list[tuple[int, str]] = []
        for tr in sorted(tracklets, key=lambda t: t.tracklet_id):
            track = matched.get(tr.tracklet_id)
            if track is None:
                continue
            cand = candidate_for_track(track)
            if cand is None or cand not in roster.candidates:
                continue
            tallest = max(tf.box.height for tf in tr.frames)
            if tallest < self.min_box_height:
                continue
            eligible.append((tr.tracklet_id, cand))

        rng = np.random.default_rng(self.seed)
        n_keep = int(round(self.coverage * len(eligible)))
        keep_idx = sorted(rng.permutation(len(eligible))[:n_keep])

        anchors: list[Anchor] = []
        for i in keep_idx:
            tid, cand = eligible[i]
            if self.noise > 0 and rng.random() < self.noise:
                wrong = [c for c in roster.candidates if c != cand]
                if wrong:
                    cand = wrong[int(rng.integers(len(wrong)))]
            anchors.append(
                Anchor(tracklet_id=tid, candidate=cand, log_lr=self.log_lr, source=self.name)
            )
        return anchors


class FaceAnchorSource:
    """The face anchor stream seam: registered, emits nothing. No owned
    footage contains harvestable faces; implementation lands when suitable
    footage exists (PRD: out of scope for v1)."""

    name = "face"

    def anchors(self, tracklets: list[Tracklet], roster: Roster) -> list[Anchor]:
        return []
