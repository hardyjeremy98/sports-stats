"""Weak per-frame possessor labels for the learned Peral estimator (SPO-82).

Turns an existing run's artifacts (tracklets.json + ball.jsonl + teams.json)
into per-frame ball-possessor labels via the Phase 1 image-space heuristic
(SPO-79) -- no new annotation, no download. For each frame the possessor
tracklet is the positive tube and the other visible players are negatives, the
supervision the per-player tube classifier (Peral block 1) needs.

THESE LABELS ARE WEAK. They are exactly as good as the nearest-player heuristic:
the "ball in front of a distant player" false-possession mode Peral et al. flag
contaminates them, and abstention frames (loose ball / no ball detected) carry
no positive. An honest evaluation of a model trained on them therefore still
needs a small HAND-LABELLED held-out set -- that is deferred (see the SPO-83
gate). Do not report a possessor-accuracy number measured against these labels
as if it were ground truth.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from matchlab_core.artifacts import ArtifactStore
from matchlab_core.schemas import (
    BallObservation,
    DetectionClass,
    TeamAssignment,
    Tracklet,
)
from matchlab_core.schemas.run import ArtifactName
from matchlab_core.stages.possession.heuristic_image import HeuristicImagePossession
from pydantic import BaseModel

_POSSESSOR_CLASSES = {DetectionClass.PLAYER, DetectionClass.GOALKEEPER}

CAVEAT = (
    "Weak labels: nearest-player-to-ball heuristic (possession-heuristic-image), "
    "noisy. The 'ball in front of a distant player' false-possession mode "
    "(Peral et al. VISAPP 2025) contaminates positives, and loose-ball / "
    "no-ball frames carry no label. An honest possessor-accuracy eval needs a "
    "hand-labelled held-out set (deferred, SPO-83)."
)


class WeakPossessorFrame(BaseModel):
    frame_idx: int
    t: float
    possessor_tracklet_id: int | None
    candidate_tracklet_ids: list[int]  # visible player/GK tracklets (positives+negatives)
    confidence: float
    margin: float


class WeakPossessorLabels(BaseModel):
    source_run: str
    estimator: str
    params: dict
    weak: bool
    caveat: str
    frames: list[WeakPossessorFrame]


def derive_weak_possessor_labels(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    **params,
) -> WeakPossessorLabels:
    """Derive weak possessor labels from a run's artifacts.

    Reads tracklets (required) plus teams/ball (optional) from the run dir,
    runs the heuristic possessor estimator, and attaches per-frame candidate
    tracklets. Writes JSON to `out_path` when given. `params` are forwarded to
    the estimator (e.g. possession_radius_px, smooth_radius).
    """
    store = ArtifactStore(run_dir)
    tracklets = store.read_json_list(ArtifactName.TRACKLETS, Tracklet)
    teams = (
        store.read_json_list(ArtifactName.TEAMS, TeamAssignment)
        if store.exists(ArtifactName.TEAMS)
        else []
    )
    ball = (
        list(store.read_jsonl(ArtifactName.BALL, BallObservation))
        if store.exists(ArtifactName.BALL)
        else []
    )

    estimator = HeuristicImagePossession(**params)
    timeline = estimator.estimate(None, tracklets, teams, ball)

    candidates_by_frame: dict[int, list[int]] = defaultdict(list)
    for tr in tracklets:
        if tr.cls in _POSSESSOR_CLASSES:
            for fr in tr.frames:
                candidates_by_frame[fr.frame_idx].append(tr.tracklet_id)

    frames = [
        WeakPossessorFrame(
            frame_idx=pf.frame_idx,
            t=pf.t,
            possessor_tracklet_id=pf.possessor_tracklet_id,
            candidate_tracklet_ids=sorted(candidates_by_frame.get(pf.frame_idx, [])),
            confidence=pf.confidence,
            margin=pf.margin,
        )
        for pf in timeline
    ]

    labels = WeakPossessorLabels(
        source_run=Path(run_dir).name,
        estimator="possession-heuristic-image",
        params=estimator.params.model_dump(mode="json"),
        weak=True,
        caveat=CAVEAT,
        frames=frames,
    )
    if out_path is not None:
        Path(out_path).write_text(labels.model_dump_json(indent=2))
    return labels
