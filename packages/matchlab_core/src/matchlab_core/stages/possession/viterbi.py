"""Viterbi-decoded possessor estimator (B3, Notion development-path step 2).

Same slot and the same evidence as `possession-heuristic-image` (SPO-79) -- the
only difference is the temporal model, which is the point: swapping the two
impls in a config is a controlled ablation of per-frame argmin + majority
smoothing against a first-order HMM with tactical priors. All the modelling
lives in `matchlab_core.possession_denoise`; this file is only the registry
adapter.
"""

from __future__ import annotations

from matchlab_core.ball_kinematics import Params as KinematicsParams
from matchlab_core.interfaces import PossessionEstimator, StageContext
from matchlab_core.possession_denoise import DenoiseParams, denoise_possession
from matchlab_core.registry import register
from matchlab_core.schemas import (
    BallObservation,
    PossessorFrame,
    TeamAssignment,
    Tracklet,
)
from matchlab_core.schemas.run import StageKind


@register(StageKind.POSSESSION, "possession-viterbi")
class ViterbiPossession(PossessionEstimator):
    def __init__(self, **params):
        # Touch corroboration is computed internally from the ball and tracklets
        # the stage already receives, so the kinematics knobs are nested rather
        # than flattened into the denoise params -- they belong to a different
        # model and share no names by accident.
        kinematics = params.pop("kinematics", None) or {}
        self.params = DenoiseParams(**params)
        self.kinematics = KinematicsParams(**kinematics)

    def estimate(
        self,
        ctx: StageContext,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
        ball: list[BallObservation],
    ) -> list[PossessorFrame]:
        return denoise_possession(
            tracklets,
            teams,
            ball,
            params=self.params,
            kinematics=self.kinematics,
        )
