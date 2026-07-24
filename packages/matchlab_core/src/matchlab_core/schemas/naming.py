"""naming.json — the re-ID engine's closed-roster naming artifact.

One row per thread (merged tracklet group == PlayerEntity). The tracer slice
writes the skeleton (all threads abstained, empty posteriors); later slices
fill posteriors/margins (decoder), anchors_consumed (anchor framework), tier
(confidence tiers), and calibration (per-modality calibration provenance)
without reshaping the artifact. Mirrored by hand in web/src/lib/types.ts.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class NamingDecision(StrEnum):
    NAMED = "named"
    ABSTAIN = "abstain"  # first-class outcome: unknown beats silently wrong


class ConfidenceTier(StrEnum):
    """Where a thread's naming decision is routed (PRD solution step 4)."""

    AUTO_ACCEPT = "auto_accept"
    ADJUDICATE = "adjudicate"  # VLM adjudicator seam; pass-through in v1
    QA = "qa"  # human Identity QA queue


class AnchorRecord(BaseModel):
    """One anchor consumed while naming a thread — the single anchor currency
    (tracklet, roster candidate, calibrated log-likelihood ratio)."""

    tracklet_id: int
    candidate: str
    log_lr: float
    source: str  # anchor-source name, e.g. "oracle-jersey"


class ThreadNaming(BaseModel):
    thread_id: int  # == PlayerEntity.player_id
    tracklet_ids: list[int]
    label: str | None = None  # roster candidate, when decision == named
    posterior: dict[str, float] = {}  # roster candidate -> probability
    margin: float | None = None  # top1 - top2 posterior
    decision: NamingDecision = NamingDecision.ABSTAIN
    tier: ConfidenceTier | None = None  # None until the tier slice routes it
    anchors_consumed: list[AnchorRecord] = []


class NamingReport(BaseModel):
    """naming.json: per-thread roster posteriors, decisions, and the naming
    provenance (roster + calibration parameters) for one run."""

    impl: str
    params: dict = {}
    roster: list[str] = []
    threads: list[ThreadNaming] = []
    calibration: dict = {}
