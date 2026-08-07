"""Tier 1 derived statistics: event-chain stats over a canonical event stream.

Source-agnostic by construction -- nothing here imports a dataset. A ground-truth
adapter (`matchlab_train.datasets.footpass_events`) and, later, the pipeline's
own spotting/attribution output both produce `MatchEvent` streams, and every
stat is a query against that one shape.

Read `stats/schema.py` first for the event vocabulary and `stats/chains.py` for
possession segmentation -- the chain rules quietly decide four of the eleven
stats, and each of them was a measured defect before it was a rule.

Design commitments carried through every module, from the source doc:

* **An abstention is not a zero.** No goal labels means GCA is `None`, not 0; no
  xG model means xA is `None`; no duels attempted means no win rate. A zero is a
  claim that something was measured.
* **Ratios over counts.** Measured on FOOTPASS val: ratios tolerate 20-40% event
  loss within 10% movement, counts only 5-10%.
* **Every stat declares what it needs and what it cannot prove**
  (`compute.STAT_REGISTRY`), including the three that cannot be validated
  against any currently available ground truth.
"""

from matchlab_core.stats.chains import Chain, ChainingResult, build_chains, possession_changes
from matchlab_core.stats.compute import STAT_REGISTRY, compute_tier1, headline_stats
from matchlab_core.stats.schema import (
    ActorKey,
    Coverage,
    EventOutcome,
    MatchEvent,
    OutcomeSource,
    PitchPoint,
    StatEventType,
    Tier1StatLine,
    Tier1StatSheet,
)

__all__ = [
    "STAT_REGISTRY",
    "ActorKey",
    "Chain",
    "ChainingResult",
    "Coverage",
    "EventOutcome",
    "MatchEvent",
    "OutcomeSource",
    "PitchPoint",
    "StatEventType",
    "Tier1StatLine",
    "Tier1StatSheet",
    "build_chains",
    "compute_tier1",
    "headline_stats",
    "possession_changes",
]
