"""Weighted Event Fusion: combine several DST models' event lists into one.

PAVE section 4, the last of its four contributions and the largest single one after
stage 1 -- worth +0.042 VAL macro-F1 to them, and the difference between 55.1 and
58.94 on the challenge split.

It is pure post-processing over `PCBASEvent` lists, so it needs no model, no GPU and
no logits: fuse whatever N models produced, score each surviving cluster by how many
of them agreed, and drop what only one found.

    group by (identity, class) -> cluster within +/-delta, greedy on descending score
    -> score = mean(contributing) * (n/N)^0.5 -> drop n < 2, except `solo_classes`

Three details are ours rather than the paper's, recorded so a miss stays attributable:

* **The cluster's frame is its seed's frame** -- the highest-scoring member. The paper
  does not say how a cluster's timestamp is chosen. The seed is the natural
  representative given the assignment is already seeded on it, and averaging frames
  would invent a timestamp no model proposed.
* **One model contributes at most once per cluster.** Otherwise a single model that
  emitted two nearby events would satisfy the >=2 agreement filter alone, which
  defeats the filter's purpose.
* **`identity` is a parameter** rather than fixed to shirt. PAVE groups by
  (team, shirt, class); our models predict role slots, and `slot` already packs team
  and role together (ADR 008). Slot is the default because that is what is native;
  "shirt" is available for fusing events that have been through the export-time
  roster remap, which is the form the reference's on-disk exchange uses.

The tackle exemption is PAVE's and is explicit in the paper: the agreement filter was
deleting the only correct tackle predictions. That is not a quirk of their run -- tackle
has 26 GT events in VAL and 174 trainable anchors in all of TRAIN, so the handful of
hits any model gets are exactly the ones a 2-of-N filter removes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.schema import CLASS_NAMES

FusionIdentity = Literal["slot", "shirt"]

# Same tolerance the metric matches at, and what PAVE clusters within.
DEFAULT_FUSION_DELTA = 12
# Minimum number of models that must agree for a cluster to survive.
DEFAULT_MIN_MODELS = 2
# Classes exempt from the agreement filter. PAVE exempts tackle explicitly.
DEFAULT_SOLO_CLASSES: tuple[str, ...] = ("tackle",)


def _identity_key(event: PCBASEvent, identity: FusionIdentity) -> tuple:
    if identity == "shirt":
        return (event.left_to_right, event.shirt_number)
    return (event.slot,)


def fuse_model_events(
    per_model: Sequence[Sequence[PCBASEvent]],
    *,
    delta: int = DEFAULT_FUSION_DELTA,
    identity: FusionIdentity = "slot",
    min_models: int = DEFAULT_MIN_MODELS,
    solo_classes: Sequence[str] = DEFAULT_SOLO_CLASSES,
) -> list[PCBASEvent]:
    """Fuse N models' event lists into one, weighted by agreement.

    Args:
        per_model: one event list per model. Empty lists are allowed and still count
            toward N -- a model that found nothing is a model that disagreed.
        delta: cluster radius in frames.
        identity: what counts as "the same player" -- role slot, or (team, shirt).
        min_models: agreement threshold; clusters with fewer supporters are dropped.
        solo_classes: class names exempt from that threshold.

    Returns:
        One event per surviving cluster, sorted by frame.
    """
    n_models = len(per_model)
    if n_models == 0:
        return []
    exempt = {CLASS_NAMES.index(name) for name in solo_classes if name in CLASS_NAMES}

    # (identity, class) -> [(model_index, event), ...]
    groups: dict[tuple, list[tuple[int, PCBASEvent]]] = {}
    for model_idx, events in enumerate(per_model):
        for event in events:
            key = (*_identity_key(event, identity), event.class_id)
            groups.setdefault(key, []).append((model_idx, event))

    fused: list[PCBASEvent] = []
    for (*_, class_id), members in groups.items():
        # Greedy assignment on DESCENDING score: the strongest proposal seeds a
        # cluster and absorbs everything unassigned within delta of it.
        remaining = sorted(members, key=lambda pair: -pair[1].score)
        assigned: set[int] = set()
        for position, (seed_model, seed) in enumerate(remaining):
            if position in assigned:
                continue
            assigned.add(position)
            cluster = [(seed_model, seed)]
            contributing = {seed_model}
            for other, (model_idx, event) in enumerate(remaining):
                if other in assigned or model_idx in contributing:
                    continue
                if abs(event.frame_idx - seed.frame_idx) <= delta:
                    assigned.add(other)
                    cluster.append((model_idx, event))
                    contributing.add(model_idx)

            n = len(contributing)
            if n < min_models and class_id not in exempt:
                continue
            mean_score = sum(event.score for _, event in cluster) / len(cluster)
            fused.append(
                seed.model_copy(
                    update={"score": mean_score * (n / n_models) ** 0.5}
                )
            )

    fused.sort(key=lambda e: (e.frame_idx, e.slot if e.slot is not None else -1))
    return fused
