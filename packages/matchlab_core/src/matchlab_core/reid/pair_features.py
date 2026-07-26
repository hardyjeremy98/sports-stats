"""Per-pair features for a calibrated merge decision (SPO-85 amendment #3).

The scalar-threshold rule treats every candidate pair identically, but measured
risk is strongly context-dependent: at the loose operating point 9% of merges
are wrong at gaps <=2 s versus 21% at >5 s. Everything needed to tell those
cases apart is already computed — this module just names it.

Pure: representations + tracklets in, a feature row per candidate pair out. No
model, no I/O, so the feature definitions can be tested against hand-computed
values independently of whatever is fitted on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from matchlab_core.reid.representation import TrackletRepresentation
from matchlab_core.schemas import Tracklet

# Order is the model's input contract; changing it invalidates fitted weights.
FEATURE_NAMES: tuple[str, ...] = (
    "affinity",
    "margin",
    "mutual_best",
    "gap_seconds",
    "min_crop_height",
    "min_fragment_frames",
    "candidate_count",
    "min_part_visibility",
)


@dataclass
class PairFeature:
    a: int
    b: int
    values: list[float] = field(default_factory=list)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)


def _mean_box_height(t: Tracklet) -> float:
    return float(np.mean([f.box.y2 - f.box.y1 for f in t.frames])) if t.frames else 0.0


def build_pair_features(
    tracklets: list[Tracklet],
    reps: dict[int, TrackletRepresentation],
    pool: dict[int, set[int]],
    affinity: dict[tuple[int, int], float],
    *,
    fps: float = 25.0,
) -> list[PairFeature]:
    """One row per gate-passing pair that has an affinity.

    `affinity` is keyed by the ordered pair (min_id, max_id). `pool` is the
    gate-passing adjacency — the same candidate set the merge rule sees, so the
    model is never trained on pairs the rule could not act on.
    """
    by_id = {t.tracklet_id: t for t in tracklets}

    def aff(x: int, y: int) -> float | None:
        return affinity.get((min(x, y), max(x, y)))

    # Best and runner-up per tracklet, for margin and mutual-best.
    ranked: dict[int, list[tuple[float, int]]] = {}
    for tid, partners in pool.items():
        scored = [(aff(tid, o), o) for o in partners]
        ranked[tid] = sorted(
            [(s, o) for s, o in scored if s is not None], key=lambda r: (-r[0], r[1])
        )

    def side_margin(x: int, y: int) -> float:
        rl = ranked.get(x, [])
        if not rl or rl[0][1] != y:
            return -1.0  # y is not x's top choice; negative marks it
        return rl[0][0] - rl[1][0] if len(rl) > 1 else rl[0][0]

    out: list[PairFeature] = []
    seen: set[tuple[int, int]] = set()
    for tid, partners in pool.items():
        for other in partners:
            key = (min(tid, other), max(tid, other))
            if key in seen:
                continue
            seen.add(key)
            score = aff(*key)
            if score is None:
                continue
            a, b = by_id[key[0]], by_id[key[1]]
            first, second = (a, b) if a.start_frame <= b.start_frame else (b, a)
            m_a, m_b = side_margin(key[0], key[1]), side_margin(key[1], key[0])
            mutual = 1.0 if (m_a >= 0.0 and m_b >= 0.0) else 0.0
            margin = min(m_a, m_b) if mutual else -1.0
            gap_s = max(0, second.start_frame - first.end_frame) / fps
            vis = [float(np.mean(reps[t].part_visibility)) if t in reps else 0.0 for t in key]
            out.append(
                PairFeature(
                    a=key[0],
                    b=key[1],
                    values=[
                        float(score),
                        float(margin),
                        mutual,
                        float(gap_s),
                        min(_mean_box_height(a), _mean_box_height(b)),
                        float(min(len(a.frames), len(b.frames))),
                        float(min(len(pool[key[0]]), len(pool[key[1]]))),
                        float(min(vis)),
                    ],
                )
            )
    return out
