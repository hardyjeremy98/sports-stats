"""Re-express the merge accounting in tracklet and player units.

`available_true_pairs` counts PAIRS, and a player split into k fragments
contributes k(k-1)/2 of them while needing only k-1 merges to be reassembled.
So pair-recall understates how much of the job is done. This reports:

  - mergeable fragments: fragments with >=1 gate-passing same-track partner
  - reunited fragments:  of those, ones that ended up sharing a thread with a
                         true partner
  - stranded fragments:  mergeable but left with no true partner in their thread
  - edges needed:        sum over players of (gate-connected fragments - 1)
  - players whole:       players whose gate-connected fragments all ended up in
                         one thread
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

from matchlab_core.reid.gates import (
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
)
from matchlab_core.reid.merge import merge_tracklets
from matchlab_core.reid.representation import build_representations, pair_similarity
from matchlab_core.reid.retrieval import gate_passing_pairs
from matchlab_train.reid_retrieval_score import gt_map_from_features, load_run

LAB = Path("/home/jeremy/code/MatchDay/lab")
RULE, MIN_MARGIN, MIN_SIMILARITY = "mutual-best", 0.07, 0.95


class DSU:
    def __init__(self):
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def analyse(run_dir: Path, fps: float = 25.0) -> dict:
    tracklets, feats, teams = load_run(run_dir)
    gt = gt_map_from_features(feats)
    reps = build_representations(feats)
    gates = [
        TemporalOverlapGate(tolerance_frames=2),
        TeamConsistencyGate(teams),
        MotionFeasibilityGate(fps=fps),
    ]

    def sim(a: int, b: int):
        if a not in reps or b not in reps:
            return None
        return pair_similarity(reps[a], reps[b], min_part_visibility=0.3)

    res = merge_tracklets(
        tracklets, gates=gates, similarity=sim, min_similarity=MIN_SIMILARITY,
        overlap_tolerance_frames=2, decision_rule=RULE, min_margin=MIN_MARGIN,
    )
    pool = gate_passing_pairs(tracklets, gates)

    # True pairs, and which of them the gates allowed through at all.
    by_track: dict[int, list[int]] = defaultdict(list)
    for tid, track in gt.items():
        by_track[track].append(tid)
    true_pairs = gate_true = 0
    for members in by_track.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                true_pairs += 1
                if b in pool[a]:
                    gate_true += 1

    # Gate-connected components per player: the merges actually reachable.
    edges_needed = 0
    mergeable: set[int] = set()
    for members in by_track.values():
        d = DSU()
        for i, a in enumerate(members):
            d.find(a)
            for b in members[i + 1 :]:
                if b in pool[a]:
                    d.union(a, b)
                    mergeable.add(a)
                    mergeable.add(b)
        comps: dict[int, int] = defaultdict(int)
        for m in members:
            comps[d.find(m)] += 1
        edges_needed += sum(v - 1 for v in comps.values() if v > 1)

    # What the engine actually produced.
    made = DSU()
    for a, b in res.merge_edges:
        made.union(a, b)
    reunited = 0
    for tid in mergeable:
        root = made.find(tid)
        if any(
            other != tid and gt.get(other) == gt.get(tid) and made.find(other) == root
            for other in by_track[gt[tid]]
        ):
            reunited += 1

    players_whole = 0
    for members in by_track.values():
        d = DSU()
        for i, a in enumerate(members):
            d.find(a)
            for b in members[i + 1 :]:
                if b in pool[a]:
                    d.union(a, b)
        comps: dict[int, list[int]] = defaultdict(list)
        for m in members:
            comps[d.find(m)].append(m)
        for comp in comps.values():
            if len(comp) > 1 and len({made.find(m) for m in comp}) == 1:
                players_whole += 1

    return {
        "true_pairs": true_pairs,
        "gate_passing_true_pairs": gate_true,
        "mergeable_fragments": len(mergeable),
        "reunited_fragments": reunited,
        "stranded_fragments": len(mergeable) - reunited,
        "edges_needed": edges_needed,
        "correct_edges": sum(1 for a, b in res.merge_edges if gt.get(a) == gt.get(b)),
        "wrong_edges": sum(1 for a, b in res.merge_edges if gt.get(a) != gt.get(b)),
        "players_whole": players_whole,
    }


if __name__ == "__main__":
    arm, seqs = sys.argv[1], sys.argv[2].split(",")
    tot: dict[str, int] = defaultdict(int)
    for s in seqs:
        r = analyse(LAB / "data/runs" / f"gt85-{arm}-{s}")
        for k, v in r.items():
            tot[k] += v
        print(
            f"{s}  mergeable={r['mergeable_fragments']:3d} reunited={r['reunited_fragments']:3d} "
            f"stranded={r['stranded_fragments']:3d} | edges needed={r['edges_needed']:3d} "
            f"correct={r['correct_edges']:3d} wrong={r['wrong_edges']:2d} | "
            f"true pairs={r['true_pairs']:3d} gate-passing={r['gate_passing_true_pairs']:3d} "
            f"| players whole={r['players_whole']:3d}",
            flush=True,
        )
    print(f"TOTAL {arm}: {dict(tot)}")
