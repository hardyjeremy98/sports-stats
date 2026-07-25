"""Operating-point sweep for the mutual-best merge rule on the GT-tracklet
harness (SPO-85 amendment).

The registered point (margin 0.07 / floor 0.95) was derived against KPR's
affinity distribution. A different embedder shifts that distribution, so the
same thresholds no longer mean the same thing — this re-derives them per arm
over frozen features, which costs seconds because nothing is re-embedded.

Selection is by the pre-registered rule, applied by `select`: maximise correct
merges subject to ZERO wrong merges on EVERY tuning sequence. Ties break toward
the larger margin (further from the frontier), then the larger floor.
"""

from __future__ import annotations

import json
import sys
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

MARGINS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
FLOORS = [0.80, 0.85, 0.90, 0.95]


class _DSU:
    def __init__(self) -> None:
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


def prepare(run_dir: Path, fps: float = 25.0) -> dict:
    """Everything that does not depend on the operating point, computed once."""
    tracklets, feats, teams = load_run(run_dir)
    gt = gt_map_from_features(feats)
    reps = build_representations(feats)
    gates = [
        TemporalOverlapGate(tolerance_frames=2),
        TeamConsistencyGate(teams),
        MotionFeasibilityGate(fps=fps),
    ]
    pool = gate_passing_pairs(tracklets, gates)
    by_track: dict[int, list[int]] = {}
    for tid, track in gt.items():
        by_track.setdefault(track, []).append(tid)
    edges_needed = 0
    for members in by_track.values():
        d = _DSU()
        for i, a in enumerate(members):
            d.find(a)
            for b in members[i + 1 :]:
                if b in pool[a]:
                    d.union(a, b)
        comps: dict[int, int] = {}
        for m in members:
            comps[d.find(m)] = comps.get(d.find(m), 0) + 1
        edges_needed += sum(v - 1 for v in comps.values() if v > 1)
    return {
        "tracklets": tracklets,
        "gt": gt,
        "reps": reps,
        "gates": gates,
        "edges_needed": edges_needed,
    }


def evaluate(prepared: dict, *, margin: float, floor: float) -> tuple[int, int]:
    """(correct, wrong) merge edges at one operating point."""
    reps, gt = prepared["reps"], prepared["gt"]

    def sim(a: int, b: int):
        if a not in reps or b not in reps:
            return None
        return pair_similarity(reps[a], reps[b], min_part_visibility=0.3)

    res = merge_tracklets(
        prepared["tracklets"],
        gates=prepared["gates"],
        similarity=sim,
        min_similarity=floor,
        overlap_tolerance_frames=2,
        decision_rule="mutual-best",
        min_margin=margin,
    )
    correct = sum(1 for a, b in res.merge_edges if gt.get(a) == gt.get(b))
    return correct, len(res.merge_edges) - correct


def select(grid: list[dict]) -> dict | None:
    """Pre-registered rule: maximise correct merges subject to zero wrong on
    every sequence; ties break toward larger margin, then larger floor."""
    clean = [c for c in grid if c["wrong_total"] == 0 and c["worst_seq_wrong"] == 0]
    if not clean:
        return None
    return sorted(clean, key=lambda c: (-c["correct_total"], -c["margin"], -c["floor"]))[0]


def sweep(arm: str, seqs: list[str]) -> list[dict]:
    prepared = {s: prepare(LAB / "data/runs" / f"gt85-{arm}-{s}") for s in seqs}
    needed = sum(p["edges_needed"] for p in prepared.values())
    grid: list[dict] = []
    for margin in MARGINS:
        for floor in FLOORS:
            per_seq = {
                s: evaluate(p, margin=margin, floor=floor) for s, p in prepared.items()
            }
            grid.append(
                {
                    "margin": margin,
                    "floor": floor,
                    "correct_total": sum(c for c, _w in per_seq.values()),
                    "wrong_total": sum(w for _c, w in per_seq.values()),
                    "worst_seq_wrong": max(w for _c, w in per_seq.values()),
                    "edges_needed": needed,
                    "per_sequence": {s: {"correct": c, "wrong": w} for s, (c, w) in per_seq.items()},
                }
            )
    return grid


if __name__ == "__main__":  # pragma: no cover - operator entry point
    arm = sys.argv[1]
    seqs = sys.argv[2].split(",")
    grid = sweep(arm, seqs)
    out = LAB / f"data/runs/opsweep-{arm}.json"
    out.write_text(json.dumps(grid, indent=2))
    print(f"{'margin':>7} {'floor':>6} {'correct':>8} {'wrong':>6} {'worst':>6}")
    for c in grid:
        print(
            f"{c['margin']:7.2f} {c['floor']:6.2f} {c['correct_total']:8d} "
            f"{c['wrong_total']:6d} {c['worst_seq_wrong']:6d}"
        )
    best = select(grid)
    print()
    print(f"edges needed: {grid[0]['edges_needed']}")
    print("SELECTED:", json.dumps(best, default=str) if best else "no zero-wrong point exists")
