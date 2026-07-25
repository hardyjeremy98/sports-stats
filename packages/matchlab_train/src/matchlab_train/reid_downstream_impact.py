"""Step 0 for the loose-threshold decision: what do merges actually do to the
metrics merging exists to improve?

Everything so far has counted merge edges. Edge counts do not answer whether a
given operating point is worth adopting: one wrong merge welds two players into
a single entity and can cost more entity purity than several correct merges
recover. This scores each operating point through the repo's own evaluator so
the numbers are comparable to every other benchmark in the tree.

Method: rebuild `players.json` from the merge groups a given operating point
produces, into a scratch copy of the run dir, then call `evaluate_run`. Nothing
is re-embedded and the source runs are never mutated.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from matchlab_core.evaluation import evaluate_run
from matchlab_core.gt import GroundTruth
from matchlab_core.reid.gates import (
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
)
from matchlab_core.reid.merge import merge_tracklets
from matchlab_core.reid.representation import build_representations, pair_similarity

from matchlab_train.reid_retrieval_score import gt_map_from_features, load_run

LAB = Path("/home/jeremy/code/MatchDay/lab")

# Files evaluate_run needs; frame_features.npz is deliberately excluded (large,
# and the evaluator never reads it).
_NEEDED = ("manifest.json", "tracklets.json", "detections.jsonl", "teams.json")


def groups_for(run_dir: Path, *, margin: float | None, floor: float | None, fps: float = 25.0):
    """Merge groups at one operating point. margin=None means no-op (each
    tracklet its own entity), the baseline every arm is compared against."""
    tracklets, feats, teams = load_run(run_dir)
    if margin is None:
        return [[t.tracklet_id] for t in tracklets], teams, feats
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
        tracklets,
        gates=gates,
        similarity=sim,
        min_similarity=floor,
        overlap_tolerance_frames=2,
        decision_rule="mutual-best",
        min_margin=margin,
    )
    return res.groups, teams, feats


def score_point(run_dir: Path, gt: GroundTruth, *, margin: float | None, floor: float | None):
    groups, teams, feats = groups_for(run_dir, margin=margin, floor=floor)
    gt_map = gt_map_from_features(feats)
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / run_dir.name
        scratch.mkdir(parents=True)
        for name in _NEEDED:
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, scratch / name)
        players = [
            {
                "player_id": n,
                "tracklet_ids": sorted(g),
                "team": (teams.get(sorted(g)[0]).value if teams.get(sorted(g)[0]) else "unknown"),
                "identity": {"kind": "none", "label": None, "confidence": 0.0, "evidence": []},
                "association_confidence": 1.0,
            }
            for n, g in enumerate(sorted(groups, key=min), start=1)
        ]
        (scratch / "players.json").write_text(json.dumps(players))
        result = evaluate_run(scratch, gt)
    wrong = correct = 0
    for g in groups:
        tracks = [gt_map.get(t) for t in g if gt_map.get(t) is not None]
        if len(g) > 1:
            correct += sum(1 for t in tracks if t == max(set(tracks), key=tracks.count)) - 1
            wrong += len(tracks) - tracks.count(max(set(tracks), key=tracks.count))
    lv = result["levels"]
    pur = result["purity"]["entity"]["post_filter"]
    return {
        "idf1_entity": lv["entity"]["idf1"],
        "hota_entity": result["hota"]["entity"]["hota"],
        "mota_entity": lv["entity"]["mota"],
        "idsw_entity": int(lv["entity"]["num_switches"]),
        "entity_purity": pur.get("mean_purity"),
        "entities": len(groups),
        "correct_edges": correct,
        "wrong_edges": wrong,
    }


POINTS = {
    "no-op": (None, None),
    "strict (0.10/0.95)": (0.10, 0.95),
    "loose (0.04/0.80)": (0.04, 0.80),
    "mid (0.06/0.85)": (0.06, 0.85),
}


if __name__ == "__main__":  # pragma: no cover - operator entry point
    arm = sys.argv[1]
    seqs = sys.argv[2].split(",")
    rows: dict[str, list[dict]] = {k: [] for k in POINTS}
    for s in seqs:
        gt = GroundTruth.model_validate_json(
            (LAB / f"data/videos/soccernet/{s}.gt.json").read_text()
        )
        for name, (m, f) in POINTS.items():
            rows[name].append(score_point(LAB / "data/runs" / f"gt85-{arm}-{s}", gt, margin=m, floor=f))
            print(f"  {s} {name}: {rows[name][-1]}", flush=True)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    print(f"\n=== {arm}: mean over {len(seqs)} tuning sequences ===")
    hdr = f"{'point':20} {'IDF1':>7} {'HOTA':>7} {'purity':>8} {'IDSW':>6} {'correct':>8} {'wrong':>6}"
    print(hdr)
    base = None
    for name in POINTS:
        rs = rows[name]
        idf1, hota = mean([r["idf1_entity"] for r in rs]), mean([r["hota_entity"] for r in rs])
        pur = mean([r["entity_purity"] for r in rs])
        line = (
            f"{name:20} {idf1:7.4f} {hota:7.4f} {pur if pur is None else round(pur, 4)!s:>8} "
            f"{sum(r['idsw_entity'] for r in rs):6d} {sum(r['correct_edges'] for r in rs):8d} "
            f"{sum(r['wrong_edges'] for r in rs):6d}"
        )
        if name == "no-op":
            base = (idf1, hota, pur)
        else:
            line += f"   Δ IDF1 {idf1 - base[0]:+.4f}  Δ HOTA {hota - base[1]:+.4f}"
            if pur is not None and base[2] is not None:
                line += f"  Δ purity {pur - base[2]:+.4f}"
        print(line)
