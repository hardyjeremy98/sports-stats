"""k-reciprocal re-ranking arm for the GT-tracklet harness (SPO-85 amendment #2).

Two stages, because `evaluate_run` is far more expensive than a merge:

  A. sweep (k1, lambda, margin, floor) scoring only merge edges — cheap, so the
     whole grid is affordable;
  B. score the finalists downstream (entity IDF1 / HOTA / purity), which is what
     the registered objective is actually stated in.

Selection: maximise mean entity IDF1 subject to mean entity purity >= 0.99.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matchlab_core.gt import GroundTruth
from matchlab_core.reid.gates import (
    MotionFeasibilityGate,
    TeamConsistencyGate,
    TemporalOverlapGate,
)
from matchlab_core.reid.merge import merge_tracklets
from matchlab_core.reid.representation import build_representations, pair_similarity
from matchlab_core.reid.rerank import k_reciprocal_rerank

from matchlab_train.reid_downstream_impact import LAB, score_point
from matchlab_train.reid_retrieval_score import gt_map_from_features, load_run

K1S = [5, 10, 20]
LAMBDAS = [0.0, 0.3, 0.6]
MARGINS = [0.02, 0.04, 0.06, 0.10]
FLOORS = [0.0, 0.2, 0.4, 0.6, 0.8]
WRONG_BUDGET = 10  # stage-A filter, ~the accepted loose point's 8


def _prepared(run_dir: Path, fps: float = 25.0) -> dict:
    tracklets, feats, teams = load_run(run_dir)
    reps = build_representations(feats)
    ids = [t.tracklet_id for t in tracklets]
    n = len(ids)
    pos = {t: i for i, t in enumerate(ids)}
    aff = np.full((n, n), np.nan)
    for i, a in enumerate(ids):
        aff[i, i] = 1.0
        for b in ids[i + 1 :]:
            s = (
                None
                if a not in reps or b not in reps
                else pair_similarity(reps[a], reps[b], min_part_visibility=0.3)
            )
            if s is not None:
                aff[i, pos[b]] = aff[pos[b], i] = float(s)
    return {
        "order": ids,
        "tracklets": tracklets,
        "gt": gt_map_from_features(feats),
        "aff": aff,
        "pos": pos,
        "gates": [
            TemporalOverlapGate(tolerance_frames=2),
            TeamConsistencyGate(teams),
            MotionFeasibilityGate(fps=fps),
        ],
    }


def _edges(prep: dict, matrix: np.ndarray, *, margin: float, floor: float):
    pos, gt = prep["pos"], prep["gt"]

    def sim(a: int, b: int):
        v = matrix[pos[a], pos[b]]
        return None if np.isnan(v) else float(v)

    res = merge_tracklets(
        prep["tracklets"],
        gates=prep["gates"],
        similarity=sim,
        min_similarity=floor,
        overlap_tolerance_frames=2,
        decision_rule="mutual-best",
        min_margin=margin,
    )
    correct = sum(1 for a, b in res.merge_edges if gt.get(a) == gt.get(b))
    return correct, len(res.merge_edges) - correct


def main() -> int:
    seqs = sys.argv[1].split(",")
    preps = {s: _prepared(LAB / "data/runs" / f"gt85-prtreid-{s}") for s in seqs}

    print("=== stage A: edge counts over the grid ===", flush=True)
    grid = []
    for k1 in K1S:
        for lam in LAMBDAS:
            mats = {
                s: k_reciprocal_rerank(p["aff"], k1=k1, k2=3, lambda_value=lam)
                for s, p in preps.items()
            }
            for margin in MARGINS:
                for floor in FLOORS:
                    c = w = 0
                    for s, p in preps.items():
                        ec, ew = _edges(p, mats[s], margin=margin, floor=floor)
                        c += ec
                        w += ew
                    grid.append(
                        {"k1": k1, "lambda": lam, "margin": margin, "floor": floor,
                         "correct": c, "wrong": w}
                    )
    (LAB / "data/runs/rerank-stageA.json").write_text(json.dumps(grid, indent=2))

    eligible = [g for g in grid if g["wrong"] <= WRONG_BUDGET]
    finalists = sorted(eligible, key=lambda g: -g["correct"])[:5]
    print(f"grid={len(grid)} eligible(wrong<={WRONG_BUDGET})={len(eligible)}")
    for g in finalists:
        print(f"  k1={g['k1']:2d} lam={g['lambda']:.1f} m={g['margin']:.2f} "
              f"f={g['floor']:.1f} -> {g['correct']} correct / {g['wrong']} wrong")

    print("\n=== stage B: downstream metrics for finalists + baseline ===", flush=True)
    gts = {
        s: GroundTruth.model_validate_json(
            (LAB / f"data/videos/soccernet/{s}.gt.json").read_text()
        )
        for s in seqs
    }

    def downstream(matrix_for, margin, floor, label):
        rows = []
        for s in seqs:
            rows.append(
                score_point(
                    LAB / "data/runs" / f"gt85-prtreid-{s}", gts[s],
                    margin=margin, floor=floor,
                    matrix=None if matrix_for is None else matrix_for[s],
                    order=preps[s]["order"],
                )
            )
        idf1 = sum(r["idf1_entity"] for r in rows) / len(rows)
        hota = sum(r["hota_entity"] for r in rows) / len(rows)
        purs = [r["entity_purity"] for r in rows if r["entity_purity"] is not None]
        pur = sum(purs) / len(purs) if purs else None
        print(
            f"{label:38} IDF1 {idf1:.4f}  HOTA {hota:.4f}  purity "
            f"{pur if pur is None else round(pur, 4)}  IDSW {sum(r['idsw_entity'] for r in rows)}"
            f"  worst-seq purity {min(purs) if purs else None}",
            flush=True,
        )
        return {"label": label, "idf1": idf1, "hota": hota, "purity": pur}

    results = [downstream(None, 0.04, 0.80, "baseline plain affinity 0.04/0.80")]
    for g in finalists:
        mats = {
            s: k_reciprocal_rerank(p["aff"], k1=g["k1"], k2=3, lambda_value=g["lambda"])
            for s, p in preps.items()
        }
        results.append(
            downstream(mats, g["margin"], g["floor"],
                       f"rerank k1={g['k1']} lam={g['lambda']} m={g['margin']} f={g['floor']}")
        )

    ok = [r for r in results if r["purity"] is not None and r["purity"] >= 0.99]
    best = max(ok, key=lambda r: r["idf1"]) if ok else None
    print("\nSELECTED (max IDF1 s.t. purity >= 0.99):", best["label"] if best else "none eligible")
    (LAB / "data/runs/rerank-stageB.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    sys.exit(main())
