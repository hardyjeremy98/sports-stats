"""Calibrated pair model for merge decisions (SPO-85 amendment #3).

Replaces the global (floor, margin) threshold with P(same player | pair
features), so risk can be spent where it is cheap (short gaps, large crops) and
refused where it is expensive. Measured motivation: at the loose operating point
9% of merges are wrong at gaps <=2 s versus 21% at >5 s.

**Every reported number is out-of-fold.** Leave-one-sequence-out: for each
tuning sequence the model is fitted on the other seven and applied only to the
held-back one. Fitting and scoring on all eight would inflate the result, which
is precisely the trap this design exists to avoid — so the in-fold number is
computed too, and a large in/out gap is reported as evidence of overfitting
rather than quietly ignored.
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
from matchlab_core.reid.pair_features import FEATURE_NAMES, build_pair_features
from matchlab_core.reid.representation import build_representations, pair_similarity
from matchlab_core.reid.retrieval import gate_passing_pairs
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matchlab_train.reid_downstream_impact import LAB, score_point
from matchlab_train.reid_retrieval_score import gt_map_from_features, load_run

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
BASELINE = {"idf1": 0.9177, "purity": 0.9936}


def prepare(run_dir: Path, fps: float = 25.0) -> dict:
    tracklets, feats, teams = load_run(run_dir)
    reps = build_representations(feats)
    gates = [
        TemporalOverlapGate(tolerance_frames=2),
        TeamConsistencyGate(teams),
        MotionFeasibilityGate(fps=fps),
    ]
    pool = gate_passing_pairs(tracklets, gates)
    aff: dict[tuple[int, int], float] = {}
    for tid, partners in pool.items():
        for other in partners:
            key = (min(tid, other), max(tid, other))
            if key in aff or key[0] not in reps or key[1] not in reps:
                continue
            s = pair_similarity(reps[key[0]], reps[key[1]], min_part_visibility=0.3)
            if s is not None:
                aff[key] = float(s)
    rows = build_pair_features(tracklets, reps, pool, aff, fps=fps)
    gt = gt_map_from_features(feats)
    X = np.stack([r.as_array() for r in rows]) if rows else np.zeros((0, len(FEATURE_NAMES)))
    y = np.asarray(
        [1 if gt.get(r.a) is not None and gt.get(r.a) == gt.get(r.b) else 0 for r in rows]
    )
    order = [t.tracklet_id for t in tracklets]
    return {"rows": rows, "X": X, "y": y, "order": order, "n": len(order)}


def _fit(X: np.ndarray, y: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    model.fit(X, y)
    return model


def _matrix_from_probs(prep: dict, probs: np.ndarray) -> np.ndarray:
    pos = {t: i for i, t in enumerate(prep["order"])}
    m = np.full((prep["n"], prep["n"]), np.nan)
    np.fill_diagonal(m, 1.0)
    for r, p in zip(prep["rows"], probs, strict=True):
        m[pos[r.a], pos[r.b]] = m[pos[r.b], pos[r.a]] = float(p)
    return m


def main() -> int:
    seqs = sys.argv[1].split(",")
    preps = {s: prepare(LAB / "data/runs" / f"gt85-prtreid-{s}") for s in seqs}
    print(f"pairs per sequence: { {s: len(p['rows']) for s, p in preps.items()} }")
    print(f"positives total: {sum(int(p['y'].sum()) for p in preps.values())}")

    # Leave-one-sequence-out: out-of-fold probabilities for every sequence.
    oof: dict[str, np.ndarray] = {}
    infold: dict[str, np.ndarray] = {}
    for s in seqs:
        tr = [t for t in seqs if t != s]
        X = np.concatenate([preps[t]["X"] for t in tr])
        y = np.concatenate([preps[t]["y"] for t in tr])
        model = _fit(X, y)
        oof[s] = model.predict_proba(preps[s]["X"])[:, 1]
        infold[s] = _fit(preps[s]["X"], preps[s]["y"]).predict_proba(preps[s]["X"])[:, 1]

    gts = {
        s: GroundTruth.model_validate_json(
            (LAB / f"data/videos/soccernet/{s}.gt.json").read_text()
        )
        for s in seqs
    }

    def score(probs: dict[str, np.ndarray], p_star: float, label: str) -> dict:
        rows = []
        for s in seqs:
            m = _matrix_from_probs(preps[s], probs[s])
            rows.append(
                score_point(
                    LAB / "data/runs" / f"gt85-prtreid-{s}", gts[s],
                    margin=0.0, floor=p_star, matrix=m, order=preps[s]["order"],
                )
            )
        idf1 = sum(r["idf1_entity"] for r in rows) / len(rows)
        hota = sum(r["hota_entity"] for r in rows) / len(rows)
        purs = [r["entity_purity"] for r in rows if r["entity_purity"] is not None]
        pur = sum(purs) / len(purs) if purs else None
        out = {
            "label": label, "p_star": p_star, "idf1": idf1, "hota": hota, "purity": pur,
            "worst_purity": min(purs) if purs else None,
            "idsw": sum(r["idsw_entity"] for r in rows),
            "correct": sum(r["correct_edges"] for r in rows),
            "wrong": sum(r["wrong_edges"] for r in rows),
        }
        print(
            f"{label:28} p*={p_star:.1f}  IDF1 {idf1:.4f}  HOTA {hota:.4f}  "
            f"purity {pur:.4f} (worst {out['worst_purity']:.4f})  IDSW {out['idsw']:3d}  "
            f"{out['correct']}✓/{out['wrong']}✗",
            flush=True,
        )
        return out

    print("\n=== out-of-fold (the honest estimate) ===")
    oof_results = [score(oof, p, "calibrated OOF") for p in THRESHOLDS]
    print("\n=== in-fold (overfitting check — NOT a result) ===")
    infold_results = [score(infold, p, "calibrated in-fold") for p in THRESHOLDS]

    eligible = [r for r in oof_results if r["purity"] is not None and r["purity"] >= 0.99]
    best = max(eligible, key=lambda r: r["idf1"]) if eligible else None
    print(f"\nBaseline to beat: IDF1 {BASELINE['idf1']:.4f} at purity {BASELINE['purity']:.4f}")
    if best is None:
        print("SELECTED: none — no out-of-fold threshold clears purity >= 0.99")
    else:
        verdict = "BEATS" if best["idf1"] > BASELINE["idf1"] else "does NOT beat"
        print(f"SELECTED: p*={best['p_star']} IDF1 {best['idf1']:.4f} -> {verdict} the baseline")
    bi = max(infold_results, key=lambda r: r["idf1"])["idf1"]
    bo = max(oof_results, key=lambda r: r["idf1"])["idf1"]
    print(f"in-fold best {bi:.4f} vs out-of-fold best {bo:.4f}  (gap {bi - bo:+.4f})")
    (LAB / "data/runs/calibrated-merge.json").write_text(
        json.dumps({"oof": oof_results, "infold": infold_results}, indent=2)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - operator entry point
    sys.exit(main())
