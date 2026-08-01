"""SPO-87 team A/B: kit-colour vs oracle team labels, everything else fixed.

The two configs differ in exactly one line (`stages.team.impl`), so every
difference below is attributable to team classification. Reports the tuning
(SNMOT-116..123) and held-out (SNMOT-124..127) tiers separately per
configs/datasets/soccernet.json; the held-out tier is the one to quote.

Usage: uv run python scripts/team_ab.py [--sequences SNMOT-124 ...] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_ARMS = {"kitcolor": "configs/pipeline.team-ab-kitcolor.yaml",
            "oracle": "configs/pipeline.team-ab-oracle.yaml",
            "siglip": "configs/pipeline.team-ab-siglip.yaml"}
# Default to the two SPO-87 arms; `--arms` selects others (e.g. the siglip
# comparison, which is a separate question from the decontamination gate).
DEFAULT_ARMS = {k: ALL_ARMS[k] for k in ("kitcolor", "oracle")}
TUNING = [f"SNMOT-{i}" for i in range(116, 124)]
HELD_OUT = [f"SNMOT-{i}" for i in range(124, 128)]


def run_arm(seq: str, arm: str, cfg: str, device: str) -> Path:
    run_id = f"team-ab-{arm}-{seq}"
    run_dir = ROOT / "data" / "runs" / run_id
    video = ROOT / "data" / "videos" / "soccernet" / f"{seq}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    proc = subprocess.run(
        ["uv", "run", "matchlab-run", "--video", str(video), "--config", cfg,
         "--device", device, "--run-id", run_id],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{run_id} failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return run_dir


def score(run_dir: Path, seq: str) -> dict:
    from matchlab_core.evaluation import evaluate_run
    from matchlab_core.gt import GroundTruth

    gt_path = ROOT / "data" / "videos" / "soccernet" / f"{seq}.gt.json"
    gt = GroundTruth.model_validate(json.loads(gt_path.read_text()))
    r = evaluate_run(run_dir, gt)
    assoc, team = r["association"], r.get("team") or {}
    gate = (team.get("gate") or {}).get("configured", {})
    return {
        "entity_idf1": r["levels"]["entity"]["idf1"],
        "tracklet_idf1": r["levels"]["tracklet"]["idf1"],
        "entity_purity": r["purity"]["entity"]["pre_filter"].get("mean_purity"),
        "merges": assoc["n_merges_total"],
        "merges_correct": assoc["n_pairs_correct"],
        "merges_wrong": assoc["n_wrong_total"],
        "merge_precision": assoc["merge_precision_all"],
        "merge_recall": assoc["merge_recall"],
        "missed_pairs": assoc["n_missed_pairs"],
        "team_accuracy": (team.get("assignment") or {}).get("accuracy"),
        "team_coverage": (team.get("assignment") or {}).get("coverage"),
        "team_by_role": (team.get("assignment") or {}).get("by_role"),
        "gate_same_player_pairs": (gate.get("same_player") or {}).get("pairs"),
        "gate_false_vetoes": (gate.get("same_player") or {}).get("vetoed"),
        "gate_false_veto_rate": gate.get("false_veto_rate"),
        "gate_opponent_pairs": (gate.get("opponents") or {}).get("pairs"),
        "gate_true_veto_rate": gate.get("true_veto_rate"),
        "gate_label_only_false_vetoes": (
            ((team.get("gate") or {}).get("label_only") or {}).get("same_player") or {}
        ).get("vetoed"),
    }


def mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarise(rows: list[dict], arm: str, seqs: list[str]) -> dict:
    sel = [r for r in rows if r["arm"] == arm and r["sequence"] in seqs]
    if not sel:
        return {}
    total = lambda k: sum(r[k] for r in sel if r[k] is not None)  # noqa: E731
    return {
        "sequences": len(sel),
        "mean_entity_idf1": mean([r["entity_idf1"] for r in sel]),
        "mean_entity_purity": mean([r["entity_purity"] for r in sel]),
        "merges": total("merges"),
        "merges_correct": total("merges_correct"),
        "merges_wrong": total("merges_wrong"),
        "merge_precision": (
            round(total("merges_correct") / total("merges"), 4) if total("merges") else None
        ),
        "missed_pairs": total("missed_pairs"),
        "mean_team_accuracy": mean([r["team_accuracy"] for r in sel]),
        "gate_same_player_pairs": total("gate_same_player_pairs"),
        "gate_false_vetoes": total("gate_false_vetoes"),
        "gate_false_veto_rate": (
            round(total("gate_false_vetoes") / total("gate_same_player_pairs"), 4)
            if total("gate_same_player_pairs") else None
        ),
        "gate_label_only_false_vetoes": total("gate_label_only_false_vetoes"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", nargs="*", default=TUNING + HELD_OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/reports/team-ab-spo87.json")
    ap.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    args = ap.parse_args()
    arms = {k: ALL_ARMS[k] for k in args.arms}

    rows: list[dict] = []
    for seq in args.sequences:
        for arm, cfg in arms.items():
            print(f"[{seq}] {arm} ...", flush=True)
            try:
                run_dir = run_arm(seq, arm, cfg, args.device)
                rows.append({"sequence": seq, "arm": arm, **score(run_dir, seq)})
            except Exception as exc:  # keep going; a partial matrix is still informative
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
                rows.append({"sequence": seq, "arm": arm, "error": str(exc)[:500]})

    ok = [r for r in rows if "error" not in r]
    payload = {
        "substrate": "oracle detect + oracle-fragment track (gap_frames 2) + osnet, "
                     "reid-engine shipped defaults, anchor_source none, jersey OFF",
        "only_variable": "stages.team.impl",
        "rows": rows,
        "tuning": {a: summarise(ok, a, TUNING) for a in arms},
        "held_out": {a: summarise(ok, a, HELD_OUT) for a in arms},
        "all": {a: summarise(ok, a, args.sequences) for a in arms},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    for tier in ("tuning", "held_out"):
        print(f"\n== {tier} ==")
        for arm in arms:
            print(f"  {arm:9s} {payload[tier].get(arm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
