"""Gap-site evaluation: score the merge engine per DECISION, not per clip.

The clip-level merge metric on 30 s SNMOT runs is 8 events pooled over six
clips -- an evaluation with no resolution (a full threshold x margin sweep
returned identical counts at every point). Every pass-1 merge decision,
however, is a gap site with a ground-truth answer, and the engine already
records its full scored hypothesis set (reid_detail.json `decisions`,
un-truncated via `reid_detail_max_candidates`). This harness replays the
associate stage over frozen run artifacts, GT-labels the tracklets, and turns
each decision into a row that supports the questions the merge counts cannot:

- was the true link IN the candidate set at all (candidate recall) or present
  but out-ranked (ranking failure)? Different fixes; indistinguishable in
  right/wrong/missed counts.
- for each wrong merge: close runner-up (denominator failure) or uncontested
  winner (candidate-set failure)? And was the chosen thread's own true
  continuation left unexplained (a global-assignment symptom)?
- per-channel LLR contributions on correct vs incorrect decisions, binned by
  gap length and candidate density.
- bootstrap noise floors for the clip-level merge metric AND the site-level
  metrics, so "no difference" claims carry a detectable-effect-size.

Verdict convention: a candidate thread is judged by its FACING END (the
`partner` tracklet the engine recorded), same edge convention as
`link_endpoints` in bootstrap_threads.py. Tracklets with no GT label or
purity < 0.8 are excluded from verdicts (never counted right OR wrong), and
their exclusion is reported.

Usage:
  uv run python -m matchlab_train.experiments.gapsite_eval \
      --runs best2-120 best2-123 best2-124 best2-125 best2-126 best2-127
Writes data/experiments/gapsite-eval/<run>.json (full retained hypothesis
sets, one row per decision) and .../summary.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

MIN_IOU_LABEL = 0.3
MIN_PURITY = 0.8

#: The round-2 best-measured operating point (pipeline.reid-best-snmot.yaml),
#: with the full hypothesis set retained. gmc off: camera-motion estimation
#: only feeds the pairwise engine's motion gate, and decoding frames per
#: replay would dominate the runtime.
BEST_PARAMS = {
    "merge_strategy": "two-pass",
    "fusion_model": "configs/reid/fusion-footpass-v1.json",
    "pass1_min_score": 4.0,
    "pass2_min_score": 2.0,
    "merge_min_margin": 0.5,
    "occupancy_coords": "formation-relative",
    "calibration_min_confidence": 0.5,
    "anchor_source": "none",
    "jersey_enabled": False,
    "gmc": False,
    "reid_detail_max_candidates": 1_000_000,
}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def gt_label_tracklets(tracklet_rows: list[dict], gt: dict) -> dict[int, int]:
    """tracklet_id -> majority GT track id, for tracklets above the purity bar.

    Per-frame IoU argmax vote against GT player/goalkeeper boxes; a tracklet
    whose majority label carries < MIN_PURITY of its labelled frames is left
    out -- it is not one player, so no per-player verdict applies to it.
    """
    gt_frames: dict[int, dict[int, np.ndarray]] = {}
    for tr in gt["tracks"]:
        if tr["role"] not in ("player", "goalkeeper"):
            continue
        gt_frames[tr["track_id"]] = {
            fr["frame_idx"]: np.array(
                [fr["box"]["x1"], fr["box"]["y1"], fr["box"]["x2"], fr["box"]["y2"]]
            )
            for fr in tr["frames"]
        }
    out: dict[int, int] = {}
    for t in tracklet_rows:
        if t.get("cls") not in (None, "player", "goalkeeper"):
            continue
        votes: dict[int, int] = {}
        n = 0
        for fr in t["frames"]:
            b = fr["box"]
            box = np.array([b["x1"], b["y1"], b["x2"], b["y2"]])
            cands = [
                (gid, fmap[fr["frame_idx"]])
                for gid, fmap in gt_frames.items()
                if fr["frame_idx"] in fmap
            ]
            if not cands:
                continue
            ious = _iou(box[None, :], np.stack([c[1] for c in cands]))[0]
            k = int(np.argmax(ious))
            if ious[k] >= MIN_IOU_LABEL:
                votes[cands[k][0]] = votes.get(cands[k][0], 0) + 1
                n += 1
        if votes:
            gid = max(votes, key=votes.get)
            if votes[gid] / n >= MIN_PURITY:
                out[t["tracklet_id"]] = gid
    return out


def replay_associate(run_dir: Path, params: dict, tmp: Path) -> tuple[dict, dict]:
    """Re-run the associate stage over a copy of `run_dir`'s artifacts.

    Returns (reid_detail, association) dicts. Imports are local so the module
    can be inspected without a synced core environment.
    """
    import matchlab_core.stages  # noqa: F401
    from matchlab_core.artifacts import ArtifactStore
    from matchlab_core.config import PipelineConfig
    from matchlab_core.interfaces import StageContext
    from matchlab_core.registry import build
    from matchlab_core.schemas import ArtifactName, TeamAssignment, Tracklet
    from matchlab_core.schemas.run import StageKind, VideoMeta

    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(run_dir, tmp)
    manifest = json.loads((tmp / "manifest.json").read_text())
    store = ArtifactStore(tmp)
    tracklets = [
        Tracklet.model_validate(r)
        for r in json.loads(store.path(ArtifactName.TRACKLETS).read_text())
    ]
    teams = [
        TeamAssignment.model_validate(r)
        for r in json.loads(store.path(ArtifactName.TEAMS).read_text())
    ]
    ctx = StageContext(
        video=VideoMeta.model_validate(manifest["video"]),
        config=PipelineConfig.model_validate(manifest["config"]),
        store=store,
    )
    build(StageKind.ASSOCIATE, "reid-engine", params).associate(ctx, tracklets, teams)
    detail = json.loads(store.path(ArtifactName.REID_DETAIL).read_text())
    assoc = json.loads(store.path(ArtifactName.ASSOCIATION).read_text())
    return detail, assoc


def site_rows(run_id: str, run_dir: Path, detail: dict, assoc: dict) -> dict:
    """One row per pass-1 decision, with the full hypothesis set and verdicts."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    seq = Path(manifest["video"]["path"]).stem
    gt = json.loads(
        (Path("data/videos/soccernet") / f"{seq}.gt.json").read_text()
    )
    raw = json.loads((run_dir / "tracklets.json").read_text())
    tracklet_rows = raw["tracklets"] if isinstance(raw, dict) else raw
    gt_of = gt_label_tracklets(tracklet_rows, gt)
    span = {
        t["tracklet_id"]: (
            min(fr["frame_idx"] for fr in t["frames"]),
            max(fr["frame_idx"] for fr in t["frames"]),
        )
        for t in tracklet_rows
        if t["frames"]
    }
    fps = manifest["video"]["fps"]

    # Final entity of each tracklet, for the "chosen thread's own true
    # continuation left unexplained" breakdown.
    ent_of: dict[int, int] = {}
    for e in assoc.get("entities", []):
        for tid in e["tracklet_ids"]:
            ent_of[tid] = e["player_id"]

    labelled = sorted(gt_of, key=lambda t: span[t][0])
    rows = []
    for d in detail.get("decisions", []):
        q = d["tracklet_id"]
        g = gt_of.get(q)
        # True predecessor: latest labelled same-player tracklet ending before
        # q starts. Its existence makes this site LINKABLE; a linkable site
        # with no true candidate in the set is a candidate-recall failure.
        pred = None
        if g is not None:
            before = [
                t for t in labelled
                if gt_of[t] == g and t != q and span[t][1] < span[q][0]
            ]
            if before:
                pred = max(before, key=lambda t: span[t][1])
        cands = []
        true_rank, true_row = None, None
        for rank, c in enumerate(
            sorted(d.get("candidates", []), key=lambda c: -c["total"]), start=1
        ):
            cg = gt_of.get(c["partner"])
            is_true = g is not None and cg == g
            cands.append({
                "rank": rank,
                "partner": c["partner"],
                "partner_gt": cg,
                "is_true": is_true,
                "total": c["total"],
                "gap_s": (span[q][0] - span[c["partner"]][1]) / fps,
                "channels": {
                    ch["name"]: {"llr": ch["llr"], "contribution": ch["contribution"]}
                    for ch in c.get("channels", [])
                },
            })
            if is_true and true_rank is None:
                true_rank, true_row = rank, cands[-1]
        chosen_gt = gt_of.get(d["chosen"]) if d.get("chosen") is not None else None
        verdict = None
        if d["decision"] == "merged" and g is not None and chosen_gt is not None:
            verdict = "right" if chosen_gt == g else "wrong"
        unexplained = None
        if verdict == "wrong":
            # Did the thread q wrongly joined lose ITS player's own next
            # fragment (it ended up in a different final entity)?
            succ = [
                t for t in labelled
                if gt_of[t] == chosen_gt and span[t][0] > span[d["chosen"]][1]
            ]
            if succ:
                nxt = min(succ, key=lambda t: span[t][0])
                unexplained = ent_of.get(nxt) != ent_of.get(d["chosen"])
        rows.append({
            "run": run_id,
            "tracklet_id": q,
            "gt_id": g,
            "decision": d["decision"],
            "chosen": d.get("chosen"),
            "chosen_gt": chosen_gt,
            "verdict": verdict,
            "total": d.get("total"),
            "runner_up": d.get("runner_up"),
            "margin": (
                None
                if d.get("total") is None or d.get("runner_up") is None
                else d["total"] - d["runner_up"]
            ),
            "threshold": d["threshold"],
            "n_candidates": d.get("n_candidates_total", len(cands)),
            "linkable": pred is not None,
            "true_pred": pred,
            "true_in_candidates": true_rank is not None,
            "true_rank": true_rank,
            "true_total": None if true_row is None else true_row["total"],
            "true_gap_s": None if true_row is None else true_row["gap_s"],
            "chosen_thread_unexplained": unexplained,
            "candidates": cands,
        })
    return {
        "run": run_id,
        "seq": seq,
        "n_tracklets": len(tracklet_rows),
        "n_labelled": len(gt_of),
        "rows": rows,
    }


def _boot_ci(stat, items: list, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    items = list(items)
    for _ in range(n):
        sample = [items[i] for i in rng.integers(0, len(items), len(items))]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(per_run: list[dict]) -> dict:
    rows = [r for pr in per_run for r in pr["rows"]]
    linkable = [r for r in rows if r["linkable"]]
    merged = [r for r in rows if r["verdict"] in ("right", "wrong")]
    wrong = [r for r in merged if r["verdict"] == "wrong"]

    def precision(sample):
        m = [r for r in sample if r["verdict"] in ("right", "wrong")]
        if not m:
            return None
        return sum(r["verdict"] == "right" for r in m) / len(m)

    def site_recall(sample):
        lk = [r for r in sample if r["linkable"]]
        if not lk:
            return None
        return sum(
            r["verdict"] == "right" for r in lk
        ) / len(lk)

    # Clip-level bootstrap: resample RUNS; site-level: resample sites.
    runs = [pr["rows"] for pr in per_run]

    def clip_precision(sample_runs):
        flat = [r for rr in sample_runs for r in rr]
        return precision(flat)

    summary = {
        "n_sites": len(rows),
        "n_linkable_sites": len(linkable),
        "n_merge_events": len(merged),
        "n_wrong": len(wrong),
        "merge_precision": precision(rows),
        "merge_precision_ci95_over_sites": _boot_ci(precision, rows),
        "merge_precision_ci95_over_clips": _boot_ci(clip_precision, runs),
        "linkable_recall": site_recall(rows),
        "linkable_recall_ci95_over_sites": _boot_ci(site_recall, rows),
        "candidate_recall": (
            None
            if not linkable
            else sum(r["true_in_candidates"] for r in linkable) / len(linkable)
        ),
        "ranking_top1_given_present": (
            None
            if not any(r["true_in_candidates"] for r in linkable)
            else sum(r["true_rank"] == 1 for r in linkable if r["true_in_candidates"])
            / sum(r["true_in_candidates"] for r in linkable)
        ),
        "linkable_missed_breakdown": {
            "true_absent_from_candidates": sum(
                1 for r in linkable
                if not r["true_in_candidates"] and r["verdict"] != "right"
            ),
            "true_present_outranked": sum(
                1 for r in linkable
                if r["true_in_candidates"] and r["true_rank"] > 1
                and r["verdict"] != "right"
            ),
            "true_top1_below_bar": sum(
                1 for r in linkable
                if r["true_rank"] == 1 and r["decision"] == "abstained"
            ),
        },
        "wrong_merges": [
            {
                "run": r["run"], "tracklet_id": r["tracklet_id"],
                "margin": r["margin"], "n_candidates": r["n_candidates"],
                "true_in_candidates": r["true_in_candidates"],
                "true_rank": r["true_rank"],
                "chosen_thread_unexplained": r["chosen_thread_unexplained"],
            }
            for r in wrong
        ],
    }

    # Per-channel LLR contribution, correct vs incorrect top-1, by gap bin and
    # candidate density.
    def bin_of(gap):
        for hi, name in ((1.2, "<1.2s"), (3.0, "1.2-3s"), (7.0, "3-7s"), (15.0, "7-15s")):
            if gap < hi:
                return name
        return ">=15s"

    chan: dict[str, dict] = {}
    for r in rows:
        if not r["candidates"]:
            continue
        top = r["candidates"][0]
        if top["partner_gt"] is None or r["gt_id"] is None:
            continue
        key_ok = "true_top1" if top["is_true"] else "false_top1"
        for name, ch in top["channels"].items():
            if ch["llr"] is None:
                continue
            b = chan.setdefault(name, {}).setdefault(bin_of(top["gap_s"]), {})
            b.setdefault(key_ok, []).append(ch["contribution"])
    summary["channel_contributions"] = {
        name: {
            gb: {
                k: {
                    "n": len(v),
                    "median": float(np.median(v)),
                }
                for k, v in kinds.items()
            }
            for gb, kinds in bins.items()
        }
        for name, bins in chan.items()
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/experiments/gapsite-eval"))
    ap.add_argument("--param", action="append", default=[],
                    help="override BEST_PARAMS, e.g. --param merge_min_margin=0.0")
    args = ap.parse_args()
    params = dict(BEST_PARAMS)
    for kv in args.param:
        k, v = kv.split("=", 1)
        params[k] = json.loads(v) if v not in ("two-pass",) else v

    args.out.mkdir(parents=True, exist_ok=True)
    per_run = []
    tmp = Path("data/runs/_gapsite-tmp")
    for run_id in args.runs:
        run_dir = Path("data/runs") / run_id
        detail, assoc = replay_associate(run_dir, params, tmp)
        pr = site_rows(run_id, run_dir, detail, assoc)
        per_run.append(pr)
        (args.out / f"{run_id}.json").write_text(json.dumps(pr, indent=1))
        n_merged = sum(r["verdict"] is not None for r in pr["rows"])
        print(f"{run_id}: {len(pr['rows'])} sites, "
              f"{sum(r['linkable'] for r in pr['rows'])} linkable, "
              f"{n_merged} judged merges", flush=True)
    if tmp.exists():
        shutil.rmtree(tmp)
    summary = summarise(per_run)
    summary["params"] = params
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("wrong_merges", "channel_contributions", "params")},
                     indent=1))
    print("wrong merges:", json.dumps(summary["wrong_merges"], indent=1))


if __name__ == "__main__":
    main()
