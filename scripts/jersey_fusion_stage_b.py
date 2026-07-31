"""Stage B driver (task 7): three-arm offline comparison -- body alone,
jersey alone, fused (body + jersey) -- on the 868-fragment SNMOT substrate.

Reads:
  * data/experiments/jersey-fusion/features/<stem>/frame_features.npz (stage A)
  * /tmp/jersey_evidence_cache.pkl (868-fragment per-crop jersey evidence cache)

Writes:
  data/experiments/jersey-fusion/report.json
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from matchlab_core.frame_features import FrameFeatures
from matchlab_core.reid.evidence import LLRCalibrator, fit_fusion_weights
from matchlab_core.reid.frontier import merge_counts
from matchlab_core.reid.jersey import (
    crop_number_logprobs,
    number_prior,
    pair_llr,
    tracklet_likelihood,
    uniform_prior,
)
from matchlab_core.reid.jersey_fusion import (
    assert_do_no_harm,
    fuse_sum,
    fuse_weighted,
    pooled_pairs,
    veto_impact,
)
from matchlab_core.reid.representation import build_representations, pair_similarity

CACHE_PATH = Path("/tmp/jersey_evidence_cache.pkl")
FEATURES_DIR = Path("data/experiments/jersey-fusion/features")
OUT_PATH = Path("data/experiments/jersey-fusion/report.json")


def load_cache() -> list[dict]:
    with CACHE_PATH.open("rb") as f:
        return pickle.load(f)


def load_all_features() -> dict[str, FrameFeatures]:
    out = {}
    for d in sorted(FEATURES_DIR.iterdir()):
        npz = d / "frame_features.npz"
        if npz.exists():
            out[d.name] = FrameFeatures.load(npz)
    return out


def verify_alignment(cache_rows: list[dict], feats: dict[str, FrameFeatures]) -> None:
    from matchlab_core.reid.jersey_fusion import verify_fragment_alignment

    meta_by_clip = {stem: ff.meta for stem, ff in feats.items()}
    verify_fragment_alignment(cache_rows, meta_by_clip)
    print(f"[stage-b] alignment OK: {len(cache_rows)} cache rows all match npz gt_track_by_fragment")


def compute_jersey_likelihoods(cache_rows: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    """One likelihood-over-100 array per (clip_stem, frag), via the SHIPPED
    rule: weights = clip(band_leg,0,1)*clip(confs,1e-6,1); Sigma-w mean
    posterior with margin_tau=2.0."""
    out = {}
    for row in cache_rows:
        stem = row["clip"].removesuffix(".mp4")
        probs = row["probs"]  # (n_crops, 3, 11)
        logprobs = np.array([crop_number_logprobs(p) for p in probs])
        band = np.clip(row["band_leg"], 0.0, 1.0)
        confs = np.clip(row["confs"], 1e-6, 1.0)
        weights = band * confs
        lik = tracklet_likelihood(logprobs, weights, margin_tau=2.0)
        out[(stem, row["frag"])] = lik
    return out


def estimate_clip_priors(
    cache_rows: list[dict], likelihoods: dict[tuple[str, int], np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-clip roster prior from that clip's own non-abstained argmax reads."""
    flat = uniform_prior()
    by_clip: dict[str, list[int]] = defaultdict(list)
    for row in cache_rows:
        stem = row["clip"].removesuffix(".mp4")
        lik = likelihoods[(stem, row["frag"])]
        if not np.allclose(lik, flat, atol=1e-9):
            by_clip[stem].append(int(np.argmax(lik)))
    return {stem: (number_prior(nums) if nums else flat) for stem, nums in by_clip.items()}


def main() -> None:
    cache_rows = load_cache()
    feats = load_all_features()
    print(f"[stage-b] loaded {len(cache_rows)} cache rows, {len(feats)} clips of features")
    verify_alignment(cache_rows, feats)

    # --- representations + body pairwise cosine ---
    reps_by_clip = {stem: build_representations(ff, max_prototypes=4) for stem, ff in feats.items()}

    keys = [(row["clip"].removesuffix(".mp4"), row["frag"]) for row in cache_rows]
    labels = {(row["clip"].removesuffix(".mp4"), row["frag"]): row["gt_track"] for row in cache_rows}
    clip_stems = sorted({k[0] for k in keys})
    n_tune = 16
    tune_clips = set(clip_stems[:n_tune])
    held_clips = set(clip_stems[n_tune:])
    print(f"[stage-b] {len(tune_clips)} tune clips, {len(held_clips)} held clips")

    pairs = pooled_pairs(keys, labels)
    print(f"[stage-b] {len(pairs)} within-clip pairs total")

    body_cosine: dict[tuple, float] = {}
    for a, b in pairs:
        stem = a[0]
        reps = reps_by_clip.get(stem, {})
        ra, rb = reps.get(a[1]), reps.get(b[1])
        if ra is None or rb is None:
            continue
        s = pair_similarity(ra, rb)
        if s is not None:
            body_cosine[(a, b)] = s

    # --- calibrate body LLRs on TUNING half only ---
    tune_same = [s for (a, b), s in body_cosine.items() if a[0] in tune_clips and labels[a] == labels[b]]
    tune_diff = [s for (a, b), s in body_cosine.items() if a[0] in tune_clips and labels[a] != labels[b]]
    print(f"[stage-b] body calibration: {len(tune_same)} same / {len(tune_diff)} diff (tune half)")
    calibrator = LLRCalibrator.fit(tune_same, tune_diff)
    body_llr = {k: calibrator.llr(s) for k, s in body_cosine.items()}

    # --- jersey LLRs ---
    likelihoods = compute_jersey_likelihoods(cache_rows)
    priors = estimate_clip_priors(cache_rows, likelihoods)
    jersey_llr: dict[tuple, float] = {}
    for a, b in pairs:
        stem = a[0]
        la, lb = likelihoods.get(a), likelihoods.get(b)
        if la is None or lb is None:
            continue
        prior = priors.get(stem, uniform_prior())
        jersey_llr[(a, b)] = pair_llr(la, lb, prior)

    # --- fused (unweighted sum) ---
    fused_sum = fuse_sum(body_llr, jersey_llr)

    # --- fitted-weight fusion, tuning half only ---
    # Episode = one anchor fragment's field of same-clip candidates (every
    # other fragment in that clip), feature = [body_llr, jersey_llr] for the
    # (anchor, candidate) pair, correct = candidate shares the anchor's
    # gt_track. This matches fit_fusion_weights's docstring: "the listwise
    # probability that the field's softmax puts on SOME correct candidate."
    candidates_by_anchor: dict[tuple, list[tuple]] = defaultdict(list)
    for a, b in pairs:
        if a[0] not in tune_clips:
            continue
        candidates_by_anchor[a].append(b)
        candidates_by_anchor[b].append(a)

    ep_index, feat_rows, is_correct = [], [], []
    for ep, (anchor, cands) in enumerate(candidates_by_anchor.items()):
        for cand in cands:
            key = (anchor, cand) if anchor < cand else (cand, anchor)
            feat_rows.append([body_llr.get(key, 0.0), jersey_llr.get(key, 0.0)])
            is_correct.append(labels[anchor] == labels[cand])
            ep_index.append(ep)
    fit_weights = fit_fusion_weights(np.array(feat_rows), np.array(ep_index), np.array(is_correct))
    print(f"[stage-b] fitted fusion weights (body, jersey): {fit_weights.tolist()}")
    fused_weighted = fuse_weighted(body_llr, jersey_llr, fit_weights)

    # --- do-no-harm ---
    checked_sum = assert_do_no_harm(body_llr, jersey_llr, fused_sum)
    checked_w = assert_do_no_harm(
        body_llr, jersey_llr, fused_weighted, body_scale=float(fit_weights[0])
    )
    print(f"[stage-b] do-no-harm OK on {checked_sum} (sum) / {checked_w} (weighted) zero-jersey pairs")

    # --- held-out metrics ---
    def held(d: dict) -> dict:
        return {(a, b): v for (a, b), v in d.items() if a[0] in held_clips}

    held_labels = {k: v for k, v in labels.items() if k[0] in held_clips}
    arms = {
        "body_only": held(body_llr),
        "jersey_only": held(jersey_llr),
        "fused_sum": held(fused_sum),
        "fused_weighted": held(fused_weighted),
    }

    def roc_auc(scores: dict) -> float | None:
        items = list(scores.items())
        y = [held_labels.get(a) == held_labels.get(b) for (a, b), _ in items]
        s = [v for _, v in items]
        pos = [v for v, yy in zip(s, y, strict=True) if yy]
        neg = [v for v, yy in zip(s, y, strict=True) if not yy]
        if not pos or not neg:
            return None
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s))
        sorted_s = np.asarray(s)[order]
        i, r = 0, 1
        while i < len(sorted_s):
            j = i
            while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            ranks[order[i : j + 1]] = (r + (r + (j - i))) / 2.0
            r += j - i + 1
            i = j + 1
        pos_rank_sum = sum(ranks[k] for k, yy in enumerate(y) if yy)
        return float((pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

    def threshold_table(scores: dict, n: int = 20) -> list[dict]:
        vals = list(scores.values())
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if lo == hi:
            thresholds = [lo]
        else:
            thresholds = list(np.linspace(lo, hi, n))
        return [
            {
                "threshold": float(t),
                **{
                    k: v
                    for k, v in merge_counts(scores, held_labels, threshold=t).items()
                    if k in ("correct", "wrong")
                },
            }
            for t in thresholds
        ]

    def matched_budget(scores: dict, budgets: list[int]) -> dict:
        table = threshold_table(scores, n=200)
        out = {}
        for b in budgets:
            afford = [r for r in table if r["wrong"] <= b]
            out[str(b)] = max(afford, key=lambda r: r["correct"]) if afford else {"correct": 0, "wrong": 0}
        return out

    report: dict = {"n_tune_clips": len(tune_clips), "n_held_clips": len(held_clips)}
    for name, scores in arms.items():
        report[name] = {
            "n_pairs": len(scores),
            "roc_auc": roc_auc(scores),
            "threshold_table": threshold_table(scores),
            "matched_wrong_budget": matched_budget(scores, [0, 5, 10, 20]),
        }

    body_scores_held = arms["body_only"]
    best_body_t = None
    best_body_correct = -1
    for row in report["body_only"]["threshold_table"]:
        if row["wrong"] == 0 and row["correct"] > best_body_correct:
            best_body_correct = row["correct"]
            best_body_t = row["threshold"]
    if best_body_t is None and report["body_only"]["threshold_table"]:
        best_body_t = min(r["threshold"] for r in report["body_only"]["threshold_table"])

    report["veto_impact_body_to_fused"] = veto_impact(
        body_scores_held, held(fused_sum), held_labels, body_threshold=best_body_t or 0.0
    )
    report["body_best_threshold_used_for_veto"] = best_body_t

    report["do_no_harm"] = {
        "checked_pairs_sum_arm": checked_sum,
        "checked_pairs_weighted_arm": checked_w,
    }
    report["fitted_fusion_weights"] = {"body": float(fit_weights[0]), "jersey": float(fit_weights[1])}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[stage-b] wrote {OUT_PATH}")

    print("=== HEADLINE (held-out, ROC-AUC) ===")
    for name in arms:
        print(f"  {name}: AUC={report[name]['roc_auc']}")


if __name__ == "__main__":
    main()
