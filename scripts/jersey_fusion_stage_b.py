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
from matchlab_core.reid.frontier import sweep as frontier_sweep
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

    # --- held-out slices (do-no-harm and every downstream metric are scored
    # on held-out ONLY -- the tuning half never contributes to a reported
    # number) ---
    def held(d: dict) -> dict:
        return {(a, b): v for (a, b), v in d.items() if a[0] in held_clips}

    held_labels = {k: v for k, v in labels.items() if k[0] in held_clips}
    held_body_llr = held(body_llr)
    held_jersey_llr = held(jersey_llr)
    held_fused_sum = held(fused_sum)
    held_fused_weighted = held(fused_weighted)

    # --- do-no-harm, held-out pairs only ---
    checked_sum = assert_do_no_harm(held_body_llr, held_jersey_llr, held_fused_sum)
    checked_w = assert_do_no_harm(
        held_body_llr, held_jersey_llr, held_fused_weighted, body_scale=float(fit_weights[0])
    )
    print(
        f"[stage-b] do-no-harm OK on {checked_sum} (sum) / {checked_w} (weighted) "
        "held-out zero-jersey pairs"
    )

    arms = {
        "body_only": held_body_llr,
        "jersey_only": held_jersey_llr,
        "fused_sum": held_fused_sum,
        "fused_weighted": held_fused_weighted,
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

    def exact_curve(scores: dict) -> list[tuple[float, int, int]]:
        """Exact prefix scan (`frontier.sweep(...).curve()`): the (score,
        cumulative-correct, cumulative-wrong) triple at EVERY distinct
        admitted score, not an interpolated grid. A linear grid of 20-200
        points can straddle the exact point where `correct` increments --
        measured to understate jersey-only's zero-wrong yield 14x (1 vs the
        true 14) and fused's by 1 (49 vs the true 50); this is the fix."""
        sw = frontier_sweep(scores, held_labels)
        return sw.curve()

    def exact_budget(curve: list[tuple[float, int, int]], budgets: list[int]) -> dict:
        out = {}
        for b in budgets:
            best = {"correct": 0, "wrong": 0, "threshold": None}
            for score, correct, wrong in curve:
                if wrong <= b and correct > best["correct"]:
                    best = {"correct": correct, "wrong": wrong, "threshold": float(score)}
            out[str(b)] = best
        return out

    def downsampled_table(curve: list[tuple[float, int, int]], n: int = 20) -> list[dict]:
        """Display-only: n evenly-spaced EXACT points from the real curve
        (not interpolated thresholds) so the report stays readable."""
        if not curve:
            return []
        idxs = sorted(set(np.linspace(0, len(curve) - 1, min(n, len(curve))).astype(int).tolist()))
        return [
            {"threshold": float(curve[i][0]), "correct": curve[i][1], "wrong": curve[i][2]}
            for i in idxs
        ]

    report: dict = {"n_tune_clips": len(tune_clips), "n_held_clips": len(held_clips)}
    curves = {}
    for name, scores in arms.items():
        curve = exact_curve(scores)
        curves[name] = curve
        report[name] = {
            "n_pairs": len(scores),
            "roc_auc": roc_auc(scores),
            "threshold_table": downsampled_table(curve),
            "matched_wrong_budget": exact_budget(curve, [0, 5, 10, 20]),
        }

    # --- veto impact: body's OWN exact zero-wrong frontier (falls back to
    # its most permissive admitted score if body has no zero-wrong point) ---
    body_curve = curves["body_only"]
    zero_wrong = [c for c in body_curve if c[2] == 0]
    if zero_wrong:
        best_body_t = float(max(zero_wrong, key=lambda c: c[1])[0])
    else:
        best_body_t = float(min(c[0] for c in body_curve)) if body_curve else 0.0

    report["veto_impact_body_to_fused"] = veto_impact(
        arms["body_only"], arms["fused_sum"], held_labels, body_threshold=best_body_t
    )
    report["body_best_threshold_used_for_veto"] = best_body_t
    report["body_has_zero_wrong_point"] = bool(zero_wrong)

    # --- per-clip spread of fused's zero-wrong merges: which clips actually
    # contribute correct merges at the winning threshold, not just a pooled
    # count (a headline concentrated in 1-2 clips would be a weaker result) ---
    fused_zero_wrong_threshold = report["fused_sum"]["matched_wrong_budget"]["0"]["threshold"]
    per_clip_merges: dict[str, int] = defaultdict(int)
    if fused_zero_wrong_threshold is not None:
        mc = merge_counts(arms["fused_sum"], held_labels, threshold=fused_zero_wrong_threshold)
        for a, b in mc["merged"]:
            if held_labels.get(a) == held_labels.get(b):
                per_clip_merges[a[0]] += 1
    report["fused_sum_zero_wrong_per_clip"] = dict(sorted(per_clip_merges.items()))
    report["fused_sum_zero_wrong_clips_represented"] = f"{len(per_clip_merges)}/{len(held_clips)}"
    report["fused_sum_zero_wrong_max_in_one_clip"] = max(per_clip_merges.values(), default=0)

    report["do_no_harm"] = {
        "note": "checked on HELD-OUT pairs only (16 held clips), not the full 32-clip pool",
        "checked_pairs_sum_arm": checked_sum,
        "checked_pairs_weighted_arm": checked_w,
    }
    report["fitted_fusion_weights"] = {"body": float(fit_weights[0]), "jersey": float(fit_weights[1])}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[stage-b] wrote {OUT_PATH}")

    print("=== HEADLINE (held-out, ROC-AUC and exact zero-wrong yield) ===")
    for name in arms:
        zw = report[name]["matched_wrong_budget"]["0"]["correct"]
        print(f"  {name}: AUC={report[name]['roc_auc']}, correct@wrong=0: {zw}")
    print(
        f"[stage-b] fused_sum zero-wrong spread: {report['fused_sum_zero_wrong_clips_represented']} "
        f"clips, max {report['fused_sum_zero_wrong_max_in_one_clip']} in one clip"
    )


if __name__ == "__main__":
    main()
