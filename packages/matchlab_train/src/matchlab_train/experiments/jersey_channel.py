"""Gate 3 (SPO jersey-ocr merge channel, task 5): does the jersey-OCR channel
help re-identification on real match footage, not the curated reader
benchmark?

Substrate is `gt_fragments.fragment_tracks` on SNMOT clips: fragments are pure
by construction (correct pairs are the GT track id, not inferred), and
crop degradation at each re-entry is the real thing the reader has to survive
-- unlike the SoccerNet jersey TEST split (gate 1), which is curated,
single-track crops.

Two things this module exists to guard against, both already measured on the
curated split and expected to be worse here:

  * OCR errors are correlated, not independent -- wrong reads have
    attractors (e.g. "27" absorbed 9 distinct true numbers on the reader
    benchmark), so a same-number read is not the independence-assuming
    confidence it looks like.
  * the false-veto rate is the headline risk: two same-numbered fragments
    can be read as different numbers and the channel then argues AGAINST a
    correct merge. Veto precision is reported as a function of threshold,
    never as a single number -- the question is whether a *safe* operating
    point exists, not whether the average is safe.

Everything below the reader call (`_estimate_clip_prior`, `_pairs_for_clip`,
`_veto_precision_curve`, `_reader_calibration`) is pure numpy/python and is
unit-tested without torch. `run()` is the only part that touches the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matchlab_core.artifacts import ArtifactStore
from matchlab_core.config import PipelineConfig
from matchlab_core.crops import sample_quality_crops
from matchlab_core.gt_fragments import FragmentResult, fragment_tracks
from matchlab_core.interfaces import StageContext
from matchlab_core.ocr.parseq import JerseyReader
from matchlab_core.reid.frontier import zero_wrong_frontier
from matchlab_core.reid.jersey import (
    N_NUMBERS,
    crop_number_logprobs,
    number_prior,
    pair_llr,
    tracklet_likelihood,
    uniform_prior,
)
from matchlab_core.video import probe
from pydantic import BaseModel

from matchlab_train.experiments.base import Experiment
from matchlab_train.gt_clips import discover_clips_with_gt
from matchlab_train.registry import register

OUT_DIR = Path("data/experiments/jersey-channel")
_FLAT_TOL = 1e-9  # a likelihood equal to uniform_prior() to float precision
_LLR_TOL = 1e-6  # "no evidence" threshold on the LLR itself (spec: 1e-6)


class Params(BaseModel):
    clips_dir: str = "data/videos/soccernet"
    max_clips: int = 32
    # Only used for video decode settings (sample_stride etc.) -- no pipeline
    # stage in this yaml is actually run.
    base_config: str = "configs/pipeline.baseline-observe.yaml"
    checkpoint: str = "data/weights/parseq-jersey.ckpt"
    device: str = "cuda"
    min_legibility: float = 0.9
    per_tracklet_crops: int = 100
    min_box_height_px: int = 60
    min_crop_confidence: float = 0.3
    max_isolation_iou: float = 0.15
    veto_thresholds: list[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    frontier_thresholds: list[float] = list(np.linspace(-10.0, 10.0, 81).tolist())


def _is_flat(likelihood: np.ndarray) -> bool:
    return bool(np.allclose(likelihood, 1.0 / N_NUMBERS, atol=_FLAT_TOL))


def _parse_jersey(value) -> int | None:
    """GT jersey field, restricted to a plain 0-99 numeric string/int. Returns
    None for non-numeric GT (blank, illegible-in-GT, letters) -- those
    fragments are excluded from the reader-accuracy measurement (task spec:
    "skip fragments whose GT jersey is non-numeric"), not counted as wrong."""
    if value is None:
        return None
    s = str(value).strip()
    if not s.isdigit():
        return None
    n = int(s)
    return n if 0 <= n < N_NUMBERS else None


def _pairs_for_clip(fragment_ids: list[int]) -> list[tuple[int, int]]:
    """Every unordered within-clip fragment pair, ordered (min, max) -- the
    key convention `frontier.py` and `pair_llr` scores are read against."""
    ids = sorted(fragment_ids)
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]


def _estimate_clip_prior(observed_numbers: list[int]) -> np.ndarray:
    """Per-match roster prior from THIS clip's own observed reads, not a
    global frequency table (a jersey roster is match-specific, not a league
    constant)."""
    if not observed_numbers:
        return uniform_prior()
    return number_prior(observed_numbers)


def _reader_calibration(rows: list[dict]) -> dict:
    """Per-fragment precision/coverage of the reader's argmax prediction
    against GT jersey, restricted to fragments with numeric GT (task spec).
    Same correct/wrong/abstained shape as gate 1's count table."""
    correct = wrong = abstained = 0
    for r in rows:
        if r["gt_jersey"] is None:
            continue
        if r["predicted"] is None:
            abstained += 1
        elif r["predicted"] == r["gt_jersey"]:
            correct += 1
        else:
            wrong += 1
    decided = correct + wrong
    attempted = decided + abstained
    return {
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "precision": (correct / decided) if decided else 0.0,
        "coverage": (decided / attempted) if attempted else 0.0,
    }


def _roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based (Mann-Whitney) AUC, no sklearn dependency. None if either
    class is empty (AUC undefined)."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = np.asarray(scores)[order]
    # average ranks over ties
    i = 0
    r = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        r += j - i + 1
        i = j + 1
    pos_rank_sum = sum(ranks[k] for k, y in enumerate(labels) if y)
    n_pos, n_neg = len(pos), len(neg)
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _veto_precision_curve(
    scores: dict[tuple[int, int], float],
    labels: dict[int, object],
    thresholds: list[float],
) -> list[dict]:
    """For each negative threshold t: among pairs with llr <= -t, the
    fraction that are genuinely different players (a true veto), plus the
    count that fired. A curve, not one number -- the spec is explicit that
    "is the veto safe on average" is the wrong question."""
    out = []
    for t in thresholds:
        fired = [(a, b) for (a, b), s in scores.items() if s <= -t]
        if not fired:
            out.append({"threshold": t, "n_fired": 0, "precision": None})
            continue
        n_correct_veto = sum(1 for a, b in fired if labels.get(a) != labels.get(b))
        out.append(
            {
                "threshold": t,
                "n_fired": len(fired),
                "precision": n_correct_veto / len(fired),
            }
        )
    return out


@register("jersey-channel")
class JerseyChannelExperiment(Experiment):
    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()
        base_cfg = PipelineConfig.from_yaml(p.base_config)

        clip_gt_pairs = discover_clips_with_gt(p.clips_dir, p.max_clips)
        if not clip_gt_pairs:
            raise FileNotFoundError(f"No clips with sibling GT under {p.clips_dir}")

        reader = JerseyReader(checkpoint=p.checkpoint, min_legibility=p.min_legibility)
        reader.prepare(device=p.device)

        global_scores: dict[tuple[int, int], float] = {}
        global_labels: dict[int, object] = {}
        reader_rows: list[dict] = []
        flat_pair_llrs: list[float] = []  # both-flat pairs; should be ~0
        per_clip: list[dict] = []

        for ci, (clip, gt) in enumerate(clip_gt_pairs):
            print(f"[jersey-channel] clip {ci + 1}/{len(clip_gt_pairs)}: {clip.name}", flush=True)
            frag: FragmentResult = fragment_tracks(gt)
            if not frag.tracklets:
                per_clip.append({"clip": clip.name, "n_fragments": 0, "skipped": "no fragments"})
                continue

            ctx = StageContext(
                video=probe(clip, sample_stride=base_cfg.video.sample_stride),
                config=base_cfg,
                store=ArtifactStore(workdir / "scratch" / clip.stem),
                device=p.device,
            )
            scored = sample_quality_crops(
                ctx,
                frag.tracklets,
                per_tracklet=p.per_tracklet_crops,
                min_box_height_px=p.min_box_height_px,
                min_confidence=p.min_crop_confidence,
                max_isolation_iou=p.max_isolation_iou,
            )

            likelihood_by_frag: dict[int, np.ndarray] = {}
            is_flat_by_frag: dict[int, bool] = {}
            observed_numbers: list[int] = []
            offset = ci * 1_000_000  # global fragment id, unique across clips

            for tr in frag.tracklets:
                fid = tr.tracklet_id
                crops = scored.get(fid, [])
                reads = reader.read(crops)
                if reads:
                    logprobs = np.array([crop_number_logprobs(r.char_probs) for r in reads])
                    weights = np.array([r.legibility for r in reads])
                    for lp in logprobs:
                        observed_numbers.append(int(np.argmax(lp)))
                else:
                    logprobs = np.zeros((0, N_NUMBERS))
                    weights = np.zeros((0,))
                likelihood = tracklet_likelihood(logprobs, weights)
                likelihood_by_frag[fid] = likelihood
                is_flat_by_frag[fid] = _is_flat(likelihood)

                predicted = int(np.argmax(likelihood)) if not is_flat_by_frag[fid] else None
                reader_rows.append(
                    {
                        "clip": clip.name,
                        "fragment": fid,
                        "gt_jersey": _parse_jersey(frag.jersey_by_fragment.get(fid)),
                        "predicted": predicted,
                        "n_reads": len(reads),
                    }
                )

            prior = _estimate_clip_prior(observed_numbers)

            clip_scores: dict[tuple[int, int], float] = {}
            clip_labels: dict[int, object] = {}
            for tr in frag.tracklets:
                gid = tr.tracklet_id + offset
                clip_labels[gid] = (ci, frag.gt_track_by_fragment[tr.tracklet_id])

            for a, b in _pairs_for_clip([tr.tracklet_id for tr in frag.tracklets]):
                llr = pair_llr(likelihood_by_frag[a], likelihood_by_frag[b], prior)
                clip_scores[(a + offset, b + offset)] = llr
                global_scores[(a + offset, b + offset)] = llr
                if is_flat_by_frag[a] and is_flat_by_frag[b]:
                    flat_pair_llrs.append(llr)

            global_labels.update(clip_labels)

            clip_frontier = zero_wrong_frontier(
                clip_scores, clip_labels, thresholds=p.frontier_thresholds
            )
            per_clip.append(
                {
                    "clip": clip.name,
                    "n_fragments": len(frag.tracklets),
                    "n_pairs": len(clip_scores),
                    "n_pairs_touched": sum(1 for s in clip_scores.values() if abs(s) > _LLR_TOL),
                    "frontier_correct": clip_frontier["correct"],
                    "frontier_wrong": clip_frontier["wrong"],
                }
            )

        # --- A: reader on SNMOT substrate ---
        reader_calibration = _reader_calibration(reader_rows)

        # --- B: channel alone, pairwise, pooled over ALL pairs ---
        touched = {pair: s for pair, s in global_scores.items() if abs(s) > _LLR_TOL}
        pairs_touched_fraction = len(touched) / len(global_scores) if global_scores else 0.0

        auc_scores = list(touched.values())
        auc_labels = [global_labels.get(a) == global_labels.get(b) for a, b in touched]
        roc_auc = _roc_auc(auc_scores, auc_labels)

        frontier = zero_wrong_frontier(global_scores, global_labels, thresholds=p.frontier_thresholds)
        veto_curve = _veto_precision_curve(global_scores, global_labels, p.veto_thresholds)

        # --- C: do-no-harm ---
        do_no_harm_violations = [abs(v) for v in flat_pair_llrs if abs(v) >= _LLR_TOL]

        report = {
            "n_clips": len(clip_gt_pairs),
            "n_fragments_total": len(global_labels),
            "n_pairs_total": len(global_scores),
            "reader_calibration_reference": {
                "note": "SoccerNet jersey TEST split (curated), see jersey-reader-gate",
                "precision": 0.8938,
                "coverage": 0.8692,
            },
            "reader_calibration_snmot": reader_calibration,
            "roc_auc": roc_auc,
            "pairs_touched_fraction": pairs_touched_fraction,
            "zero_wrong_frontier": {
                "correct": frontier["correct"],
                "wrong": frontier["wrong"],
                "threshold": frontier["threshold"],
            },
            "veto_precision_curve": veto_curve,
            "do_no_harm": {
                "n_both_flat_pairs": len(flat_pair_llrs),
                "n_violations": len(do_no_harm_violations),
                "max_abs_llr_when_both_flat": max(do_no_harm_violations, default=0.0),
            },
            "per_clip": per_clip,
        }

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "gate3.json").write_text(json.dumps(report, indent=2))
        self.write_result(workdir, report)
        return report
