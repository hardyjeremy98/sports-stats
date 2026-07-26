"""Phase A: does occupancy carry re-ID evidence, and is its strength pair-dependent?

Pre-registered in `docs/superpowers/specs/2026-07-27-position-evidence-reid.md` §3.

    H1  pooled AUC of the calibrated occupancy LLR separating same-player from
        different-player fragment pairs.  PASS >= 0.70, FAIL < 0.60.
    H2  mean |LLR| for different-player pairs, bucketed by role distance.
        PASS: distant-role >= 1.5x same-role AND same-role is the weakest bucket.
        FAIL: ratio < 1.2.

Registered analysis choices, fixed before any arm ran:

  * H2 is measured on SAME-TEAM impostor pairs only. Roles are per-team and the
    teams attack in opposite directions, so a cross-team "left back vs left
    back" pair occupies mirrored zones -- comparing it to a same-team pair would
    measure attacking direction, not role. Same-team impostors are also the case
    the design exists for: same-kit teammates are where appearance fails.
  * The calibrator is fitted on TRAIN halves and evaluated on VAL halves, never
    the same half.
  * `role_distance` is analysis-only. It never enters a footprint, an LLR, or a
    merge decision.

Added after first execution (recorded, not silently folded in): a broadcast half
yields ~3000 observable fragments at min_frames=25, i.e. ~4.5M candidate pairs,
so pairs are SUBSAMPLED and the distance is computed as a vectorised matrix
operation. Results are additionally stratified by fragment duration, because a
one-second fragment's "footprint" is a point, not a distribution -- if the
channel is duration-dependent, the deployed system must abstain on short
fragments rather than average that weakness into every decision.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from matchlab_core.reid.evidence import LLRCalibrator
from matchlab_core.reid.frontier import sweep
from matchlab_core.reid.occupancy import DEFAULT_GRID, Footprint, build_footprint

from matchlab_train.datasets.footpass import (
    COL,
    ROLE_NAMES,
    FootpassHalf,
    half_keys,
    load_half,
)

TACTICAL = Path("data/footpass/tactical")
TRAIN_H5 = TACTICAL / "train_tactical_data.h5"
VAL_H5 = TACTICAL / "val_tactical_data.h5"
FPS = 25.0

# Canonical role anchor positions on a normalised pitch (x along the length
# toward the opponent goal, y across the width). ANALYSIS ONLY.
ROLE_ANCHORS: dict[int, tuple[float, float]] = {
    1: (0.05, 0.50),   # GK
    2: (0.25, 0.85),   # LB
    3: (0.15, 0.65),   # LCB
    4: (0.15, 0.50),   # MCB
    5: (0.15, 0.35),   # RCB
    6: (0.50, 0.85),   # LM
    7: (0.50, 0.15),   # RM
    8: (0.35, 0.50),   # DM
    9: (0.60, 0.50),   # AM
    10: (0.80, 0.85),  # LW
    11: (0.80, 0.15),  # RW
    12: (0.90, 0.50),  # CF
    13: (0.25, 0.15),  # RB
}


@dataclass
class Fragment:
    player_id: int
    start: int
    end: int
    xs: np.ndarray
    ys: np.ndarray
    role: int
    team: int
    footprint: Footprint | None = field(default=None, repr=False)

    @property
    def duration_s(self) -> float:
        return len(self.xs) / FPS


def role_distance(role_a: int, role_b: int) -> float:
    a = ROLE_ANCHORS.get(int(role_a))
    b = ROLE_ANCHORS.get(int(role_b))
    if a is None or b is None:
        return float("nan")
    return float(math.dist(a, b))


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return min(a[1], b[1]) >= max(a[0], b[0])


def _rankdata(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(1, len(v) + 1, dtype=np.float64)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auc(same, diff) -> float:
    """P(a random different-player pair scores higher than a random same-player
    pair), ties half. Scores are LOWER-is-more-same-like (footprint distance),
    so a perfect separator gives 1.0. Mann-Whitney; no sklearn dependency."""
    s = np.asarray(list(same), dtype=np.float64)
    d = np.asarray(list(diff), dtype=np.float64)
    if not len(s) or not len(d):
        return float("nan")
    ranks = _rankdata(np.concatenate([s, d]))
    u_same = float(ranks[: len(s)].sum()) - len(s) * (len(s) + 1) / 2.0
    return float(1.0 - u_same / (len(s) * len(d)))


def relative_coords(half: FootpassHalf) -> np.ndarray:
    """(n_rows, 2) formation-relative position, NaN where not observable.

    Variant B. A player is observable only when play is near them, so absolute
    occupancy partly measures where the BALL was, not where the player's role
    is -- a selection effect, not a coordinate shift (FOOTPASS positions are
    pitch coordinates). Subtracting the team's OBSERVABLE centroid per frame
    removes the shared play-location component and leaves formation-relative
    position, which is the part that should identify a player.

    Only observable teammates enter the centroid: the deployed system cannot
    see the others either.
    """
    rows = half.rows
    out = np.full((len(rows), 2), np.nan, dtype=np.float64)
    observable = (~np.isnan(rows[:, COL.ROI_X])).astype(np.float64)
    key = rows[:, COL.FRAME].astype(np.int64) * 4 + rows[:, COL.TEAM].astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    n_groups = int(inv.max()) + 1 if len(inv) else 0
    counts = np.bincount(inv, weights=observable, minlength=n_groups)
    valid = counts > 0
    for axis, col in ((0, COL.X), (1, COL.Y)):
        vals = np.nan_to_num(rows[:, col].astype(np.float64))
        sums = np.bincount(inv, weights=vals * observable, minlength=n_groups)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=valid)
        rel = rows[:, col].astype(np.float64) - means[inv] + 0.5
        out[:, axis] = np.where((observable > 0) & valid[inv], rel, np.nan)
    return out


def build_fragments(
    half: FootpassHalf,
    *,
    max_gap_frames: int = 2,
    min_frames: int = 25,
    relative: bool = False,
) -> list[Fragment]:
    """One fragment per observable span, carrying ONLY observable positions.

    FOOTPASS knows a player's position while off-camera; the deployed system
    never will, so those rows are dropped rather than interpolated.

    `relative=True` selects variant B (formation-relative coordinates, see
    `relative_coords`) instead of absolute pitch position.
    """
    rows = half.rows
    coords = (
        relative_coords(half)
        if relative
        else np.stack([rows[:, COL.X], rows[:, COL.Y]], axis=1).astype(np.float64)
    )
    order = np.lexsort((rows[:, COL.FRAME], rows[:, COL.PLAYER_ID]))
    rows = rows[order]
    coords = coords[order]
    pids = rows[:, COL.PLAYER_ID]
    boundaries = np.flatnonzero(np.diff(pids)) + 1
    out: list[Fragment] = []
    for chunk in np.split(np.arange(len(rows)), boundaries):
        if not len(chunk):
            continue
        sub = rows[chunk]
        sub_coords = coords[chunk]
        pid = int(sub[0, COL.PLAYER_ID])
        frames = sub[:, COL.FRAME].astype(int)
        visible = ~np.isnan(sub[:, COL.ROI_X]) & ~np.isnan(sub_coords).any(axis=1)
        vis_frames = frames[visible]
        if not len(vis_frames):
            continue
        role, team = int(sub[0, COL.ROLE]), int(sub[0, COL.TEAM])
        # split visible frames into spans, bridging gaps <= max_gap_frames
        breaks = np.flatnonzero(np.diff(vis_frames) > max_gap_frames + 1) + 1
        for span_idx in np.split(np.arange(len(vis_frames)), breaks):
            if len(span_idx) < min_frames:
                continue
            start, end = int(vis_frames[span_idx[0]]), int(vis_frames[span_idx[-1]])
            sel = visible & (frames >= start) & (frames <= end)
            out.append(
                Fragment(
                    player_id=pid,
                    start=start,
                    end=end,
                    xs=sub_coords[sel, 0],
                    ys=sub_coords[sel, 1],
                    role=role,
                    team=team,
                )
            )
    return out


def footprint_matrix(
    frags: list[Fragment], *, grid: tuple[int, int] = DEFAULT_GRID, sigma: float = 1.0
) -> np.ndarray:
    """(N, gx*gy) row-stochastic matrix of footprints, in fragment order."""
    mats = []
    for f in frags:
        if f.footprint is None:
            f.footprint = build_footprint(f.xs, f.ys, grid=grid, sigma=sigma)
        mats.append(f.footprint.grid.ravel())
    return np.asarray(mats, dtype=np.float64) if mats else np.zeros((0, grid[0] * grid[1]))


def pairwise_js(mat: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray, *, chunk: int = 200_000):
    """Vectorised Jensen-Shannon distance for selected index pairs.

    Same value as `occupancy.js_distance`, computed in blocks so that millions
    of candidate pairs per half stay tractable.
    """
    out = np.empty(len(i_idx), dtype=np.float64)
    for s in range(0, len(i_idx), chunk):
        e = min(s + chunk, len(i_idx))
        p = mat[i_idx[s:e]]
        q = mat[j_idx[s:e]]
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = np.where(p > 0, p * np.log2(p / np.maximum(m, 1e-12)), 0.0)
            tq = np.where(q > 0, q * np.log2(q / np.maximum(m, 1e-12)), 0.0)
        div = 0.5 * tp.sum(axis=1) + 0.5 * tq.sum(axis=1)
        out[s:e] = np.sqrt(np.clip(div, 0.0, 1.0))
    return out


def sample_pairs(
    frags: list[Fragment],
    rng: np.random.Generator,
    *,
    max_same: int = 40_000,
    max_diff: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping (same-player, different-player) index pairs.

    Temporally overlapping fragments are excluded: they can never be merge
    candidates, so including them would pad the negative set with pairs the
    engine never scores (mirrors TemporalOverlapGate).
    """
    n = len(frags)
    pid = np.array([f.player_id for f in frags])
    start = np.array([f.start for f in frags])
    end = np.array([f.end for f in frags])

    def keep(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return np.minimum(end[i], end[j]) < np.maximum(start[i], start[j])

    same_i, same_j = [], []
    for p in np.unique(pid):
        idx = np.flatnonzero(pid == p)
        if len(idx) < 2:
            continue
        ii, jj = zip(*combinations(idx, 2), strict=True)
        same_i.extend(ii)
        same_j.extend(jj)
    si = np.asarray(same_i, dtype=int)
    sj = np.asarray(same_j, dtype=int)
    if len(si):
        m = keep(si, sj)
        si, sj = si[m], sj[m]
        if len(si) > max_same:
            pick = rng.choice(len(si), max_same, replace=False)
            si, sj = si[pick], sj[pick]

    # Different-player pairs: rejection-sample rather than enumerate ~N^2/2.
    di, dj = [], []
    target = max_diff
    tries = 0
    while sum(len(x) for x in di) < target and tries < 40 and n > 1:
        tries += 1
        a = rng.integers(0, n, size=target)
        b = rng.integers(0, n, size=target)
        m = (pid[a] != pid[b]) & keep(a, b)
        di.append(a[m])
        dj.append(b[m])
    dii = np.concatenate(di)[:target] if di else np.zeros(0, dtype=int)
    djj = np.concatenate(dj)[:target] if dj else np.zeros(0, dtype=int)
    lo = np.minimum(dii, djj)
    hi = np.maximum(dii, djj)
    uniq = np.unique(np.stack([lo, hi], axis=1), axis=0) if len(lo) else np.zeros((0, 2), int)
    return np.stack([si, sj], axis=1) if len(si) else np.zeros((0, 2), int), uniq


def _half_scores(path: Path, key: str, rng, *, min_frames: int, max_diff: int, relative: bool):
    frags = build_fragments(load_half(path, key), min_frames=min_frames, relative=relative)
    if len(frags) < 2:
        return frags, np.zeros((0, 2), int), np.zeros(0), np.zeros((0, 2), int), np.zeros(0)
    mat = footprint_matrix(frags)
    same, diff = sample_pairs(frags, rng, max_diff=max_diff)
    ds = pairwise_js(mat, same[:, 0], same[:, 1]) if len(same) else np.zeros(0)
    dd = pairwise_js(mat, diff[:, 0], diff[:, 1]) if len(diff) else np.zeros(0)
    return frags, same, ds, diff, dd


def phase_a(
    *,
    train_path: Path = TRAIN_H5,
    val_path: Path = VAL_H5,
    n_train_halves: int = 4,
    min_frames: int = 50,
    max_diff: int = 200_000,
    relative: bool = False,
    max_bins: int = 200,
) -> dict:
    rng = np.random.default_rng(0)
    train_keys = half_keys(train_path)[:n_train_halves]
    val_keys = half_keys(val_path)

    fit_same, fit_diff = [], []
    for key in train_keys:
        _, _, ds, _, dd = _half_scores(
            train_path, key, rng, min_frames=min_frames, max_diff=max_diff, relative=relative
        )
        fit_same.append(ds)
        fit_diff.append(dd)
    cal = LLRCalibrator.fit(
        np.concatenate(fit_same), np.concatenate(fit_diff), max_bins=max_bins
    )

    val_same_d, val_diff_d = [], []
    dur_same, dur_diff = [], []
    role_rows: list[tuple[float, int, int]] = []  # (distance, role_a, role_b) same-team only
    per_half = {}
    for key in val_keys:
        frags, same, ds, diff, dd = _half_scores(
            val_path, key, rng, min_frames=min_frames, max_diff=max_diff, relative=relative
        )
        per_half[key] = {
            "fragments": len(frags),
            "same_pairs": int(len(ds)),
            "diff_pairs": int(len(dd)),
        }
        val_same_d.append(ds)
        val_diff_d.append(dd)
        if len(same):
            dur_same.append(
                np.minimum(
                    [frags[i].duration_s for i in same[:, 0]],
                    [frags[j].duration_s for j in same[:, 1]],
                )
            )
        if len(diff):
            dur_diff.append(
                np.minimum(
                    [frags[i].duration_s for i in diff[:, 0]],
                    [frags[j].duration_s for j in diff[:, 1]],
                )
            )
            for (i, j), d in zip(diff, dd, strict=True):
                if frags[i].team == frags[j].team:
                    role_rows.append((float(d), frags[i].role, frags[j].role))

    same_d = np.concatenate(val_same_d)
    diff_d = np.concatenate(val_diff_d)
    dur_s = np.concatenate(dur_same) if dur_same else np.zeros(0)
    dur_d = np.concatenate(dur_diff) if dur_diff else np.zeros(0)

    same_llr = np.array([-cal.llr(d) for d in same_d])
    diff_llr = np.array([-cal.llr(d) for d in diff_d])
    auc_llr = auc(same_llr, diff_llr)
    auc_raw = auc(same_d, diff_d)

    # Stratify by the shorter fragment's duration: a one-second footprint is a
    # point, and the deployed channel must know when to abstain.
    strata = {}
    for lo, hi, name in ((0, 4, "<4s"), (4, 10, "4-10s"), (10, 30, "10-30s"), (30, 1e9, ">30s")):
        ms = (dur_s >= lo) & (dur_s < hi)
        md = (dur_d >= lo) & (dur_d < hi)
        strata[name] = {
            "n_same": int(ms.sum()),
            "n_diff": int(md.sum()),
            "auc": auc(same_d[ms], diff_d[md]) if ms.sum() and md.sum() else float("nan"),
        }

    buckets: dict[str, list[float]] = {"same_role": [], "near_role": [], "distant_role": []}
    role_pair: dict[str, list[float]] = {}
    for d, ra, rb in role_rows:
        rd = role_distance(ra, rb)
        if math.isnan(rd):
            continue
        v = abs(cal.llr(d))
        k = "same_role" if rd == 0.0 else ("near_role" if rd <= 0.3 else "distant_role")
        buckets[k].append(v)
        names = sorted([ROLE_NAMES.get(ra, "?"), ROLE_NAMES.get(rb, "?")])
        role_pair.setdefault("/".join(names), []).append(v)

    means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in buckets.items()}
    counts = {k: len(v) for k, v in buckets.items()}
    ratio = (
        means["distant_role"] / means["same_role"]
        if means["same_role"] and means["same_role"] > 0
        else float("nan")
    )
    finite = [v for v in means.values() if not math.isnan(v)]
    same_is_weakest = bool(finite) and means["same_role"] == min(finite)

    return {
        "train_keys": train_keys,
        "val_keys": val_keys,
        "min_frames": min_frames,
        "variant": "B-formation-relative" if relative else "A-absolute",
        "per_half": per_half,
        "n_same_pairs": int(len(same_d)),
        "n_diff_pairs": int(len(diff_d)),
        "auc_llr": auc_llr,
        "auc_raw_distance": auc_raw,
        "h1_pass": bool(auc_llr >= 0.70),
        "h1_fail": bool(auc_llr < 0.60),
        "auc_by_duration": strata,
        "role_bucket_mean_abs_llr": means,
        "role_bucket_counts": counts,
        "distant_over_same_ratio": ratio,
        "same_role_is_weakest": same_is_weakest,
        "h2_pass": bool(ratio >= 1.5 and same_is_weakest),
        "h2_fail": bool(ratio < 1.2),
        "role_pair_mean_abs_llr": dict(
            sorted(
                ((k, float(np.mean(v))) for k, v in role_pair.items() if len(v) >= 100),
                key=lambda kv: kv[1],
            )
        ),
        "calibrator": cal.to_dict(),
    }


def merge_frontier(
    *,
    val_path: Path = VAL_H5,
    train_path: Path = TRAIN_H5,
    n_train_halves: int = 4,
    min_frames: int = 50,
    max_diff: int = 200_000,
    relative: bool = True,
    max_bins: int = 200,
) -> dict:
    """How many merges can the position channel repair with ZERO wrong merges?

    The number that matters, and the one directly comparable to appearance's
    13-14 of 125 on the SoccerNet GT-fragment harness (SPO-85). Scored per half,
    since do-no-harm is a per-sequence standard.
    """
    rng = np.random.default_rng(0)
    fit_same, fit_diff = [], []
    for key in half_keys(train_path)[:n_train_halves]:
        _, _, ds, _, dd = _half_scores(
            train_path, key, rng, min_frames=min_frames, max_diff=max_diff, relative=relative
        )
        fit_same.append(ds)
        fit_diff.append(dd)
    cal = LLRCalibrator.fit(
        np.concatenate(fit_same), np.concatenate(fit_diff), max_bins=max_bins
    )

    per_half = {}
    for key in half_keys(val_path):
        frags, same, ds, diff, dd = _half_scores(
            val_path, key, rng, min_frames=min_frames, max_diff=max_diff, relative=relative
        )
        labels = {i: f.player_id for i, f in enumerate(frags)}
        scores: dict[tuple[int, int], float] = {}
        for (i, j), d in zip(same, ds, strict=True):
            scores[(int(i), int(j))] = cal.llr(float(d))
        for (i, j), d in zip(diff, dd, strict=True):
            scores[(int(i), int(j))] = cal.llr(float(d))
        # merges needed = one per extra fragment of each fragmented player
        counts: dict[int, int] = {}
        for f in frags:
            counts[f.player_id] = counts.get(f.player_id, 0) + 1
        needed = sum(c - 1 for c in counts.values() if c > 1)
        sw = sweep(scores, labels)
        curve = sw.curve()
        # Best correct-count at each wrong-merge budget: one pass down the
        # ranked admission list, so the whole curve costs what one point did.
        best_at: dict[int, tuple[int, int]] = {}
        for budget in (0, 1, 5, 10, 25, 50):
            best_c, best_w = 0, 0
            for _s, c, w in curve:
                if w <= budget and c > best_c:
                    best_c, best_w = c, w
            best_at[budget] = (best_c, best_w)
        loose_c, loose_w = (curve[-1][1], curve[-1][2]) if curve else (0, 0)
        per_half[key] = {
            "fragments": len(frags),
            "merges_needed": needed,
            "admitted_pairs": len(curve),
            "frontier_correct": best_at[0][0],
            "frontier_wrong": best_at[0][1],
            "loose_correct": loose_c,
            "loose_wrong": loose_w,
            "budget_curve": {
                str(b): {"correct": c, "wrong": w} for b, (c, w) in best_at.items()
            },
        }
    tot = {
        k: sum(v[k] for v in per_half.values())
        for k in ("merges_needed", "frontier_correct", "frontier_wrong",
                  "loose_correct", "loose_wrong")
    }
    tot["budget_curve"] = {
        str(b): {
            "correct": sum(v["budget_curve"][str(b)]["correct"] for v in per_half.values()),
            "wrong": sum(v["budget_curve"][str(b)]["wrong"] for v in per_half.values()),
        }
        for b in (0, 1, 5, 10, 25, 50)
    }
    return {"variant": "B-formation-relative" if relative else "A-absolute",
            "min_frames": min_frames, "per_half": per_half, "total": tot}


def _verdict(pass_: bool, fail: bool) -> str:
    return "PASS" if pass_ else ("FAIL" if fail else "INCONCLUSIVE")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A occupancy-LLR falsification test")
    ap.add_argument("--min-frames", type=int, default=50)
    ap.add_argument("--train-halves", type=int, default=4)
    ap.add_argument("--max-diff", type=int, default=200_000)
    ap.add_argument("--relative", action="store_true", help="variant B: formation-relative")
    ap.add_argument("--frontier", action="store_true", help="merge-frontier accounting")
    ap.add_argument("--out", type=Path, default=Path("docs/reports/2026-07-27-phase-a-raw.json"))
    args = ap.parse_args()

    if args.frontier:
        fr = merge_frontier(min_frames=args.min_frames, relative=args.relative,
                            n_train_halves=args.train_halves, max_diff=args.max_diff)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fr, indent=2))
        print(f"variant={fr['variant']} min_frames={fr['min_frames']}")
        for k, v in fr['per_half'].items():
            print(f"  {k}: needed={v['merges_needed']} admitted={v['admitted_pairs']}"
                  f" frontier={v['frontier_correct']}"
                  f" | loose {v['loose_correct']}c/{v['loose_wrong']}w")
        print(f"TOTAL needed={fr['total']['merges_needed']} "
              f"frontier={fr['total']['frontier_correct']} "
              f"loose={fr['total']['loose_correct']}c/{fr['total']['loose_wrong']}w")
        print("  budget curve (per-half wrong-merge budget -> pooled correct):")
        for b, v in fr['total']['budget_curve'].items():
            print(f"    <={b:>2} wrong/half: {v['correct']:>4} correct, {v['wrong']:>4} wrong")
        print(f"written: {args.out}")
        return

    r = phase_a(
        n_train_halves=args.train_halves,
        min_frames=args.min_frames,
        max_diff=args.max_diff,
        relative=args.relative,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2))

    print(f"variant={r['variant']} min_frames={r['min_frames']}  pairs: same={r['n_same_pairs']} diff={r['n_diff_pairs']}")
    print(
        f"H1 AUC(calibrated LLR)={r['auc_llr']:.4f}  [raw distance {r['auc_raw_distance']:.4f}]"
        f"  -> {_verdict(r['h1_pass'], r['h1_fail'])}"
    )
    for k, v in r["auc_by_duration"].items():
        print(f"    {k:>7}: AUC {v['auc']:.4f}  (same {v['n_same']}, diff {v['n_diff']})")
    print(f"H2 mean|LLR| by role bucket: {r['role_bucket_mean_abs_llr']}")
    print(f"    counts: {r['role_bucket_counts']}")
    print(
        f"    distant/same ratio={r['distant_over_same_ratio']:.3f}"
        f"  same_role_weakest={r['same_role_is_weakest']}"
        f"  -> {_verdict(r['h2_pass'], r['h2_fail'])}"
    )
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
