"""Two-pass bootstrap: merge into accumulated THREADS, not into single fragments.

The multi-input harness scores a query fragment against candidate FRAGMENTS. On
FOOTPASS a fragment lasts a median 6.6 s and touches 3 of 96 pitch cells, so
"the candidate that typically occupies that space" is being asked of a smudge.
With oracle threading, representing a candidate by everything seen of it so far
is worth +12.8 rank-1 on body ID and +7.4 on occupancy.

Here that is done WITHOUT the oracle. Fragments are processed in time order; each
either joins an existing thread or starts a new one, and a thread accumulates
its members' territory, appearance and exit point as it grows. So the evidence a
later fragment is judged against is evidence the system built for itself -- which
is the whole risk: a wrong merge poisons a thread's territory and appearance for
every decision that follows, and nothing downstream can tell.

That is what the do-no-harm operating point is for, and why this reports
correct/wrong merges rather than rank-1: a ranking metric cannot see a thread
that quietly fused two players 40 minutes ago.

Calibrators and fusion weights come from OTHER matches, fitted on oracle threads
there. The distribution they are fitted on is therefore cleaner than the one
they are applied to -- noted rather than corrected, and a reason to read the
numbers as optimistic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from matchlab_core.reid.evidence import LLRCalibrator, fit_fusion_weights
from matchlab_core.reid.occupancy import js_distance
from matchlab_core.reid.threads import ThreadState
from matchlab_core.reid.transition import TransitionPrior, displacement

from matchlab_train.datasets.footpass import COL, load_half
from matchlab_train.experiments.multi_input import VAL_H5, load_appearance
from matchlab_train.experiments.position_evidence import build_fragments

FPS = 25.0
MATCHES = ("game_18", "game_24", "game_47")
CHANNELS = ("body", "occupancy", "gap")


def half_frames(key: str):
    """Fragments plus the per-fragment endpoints the transition prior needs."""
    half = load_half(VAL_H5, key)
    frags = build_fragments(half, min_frames=50, relative=True)
    rows = half.rows
    first_xy, last_xy = [], []
    for f in frags:
        sel = (rows[:, COL.PLAYER_ID] == f.player_id) & ~np.isnan(rows[:, COL.ROI_X])
        sub = rows[sel]
        sub = sub[(sub[:, COL.FRAME] >= f.start) & (sub[:, COL.FRAME] <= f.end)]
        sub = sub[np.argsort(sub[:, COL.FRAME])]
        first_xy.append(sub[0, [COL.X, COL.Y]].astype(float))
        last_xy.append(sub[-1, [COL.X, COL.Y]].astype(float))
    app = load_appearance(key) or {}
    return frags, np.array(first_xy), np.array(last_xy), app


def initial_states(frags, first_xy, last_xy, app) -> list[ThreadState]:
    out = []
    for i, f in enumerate(frags):
        e = app.get(i)
        if e is not None:
            e = np.asarray(e, dtype=np.float64)
            n = np.linalg.norm(e)
            e = e / n if n > 1e-9 else None
        out.append(
            ThreadState.from_fragment(
                f.xs, f.ys, embedding=e, start=f.start, end=f.end, exit_xy=last_xy[i]
            )
        )
    return out


def raw_row(state: ThreadState, state_fp, i: int, states, q_fp, q_proto, first_xy):
    """One row of channel raw scores for joining fragment `i` to `state`.

    Footprints and prototypes are passed in rather than recomputed: blurring a
    grid per comparison turns this into ~1.2M convolutions per sweep, which is
    what made the first attempt unrunnable.
    """
    q = states[i]
    proto = state.prototype
    qp = q_proto[i]
    body = float(proto @ qp) if (proto is not None and qp is not None) else np.nan
    dx, dy = displacement(state.exit_xy[None, :], first_xy[i][None, :])
    return [
        body,
        js_distance(q_fp[i], state_fp),
        (q.first_start - state.last_end) / FPS,
        float(dx[0]),
        float(dy[0]),
    ]


def oracle_pairs(frags, states, first_xy):
    """Scored (thread-so-far, next fragment) pairs under GT threading.

    Used only to FIT calibrators and weights, and only on matches that are not
    being evaluated.
    """
    pid = np.array([f.player_id for f in frags])
    team = np.array([f.team for f in frags])
    start = np.array([f.start for f in frags])
    q_fp = [s.footprint() for s in states]
    q_proto = [s.prototype for s in states]
    p_team = {int(p): int(team[np.flatnonzero(pid == p)[0]]) for p in np.unique(pid)}

    grown: dict[int, ThreadState] = {}
    grown_fp: dict[int, object] = {}
    rows, labels = [], []
    for i in np.argsort(start):
        for p, st in grown.items():
            if p_team[p] != team[i] or st.last_end >= start[i]:
                continue
            rows.append(raw_row(st, grown_fp[p], int(i), states, q_fp, q_proto, first_xy))
            labels.append(p == pid[i])
        key = int(pid[i])
        grown[key] = grown[key].merged_with(states[i]) if key in grown else states[i]
        grown_fp[key] = grown[key].footprint()
    return np.array(rows, dtype=float), np.array(labels, dtype=bool)


def fit_from(matches: list[str]):
    rows, labels = [], []
    for m in matches:
        for key in (f"{m}_H1", f"{m}_H2"):
            frags, first_xy, last_xy, app = half_frames(key)
            states = initial_states(frags, first_xy, last_xy, app)
            r, y = oracle_pairs(frags, states, first_xy)
            if len(r):
                rows.append(r)
                labels.append(y)
    r = np.concatenate(rows)
    y = np.concatenate(labels)
    cals = {}
    for j, name in enumerate(CHANNELS):
        col = r[:, j]
        ok = ~np.isnan(col)
        cals[name] = LLRCalibrator.fit(col[ok & y], col[ok & ~y], max_bins=200)
    prior = TransitionPrior.fit(r[:, 2], r[:, 3], r[:, 4], y)
    llr = channel_llrs(r, cals, prior)
    # Weights are fitted per-decision here rather than per-field: a greedy
    # thread assignment has no fixed candidate set to normalise over.
    w = fit_fusion_weights(llr, np.arange(len(llr)) // 64, y)
    return cals, prior, w


def channel_llrs(r: np.ndarray, cals, prior) -> np.ndarray:
    cols = []
    for j, name in enumerate(CHANNELS):
        col = r[:, j]
        cols.append(
            np.array([cals[name].llr(float(v)) if not np.isnan(v) else 0.0 for v in col])
        )
    cols.append(np.asarray(prior.llr(r[:, 2], r[:, 3], r[:, 4])))
    return np.stack(cols, axis=1)


def thread_half(key, cals, prior, w, *, min_score: float, min_margin: float) -> dict:
    frags, first_xy, last_xy, app = half_frames(key)
    states = initial_states(frags, first_xy, last_xy, app)
    pid = np.array([f.player_id for f in frags])
    team = np.array([f.team for f in frags])
    start = np.array([f.start for f in frags])

    q_fp = [s.footprint() for s in states]
    q_proto = [s.prototype for s in states]

    threads: list[ThreadState] = []
    t_fp: list[object] = []
    t_team: list[int] = []
    t_members: list[list[int]] = []
    correct = wrong = 0

    for i in np.argsort(start):
        live = [
            k
            for k in range(len(threads))
            if t_team[k] == team[i] and threads[k].last_end < start[i]
        ]
        best_k, best_s, runner = None, -np.inf, -np.inf
        if live:
            r = np.array(
                [
                    raw_row(threads[k], t_fp[k], int(i), states, q_fp, q_proto, first_xy)
                    for k in live
                ]
            )
            scores = channel_llrs(r, cals, prior) @ w
            o = np.argsort(-scores)
            best_k, best_s = live[o[0]], float(scores[o[0]])
            runner = float(scores[o[1]]) if len(o) > 1 else -np.inf

        if best_k is not None and best_s >= min_score and (best_s - runner) >= min_margin:
            majority = Counter(pid[t_members[best_k]]).most_common(1)[0][0]
            correct += majority == pid[i]
            wrong += majority != pid[i]
            threads[best_k] = threads[best_k].merged_with(states[i])
            t_fp[best_k] = threads[best_k].footprint()
            t_members[best_k].append(int(i))
        else:
            threads.append(states[i])
            t_fp.append(q_fp[int(i)])
            t_team.append(int(team[i]))
            t_members.append([int(i)])

    sizes = [len(m) for m in t_members]
    pure = sum(1 for m in t_members if len(set(pid[m])) == 1)
    return {
        "key": key,
        "fragments": len(frags),
        "true_players": int(len(set(pid.tolist()))),
        "threads": len(threads),
        "merges": correct + wrong,
        "correct": correct,
        "wrong": wrong,
        "pure_threads": pure,
        "largest_thread": max(sizes) if sizes else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument(
        "--out", type=Path, default=Path("docs/reports/2026-07-28-bootstrap-threads.json")
    )
    args = ap.parse_args()

    results = []
    for held_out in MATCHES:
        others = [m for m in MATCHES if m != held_out]
        print(f"fit on {others} -> evaluate {held_out}", flush=True)
        cals, prior, w = fit_from(others)
        print(f"  weights {dict(zip((*CHANNELS, 'transition'), w.round(3), strict=True))}",
              flush=True)
        for ms in args.min_score:
            for key in (f"{held_out}_H1", f"{held_out}_H2"):
                r = thread_half(key, cals, prior, w, min_score=ms,
                                min_margin=args.min_margin)
                r["min_score"] = ms
                results.append(r)
                print(
                    f"  {key} thr={ms:4.1f}  merges {r['merges']:5d}"
                    f"  correct {r['correct']:5d}  wrong {r['wrong']:4d}"
                    f"  threads {r['threads']:4d} (true {r['true_players']})",
                    flush=True,
                )

    print(f"\n{'threshold':>10s} {'merges':>8s} {'correct':>8s} {'wrong':>7s} {'purity':>8s}")
    for ms in args.min_score:
        rows = [r for r in results if r["min_score"] == ms]
        c = sum(r["correct"] for r in rows)
        wr = sum(r["wrong"] for r in rows)
        pt = sum(r["pure_threads"] for r in rows)
        th = sum(r["threads"] for r in rows)
        print(f"{ms:10.1f} {c + wr:8d} {c:8d} {wr:7d} {100 * pt / max(th, 1):7.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
