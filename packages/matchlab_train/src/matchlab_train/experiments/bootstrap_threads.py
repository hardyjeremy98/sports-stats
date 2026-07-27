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
import copy
import json
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Corruption:
    """How far to degrade GT fragments toward real tracker output.

    `contaminate` -- fraction of fragments that get another player spliced over
    their tail, which is what an ID switch inside a tracklet looks like. The
    fragment keeps the MAJORITY player's label, so the corruption shows up as
    damage rather than as a relabelled-and-therefore-correct fragment.

    `oversegment` -- probability a fragment is split in two. Real trackers break
    identity more often than an observability span does, and shorter fragments
    carry less of everything.

    Splices are same-team only: a cross-team splice would be caught by the team
    gate and so would understate the damage, and real ID switches are
    overwhelmingly within a team anyway.
    """

    contaminate: float = 0.0
    oversegment: float = 0.0
    tail: float = 0.4
    seed: int = 0


def corrupt_fragments(frags, embeddings: dict, c: Corruption):
    """Return (fragments, embeddings) degraded per `c`. Pure; inputs untouched."""
    rng = np.random.default_rng(c.seed)
    out = [copy.copy(f) for f in frags]
    emb = {k: np.array(v, dtype=float) for k, v in embeddings.items()}
    # Endpoints live in ABSOLUTE pitch coordinates while xs/ys are
    # formation-relative, so they cannot be recovered from the corrupted
    # positions. `origin` and `donor` let the caller remap them instead.
    for i, f in enumerate(out):
        f.origin, f.donor = i, -1

    if c.contaminate > 0:
        by_team: dict[int, list[int]] = {}
        for i, f in enumerate(out):
            by_team.setdefault(int(f.team), []).append(i)
        for i, f in enumerate(out):
            pool = [j for j in by_team.get(int(f.team), []) if out[j].player_id != f.player_id]
            if not pool or rng.random() >= c.contaminate:
                continue
            j = int(rng.choice(pool))
            other = out[j]
            n = len(f.xs)
            take = max(1, min(int(round(n * c.tail)), n - 1, len(other.xs)))
            src = int(rng.integers(0, max(len(other.xs) - take, 0) + 1))
            f.xs = np.concatenate([f.xs[: n - take], other.xs[src : src + take]])
            f.ys = np.concatenate([f.ys[: n - take], other.ys[src : src + take]])
            # The fragment now ENDS on the other player, so that is where a
            # transition prior would think this identity was last seen.
            f.donor = j
            if i in emb and j in emb:
                frac = take / n
                mixed = (1.0 - frac) * emb[i] + frac * emb[j]
                norm = np.linalg.norm(mixed)
                emb[i] = mixed / norm if norm > 1e-9 else mixed

    if c.oversegment > 0:
        split_out, split_emb = [], {}
        for i, f in enumerate(out):
            n = len(f.xs)
            if n < 4 or rng.random() >= c.oversegment:
                split_emb[len(split_out)] = emb[i] if i in emb else None
                split_out.append(f)
                continue
            cut = int(rng.integers(2, n - 1))
            span = max(f.end - f.start, 1)
            mid = f.start + int(span * cut / n)
            a, b = copy.copy(f), copy.copy(f)
            a.xs, a.ys, a.end = f.xs[:cut], f.ys[:cut], mid
            b.xs, b.ys, b.start = f.xs[cut:], f.ys[cut:], mid + 1
            for part in (a, b):
                split_emb[len(split_out)] = emb[i] if i in emb else None
                split_out.append(part)
        out = split_out
        emb = {k: v for k, v in split_emb.items() if v is not None}

    return out, emb


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
    import gc

    rows, labels = [], []
    for m in matches:
        for key in (f"{m}_H1", f"{m}_H2"):
            frags, first_xy, last_xy, app = half_frames(key)
            states = initial_states(frags, first_xy, last_xy, app)
            r, y = oracle_pairs(frags, states, first_xy)
            if len(r):
                rows.append(r)
                labels.append(y)
            # A FOOTPASS half is ~1.6M rows and the fragments hold slices of it.
            # Only the scored pairs are needed past this point, and holding four
            # halves at once is what makes this die under memory pressure.
            del frags, first_xy, last_xy, app, states, r, y
            gc.collect()
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


def thread_half(
    key,
    cals,
    prior,
    w,
    *,
    min_score: float,
    min_margin: float,
    accumulate: bool = True,
    corruption: Corruption | None = None,
) -> dict:
    """Greedy sequential threading.

    `accumulate=False` is the control that isolates what accumulation is worth:
    identical decision rule, identical channels, but a thread is represented by
    its most recent fragment alone rather than by everything seen of it. Without
    this arm the headline number is unattributable -- it could be the threading
    rule rather than the accumulated evidence.
    """
    frags, first_xy, last_xy, app = half_frames(key)
    if corruption is not None:
        frags, app = corrupt_fragments(frags, app, corruption)
        # Remap endpoints: a contaminated fragment now ENDS on its donor, so
        # that is where the transition prior should think this identity was
        # last seen. Getting this wrong would hide most of the damage.
        origin = np.array([f.origin for f in frags])
        donor = np.array([f.donor for f in frags])
        first_xy = first_xy[origin]
        last_xy = np.where(
            (donor >= 0)[:, None], last_xy[np.maximum(donor, 0)], last_xy[origin]
        )
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
            if accumulate:
                threads[best_k] = threads[best_k].merged_with(states[i])
                t_fp[best_k] = threads[best_k].footprint()
            else:
                # Control: the thread advances to the joining fragment and
                # forgets everything before it.
                threads[best_k] = states[i]
                t_fp[best_k] = q_fp[int(i)]
            t_members[best_k].append(int(i))
        else:
            threads.append(states[i])
            t_fp.append(q_fp[int(i)])
            t_team.append(int(team[i]))
            t_members.append([int(i)])

    sizes = [len(m) for m in t_members]
    pure = sum(1 for m in t_members if len(set(pid[m])) == 1)
    n_players = int(len(set(pid.tolist())))
    # int() throughout: the counters accumulate numpy bools into np.int64, which
    # json.dumps refuses -- and it refuses AFTER the run has printed, so the
    # results look complete right up to the point where nothing is saved.
    return {
        "key": key,
        "fragments": len(frags),
        "true_players": n_players,
        "merges_needed": len(frags) - n_players,
        "threads": len(threads),
        "merges": int(correct + wrong),
        "correct": int(correct),
        "wrong": int(wrong),
        "pure_threads": int(pure),
        "largest_thread": int(max(sizes)) if sizes else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--matches", nargs="+", default=list(MATCHES),
                    help="held-out matches to evaluate; the rest are always fitted on")
    ap.add_argument(
        "--contaminate", type=float, nargs="+", default=[0.0],
        help="fraction of EVALUATED fragments given an ID switch. Corruption is "
             "applied only to the evaluated match, because the question is what "
             "happens when a real tracker feeds a system calibrated on clean data.",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("docs/reports/2026-07-28-bootstrap-threads.json")
    )
    args = ap.parse_args()

    results = []
    for held_out in args.matches:
        others = [m for m in MATCHES if m != held_out]
        print(f"fit on {others} -> evaluate {held_out}", flush=True)
        cals, prior, w = fit_from(others)
        print(f"  weights {dict(zip((*CHANNELS, 'transition'), w.round(3), strict=True))}",
              flush=True)
        for ms in args.min_score:
            for cont in args.contaminate:
                corr = None if cont <= 0 else Corruption(contaminate=cont, seed=0)
                for key in (f"{held_out}_H1", f"{held_out}_H2"):
                    for acc in (True, False):
                        r = thread_half(key, cals, prior, w, min_score=ms,
                                        min_margin=args.min_margin, accumulate=acc,
                                        corruption=corr)
                        r["min_score"] = ms
                        r["accumulate"] = acc
                        r["contaminate"] = cont
                        results.append(r)
                        print(
                            f"  {key} thr={ms:4.1f} c={cont:4.2f}"
                            f" {'ACCUM' if acc else 'recent':>6s}"
                            f"  merges {r['merges']:5d}"
                            f"  correct {r['correct']:5d}  wrong {r['wrong']:4d}"
                            f"  threads {r['threads']:4d} (true {r['true_players']})",
                            flush=True,
                        )

    print(f"\n{'threshold':>10s} {'contam':>7s} {'candidate':>10s} {'merges':>8s}"
          f" {'correct':>8s} {'wrong':>7s} {'precision':>10s} {'coverage':>9s}"
          f" {'purity':>8s}")
    summary = {}
    for ms in args.min_score:
      for cont in args.contaminate:
        for acc in (True, False):
            rows = [
                r for r in results
                if r["min_score"] == ms and r["accumulate"] == acc
                and r["contaminate"] == cont
            ]
            if not rows:
                continue
            c = sum(r["correct"] for r in rows)
            wr = sum(r["wrong"] for r in rows)
            pt = sum(r["pure_threads"] for r in rows)
            th = sum(r["threads"] for r in rows)
            # Coverage against the merges the footage actually requires. Precision
            # alone flatters an abstaining system: merging nothing scores 100%.
            need = sum(r["merges_needed"] for r in rows)
            summary[f"thr={ms}/c={cont}/{'accum' if acc else 'recent'}"] = {
                "correct": c, "wrong": wr, "merges_needed": need,
                "precision": round(c / max(c + wr, 1), 4),
                "coverage": round(c / max(need, 1), 4),
            }
            print(
                f"{ms:10.1f} {cont:7.2f} {'ACCUM' if acc else 'recent':>10s}"
                f" {c + wr:8d} {c:8d}"
                f" {wr:7d} {100 * c / max(c + wr, 1):9.2f}%"
                f" {100 * c / max(need, 1):8.2f}% {100 * pt / max(th, 1):7.1f}%"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_half": results}, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
