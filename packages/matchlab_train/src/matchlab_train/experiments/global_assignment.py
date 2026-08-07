"""Global-assignment A/B for the two-pass FOOTPASS bootstrap harness.

The greedy two-pass merge has no mutual-exclusivity resolution: pass 1 lets
the earlier-starting of two temporally-overlapping fragments take a thread
unconditionally, and pass 2's score-ordered sweep is a maximal matching, not a
maximum-weight one. This experiment swaps ONLY the decision resolution --
scoring, channels, calibrators, weights and verdict accounting are exactly
`bootstrap_threads`' -- so the A/B isolates the decision rule.

Pass 1 (rule "hungarian"): fragments are grouped into mutual-overlap CLIQUES
(extend the batch while next.start <= min(end of batch)); within a clique at
most one member may join any given thread, which is exactly the bipartite
assignment `scipy.optimize.linear_sum_assignment` solves. Cliques -- not
overlap-graph components -- are deliberate: no clique member can be eligible
for a thread an earlier member joined (they overlap), so every eligible
(fragment, thread) score is identical under greedy and Hungarian and the A/B
carries zero evidence-staleness confound. The price is that chained conflicts
straddling a batch boundary are NOT resolved; `split_conflict_pairs` counts
them so the covered fraction of the conflict population is known.

Pass 2 (rule "matching"): per round, `networkx.max_weight_matching` (exact,
general graph) over edges with score >= pass2_score, edge weight
(score - pass2_score) + EPS -- a surplus objective, chosen for the do-no-harm
posture (one high-surplus merge beats two marginal ones; merge count is not
the objective).

Usage:
  uv run python -m matchlab_train.experiments.global_assignment \
      --max-gap-frames 30 --min-score 2 2.5 3 3.5 4 4.5 5 5.5 6
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from matchlab_train.experiments import bootstrap_threads as bt

#: Dummy "start a new thread" columns are priced just under min_score so the
#: LP never forces a below-threshold join, while a singleton batch still
#: reduces exactly to greedy's inclusive `best_s >= min_score` rule.
EPS = 1e-9
#: Ineligible (team / overlap / no-required-evidence) sentinel. Finite --
#: scipy rejects inf -- and unreachable: each fragment's own dummy column
#: strictly dominates it.
NEG = -1e12


def build_batches(order: np.ndarray, start: np.ndarray, end: np.ndarray) -> list[list[int]]:
    """Mutual-overlap cliques over start-ordered fragments.

    Extend while the next fragment starts at or before every current member's
    end (inclusive ends: overlap iff b.start <= a.end, the same convention as
    the harness envelope gate's complement). Every pair in a batch overlaps,
    so at most one member can join any given thread.
    """
    batches: list[list[int]] = []
    cur: list[int] = []
    min_end = None
    for i in order:
        i = int(i)
        if cur and start[i] <= min_end:
            cur.append(i)
            min_end = min(min_end, int(end[i]))
        else:
            if cur:
                batches.append(cur)
            cur, min_end = [i], int(end[i])
    if cur:
        batches.append(cur)
    return batches


def conflict_pair_counts(
    batches: list[list[int]], start, end, team
) -> tuple[int, int]:
    """(within-batch, split-across-batches) same-team overlapping pair counts.

    A "conflict pair" is two same-team fragments whose spans overlap -- the
    population whose competition greedy resolves by arrival order. Split pairs
    are the ones clique batching cannot see (design scope, cold-review
    finding 1).
    """
    n = len(start)
    batch_id = np.empty(n, dtype=np.int64)
    for bi, b in enumerate(batches):
        batch_id[b] = bi
    s = np.asarray(start)[:, None]
    e = np.asarray(end)[:, None]
    t = np.asarray(team)
    overlap = (s.T <= e) & (s <= e.T) & (t[:, None] == t[None, :])
    overlap &= np.triu(np.ones((n, n), dtype=bool), k=1)
    same_batch = batch_id[:, None] == batch_id[None, :]
    within = int(np.count_nonzero(overlap & same_batch))
    split = int(np.count_nonzero(overlap & ~same_batch))
    return within, split


def _matching_selection(cands, scores, min_score: float) -> list[int]:
    """Indices into `cands` chosen by exact max-weight matching (surplus)."""
    g = nx.Graph()
    for idx, ((x, y), s) in enumerate(zip(cands, scores)):
        if s >= min_score:
            g.add_edge(x, y, weight=float(s - min_score) + EPS, idx=idx)
    if not g.edges:
        return []
    mate = nx.max_weight_matching(g, maxcardinality=False)
    return [g.edges[x, y]["idx"] for x, y in mate]


def _greedy_selection(cands, scores, min_score: float) -> list[int]:
    """bootstrap_threads.agglomerate's sweep: score order, first-come `used`."""
    order = np.argsort(-scores)
    used: set[int] = set()
    out = []
    for idx in order:
        if scores[idx] < min_score:
            break
        x, y = cands[idx]
        if x in used or y in used:
            continue
        used.update((x, y))
        out.append(int(idx))
    return out


def agglomerate(
    threads, t_fp, t_team, t_members, pid, start, end, cals, prior, w, *,
    min_score, rounds=8, rule="greedy",
):
    """bootstrap_threads.agglomerate with the pair-selection rule swappable.

    Verdict accounting (link_endpoints edge verdicts + majority) is identical;
    with rule="greedy" the merge set per round is identical too.
    """
    threads = list(threads)
    t_fp = list(t_fp)
    t_members = [list(m) for m in t_members]
    correct = wrong = maj_correct = maj_wrong = 0
    edges: list[tuple[int, int, bool]] = []
    for _ in range(rounds):
        live = [k for k in range(len(threads)) if threads[k] is not None]
        cands = []
        for ai, a in enumerate(live):
            for b in live[ai + 1:]:
                if t_team[a] != t_team[b]:
                    continue
                x, y = (a, b) if threads[a].last_end <= threads[b].last_end else (b, a)
                if not bt.members_disjoint(t_members[x], t_members[y], start, end):
                    continue
                cands.append((x, y))
        if not cands:
            break
        rows = np.array(
            [bt.thread_pair_row(threads[x], t_fp[x], threads[y], t_fp[y]) for x, y in cands]
        )
        scores = bt.apply_weights(bt.channel_llrs(rows, cals, prior), rows[:, 2], w)
        pick = (_greedy_selection if rule == "greedy" else _matching_selection)(
            cands, scores, min_score
        )
        if not pick:
            break
        for idx in pick:
            x, y = cands[idx]
            a, b = bt.link_endpoints(t_members[x], t_members[y], start)
            ok = pid[a] == pid[b]
            correct += ok
            wrong += not ok
            edges.append((a, b, bool(ok)))
            mx = Counter(pid[t_members[x]]).most_common(1)[0][0]
            my = Counter(pid[t_members[y]]).most_common(1)[0][0]
            maj_correct += mx == my
            maj_wrong += mx != my
            threads[x] = threads[x].merged_with(threads[y])
            t_fp[x] = threads[x].footprint()
            t_members[x] = t_members[x] + t_members[y]
            threads[y] = None
            t_members[y] = []
    keep = [k for k in range(len(threads)) if threads[k] is not None]
    return (
        [threads[k] for k in keep], [t_fp[k] for k in keep],
        [t_members[k] for k in keep],
        int(correct), int(wrong), int(maj_correct), int(maj_wrong), edges,
    )


def thread_half(
    key, cals, prior, w, *,
    min_score: float,
    p1: str = "hungarian",
    p2: str | None = "greedy",
    pass2_score: float = 4.0,
) -> dict:
    """bootstrap_threads.thread_half with swappable decision rules.

    p1: "greedy" (batches forced to singletons -- reproduces the incumbent
    exactly, tested) or "hungarian". p2: None (off), "greedy", "matching".
    Scoring is bootstrap_threads' own; margin fixed at the shipped operating
    point (0).
    """
    frags, first_xy, last_xy, app = bt.half_frames(key)
    states = bt.initial_states(frags, first_xy, last_xy, app)
    pid = np.array([f.player_id for f in frags])
    team = np.array([f.team for f in frags])
    start = np.array([f.start for f in frags])
    end = np.array([f.end for f in frags])
    q_fp = [s.footprint() for s in states]
    q_proto = [s.prototype for s in states]

    order = np.argsort(start)
    # Clique stats always describe the REAL overlap structure (reported for
    # every arm); the greedy arm merely SOLVES singleton batches over it.
    cliques = build_batches(order, start, end)
    batches = [[int(i)] for i in order] if p1 == "greedy" else cliques
    within, split = conflict_pair_counts(cliques, start, end, team)

    threads: list = []
    t_fp: list = []
    t_team: list[int] = []
    t_members: list[list[int]] = []
    correct = wrong = maj_correct = maj_wrong = 0
    p1_edges: list[tuple[int, int, bool]] = []

    def join(k: int, i: int) -> None:
        nonlocal correct, wrong, maj_correct, maj_wrong
        a, b = bt.link_endpoints(t_members[k], [i], start)
        ok = pid[a] == pid[b]
        correct += ok
        wrong += not ok
        p1_edges.append((a, b, bool(ok)))
        majority = Counter(pid[t_members[k]]).most_common(1)[0][0]
        maj_correct += majority == pid[i]
        maj_wrong += majority != pid[i]
        threads[k] = threads[k].merged_with(states[i])
        t_fp[k] = threads[k].footprint()
        t_members[k].append(i)

    def new_thread(i: int) -> None:
        threads.append(states[i])
        t_fp.append(q_fp[i])
        t_team.append(int(team[i]))
        t_members.append([i])

    for batch in batches:
        # Live threads and scores against PRE-BATCH thread states. Within a
        # clique no member is eligible for a thread another member joins
        # (they overlap), so these scores equal the greedy path's.
        nt = len(threads)
        mat = np.full((len(batch), nt + len(batch)), NEG)
        for r, i in enumerate(batch):
            live = [k for k in range(nt)
                    if t_team[k] == team[i] and threads[k].last_end < start[i]]
            if live:
                rows = np.array([
                    bt.raw_row(threads[k], t_fp[k], i, states, q_fp, q_proto, first_xy)
                    for k in live
                ])
                mat[r, live] = bt.apply_weights(
                    bt.channel_llrs(rows, cals, prior), rows[:, 2], w
                )
            mat[r, nt + r] = min_score - EPS  # this fragment's "new thread"
        rr, cc = linear_sum_assignment(mat, maximize=True)
        for r, c in zip(rr, cc):
            i = batch[r]
            if c < nt and mat[r, c] >= min_score:
                join(int(c), i)
            else:
                new_thread(i)

    p2_c = p2_w = p2_mc = p2_mw = 0
    p2_edges: list = []
    if p2 is not None:
        (threads, t_fp, t_members, p2_c, p2_w, p2_mc, p2_mw, p2_edges) = agglomerate(
            threads, t_fp, t_team, t_members, pid, start, end,
            cals, prior, w, min_score=pass2_score, rule=p2,
        )

    sizes = [len(m) for m in t_members]
    pure = sum(1 for m in t_members if len(set(pid[m])) == 1)
    n_players = int(len(set(pid.tolist())))
    return {
        "key": key,
        "fragments": len(frags),
        "true_players": n_players,
        "merges_needed": len(frags) - n_players,
        "threads": len(t_members),
        "merges": int(correct + wrong + p2_c + p2_w),
        "correct": int(correct + p2_c),
        "wrong": int(wrong + p2_w),
        "pass1_correct": int(correct),
        "pass1_wrong": int(wrong),
        "pass2_correct": int(p2_c),
        "pass2_wrong": int(p2_w),
        "majority_correct": int(maj_correct + p2_mc),
        "majority_wrong": int(maj_wrong + p2_mw),
        "pure_threads": int(pure),
        "largest_thread": int(max(sizes)) if sizes else 0,
        "n_batches": len(cliques),
        "n_multi_batches": int(sum(len(b) > 1 for b in cliques)),
        "max_batch": int(max(len(b) for b in cliques)) if cliques else 0,
        "conflict_pairs_within_batch": int(within),
        "conflict_pairs_split": int(split),
        "pass1_edges": [(int(a), int(b), ok) for a, b, ok in p1_edges],
        "pass2_edges": [(int(a), int(b), ok) for a, b, ok in p2_edges],
    }


ARMS = (
    ("greedy", "greedy"),
    ("hungarian", "greedy"),
    ("greedy", "matching"),
    ("hungarian", "matching"),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, nargs="+",
                    default=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    ap.add_argument("--pass2-score", type=float, default=2.0)
    ap.add_argument("--max-gap-frames", type=int, default=30)
    ap.add_argument("--matches", nargs="+", default=list(bt.MATCHES))
    ap.add_argument("--out", type=Path,
                    default=Path("docs/reports/2026-08-03-global-assignment.json"))
    args = ap.parse_args()
    bt.MAX_GAP_FRAMES = args.max_gap_frames

    import networkx
    import scipy
    results = []
    for held_out in args.matches:
        others = [m for m in bt.MATCHES if m != held_out]
        print(f"fit on {others} -> evaluate {held_out}", flush=True)
        cals, prior, w = bt.fit_from(others)
        for ms in args.min_score:
            for key in (f"{held_out}_H1", f"{held_out}_H2"):
                for p1, p2 in ARMS:
                    r = thread_half(key, cals, prior, w, min_score=ms,
                                    p1=p1, p2=p2, pass2_score=args.pass2_score)
                    r |= {"min_score": ms, "p1": p1, "p2": p2,
                          "pass2_score": args.pass2_score}
                    results.append(r)
                    print(
                        f"  {key} thr={ms:4.1f} {p1[:4]}/{p2[:5]}"
                        f"  correct {r['correct']:5d}  wrong {r['wrong']:4d}"
                        f"  (p2 {r['pass2_correct']:4d}/{r['pass2_wrong']:3d})"
                        f"  threads {r['threads']:4d}"
                        f"  multi-batches {r['n_multi_batches']}"
                        f" split-conflicts {r['conflict_pairs_split']}",
                        flush=True,
                    )

    print(f"\n{'threshold':>10s} {'arm':>16s} {'merges':>8s} {'correct':>8s}"
          f" {'wrong':>7s} {'precision':>10s} {'coverage':>9s}")
    summary = {}
    for ms in args.min_score:
        for p1, p2 in ARMS:
            rows = [r for r in results
                    if r["min_score"] == ms and r["p1"] == p1 and r["p2"] == p2]
            if not rows:
                continue
            c = sum(r["correct"] for r in rows)
            wr = sum(r["wrong"] for r in rows)
            need = sum(r["merges_needed"] for r in rows)
            arm = f"{p1}/{p2}"
            summary[f"thr={ms}/{arm}"] = {
                "correct": c, "wrong": wr, "merges_needed": need,
                "precision": round(c / max(c + wr, 1), 4),
                "coverage": round(c / max(need, 1), 4),
            }
            print(f"{ms:10.1f} {arm:>16s} {c + wr:8d} {c:8d} {wr:7d}"
                  f" {100 * c / max(c + wr, 1):9.2f}%"
                  f" {100 * c / max(need, 1):8.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "provenance": {
            "scipy": scipy.__version__, "networkx": networkx.__version__,
            "max_gap_frames": args.max_gap_frames, "coords": bt.COORDS,
            "pass2_score": args.pass2_score,
        },
        "summary": summary,
        "per_half": results,
    }, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
