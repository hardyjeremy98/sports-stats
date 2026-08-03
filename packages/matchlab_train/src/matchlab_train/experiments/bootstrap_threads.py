"""Two-pass bootstrap: merge into accumulated THREADS, not into single fragments.

The multi-input harness scores a query fragment against candidate FRAGMENTS. On
FOOTPASS a fragment lasts a median 8.2 s (12.2 s mean) and touches, on game_18_H1,
a median of 3 of 96 pitch cells -- so "the candidate that typically occupies that
space" is being asked of a smudge.

With oracle threading, representing a candidate by everything seen of it so far
is worth +12.8 rank-1 on body ID and +7.4 on occupancy.

NOTE these fragments are GT observability spans, NOT tracker tracklets: they split
whenever the player is off-camera for more than 2 frames, whereas a tracker bridges
short occlusions with its motion buffer and can carry an ID switch inside a single
tracklet. Real tracklets measured on SNMOT under the same >=2 s filter run to a
median 10.0 s against these 8.2 s, so this substrate is MORE fragmented than
reality -- harder in fragment count and in evidence per fragment, easier in every
other respect.

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
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matchlab_core.reid.evidence import (
    LLRCalibrator,
    clip_transition,
    fit_fusion_weights,
)
from matchlab_core.reid.jersey import number_prior, pair_llr, uniform_prior
from matchlab_core.reid.occupancy import js_distance
from matchlab_core.reid.threads import ThreadState
from matchlab_core.reid.transition import TransitionPrior, displacement
from matchlab_core.reid.twopass import _pool_jersey

from matchlab_train.datasets.footpass import COL, load_half
from matchlab_train.experiments.multi_input import VAL_H5, load_appearance
from matchlab_train.experiments.position_evidence import build_fragments

FPS = 25.0
MATCHES = ("game_18", "game_24", "game_47")
CHANNELS = ("body", "occupancy", "gap")
# How long a player may be out of frame before the span is cut. The default of 2
# frames (80 ms) makes fragments finer than any real tracker's output: a tracker
# bridges short absences with its motion buffer and only surrenders after ~1 s,
# so re-ID is asked about the LONGER gaps. Raise this to make the units
# tracker-shaped, so that every required merge is a genuine re-ID decision
# rather than a re-link the tracker would have made internally.
MAX_GAP_FRAMES = 2


def link_endpoints(members_x, members_y, start) -> tuple[int, int]:
    """The two tracklets a merge edge actually bridges: x's latest, y's earliest.

    Scoring against these rather than against a thread's majority player keeps
    the verdict fixed to the link that was made. A thread that has already
    absorbed the wrong player would otherwise start marking correct
    continuations wrong, and wrong ones right -- charging one bad merge again at
    every merge that follows it.

    The facing ends are found by start time rather than by list position. Both
    passes do happen to build `t_members` chronologically, but relying on that
    would fail silently and invisibly if the pass-2 overlap gate were ever
    relaxed: the verdict would quietly begin scoring the wrong pair.
    """
    x = min(members_x, key=lambda m: (-start[m], m))
    y = min(members_y, key=lambda m: (start[m], m))
    return int(x), int(y)


def members_disjoint(mx, my, start, end) -> bool:
    """Do two threads' tracklets interleave without ever overlapping?

    The physical constraint is that one player cannot be in two places at once,
    which is a statement about the TRACKLETS. Testing it on the threads'
    outer envelopes instead -- `x.last_end < y.first_start` -- is strictly
    stronger, and wrongly: two threads of the same player routinely interleave
    (x covers 0-10s and 100-110s, y covers 50-60s) with no tracklet overlapping
    any other. The envelope test blocks that pair permanently.

    It also gets worse as merging proceeds: every merge widens an envelope, so
    agglomeration raises its own floor round by round.

    Envelopes are checked first as a cheap sufficient condition -- disjoint
    envelopes imply disjoint members -- so the interval scan only runs on the
    pairs the envelope test would have rejected.
    """
    ax, bx = start[mx], end[mx]
    ay, by = start[my], end[my]
    if bx.max() < ay.min() or by.max() < ax.min():
        return True
    ox, oy = np.argsort(ax), np.argsort(ay)
    i = j = 0
    while i < len(ox) and j < len(oy):
        p, q = ox[i], oy[j]
        if ax[p] <= by[q] and ay[q] <= bx[p]:
            return False
        if bx[p] < by[q]:
            i += 1
        else:
            j += 1
    return True


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


CACHE = Path("data/experiments/bootstrap-cache")

# Cache-variant knobs. These MUST appear in every cache filename that depends
# on them: the caches were previously keyed only by (key, MAX_GAP_FRAMES), so
# changing the coordinate convention or adding per-fragment data silently
# returned stale pickles fitted under the old semantics -- the exact failure
# class of the positional-cache alignment bug (2026-07-27) and the occupancy
# fit/serve mismatch this revision exists to fix.
#: "rel" = formation-relative xs/ys (the historical harness behaviour; what
#: fusion-footpass-v1 was fitted on), "abs" = absolute pitch coordinates (what
#: reid_engine actually serves).
COORDS = "rel"
#: Last-K/first-K absolute observation rows cached per fragment, for endpoint
#: velocity estimation (item 3). Part of the cache key.
TAIL_K = 30
#: Gap-conditioned body calibration: upper edges (seconds) of the gap bins,
#: each bin getting its own body LLR calibrator, the last (open) bin falling
#: through to the flat calibrator. () = flat everywhere (historical). Part of
#: the fit cache key.
GAP_BINS: tuple[float, ...] = ()
#: Gap-conditioned channel WEIGHTS (upper edges, seconds; one open last bin).
#: Distinct from GAP_BINS (per-bin body CALIBRATORS): a jointly-weighted gap
#: channel can shift the decision boundary per bin (an offset in the fused
#: sum) but cannot rescale how many nats an appearance margin is WORTH per
#: bin (a slope). Per-bin weights are that slope experiment. Fitted as one
#: conditional logit over block-expanded features (llr x bin indicator), so
#: bins share episodes and normalisation. () = one global weight vector.
WEIGHT_GAP_BINS: tuple[float, ...] = ()
#: Negative-side bound (nats) on the transition channel's LLR; the positive
#: side keeps the global saturate. 6.0 = the historical symmetric bound.
#: Part of the FIT cache key (weights are fitted on the clipped column); the
#: measured case for sweeping it down is the channel's own flip counts
#: (veto half 1 correct / 31 harmful blocks, transition.py 2026-07-28).
TRANS_NEG_CLAMP = 6.0
# The fragmentation the appearance extractor was run at. `fragment_embeddings.pkl`
# is keyed by position in THAT list, so any other setting must be remapped.
APPEARANCE_GAP_FRAMES = 2


def _base_fragments(key: str):
    """Fragments at the fragmentation the appearance cache was built for."""
    for name in (f"{key}-g{APPEARANCE_GAP_FRAMES}.pkl", f"{key}.pkl"):
        p = CACHE / name
        if p.exists():
            return pickle.loads(p.read_bytes())[0]
    return build_fragments(
        load_half(VAL_H5, key),
        max_gap_frames=APPEARANCE_GAP_FRAMES,
        min_frames=50,
        relative=True,
    )


def appearance_for(key: str, frags) -> dict:
    """Embeddings indexed to `frags`, remapping across fragmentations if needed.

    Reading the cache positionally at a different `max_gap_frames` is what
    invalidated every figure published on 2026-07-30, so the mismatch is
    repaired here rather than left to the caller to notice.
    """
    app = load_appearance(key) or {}
    if not app or len(app) == len(frags):
        return app
    app = remap_appearance(frags, _base_fragments(key), app)
    print(f"  remapped appearance for {key}: {len(app)}/{len(frags)} fragments")
    return app


def half_frames(key: str):
    """Fragments plus the per-fragment endpoints the transition prior needs.

    Cached, and the endpoint lookup is done by sorting the observable rows once
    and slicing per fragment. The naive version masked all ~1.6M rows once per
    fragment -- 2,500 full-array scans per half -- which was both slow and the
    reason this died under memory pressure.
    """
    cache = CACHE / f"{key}-g{MAX_GAP_FRAMES}-{COORDS}-t{TAIL_K}.pkl"
    if cache.exists():
        frags, first_xy, last_xy = pickle.loads(cache.read_bytes())
        return frags, first_xy, last_xy, appearance_for(key, frags)

    half = load_half(VAL_H5, key)
    frags = build_fragments(
        half, max_gap_frames=MAX_GAP_FRAMES, min_frames=50,
        relative=(COORDS == "rel"),
    )
    rows = half.rows
    obs = rows[~np.isnan(rows[:, COL.ROI_X])]
    order = np.lexsort((obs[:, COL.FRAME], obs[:, COL.PLAYER_ID]))
    obs = obs[order]
    pids = obs[:, COL.PLAYER_ID]
    frames = obs[:, COL.FRAME]

    first_xy, last_xy = [], []
    for f in frags:
        lo = int(np.searchsorted(pids, f.player_id, "left"))
        hi = int(np.searchsorted(pids, f.player_id, "right"))
        block = frames[lo:hi]
        a = lo + int(np.searchsorted(block, f.start, "left"))
        b = lo + int(np.searchsorted(block, f.end, "right"))
        first_xy.append(obs[a, [COL.X, COL.Y]].astype(float))
        last_xy.append(obs[b - 1, [COL.X, COL.Y]].astype(float))
        # Absolute-coordinate endpoint tails for velocity estimation: the
        # fragment's own xs/ys may be formation-relative (COORDS="rel"), whose
        # derivative is player-minus-centroid velocity -- wrong for a
        # transition prediction. (frame, x, y) rows, real timestamps, because
        # spans can contain gaps up to MAX_GAP_FRAMES.
        f.tail_xy = obs[max(b - TAIL_K, a):b][:, [COL.FRAME, COL.X, COL.Y]].astype(float)
        f.head_xy = obs[a:min(a + TAIL_K, b)][:, [COL.FRAME, COL.X, COL.Y]].astype(float)
    first_xy, last_xy = np.array(first_xy), np.array(last_xy)

    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps((frags, first_xy, last_xy)))
    return frags, first_xy, last_xy, appearance_for(key, frags)


def remap_appearance(frags, base_frags, base_app: dict) -> dict:
    """Re-index positionally-keyed embeddings from one fragmentation onto another.

    `fragment_embeddings.pkl` is keyed by position in the `max_gap_frames=2`
    fragment list, so it cannot be used directly at any other setting. A coarser
    span is by construction the union of consecutive finer spans of the SAME
    player, so each coarse fragment's embedding is the frame-count-weighted mean
    of the fine embeddings it contains -- exact re-indexing rather than an
    approximation.

    Containment is checked on player id as well as span. Two players occupy
    overlapping frame ranges constantly, so matching on time alone would pool
    strangers into one prototype.
    """
    by_player: dict[int, list] = {}
    for j, g in enumerate(base_frags):
        if j in base_app:
            by_player.setdefault(int(g.player_id), []).append((g, base_app[j]))

    out: dict[int, np.ndarray] = {}
    for i, f in enumerate(frags):
        acc, wt = None, 0.0
        for g, emb in by_player.get(int(f.player_id), ()):
            if g.start >= f.start and g.end <= f.end:
                v = np.asarray(emb, dtype=float) * len(g.xs)
                acc = v if acc is None else acc + v
                wt += len(g.xs)
        if acc is None or wt <= 0:
            continue
        mean = acc / wt
        norm = float(np.linalg.norm(mean))
        out[i] = mean / norm if norm > 1e-9 else mean
    return out


def check_appearance_alignment(frags, app) -> None:
    """Fail loudly if the embedding cache was built for a different fragmentation.

    `load_appearance` returns {fragment_index: embedding} from a file with no
    record of the `max_gap_frames`/`min_frames` it was generated under, and
    `initial_states` looks embeddings up POSITIONALLY. Changing the
    fragmentation therefore silently hands each fragment some other span's
    embedding -- which is exactly what happened on 2026-07-30: every
    `max_gap_frames=30` run scored with 58.3% of embeddings belonging to the
    wrong player, on the highest-weighted channel, and every published figure
    had to be withdrawn. Nothing detected it because nothing checked.

    The file enumerates 0..N-1 for its own N, so an out-of-range key proves it
    was built for a different, longer fragment list. Note the converse is NOT an
    error: after `remap_appearance` some fragments legitimately have no
    embedding, and missing evidence is neutral rather than a fault (ADR 003).

    This does not catch a same-length misalignment. The durable fix is to key
    `fragment_embeddings.pkl` by (player_id, start, end) in the extractor, which
    is not in this tree.
    """
    if app and max(app) >= len(frags):
        raise ValueError(
            f"appearance cache indexes fragment {max(app)} but there are only "
            f"{len(frags)} fragments -- it was generated for a different "
            f"fragmentation. Remap it with remap_appearance() rather than "
            f"indexing across the two (MAX_GAP_FRAMES={MAX_GAP_FRAMES})."
        )


def initial_states(frags, first_xy, last_xy, app) -> list[ThreadState]:
    check_appearance_alignment(frags, app)
    out = []
    for i, f in enumerate(frags):
        e = app.get(i)
        if e is not None:
            e = np.asarray(e, dtype=np.float64)
            n = np.linalg.norm(e)
            e = e / n if n > 1e-9 else None
        out.append(
            ThreadState.from_fragment(
                f.xs, f.ys, embedding=e, start=f.start, end=f.end,
                exit_xy=last_xy[i], entry_xy=first_xy[i],
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


def thread_pair_row(a: ThreadState, a_fp, b: ThreadState, b_fp):
    """Channel raw scores for joining thread `b` onto thread `a` (a ends first).

    The pass-2 counterpart of `raw_row`. Both sides are accumulated here, which
    is the whole point: pass 1 always had a single fragment on the query side,
    so half the evidence in every comparison was a 6.6-second smudge.
    """
    pa, pb = a.prototype, b.prototype
    body = float(pa @ pb) if (pa is not None and pb is not None) else np.nan
    dx, dy = displacement(a.exit_xy[None, :], b.entry_xy[None, :])
    return [
        body,
        js_distance(b_fp, a_fp),
        (b.first_start - a.last_end) / FPS,
        float(dx[0]),
        float(dy[0]),
    ]


def _fuse(rows, cals, prior, w, scorer, ctx_builder):
    """Fused score per candidate row: the incumbent, or a plug-in scorer.

    The plug-in exists so that arms testing a richer INPUT representation share
    this module's decision rule, candidate generation, gates and accounting
    verbatim -- otherwise a frontier difference is not attributable to the
    representation. `scorer=None` is the incumbent and is bit-identical to it
    (test_edge_scorer.py), so the default path is unchanged.
    """
    if scorer is None:
        return apply_weights(channel_llrs(rows, cals, prior), rows[:, 2], w)
    return scorer.score(rows, ctx_builder())


def agglomerate(
    threads, t_fp, t_team, t_members, pid, start, end, cals, prior, w, *,
    min_score, rounds=8, gate="members", jersey=None, scorer=None
):
    """Repeatedly merge the best-scoring compatible pair of THREADS.

    Greedy agglomeration rather than one sweep: every merge strengthens the
    surviving thread, so a pair that was below threshold can clear it once one
    side has absorbed more evidence. That is the actual two-pass idea -- pass 1
    only ever compared a thread against a lone fragment.

    Compatibility is the same pair of physically-certain constraints used
    everywhere else: same team, and non-overlapping spans, because one player
    cannot be in two places at once.
    """
    threads = list(threads)
    t_fp = list(t_fp)
    t_members = [list(m) for m in t_members]
    t_jersey, j_prior, j_weight = (
        (list(jersey[0]), jersey[1], jersey[2]) if jersey is not None
        else (None, None, 0.0)
    )
    correct = wrong = 0
    maj_correct = maj_wrong = 0
    for _ in range(rounds):
        live = [k for k in range(len(threads)) if threads[k] is not None]
        cands = []
        for ai, a in enumerate(live):
            for b in live[ai + 1 :]:
                if t_team[a] != t_team[b]:
                    continue
                x, y = (a, b) if threads[a].last_end <= threads[b].last_end else (b, a)
                if gate == "members":
                    if not members_disjoint(t_members[x], t_members[y], start, end):
                        continue
                elif threads[x].last_end >= threads[y].first_start:
                    continue
                cands.append((x, y))
        if not cands:
            break
        rows = np.array(
            [thread_pair_row(threads[x], t_fp[x], threads[y], t_fp[y]) for x, y in cands]
        )

        def _ctx(cands=cands, threads=threads):
            # Pass 2 has no per-query field: `cands` is a global pool of every
            # compatible thread pair. The field a cohort-normalising arm needs
            # is therefore defined PER THREAD -- the pairs sharing thread x --
            # which is the closest well-defined analogue of pass 1's field.
            from matchlab_train.experiments.edge_scorer import ScoreContext

            owner = np.array([x for x, _ in cands], dtype=np.int64)
            size = np.bincount(owner, minlength=owner.max() + 1)[owner].astype(float)
            return ScoreContext(
                episode=owner,
                n_frames_a=np.array([threads[x].n_frames for x, _ in cands], float),
                n_frames_b=np.array([threads[y].n_frames for _, y in cands], float),
                n_fragments_a=np.array(
                    [threads[x].n_fragments for x, _ in cands], float
                ),
                n_fragments_b=np.array(
                    [threads[y].n_fragments for _, y in cands], float
                ),
                field_size=size,
            )

        scores = _fuse(rows, cals, prior, w, scorer, _ctx)
        if t_jersey is not None:
            scores = scores + j_weight * np.array(
                [pair_llr(t_jersey[x], t_jersey[y], j_prior) for x, y in cands]
            )
        order = np.argsort(-scores)
        used: set[int] = set()
        n_merged = 0
        for idx in order:
            if scores[idx] < min_score:
                break
            x, y = cands[idx]
            if x in used or y in used:
                continue
            used.update((x, y))
            # Scored while both sides are still intact. The verdict is on the
            # edge itself -- the facing ends of the two threads -- with the
            # majority rule kept alongside so the two can be compared.
            a, b = link_endpoints(t_members[x], t_members[y], start)
            correct += pid[a] == pid[b]
            wrong += pid[a] != pid[b]
            mx = Counter(pid[t_members[x]]).most_common(1)[0][0]
            my = Counter(pid[t_members[y]]).most_common(1)[0][0]
            maj_correct += mx == my
            maj_wrong += mx != my
            threads[x] = threads[x].merged_with(threads[y])
            t_fp[x] = threads[x].footprint()
            t_members[x] = t_members[x] + t_members[y]
            if t_jersey is not None:
                t_jersey[x] = _pool_jersey(t_jersey[x], t_jersey[y])
            threads[y] = None
            t_members[y] = []
            n_merged += 1
        if not n_merged:
            break
    keep = [k for k in range(len(threads)) if threads[k] is not None]
    return (
        [threads[k] for k in keep],
        [t_fp[k] for k in keep],
        [t_members[k] for k in keep],
        int(correct),
        int(wrong),
        int(maj_correct),
        int(maj_wrong),
    )


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
    rows, labels, episode = [], [], []
    for i in np.argsort(start):
        for p, st in grown.items():
            if p_team[p] != team[i] or st.last_end >= start[i]:
                continue
            rows.append(raw_row(st, grown_fp[p], int(i), states, q_fp, q_proto, first_xy))
            labels.append(p == pid[i])
            # One decision = one query fragment against its whole candidate
            # field. Blocking by row count instead pooled 2-3 unrelated
            # queries, each with its own positive, into one softmax.
            episode.append(int(i))
        key = int(pid[i])
        grown[key] = grown[key].merged_with(states[i]) if key in grown else states[i]
        grown_fp[key] = grown[key].footprint()
    return (np.array(rows, dtype=float), np.array(labels, dtype=bool),
            np.array(episode, dtype=np.int64))


def fit_from(matches: list[str]):
    import gc

    # Cached on the fit-match set: nothing about it varies across operating
    # points, and rebuilding it per sweep is what kept dying under memory
    # pressure from concurrent jobs.
    bins_tok = "b" + "_".join(f"{b:g}" for b in GAP_BINS) if GAP_BINS else "flat"
    if WEIGHT_GAP_BINS:
        bins_tok += "-wb" + "_".join(f"{b:g}" for b in WEIGHT_GAP_BINS)
    if TRANS_NEG_CLAMP != 6.0:
        # Weights are fitted on the clipped transition column; an unkeyed
        # sweep would silently return clamp-6 weights for every arm.
        bins_tok += f"-tnc{TRANS_NEG_CLAMP:g}"
    fit_cache = CACHE / (
        f"fit-{'+'.join(sorted(matches))}-g{MAX_GAP_FRAMES}-{COORDS}-{bins_tok}.pkl"
    )
    if fit_cache.exists():
        cals_d, prior_d, w = pickle.loads(fit_cache.read_bytes())
        cals = {k: LLRCalibrator.from_dict(v) for k, v in cals_d.items()
                if k != "body_by_gap"}
        if "body_by_gap" in cals_d:
            cals["body_by_gap"] = [
                (hi, LLRCalibrator.from_dict(cd)) for hi, cd in cals_d["body_by_gap"]
            ]
        return cals, TransitionPrior.from_dict(prior_d), w

    rows, labels, episodes = [], [], []
    for m in matches:
        for key in (f"{m}_H1", f"{m}_H2"):
            frags, first_xy, last_xy, app = half_frames(key)
            states = initial_states(frags, first_xy, last_xy, app)
            r, y, ep = oracle_pairs(frags, states, first_xy)
            if len(r):
                rows.append(r)
                labels.append(y)
                episodes.append(ep)
            # A FOOTPASS half is ~1.6M rows and the fragments hold slices of it.
            # Only the scored pairs are needed past this point, and holding four
            # halves at once is what makes this die under memory pressure.
            del frags, first_xy, last_xy, app, states, r, y
            gc.collect()
    r = np.concatenate(rows)
    y = np.concatenate(labels)
    # Episode ids are per-half fragment indices, so offset each half before
    # concatenating or two halves' query 7 would be pooled into one decision.
    eps, off = [], 0
    for e in episodes:
        eps.append(e + off)
        off += int(e.max()) + 1 if len(e) else 0
    ep_index = np.concatenate(eps)
    cals = {}
    for j, name in enumerate(CHANNELS):
        col = r[:, j]
        ok = ~np.isnan(col)
        cals[name] = LLRCalibrator.fit(col[ok & y], col[ok & ~y], max_bins=200)
    if GAP_BINS:
        gaps = r[:, 2]
        body = r[:, 0]
        ok = ~np.isnan(body) & ~np.isnan(gaps)
        lo = 0.0
        by_gap = []
        for hi in GAP_BINS:
            sel = ok & (gaps >= lo) & (gaps < hi)
            n_same, n_diff = int((sel & y).sum()), int((sel & ~y).sum())
            print(f"  gap bin [{lo:g},{hi:g})s: same {n_same} diff {n_diff}")
            # A thin bin gets no calibrator of its own: LLRCalibrator.fit
            # degrades quietly on scraps, and a noisy per-bin mapping is worse
            # than the flat one it would shadow.
            if n_same >= 200 and n_diff >= 200:
                by_gap.append(
                    (hi, LLRCalibrator.fit(body[sel & y], body[sel & ~y],
                                           max_bins=200))
                )
            lo = hi
        cals["body_by_gap"] = by_gap
    prior = TransitionPrior.fit(r[:, 2], r[:, 3], r[:, 4], y)
    llr = channel_llrs(r, cals, prior)
    # One episode = one query fragment against its whole candidate field, which
    # is what the conditional logit expects. This was `arange(n) // 64`, a fixed
    # block size that straddled 2-3 unrelated queries and normalised the softmax
    # over several positives at once.
    if WEIGHT_GAP_BINS:
        # One conditional logit over block-expanded features: row i's LLRs sit
        # in the column block of its own gap bin, zeros elsewhere. Episodes
        # (query fields) are unchanged, so candidates at different gaps still
        # compete inside one softmax -- which is the decision being modelled.
        wb = _weight_bin(r[:, 2], WEIGHT_GAP_BINS)
        n_bins = len(WEIGHT_GAP_BINS) + 1
        n_ch = llr.shape[1]
        expanded = np.zeros((len(llr), n_bins * n_ch))
        for b in range(n_bins):
            sel = wb == b
            expanded[sel, b * n_ch:(b + 1) * n_ch] = llr[sel]
            print(f"  weight bin {b}: same {int((sel & y).sum())} "
                  f"diff {int((sel & ~y).sum())}")
        w_flat = fit_fusion_weights(expanded, ep_index, y)
        w = {"edges": tuple(WEIGHT_GAP_BINS),
             "w": w_flat.reshape(n_bins, n_ch)}
    else:
        w = fit_fusion_weights(llr, ep_index, y)
    CACHE.mkdir(parents=True, exist_ok=True)
    cals_d = {k: v.to_dict() for k, v in cals.items() if k != "body_by_gap"}
    if "body_by_gap" in cals:
        cals_d["body_by_gap"] = [(hi, c.to_dict()) for hi, c in cals["body_by_gap"]]
    fit_cache.write_bytes(pickle.dumps((cals_d, prior.to_dict(), w)))
    return cals, prior, w


def _weight_bin(gaps: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Row -> weight-bin index (0..len(edges)); the last bin is open."""
    return np.searchsorted(np.asarray(edges, dtype=float), gaps, side="right")


def apply_weights(llr: np.ndarray, gaps: np.ndarray, w) -> np.ndarray:
    """Fused score per row: global (`w` is a vector) or gap-binned (`w` is
    {"edges": ..., "w": (n_bins, n_channels)} -- each pair scored with the
    weight vector of ITS OWN gap bin)."""
    if not isinstance(w, dict):
        return llr @ w
    wb = _weight_bin(np.asarray(gaps, dtype=float), tuple(w["edges"]))
    return np.einsum("ij,ij->i", llr, np.asarray(w["w"])[wb])


def channel_llrs(r: np.ndarray, cals, prior) -> np.ndarray:
    by_gap = cals.get("body_by_gap") or []

    def body_cal(gap: float):
        for hi, cal in by_gap:
            if gap < hi:
                return cal
        return cals["body"]

    cols = []
    for j, name in enumerate(CHANNELS):
        col = r[:, j]
        if name == "gap":
            # Pass-2 interleaved thread pairs carry a negative envelope gap,
            # outside the calibrator's fitted support; the engine serves the
            # clamped 0 (score_channels), so the harness must too.
            col = np.maximum(col, 0.0)
        if name == "body" and by_gap:
            cols.append(np.array([
                body_cal(float(g)).llr(float(v)) if not np.isnan(v) else 0.0
                for v, g in zip(col, r[:, 2])
            ]))
            continue
        cols.append(
            np.array([cals[name].llr(float(v)) if not np.isnan(v) else 0.0 for v in col])
        )
    # Bounded like every other channel. The prior itself is deliberately
    # unbounded -- a 60 m move in 1 s costs it tens of nats, which is the point --
    # but `fit_fusion_weights` gives each channel ONE linear coefficient, and a
    # column reaching -3754 nats against everything else's +/-6 cannot share one.
    # Reporting the ratio honestly is the prior's job; putting channels on a
    # common footing is this layer's. The negative side's bound is swept
    # separately (TRANS_NEG_CLAMP); dt <= 0 abstains outright -- the sigma
    # floor makes the prior's output there an artifact, and the engine's
    # score_channels gate does the same.
    t = np.asarray(clip_transition(
        prior.llr(r[:, 2], r[:, 3], r[:, 4]), TRANS_NEG_CLAMP
    ))
    t = np.where(r[:, 2] > 0.0, t, 0.0)
    cols.append(t)
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
    pass2: bool = False,
    pass2_score: float = 4.0,
    gate: str = "members",
    jersey: tuple[list, np.ndarray, float] | None = None,
    scorer=None,
) -> dict:
    """Greedy sequential threading.

    `accumulate=False` is the control that isolates what accumulation is worth:
    identical decision rule, identical channels, but a thread is represented by
    its most recent fragment alone rather than by everything seen of it. Without
    this arm the headline number is unattributable -- it could be the threading
    rule rather than the accumulated evidence.

    `jersey` = (per-fragment number likelihoods, number prior, weight): the
    engine's jersey-OCR merge channel served exactly as reid_engine does --
    fixed weight, `pair_llr` against the pooled thread belief, `_pool_jersey`
    accumulation -- so with/without is a pure channel ablation. Incompatible
    with `corruption` (contaminated fragments would need their jersey evidence
    remapped, which nothing here does).
    """
    frags, first_xy, last_xy, app = half_frames(key)
    if jersey is not None and corruption is not None:
        raise ValueError("jersey evidence is not corruption-remapped; run one or the other")
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
    end = np.array([f.end for f in frags])

    q_fp = [s.footprint() for s in states]
    q_proto = [s.prototype for s in states]
    q_jersey, j_prior, j_weight = jersey if jersey is not None else (None, None, 0.0)

    threads: list[ThreadState] = []
    t_fp: list[object] = []
    t_team: list[int] = []
    t_members: list[list[int]] = []
    t_jersey: list[np.ndarray] = []
    correct = wrong = 0
    maj_correct = maj_wrong = 0

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
            def _ctx(live=live, i=i):
                from matchlab_train.experiments.edge_scorer import ScoreContext

                n = len(live)
                return ScoreContext(
                    episode=np.zeros(n, dtype=np.int64),  # one decision
                    n_frames_a=np.array([threads[k].n_frames for k in live], float),
                    n_frames_b=np.full(n, float(states[int(i)].n_frames)),
                    n_fragments_a=np.array(
                        [threads[k].n_fragments for k in live], float
                    ),
                    n_fragments_b=np.ones(n),
                    field_size=np.full(n, float(n)),
                )

            scores = _fuse(r, cals, prior, w, scorer, _ctx)
            if q_jersey is not None:
                scores = scores + j_weight * np.array(
                    [pair_llr(t_jersey[k], q_jersey[i], j_prior) for k in live]
                )
            o = np.argsort(-scores)
            best_k, best_s = live[o[0]], float(scores[o[0]])
            runner = float(scores[o[1]]) if len(o) > 1 else -np.inf

        if best_k is not None and best_s >= min_score and (best_s - runner) >= min_margin:
            a, b = link_endpoints(t_members[best_k], [int(i)], start)
            correct += pid[a] == pid[b]
            wrong += pid[a] != pid[b]
            majority = Counter(pid[t_members[best_k]]).most_common(1)[0][0]
            maj_correct += majority == pid[i]
            maj_wrong += majority != pid[i]
            if accumulate:
                threads[best_k] = threads[best_k].merged_with(states[i])
                t_fp[best_k] = threads[best_k].footprint()
                if q_jersey is not None:
                    t_jersey[best_k] = _pool_jersey(t_jersey[best_k], q_jersey[i])
            else:
                # Control: the thread advances to the joining fragment and
                # forgets everything before it.
                threads[best_k] = states[i]
                t_fp[best_k] = q_fp[int(i)]
                if q_jersey is not None:
                    t_jersey[best_k] = q_jersey[i]
            t_members[best_k].append(int(i))
        else:
            threads.append(states[i])
            t_fp.append(q_fp[int(i)])
            t_team.append(int(team[i]))
            t_members.append([int(i)])
            if q_jersey is not None:
                t_jersey.append(q_jersey[i])

    p2_correct = p2_wrong = 0
    p2_maj_correct = p2_maj_wrong = 0
    if pass2:
        (threads, t_fp, t_members, p2_correct, p2_wrong,
         p2_maj_correct, p2_maj_wrong) = agglomerate(
            threads, t_fp, t_team, t_members, pid, start, end,
            cals, prior, w, min_score=pass2_score, gate=gate,
            jersey=(t_jersey, j_prior, j_weight) if q_jersey is not None else None,
            scorer=scorer,
        )

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
        "merges": int(correct + wrong + p2_correct + p2_wrong),
        "correct": int(correct + p2_correct),
        "wrong": int(wrong + p2_wrong),
        "pass1_correct": int(correct),
        "pass1_wrong": int(wrong),
        "pass2_correct": int(p2_correct),
        "pass2_wrong": int(p2_wrong),
        # Legacy accounting, kept only so the change in what is measured is
        # visible rather than silent.
        "majority_correct": int(maj_correct + p2_maj_correct),
        "majority_wrong": int(maj_wrong + p2_maj_wrong),
        "pure_threads": int(pure),
        "largest_thread": int(max(sizes)) if sizes else 0,
    }


def main() -> None:
    global MAX_GAP_FRAMES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, nargs="+", default=[0.0, 2.0, 4.0, 6.0])
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--max-gap-frames", type=int, default=2,
                    help="frames a player may be out of frame before the span is cut; "
                         "~30 (1.2s) approximates a real tracker's buffer")
    ap.add_argument("--gate", choices=("envelope", "members"), default="members",
                    help="pass-2 compatibility test: thread envelopes (legacy) or "
                         "tracklet-interval disjointness (physically correct)")
    ap.add_argument("--pass2-score", type=float, nargs="+", default=[4.0],
                    help="threshold for thread-to-thread agglomeration in pass 2")
    ap.add_argument("--matches", nargs="+", default=list(MATCHES),
                    help="held-out matches to evaluate; the rest are always fitted on")
    ap.add_argument(
        "--contaminate", type=float, nargs="+", default=[0.0],
        help="fraction of EVALUATED fragments given an ID switch. Corruption is "
             "applied only to the evaluated match, because the question is what "
             "happens when a real tracker feeds a system calibrated on clean data.",
    )
    ap.add_argument(
        "--trans-neg-clamp", type=float, default=6.0,
        help="negative-side bound (nats) on the transition channel; 6.0 is the "
             "historical symmetric saturate, 0.0 flattens all vetoes to 0. "
             "Weights are refit per value (fit cache keyed).",
    )
    ap.add_argument(
        "--jersey-weight", type=float, default=0.0,
        help="fixed fused weight for the jersey-OCR channel (engine convention "
             "jersey_weight_twopass=1.0); 0 = channel absent. Needs the cached "
             "posteriors from matchlab_train.experiments.footpass_jersey.",
    )
    ap.add_argument(
        "--trans-off", action="store_true",
        help="serve with the transition weight zeroed (fit unchanged): the "
             "channel-off control that shows what the positive half is worth",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("docs/reports/2026-07-28-bootstrap-threads.json")
    )
    args = ap.parse_args()
    MAX_GAP_FRAMES = args.max_gap_frames
    global TRANS_NEG_CLAMP
    TRANS_NEG_CLAMP = args.trans_neg_clamp

    results = []
    for held_out in args.matches:
        others = [m for m in MATCHES if m != held_out]
        print(f"fit on {others} -> evaluate {held_out}", flush=True)
        cals, prior, w = fit_from(others)
        if args.trans_off:
            if isinstance(w, dict):
                w = {"edges": w["edges"], "w": np.asarray(w["w"], dtype=float).copy()}
                w["w"][:, -1] = 0.0
            else:
                w = np.asarray(w, dtype=float).copy()
                w[-1] = 0.0
        if isinstance(w, dict):
            print(f"  gap-binned weights (edges {w['edges']}):\n{np.round(w['w'], 3)}",
                  flush=True)
        else:
            print(f"  weights {dict(zip((*CHANNELS, 'transition'), w.round(3), strict=True))}",
                  flush=True)
        for ms in args.min_score:
            for cont in args.contaminate:
                corr = None if cont <= 0 else Corruption(contaminate=cont, seed=0)
                for key in (f"{held_out}_H1", f"{held_out}_H2"):
                    jersey = None
                    if args.jersey_weight > 0.0:
                        # Lazy import: footpass_jersey imports this module.
                        from matchlab_train.experiments.footpass_jersey import (
                            fragment_posteriors,
                        )

                        post = fragment_posteriors(
                            key, max_gap_frames=args.max_gap_frames
                        )
                        flat = uniform_prior()
                        q_jersey = [
                            post["likelihood"].get(fi, flat)
                            for fi in range(post["n_fragments"])
                        ]
                        # Engine convention: the prior comes from this run's
                        # own confident reads, never from GT.
                        j_prior = (
                            number_prior(post["confident_numbers"])
                            if post["confident_numbers"] else uniform_prior()
                        )
                        jersey = (q_jersey, j_prior, args.jersey_weight)
                    arms = [(True, None), (False, None)]
                    arms += [(True, p) for p in args.pass2_score]
                    for acc, p2s in arms:
                        p2 = p2s is not None
                        r = thread_half(key, cals, prior, w, min_score=ms,
                                        min_margin=args.min_margin, accumulate=acc,
                                        corruption=corr, pass2=p2,
                                        pass2_score=p2s if p2 else 0.0,
                                        gate=args.gate, jersey=jersey)
                        r["min_score"] = ms
                        r["accumulate"] = acc
                        r["pass2"] = p2
                        r["pass2_score"] = p2s
                        r["contaminate"] = cont
                        results.append(r)
                        arm = (f"ACCUM+P2@{p2s:g}" if p2 else "ACCUM") if acc else "recent"
                        print(
                            f"  {key} thr={ms:4.1f} c={cont:4.2f} {arm:>9s}"
                            f"  merges {r['merges']:5d}"
                            f"  correct {r['correct']:5d}  wrong {r['wrong']:4d}"
                            f"  (p2 {r['pass2_correct']:4d}/{r['pass2_wrong']:3d})"
                            f"  threads {r['threads']:4d} (true {r['true_players']})",
                            flush=True,
                        )

    print(f"\n{'threshold':>10s} {'contam':>7s} {'candidate':>10s} {'merges':>8s}"
          f" {'correct':>8s} {'wrong':>7s} {'precision':>10s} {'coverage':>9s}"
          f" {'purity':>8s} {'majority%':>9s}")
    summary = {}
    for ms in args.min_score:
      for cont in args.contaminate:
        arms = [(True, None), (False, None)] + [(True, p) for p in args.pass2_score]
        for acc, p2s in arms:
            rows = [
                r for r in results
                if r["min_score"] == ms and r["accumulate"] == acc
                and r["contaminate"] == cont and r["pass2_score"] == p2s
            ]
            if not rows:
                continue
            c = sum(r["correct"] for r in rows)
            wr = sum(r["wrong"] for r in rows)
            mc = sum(r["majority_correct"] for r in rows)
            mw = sum(r["majority_wrong"] for r in rows)
            pt = sum(r["pure_threads"] for r in rows)
            th = sum(r["threads"] for r in rows)
            # Coverage against the merges the footage actually requires. Precision
            # alone flatters an abstaining system: merging nothing scores 100%.
            need = sum(r["merges_needed"] for r in rows)
            arm = (f"accum+p2@{p2s:g}" if p2s is not None else "accum") if acc else "recent"
            summary[f"thr={ms}/c={cont}/{arm}"] = {
                "correct": c, "wrong": wr, "merges_needed": need,
                "precision": round(c / max(c + wr, 1), 4),
                "coverage": round(c / max(need, 1), 4),
                "majority_correct": mc, "majority_wrong": mw,
                "majority_precision": round(mc / max(mc + mw, 1), 4),
            }
            print(
                f"{ms:10.1f} {cont:7.2f} {arm:>10s}"
                f" {c + wr:8d} {c:8d}"
                f" {wr:7d} {100 * c / max(c + wr, 1):9.2f}%"
                f" {100 * c / max(need, 1):8.2f}% {100 * pt / max(th, 1):7.1f}%"
                f" {100 * mc / max(mc + mw, 1):9.2f}%"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_half": results}, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
