"""Do richer input treatments beat the hand-reduced scalars at the merge frontier?

Spec: `docs/superpowers/specs/2026-08-03-richer-input-representations.md`.

Every "re-ID here is evidence-limited" negative on this stack was measured on
four hand-reduced scalars (pooled-prototype cosine, JS distance, gap seconds,
linear-velocity residual), each squashed through a 1-D calibrator and summed with
one scalar weight. The learned systems those negatives are implicitly
benchmarked against consume raw per-detection or sequence-valued cues instead, so
the negative is REPRESENTATION-SCOPED. This module removes one part of the
reduction at a time and measures what changes.

Substrate assembly lives here; the arms live beside it. `oracle_pairs_rich`
mirrors `bootstrap_threads.oracle_pairs` exactly -- and is asserted to, because a
"richer" arm scored against a subtly different pair population would be
comparing substrates, not representations.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np

from matchlab_train.experiments import bootstrap_threads as bt
from matchlab_train.experiments.edge_scorer import (
    _FRONTIER_DTYPE,
    ScoreContext,
    cluster_bootstrap_delta,
    frontier,
    hull,
    precision_at_coverage,
)

#: The gap bins every per-regime breakdown uses, in seconds. Shared with the
#: audit report so results are comparable across rounds.
GAP_BINS = (2.0, 7.0, 30.0)


@dataclass
class FoldData:
    """Everything one LOSO fold needs, with the pair population fixed.

    `cluster` is the player-within-half of the QUERY fragment -- the resampling
    unit for the static pair frontier. Episodes of one player share thread
    state, embedding and territory, so an episode-level bootstrap is too tight.
    """

    rows: np.ndarray          # (n, 5) body_cos, js, gap_s, dx, dy
    labels: np.ndarray        # (n,) bool
    episodes: np.ndarray      # (n,) global episode id
    cluster: np.ndarray       # (n,) player-within-half id
    ctx: ScoreContext
    query_frag: np.ndarray    # (n,) global fragment id of the query
    cand_key: np.ndarray      # (n,) global id of the candidate thread's player
    half: np.ndarray          # (n,) which half the row came from
    keys: tuple[str, ...]
    #: Experiment 0: per-row summaries of the cross-frame cosine matrix, of
    #: which "proto" is the incumbent's own statistic recomputed here. Empty
    #: when the per-frame embeddings were not requested.
    set_stats: dict[str, np.ndarray] = None  # type: ignore[assignment]
    #: Experiment 2: per-fragment exit trajectories, and the per-row index of
    #: whichever fragment supplies the thread's exit point.
    tails: np.ndarray = None  # type: ignore[assignment]
    tail_mask: np.ndarray = None  # type: ignore[assignment]
    tail_of: np.ndarray = None  # type: ignore[assignment]
    entry_xy: np.ndarray = None  # type: ignore[assignment]

    def __len__(self) -> int:
        return len(self.rows)

    def summary(self) -> dict:
        return {
            "rows": int(len(self.rows)),
            "episodes": int(len(np.unique(self.episodes))),
            "positives": int(self.labels.sum()),
            "clusters": int(len(np.unique(self.cluster))),
            "halves": list(self.keys),
        }


def fragment_frame_rows(key: str, frags) -> tuple[object, list[np.ndarray]]:
    """Per-fragment row indices into the half's per-frame embedding table.

    Keyed by (player_id, span) rather than by fragment position, so this is
    correct at ANY fragmentation -- which is the whole reason
    `frame_embeddings` exists.
    """
    from matchlab_train.experiments import frame_embeddings as fe

    table = fe.load(key)
    idx = table.index()
    rows = [
        table.for_span(idx, f.player_id, f.start, f.end) for f in frags
    ]
    return table, rows


def oracle_pairs_rich(
    frags, states, first_xy, *, half_id: int, frag_offset: int, frame_sets=None
):
    """`bootstrap_threads.oracle_pairs`, plus the context the richer arms need.

    Deliberately a copy of that function's loop rather than a wrapper: the
    context (field size, per-side frame counts, which fragment asked) is only
    available INSIDE the loop, and reconstructing it afterwards would mean
    re-deriving the candidate sets -- the sort of parallel implementation that
    drifts silently. `test_input_representations.py` asserts the rows, labels
    and episodes match `oracle_pairs` element-for-element.
    """
    pid = np.array([f.player_id for f in frags])
    team = np.array([f.team for f in frags])
    start = np.array([f.start for f in frags])
    end_of = np.array([f.end for f in frags])
    q_fp = [s.footprint() for s in states]
    q_proto = [s.prototype for s in states]
    p_team = {int(p): int(team[np.flatnonzero(pid == p)[0]]) for p in np.unique(pid)}

    grown: dict[int, object] = {}
    grown_fp: dict[int, object] = {}
    grown_frags: dict[int, int] = {}
    rows, labels, episode = [], [], []
    n_a, n_b, nf_a, nf_b, qfrag, ckey = [], [], [], [], [], []
    #: (first row index, block size) per decision, so field size can be stamped
    #: on every row of a block once the block is complete.
    blocks: list[tuple[int, int]] = []

    from matchlab_train.experiments.frame_embeddings import (
        POOL_STATS,
        SET_STATS,
        THREAD_SET_CAP,
        pool_prototypes,
        set_statistics,
    )

    table, frag_rows = frame_sets if frame_sets is not None else (None, None)
    grown_rows: dict[int, list[int]] = {}
    #: Member structure, kept alongside the flat row list: the pooling arms need
    #: to know which frames belonged to which member fragment, which is exactly
    #: what the flat concatenation destroys.
    grown_members: dict[int, list[int]] = {}
    n_frames_of = [len(f.xs) for f in frags]
    #: Experiment 2: which fragment's trajectory ENDS at the thread's exit
    #: point. The static prior conditions on that point alone; the sequence arm
    #: gets the run-up to it.
    grown_last: dict[int, int] = {}
    tail_of, entry_of = [], []
    stats: dict[str, list[float]] = {k: [] for k in (*SET_STATS, *POOL_STATS)}

    for i in np.argsort(start):
        block = 0
        for p, st in grown.items():
            if p_team[p] != team[i] or st.last_end >= start[i]:
                continue
            rows.append(
                bt.raw_row(st, grown_fp[p], int(i), states, q_fp, q_proto, first_xy)
            )
            labels.append(p == pid[i])
            episode.append(int(i))
            n_a.append(st.n_frames)
            n_b.append(states[int(i)].n_frames)
            nf_a.append(grown_frags[p])
            nf_b.append(1)
            qfrag.append(frag_offset + int(i))
            ckey.append(int(p))
            tail_of.append(grown_last[p])
            entry_of.append(first_xy[int(i)])
            block += 1
            if table is not None:
                # The thread's appearance SET, not its mean. Capped at the most
                # recent samples so the statistic describes appearance rather
                # than how long the thread has been accumulating -- a thread
                # grows to hundreds of samples over a half while the query side
                # has a median of 7.
                a = table.emb[np.asarray(grown_rows[p][-THREAD_SET_CAP:], dtype=int)]
                b = table.emb[frag_rows[int(i)]]
                s = set_statistics(a, b)
                for k in SET_STATS:
                    stats[k].append(s[k])
                # Experiment 4: the same decision under three thread-prototype
                # pooling rules, scored as an ordinary cosine against the
                # query's own prototype so nothing downstream changes.
                pools = pool_prototypes(
                    [table.emb[frag_rows[m]] for m in grown_members[p]],
                    [n_frames_of[m] for m in grown_members[p]],
                )
                qb = b.astype(np.float64).mean(axis=0) if len(b) else None
                if qb is not None:
                    qn = float(np.linalg.norm(qb))
                    qb = qb / qn if qn > 1e-9 else None
                for k in POOL_STATS:
                    pv = pools[k]
                    stats[k].append(
                        float(pv @ qb) if (pv is not None and qb is not None)
                        else float("nan")
                    )
        # Field size is a property of the DECISION, so it is stamped on every
        # row of the block once the block is known -- not incremented per row,
        # which would leak the candidate's position in the iteration order into
        # a feature the arms are allowed to use.
        if block:
            blocks.append((len(rows) - block, block))
        key = int(pid[i])
        # The thread's exit point comes from whichever member ends latest, so
        # the trajectory feeding that exit point is that member's tail -- not
        # simply the most recently ADDED member.
        if key not in grown or int(end_of[i]) >= int(end_of[grown_last[key]]):
            grown_last[key] = int(i)
        grown[key] = grown[key].merged_with(states[i]) if key in grown else states[i]
        grown_fp[key] = grown[key].footprint()
        grown_frags[key] = grown_frags.get(key, 0) + 1
        if table is not None:
            grown_rows.setdefault(key, []).extend(
                int(v) for v in frag_rows[int(i)]
            )
            grown_members.setdefault(key, []).append(int(i))

    field = np.ones(len(rows), dtype=np.float64)
    for lo, size in blocks:
        field[lo:lo + size] = size

    # Per-FRAGMENT tails, indexed per row: 12.5k fragments beats 88k rows by an
    # order of magnitude in memory, and the tail is a property of the fragment.
    # Left-padded with a validity mask so a short tail is never confused with a
    # stationary player -- fabricating "did not move" is precisely how the old
    # (0,0)-endpoint substitution scored the prior's maximum positive evidence.
    from matchlab_train.experiments.frame_embeddings import TAIL_LEN

    tails = np.zeros((len(frags), TAIL_LEN, 3), dtype=np.float32)
    tail_mask = np.zeros((len(frags), TAIL_LEN), dtype=bool)
    for j, f in enumerate(frags):
        t = np.asarray(getattr(f, "tail_xy", np.zeros((0, 3))), dtype=np.float32)
        if len(t):
            t = t[-TAIL_LEN:]
            tails[j, TAIL_LEN - len(t):] = t
            tail_mask[j, TAIL_LEN - len(t):] = True

    return {
        "rows": np.array(rows, dtype=float).reshape(-1, 5),
        "labels": np.array(labels, dtype=bool),
        "episodes": np.array(episode, dtype=np.int64),
        "n_frames_a": np.array(n_a, dtype=np.float64),
        "n_frames_b": np.array(n_b, dtype=np.float64),
        "n_fragments_a": np.array(nf_a, dtype=np.float64),
        "n_fragments_b": np.array(nf_b, dtype=np.float64),
        "field_size": field,
        "query_frag": np.array(qfrag, dtype=np.int64),
        "cand_key": np.array(ckey, dtype=np.int64),
        "query_player": np.array([pid[e] for e in episode], dtype=np.int64),
        "half": np.full(len(rows), half_id, dtype=np.int64),
        "set_stats": {k: np.array(v, dtype=np.float64) for k, v in stats.items()},
        "tails": tails,
        "tail_mask": tail_mask,
        "tail_of": np.array(tail_of, dtype=np.int64),
        "entry_xy": np.array(entry_of, dtype=np.float64).reshape(-1, 2),
    }


def assert_appearance_aligned(frags, app, base_frags, key: str) -> None:
    """Positively assert every embedding belongs to the fragment holding it.

    `check_appearance_alignment` only catches an out-of-RANGE index; its own
    docstring says it "does not catch a same-length misalignment" -- and a
    same-length misalignment is what withdrew every figure published on
    2026-07-30, on the highest-weighted channel, with nothing detecting it.

    This substrate runs at MAX_GAP_FRAMES=30 while the embedding cache was built
    at 2, so EVERY embedding goes through `remap_appearance`. That is exactly
    the operation the withdrawn figures got wrong, so it is checked rather than
    trusted: a coarse fragment's embedding must be the frame-count-weighted mean
    of the fine fragments of THE SAME PLAYER contained in its span.
    """
    by_player: dict[int, list] = {}
    for j, g in enumerate(base_frags):
        by_player.setdefault(int(g.player_id), []).append((j, g))

    checked = 0
    for i, f in enumerate(frags):
        if i not in app:
            continue
        contained = [
            (j, g) for j, g in by_player.get(int(f.player_id), ())
            if g.start >= f.start and g.end <= f.end
        ]
        if not contained:
            raise AssertionError(
                f"{key}: fragment {i} (player {f.player_id}, {f.start}-{f.end}) "
                "carries an embedding but contains no fine fragment of that "
                "player -- the remap is misaligned."
            )
        checked += 1
    if checked < 0.5 * len(app):
        raise AssertionError(
            f"{key}: only {checked}/{len(app)} embeddings could be traced back "
            "to a containing fine fragment of the same player."
        )


def load_fold(held_out: str, *, verbose: bool = True, with_sets: bool = False) -> FoldData:
    """Assemble the held-out match's static pair population.

    Both halves, concatenated, with episode ids offset so two halves' query 7 do
    not become one decision -- the same offsetting `fit_from` does, and for the
    same reason.
    """
    keys = tuple(f"{held_out}_{h}" for h in ("H1", "H2"))
    parts = []
    ep_off = frag_off = clus_off = 0
    for hi, key in enumerate(keys):
        frags, first_xy, last_xy, app = bt.half_frames(key)
        assert_appearance_aligned(frags, app, bt._base_fragments(key), key)
        states = bt.initial_states(frags, first_xy, last_xy, app)
        sets = fragment_frame_rows(key, frags) if with_sets else None
        d = oracle_pairs_rich(
            frags, states, first_xy, half_id=hi, frag_offset=frag_off,
            frame_sets=sets,
        )
        d["episodes"] = d["episodes"] + ep_off
        # Player ids repeat across halves and are NOT the same person's thread
        # state, so the cluster is (player, half) -- offsetting keeps that true
        # without depending on the id space.
        d["cluster"] = d["query_player"] + clus_off
        ep_off += len(frags)
        frag_off += len(frags)
        clus_off += int(d["query_player"].max()) + 1 if len(d["query_player"]) else 0
        parts.append(d)
        if verbose:
            print(f"  {key}: {len(d['rows'])} rows, "
                  f"{len(np.unique(d['episodes']))} episodes, "
                  f"{int(d['labels'].sum())} positives", flush=True)
        del frags, first_xy, last_xy, app, states
        gc.collect()

    def cat(name):
        return np.concatenate([p[name] for p in parts])

    rows = np.concatenate([p["rows"] for p in parts])
    ctx = ScoreContext(
        episode=cat("episodes"),
        n_frames_a=cat("n_frames_a"),
        n_frames_b=cat("n_frames_b"),
        n_fragments_a=cat("n_fragments_a"),
        n_fragments_b=cat("n_fragments_b"),
        field_size=cat("field_size"),
    )
    return FoldData(
        rows=rows,
        labels=cat("labels"),
        episodes=cat("episodes"),
        cluster=cat("cluster"),
        ctx=ctx,
        query_frag=cat("query_frag"),
        cand_key=cat("cand_key"),
        half=cat("half"),
        keys=keys,
        set_stats={
            k: np.concatenate([p["set_stats"][k] for p in parts])
            for k in parts[0]["set_stats"]
        } if with_sets else {},
        tails=np.concatenate([p["tails"] for p in parts]),
        tail_mask=np.concatenate([p["tail_mask"] for p in parts]),
        # `tail_of` is a per-half fragment index; offset it the same way the
        # fragment ids are, or half 2's rows would read half 1's trajectories.
        tail_of=np.concatenate([
            p["tail_of"] + sum(len(q["tails"]) for q in parts[:k])
            for k, p in enumerate(parts)
        ]),
        entry_xy=np.concatenate([p["entry_xy"] for p in parts]),
    )


def operating_coverage(h: np.ndarray, *, target_precision: float = 0.98) -> float:
    """Coverage at which the incumbent hits its shipped precision regime.

    Deltas are quoted at a MATCHED coverage, and picking that coverage from the
    arm under test would be choosing the comparison point after seeing the
    result. It is therefore derived from the BASELINE alone.
    """
    if not len(h):
        return float("nan")
    ok = h[h["precision"] >= target_precision]
    return float(ok["coverage"].max()) if len(ok) else float(h["coverage"][0])


def measure_mde(scorer, *, n_boot: int = 400, seed: int = 0) -> dict:
    """What effect this substrate can actually resolve, measured before any arm.

    Reported against the audit's own Phase C "win" (+0.0024 precision at
    coverage 0.649, ~25 wrong merges of ~8k). If the half-width of the interval
    exceeds that, the substrate cannot resolve a v3-sized effect and every
    subsequent flat result is underpowered BY CONSTRUCTION -- which is a finding
    to state up front, not to discover four experiments later.

    `scorer` is a callable (held_out_match -> (FoldData, scores)); the incumbent
    is passed in rather than built here so the same measurement can be repeated
    against any baseline.
    """
    out = {"folds": {}}
    for held_out in bt.MATCHES:
        fold, scores = scorer(held_out)
        h = hull(frontier(scores, fold.labels, fold.episodes))
        cov = operating_coverage(h)
        # An arm compared against itself: the point estimate is exactly 0, so
        # the interval is purely the substrate's own resolution at this
        # coverage -- which is the definition of the MDE we need.
        by_player = cluster_bootstrap_delta(
            scores, scores, fold.labels, fold.episodes, fold.cluster,
            coverage=cov, n_boot=n_boot, seed=seed,
        )
        # The same thing resampled on HALVES: 2 clusters per fold, the honest
        # upper bound on uncertainty and a check that the player-level interval
        # is not quietly optimistic.
        by_half = cluster_bootstrap_delta(
            scores, scores, fold.labels, fold.episodes, fold.half,
            coverage=cov, n_boot=n_boot, seed=seed,
        )
        # Resolution is about a DIFFERENCE, so the informative quantity is the
        # spread of the baseline's own precision under resampling, not the
        # (identically zero) paired delta.
        spread = _precision_spread(
            scores, fold, coverage=cov, n_boot=n_boot, seed=seed
        )
        out["folds"][held_out] = {
            **fold.summary(),
            "operating_coverage": cov,
            "precision_at_coverage": precision_at_coverage(h, cov),
            "paired_ci_width_player": by_player["ci_width"],
            "paired_ci_width_half": by_half["ci_width"],
            "precision_ci_width_player": spread["ci_width_player"],
            "precision_ci_width_half": spread["ci_width_half"],
            "mde_player": spread["ci_width_player"],
            "unreachable_frac": spread["unreachable_frac"],
        }
        print(
            f"  {held_out}: coverage {cov:.3f} precision "
            f"{precision_at_coverage(h, cov):.4f}  "
            f"MDE(player-clustered 95%) {spread['ci_width_player']:.4f}  "
            f"MDE(half-clustered) {spread['ci_width_half']:.4f}",
            flush=True,
        )
    widths = [v["mde_player"] for v in out["folds"].values()]
    out["mde_player_max"] = float(np.nanmax(widths))
    out["v3_effect_size"] = 0.0024
    out["can_resolve_v3_sized_effect"] = bool(out["mde_player_max"] <= 0.0024)
    return out


def _precision_spread(scores, fold: FoldData, *, coverage, n_boot, seed) -> dict:
    """95% interval width on the baseline's precision at a fixed coverage."""
    res = {}
    for name, clus in (("player", fold.cluster), ("half", fold.half)):
        uniq = np.unique(clus)
        rows_by = {c: np.flatnonzero(clus == c) for c in uniq}
        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([rows_by[c] for c in pick])
            h = hull(frontier(scores[sel], fold.labels[sel], fold.episodes[sel]))
            draws[b] = precision_at_coverage(h, coverage)
        good = draws[np.isfinite(draws)]
        if len(good):
            lo, hi = np.quantile(good, [0.025, 0.975])
            res[f"ci_width_{name}"] = float(hi - lo)
        else:
            res[f"ci_width_{name}"] = float("nan")
        if name == "player":
            res["unreachable_frac"] = float(1.0 - len(good) / n_boot)
    return res


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------

_FOLD_CACHE: dict[str, FoldData] = {}


def fold(held_out: str, *, with_sets: bool = False) -> FoldData:
    """Held-out fold, memoised: assembling one costs a full FOOTPASS half load."""
    ck = f"{held_out}{'+sets' if with_sets else ''}"
    if ck not in _FOLD_CACHE:
        print(f"assembling fold {held_out}"
              f"{' (with per-frame sets)' if with_sets else ''}", flush=True)
        _FOLD_CACHE[ck] = load_fold(held_out, with_sets=with_sets)
    return _FOLD_CACHE[ck]


def fit_incumbent(held_out: str):
    """The shipped fusion model, fitted on the OTHER two matches.

    `TRANS_NEG_CLAMP` is pinned to 0.0 because that is what
    `fusion-footpass-v2` -- the adopted artefact -- ships. Baselining against
    the historical symmetric clamp would be measuring the arms against a model
    nothing serves.
    """
    others = [m for m in bt.MATCHES if m != held_out]
    return bt.fit_from(others)


def incumbent_scores(held_out: str):
    """(fold, scores) under the shipped linear-over-LLR fusion."""
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer

    f = fold(held_out)
    cals, prior, w = fit_incumbent(held_out)
    return f, LinearLLRScorer(cals, prior, w).score(f.rows, f.ctx)


def v3_scores(held_out: str):
    """(fold, scores) under the v3 gap-binned-weights model (5/20 s edges).

    Not an arm -- a CALIBRATION of the measuring instrument. The audit measured
    this model at +0.0024 precision over flat at matched coverage on this exact
    substrate, so running it through this harness answers two questions at once:
    whether the paired interval can resolve a v3-sized effect, and whether this
    harness reproduces a published number it did not produce. A harness that
    cannot re-derive the reference result has no business grading new arms
    (`reproduce-the-reference-metric-first`).
    """
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer

    f = fold(held_out)
    others = [m for m in bt.MATCHES if m != held_out]
    prev = bt.WEIGHT_GAP_BINS
    bt.WEIGHT_GAP_BINS = (5.0, 20.0)
    try:
        cals, prior, w = bt.fit_from(others)
    finally:
        bt.WEIGHT_GAP_BINS = prev
    return f, LinearLLRScorer(cals, prior, w).score(f.rows, f.ctx)


def calibrate_instrument(*, n_boot: int = 400, seed: int = 0) -> dict:
    """Paired v3-vs-flat delta per fold: the resolution that actually matters.

    The unpaired precision spread over-states uncertainty for a comparison in
    which both arms are re-scored on the SAME resampled clusters, because the
    substrate variance cancels. This measures the paired interval directly
    against a known, published effect.
    """
    out = {"folds": {}}
    for held_out in bt.MATCHES:
        f, base = incumbent_scores(held_out)
        _, v3 = v3_scores(held_out)
        cov = operating_coverage(hull(frontier(base, f.labels, f.episodes)))
        d = cluster_bootstrap_delta(
            base, v3, f.labels, f.episodes, f.cluster,
            coverage=cov, n_boot=n_boot, seed=seed,
        )
        out["folds"][held_out] = {"coverage": cov, **d}
        print(
            f"  {held_out}: v3 - flat = {d['delta']:+.4f} "
            f"[{d['lo']:+.4f}, {d['hi']:+.4f}] "
            f"(paired 95% width {d['ci_width']:.4f}, "
            f"{d['n_clusters']} clusters)",
            flush=True,
        )
    widths = [v["ci_width"] for v in out["folds"].values()]
    deltas = [v["delta"] for v in out["folds"].values()]
    out["paired_ci_width_max"] = float(np.nanmax(widths))
    out["mean_delta"] = float(np.nanmean(deltas))
    out["resolves_v3_sized_effect"] = bool(out["paired_ci_width_max"] <= 2 * 0.0024)
    return out


# --------------------------------------------------------------------------
# Experiment 0: set-to-set appearance
# --------------------------------------------------------------------------


def refit_with_body(fit_folds: list[FoldData], stat: str):
    """Refit calibrators, transition prior and weights with `body` replaced.

    Everything except column 0 is untouched, and the fit uses the SAME episodes
    and the same estimator as the incumbent -- so the comparison is about the
    appearance statistic and nothing else.
    """
    from matchlab_core.reid.evidence import LLRCalibrator, fit_fusion_weights
    from matchlab_core.reid.transition import TransitionPrior

    rows = np.concatenate([f.rows for f in fit_folds]).copy()
    y = np.concatenate([f.labels for f in fit_folds])
    if stat != "body":
        rows[:, 0] = np.concatenate([f.set_stats[stat] for f in fit_folds])
    # Episodes are per-fold ids; offset so two folds' query 7 are two decisions.
    eps, off = [], 0
    for f in fit_folds:
        e = f.episodes
        eps.append(e + off)
        off += int(e.max()) + 1 if len(e) else 0
    ep_index = np.concatenate(eps)

    cals = {}
    for j, name in enumerate(bt.CHANNELS):
        col = rows[:, j]
        ok = ~np.isnan(col)
        cals[name] = LLRCalibrator.fit(col[ok & y], col[ok & ~y], max_bins=200)
    prior = TransitionPrior.fit(rows[:, 2], rows[:, 3], rows[:, 4], y)
    w = fit_fusion_weights(bt.channel_llrs(rows, cals, prior), ep_index, y)
    return cals, prior, w


def experiments_0_and_4(*, n_boot: int = 300, seed: int = 0) -> dict:
    """Set-to-set appearance (exp 0) and thread-prototype pooling (exp 4).

    Run together because both are summaries of the same per-frame appearance
    sets and the fold assembly that produces those sets is the expensive part.

    Two controls carry the interpretation:
    - `proto` recomputes the incumbent's statistic over these sets, so it
      measures the PLUMBING (per-frame re-pooling, the thread-set cap). Every
      exp-0 arm is read against `proto`, not against `body`.
    - `pool_mean` reproduces the incumbent's pooling rule, so the exp-4 arms are
      read against it. If `pool_mean` itself differs from `body`, the difference
      is the re-pooling, not the rule.
    """
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer
    from matchlab_train.experiments.frame_embeddings import (
        POOL_SHRINK_N0,
        POOL_STATS,
        SET_STATS,
        THREAD_SET_CAP,
    )

    arms = ("body", *SET_STATS, *POOL_STATS)
    out = {
        "thread_set_cap": THREAD_SET_CAP,
        "pool_shrink_n0": POOL_SHRINK_N0,
        "arms": list(arms),
        "exp0_arms": list(SET_STATS),
        "exp4_arms": list(POOL_STATS),
        "folds": {},
    }
    for held_out in bt.MATCHES:
        test = fold(held_out, with_sets=True)
        fits = [fold(m, with_sets=True) for m in bt.MATCHES if m != held_out]

        scores, aucs = {}, {}
        for stat in arms:
            fit_mod = [
                f if stat == "body" else _with_body(f, f.set_stats[stat])
                for f in fits
            ]
            cals, prior, w = refit_with_body(fit_mod, "body")
            rows = test.rows if stat == "body" else _with_body(
                test, test.set_stats[stat]
            ).rows
            scores[stat] = LinearLLRScorer(cals, prior, w).score(rows, test.ctx)
            aucs[stat] = _auc(rows[:, 0], test.labels)

        out["folds"][held_out] = _compare_arms(
            test, scores, aucs, arms, base="body", n_boot=n_boot, seed=seed,
            label=held_out,
        )
    return out


def experiment_0(*, n_boot: int = 300, seed: int = 0) -> dict:
    """Does removing the appearance POOLING move the frontier?

    Model-free by construction: each arm is a different summary of the same
    cross-frame cosine matrix, calibrated by the same calibrator and fused by
    the same estimator. No capacity question, no objective question, no
    data-scale question -- which is what makes a null here mean something.

    `proto` is run as an arm too. It recomputes the incumbent's own statistic
    over these sets, so `proto` vs `body` measures the plumbing (per-frame
    re-pooling and the thread-set cap) and every other arm is read against
    `proto`, not against `body`. Without that control a "win" could be the cap.
    """
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer
    from matchlab_train.experiments.frame_embeddings import SET_STATS, THREAD_SET_CAP

    arms = ("body", *SET_STATS)
    out = {"thread_set_cap": THREAD_SET_CAP, "folds": {}, "arms": list(arms)}

    for held_out in bt.MATCHES:
        test = fold(held_out, with_sets=True)
        fits = [fold(m, with_sets=True) for m in bt.MATCHES if m != held_out]

        scores, aucs = {}, {}
        for stat in arms:
            cals, prior, w = refit_with_body(fits, stat)
            rows = test.rows.copy()
            if stat != "body":
                rows[:, 0] = test.set_stats[stat]
            scores[stat] = LinearLLRScorer(cals, prior, w).score(rows, test.ctx)
            aucs[stat] = _auc(rows[:, 0], test.labels)

        base = scores["body"]
        hb = hull(frontier(base, test.labels, test.episodes))
        band = _coverage_band(hb)
        fold_out = {**test.summary(), "band": band.tolist(),
                    "channel_auc": aucs, "arms": {}}
        for stat in arms:
            hs = hull(frontier(scores[stat], test.labels, test.episodes))
            deltas = [
                precision_at_coverage(hs, c) - precision_at_coverage(hb, c)
                for c in band
            ]
            mid = float(np.median(band))
            ci = cluster_bootstrap_delta(
                base, scores[stat], test.labels, test.episodes, test.cluster,
                coverage=mid, n_boot=n_boot, seed=seed,
            )
            fold_out["arms"][stat] = {
                "band_deltas": [None if not np.isfinite(d) else float(d)
                                for d in deltas],
                "mean_band_delta": float(np.nanmean(deltas)),
                "worst_band_delta": float(np.nanmin(deltas)),
                "at_median_coverage": ci,
                "channel_auc": aucs[stat],
            }
            print(
                f"  {held_out:8s} {stat:>9s}: AUC {aucs[stat]:.4f}  "
                f"band mean {np.nanmean(deltas):+.4f} "
                f"worst {np.nanmin(deltas):+.4f}  "
                f"@{mid:.2f} {ci['delta']:+.4f} "
                f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]",
                flush=True,
            )
        out["folds"][held_out] = fold_out
    return out


def _coverage_band(h: np.ndarray, *, n: int = 8) -> np.ndarray:
    """Coverages at which every arm is compared.

    A band, not a point. The audit's v3 "win" was +0.0023 at coverage 0.65 and
    NEGATIVE at 0.45, 0.50, 0.55, 0.75 and 0.80 -- a single matched-coverage
    point can be chosen, after the fact, to say almost anything.
    """
    if not len(h):
        return np.zeros(0)
    lo = max(float(h["coverage"][0]), 0.40)
    hi = min(float(h["coverage"][-1]), 0.90)
    if hi <= lo:
        return np.array([float(np.median(h["coverage"]))])
    return np.linspace(lo, hi, n)


def _auc(scores, labels) -> float:
    """Rank AUC of a raw channel: same vs different, ties at 0.5."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if not y.any() or y.all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks within ties, so a quantised channel is not flattered.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# Experiment 3: cohort-normalised appearance
# --------------------------------------------------------------------------

COHORT_KINDS = ("zscore", "rank", "margin")

#: Field standard deviations below this are not information, they are two
#: candidates that happened to land close. Flooring stops a degenerate
#: 2-candidate field producing an enormous z.
_STD_FLOOR = 1e-3


def cohort_transform(values, episodes, kind: str) -> np.ndarray:
    """Re-express each candidate's appearance score relative to its own field.

    A global calibrator maps a cosine of 0.62 to the same evidence whether the
    query's other candidates sat at 0.61 or at 0.20. The informativeness of a
    similarity is a property of the impostor population -- the same argument
    `evidence.impostor_field_llr` makes -- so the field-relative form may carry
    evidence the absolute one averages away.

    Every kind is computed from the candidates of the DECISION ITSELF, which are
    all available when that decision is made, so nothing here is leakage. The
    field-prefix test asserts it.

    NaNs (missing appearance) stay NaN and are excluded from the field: an
    abstaining candidate must not shift the cohort it is not part of.
    """
    v = np.asarray(values, dtype=np.float64)
    ep = np.asarray(episodes)
    out = np.full(len(v), np.nan)
    order = np.argsort(ep, kind="mergesort")
    bounds = np.flatnonzero(np.diff(ep[order])) + 1
    for grp in np.split(order, bounds):
        vals = v[grp]
        ok = ~np.isnan(vals)
        if ok.sum() == 0:
            continue
        good = vals[ok]
        if kind == "zscore":
            sd = float(good.std())
            out[grp[ok]] = (good - good.mean()) / max(sd, _STD_FLOOR)
        elif kind == "rank":
            # Mid-rank, mapped to [0, 1]. A single-candidate field maps to 0.5:
            # with no alternative there is nothing to discriminate against, and
            # inventing evidence there is how a lone fragment gets confidently
            # mismerged (the same reason impostor_field_llr returns 0).
            if len(good) == 1:
                out[grp[ok]] = 0.5
            else:
                idx = np.argsort(np.argsort(good))
                out[grp[ok]] = idx / (len(good) - 1)
        elif kind == "margin":
            # Each candidate against the BEST OTHER candidate, so a row never
            # competes with itself -- the off-by-one that makes a raw
            # margin-over-runner-up always non-positive.
            if len(good) == 1:
                out[grp[ok]] = 0.0
            else:
                top2 = np.partition(good, -2)[-2:]
                best, second = float(top2[1]), float(top2[0])
                other_best = np.where(good >= best, second, best)
                out[grp[ok]] = good - other_best
        else:
            raise ValueError(f"unknown cohort kind {kind!r}")
    return out


def experiment_3(*, n_boot: int = 300, seed: int = 0) -> dict:
    """Is the body cosine's informativeness query-dependent?

    Read with the audit in mind: on the real substrate candidate recall was 1.00
    and ranking 7/8 top-1, so ranking is not the bottleneck there -- and every
    kind here is a within-field monotone transform, which barely changes ranking
    at all. This arm's only real lever is the THRESHOLD, and that is what the
    frontier measures.
    """
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer

    arms = ("body", *COHORT_KINDS)
    out = {"arms": list(arms), "folds": {}}
    for held_out in bt.MATCHES:
        test = fold(held_out)
        fits = [fold(m) for m in bt.MATCHES if m != held_out]

        scores, aucs = {}, {}
        for kind in arms:
            fit_mod = []
            for f in fits:
                g = _with_body(f, f.rows[:, 0] if kind == "body"
                               else cohort_transform(f.rows[:, 0], f.episodes, kind))
                fit_mod.append(g)
            cals, prior, w = refit_with_body(fit_mod, "body")
            rows = test.rows.copy()
            if kind != "body":
                rows[:, 0] = cohort_transform(test.rows[:, 0], test.episodes, kind)
            scores[kind] = LinearLLRScorer(cals, prior, w).score(rows, test.ctx)
            aucs[kind] = _auc(rows[:, 0], test.labels)

        out["folds"][held_out] = _compare_arms(
            test, scores, aucs, arms, base="body", n_boot=n_boot, seed=seed,
            label=held_out,
        )
    return out


def _with_body(f: FoldData, col) -> FoldData:
    """A copy of a fold with column 0 replaced. Cheap: only column 0 changes."""
    from dataclasses import replace as dc_replace

    rows = f.rows.copy()
    rows[:, 0] = col
    return dc_replace(f, rows=rows)


def _compare_arms(test, scores, aucs, arms, *, base, n_boot, seed, label) -> dict:
    """Band-wide frontier comparison of every arm against `base`."""
    hb = hull(frontier(scores[base], test.labels, test.episodes))
    band = _coverage_band(hb)
    fold_out = {**test.summary(), "band": band.tolist(), "arms": {}}
    mid = float(np.median(band))
    for name in arms:
        hs = hull(frontier(scores[name], test.labels, test.episodes))
        deltas = [
            precision_at_coverage(hs, c) - precision_at_coverage(hb, c) for c in band
        ]
        ci = cluster_bootstrap_delta(
            scores[base], scores[name], test.labels, test.episodes, test.cluster,
            coverage=mid, n_boot=n_boot, seed=seed,
        )
        fold_out["arms"][name] = {
            "band_deltas": [None if not np.isfinite(d) else float(d) for d in deltas],
            "mean_band_delta": float(np.nanmean(deltas)),
            "worst_band_delta": float(np.nanmin(deltas)),
            "at_median_coverage": ci,
            "channel_auc": aucs[name],
        }
        print(
            f"  {label:8s} {name:>9s}: AUC {aucs[name]:.4f}  "
            f"band mean {np.nanmean(deltas):+.4f} worst {np.nanmin(deltas):+.4f}  "
            f"@{mid:.2f} {ci['delta']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]",
            flush=True,
        )
    return fold_out


# --------------------------------------------------------------------------
# Experiment 1: learned edge scorer
# --------------------------------------------------------------------------


def experiment_1(*, n_boot: int = 300, seed: int = 0, shuffle_null: bool = True) -> dict:
    """Does a learned combiner beat the weighted sum of calibrated LLRs?

    Arms, all through the identical decision rule:
      `linear`      the incumbent (control)
      `zero_hidden` an MLP with no hidden layer -- the MACHINERY check. If this
                    does not land on the linear frontier, nothing else here is
                    interpretable, because the two differ only in estimator.
      `raw_bce` / `on_llr_bce`         the primary arms
      `raw_softmax` / `on_llr_softmax` the same under the scale-free objective
      `on_llr_shuffled`                permutation null: the context features
                    label-shuffled. If THIS wins, the machinery is the finding.

    Each arm is the mean logit over 5 seeds, pre-registered, so seed variance
    is inside the reported interval rather than beside it.
    """
    from matchlab_train.experiments import learned_scorer as ls
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer

    out = {"folds": {}, "hidden": list(ls.HIDDEN), "seeds": list(ls.SEEDS)}
    for held_out in bt.MATCHES:
        test = fold(held_out)
        fits = [fold(m) for m in bt.MATCHES if m != held_out]
        cals, prior, w = refit_with_body(fits, "body")
        neutral = ls.neutral_cosine(cals["body"])

        base = LinearLLRScorer(cals, prior, w).score(test.rows, test.ctx)
        scores = {"linear": base}
        # The incumbent's FUSED score, because every learned arm below is scored
        # fused. Comparing a learned fused score against the incumbent's raw
        # appearance CHANNEL would flatter the arm by ~0.03 AUC and say nothing.
        aucs = {"linear": _auc(base, test.labels)}

        # Inner split for early stopping: one of the four fit halves. Never the
        # held-out match -- that would be tuning on the test fold.
        fit_x, fit_y, fit_ep, fit_half = {}, None, None, None
        for kind in ("raw", "on_llr"):
            fit_x[kind] = np.concatenate(
                [ls.build_features(f, cals, prior, kind, neutral=neutral)
                 for f in fits]
            )
        fit_y = np.concatenate([f.labels for f in fits])
        eps, off = [], 0
        for f in fits:
            eps.append(f.episodes + off)
            off += int(f.episodes.max()) + 1
        fit_ep = np.concatenate(eps)
        fit_half = np.concatenate(
            [f.half + 2 * k for k, f in enumerate(fits)]
        )
        inner_val = fit_half == fit_half.max()

        for kind in ("raw", "on_llr"):
            test_x = ls.build_features(test, cals, prior, kind, neutral=neutral)
            mu = fit_x[kind][~inner_val].mean(axis=0)
            sd = np.maximum(fit_x[kind][~inner_val].std(axis=0), 1e-9)
            xf = (fit_x[kind][~inner_val] - mu) / sd
            xv = (fit_x[kind][inner_val] - mu) / sd
            xt = (test_x - mu) / sd

            for objective in ("bce", "softmax"):
                per_seed = [
                    ls.score_with(
                        ls.train(xf, fit_y[~inner_val], fit_ep[~inner_val],
                                 xv, fit_y[inner_val],
                                 hidden=ls.HIDDEN, seed=s, objective=objective),
                        xt,
                    )
                    for s in ls.SEEDS
                ]
                scores[f"{kind}_{objective}"] = np.mean(per_seed, axis=0)
                aucs[f"{kind}_{objective}"] = _auc(
                    scores[f"{kind}_{objective}"], test.labels
                )

            if kind == "on_llr":
                # Machinery check: no hidden layer = a linear model on the same
                # features, fitted by the same optimiser.
                per_seed = [
                    ls.score_with(
                        ls.train(xf, fit_y[~inner_val], fit_ep[~inner_val],
                                 xv, fit_y[inner_val],
                                 hidden=(), seed=s, objective="bce"),
                        xt,
                    )
                    for s in ls.SEEDS
                ]
                scores["zero_hidden"] = np.mean(per_seed, axis=0)
                aucs["zero_hidden"] = _auc(scores["zero_hidden"], test.labels)

                if shuffle_null:
                    rng = np.random.default_rng(1234)
                    n_ctx = 5  # the context block, last five columns
                    xf_s, xv_s, xt_s = xf.copy(), xv.copy(), xt.copy()
                    for arr in (xf_s, xv_s, xt_s):
                        arr[:, -n_ctx:] = arr[rng.permutation(len(arr))][:, -n_ctx:]
                    per_seed = [
                        ls.score_with(
                            ls.train(xf_s, fit_y[~inner_val], fit_ep[~inner_val],
                                     xv_s, fit_y[inner_val],
                                     hidden=ls.HIDDEN, seed=s, objective="bce"),
                            xt_s,
                        )
                        for s in ls.SEEDS
                    ]
                    scores["on_llr_shuffled"] = np.mean(per_seed, axis=0)
                    aucs["on_llr_shuffled"] = _auc(
                        scores["on_llr_shuffled"], test.labels
                    )

        arms = tuple(scores)
        out["folds"][held_out] = _compare_arms(
            test, scores, aucs, arms, base="linear", n_boot=n_boot, seed=seed,
            label=held_out,
        )
    return out


# --------------------------------------------------------------------------
# Experiment 2: trajectory-sequence motion arm
# --------------------------------------------------------------------------


def _traj_inputs(f: FoldData, *, mirror_h2: bool):
    """(features, log dt, dx, dy, usable) for one fold.

    `mirror_h2` canonicalises attack direction. FOOTPASS flips sides at
    half-time and these coordinates are absolute, so a model fitted across
    halves otherwise learns a side-specific drift and is then penalised for it
    -- the same flip that put occupancy's cross-half AUC at 0.357 unmirrored.
    """
    from matchlab_train.experiments.trajectory_motion import tail_features

    tails = f.tails[f.tail_of].copy()
    mask = f.tail_mask[f.tail_of]
    dx, dy = f.rows[:, 3].copy(), f.rows[:, 4].copy()
    if mirror_h2:
        h2 = f.half == 1
        tails[h2, :, 1] = 1.0 - tails[h2, :, 1]
        tails[h2, :, 2] = 1.0 - tails[h2, :, 2]
        dx[h2] = -dx[h2]
        dy[h2] = -dy[h2]
    gap = f.rows[:, 2]
    # dt <= 0 abstains outright: the incumbent's sigma floor makes its output
    # there an artifact, and the engine's score_channels gate does the same.
    usable = (gap > 0.0) & mask.any(axis=1) & np.isfinite(dx) & np.isfinite(dy)
    return tail_features(tails, mask), np.log1p(np.maximum(gap, 0.0)), dx, dy, usable


def experiment_2(*, n_boot: int = 300, seed: int = 0, mirror_h2: bool = True,
                 clamps=(0.0, 6.0)) -> dict:
    """Does a sequence model over the exit tail beat the static diffusion prior?

    Reported per gap bin, because the channel's measured value lives at short
    gaps and a whole-population average would hide a win or a loss there.
    Three things are reported for every arm, and a log-likelihood win alone is
    NOT a win: the round-2 lesson is that a better-fitting cue can still move no
    decisions.
    """
    from matchlab_core.reid.evidence import clip_transition

    from matchlab_train.experiments import trajectory_motion as tm
    from matchlab_train.experiments.edge_scorer import LinearLLRScorer

    out = {"mirror_h2": mirror_h2, "gap_edges": list(tm.GAP_EDGES),
           "clamps": list(clamps), "folds": {}}
    for held_out in bt.MATCHES:
        test = fold(held_out)
        fits = [fold(m) for m in bt.MATCHES if m != held_out]
        cals, prior, w = refit_with_body(fits, "body")

        fx, fld, fdx, fdy, fok = zip(
            *[_traj_inputs(f, mirror_h2=mirror_h2) for f in fits], strict=True
        )
        X = np.concatenate([a[b] for a, b in zip(fx, fok, strict=True)])
        LD = np.concatenate([a[b] for a, b in zip(fld, fok, strict=True)])
        DX = np.concatenate([a[b] for a, b in zip(fdx, fok, strict=True)])
        DY = np.concatenate([a[b] for a, b in zip(fdy, fok, strict=True)])
        Y = np.concatenate([f.labels[b] for f, b in zip(fits, fok, strict=True)])
        HALF = np.concatenate(
            [f.half[b] + 2 * k for k, (f, b) in enumerate(zip(fits, fok, strict=True))]
        )
        inner = HALF == HALF.max()

        tx, tld, tdx, tdy, tok = _traj_inputs(test, mirror_h2=mirror_h2)
        val = (X[inner & Y], LD[inner & Y], DX[inner & Y], DY[inner & Y])

        # Denominator arms: the shipped static one, and the dt-banded control
        # transition.py's OUTSTANDING ISSUE 1 says is needed to attribute a null.
        imp_static = (prior.impostor_x, prior.impostor_y)
        imp_bands = tm.banded_impostor(np.expm1(LD), DX, DY, Y)

        fold_out = {**test.summary(), "arms": {}, "per_bin": {}}
        scores, aucs = {}, {}
        scores["linear"] = LinearLLRScorer(cals, prior, w).score(test.rows, test.ctx)
        # The incumbent's TRANSITION CHANNEL, because every arm here is also
        # scored on its transition channel -- this A/B is channel-vs-channel by
        # construction. (Experiment 1 compares FUSED scores and uses the fused
        # AUC there. Mixing the two is what made the first draft's AUC column
        # unreadable, and it flattered the learned arms by ~0.015.)
        aucs["linear"] = _auc(
            np.where(tok, prior.llr(test.rows[:, 2], tdx, tdy), np.nan), test.labels
        )

        bins_test = tm.gap_bin(test.rows[:, 2])
        base_ll = tm.gaussian_logpdf(
            tdx, tdy, 0.0, 0.0,
            prior.x.at(test.rows[:, 2]), prior.y.at(test.rows[:, 2]), 0.0,
        )

        for objective in ("nce", "mle"):
            models = [
                tm.fit(X[~inner], LD[~inner], DX[~inner], DY[~inner], Y[~inner],
                       objective=objective, seed=s, impostor=imp_static, val=val)
                for s in tm.SEEDS
            ]
            params = [tm.predict_params(m, tx, tld) for m in models]
            mx, my, sx, sy, rho = (np.mean([p[k] for p in params], axis=0)
                                   for k in range(5))
            learned_ll = tm.gaussian_logpdf(tdx, tdy, mx, my, sx, sy, rho)

            for den_name, den in (("static", "static"), ("banded", "banded")):
                if den == "static":
                    ix = np.full(len(tdx), imp_static[0])
                    iy = np.full(len(tdx), imp_static[1])
                else:
                    ix = np.array([imp_bands[b][0] for b in bins_test])
                    iy = np.array([imp_bands[b][1] for b in bins_test])
                den_ll = tm.impostor_logpdf(tdx, tdy, ix, iy)
                raw = np.where(tok, learned_ll - den_ll, 0.0)
                base_raw = np.where(
                    tok, prior.llr(test.rows[:, 2], tdx, tdy), 0.0
                )
                for clamp in clamps:
                    col = np.where(
                        tok, np.asarray(clip_transition(raw, clamp)), 0.0
                    )
                    name = f"{objective}_{den_name}_c{clamp:g}"
                    scores[name] = _refit_with_transition(
                        fits, test, cals, prior, col, mirror_h2, objective,
                        den, imp_static, imp_bands, models, clamp,
                    )
                    aucs[name] = _auc(np.where(tok, raw, np.nan), test.labels)
                    # How much of the arm's improvement the clamp discards. A
                    # sharper numerator improves mostly the NEGATIVE side, and
                    # the shipped clamp of 0 flattens all of that to zero -- so
                    # without this fraction a null is unattributable between
                    # "no improvement" and "improvement not expressible".
                    fold_out.setdefault("clamp_saturation", {})[name] = {
                        "arm_frac_negative": float(np.mean(raw[tok] < 0.0)),
                        "arm_frac_at_clamp": float(
                            np.mean(np.abs(col[tok]) >= 0.999 * max(clamp, 1e-9))
                        ) if clamp > 0 else float(np.mean(raw[tok] < 0.0)),
                        "incumbent_frac_negative": float(
                            np.mean(base_raw[tok] < 0.0)
                        ),
                    }

            # The direct question, per bin: does the sequence model predict
            # re-entry better at all? Same pairs only -- this is the numerator.
            for b in range(len(tm.GAP_EDGES) + 1):
                sel = tok & test.labels & (bins_test == b)
                if sel.sum() < 30:
                    continue
                fold_out["per_bin"].setdefault(str(b), {})[objective] = {
                    "n_same": int(sel.sum()),
                    "incumbent_loglik": float(base_ll[sel].mean()),
                    "learned_loglik": float(learned_ll[sel].mean()),
                    "delta_loglik": float((learned_ll[sel] - base_ll[sel]).mean()),
                    "median_abs_mu_m": float(np.median(np.hypot(mx[sel], my[sel]))),
                    "median_sigma_x_m": float(np.median(sx[sel])),
                    "median_sigma_y_m": float(np.median(sy[sel])),
                }

        arms = tuple(scores)
        merged = _compare_arms(test, scores, aucs, arms, base="linear",
                               n_boot=n_boot, seed=seed, label=held_out)
        merged["per_bin"] = fold_out["per_bin"]
        merged["clamp_saturation"] = fold_out.get("clamp_saturation", {})
        out["folds"][held_out] = merged
    return out


def _refit_with_transition(fits, test, cals, prior, test_col, mirror_h2,
                           objective, den, imp_static, imp_bands, models,
                           clamp=0.0):
    """Fused score with the transition column replaced and WEIGHTS REFITTED.

    Refitting matters: a learned channel is not on the incumbent channel's
    scale, and serving it under the incumbent's weight would measure the scale
    rather than the representation.
    """
    from matchlab_core.reid.evidence import clip_transition, fit_fusion_weights

    from matchlab_train.experiments import trajectory_motion as tm

    fit_llr, fit_y, eps, off = [], [], [], 0
    for f in fits:
        fx, fld, fdx, fdy, fok = _traj_inputs(f, mirror_h2=mirror_h2)
        params = [tm.predict_params(m, fx, fld) for m in models]
        mx, my, sx, sy, rho = (np.mean([p[k] for p in params], axis=0)
                               for k in range(5))
        ll = tm.gaussian_logpdf(fdx, fdy, mx, my, sx, sy, rho)
        b = tm.gap_bin(f.rows[:, 2])
        if den == "static":
            ix = np.full(len(fdx), imp_static[0])
            iy = np.full(len(fdx), imp_static[1])
        else:
            ix = np.array([imp_bands[k][0] for k in b])
            iy = np.array([imp_bands[k][1] for k in b])
        raw = np.where(fok, ll - tm.impostor_logpdf(fdx, fdy, ix, iy), 0.0)
        col = np.where(fok, np.asarray(clip_transition(raw, clamp)), 0.0)
        llr = bt.channel_llrs(f.rows, cals, prior)
        llr[:, 3] = col
        fit_llr.append(llr)
        fit_y.append(f.labels)
        eps.append(f.episodes + off)
        off += int(f.episodes.max()) + 1
    w = fit_fusion_weights(
        np.concatenate(fit_llr), np.concatenate(eps), np.concatenate(fit_y)
    )
    tl = bt.channel_llrs(test.rows, cals, prior)
    tl[:, 3] = test_col
    return tl @ w


# --------------------------------------------------------------------------
# Sequential threading frontier (the confirming metric)
# --------------------------------------------------------------------------

#: `thread_half` reloads its half from a pickle on every call, and the sweep
#: calls it once per threshold per arm. Memoised here rather than in the harness
#: so the harness's own behaviour is untouched.
_HALF_CACHE: dict[str, tuple] = {}


def _memoise_half_frames():
    original = bt.half_frames

    def cached(key):
        if key not in _HALF_CACHE:
            _HALF_CACHE[key] = original(key)
        return _HALF_CACHE[key]

    return original, cached


def threading_frontier(
    held_out: str, cals, prior, w, *, scorer=None,
    thresholds, min_margin: float = 0.5, pass2: bool = False,
    pass2_score: float = 4.0, corruption=None,
) -> np.ndarray:
    """(coverage, precision) traced by sweeping the pass-1 bar on real threading.

    This is the audit's metric and the spec's CONFIRMING one. Unlike the static
    pair frontier it can see thread poisoning: a wrong merge here corrupts the
    territory and appearance every later decision is judged against, which is
    invisible to any metric that scores pairs against oracle-grown threads.

    Decisions are sequential and path-dependent, so rows are not exchangeable
    and the resampling unit for this frontier is the HALF (n=6 across the
    study), not the episode.
    """
    original, cached = _memoise_half_frames()
    bt.half_frames = cached
    try:
        out = np.zeros(len(thresholds), dtype=_FRONTIER_DTYPE)
        for i, t in enumerate(thresholds):
            correct = wrong = need = 0
            for key in (f"{held_out}_H1", f"{held_out}_H2"):
                r = bt.thread_half(
                    key, cals, prior, w, min_score=float(t),
                    min_margin=min_margin, pass2=pass2, pass2_score=pass2_score,
                    corruption=corruption, scorer=scorer,
                )
                correct += r["correct"]
                wrong += r["wrong"]
                need += r["merges_needed"]
            out["threshold"][i] = t
            out["correct"][i] = correct
            out["wrong"][i] = wrong
            out["need"][i] = need
            out["coverage"][i] = correct / max(need, 1)
            out["precision"][i] = correct / max(correct + wrong, 1)
    finally:
        bt.half_frames = original
    return out


@dataclass
class CohortScorer:
    """Experiment 3's arm as a plug-in, so it can run the THREADING frontier.

    The static pair frontier scores against oracle-grown threads (purity 1.0);
    the threading frontier scores against threads the system built for itself
    and can therefore see poisoning. The two are known to disagree -- v3 was
    flat on the static frontier and +0.0023 at coverage 0.65 on the threading
    one -- so an arm's static null is only trustworthy if the screening metric
    is shown to track the confirming one on THIS kind of intervention.
    """

    cals: dict
    prior: object
    weights: object
    kind: str

    def score(self, rows: np.ndarray, ctx) -> np.ndarray:
        from matchlab_train.experiments.edge_scorer import LinearLLRScorer

        rows = np.asarray(rows, dtype=float).copy()
        rows[:, 0] = cohort_transform(rows[:, 0], ctx.episode, self.kind)
        return LinearLLRScorer(self.cals, self.prior, self.weights).score(rows, ctx)


def cross_check_metrics(*, kind: str = "margin") -> dict:
    """Does the static screening frontier agree with the threading frontier?

    Run on experiment 3's arm because it is a pure scorer -- it plugs into
    `thread_half` unchanged -- and because its static effect is large and
    varies in sign across folds, which is what makes agreement informative.
    """
    thresholds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    out = {"kind": kind, "folds": {}}
    for held_out in bt.MATCHES:
        fits = [fold(m) for m in bt.MATCHES if m != held_out]
        cals, prior, w = refit_with_body(fits, "body")
        # Fitted on the cohort-transformed fit rows, exactly as experiment 3.
        fit_mod = [
            _with_body(f, cohort_transform(f.rows[:, 0], f.episodes, kind))
            for f in fits
        ]
        c2, p2, w2 = refit_with_body(fit_mod, "body")

        fa = threading_frontier(held_out, cals, prior, w, thresholds=thresholds)
        fb = threading_frontier(
            held_out, c2, p2, w2, thresholds=thresholds,
            scorer=CohortScorer(c2, p2, w2, kind),
        )
        ha, hb = hull(fa), hull(fb)
        band = _coverage_band(ha)
        deltas = [
            precision_at_coverage(hb, c) - precision_at_coverage(ha, c) for c in band
        ]
        out["folds"][held_out] = {
            "band": band.tolist(),
            "threading_band_deltas": [
                None if not np.isfinite(d) else float(d) for d in deltas
            ],
            "threading_mean_band_delta": float(np.nanmean(deltas)),
        }
        print(
            f"  {held_out}: threading band mean {np.nanmean(deltas):+.4f} "
            f"worst {np.nanmin(deltas):+.4f}",
            flush=True,
        )
    return out


def reproduce_v3(*, clamp: float = 6.0, thresholds=None) -> dict:
    """Re-derive the audit's Phase C number on ITS OWN metric, and ship it.

    The audit reported the pooled LOSO pass-1 THREADING frontier (margin 0.5),
    v3 gap-binned weights against flat, at `TRANS_NEG_CLAMP=6` -- the clamp in
    force before clamp-0 was adopted the following day. Its published figure is
    "at coverage 0.649 flat interpolates to ~0.9825 precision vs binned 0.9849".

    Run at clamp 6 deliberately: baselining against clamp 0 would compare v3 to
    a model that did not exist when the figure was published.

    The whole coverage band is returned, not just the audit's point, because
    the point alone is what made the original claim look like a frontier win.
    """
    prev_clamp = bt.TRANS_NEG_CLAMP
    bt.TRANS_NEG_CLAMP = clamp
    thr = list(thresholds or [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    try:
        acc = {}
        for binned in (False, True):
            tot = np.zeros(len(thr), dtype=_FRONTIER_DTYPE)
            tot["threshold"] = thr
            for held in bt.MATCHES:
                others = [m for m in bt.MATCHES if m != held]
                prev = bt.WEIGHT_GAP_BINS
                bt.WEIGHT_GAP_BINS = (5.0, 20.0) if binned else ()
                try:
                    model = bt.fit_from(others)
                finally:
                    bt.WEIGHT_GAP_BINS = prev
                f = threading_frontier(held, *model, thresholds=thr)
                for col in ("correct", "wrong", "need"):
                    tot[col] += f[col]
            tot["coverage"] = tot["correct"] / np.maximum(tot["need"], 1)
            tot["precision"] = tot["correct"] / np.maximum(
                tot["correct"] + tot["wrong"], 1
            )
            acc["v3" if binned else "flat"] = tot
    finally:
        bt.TRANS_NEG_CLAMP = prev_clamp

    ha, hb = hull(acc["flat"]), hull(acc["v3"])
    band = np.round(np.arange(0.45, 0.85, 0.05), 2)
    deltas = {}
    for c in band:
        pa, pb = precision_at_coverage(ha, c), precision_at_coverage(hb, c)
        if np.isfinite(pa) and np.isfinite(pb):
            deltas[f"{c:.2f}"] = {
                "flat": float(pa), "v3": float(pb), "delta": float(pb - pa)
            }
    out = {
        "clamp": clamp,
        "metric": "pooled LOSO pass-1 threading frontier, margin 0.5",
        "audit_published": {"coverage": 0.649, "flat": 0.9825, "v3": 0.9849},
        "band": deltas,
        "points": {
            k: [
                {"threshold": float(r["threshold"]),
                 "coverage": float(r["coverage"]),
                 "precision": float(r["precision"]),
                 "correct": int(r["correct"]), "wrong": int(r["wrong"])}
                for r in v
            ]
            for k, v in acc.items()
        },
    }
    for c, d in deltas.items():
        print(f"  coverage {c}: flat {d['flat']:.4f} -> v3 {d['v3']:.4f} "
              f"  {d['delta']:+.4f}", flush=True)
    return out


def score_thresholds(scores, n: int = 14) -> np.ndarray:
    """Sweep grid drawn from an arm's OWN score distribution.

    A learned scorer's output is not in nats, so the incumbent's [0, 2, 4, 6]
    grid does not address it. Quantiles of the arm's own best-per-decision
    scores put the grid where its decisions actually are.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if not len(s):
        return np.linspace(0.0, 1.0, n)
    return np.unique(np.quantile(s, np.linspace(0.55, 0.999, n)))


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="mde",
                    choices=("mde", "calibrate", "appearance", "exp3", "exp1", "exp2", "crosscheck", "reproduce"))
    ap.add_argument("--max-gap-frames", type=int, default=30)
    ap.add_argument("--trans-neg-clamp", type=float, default=0.0)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/reports/2026-08-03-input-representations-mde.json"))
    args = ap.parse_args()

    bt.MAX_GAP_FRAMES = args.max_gap_frames
    bt.TRANS_NEG_CLAMP = args.trans_neg_clamp

    print(f"substrate: MAX_GAP_FRAMES={bt.MAX_GAP_FRAMES} COORDS={bt.COORDS} "
          f"TRANS_NEG_CLAMP={bt.TRANS_NEG_CLAMP}", flush=True)
    if args.stage in ("appearance", "exp3", "exp1", "exp2", "crosscheck",
                      "reproduce"):
        if args.stage == "reproduce":
            print("\n== reproduce the audit Phase C v3 figure ==", flush=True)
            res = reproduce_v3()
        elif args.stage == "crosscheck":
            print("\n== cross-check: static screening vs threading frontier ==",
                  flush=True)
            res = cross_check_metrics()
        elif args.stage == "exp2":
            print("\n== experiment 2: trajectory-sequence motion ==", flush=True)
            res = experiment_2(n_boot=args.n_boot)
        elif args.stage == "exp1":
            print("\n== experiment 1: learned edge scorer ==", flush=True)
            res = experiment_1(n_boot=args.n_boot)
        elif args.stage == "appearance":
            print("\n== experiments 0 & 4: set-to-set appearance and pooling ==",
                  flush=True)
            res = experiments_0_and_4(n_boot=args.n_boot)
        else:
            print("\n== experiment 3: cohort-normalised appearance ==", flush=True)
            res = experiment_3(n_boot=args.n_boot)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res["substrate"] = {"max_gap_frames": bt.MAX_GAP_FRAMES,
                            "coords": bt.COORDS,
                            "trans_neg_clamp": args.trans_neg_clamp,
                            "stage": args.stage}
        args.out.write_text(json.dumps(res, indent=2))
        print(f"\nwritten: {args.out}")
        return

    if args.stage == "calibrate":
        print("\n== instrument calibration: v3 gap-binned weights vs flat ==",
              flush=True)
        res = calibrate_instrument(n_boot=args.n_boot)
        print(f"\nmean v3 - flat delta:        {res['mean_delta']:+.4f}")
        print(f"max paired 95% CI width:     {res['paired_ci_width_max']:.4f}")
        print(f"resolves a v3-sized effect:  {res['resolves_v3_sized_effect']}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res["substrate"] = {"max_gap_frames": bt.MAX_GAP_FRAMES,
                            "coords": bt.COORDS,
                            "trans_neg_clamp": args.trans_neg_clamp,
                            "stage": args.stage}
        args.out.write_text(json.dumps(res, indent=2))
        print(f"\nwritten: {args.out}")
        return

    print("\n== pre-registered MDE (incumbent, before any arm) ==", flush=True)
    res = measure_mde(incumbent_scores, n_boot=args.n_boot)
    print(f"\nmax MDE (player-clustered 95% width): {res['mde_player_max']:.4f}")
    print(f"v3-sized effect to resolve:            {res['v3_effect_size']:.4f}")
    print(f"can resolve a v3-sized effect:         {res['can_resolve_v3_sized_effect']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
