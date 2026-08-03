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


def oracle_pairs_rich(frags, states, first_xy, *, half_id: int, frag_offset: int):
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
            block += 1
        # Field size is a property of the DECISION, so it is stamped on every
        # row of the block once the block is known -- not incremented per row,
        # which would leak the candidate's position in the iteration order into
        # a feature the arms are allowed to use.
        if block:
            blocks.append((len(rows) - block, block))
        key = int(pid[i])
        grown[key] = grown[key].merged_with(states[i]) if key in grown else states[i]
        grown_fp[key] = grown[key].footprint()
        grown_frags[key] = grown_frags.get(key, 0) + 1

    field = np.ones(len(rows), dtype=np.float64)
    for lo, size in blocks:
        field[lo:lo + size] = size

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


def load_fold(held_out: str, *, verbose: bool = True) -> FoldData:
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
        d = oracle_pairs_rich(frags, states, first_xy, half_id=hi, frag_offset=frag_off)
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


def fold(held_out: str) -> FoldData:
    """Held-out fold, memoised: assembling one costs a full FOOTPASS half load."""
    if held_out not in _FOLD_CACHE:
        print(f"assembling fold {held_out}", flush=True)
        _FOLD_CACHE[held_out] = load_fold(held_out)
    return _FOLD_CACHE[held_out]


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
    ap.add_argument("--stage", default="mde", choices=("mde", "calibrate"))
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
    if args.stage == "calibrate":
        print("\n== instrument calibration: v3 gap-binned weights vs flat ==",
              flush=True)
        res = calibrate_instrument(n_boot=args.n_boot)
        print(f"\nmean v3 - flat delta:        {res['mean_delta']:+.4f}")
        print(f"max paired 95% CI width:     {res['paired_ci_width_max']:.4f}")
        print(f"resolves a v3-sized effect:  {res['resolves_v3_sized_effect']}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
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
