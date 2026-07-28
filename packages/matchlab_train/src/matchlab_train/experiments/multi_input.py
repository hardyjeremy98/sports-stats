"""Multi-input tracklet merging: every cue is an input, combined two ways.

Spec: docs/superpowers/specs/2026-07-27-multi-input-set-scoring.md

Merging is CHOOSING one candidate from a field using every cue at once. No cue
gates and no cue decides; each contributes evidence. Two combiners are compared
on identical episodes and identical features:

    sum    calibrated log-likelihood ratios added. Zero learned parameters.
           Correct if channels are conditionally independent -- they are not.
    model  permutation-equivariant set scorer over the candidate field, so it
           can express channel correlation, quality interactions, and (via the
           h - context term) how a candidate differs from the field it competes
           in -- none of which a per-pair sum can represent.

Channels are ablated in and out, body appearance included, so the value of each
input and of the combiner are separately attributable.

Splits are at MATCH level: both halves of a match share 22 players, so a
half-level split leaks identities.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matchlab_core.reid.episodes import build_episodes
from matchlab_core.reid.evidence import LLRCalibrator, saturate
from matchlab_core.reid.transition import TransitionPrior, displacement

from matchlab_train.datasets.footpass import COL, half_keys, load_half
from matchlab_train.experiments.position_evidence import (
    TRAIN_H5,
    VAL_H5,
    build_fragments,
    footprint_matrix,
    pairwise_js,
)

FPS = 25.0
MIN_FRAMES = 50
CACHE = Path("data/experiments/multi-input-cache")

# Channel name -> (raw feature index within a candidate row, lower_is_better)
CHANNELS: dict[str, tuple[int, bool]] = {
    "body": (0, False),        # appearance similarity, higher = more same
    "occupancy": (1, True),    # footprint JS distance, lower = more same
    "continuity": (2, True),   # extrapolation error, lower = more same
    "gap": (3, True),          # seconds apart, lower = more same
}
# `transition` is not in CHANNELS because it does not have a scalar raw score to
# hand to LLRCalibrator: the TransitionPrior IS its calibrator, taking (dt, dx,
# dy) jointly. Columns 4 and 5 carry the exit->entry displacement in metres.
DX, DY = 4, 5
TRANSITION = "transition"
N_RAW = 6
# Bumped whenever the raw column layout changes -- a stale pickle would
# otherwise be silently reinterpreted under the new schema.
CACHE_SCHEMA = "v2"
QUALITY_NAMES = (
    "log_field_size",
    "min_duration_s",
    "max_duration_s",
    "min_obs_frames",
    "query_duration_s",
)


@dataclass
class HalfData:
    """Episodes for one half, as flat arrays."""

    raw: np.ndarray            # (n_pairs, 4) channel raw scores
    qual: np.ndarray           # (n_pairs, len(QUALITY_NAMES))
    ep_index: np.ndarray       # (n_pairs,) which episode each row belongs to
    is_correct: np.ndarray     # (n_pairs,) bool
    n_episodes: int


def _endpoints(half, frags):
    rows = half.rows
    last_xy, last_v, first_xy = [], [], []
    for f in frags:
        sel = (rows[:, COL.PLAYER_ID] == f.player_id) & ~np.isnan(rows[:, COL.ROI_X])
        sub = rows[sel]
        sub = sub[(sub[:, COL.FRAME] >= f.start) & (sub[:, COL.FRAME] <= f.end)]
        sub = sub[np.argsort(sub[:, COL.FRAME])]
        last_xy.append(sub[-1, [COL.X, COL.Y]].astype(float))
        last_v.append(sub[-1, [COL.VX, COL.VY]].astype(float))
        first_xy.append(sub[0, [COL.X, COL.Y]].astype(float))
    return np.array(last_xy), np.array(last_v), np.array(first_xy)


def extract_half(
    path: Path, key: str, *, appearance: dict[int, np.ndarray] | None = None
) -> HalfData:
    """Build every episode's candidate rows for one half.

    `appearance` maps fragment index -> embedding. When absent the body channel
    is filled with NaN and is dropped by any arm that includes it (so a
    no-video run cannot silently score a zeroed body channel as evidence).
    """
    half = load_half(path, key)
    frags = build_fragments(half, min_frames=MIN_FRAMES, relative=True)
    if len(frags) < 2:
        return HalfData(np.zeros((0, N_RAW)), np.zeros((0, len(QUALITY_NAMES))),
                        np.zeros(0, int), np.zeros(0, bool), 0)
    mat = footprint_matrix(frags)
    last_xy, last_v, first_xy = _endpoints(half, frags)
    dur = np.array([len(f.xs) / FPS for f in frags])
    nobs = np.array([len(f.xs) for f in frags], dtype=float)

    emb = None
    if appearance is not None:
        emb = np.zeros((len(frags), next(iter(appearance.values())).shape[0]))
        for i, v in appearance.items():
            if i < len(frags):
                emb[i] = v
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-9)

    episodes = build_episodes(frags)
    raw, qual, ep_idx, corr = [], [], [], []
    for e_i, ep in enumerate(episodes):
        a = ep.query
        cand = np.asarray(ep.candidates)
        gap_s = (np.array([frags[c].start for c in cand]) - frags[a].end) / FPS
        occ = pairwise_js(mat, np.full(len(cand), a), cand)
        horizon = np.clip(gap_s, 0, 3.0)[:, None]
        pred = last_xy[a][None, :] + last_v[a][None, :] * horizon / FPS
        cont = np.linalg.norm(first_xy[cand] - pred, axis=1)
        body = (emb[cand] @ emb[a]) if emb is not None else np.full(len(cand), np.nan)

        dxm, dym = displacement(last_xy[a][None, :], first_xy[cand])
        raw.append(np.stack([body, occ, cont, gap_s, dxm, dym], axis=1))
        qual.append(
            np.stack(
                [
                    np.full(len(cand), np.log1p(len(cand))),
                    np.minimum(dur[cand], dur[a]),
                    np.maximum(dur[cand], dur[a]),
                    np.minimum(nobs[cand], nobs[a]),
                    np.full(len(cand), dur[a]),
                ],
                axis=1,
            )
        )
        ep_idx.append(np.full(len(cand), e_i))
        mask = np.zeros(len(cand), bool)
        for c in ep.correct:
            mask[list(cand).index(c)] = True
        corr.append(mask)

    return HalfData(
        raw=np.concatenate(raw),
        qual=np.concatenate(qual),
        ep_index=np.concatenate(ep_idx),
        is_correct=np.concatenate(corr),
        n_episodes=len(episodes),
    )


APPEARANCE_DIR = Path("data/experiments/footpass-appearance")


def load_appearance(key: str) -> dict[int, np.ndarray] | None:
    """Per-fragment PRTreID prototypes, if they have been extracted."""
    p = APPEARANCE_DIR / key / "fragment_embeddings.pkl"
    if not p.exists():
        return None
    return pickle.loads(p.read_bytes())


def load_block(path: Path, keys: list[str], *, tag: str) -> list[HalfData]:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for key in keys:
        app = load_appearance(key)
        # Cache key records whether body ID is present: a cache built without
        # embeddings must never be reused for a body arm.
        cache = CACHE / f"{tag}-{CACHE_SCHEMA}-{key}{'-body' if app else ''}.pkl"
        if cache.exists():
            out.append(pickle.loads(cache.read_bytes()))
            continue
        hd = extract_half(path, key, appearance=app)
        cache.write_bytes(pickle.dumps(hd))
        out.append(hd)
    return out


def fit_calibrators(blocks: list[HalfData]) -> dict:
    raw = np.concatenate([b.raw for b in blocks])
    corr = np.concatenate([b.is_correct for b in blocks])
    cals: dict = {}
    for name, (idx, _lower) in CHANNELS.items():
        col = raw[:, idx]
        ok = ~np.isnan(col)
        if not ok.any():
            continue
        cals[name] = LLRCalibrator.fit(col[ok & corr], col[ok & ~corr], max_bins=200)
    if raw.shape[1] > DY:
        cals[TRANSITION] = TransitionPrior.fit(
            raw[:, CHANNELS["gap"][0]], raw[:, DX], raw[:, DY], corr
        )
    return cals


def llr_matrix(hd: HalfData, cals: dict, names: list[str]) -> np.ndarray:
    cols = []
    for n in names:
        if n == TRANSITION:
            # Bounded to match the calibrated channels -- see channel_llrs in
            # bootstrap_threads.py for why the fusion layer owns this, not the prior.
            cols.append(
                np.asarray(
                    saturate(
                        cals[n].llr(hd.raw[:, CHANNELS["gap"][0]], hd.raw[:, DX], hd.raw[:, DY])
                    )
                )
            )
            continue
        idx = CHANNELS[n][0]
        cal = cals[n]
        col = hd.raw[:, idx]
        cols.append(np.array([cal.llr(float(v)) if not np.isnan(v) else 0.0 for v in col]))
    return np.stack(cols, axis=1) if cols else np.zeros((len(hd.raw), 0))


def rank1_from_scores(scores: np.ndarray, hd: HalfData) -> tuple[float, float, int]:
    """rank-1 and top-5: does the highest-scoring candidate belong to the same player?

    Episodes with no correct candidate are excluded from the rate (they are the
    abstention population and are reported separately).
    """
    hits = top5 = n = 0
    for e in range(hd.n_episodes):
        sel = hd.ep_index == e
        if not hd.is_correct[sel].any():
            continue
        n += 1
        order = np.argsort(-scores[sel])
        correct_sorted = hd.is_correct[sel][order]
        r = int(np.flatnonzero(correct_sorted)[0]) + 1
        hits += r == 1
        top5 += r <= 5
    return (hits / n if n else float("nan"), top5 / n if n else float("nan"), n)


def episode_rows(hd: HalfData, cals, names: list[str], *, max_neg: int | None = None,
                 rng=None):
    """Per-episode (features, correct_mask). Features = channel LLRs + quality.

    `max_neg` subsamples negatives for TRAINING only; evaluation always uses the
    full field, because the field size is part of the problem.
    """
    llr = llr_matrix(hd, cals, names)
    q = hd.qual
    feats = np.concatenate([llr, q], axis=1)
    for e in range(hd.n_episodes):
        sel = np.flatnonzero(hd.ep_index == e)
        corr = hd.is_correct[sel]
        if max_neg is not None and len(sel) > max_neg:
            pos = np.flatnonzero(corr)
            neg = np.flatnonzero(~corr)
            keep_neg = rng.choice(neg, min(max_neg, len(neg)), replace=False)
            idx = np.sort(np.concatenate([pos, keep_neg]))
            sel, corr = sel[idx], corr[idx]
        yield feats[sel], corr


def train_and_eval_model(data, cals, names, *, epochs, args, seed=0, device="cpu"):
    import torch
    from torch import optim

    from matchlab_train.models.set_scorer import SetScorer, listwise_loss, pad_batch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Field size must match evaluation: the model's context term is a pool over
    # the field, so training on 64-candidate fields and evaluating on ~550 shifts
    # its input distribution by an order of magnitude. Subsample EPISODES (cheap,
    # unbiased) rather than candidates within an episode.
    train = []
    for hd in data["T"]:
        rows = list(episode_rows(hd, cals, names, max_neg=args.max_neg, rng=rng))
        train.extend(rows)
    if args.max_train_episodes and len(train) > args.max_train_episodes:
        pick = rng.choice(len(train), args.max_train_episodes, replace=False)
        train = [train[i] for i in pick]
    if not train:
        return None
    d = train[0][0].shape[1]
    mu = np.mean(np.concatenate([f for f, _ in train]), axis=0)
    sd = np.std(np.concatenate([f for f, _ in train]), axis=0) + 1e-6

    model = SetScorer(d).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    bs = 8
    for ep in range(epochs):
        rng.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), bs):
            chunk = train[i : i + bs]
            x, m, c, has = pad_batch([(f - mu) / sd for f, _ in chunk],
                                     [c for _, c in chunk], device)
            loss = listwise_loss(model(x, m), c, has)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(chunk)
        print(f"    epoch {ep + 1}/{epochs}  loss {tot / len(train):.4f}")

    model.eval()
    r1s, t5s, ns = [], [], []
    with torch.no_grad():
        for hd in data["E"]:
            if not hd.n_episodes:
                continue
            hits = top5 = n = 0
            for f, c in episode_rows(hd, cals, names):
                if not c.any():
                    continue
                x, m, _, _ = pad_batch([(f - mu) / sd], [c], device)
                logits = model(x, m)[0, : len(f)].cpu().numpy()
                order = np.argsort(-logits)
                r = int(np.flatnonzero(c[order])[0]) + 1
                hits += r == 1
                top5 += r <= 5
                n += 1
            if n:
                r1s.append(hits / n)
                t5s.append(top5 / n)
                ns.append(n)
    w = np.array(ns, dtype=float)
    return {
        "rank1": float(np.average(r1s, weights=w)),
        "top5": float(np.average(t5s, weights=w)),
        "n": int(w.sum()),
        "per_half_rank1": [float(x) for x in r1s],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-halves", type=int, default=4)
    ap.add_argument("--train-halves", type=int, default=16)
    ap.add_argument("--dev-halves", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--max-neg", type=int, default=None,
                    help="cap negatives per training episode; default = full field")
    ap.add_argument("--max-train-episodes", type=int, default=4000)
    ap.add_argument("--out", type=Path,
                    default=Path("docs/reports/2026-07-27-multi-input-raw.json"))
    args = ap.parse_args()

    tk = half_keys(TRAIN_H5)
    n_c, n_t, n_d = args.calib_halves, args.train_halves, args.dev_halves
    blocks = {
        "C": tk[:n_c],
        "T": tk[n_c : n_c + n_t],
        "D": tk[n_c + n_t : n_c + n_t + n_d],
    }
    print(f"blocks  C={len(blocks['C'])}  T={len(blocks['T'])}  D={len(blocks['D'])}"
          f"  E={len(half_keys(VAL_H5))} (val)")

    data = {k: load_block(TRAIN_H5, v, tag="train") for k, v in blocks.items()}
    data["E"] = load_block(VAL_H5, half_keys(VAL_H5), tag="val")
    for k, v in data.items():
        print(f"  {k}: {sum(b.n_episodes for b in v)} episodes, "
              f"{sum(len(b.raw) for b in v)} candidate rows")

    cals = fit_calibrators(data["C"])
    print(f"calibrated channels: {sorted(cals)}")

    arms = [
        ("occupancy",),
        ("continuity",),
        ("gap",),
        ("occupancy", "continuity"),
        ("occupancy", "continuity", "gap"),
    ]
    if "body" in cals:
        arms = [
            ("body",),
            ("occupancy",),
            ("body", "occupancy"),
            ("body", "occupancy", "continuity"),
            ("body", "occupancy", "continuity", "gap"),
            ("occupancy", "continuity", "gap"),
        ]

    results = {"sum": {}, "model": {}}
    for arm in arms:
        names = [n for n in arm if n in cals]
        if not names:
            continue
        r1s, t5s, ns = [], [], []
        for hd in data["E"]:
            if not hd.n_episodes:
                continue
            s = llr_matrix(hd, cals, names).sum(axis=1)
            r1, t5, n = rank1_from_scores(s, hd)
            r1s.append(r1)
            t5s.append(t5)
            ns.append(n)
        w = np.array(ns, dtype=float)
        results["sum"][" + ".join(names)] = {
            "rank1": float(np.average(r1s, weights=w)),
            "top5": float(np.average(t5s, weights=w)),
            "n": int(w.sum()),
            "per_half_rank1": [float(x) for x in r1s],
        }

    print(f"\n{'SUM baseline':46s} {'rank-1':>8s} {'top-5':>8s}")
    for k, v in results["sum"].items():
        print(f"{k:46s} {100 * v['rank1']:7.2f}% {100 * v['top5']:7.2f}%")

    if not args.skip_model:
        print("\ntraining set scorer per arm...")
        for arm in arms:
            names = [n for n in arm if n in cals]
            if not names:
                continue
            label = " + ".join(names)
            print(f"  arm: {label}")
            r = train_and_eval_model(data, cals, names, epochs=args.epochs, args=args)
            if r:
                results["model"][label] = r
        print(f"\n{'arm':40s} {'sum r@1':>9s} {'model r@1':>10s} {'delta':>8s}")
        for k in results["sum"]:
            if k in results["model"]:
                a, b = results["sum"][k]["rank1"], results["model"][k]["rank1"]
                print(f"{k:40s} {100 * a:8.2f}% {100 * b:9.2f}% {100 * (b - a):+7.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
