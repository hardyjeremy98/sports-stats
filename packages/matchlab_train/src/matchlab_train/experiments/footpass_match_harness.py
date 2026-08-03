"""Whole-match frozen-input re-ID harness on FOOTPASS.

The gap this closes: every existing FOOTPASS harness is per-HALF, so no
calibration or evaluation pair has ever crossed half-time -- which is exactly
where the occupancy channel currently fails (cross-half AUC 0.357 unmirrored
vs 0.820 mirrored, measured 2026-08-02). This module joins a game's two halves
into one fragment list with club-normalised team labels and a global frame
axis, freezes every channel input per fragment, and turns pair scoring into a
vectorised pass over cached arrays -- so changing a channel's internals (grid,
sigma, mirroring, fusion weights) re-scores a full match in seconds, not hours.

Frozen per-fragment inputs, one pickle per (game, fragmentation):
  - formation-relative xs/ys        (occupancy; footprints are REBUILT from
    these at eval time, so grid/sigma/mirror are free parameters, not baked in)
  - absolute entry/exit endpoints   (transition)
  - global start/end frames         (gap; H2 offset by H1's span + half-time)
  - prtreid fragment embedding      (body; from the existing appearance cache,
    remapped across fragmentations by bootstrap_threads.appearance_for)
  - GT shirt number + role          (jersey channel labels; FOOTPASS SHIRT col)
  - club                            (TEAM is PITCH SIDE and flips at half-time;
    resolved to a stable club id by majority vote over shared player ids)

Usage:
  uv run python -m matchlab_train.experiments.footpass_match_harness \
      --games game_18 game_24 game_47 --max-gap-frames 30 --mirror both

Prints per-channel AUCs split same-half / cross-half and writes the raw pair
table (one row per candidate pair, all channels) to
data/experiments/footpass-match/<game>-pairs-*.npz for offline recombination.

Measured verdict (2026-08-02, fused LOMO over the three val games): occupancy
served raw is dead weight over whole matches (AUC 0.854 vs 0.855 without it);
an EXPLICIT per-half flip -- rotate the later half's footprints 180 degrees,
the boundary is known on this substrate -- recovers +3.4 points (0.888,
cross-half 0.873). The flip is the adopted production direction but DEFERRED
pending half-boundary estimation in the pre-run calibration sequence.
min(JS, JS-mirrored) is NOT an acceptable stand-in: it collapses within-half
cross-flank impostors (LB<->RB / LW<->RW) below the genuine median. Details:
docs/implementation-status.md, 2026-08-02 occupancy entry.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matchlab_core.reid.occupancy import _blur

from matchlab_train.datasets.footpass import COL, load_half
from matchlab_train.experiments import bootstrap_threads as bt
from matchlab_train.experiments.position_evidence import VAL_H5, auc

FPS = 25.0
#: Nominal half-time break inserted between the halves on the global frame
#: axis. Only the gap channel sees it; its exact value is a convention.
HALF_BREAK_FRAMES = int(15 * 60 * FPS)

MATCH_CACHE = Path("data/experiments/footpass-match")


@dataclass
class MatchFragment:
    player_id: int
    club: int  # stable across halves (0/1); NOT the raw side-coded TEAM
    half: int
    start: int  # global frame axis (H2 offset past H1 + half-time break)
    end: int
    xs: np.ndarray  # formation-relative, observable frames only
    ys: np.ndarray
    entry_xy: np.ndarray  # absolute pitch coords
    exit_xy: np.ndarray
    embedding: np.ndarray | None
    shirt: int
    role: int


def club_of_side(h1_rows: np.ndarray, h2_rows: np.ndarray) -> dict[tuple[int, int], int]:
    """(half, side) -> club, by majority vote over players seen in both halves.

    FOOTPASS TEAM encodes pitch side, which flips at half-time -- pairing
    cross-half fragments on equal TEAM codes silently yields zero same-player
    pairs (found 2026-08-02). H1's side codes are adopted as the club ids.
    """
    side1 = {int(r[COL.PLAYER_ID]): int(r[COL.TEAM]) for r in h1_rows}
    side2 = {int(r[COL.PLAYER_ID]): int(r[COL.TEAM]) for r in h2_rows}
    votes_flip = [side1[p] != side2[p] for p in side1.keys() & side2.keys()]
    if not votes_flip:
        raise ValueError("no shared player ids between halves; cannot map clubs")
    flip = np.mean(votes_flip) >= 0.5
    return {
        (1, 0): 0, (1, 1): 1,
        (2, 0): 1 if flip else 0,
        (2, 1): 0 if flip else 1,
    }


def load_match(
    game: str, *, h5: Path = VAL_H5, max_gap_frames: int = 30
) -> list[MatchFragment]:
    """Both halves of `game` as one frozen fragment list. Cached.

    Fragmentation, endpoint tails and appearance reuse bootstrap_threads'
    per-half caches (and their appearance re-indexing), so the expensive parts
    are shared with every other harness rather than recomputed.
    """
    MATCH_CACHE.mkdir(parents=True, exist_ok=True)
    # v2: plain dicts, not dataclass instances -- a pickle of MatchFragment
    # written by a `python -m` run records the class under `__main__` and is
    # unloadable from any importing process.
    cache = MATCH_CACHE / f"{game}-g{max_gap_frames}-{bt.COORDS}-v2.pkl"
    if cache.exists():
        return [MatchFragment(**d) for d in pickle.loads(cache.read_bytes())]

    prev_gap = bt.MAX_GAP_FRAMES
    bt.MAX_GAP_FRAMES = max_gap_frames
    try:
        halves = {}
        for half_no in (1, 2):
            key = f"{game}_H{half_no}"
            frags, first_xy, last_xy, app = bt.half_frames(key)
            bt.check_appearance_alignment(frags, app)
            halves[half_no] = (frags, first_xy, last_xy, app, load_half(h5, key))
    finally:
        bt.MAX_GAP_FRAMES = prev_gap

    clubs = club_of_side(halves[1][4].rows, halves[2][4].rows)
    h1_span = int(np.nanmax(halves[1][4].rows[:, COL.FRAME])) + 1
    out: list[MatchFragment] = []
    for half_no in (1, 2):
        frags, first_xy, last_xy, app, half = halves[half_no]
        offset = 0 if half_no == 1 else h1_span + HALF_BREAK_FRAMES
        # SHIRT is player-constant within a half; one lookup per player.
        rows = half.rows
        shirt = {
            int(pid): int(rows[rows[:, COL.PLAYER_ID] == pid][0, COL.SHIRT])
            for pid in np.unique(rows[:, COL.PLAYER_ID]).astype(int)
        }
        for i, f in enumerate(frags):
            emb = app.get(i)
            out.append(
                MatchFragment(
                    player_id=int(f.player_id),
                    club=clubs[(half_no, int(f.team))],
                    half=half_no,
                    start=int(f.start) + offset,
                    end=int(f.end) + offset,
                    xs=np.asarray(f.xs, dtype=np.float64),
                    ys=np.asarray(f.ys, dtype=np.float64),
                    entry_xy=np.asarray(first_xy[i], dtype=np.float64),
                    exit_xy=np.asarray(last_xy[i], dtype=np.float64),
                    embedding=None if emb is None else np.asarray(emb, np.float64),
                    shirt=shirt.get(int(f.player_id), -1),
                    role=int(f.role),
                )
            )
    cache.write_bytes(pickle.dumps([vars(f) for f in out]))
    return out


def footprint_matrix(
    frags: list[MatchFragment], *, grid: tuple[int, int] = (12, 8), sigma: float = 1.0
) -> np.ndarray:
    """(N, gy*gx) row-stochastic footprints, rebuilt from raw coords.

    Rebuilding here (rather than freezing footprints) is what makes grid and
    sigma sweepable without touching the cache.
    """
    gx, gy = grid
    mats = np.zeros((len(frags), gy, gx), dtype=np.float64)
    for i, f in enumerate(frags):
        ix = np.clip((np.clip(f.xs, 0.0, 1.0) * gx).astype(int), 0, gx - 1)
        iy = np.clip((np.clip(f.ys, 0.0, 1.0) * gy).astype(int), 0, gy - 1)
        np.add.at(mats[i], (iy, ix), 1.0)
        mats[i] = _blur(mats[i], sigma)
        t = mats[i].sum()
        mats[i] = mats[i] / t if t > 0 else 1.0 / (gy * gx)
    return mats.reshape(len(frags), gy * gx)


def js_pairs(fp: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunk: int = 200_000):
    """Vectorised JS distance for footprint row pairs (ia[k], ib[k])."""
    out = np.empty(len(ia), dtype=np.float64)
    for lo in range(0, len(ia), chunk):
        p = fp[ia[lo : lo + chunk]]
        q = fp[ib[lo : lo + chunk]]
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kp = np.where(p > 0, p * np.log2(p / np.maximum(m, 1e-300)), 0.0).sum(1)
            kq = np.where(q > 0, q * np.log2(q / np.maximum(m, 1e-300)), 0.0).sum(1)
        out[lo : lo + chunk] = np.sqrt(np.clip(0.5 * kp + 0.5 * kq, 0.0, 1.0))
    return out


def pair_table(
    frags: list[MatchFragment],
    *,
    grid: tuple[int, int] = (12, 8),
    sigma: float = 1.0,
    max_gap_s: float | None = None,
) -> dict[str, np.ndarray]:
    """All ordered same-club non-overlapping candidate pairs, every channel raw.

    Columns (parallel arrays): same, cross_half, gap_s, body, occ, occ_mirror,
    jump, shirt_eq, ia, ib. Missing evidence is NaN (body without embeddings,
    shirt_eq when either shirt is unknown) -- neutral, never an exclusion.
    """
    n = len(frags)
    start = np.array([f.start for f in frags])
    end = np.array([f.end for f in frags])
    club = np.array([f.club for f in frags])
    half = np.array([f.half for f in frags])
    pid = np.array([f.player_id for f in frags])
    shirt = np.array([f.shirt for f in frags])

    ia, ib = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ia, ib = ia.ravel(), ib.ravel()
    keep = (club[ia] == club[ib]) & (end[ia] < start[ib])
    if max_gap_s is not None:
        keep &= (start[ib] - end[ia]) / FPS <= max_gap_s
    ia, ib = ia[keep], ib[keep]

    fp = footprint_matrix(frags, grid=grid, sigma=sigma)
    gy_gx = fp.shape[1]
    gx, gy = grid
    fp_m = np.flip(fp.reshape(-1, gy, gx), axis=(1, 2)).reshape(-1, gy_gx)

    emb = np.full((n, 1), np.nan)
    dims = {len(f.embedding) for f in frags if f.embedding is not None}
    if dims:
        (d,) = dims
        emb = np.full((n, d), np.nan)
        for i, f in enumerate(frags):
            if f.embedding is not None:
                v = f.embedding
                nv = np.linalg.norm(v)
                emb[i] = v / nv if nv > 1e-9 else v
    body = np.einsum("ij,ij->i", emb[ia], emb[ib])  # NaN propagates when missing

    jump = np.linalg.norm(
        np.stack([f.exit_xy for f in frags])[ia]
        - np.stack([f.entry_xy for f in frags])[ib],
        axis=1,
    )
    shirt_eq = np.where(
        (shirt[ia] >= 0) & (shirt[ib] >= 0),
        (shirt[ia] == shirt[ib]).astype(np.float64),
        np.nan,
    )
    return {
        "same": (pid[ia] == pid[ib]),
        "cross_half": (half[ia] != half[ib]),
        "gap_s": (start[ib] - end[ia]) / FPS,
        "body": body,
        "occ": js_pairs(fp, ia, ib),
        "occ_mirror": js_mixed(fp, fp_m, ia, ib),
        "jump": jump,
        "shirt_eq": shirt_eq,
        "ia": ia,
        "ib": ib,
    }


def js_mixed(fp: np.ndarray, fp_m: np.ndarray, ia, ib, chunk: int = 200_000):
    """JS distance between fp[ia] and MIRRORED fp[ib]."""
    out = np.empty(len(ia), dtype=np.float64)
    for lo in range(0, len(ia), chunk):
        p = fp[ia[lo : lo + chunk]]
        q = fp_m[ib[lo : lo + chunk]]
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kp = np.where(p > 0, p * np.log2(p / np.maximum(m, 1e-300)), 0.0).sum(1)
            kq = np.where(q > 0, q * np.log2(q / np.maximum(m, 1e-300)), 0.0).sum(1)
        out[lo : lo + chunk] = np.sqrt(np.clip(0.5 * kp + 0.5 * kq, 0.0, 1.0))
    return out


def channel_aucs(t: dict[str, np.ndarray], *, mirror: str = "off") -> dict[str, dict]:
    """AUC per channel, overall / same-half / cross-half.

    Convention matches position_evidence.auc: scores are HIGHER-means-more-
    DIFFERENT, so body cosine is negated. `mirror`: "off" scores occ raw;
    "cross" mirrors the later fragment only across halves; "min" takes
    min(raw, mirrored) everywhere.
    """
    if mirror == "off":
        occ = t["occ"]
    elif mirror == "cross":
        occ = np.where(t["cross_half"], t["occ_mirror"], t["occ"])
    elif mirror == "min":
        occ = np.minimum(t["occ"], t["occ_mirror"])
    else:
        raise ValueError(mirror)
    chans = {"body": -t["body"], "occ": occ, "jump": t["jump"], "gap": t["gap_s"]}
    strata = {
        "all": np.ones(len(occ), bool),
        "same_half": ~t["cross_half"],
        "cross_half": t["cross_half"],
    }
    out: dict[str, dict] = {}
    for name, score in chans.items():
        ok = ~np.isnan(score)
        out[name] = {
            s: auc(score[ok & m & t["same"]], score[ok & m & ~t["same"]])
            for s, m in strata.items()
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["game_18"])
    ap.add_argument("--max-gap-frames", type=int, default=30)
    ap.add_argument("--grid", default="12x8")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--max-gap-s", type=float, default=None)
    ap.add_argument("--mirror", default="both", choices=["off", "cross", "min", "both"])
    ap.add_argument("--save-pairs", action="store_true")
    args = ap.parse_args()
    gx, gy = (int(v) for v in args.grid.split("x"))

    for game in args.games:
        frags = load_match(game, max_gap_frames=args.max_gap_frames)
        n_emb = sum(f.embedding is not None for f in frags)
        print(f"\n{game}: {len(frags)} fragments ({n_emb} with embeddings)")
        t = pair_table(frags, grid=(gx, gy), sigma=args.sigma, max_gap_s=args.max_gap_s)
        ns, nd = int(t["same"].sum()), int((~t["same"]).sum())
        nc = int(t["cross_half"].sum())
        print(f"  pairs: {ns} same / {nd} diff ({nc} cross-half)")
        modes = ["off", "cross", "min"] if args.mirror == "both" else [args.mirror]
        for mode in modes:
            res = channel_aucs(t, mirror=mode)
            for ch, by in res.items():
                if ch == "occ" or mode == modes[0]:
                    line = "  ".join(f"{s}={v:.3f}" for s, v in by.items())
                    print(f"  {ch:12s} [mirror={mode if ch == 'occ' else '-'}]  {line}")
        if args.save_pairs:
            out = MATCH_CACHE / (
                f"{game}-pairs-g{args.max_gap_frames}-{args.grid}-s{args.sigma}.npz"
            )
            np.savez_compressed(out, **{k: v for k, v in t.items()})
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
