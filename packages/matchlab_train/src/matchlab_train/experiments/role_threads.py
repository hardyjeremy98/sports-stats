"""3b on MERGED THREADS, with off-camera position interpolation.

Corrects two framing errors in `role_headroom`:

1. **Role feeds DST (action spotting), not re-ID.** DST's `teamvec` consumes
   `slot = LEFT_TO_RIGHT * 13 + (ROLE_ID - 1)`, 26 slots
   (`docs/reference/footpass-setup.md:159`). Re-ID uses occupancy, which is
   already built. So role is a consumer of re-ID output, and the correct unit
   is a MERGED THREAD -- minutes of evidence -- not a 7.6 s observable span.
   `role_headroom` measured raw spans to avoid a circularity that does not
   exist.

2. **Identity across gaps is available.** Inside a merged thread the player is
   known through off-camera gaps, so their position can be interpolated between
   last-seen and next-seen. The 2026-08-05 centroid work measured that at 94.2%
   of centroid error removed but rejected it as "uses identity, not
   deployable" -- true for a re-ID INPUT, false for a post-re-ID consumer.

Fixes a fit/serve flaw cold review found in `role_headroom`: templates were
always fitted on visible rows while the all-11 arm was served on every row, so
that arm ran off-distribution. Here the position arm is applied identically to
fitting and serving.

Arms
----
position : `observed`   on-camera samples only (what role_headroom used)
           `interp`     linear fill of off-camera gaps within the thread
           `interp-warp` concave fill -- players move fast early in a gap then
                        settle, the heuristic the TimeWarp work fitted
           `gt`         true off-camera positions (ceiling for interpolation)
threads  : `oracle` (all of a player's data in a half -- a perfect merge) and
           `frag-N` (thread split into N pieces, to show the cost of imperfect
           merging without needing a real re-ID run)
estimator: `mean-nn` (nearest template mean) and `gauss` (Mahalanobis under a
           per-role full covariance -- the Bialkowski/SoccerCPD cost)

Direction is ORACLE (`COL.TEAM`); in production 3a supplies the LEFT_TO_RIGHT
bit that the DST slot encoding needs.

Usage:
  uv run python -m matchlab_train.experiments.role_threads --train-halves 24
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from matchlab_core.formation import TimeWarp

from matchlab_train.datasets.footpass import COL, half_keys, load_half
from matchlab_train.experiments.position_evidence import TRAIN_H5, VAL_H5
from matchlab_train.experiments.role_headroom import (
    STRIDE,
    _centroid_lookup,
    canonical_xy,
)

GK = 1
N_ROLES = 13


def thread_positions(
    rows: np.ndarray, position_arm: str, warp: TimeWarp
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(player_id, frame, xy, role) after applying the position arm.

    `observed` keeps on-camera samples. `interp`/`interp-warp` additionally
    place the player during off-camera gaps by interpolating between the
    bracketing observed samples -- legitimate here because a merged thread
    carries identity through the gap. `gt` uses the true off-camera position.
    """
    xy = canonical_xy(rows)
    fr = rows[:, COL.FRAME].astype(np.int64)
    pid = rows[:, COL.PLAYER_ID].astype(np.int64)
    role = rows[:, COL.ROLE].astype(np.int64)
    vis = ~np.isnan(rows[:, COL.ROI_X])

    if position_arm == "gt":
        return pid, fr, xy, role
    if position_arm == "observed":
        return pid[vis], fr[vis], xy[vis], role[vis]

    out_p, out_f, out_xy, out_r = [], [], [], []
    for p in np.unique(pid):
        m = pid == p
        f_p, xy_p, v_p = fr[m], xy[m], vis[m]
        order = np.argsort(f_p, kind="stable")
        f_p, xy_p, v_p = f_p[order], xy_p[order], v_p[order]
        obs = np.flatnonzero(v_p)
        if len(obs) == 0:
            continue
        filled = xy_p.copy()
        keep = v_p.copy()
        for a, b in zip(obs[:-1], obs[1:]):
            if b - a <= 1:
                continue
            gap = np.arange(a + 1, b)
            tau = (f_p[gap] - f_p[a]) / max(f_p[b] - f_p[a], 1)
            wx, wy = warp(tau)
            filled[gap, 0] = xy_p[a, 0] + wx * (xy_p[b, 0] - xy_p[a, 0])
            filled[gap, 1] = xy_p[a, 1] + wy * (xy_p[b, 1] - xy_p[a, 1])
            keep[gap] = True
        out_p.append(np.full(int(keep.sum()), p, dtype=np.int64))
        out_f.append(f_p[keep])
        out_xy.append(filled[keep])
        out_r.append(np.full(int(keep.sum()), role[m][0], dtype=np.int64))
    if not out_p:
        return (np.empty(0, np.int64),) * 2 + (np.empty((0, 2)), np.empty(0, np.int64))
    return (
        np.concatenate(out_p), np.concatenate(out_f),
        np.concatenate(out_xy), np.concatenate(out_r),
    )


def half_features(
    h5, key: str, position_arm: str, centroid_arm: str, warp: TimeWarp
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(pid, frame, feature, role) in the spread-normalised relative frame."""
    rows = load_half(h5, key).rows
    rows = rows[rows[:, COL.FRAME].astype(np.int64) % STRIDE == 0]
    cent = _centroid_lookup(rows, centroid_arm)
    team_of = {
        int(p): int(rows[rows[:, COL.PLAYER_ID] == p][0, COL.TEAM])
        for p in np.unique(rows[:, COL.PLAYER_ID])
    }
    pid, fr, xy, role = thread_positions(rows, position_arm, warp)
    if len(pid) == 0:
        return pid, fr, np.empty((0, 2)), role

    c = np.full_like(xy, np.nan)
    for i in range(len(xy)):
        got = cent.get((team_of[int(pid[i])], int(fr[i])))
        if got is not None:
            c[i] = got
    rel = xy - c

    # Spread normalisation: team RMS radius per (team, frame). EFPI's one
    # substantive finding is that unscaled matching against fixed templates
    # yields illogical labels, because team spread varies with phase of play.
    acc: dict[tuple[int, int], list[float]] = defaultdict(list)
    keys = [(team_of[int(p)], int(f)) for p, f in zip(pid, fr)]
    for i, k in enumerate(keys):
        if np.isfinite(rel[i]).all():
            acc[k].append(float(rel[i] @ rel[i]))
    scale = {k: float(np.sqrt(np.mean(v))) for k, v in acc.items() if v}
    s = np.array([scale.get(k, np.nan) for k in keys])
    with np.errstate(invalid="ignore", divide="ignore"):
        feat = rel / np.where(s[:, None] > 1e-6, s[:, None], np.nan)
    ok = np.isfinite(feat).all(axis=1)
    return pid[ok], fr[ok], feat[ok], role[ok]


def fit(keys, position_arm, centroid_arm, warp, *, limit):
    """role -> (mean, inverse covariance). Fitted on TRAIN with the SAME arms."""
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for k in keys[:limit]:
        _, _, feat, role = half_features(TRAIN_H5, k, position_arm, centroid_arm, warp)
        for r in np.unique(role):
            acc[int(r)].append(feat[role == r])
    out = {}
    pooled = np.concatenate([np.concatenate(v) for v in acc.values()])
    pooled_cov = np.cov(pooled.T) + 1e-6 * np.eye(2)
    for r, v in acc.items():
        z = np.concatenate(v)
        if len(z) < 50:
            continue
        cov = np.cov(z.T) + 1e-6 * np.eye(2)
        # Shrink toward the pooled covariance: rare roles (MCB has 22 team-
        # halves in all of TRAIN) otherwise get an ill-conditioned estimate.
        cov = 0.7 * cov + 0.3 * pooled_cov
        out[r] = (z.mean(axis=0), np.linalg.inv(cov))
    return out


def evaluate(templates, position_arm, centroid_arm, warp, *, n_split: int, est: str):
    """Role accuracy over threads. n_split=1 is a perfect merge."""
    roles = sorted(templates)
    means = np.stack([templates[r][0] for r in roles])
    hit = tot = 0
    per_half = []
    for key in half_keys(VAL_H5):
        pid, fr, feat, role = half_features(
            VAL_H5, key, position_arm, centroid_arm, warp
        )
        h = t = 0
        for p in np.unique(pid):
            m = pid == p
            f_p, x_p, r_p = fr[m], feat[m], role[m][0]
            if r_p == GK or len(f_p) < 10:
                continue
            order = np.argsort(f_p, kind="stable")
            x_p = x_p[order]
            for chunk in np.array_split(x_p, n_split):
                if len(chunk) < 10:
                    continue
                mu = chunk.mean(axis=0)
                if est == "mean-nn":
                    j = int(np.argmin(((means - mu) ** 2).sum(axis=1)))
                else:  # gauss: Mahalanobis under each role's covariance
                    d = [
                        float((mu - templates[r][0]) @ templates[r][1] @ (mu - templates[r][0]))
                        for r in roles
                    ]
                    j = int(np.argmin(d))
                t += 1
                h += roles[j] == r_p
        hit += h
        tot += t
        per_half.append(h / max(t, 1))
    return hit / max(tot, 1), np.array(per_half), tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-halves", type=int, default=24)
    a = ap.parse_args()
    tkeys = half_keys(TRAIN_H5)
    warp_lin, warp_cc = TimeWarp(), TimeWarp(k_x=2.0, k_y=2.0)

    print("== 3b ON MERGED THREADS (role feeds DST action spotting) ==")
    print("Unit = merged thread. Direction ORACLE. Centroid = module (imputed).")
    print("Spread-normalised formation-relative frame. exGK. chance ~0.10.")
    print(f"Templates fitted on {a.train_halves} TRAIN halves with MATCHED arms.")
    print()
    print(f"{'position':<13}{'estimator':<11}{'merge':<12}{'acc':>7}{'per-match sd':>14}{'n':>7}")

    for position_arm, warp in (
        ("observed", warp_lin), ("interp", warp_lin),
        ("interp-warp", warp_cc), ("gt", warp_lin),
    ):
        tpl = {}
        for est in ("mean-nn", "gauss"):
            for n_split, label in ((1, "perfect"), (4, "4 pieces"), (16, "16 pieces")):
                if est not in tpl:
                    tpl[est] = fit(
                        tkeys, position_arm, "module", warp, limit=a.train_halves
                    )
                acc, per_half, n = evaluate(
                    tpl[est], position_arm, "module", warp,
                    n_split=n_split, est=est,
                )
                print(f"{position_arm:<13}{est:<11}{label:<12}{acc:>7.3f}"
                      f"{per_half.std():>14.3f}{n:>7}")


if __name__ == "__main__":
    main()
