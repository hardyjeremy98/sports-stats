"""3b headroom probe: is there room above the trivial baseline?

Cold review measured a two-float baseline -- per-unit mean (x, y), nearest role
anchor, no assignment, no covariance -- at 0.662 exact-role excluding GK. That
is 73% of the way from chance (0.091) to SoccerCPD's 0.865, and SoccerCPD has
10/10 visibility, known identity and minute-long segments, so it is a
full-information ceiling rather than a target. Before building an estimator,
establish the floor and the achievable ceiling under the ACTUAL broadcast
visibility on this substrate.

Protocol notes that matter more than the code
---------------------------------------------
* **Direction is ORACLE**, from FOOTPASS `COL.TEAM` (pitch side). 3a cannot run
  here: every H5 key is a single half, genuinely one epoch, so it correctly
  returns `single-epoch` and supplies no direction. The 3a->3b integration is
  unmeasured. Canonicalisation mirrors (x, y) -> (1-x, 1-y) for the side that
  attacks -x, which is a 180 rotation and so preserves handedness (LB stays LB).
* **Templates are fitted on TRAIN halves only**; VAL is untouched by fitting.
* **VAL contains exactly ONE formation** (GK LB LCB RCB LM RM AM LW RW CF RB);
  MCB and DM never appear. So a formation-restriction arm is unmeasurable here
  and is not run.
* **Headline excludes GK.** GK is observable only 9% of the time but trivially
  separable when seen, so including it inflates every arm.
* Two evaluation units: `player-half` (matches the cold-review baseline, so the
  numbers are comparable) and `fragment` (observable spans -- what re-ID
  actually gets, and therefore the realistic one).

Reading the numbers (2026-08-05, 6 random 24-half TRAIN subsets, fragment
unit, exGK; chance ~0.10):

    relative-norm / module    0.443   <- best DEPLOYABLE
    relative-norm / oracle    0.457   <- ceiling (reads off-camera GT)
    relative-norm / observed  0.409
    relative      / observed  0.393
    absolute      / observed  0.278

Match-clustered CIs (n=3 matches, t=4.303) are wide and mostly cross zero, so
the load-bearing evidence is SIGN CONSISTENCY across template subsets, which
was 6/6 for every delta below. Never quote n=5718 fragments as an inferential
n -- they come from 3 matches.

Two caveats that make these numbers OPTIMISTIC:
* `MIN_FRAMES` discards ~58% of observable spans -- the SHORT ones, where a
  mean position is least informative. Real re-ID sees worse.
* Absolute levels move up to 8 points with the number of TRAIN halves (more
  formations blur the role means), so `--train-halves` must be reported with
  any level. Deltas are far more stable than levels.

Usage:
  uv run python -m matchlab_train.experiments.role_headroom --train-halves 24
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from matchlab_core.formation import TrackSpan, estimate_team_centroids

from matchlab_train.datasets.footpass import (
    COL,
    half_keys,
    load_half,
    observable_spans,
)
from matchlab_train.experiments.position_evidence import TRAIN_H5, VAL_H5

FPS = 25.0
STRIDE = 25  # 1 Hz
MIN_FRAMES = 10  # sampled frames a unit needs to be scored
GK = 1


def canonical_xy(rows: np.ndarray) -> np.ndarray:
    """Absolute pitch coords, mirrored so every team attacks +x.

    Side 0 defends the left goal in 6/6 val halves (GK median x < 0.5), so side
    0 attacks +x and side 1 is mirrored by a 180 rotation.
    """
    xy = rows[:, [COL.X, COL.Y]].astype(np.float64).copy()
    flip = rows[:, COL.TEAM].astype(np.int64) == 1
    xy[flip] = 1.0 - xy[flip]
    return xy


def _centroid_lookup(rows: np.ndarray, arm: str) -> dict[tuple[int, int], np.ndarray]:
    """(team, frame) -> centroid, in the canonical (already mirrored) frame."""
    xy = canonical_xy(rows)
    fr = rows[:, COL.FRAME].astype(np.int64)
    team = rows[:, COL.TEAM].astype(np.int64)
    vis = ~np.isnan(rows[:, COL.ROI_X])
    out: dict[tuple[int, int], np.ndarray] = {}

    for t in (0, 1):
        m_t = team == t
        if arm == "oracle":
            sel = m_t  # all 11, including off-camera -- not deployable
        elif arm == "observed":
            sel = m_t & vis
        elif arm == "module":
            # The deployable imputing estimator. impute=True is NOT the module
            # default and the module has no callers, so it is passed explicitly.
            #
            # CRITICAL: one TrackSpan per CONTIGUOUS OBSERVABLE RUN, not one per
            # player. Keying by player_id gives 11 spans each running kickoff to
            # final whistle, so `_departures_and_returns` sees 11 starts all at
            # frame ~50 and 11 ends all at ~75300 -- no camera-exit brackets at
            # all, and imputation never fires. Measured with the per-player
            # form: `applied` was 0.000-0.003 on three of six val halves (i.e.
            # bit-identical to `observed`) and fired elsewhere only on
            # substitutions. Runs are also what re-ID actually holds.
            pid = rows[:, COL.PLAYER_ID].astype(np.int64)
            spans = []
            tid = 0
            for p in np.unique(pid[m_t]):
                idx = np.flatnonzero(m_t & vis & (pid == p))
                if len(idx) == 0:
                    continue
                idx = idx[np.argsort(fr[idx], kind="stable")]
                f_p, xy_p = fr[idx], xy[idx]
                # Break a run where the sampled gap exceeds two strides; rows
                # are already decimated to STRIDE, so consecutive samples sit
                # one stride apart.
                cut = np.flatnonzero(np.diff(f_p) > 2 * STRIDE) + 1
                for run in np.split(np.arange(len(f_p)), cut):
                    if len(run) < 2:
                        continue
                    tid += 1
                    spans.append(
                        TrackSpan(
                            track_id=tid, team=0,
                            frames=f_p[run].astype(np.int64),
                            xy=xy_p[run].astype(np.float64),
                        )
                    )
            if spans:
                series = estimate_team_centroids(spans, roster_size=11, impute=True)
                if 0 in series:
                    s = series[0]
                    for f, c in zip(s.frames, s.xy):
                        if np.isfinite(c).all():
                            out[(t, int(f))] = c
            continue
        else:
            raise ValueError(arm)

        f_sel, xy_sel = fr[sel], xy[sel]
        order = np.argsort(f_sel, kind="stable")
        f_sel, xy_sel = f_sel[order], xy_sel[order]
        uniq, start = np.unique(f_sel, return_index=True)
        bounds = np.append(start, len(f_sel))
        for i, f in enumerate(uniq):
            out[(t, int(f))] = xy_sel[bounds[i]:bounds[i + 1]].mean(axis=0)
    return out


def features(rows: np.ndarray, frame_arm: str, centroid_arm: str) -> np.ndarray:
    """Per-row 2-D feature in the requested coordinate frame; NaN where undefined."""
    xy = canonical_xy(rows)
    if frame_arm == "absolute":
        return xy

    cent = _centroid_lookup(rows, centroid_arm)
    fr = rows[:, COL.FRAME].astype(np.int64)
    team = rows[:, COL.TEAM].astype(np.int64)
    c = np.full_like(xy, np.nan)
    for i in range(len(xy)):
        got = cent.get((int(team[i]), int(fr[i])))
        if got is not None:
            c[i] = got
    rel = xy - c
    if frame_arm == "relative":
        return rel
    if frame_arm == "relative-norm":
        # EFPI's one substantive finding: unscaled matching against fixed
        # templates yields illogical labels, because team spread varies with
        # phase of play. Normalise each frame by the team's RMS radius.
        scale: dict[tuple[int, int], float] = {}
        key = list(zip(team.tolist(), fr.tolist()))
        acc: dict[tuple[int, int], list[float]] = defaultdict(list)
        vis = ~np.isnan(rows[:, COL.ROI_X])
        for i, k in enumerate(key):
            if vis[i] and np.isfinite(rel[i]).all():
                acc[k].append(float(rel[i] @ rel[i]))
        for k, v in acc.items():
            scale[k] = float(np.sqrt(np.mean(v))) if v else np.nan
        s = np.array([scale.get(k, np.nan) for k in key])
        with np.errstate(invalid="ignore", divide="ignore"):
            return rel / np.where(s[:, None] > 1e-6, s[:, None], np.nan)
    raise ValueError(frame_arm)


def fit_templates(
    keys: list[str], frame_arm: str, centroid_arm: str, *, h5, limit: int
) -> dict[int, np.ndarray]:
    """role -> mean feature, fitted on TRAIN only."""
    acc: dict[int, list[np.ndarray]] = defaultdict(list)
    for k in keys[:limit]:
        rows = load_half(h5, k).rows
        rows = rows[rows[:, COL.FRAME].astype(np.int64) % STRIDE == 0]
        feat = features(rows, frame_arm, centroid_arm)
        vis = ~np.isnan(rows[:, COL.ROI_X])
        role = rows[:, COL.ROLE].astype(np.int64)
        ok = vis & np.isfinite(feat).all(axis=1)
        for r in np.unique(role[ok]):
            acc[int(r)].append(feat[ok & (role == r)])
    return {
        r: np.concatenate(v).mean(axis=0) for r, v in acc.items() if len(v)
    }


def score(
    frame_arm: str, centroid_arm: str, visibility: str, unit: str,
    templates: dict[int, np.ndarray], *, games: list[str],
) -> tuple[float, float, int]:
    """(exact accuracy exGK, incl GK, n units) on VAL."""
    roles = sorted(templates)
    T = np.stack([templates[r] for r in roles])
    hit_ex = tot_ex = hit_all = tot_all = 0

    for key in games:
        half = load_half(VAL_H5, key)
        rows = half.rows
        rows = rows[rows[:, COL.FRAME].astype(np.int64) % STRIDE == 0]
        feat = features(rows, frame_arm, centroid_arm)
        vis = ~np.isnan(rows[:, COL.ROI_X])
        keep = np.isfinite(feat).all(axis=1)
        if visibility == "observable":
            keep &= vis
        pid = rows[:, COL.PLAYER_ID].astype(np.int64)
        role = rows[:, COL.ROLE].astype(np.int64)
        fr = rows[:, COL.FRAME].astype(np.int64)

        units: dict[tuple, list[int]] = defaultdict(list)
        if unit == "player-half":
            for i in np.flatnonzero(keep):
                units[(int(pid[i]),)].append(i)
        else:  # fragment: observable spans, what re-ID actually gets
            spans = {
                p: observable_spans(half, p, max_gap_frames=30)
                for p in half.player_ids
            }
            bounds = {p: np.array(s) for p, s in spans.items() if s}
            for i in np.flatnonzero(keep):
                b = bounds.get(int(pid[i]))
                if b is None:
                    continue
                j = np.flatnonzero((b[:, 0] <= fr[i]) & (fr[i] <= b[:, 1]))
                if len(j):
                    units[(int(pid[i]), int(j[0]))].append(i)

        for _u, idx in units.items():
            if len(idx) < MIN_FRAMES:
                continue
            mean = feat[idx].mean(axis=0)
            pred = roles[int(np.argmin(((T - mean) ** 2).sum(axis=1)))]
            truth = int(role[idx[0]])
            tot_all += 1
            hit_all += pred == truth
            if truth != GK:
                tot_ex += 1
                hit_ex += pred == truth
    return (
        hit_ex / max(tot_ex, 1), hit_all / max(tot_all, 1), tot_ex,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-halves", type=int, default=48)
    args = ap.parse_args()
    games = half_keys(VAL_H5)

    print("== 3b HEADROOM PROBE ==")
    print("Trivial estimator throughout: per-unit MEAN feature -> nearest role")
    print("anchor. No assignment, no covariance. Direction is ORACLE (3a cannot")
    print("run on single-half keys). Templates fitted on TRAIN only.")
    print("chance = 1/11 = 0.091 | SoccerCPD 0.865 is a FULL-INFORMATION")
    print("ceiling (10/10 visible, known identity, minute segments), not a bar.")
    print()
    print(f"{'frame':<15}{'centroid':<10}{'visible':<12}{'unit':<13}"
          f"{'exGK':>7}{'allroles':>9}{'n':>7}")

    combos = [
        ("absolute", "-", "all-11"), ("absolute", "-", "observable"),
        ("relative", "oracle", "all-11"), ("relative", "oracle", "observable"),
        ("relative", "module", "observable"),
        ("relative", "observed", "observable"),
        ("relative-norm", "oracle", "observable"),
        ("relative-norm", "module", "observable"),
        ("relative-norm", "observed", "observable"),
    ]
    tkeys = half_keys(TRAIN_H5)
    cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for frame_arm, cent, visb in combos:
        ck = (frame_arm, cent)
        if ck not in cache:
            cache[ck] = fit_templates(
                tkeys, frame_arm, cent if cent != "-" else "observed",
                h5=TRAIN_H5, limit=args.train_halves,
            )
        for unit in ("player-half", "fragment"):
            ex, alr, n = score(
                frame_arm, cent if cent != "-" else "observed", visb, unit,
                cache[ck], games=games,
            )
            print(f"{frame_arm:<15}{cent:<10}{visb:<12}{unit:<13}"
                  f"{ex:>7.3f}{alr:>9.3f}{n:>7}")

    print()
    print("Reading: 'all-11' rows are an ORACLE upper bound (they read")
    print("off-camera GT positions). 'observable' is the real setting. The gap")
    print("between them is the ceiling that better centroid/visibility handling")
    print("could buy; the gap above the best trivial row is 3b's headroom.")


if __name__ == "__main__":
    main()
