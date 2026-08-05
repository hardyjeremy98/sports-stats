"""3a validation: attacking direction and half-boundary against FOOTPASS GT.

Protocol notes that matter more than the code
---------------------------------------------
* **Input is club labels, answer is pitch side.** FOOTPASS `COL.TEAM` is the
  pitch SIDE and does not flip at half-time (verified: side 0's GK has median
  x < 0.5 in 6/6 halves). Feeding TEAM to the estimator would make the flip
  undetectable. The estimator is fed CLUB labels -- what an appearance-based
  team stage emits -- resolved by `footpass_match_harness.club_of_side`.
* **GT direction** = goalkeeper (ROLE == 1) median x per (club, half) over ALL
  GT rows. **GT boundary** = the H1/H2 key split.
* **Observability discipline**: the estimator sees only rows with ROI_X
  non-NaN (58% of rows are off-camera). Everything else is oracle.
* **Unequal-phi concatenation is the primary protocol.** `score(t)` peaks at
  n/2 under the null and `min_half` narrows the window toward the centre, so
  with two equal halves butted together a constant "boundary = n/2" estimator
  scores near-perfectly. Truncating H1 to a fraction phi of its length sweeps
  the true boundary across the timeline and destroys that shortcut. The
  midpoint and permuted-series baselines are printed in the same table so the
  comparison cannot be skipped.
* **No half-time gap.** `footpass_match_harness.HALF_BREAK_FRAMES` inserts a
  15-minute void; finding a boundary inside an empty region is not the task.
  The gapped case is reported separately, labelled easy.

Comparison bar (prior art): SoccerNet game-clock OCR locates half-starts within
+-2 s for 90% of matches -- a cue needing a broadcast clock overlay, absent on
the target amateur footage. No published accuracy exists for estimating
direction from tracklets; the sn-gamestate GSR baseline uses the same
centroid-ordering cue but never faces a flip (30-second sequences).

Usage:
  uv run python -m matchlab_train.experiments.direction_eval
"""

from __future__ import annotations

import argparse

import numpy as np
from matchlab_core.formation.direction import (
    estimate_direction,
    evaluate_probes,
    probe_fn_from_arrays,
)

from matchlab_train.datasets.footpass import COL, load_half
from matchlab_train.experiments.footpass_match_harness import club_of_side
from matchlab_train.experiments.position_evidence import VAL_H5

GAMES = ["game_18", "game_24", "game_47"]
FPS = 25.0
PHIS = (0.25, 0.40, 0.55, 0.70, 0.85)


def gt_side_defends_left(rows: np.ndarray) -> dict[int, bool]:
    """side code -> defends the LEFT goal, from the GK's median x (all GT rows)."""
    out = {}
    for side in (0, 1):
        m = (rows[:, COL.TEAM] == side) & (rows[:, COL.ROLE] == 1)
        if not m.any():
            raise ValueError(f"no GK rows for side {side}")
        out[side] = float(np.nanmedian(rows[m, COL.X])) < 0.5
    return out


def assert_clubs_actually_swap_ends(h1: np.ndarray, h2: np.ndarray, clubs) -> None:
    """Provenance guard on `club_of_side`.

    It resolves the flip by `mean(votes_flip) >= 0.5`, so an exact tie declares
    a flip, and the entire occupancy baseline collapses if that bit is wrong
    for any game. Verify directly that each resolved CLUB's goalkeeper really
    does change ends between the halves.
    """
    for half_no, rows in ((1, h1), (2, h2)):
        sides = gt_side_defends_left(rows)
        for side, left in sides.items():
            club = clubs[(half_no, side)]
            if half_no == 1:
                assert_clubs_actually_swap_ends.first = getattr(
                    assert_clubs_actually_swap_ends, "first", {}
                )
                assert_clubs_actually_swap_ends.first[club] = left
            else:
                was = assert_clubs_actually_swap_ends.first.get(club)
                if was is not None and was == left:
                    raise AssertionError(
                        f"club {club} defends the same end in both halves -- "
                        "club_of_side resolved the flip bit wrongly"
                    )


def build_match(game: str, phi: float, *, gap: bool):
    """Concatenated observable arrays with CLUB labels, plus the GT boundary.

    H1 is truncated to `phi` of its length so the true boundary is not the
    midpoint. Returns (frames, xs, clubs, boundary, total_frames).
    """
    h1 = load_half(VAL_H5, f"{game}_H1")
    h2 = load_half(VAL_H5, f"{game}_H2")
    clubs = club_of_side(h1.rows, h2.rows)
    assert_clubs_actually_swap_ends(h1.rows, h2.rows, clubs)

    r1, r2 = h1.rows, h2.rows
    # FOOTPASS H2 frame indices CONTINUE the match timeline -- they do not
    # restart at zero (game_18_H1 ends at 75307, H2 starts at 75525). Adding
    # H1's span as an offset, as `footpass_match_harness.load_match` does,
    # double-counts it and opens a ~50-minute void between the halves. Rebase
    # each half on its own first frame instead.
    f1min = int(np.nanmin(r1[:, COL.FRAME]))
    f1max = int(np.nanmax(r1[:, COL.FRAME]))
    f2min = int(np.nanmin(r2[:, COL.FRAME]))
    cut = f1min + int((f1max - f1min) * phi)
    r1 = r1[r1[:, COL.FRAME] <= cut]

    break_frames = int(15 * 60 * FPS) if gap else 0
    cut0 = cut - f1min  # rebased end of the truncated first epoch
    boundary = cut0 + 1 + break_frames // 2

    parts = []
    for half_no, rows, shift in (
        (1, r1, -f1min),
        (2, r2, -f2min + cut0 + 1 + break_frames),
    ):
        obs = rows[~np.isnan(rows[:, COL.ROI_X])]
        side = obs[:, COL.TEAM].astype(np.int64)
        parts.append((
            obs[:, COL.FRAME].astype(np.int64) + shift,
            obs[:, COL.X].astype(np.float64),
            np.array([clubs[(half_no, int(s))] for s in side], dtype=np.int64),
        ))
    frames = np.concatenate([p[0] for p in parts])
    xs = np.concatenate([p[1] for p in parts])
    cl = np.concatenate([p[2] for p in parts])
    total = int(frames.max()) + 1
    return frames, xs, cl, boundary, total


def per_probe_accuracy_by_third(game: str) -> None:
    """Per-probe sign accuracy conditioned on where the play is.

    The pan bias is SELECTION TRUNCATION, not an additive offset: with play
    deep in one end, both teams' visible members sit at high x, so d shrinks or
    inverts, and the deep team's extreme-x players (GK, back line) are the ones
    cropped. That makes the error a function of play location -- correlated
    with the estimand -- so the mean accuracy understates the damage where it
    matters. Headline the worst third.
    """
    for half_no in (1, 2):
        half = load_half(VAL_H5, f"{game}_H{half_no}")
        rows = half.rows
        sides = gt_side_defends_left(rows)
        want = -1.0 if sides[0] else +1.0  # d = x0 - x1; side0 left => d < 0
        obs = rows[~np.isnan(rows[:, COL.ROI_X])]
        fr = obs[:, COL.FRAME].astype(np.int64)
        side = obs[:, COL.TEAM].astype(np.int64)
        x = obs[:, COL.X].astype(np.float64)

        frames = np.unique(fr)[::25]
        fn = probe_fn_from_arrays(fr, x, side, observable=np.ones(len(fr), bool))
        p = evaluate_probes(fn, frames, min_per_team=3)
        # play location proxy: mean x of ALL observable players at that frame
        loc = []
        for f in p.frames:
            m = fr == f
            loc.append(float(x[m].mean()) if m.any() else np.nan)
        loc = np.asarray(loc)
        thirds = np.digitize(loc, [1 / 3, 2 / 3])
        accs = []
        for t in (0, 1, 2):
            m = thirds == t
            accs.append(float((np.sign(p.d[m]) == want).mean()) if m.any() else np.nan)
        overall = float((np.sign(p.d) == want).mean())
        print(
            f"  {game}_H{half_no}: per-probe acc overall {overall:.3f} | "
            f"by play third {accs[0]:.3f} {accs[1]:.3f} {accs[2]:.3f} | "
            f"WORST {np.nanmin(accs):.3f}  (n={len(p.d)} probes)"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=float, default=30.0)
    ap.add_argument("--games", nargs="*", default=GAMES)
    args = ap.parse_args()

    print("== V1b: per-probe sign accuracy conditioned on play location ==")
    print("(pan bias is selection truncation, so the mean overstates; worst third governs)")
    for g in args.games:
        per_probe_accuracy_by_third(g)

    print()
    print("== V2: boundary error, UNEQUAL-phi protocol, NO half-time gap ==")
    print("phi truncates H1, sweeping the true boundary off the midpoint.")
    print(f"{'game':<9} {'phi':>5} {'true(s)':>8} {'est(s)':>8} {'err(s)':>8} "
          f"{'MIDerr':>8} {'z':>6} {'probes':>7}  reason")
    rows_out = []
    for g in args.games:
        for phi in PHIS:
            fr, xs, cl, b, total = build_match(g, phi, gap=False)
            fn = probe_fn_from_arrays(fr, xs, cl, observable=np.ones(len(fr), bool))
            est = estimate_direction(
                fn, total_frames=total, fps=FPS, coarse_seconds=args.coarse,
                min_half_seconds=8 * 60.0,
            )
            mid_err = abs(total // 2 - b) / FPS
            if est.ok:
                err = abs(est.boundary - b) / FPS
                rows_out.append((err, mid_err))
                print(f"{g:<9} {phi:>5.2f} {b/FPS:>8.0f} {est.boundary/FPS:>8.0f} "
                      f"{err:>8.1f} {mid_err:>8.1f} {est.z:>6.1f} {est.probes_used:>7}  ok")
            else:
                print(f"{g:<9} {phi:>5.2f} {b/FPS:>8.0f} {'--':>8} {'ABSTAIN':>8} "
                      f"{mid_err:>8.1f} {est.z:>6.1f} {est.probes_used:>7}  {est.reason}")

    if rows_out:
        e = np.array([r[0] for r in rows_out])
        m = np.array([r[1] for r in rows_out])
        print(f"\n  resolved {len(e)}/{len(args.games)*len(PHIS)}  "
              f"median |err| {np.median(e):.1f}s  max {e.max():.1f}s  "
              f"| midpoint baseline median {np.median(m):.1f}s")
        print(f"  within +-30 s: {(e <= 30).mean():.0%}   "
              f"(bar: SoccerNet clock OCR 90% within +-2 s, needs a clock overlay)")

    print()
    print("== V2b: equal halves WITH the 15-min gap -- the easy case, for contrast ==")
    for g in args.games:
        fr, xs, cl, b, total = build_match(g, 1.0, gap=True)
        fn = probe_fn_from_arrays(fr, xs, cl, observable=np.ones(len(fr), bool))
        est = estimate_direction(
            fn, total_frames=total, fps=FPS, coarse_seconds=args.coarse,
            min_half_seconds=8 * 60.0,
        )
        mid_err = abs(total // 2 - b) / FPS
        err = abs(est.boundary - b) / FPS if est.ok else float("nan")
        print(f"  {g}: err {err:.1f}s  (midpoint baseline {mid_err:.1f}s) "
              f"z={est.z:.1f} probes={est.probes_used} {est.reason}")

    print()
    print("== V3: probe-budget sweep (phi=0.55, no gap) ==")
    print("Report against probes actually taken; S0 must exceed the error")
    print("autocorrelation length or a whole cluster of probes goes wrong together.")
    print(f"{'S0(s)':>6} {'resolved':>9} {'med err(s)':>11} {'max err(s)':>11} {'med probes':>11}")
    for s0 in (5.0, 15.0, 30.0, 60.0, 120.0):
        errs, probes, res = [], [], 0
        for g in args.games:
            fr, xs, cl, b, total = build_match(g, 0.55, gap=False)
            fn = probe_fn_from_arrays(fr, xs, cl, observable=np.ones(len(fr), bool))
            est = estimate_direction(
                fn, total_frames=total, fps=FPS, coarse_seconds=s0,
                min_half_seconds=8 * 60.0,
            )
            probes.append(est.probes_used)
            if est.ok:
                res += 1
                errs.append(abs(est.boundary - b) / FPS)
        if errs:
            print(f"{s0:>6.0f} {res:>4}/{len(args.games):<4} {np.median(errs):>11.1f} "
                  f"{max(errs):>11.1f} {np.median(probes):>11.0f}")
        else:
            print(f"{s0:>6.0f} {res:>4}/{len(args.games):<4} {'--':>11} {'--':>11} "
                  f"{np.median(probes):>11.0f}")

    print()
    print("== V3b: error autocorrelation length of the per-probe sign ==")
    for g in args.games:
        half = load_half(VAL_H5, f"{g}_H1")
        rows = half.rows
        sides = gt_side_defends_left(rows)
        want = -1.0 if sides[0] else +1.0
        obs = rows[~np.isnan(rows[:, COL.ROI_X])]
        fr = obs[:, COL.FRAME].astype(np.int64)
        x = obs[:, COL.X].astype(np.float64)
        side = obs[:, COL.TEAM].astype(np.int64)
        frames = np.unique(fr)[::25]
        fn = probe_fn_from_arrays(fr, x, side, observable=np.ones(len(fr), bool))
        p = evaluate_probes(fn, frames, min_per_team=3)
        err = (np.sign(p.d) != want).astype(np.float64)
        err = err - err.mean()
        denom = float((err * err).sum())
        lags = [1, 5, 10, 20, 30, 60, 120]
        ac = [float((err[:-k] * err[k:]).sum()) / denom for k in lags]
        first_below = next((k for k, a in zip(lags, ac) if a < 0.1), None)
        print(f"  {g}_H1: autocorr at lags(s) {lags} = "
              f"{[round(a, 2) for a in ac]} -> drops below 0.1 at ~{first_below}s")

    print()
    print("== V4: team-label noise ==")
    print("(a) independent symmetric flips -- the EASY model: sign stays unbiased")
    print(f"{'p':>6} {'resolved':>9} {'med err(s)':>11} {'max err(s)':>11}")
    rng = np.random.default_rng(0)
    for p_flip in (0.0, 0.05, 0.10, 0.20, 0.35):
        errs, res = [], 0
        for g in args.games:
            fr, xs, cl, b, total = build_match(g, 0.55, gap=False)
            noisy = cl.copy()
            m = rng.random(len(noisy)) < p_flip
            noisy[m] = 1 - noisy[m]
            fn = probe_fn_from_arrays(fr, xs, noisy, observable=np.ones(len(fr), bool))
            est = estimate_direction(
                fn, total_frames=total, fps=FPS, coarse_seconds=args.coarse,
                min_half_seconds=8 * 60.0,
            )
            if est.ok:
                res += 1
                errs.append(abs(est.boundary - b) / FPS)
        print(f"{p_flip:>6.2f} {res:>4}/{len(args.games):<4} "
              f"{np.median(errs) if errs else float('nan'):>11.1f} "
              f"{max(errs) if errs else float('nan'):>11.1f}")

    print()
    print("(b) BURSTY/ASYMMETRIC -- all of one club absorbed by the other for a window.")
    print("    This OFFSETS d and can flip its sign; symmetric noise cannot.")
    print(f"{'win(min)':>9} {'resolved':>9} {'med err(s)':>11} {'max err(s)':>11}  worst outcome")
    for win_min in (0.0, 2.0, 5.0, 10.0, 20.0):
        errs, res, uncovered = [], 0, [0]
        for g in args.games:
            fr, xs, cl, b, total = build_match(g, 0.55, gap=False)
            noisy = cl.copy()
            w = int(win_min * 60 * FPS)
            start = b - w // 2  # straddling the boundary: the worst place for it
            m = (fr >= start) & (fr < start + w) & (noisy == 0)
            noisy[m] = 1
            fn = probe_fn_from_arrays(fr, xs, noisy, observable=np.ones(len(fr), bool))
            est = estimate_direction(
                fn, total_frames=total, fps=FPS, coarse_seconds=args.coarse,
                min_half_seconds=8 * 60.0,
            )
            if est.ok:
                res += 1
                errs.append(abs(est.boundary - b) / FPS)
                # The only fabrication that matters is an error the reported
                # confidence band does NOT cover -- that is what makes
                # `epoch_of_fragment` hand back a confident wrong epoch.
                uncovered[0] += int(abs(est.boundary - b) > est.ci_frames)
        worst = "abstained" if not errs else (
            "FABRICATED" if max(uncovered) > 0 else "error inside reported CI")
        print(f"{win_min:>9.0f} {res:>4}/{len(args.games):<4} "
              f"{np.median(errs) if errs else float('nan'):>11.1f} "
              f"{max(errs) if errs else float('nan'):>11.1f}  {worst}")

    print()
    print("== V1: absolute direction accuracy per (club, epoch), n=6 halves ==")
    ok = tot = 0
    for g in args.games:
        h1 = load_half(VAL_H5, f"{g}_H1")
        h2 = load_half(VAL_H5, f"{g}_H2")
        clubs = club_of_side(h1.rows, h2.rows)
        fr, xs, cl, b, total = build_match(g, 1.0, gap=False)
        fn = probe_fn_from_arrays(fr, xs, cl, observable=np.ones(len(fr), bool))
        est = estimate_direction(
            fn, total_frames=total, fps=FPS, coarse_seconds=args.coarse,
            min_half_seconds=8 * 60.0,
        )
        if not est.ok:
            print(f"  {g}: ABSTAIN ({est.reason})")
            continue
        for half_no, rows in ((1, h1.rows), (2, h2.rows)):
            sides = gt_side_defends_left(rows)
            epoch = 0 if half_no == 1 else 1
            for side, left in sides.items():
                club = clubs[(half_no, side)]
                got = est.attacks_positive_x(club, epoch)
                tot += 1
                ok += int(got is left)  # defends left => attacks +x
        print(f"  {g}: running {ok}/{tot}")
    print(f"  absolute direction correct {ok}/{tot} (n=6 halves, 2 clubs each)")


if __name__ == "__main__":
    main()
