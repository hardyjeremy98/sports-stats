"""What formation prediction gets WRONG, and whether the mistakes matter.

`formation_pred` measures how OFTEN the predicted 10-slot inventory is exactly
right. That number alone cannot say whether a miss is harmless or destructive,
and on this data the distinction is the whole story: 163 of 192 team-halves
(85%) sit in two classes that differ by a SINGLE role -- attacking midfielder
versus defensive midfielder -- and are otherwise identical, including their
line structure. Confusing those two leaves the defender/midfielder/attacker
counts exactly right and misplaces one player by one line. Confusing a
four-defender shape with a five-defender one does not.

So this harness reports three accuracies, not one:

    exact       the predicted inventory equals the true one
    shape       the (defenders, midfielders, attackers) counts are right,
                whatever the individual roles are
    line-safe   every predicted role is in the same LINE as the role it
                displaced (a strictly weaker condition than `exact`, and the
                one that matters to a downstream consumer reading position)

and then the failure MODE: the class confusion table, the role-substitution
census (which role gets hallucinated in place of which), and concrete
team-halves printed side by side.

Lines follow the FOOTPASS role ids (`datasets.footpass.ROLE_NAMES`):
defenders LB/LCB/MCB/RCB/RB, midfielders LM/RM/DM/AM, attackers LW/RW/CF.
The keeper is excluded everywhere, as in `formation_pred`.

Same protocol as `formation_pred`: FOOTPASS TRAIN, 5-fold held out by match,
every fit inside the training fold, oracle inputs throughout. Read its module
docstring for the caveats -- they all apply here unchanged.

Usage:
  uv run python -m matchlab_train.experiments.formation_errors --arm interp-warp
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from matchlab_train.datasets.footpass import ROLE_NAMES, half_keys
from matchlab_train.experiments.formation_pred import (
    DESCRIPTOR_ARMS,
    GK_PROXY,
    WARP_CONCAVE,
    WARP_LINEAR,
    fit_arms,
    match_of,
    predict,
    team_halves,
)
from matchlab_train.experiments.position_evidence import TRAIN_H5

DEFENDERS = frozenset({2, 3, 4, 5, 13})   # LB LCB MCB RCB RB
MIDFIELDERS = frozenset({6, 7, 8, 9})     # LM RM DM AM
ATTACKERS = frozenset({10, 11, 12})       # LW RW CF

LINE_OF = {r: "D" for r in DEFENDERS} | {r: "M" for r in MIDFIELDERS} | {
    r: "A" for r in ATTACKERS
}


def shape(inv: tuple[int, ...]) -> tuple[int, int, int]:
    """(defenders, midfielders, attackers) -- the familiar 4-3-3 reading."""
    c = Counter(LINE_OF.get(r, "?") for r in inv)
    return c["D"], c["M"], c["A"]


def fmt_shape(s: tuple[int, int, int]) -> str:
    return "-".join(str(v) for v in s)


def fmt_inv(inv: tuple[int, ...]) -> str:
    return "-".join(ROLE_NAMES.get(r, str(r)) for r in inv)


def diff(truth: tuple[int, ...], pred: tuple[int, ...]):
    """(roles the prediction missed, roles it invented). Sets, so a pure
    substitution shows as one of each and the size is the edit distance."""
    return tuple(sorted(set(truth) - set(pred))), tuple(sorted(set(pred) - set(truth)))


def collect(arm: str, position_arm: str, folds: int, seed: int):
    """Held-out (team-half, truth, prediction) for every team-half."""
    warp = WARP_CONCAVE if position_arm == "interp-warp" else WARP_LINEAR
    keys = half_keys(TRAIN_H5)
    cache = {k: team_halves(TRAIN_H5, k, position_arm, warp) for k in keys}
    matches = sorted({match_of(k) for k in keys})
    out = []
    for fold in np.array_split(np.random.default_rng(seed).permutation(len(matches)),
                               folds):
        test_m = {matches[i] for i in fold}
        train = [t for k in keys if match_of(k) not in test_m for t in cache[k]]
        fitted = fit_arms(train)
        for k in keys:
            if match_of(k) not in test_m:
                continue
            for th in cache[k]:
                if th.inventory:
                    out.append((th, th.inventory, predict(arm, th, fitted)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="interp-warp",
                    choices=["observed", "interp", "interp-warp", "gt"])
    ap.add_argument("--method", default="template-match", choices=DESCRIPTOR_ARMS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--gk-proxy", action="store_true")
    a = ap.parse_args()
    GK_PROXY[0] = a.gk_proxy

    print(f"collecting held-out predictions ({a.method}, position arm {a.arm}) ...",
          flush=True)
    rows = collect(a.method, a.arm, a.folds, a.seed)
    n = len(rows)

    exact = sum(t == p for _, t, p in rows)
    shape_ok = sum(shape(t) == shape(p) for _, t, p in rows)
    # line-safe: every substitution stays inside its own line, so no player
    # moves between defence, midfield and attack.
    line_safe = 0
    for _, t, p in rows:
        miss, extra = diff(t, p)
        if Counter(LINE_OF.get(r) for r in miss) == Counter(
            LINE_OF.get(r) for r in extra
        ):
            line_safe += 1

    print(f"\n== HOW OFTEN, AND HOW BADLY ({n} held-out team-halves) ==")
    print(f"  exact inventory      {exact/n:.3f}  ({exact}/{n})")
    print(f"  shape D-M-A correct  {shape_ok/n:.3f}  ({shape_ok}/{n})")
    print(f"  line-safe            {line_safe/n:.3f}  ({line_safe}/{n})")
    wrong = n - exact
    if wrong:
        print(f"\n  Of the {wrong} WRONG predictions, "
              f"{shape_ok - exact} ({(shape_ok-exact)/wrong:.1%}) still have the "
              f"right number of defenders, midfielders and attackers.")

    print("\n== EDIT DISTANCE: how many role slots are wrong at a time ==")
    dist = Counter(len(diff(t, p)[0]) for _, t, p in rows)
    for d in sorted(dist):
        tag = "correct" if d == 0 else f"{d} slot{'s' if d > 1 else ''} differ"
        print(f"  {tag:<16}{dist[d]:4d}  ({dist[d]/n:.3f})")

    print("\n== THE FAILURE MODE: which role replaces which ==")
    subs = Counter()
    for _, t, p in rows:
        miss, extra = diff(t, p)
        if len(miss) == 1 and len(extra) == 1:
            subs[(miss[0], extra[0])] += 1
    tot_sub = sum(subs.values())
    print(f"  single-slot substitutions: {tot_sub} of {wrong} errors "
          f"({tot_sub/max(wrong,1):.1%})")
    for (m, e), c in subs.most_common(12):
        same = "SAME LINE" if LINE_OF.get(m) == LINE_OF.get(e) else "crosses lines"
        print(f"    truth {ROLE_NAMES.get(m,m):<4} -> predicted "
              f"{ROLE_NAMES.get(e,e):<4}  {c:4d}   {same}")

    print("\n== CLASS CONFUSION (true shape -> predicted shape) ==")
    conf = Counter((fmt_shape(shape(t)), fmt_shape(shape(p))) for _, t, p in rows)
    by_true = defaultdict(int)
    for (tt, _), c in conf.items():
        by_true[tt] += c
    for tt in sorted(by_true, key=lambda k: -by_true[k]):
        got = {pp: c for (t2, pp), c in conf.items() if t2 == tt}
        right = got.get(tt, 0)
        detail = "  ".join(f"{k}:{v}" for k, v in
                           sorted(got.items(), key=lambda kv: -kv[1]) if k != tt)
        print(f"  true {tt:<7} n={by_true[tt]:4d}  correct {right/by_true[tt]:.3f}"
              f"   {'wrong -> ' + detail if detail else ''}")

    print(f"\n== EXAMPLES (first {a.examples} misses) ==")
    shown = 0
    for th, t, p in rows:
        if t == p or shown >= a.examples:
            continue
        miss, extra = diff(t, p)
        crosses = Counter(LINE_OF.get(r) for r in miss) != Counter(
            LINE_OF.get(r) for r in extra
        )
        print(f"\n  {th.key} team {th.team}   "
              f"{'LINE CHANGE' if crosses else 'same lines'}")
        print(f"    truth     {fmt_shape(shape(t))}  {fmt_inv(t)}")
        print(f"    predicted {fmt_shape(shape(p))}  {fmt_inv(p)}")
        print(f"    dropped {fmt_inv(miss) or '-'}   added {fmt_inv(extra) or '-'}")
        shown += 1

    print("\n== WHERE THE ERRORS LIVE: accuracy by true class prevalence ==")
    per_class = defaultdict(lambda: [0, 0])
    for _, t, p in rows:
        per_class[t][0] += int(t == p)
        per_class[t][1] += 1
    for inv, (h, c) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
        print(f"  n={c:4d}  acc {h/c:.3f}  {fmt_shape(shape(inv)):<7} {fmt_inv(inv)}")


if __name__ == "__main__":
    main()
