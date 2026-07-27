"""The six-arm ablation WITH body ID, sum vs set-scorer.

Only the three FOOTPASS val matches have video, so appearance exists only there.
Splits therefore rotate whole MATCHES with no overlap of any kind:

    fold 1: calibrate game_18 | train game_24 | eval game_47
    fold 2: calibrate game_24 | train game_47 | eval game_18
    fold 3: calibrate game_47 | train game_18 | eval game_24

Both halves of a match share 22 players, so a half-level split would leak
identities; a match-level rotation cannot. Every half is evaluated exactly once,
by a model that never saw that match in calibration or training.

The question this answers is not "is position better than body ID" -- it is not,
by a distance. It is whether position rescues the cases body ID gets wrong
without breaking the ones it gets right, and whether a learned combiner can
capture that trade where a uniform sum provably cannot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matchlab_core.reid.evidence import fit_fusion_weights

from matchlab_train.experiments.multi_input import (
    VAL_H5,
    extract_half,
    fit_calibrators,
    llr_matrix,
    load_appearance,
    rank1_from_scores,
    train_and_eval_model,
)

MATCHES = ("game_18", "game_24", "game_47")
FOLDS = [
    {"cal": "game_18", "train": "game_24", "eval": "game_47"},
    {"cal": "game_24", "train": "game_47", "eval": "game_18"},
    {"cal": "game_47", "train": "game_18", "eval": "game_24"},
]
ARMS = [
    ("body",),
    ("occupancy",),
    ("continuity",),
    ("gap",),
    ("transition",),
    ("body", "occupancy"),
    ("body", "transition"),
    ("body", "occupancy", "transition"),
    ("body", "occupancy", "continuity"),
    ("body", "occupancy", "continuity", "gap"),
    ("body", "occupancy", "continuity", "gap", "transition"),
    ("occupancy", "continuity", "gap"),
    ("occupancy", "transition"),
    ("occupancy", "continuity", "gap", "transition"),
]
MODEL_ARMS = [
    ("body",),
    ("body", "occupancy", "continuity", "gap"),
    ("body", "occupancy", "continuity", "gap", "transition"),
    ("occupancy", "continuity", "gap", "transition"),
]


def halves(match: str) -> list[str]:
    return [f"{match}_H1", f"{match}_H2"]


def load(match: str):
    return [extract_half(VAL_H5, k, appearance=load_appearance(k)) for k in halves(match)]


def conditional_effect(he, cals) -> dict:
    """Where body ID fails, does position rescue it -- and at what cost?"""
    body = llr_matrix(he, cals, ["body"]).sum(axis=1)
    pos = llr_matrix(
        he, cals, [c for c in ("occupancy", "continuity", "gap", "transition") if c in cals]
    ).sum(axis=1)
    fixed = broken = wrong = right = 0
    for e in range(he.n_episodes):
        sel = he.ep_index == e
        c = he.is_correct[sel]
        if not c.any():
            continue
        b, p = body[sel], pos[sel]
        body_ok = bool(c[np.argmax(b)])
        both_ok = bool(c[np.argmax(b + p)])
        if body_ok:
            right += 1
            broken += not both_ok
        else:
            wrong += 1
            fixed += both_ok
    return {
        "body_wrong": wrong,
        "position_fixes": fixed,
        "body_right": right,
        "position_breaks": broken,
        "net": fixed - broken,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--max-train-episodes", type=int, default=4000)
    ap.add_argument("--max-neg", type=int, default=None)
    ap.add_argument(
        "--out", type=Path, default=Path("docs/reports/2026-07-27-multi-input-body-raw.json")
    )
    args = ap.parse_args()

    cache = {m: load(m) for m in MATCHES}
    for m in MATCHES:
        print(f"{m}: {sum(h.n_episodes for h in cache[m])} episodes")

    sums: dict[str, list[float]] = {}
    weighted: dict[str, list[float]] = {}
    models: dict[str, list[float]] = {}
    weights_log: list[dict] = []
    cond: list[dict] = []

    for fold in FOLDS:
        cals = fit_calibrators(cache[fold["cal"]])
        ev = cache[fold["eval"]]
        print(f"\nfold: cal={fold['cal']} train={fold['train']} eval={fold['eval']}"
              f"  channels={sorted(cals)}")

        for h in ev:
            cond.append({"eval": fold["eval"], **conditional_effect(h, cals)})

        tr = cache[fold["train"]]
        for arm in ARMS:
            names = [n for n in arm if n in cals]
            if len(names) != len(arm):
                continue
            label = " + ".join(names)
            r1 = []
            for h in ev:
                s = llr_matrix(h, cals, names).sum(axis=1)
                r1.append(rank1_from_scores(s, h)[0])
            sums.setdefault(label, []).extend(r1)

            # Fitted on the TRAIN match: a third match, disjoint from both the
            # calibration match and the evaluated one.
            if len(names) > 1:
                w = fit_fusion_weights(
                    np.concatenate([llr_matrix(h, cals, names) for h in tr]),
                    np.concatenate([h.ep_index + 10_000 * i for i, h in enumerate(tr)]),
                    np.concatenate([h.is_correct for h in tr]),
                )
                weighted.setdefault(label, []).extend(
                    rank1_from_scores(llr_matrix(h, cals, names) @ w, h)[0] for h in ev
                )
                weights_log.append({"eval": fold["eval"], "arm": label,
                                    "w": dict(zip(names, w.round(4), strict=True))})

        if not args.skip_model:
            data = {"T": cache[fold["train"]], "E": ev}
            for arm in MODEL_ARMS:
                names = [n for n in arm if n in cals]
                if len(names) != len(arm):
                    continue
                label = " + ".join(names)
                print(f"  model arm: {label}")
                r = train_and_eval_model(
                    data, cals, names, epochs=args.epochs, args=args
                )
                if r:
                    models.setdefault(label, []).extend(r["per_half_rank1"])

    print(f"\n{'inputs':44s} {'sum r@1':>9s} {'fitted':>9s} {'model r@1':>10s}")
    for k, v in sums.items():
        m = models.get(k)
        f = weighted.get(k)
        ms = f"{100 * float(np.mean(m)):9.2f}%" if m else "        --"
        fs = f"{100 * float(np.mean(f)):8.2f}%" if f else "       --"
        print(f"{k:44s} {100 * float(np.mean(v)):8.2f}% {fs} {ms}")

    tot = {
        k: sum(c[k] for c in cond)
        for k in ("body_wrong", "position_fixes", "body_right", "position_breaks", "net")
    }
    print(f"\nposition where body ID fails (all 6 halves): {tot}")
    print(f"  rescue rate {100 * tot['position_fixes'] / max(tot['body_wrong'], 1):.1f}%"
          f"   damage rate {100 * tot['position_breaks'] / max(tot['body_right'], 1):.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "sum": sums,
                "fitted_weights": weighted,
                "model": models,
                "weights": weights_log,
                "conditional": cond,
                "totals": tot,
            },
            indent=2,
        )
    )
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
