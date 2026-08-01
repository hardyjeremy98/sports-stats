"""Turn `detector_bench.py score` / `score-ball` JSON into markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(v, nd=4):
    return "--" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def player_table(report: dict) -> str:
    rows = []
    for name, r in report["candidates"].items():
        if r.get("status") != "ok":
            continue
        p, b = r["pooled"], r["best_f1_operating_point"]
        bins = p.get("by_height_bin", {})
        small = None
        for k, v in (bins.items() if isinstance(bins, dict) else []):
            if isinstance(v, dict) and ("<25" in k or "0-25" in k or k.startswith("<")):
                small = v.get("recall")
        rows.append({
            "name": name, "ap": p["ap"], "f1": b["f1"], "conf": b["conf"],
            "p": b["precision"], "r": b["recall"],
            "small_recall": small,
            "dup": p.get("duplicates", {}).get("duplicate_rate"),
            "contaminated": r.get("contaminated"),
            "trained_on": r.get("trained_on", ""),
        })
    rows.sort(key=lambda x: (x["ap"] is None, -(x["ap"] or 0)))

    out = ["| Detector | AP@0.5 | best F1 | @conf | P | R | dup rate | tier |",
           "|---|---|---|---|---|---|---|---|"]
    for x in rows:
        tier = "CONTAMINATED" if x["contaminated"] else "clean"
        out.append(f"| `{x['name']}` | {_f(x['ap'])} | {_f(x['f1'])} | {x['conf']} | "
                   f"{_f(x['p'])} | {_f(x['r'])} | {_f(x['dup'])} | {tier} |")
    return "\n".join(out)


def height_table(report: dict, names: list[str]) -> str:
    """Recall by GT box height -- where a detector actually loses players."""
    bins: list[str] = []
    per = {}
    for n in names:
        r = report["candidates"].get(n)
        if not r or r.get("status") != "ok":
            continue
        src = r.get("pooled_at_best_f1") or r["pooled"]
        hb = src.get("by_height_bin") or {}
        if isinstance(hb, list):
            hb = {str(d.get("bin", i)): d for i, d in enumerate(hb)}
        per[n] = hb
        for k in hb:
            if k not in bins:
                bins.append(k)
    if not per:
        return "_no height-bin data_"
    out = ["| Detector | " + " | ".join(bins) + " |",
           "|---" * (len(bins) + 1) + "|"]
    for n, hb in per.items():
        cells = []
        for b in bins:
            v = hb.get(b) or {}
            cells.append(_f(v.get("recall")) if isinstance(v, dict) else "--")
        out.append(f"| `{n}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def ball_table(report: dict) -> str:
    rows = []
    for name, r in report["candidates"].items():
        if r.get("status") != "ok":
            continue
        b = r["best_f1_operating_point"]
        rows.append((name, b["f1"], b["precision"], b["recall"],
                     b["false_alarm_rate"], r["n_gt_frames"],
                     r.get("contaminated")))
    rows.sort(key=lambda x: -x[1])
    out = ["| Ball detector | best F1 | P | R | false-alarm rate | ball-GT frames |",
           "|---|---|---|---|---|---|"]
    for n, f1, p, rc, fa, ngt, _c in rows:
        out.append(f"| `{n}` | {_f(f1)} | {_f(p)} | {_f(rc)} | {_f(fa)} | {ngt} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--kind", choices=["player", "ball"], default="player")
    ap.add_argument("--heights", nargs="*", default=None)
    args = ap.parse_args()
    rep = json.loads(Path(args.json_path).read_text())
    print(f"sequences: {', '.join(rep['sequences'])}  stride={rep['stride']}\n")
    if args.kind == "ball":
        print(ball_table(rep))
    else:
        print(player_table(rep))
        if args.heights:
            print("\nRecall by GT box height (px):\n")
            print(height_table(rep, args.heights))


if __name__ == "__main__":
    main()
