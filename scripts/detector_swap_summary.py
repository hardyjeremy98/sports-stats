"""Summarise the detector-swap end-to-end benchmark as a markdown table.

Reads the per-run `eval.json` files an experiment produced, folds each through
`matchlab_core.evaluation.headline_metrics`, and reports the per-candidate mean
over sequences plus the delta. Only sequences BOTH candidates completed are
included, so a failed run can never make one arm look better by scoring an
easier subset.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# (key, label, higher_is_better)
METRICS = [
    ("detection_ap", "detection AP", True),
    ("detection_recall", "detection recall", True),
    ("hota_tracklet", "HOTA (tracklet)", True),
    ("hota_entity", "HOTA (entity)", True),
    ("idf1_tracklet", "IDF1 (tracklet)", True),
    ("idf1_entity", "IDF1 (entity)", True),
    ("tracklet_purity", "tracklet purity", True),
    ("entity_purity", "entity purity", True),
    ("idsw_tracklet", "ID switches (tracklet)", False),
    ("idsw_persistent_tracklet", "persistent ID switches", False),
    ("mixed_track_seconds", "mixed-identity seconds", False),
    ("detection_miss_burst_p95", "miss-burst p95 (frames)", False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--baseline", default="v1-hardened-incumbent")
    ap.add_argument("--candidate", default="detector-swap-mobadam")
    args = ap.parse_args()

    from matchlab_core.evaluation import headline_metrics

    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in sorted(Path(args.experiment_dir).glob("runs/*/eval.json")):
        name = p.parent.name
        for cand in (args.baseline, args.candidate):
            if name.startswith(cand + "-"):
                seq = name[len(cand) + 1:]
                rows[cand][seq] = headline_metrics(json.loads(p.read_text()))
                break

    base, cand = rows.get(args.baseline, {}), rows.get(args.candidate, {})
    shared = sorted(set(base) & set(cand))
    missing = sorted((set(base) | set(cand)) - set(shared))
    if not shared:
        raise SystemExit(f"no sequences completed by both arms (base={list(base)}, cand={list(cand)})")

    print(f"Sequences compared ({len(shared)}): {', '.join(shared)}")
    if missing:
        print(f"EXCLUDED (only one arm completed): {', '.join(missing)}")
    print()
    print(f"| Metric | {args.baseline} | {args.candidate} | Δ |")
    print("|---|---|---|---|")
    for key, label, higher in METRICS:
        b = [base[s].get(key) for s in shared]
        c = [cand[s].get(key) for s in shared]
        if any(v is None for v in b + c):
            continue
        mb, mc = statistics.fmean(b), statistics.fmean(c)
        d = mc - mb
        good = (d > 0) if higher else (d < 0)
        arrow = "" if abs(d) < 1e-9 else (" ✅" if good else " ⚠️")
        print(f"| {label} | {mb:.4f} | {mc:.4f} | {d:+.4f}{arrow} |")


if __name__ == "__main__":
    main()
