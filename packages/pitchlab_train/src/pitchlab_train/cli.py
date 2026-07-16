from __future__ import annotations

import argparse
import json
import sys

from pitchlab_train import registry
from pitchlab_train.config import ExperimentConfig


def main() -> int:
    parser = argparse.ArgumentParser(prog="pitchlab-train")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an experiment from a YAML config")
    run_p.add_argument("config", help="Path to experiment YAML")

    sub.add_parser("tasks", help="List available experiment tasks")

    labels_p = sub.add_parser(
        "export-labels", help="Export QA-derived labeled examples to JSONL"
    )
    labels_p.add_argument("--out", default="data/experiments", help="Output root")

    reid_p = sub.add_parser(
        "export-reid", help="Export identity-QA pair verdicts as re-ID training pairs"
    )
    reid_p.add_argument("--out", default="data/experiments", help="Output root")

    sn_p = sub.add_parser(
        "ingest-soccernet",
        help="Register SoccerNet tracking sequences as Lab videos with ground truth",
    )
    sn_p.add_argument("--root", default="data/soccernet/tracking", help="Dataset root")
    sn_p.add_argument("--split", default="test", choices=["train", "test", "challenge"])
    sn_p.add_argument("--limit", type=int, default=None, help="Max sequences to ingest")
    sn_p.add_argument("--sequences", nargs="*", default=None, help="Specific sequence names")

    sm_p = sub.add_parser(
        "ingest-sportsmot",
        help="Register SportsMOT sequences as Lab videos with ground truth",
    )
    sm_p.add_argument("--root", default="data/sportsmot", help="Dataset root")
    sm_p.add_argument("--split", default="val")
    sm_p.add_argument("--limit", type=int, default=None, help="Max sequences to ingest")
    sm_p.add_argument("--sequences", nargs="*", default=None, help="Specific sequence names")
    sm_p.add_argument(
        "--role", default="tuning", choices=["tuning", "held_out"],
        help="Split-manifest role to record for newly ingested sequences",
    )

    ed_p = sub.add_parser(
        "export-detections",
        help="Export a run's detections as a frozen MOT det.txt + provenance sidecar",
    )
    ed_p.add_argument("--run-dir", required=True, help="Run directory to export from")
    ed_p.add_argument("--out", required=True, help="Output directory for det.txt + sidecar")
    ed_p.add_argument(
        "--include-ball", action="store_true",
        help="Include ball detections (excluded by default -- external MOT trackers track persons)",
    )

    st_p = sub.add_parser(
        "ingest-soccertrack",
        help="Register SoccerTrack sequences as Lab videos with ground truth",
    )
    st_p.add_argument("--root", default="data/soccertrack", help="Dataset root")
    st_p.add_argument("--limit", type=int, default=None, help="Max sequences to ingest")
    st_p.add_argument("--sequences", nargs="*", default=None, help="Specific sequence names")
    st_p.add_argument(
        "--role", default="tuning", choices=["tuning", "held_out"],
        help="Split-manifest role to record for newly ingested sequences",
    )

    args = parser.parse_args()

    if args.command == "tasks":
        for task in registry.available():
            print(task)
        return 0

    if args.command == "export-labels":
        from pathlib import Path

        from pitchlab_train.datasets.qa_labels import export_qa_labels

        export_qa_labels(Path(args.out))
        return 0

    if args.command == "export-reid":
        from pathlib import Path

        from pitchlab_train.datasets.reid_labels import export_reid_labels

        export_reid_labels(Path(args.out))
        return 0

    if args.command == "export-detections":
        from pitchlab_core.exchange import DEFAULT_INCLUDE_CLASSES, export_frozen_detections
        from pitchlab_core.provenance import sha256_file

        include_classes = DEFAULT_INCLUDE_CLASSES
        if args.include_ball:
            include_classes = (*DEFAULT_INCLUDE_CLASSES, "ball")

        out_dir = export_frozen_detections(
            args.run_dir, args.out, include_classes=include_classes
        )
        det_txt_hash = sha256_file(out_dir / "det.txt")
        print(f"exported to {out_dir}")
        print(f"det.txt sha256: {det_txt_hash}")
        return 0

    if args.command == "ingest-soccernet":
        from pathlib import Path

        from pitchlab_train.datasets.soccernet_tracking import ingest_soccernet

        registered = ingest_soccernet(
            Path(args.root), split=args.split, limit=args.limit, sequences=args.sequences
        )
        print(f"ingested {len(registered)} sequences")
        return 0

    if args.command == "ingest-sportsmot":
        from pathlib import Path

        from pitchlab_train.datasets.sportsmot import ingest_sportsmot

        registered = ingest_sportsmot(
            Path(args.root),
            split=args.split,
            limit=args.limit,
            sequences=args.sequences,
            role=args.role,
        )
        print(f"ingested {len(registered)} sequences")
        return 0

    if args.command == "ingest-soccertrack":
        from pathlib import Path

        from pitchlab_train.datasets.soccertrack import ingest_soccertrack

        registered = ingest_soccertrack(
            Path(args.root), limit=args.limit, sequences=args.sequences, role=args.role
        )
        print(f"ingested {len(registered)} sequences")
        return 0

    config = ExperimentConfig.from_yaml(args.config)
    experiment = registry.build(config.task, config)
    print(f"running experiment '{config.name}' (task={config.task})", flush=True)
    result = experiment.run()
    print(json.dumps(result.get("summary", result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
