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

    if args.command == "ingest-soccernet":
        from pathlib import Path

        from pitchlab_train.datasets.soccernet_tracking import ingest_soccernet

        registered = ingest_soccernet(
            Path(args.root), split=args.split, limit=args.limit, sequences=args.sequences
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
