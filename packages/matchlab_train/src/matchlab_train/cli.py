from __future__ import annotations

import argparse
import json
import sys

from matchlab_train import registry
from matchlab_train.config import ExperimentConfig


def main() -> int:
    parser = argparse.ArgumentParser(prog="matchlab-train")
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

    it_p = sub.add_parser(
        "import-tracklets",
        help="Import an external tracker's MOT output as a scoreable run directory",
    )
    it_p.add_argument("--mot", required=True, help="MOT-format tracklet file")
    it_p.add_argument("--sidecar", required=True, help="ExternalProvenance sidecar JSON")
    it_p.add_argument("--out", required=True, help="Output run directory")
    it_p.add_argument("--fps", type=float, required=True, help="Video fps")
    it_p.add_argument("--frame-count", type=int, required=True, help="Video frame count")
    it_p.add_argument("--sample-stride", type=int, default=1, help="Video sample stride")
    it_p.add_argument(
        "--frozen-detections", default=None,
        help="Frozen-detections export dir to cross-check the sidecar's det.txt hash against",
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

    sb_p = sub.add_parser(
        "ingest-soccernet-ball",
        help="Register SoccerNet Ball Action Spotting matches as Lab videos with "
        "action-event ground truth",
    )
    sb_p.add_argument("--root", default="data/soccernet/ball", help="Dataset root")
    sb_p.add_argument("--split", default="test", choices=["train", "valid", "test", "challenge"])
    sb_p.add_argument("--limit", type=int, default=None, help="Max matches to ingest")
    sb_p.add_argument("--sequences", nargs="*", default=None, help="Specific match names")
    sb_p.add_argument(
        "--role", default="tuning", choices=["tuning", "held_out"],
        help="Split-manifest role to record for newly ingested matches",
    )

    pl_p = sub.add_parser(
        "derive-possessor-labels",
        help="Derive WEAK per-frame possessor labels from a run's artifacts "
        "(SPO-82; Peral training supervision, no new annotation)",
    )
    pl_p.add_argument("--run-dir", required=True, help="Run dir (needs tracklets/ball/teams)")
    pl_p.add_argument("--out", required=True, help="Output labels JSON path")

    args = parser.parse_args()

    if args.command == "tasks":
        for task in registry.available():
            print(task)
        return 0

    if args.command == "export-labels":
        from pathlib import Path

        from matchlab_train.datasets.qa_labels import export_qa_labels

        export_qa_labels(Path(args.out))
        return 0

    if args.command == "export-reid":
        from pathlib import Path

        from matchlab_train.datasets.reid_labels import export_reid_labels

        export_reid_labels(Path(args.out))
        return 0

    if args.command == "export-detections":
        from matchlab_core.exchange import DEFAULT_INCLUDE_CLASSES, export_frozen_detections
        from matchlab_core.provenance import sha256_file

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

    if args.command == "import-tracklets":
        from matchlab_core.exchange import import_mot_tracklets

        out_dir = import_mot_tracklets(
            args.mot,
            args.sidecar,
            args.out,
            fps=args.fps,
            frame_count=args.frame_count,
            sample_stride=args.sample_stride,
            frozen_detections_dir=args.frozen_detections,
        )
        n_tracklets = len(json.loads((out_dir / "tracklets.json").read_text()))
        print(f"imported to {out_dir}")
        print(f"{n_tracklets} tracklets")
        return 0

    if args.command == "ingest-soccernet":
        from pathlib import Path

        from matchlab_train.datasets.soccernet_tracking import ingest_soccernet

        registered = ingest_soccernet(
            Path(args.root), split=args.split, limit=args.limit, sequences=args.sequences
        )
        print(f"ingested {len(registered)} sequences")
        return 0

    if args.command == "ingest-sportsmot":
        from pathlib import Path

        from matchlab_train.datasets.sportsmot import ingest_sportsmot

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

        from matchlab_train.datasets.soccertrack import ingest_soccertrack

        registered = ingest_soccertrack(
            Path(args.root), limit=args.limit, sequences=args.sequences, role=args.role
        )
        print(f"ingested {len(registered)} sequences")
        return 0

    if args.command == "ingest-soccernet-ball":
        from pathlib import Path

        from matchlab_train.datasets.soccernet_ball import ingest_soccernet_ball

        registered = ingest_soccernet_ball(
            Path(args.root),
            split=args.split,
            limit=args.limit,
            sequences=args.sequences,
            role=args.role,
        )
        print(f"ingested {len(registered)} matches")
        return 0

    if args.command == "derive-possessor-labels":
        from matchlab_train.datasets.possessor_labels import derive_weak_possessor_labels

        labels = derive_weak_possessor_labels(args.run_dir, out_path=args.out)
        print(f"wrote {len(labels.frames)} possessor-label frames to {args.out}")
        print(f"WEAK LABELS -- {labels.caveat}")
        return 0

    config = ExperimentConfig.from_yaml(args.config)
    experiment = registry.build(config.task, config)
    print(f"running experiment '{config.name}' (task={config.task})", flush=True)
    result = experiment.run()
    print(json.dumps(result.get("summary", result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
