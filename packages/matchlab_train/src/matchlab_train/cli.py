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

    ap_p = sub.add_parser(
        "audit-possessor-labels",
        help="Profile WEAK possessor labels on GT/oracle inputs (SPO-83; "
        "label-risk profile, not an accuracy measure)",
    )
    ap_p.add_argument(
        "--root",
        default="data/soccernet/tracking/test",
        help="Dir of SNMOT sequence dirs (each with gameinfo.ini)",
    )
    ap_p.add_argument("--limit", type=int, default=None, help="Audit at most N sequences")
    ap_p.add_argument("--out", required=True, help="Output report JSON path")
    ap_p.add_argument(
        "--min-ball-coverage",
        type=float,
        default=0.5,
        help="Exclude sequences with GT ball coverage below this from the aggregate",
    )

    xv_p = sub.add_parser(
        "crossval-events",
        help="Cross-validate possession-derived events against ball-trajectory "
        "touches on GT/oracle inputs (B3; corroboration, not accuracy)",
    )
    xv_p.add_argument(
        "--root",
        default="data/soccernet/tracking/test",
        help="Dir of SNMOT sequence dirs (each with gameinfo.ini)",
    )
    xv_p.add_argument("--limit", type=int, default=None, help="Use at most N sequences")
    xv_p.add_argument("--out", required=True, help="Output report JSON path")
    xv_p.add_argument(
        "--tolerance-frames", type=int, default=6,
        help="Max frame distance for an event and a touch to count as agreeing",
    )
    xv_p.add_argument(
        "--estimator",
        default="possession-heuristic-image",
        choices=["possession-heuristic-image", "possession-viterbi"],
        help="Which possession estimator derives the events (the denoiser ablation)",
    )

    loc_p = sub.add_parser(
        "spot-localization",
        help="Score a spotting signal against SNMOT's one-action-per-clip labels "
        "(B3; localisation/recall only -- precision is unsupported)",
    )
    loc_p.add_argument("--root", default="data/soccernet/tracking/test")
    loc_p.add_argument("--limit", type=int, default=None)
    loc_p.add_argument("--out", required=True, help="Output report JSON path")
    loc_p.add_argument(
        "--signal",
        default="ball-trajectory",
        choices=["ball-trajectory", "possession", "possession-viterbi"],
        help="'possession' is the SPO-79 heuristic; 'possession-viterbi' the denoised "
        "timeline. Comparing the two is a GUARD (nearest-prediction matching means "
        "removing events can only raise the error), never a score",
    )
    fs_p = sub.add_parser(
        "footpass-stats",
        help="Summarise a FOOTPASS/PCBAS tactical HDF5 split (SPO-96 Phase 0 ingest "
        "gate): halves, rows, events per class, bbox coverage, slots seen",
    )
    fs_p.add_argument("--h5", required=True, help="Path to <split>_tactical_data.h5")
    fs_p.add_argument("--out", default=None, help="Write the report JSON here")

    pcl_p = sub.add_parser(
        "publish-pcbas-half",
        help="Score one PCBAS half against its tactical GT and register it as a Lab run",
    )
    pcl_p.add_argument("--key", required=True, help="Half key, e.g. game_18_H1")
    pcl_p.add_argument("--h5", default="data/footpass/tactical/val_tactical_data.h5")
    pcl_p.add_argument("--playbyplay", required=True, help="Stage-2 predictions JSON")
    pcl_p.add_argument("--video-root", default="data/footpass/videos_352x640")
    pcl_p.add_argument("--run-id", default=None, help="Default: pcbas-<key>")
    pcl_p.add_argument("--runs-dir", default="data/runs")
    pcl_p.add_argument("--label", default=None, help="Run label shown in the Lab")
    pcl_p.add_argument("--delta", type=int, default=None, help="Match tolerance, frames")
    pcl_p.add_argument("--conf-thresh", type=float, default=None)
    pcl_p.add_argument(
        "--no-register", action="store_true",
        help="Write the run directory but do not touch the Lab database",
    )

    g1_p = sub.add_parser(
        "gate1-calibration-eval",
        help="Gate 1 (SPO-60): reproduce PnLCalib's SoccerNet-Calibration test-split "
        "accuracy under the official challenge protocol",
    )
    g1_p.add_argument(
        "--soccernet-dir", default="data/soccernet/calibration",
        help="SoccerNet-Calibration root (contains <split>/ with <id>.jpg + <id>.json)",
    )
    g1_p.add_argument("--split", default="test", choices=["train", "valid", "test", "challenge"])
    g1_p.add_argument(
        "--thresholds", default="5,10,20",
        help="Comma-separated pixel thresholds for JaC@t (default 5,10,20)",
    )
    g1_p.add_argument(
        "--out", default="data/reports/gate1-calibration",
        help="Output directory for the gate record JSON + markdown summary",
    )
    g1_p.add_argument(
        "--predictor-cmd", default=None,
        help="Predictor command (camera-output mode); driven via the job-manifest "
        "contract. Omit and pass --prediction-dir to score precomputed predictions.",
    )
    g1_p.add_argument(
        "--prediction-dir", default=None,
        help="Directory of precomputed camera_<id>.json predictions (skip prediction)",
    )
    g1_p.add_argument(
        "--scorer-cmd", required=True,
        help="Official scorer command (sn-calibration adapter in the sibling checkout)",
    )
    g1_p.add_argument(
        "--predictor-params", default=None,
        help="JSON object merged into the job manifest's params for the predictor "
        '(e.g. weight paths: \'{"weights_kp": "...", "weights_line": "...", '
        '"device": "cuda:0"}\')',
    )
    g1_p.add_argument("--tolerance", type=float, default=0.03, help="Gate tolerance (absolute)")
    g1_p.add_argument("--resolution-width", type=int, default=960)
    g1_p.add_argument("--resolution-height", type=int, default=540)

    args = parser.parse_args()

    if args.command == "gate1-calibration-eval":
        import shlex
        from pathlib import Path

        from matchlab_train.calibration_gate import run_gate1_calibration_eval

        record = run_gate1_calibration_eval(
            soccernet_dir=Path(args.soccernet_dir),
            split=args.split,
            thresholds=[int(t) for t in args.thresholds.split(",")],
            out_dir=Path(args.out),
            scorer_command=shlex.split(args.scorer_cmd),
            predictor_command=shlex.split(args.predictor_cmd) if args.predictor_cmd else None,
            predictor_params=json.loads(args.predictor_params) if args.predictor_params else None,
            prediction_dir=args.prediction_dir,
            tolerance=args.tolerance,
            resolution_width=args.resolution_width,
            resolution_height=args.resolution_height,
        )
        print(json.dumps(record["measured"], indent=2))
        print(f"gate passed: {record['passed']}")
        print(f"gate record written under {args.out}")
        return 0 if record["passed"] else 1

    if args.command == "footpass-stats":
        from pathlib import Path

        from matchlab_train.datasets.footpass_pcbas import split_stats

        stats = split_stats(args.h5)
        payload = stats.model_dump(mode="json")
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2))
            print(f"wrote {args.out}")
        print(
            f"{stats.n_halves} halves, {stats.n_rows:,} rows, {stats.n_events:,} events, "
            f"{stats.frac_events_with_bbox:.1%} of events have a bbox, "
            f"{len(stats.slots_seen)} slots seen"
        )
        for name, count in sorted(stats.events_per_class.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<10} {count:>7,}")
        return 0

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

    if args.command == "publish-pcbas-half":
        from pathlib import Path

        from matchlab_core.pcbas.eval import DEFAULT_CONF_THRESH, DEFAULT_DELTA

        from matchlab_train.datasets.footpass_lab import publish_half
        from matchlab_train.datasets.footpass_pcbas import parse_key

        game_id, _ = parse_key(args.key)
        run_id = args.run_id or f"pcbas-{args.key}"
        metrics = publish_half(
            h5_path=Path(args.h5),
            playbyplay_path=Path(args.playbyplay),
            video_path=Path(args.video_root) / f"{game_id}.mp4",
            run_dir=Path(args.runs_dir) / run_id,
            key=args.key,
            run_id=run_id,
            delta=args.delta if args.delta is not None else DEFAULT_DELTA,
            conf_thresh=(
                args.conf_thresh if args.conf_thresh is not None else DEFAULT_CONF_THRESH
            ),
            label=args.label,
            register=not args.no_register,
        )
        for k, v in metrics.items():
            print(f"{k}: {v}")
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

    if args.command == "audit-possessor-labels":
        from pathlib import Path

        from matchlab_train.datasets.possessor_audit import audit_soccernet_tracking

        report = audit_soccernet_tracking(
            args.root, limit=args.limit, min_ball_coverage=args.min_ball_coverage
        )
        Path(args.out).write_text(report.model_dump_json(indent=2))
        excluded = [s.sequence for s in report.sequences if s.excluded]
        agg = report.aggregate
        print(f"wrote audit of {len(report.sequences)} sequences to {args.out}")
        print(f"excluded (ball coverage < {report.min_ball_coverage}): {excluded or 'none'}")
        print(f"aggregate label coverage: {agg.coverage:.3f} over {agg.total_frames} frames")
        print(f"NOT AN ACCURACY MEASURE -- {report.caveat}")
        return 0

    if args.command == "crossval-events":
        from pathlib import Path

        from matchlab_train.datasets.possessor_audit import crossval_soccernet_tracking

        report = crossval_soccernet_tracking(
            args.root,
            limit=args.limit,
            tolerance_frames=args.tolerance_frames,
            estimator=args.estimator,
        )
        Path(args.out).write_text(report.model_dump_json(indent=2))
        a = report.aggregate
        print(f"wrote cross-validation of {len(report.sequences)} sequences to {args.out}")
        print(f"estimator={report.estimator}")
        print(
            f"events={a.n_events} touches={a.n_touches} matched={a.matched} "
            f"possession_only={a.possession_only} trajectory_only={a.trajectory_only}"
        )
        print(
            f"agreement_rate={a.agreement_rate:.3f} (matched/events)  "
            f"touch_recall={a.touch_recall:.3f} (matched/touches)  "
            f"tolerance=+/-{a.tolerance_frames} frames"
        )
        print(f"NOT ACCURACY -- {report.caveat}")
        return 0

    if args.command == "spot-localization":
        from pathlib import Path

        from matchlab_train.datasets.possessor_audit import localize_soccernet_tracking

        report = localize_soccernet_tracking(
            args.root, limit=args.limit, signal=args.signal
        )
        Path(args.out).write_text(report.model_dump_json(indent=2))
        b, n = report.ball_contact, report.non_ball
        print(f"wrote localisation report ({report.signal}) to {args.out}")
        print(
            f"ball-contact classes: n={b.n} median={b.median_error_frames}f "
            f"({b.median_error_seconds}s) within-5f={b.within_5_frames:.0%} "
            f"within-25f={b.within_25_frames:.0%}"
        )
        print(
            f"non-ball classes:     n={n.n} median={n.median_error_frames}f "
            f"within-5f={n.within_5_frames:.0%}  (a ball spotter SHOULD miss these)"
        )
        print(f"LOCALISATION ONLY -- {report.caveat}")
        return 0

    config = ExperimentConfig.from_yaml(args.config)
    experiment = registry.build(config.task, config)
    print(f"running experiment '{config.name}' (task={config.task})", flush=True)
    result = experiment.run()
    print(json.dumps(result.get("summary", result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
