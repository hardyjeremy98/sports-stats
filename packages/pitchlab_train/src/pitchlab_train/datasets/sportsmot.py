"""Ingest SportsMOT sequences (data/sportsmot/<split>/<seq>/{img1,gt/gt.txt,
seqinfo.ini}, standard MOT17-style layout) into the Lab: stitch each
sequence's frames into a browser-playable mp4, convert its MOT ground truth
to the GroundTruth JSON form, register both as a Video row (videos.gt_path),
and record/merge a tuning|held_out entry for the sequence in
`configs/datasets/sportsmot.json`. Mirrors `ingest_soccernet` (same on-disk
layout family, same registration path).

SportsMOT annotates players only (see `load_sportsmot_sequence`'s docstring
in `pitchlab_core.gt`) -- no ball/referee distinction, no team labels.

Requires the server package (like qa_labels) and ffmpeg on PATH (frame
stitching).
"""

from __future__ import annotations

from pathlib import Path

from pitchlab_core.gt import load_sportsmot_sequence

from pitchlab_train.datasets.stitch import stitch_frames_to_mp4


def ingest_sportsmot(
    root: Path,
    split: str = "val",
    limit: int | None = None,
    sequences: list[str] | None = None,
    role: str = "tuning",
) -> list[tuple[int, str]]:
    from pitchlab_core.video import probe
    from pitchlab_server.api.videos import register_video_file
    from pitchlab_server.db import init_db, session
    from pitchlab_server.settings import get_settings

    from pitchlab_train.datasets.manifest import update_tier_manifest

    root = Path(root)
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"SportsMOT split dir not found: {split_dir}")
    seq_dirs = sorted(
        d
        for d in split_dir.iterdir()
        if d.is_dir() and (d / "gt" / "gt.txt").exists() and (d / "img1").is_dir()
    )
    if sequences:
        wanted = set(sequences)
        seq_dirs = [d for d in seq_dirs if d.name in wanted]
    if limit:
        seq_dirs = seq_dirs[:limit]
    if not seq_dirs:
        raise FileNotFoundError(
            f"No sequences with gt/gt.txt + img1/ under {split_dir}"
        )

    init_db()
    settings = get_settings()
    dest_dir = settings.videos_dir / "sportsmot"
    dest_dir.mkdir(parents=True, exist_ok=True)

    registered: list[tuple[int, str]] = []
    manifest_entries: list[dict] = []
    with session() as db:
        for seq_dir in seq_dirs:
            gt = load_sportsmot_sequence(seq_dir)
            mp4 = dest_dir / f"{seq_dir.name}.mp4"
            if not mp4.exists():
                stitch_frames_to_mp4(seq_dir / "img1", gt.fps, mp4)
            gt_path = dest_dir / f"{seq_dir.name}.gt.json"
            gt_path.write_text(gt.model_dump_json())

            video = register_video_file(db, f"{seq_dir.name}.mp4", mp4, probe(mp4))
            video.gt_path = str(gt_path)
            db.commit()
            registered.append((video.id, seq_dir.name))
            manifest_entries.append(
                {"name": seq_dir.name, "video": str(mp4), "gt": str(gt_path), "role": role}
            )
            print(
                f"registered video {video.id}: {seq_dir.name} "
                f"({len(gt.tracks)} gt tracks, {gt.seq_length} frames)",
                flush=True,
            )

    update_tier_manifest(
        tier="sportsmot",
        dataset="sportsmot",
        source_split=split,
        entries=manifest_entries,
    )
    return registered
