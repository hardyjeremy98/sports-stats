"""Ingest SoccerNet tracking sequences (data/soccernet/tracking/<split>/SNMOT-*)
into the Lab: stitch each sequence's frames into a browser-playable mp4,
convert its MOT ground truth to the GroundTruth JSON form, and register both
as a Video row (videos.gt_path). Runs on these videos are then scored against
the labels automatically.

Requires the server package (like qa_labels) and ffmpeg on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pitchlab_core.gt import load_soccernet_sequence


def ingest_soccernet(
    root: Path,
    split: str = "test",
    limit: int | None = None,
    sequences: list[str] | None = None,
) -> list[tuple[int, str]]:
    from pitchlab_core.video import probe
    from pitchlab_server.api.videos import register_video_file
    from pitchlab_server.db import init_db, session
    from pitchlab_server.settings import get_settings

    split_dir = root / split
    seq_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir() and (d / "gt" / "gt.txt").exists())
    if sequences:
        wanted = set(sequences)
        seq_dirs = [d for d in seq_dirs if d.name in wanted]
    if limit:
        seq_dirs = seq_dirs[:limit]
    if not seq_dirs:
        raise FileNotFoundError(f"No sequences with gt under {split_dir}")

    init_db()
    settings = get_settings()
    dest_dir = settings.videos_dir / "soccernet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    registered: list[tuple[int, str]] = []
    with session() as db:
        for seq_dir in seq_dirs:
            gt = load_soccernet_sequence(seq_dir)
            mp4 = dest_dir / f"{seq_dir.name}.mp4"
            if not mp4.exists():
                _stitch(seq_dir / "img1", gt.fps, mp4)
            gt_path = dest_dir / f"{seq_dir.name}.gt.json"
            gt_path.write_text(gt.model_dump_json())

            video = register_video_file(db, f"{seq_dir.name}.mp4", mp4, probe(mp4))
            video.gt_path = str(gt_path)
            db.commit()
            registered.append((video.id, seq_dir.name))
            print(
                f"registered video {video.id}: {seq_dir.name} "
                f"({len(gt.tracks)} gt tracks, {gt.seq_length} frames)",
                flush=True,
            )
    return registered


def _stitch(img_dir: Path, fps: float, dest: Path) -> None:
    """Frames -> h264 mp4. ffmpeg CLI rather than cv2: browsers can't play
    cv2's mp4v, and yuv420p + libx264 is the compatibility baseline."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(img_dir / "%06d.jpg"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
