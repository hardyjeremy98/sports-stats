"""Ingest SoccerTrack sequences into the Lab.

Unlike SoccerNet/SportsMOT (raw `img1/` frame dumps needing stitching),
SoccerTrack ships pre-encoded match video with a separate bbox-annotation
CSV -- there is no frame-stitching step here, and ffmpeg is not required
(`probe()` reads the video via OpenCV).

Expected layout: some `*.mp4` (or `*.MP4`) file under `root` (searched
recursively) with a same-stem `*.csv` bbox annotation file in the *same
directory* -- e.g. `root/match1/match1.mp4` + `root/match1/match1.csv`, or a
flat `root/match1.mp4` + `root/match1.csv`. Discovery is intentionally
simple: exactly "same directory, same stem"; anything else (video with no
matching CSV, CSV with no matching video) is silently skipped rather than
erroring, since a partially-populated download directory is normal. If
*nothing* is found, raises `FileNotFoundError` naming `root` and the
expected layout.

fps/width/height for `load_soccertrack_sequence` are read from `probe(video)`
(the same probe used at Video registration) since SoccerTrack CSVs carry no
seqinfo of their own.

Caveat carried from Task 3: SoccerTrack CSV frame numbers are ASSUMED
0-based (unverified against a real released dataset file) -- see
`pitchlab_core.gt.load_soccertrack_sequence`.

Requires the server package (like qa_labels).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    videos = sorted(set(root.rglob("*.mp4")) | set(root.rglob("*.MP4")))
    pairs: list[tuple[Path, Path]] = []
    for video in videos:
        csv = video.with_suffix(".csv")
        if csv.is_file():
            pairs.append((video, csv))
    return pairs


def ingest_soccertrack(
    root: Path,
    limit: int | None = None,
    sequences: list[str] | None = None,
    role: str = "tuning",
) -> list[tuple[int, str]]:
    from pitchlab_core.gt import load_soccertrack_sequence
    from pitchlab_core.video import probe
    from pitchlab_server.api.videos import register_video_file
    from pitchlab_server.db import init_db, session
    from pitchlab_server.settings import get_settings

    from pitchlab_train.datasets.manifest import update_tier_manifest

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"SoccerTrack root not found: {root}")
    pairs = _discover_pairs(root)
    if sequences:
        wanted = set(sequences)
        pairs = [(v, c) for v, c in pairs if v.stem in wanted]
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise FileNotFoundError(
            f"No SoccerTrack (video, csv) pairs found under {root}; expected "
            f"a *.mp4/*.MP4 with a same-stem *.csv in the same directory "
            f"(e.g. match1/match1.mp4 + match1/match1.csv)"
        )

    init_db()
    settings = get_settings()
    dest_dir = settings.videos_dir / "soccertrack"
    dest_dir.mkdir(parents=True, exist_ok=True)

    registered: list[tuple[int, str]] = []
    manifest_entries: list[dict] = []
    with session() as db:
        for video_src, csv_src in pairs:
            name = video_src.stem
            mp4 = dest_dir / f"{name}.mp4"
            if video_src.resolve() != mp4.resolve():
                shutil.copyfile(video_src, mp4)
            meta = probe(mp4)
            gt = load_soccertrack_sequence(
                csv_src, fps=meta.fps, width=meta.width, height=meta.height
            )
            gt_path = dest_dir / f"{name}.gt.json"
            gt_path.write_text(gt.model_dump_json())

            video = register_video_file(db, f"{name}.mp4", mp4, meta)
            video.gt_path = str(gt_path)
            db.commit()
            registered.append((video.id, name))
            manifest_entries.append(
                {"name": name, "video": str(mp4), "gt": str(gt_path), "role": role}
            )
            print(
                f"registered video {video.id}: {name} "
                f"({len(gt.tracks)} gt tracks, {gt.seq_length} frames)",
                flush=True,
            )

    update_tier_manifest(
        tier="soccertrack",
        dataset="soccertrack",
        source_split="all",
        entries=manifest_entries,
    )
    return registered
