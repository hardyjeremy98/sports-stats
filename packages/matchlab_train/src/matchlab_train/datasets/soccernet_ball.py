"""Ingest SoccerNet Ball Action Spotting matches
(`data/soccernet/ball/<split>/<match>/{video, Labels-ball.json}`) into the
Lab as timestamped action-event ground truth (SPO-47): parse each match's
`Labels-ball.json` into an `EventGroundTruth` (`matchlab_core.event_gt`),
register the match video as a Lab Video row (`videos.gt_path`), and record a
tuning|held_out entry for it in `configs/datasets/soccernet-ball.json`.
Mirrors `ingest_sportsmot`/`ingest_soccertrack` (same registration +
`update_tier_manifest` path, same `--role` CLI argument). Unlike the
tracking tiers there is no frame-stitching step -- SoccerNet ball-action
matches ship a real, already-encoded video file.

CONCERN (documented, not resolved -- flagging per SPO-47's brief rather than
guessing): the real SoccerNet Ball Action Spotting release is understood to
package each match as TWO half-video files sharing one `Labels-ball.json`
(whose `half` field picks between them), not the single "one video per
match" layout `ingest_soccernet_ball` assumes below. This repo has no
downloaded copy of the dataset to verify against (ingest is deliberately
network-free per the brief), so rather than silently guessing which half's
video to register, `ingest_soccernet_ball` raises loudly -- naming the match
dir and the video files it found -- whenever a match dir does not contain
exactly one video file. Revisit this (and the single-mp4-per-match
assumption in `configs/datasets/soccernet-ball.json`'s notes) once a real
`Labels-ball.json` + match directory has actually been downloaded.

Requires the server package (like the other ingest adapters) for the
registration path; `load_soccernet_ball_labels` itself is pure/network-free
and does not import it.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from matchlab_core.event_gt import EventGroundTruth, GroundTruthEvent

_VIDEO_EXTS = (".mp4", ".mkv")


def load_soccernet_ball_labels(
    labels_json_path: str | Path, *, fps: float, sequence: str | None = None
) -> EventGroundTruth:
    """Parse a SoccerNet Ball Action Spotting `Labels-ball.json` into an
    `EventGroundTruth`.

    File shape: `{"UrlLocal": "...", "annotations": [{"gameTime": "<half> -
    MM:SS", "label": ..., "position": "<milliseconds>", "team": ...,
    "visibility": ...}, ...]}`, where `position` is milliseconds elapsed
    since the start of that half (NOT match time).

    Conversion, per annotation:
    - `half = int(gameTime.split("-")[0])` (`gameTime`'s "<half> - MM:SS"
      prefix; `int()` tolerates the surrounding whitespace).
    - `t = position_ms / 1000.0` (seconds within the half).
    - `frame_idx = round(t * fps)`, using round-HALF-UP rather than
      Python's builtin `round()` (round-half-to-even): position "12340" @
      fps 25 gives `t=12.34`, `t*fps == 308.5` exactly, and the spec's
      worked example for this case is frame 309 -- banker's rounding would
      give 308. `math.floor(t * fps + 0.5)` gets half-up for the
      non-negative values event timestamps always are.
    - `class_ = label` verbatim (no taxonomy mapping -- same policy as
      `SpottedEvent`, see `matchlab_core.schemas.spotting`).

    `visibility` is NOT filtered on: every annotation becomes a
    `GroundTruthEvent` regardless of its `visibility` value. Kept because
    the ball-action task cares about occurrence, and a "not shown" event
    (ball off-screen but the action still happened, per the SoccerNet task
    definition) is still a real, scoreable spot; revisit if the avg-mAP
    metric task (a later task) decides visibility-gating is needed.

    Tolerant of missing optional keys (`team`, `visibility`, `UrlLocal`) --
    only `gameTime`, `label`, and `position` feed the conversion above.
    """
    labels_json_path = Path(labels_json_path)
    raw: dict[str, Any] = json.loads(labels_json_path.read_text())

    events: list[GroundTruthEvent] = []
    for ann in raw.get("annotations", []):
        half = int(ann["gameTime"].split("-")[0])
        position_ms = float(ann["position"])
        t = position_ms / 1000.0
        frame_idx = math.floor(t * fps + 0.5)  # round-half-up, see docstring
        events.append(GroundTruthEvent(class_=ann["label"], frame_idx=frame_idx, t=t, half=half))

    return EventGroundTruth(source="soccernet-ball", sequence=sequence, fps=fps, events=events)


def ingest_soccernet_ball(
    root: Path,
    split: str = "test",
    limit: int | None = None,
    sequences: list[str] | None = None,
    role: str = "tuning",
) -> list[tuple[int, str]]:
    from matchlab_core.video import probe
    from matchlab_server.api.videos import register_video_file
    from matchlab_server.db import init_db, session
    from matchlab_server.settings import get_settings

    from matchlab_train.datasets.manifest import update_tier_manifest

    root = Path(root)
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"SoccerNet ball-action split dir not found: {split_dir}")

    match_dirs = sorted(
        d for d in split_dir.iterdir() if d.is_dir() and (d / "Labels-ball.json").is_file()
    )
    if sequences:
        wanted = set(sequences)
        match_dirs = [d for d in match_dirs if d.name in wanted]
    if limit:
        match_dirs = match_dirs[:limit]
    if not match_dirs:
        raise FileNotFoundError(f"No match directories with Labels-ball.json under {split_dir}")

    init_db()
    settings = get_settings()
    dest_dir = settings.videos_dir / "soccernet-ball"
    dest_dir.mkdir(parents=True, exist_ok=True)

    registered: list[tuple[int, str]] = []
    manifest_entries: list[dict] = []
    with session() as db:
        for match_dir in match_dirs:
            videos = sorted(p for ext in _VIDEO_EXTS for p in match_dir.glob(f"*{ext}"))
            if len(videos) != 1:
                raise ValueError(
                    f"Expected exactly one video file in {match_dir}, found "
                    f"{len(videos)} ({[v.name for v in videos]}) -- SoccerNet ball-action "
                    f"matches may actually ship one video per half; see this module's "
                    f"docstring (soccernet_ball.py) for the open concern."
                )
            src_video = videos[0]
            meta = probe(src_video)

            gt = load_soccernet_ball_labels(
                match_dir / "Labels-ball.json", fps=meta.fps, sequence=match_dir.name
            )
            mp4 = dest_dir / f"{match_dir.name}{src_video.suffix}"
            if not mp4.exists():
                shutil.copy2(src_video, mp4)
            gt_path = dest_dir / f"{match_dir.name}.gt.json"
            gt_path.write_text(gt.model_dump_json())

            video = register_video_file(db, mp4.name, mp4, meta)
            video.gt_path = str(gt_path)
            db.commit()
            registered.append((video.id, match_dir.name))
            manifest_entries.append(
                {"name": match_dir.name, "video": str(mp4), "gt": str(gt_path), "role": role}
            )
            print(
                f"registered video {video.id}: {match_dir.name} ({len(gt.events)} gt events)",
                flush=True,
            )

    update_tier_manifest(
        tier="soccernet-ball",
        dataset="soccernet-ball",
        source_split=split,
        entries=manifest_entries,
    )
    return registered
