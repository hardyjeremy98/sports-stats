"""Ground-truth tracking labels.

Ground truth belongs to a *video*, not a run: every run over a labelled clip is
scored against the same reference. The canonical on-disk form is a single JSON
file (`GroundTruth`) stored next to the video and pointed at by the server's
`videos.gt_path` column; `load_soccernet_sequence`, `load_sportsmot_sequence`,
and `load_soccertrack_sequence` each convert one dataset tier's on-disk layout
into that form.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from pydantic import BaseModel

from pitchlab_core.schemas.geometry import Box

# Normalized roles. SoccerNet's gameinfo uses "goalkeepers" for some sequences
# and marks non-participants (staff, photographers) as "other".
ROLES = ("player", "goalkeeper", "referee", "ball", "other")


class GroundTruthFrame(BaseModel):
    frame_idx: int  # 0-based, same convention as pipeline artifacts
    box: Box


class GroundTruthTrack(BaseModel):
    track_id: int
    role: str = "player"  # one of ROLES
    team: str | None = None  # "left" | "right" (camera-relative), None for refs/ball
    jersey: str | None = None  # jersey number when identified; letters = unidentified
    frames: list[GroundTruthFrame]


class GroundTruth(BaseModel):
    source: str = "unknown"  # e.g. "soccernet-tracking"
    sequence: str | None = None
    fps: float = 25.0
    width: int = 0
    height: int = 0
    seq_length: int = 0
    tracks: list[GroundTruthTrack] = []


def load_soccernet_sequence(seq_dir: str | Path) -> GroundTruth:
    """Parse a SoccerNet tracking sequence dir (gt/gt.txt + gameinfo.ini +
    seqinfo.ini) into a GroundTruth.

    gt.txt rows are MOT format: frame(1-based), track_id, x, y, w, h, conf, ...
    gameinfo.ini maps track ids to roles: "trackletID_5= player team right;75".
    """
    seq_dir = Path(seq_dir)
    seq_info = _read_ini(seq_dir / "seqinfo.ini")
    game_info = _read_ini(seq_dir / "gameinfo.ini")

    meta: dict[int, tuple[str, str | None, str | None]] = {}
    for key, value in game_info.items():
        if not key.lower().startswith("trackletid_"):
            continue
        tid = int(key.split("_", 1)[1])
        meta[tid] = _parse_role(value)

    frames_by_track: dict[int, list[GroundTruthFrame]] = {}
    for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame, tid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        frames_by_track.setdefault(tid, []).append(
            GroundTruthFrame(frame_idx=frame - 1, box=Box(x1=x, y1=y, x2=x + w, y2=y + h))
        )

    tracks = []
    for tid in sorted(frames_by_track):
        role, team, jersey = meta.get(tid, ("player", None, None))
        frames = sorted(frames_by_track[tid], key=lambda f: f.frame_idx)
        tracks.append(
            GroundTruthTrack(track_id=tid, role=role, team=team, jersey=jersey, frames=frames)
        )

    return GroundTruth(
        source="soccernet-tracking",
        sequence=str(seq_info.get("name", seq_dir.name)),
        fps=float(seq_info.get("framerate", 25)),
        width=int(seq_info.get("imwidth", 0)),
        height=int(seq_info.get("imheight", 0)),
        seq_length=int(seq_info.get("seqlength", 0)),
        tracks=tracks,
    )


def load_sportsmot_sequence(seq_dir: str | Path) -> GroundTruth:
    """Parse a SportsMOT sequence dir (seqinfo.ini + gt/gt.txt) into a GroundTruth.

    SportsMOT is standard MOT17-style layout: gt.txt rows are
    `frame(1-based),track_id,x,y,w,h,conf,-1,-1,-1`. SportsMOT annotates
    players only -- no ball/referee distinction and no team labels -- so
    every track maps to role="player", team=None, jersey=None. Rows with
    conf == 0 are MOT's "ignore region" convention, not real detections, and
    are skipped.
    """
    seq_dir = Path(seq_dir)
    seq_info_path = seq_dir / "seqinfo.ini"
    if not seq_info_path.is_file():
        raise FileNotFoundError(f"SportsMOT seqinfo.ini not found: {seq_info_path}")
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"SportsMOT gt.txt not found: {gt_path}")

    seq_info = _read_ini(seq_info_path)

    frames_by_track: dict[int, list[GroundTruthFrame]] = {}
    for line in gt_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        frame, tid = int(parts[0]), int(parts[1])
        x, y, w, h = (float(v) for v in parts[2:6])
        conf = float(parts[6])
        if conf == 0:
            continue
        frames_by_track.setdefault(tid, []).append(
            GroundTruthFrame(frame_idx=frame - 1, box=Box(x1=x, y1=y, x2=x + w, y2=y + h))
        )

    tracks = []
    for tid in sorted(frames_by_track):
        frames = sorted(frames_by_track[tid], key=lambda f: f.frame_idx)
        tracks.append(
            GroundTruthTrack(track_id=tid, role="player", team=None, jersey=None, frames=frames)
        )

    return GroundTruth(
        source="sportsmot",
        sequence=str(seq_info.get("name", seq_dir.name)),
        fps=float(seq_info.get("framerate", 25)),
        width=int(seq_info.get("imwidth", 0)),
        height=int(seq_info.get("imheight", 0)),
        seq_length=int(seq_info.get("seqlength", 0)),
        tracks=tracks,
    )


# SoccerTrack ball TeamID and the synthetic track_id assigned to it. Player
# track_ids are `team_id * 1000 + player_id`, which never collides with this
# since player team_ids are only ever 0 or 1 (max player track_id < 2000).
_SOCCERTRACK_BALL_TEAM_ID = 3
_SOCCERTRACK_BALL_TRACK_ID = 9999
_SOCCERTRACK_ATTRS = ("bb_left", "bb_top", "bb_width", "bb_height")


def load_soccertrack_sequence(
    csv_path: str | Path, *, fps: float, width: int, height: int
) -> GroundTruth:
    """Parse a SoccerTrack (Scott et al. / sportslabkit) bounding-box CSV into
    a GroundTruth.

    SoccerTrack CSVs carry no seqinfo -- fps/width/height must be supplied by
    the caller. The file has a 3-row header (row 1 TeamID, row 2 PlayerID,
    row 3 attribute name) followed by data rows whose first column is the
    frame number (already 0-based; SoccerTrack does not use MOT's 1-based
    convention) and whose remaining columns come in groups of 4:
    bb_left, bb_top, bb_width, bb_height for one (team, player).

    TeamID 0/1 map to team="left"/"right" -- an arbitrary-but-stable
    convention (SoccerTrack team ids are NOT camera-relative the way
    SoccerNet's are). TeamID 3 (labelled BALL in some releases) maps to
    role="ball", team=None. track_id is deterministic and collision-free:
    `team_id * 1000 + player_id` for players, and a fixed 9999 for the ball.
    Empty cells (player not visible that frame) emit no GroundTruthFrame;
    float parsing is NaN-safe (an all-NaN cell group is treated as empty).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"SoccerTrack CSV not found: {csv_path}")

    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 3:
        raise ValueError(
            f"SoccerTrack CSV header malformed (expected 3 header rows, found "
            f"{len(rows)} total row(s)): {csv_path}"
        )
    team_row, player_row, attr_row = rows[0], rows[1], rows[2]
    n_cols = len(team_row)
    if len(player_row) != n_cols or len(attr_row) != n_cols:
        raise ValueError(
            f"SoccerTrack CSV header rows have mismatched column counts "
            f"(team={n_cols}, player={len(player_row)}, attr={len(attr_row)}): {csv_path}"
        )
    if n_cols < 5 or (n_cols - 1) % 4 != 0:
        raise ValueError(
            f"SoccerTrack CSV header column count ({n_cols}) is not "
            f"1 (frame) + a positive multiple of 4 (bbox fields): {csv_path}"
        )

    groups: list[tuple[int, int]] = []  # (team_id, player_id) per 4-column group, in order
    for g in range((n_cols - 1) // 4):
        base = 1 + g * 4
        teams, players = team_row[base : base + 4], player_row[base : base + 4]
        attrs = [a.strip().lower() for a in attr_row[base : base + 4]]
        if len(set(teams)) != 1 or len(set(players)) != 1:
            raise ValueError(
                f"SoccerTrack CSV header group {g} has inconsistent TeamID/PlayerID "
                f"across its 4 columns: {csv_path}"
            )
        if tuple(attrs) != _SOCCERTRACK_ATTRS:
            raise ValueError(
                f"SoccerTrack CSV header group {g} attribute columns are not "
                f"{_SOCCERTRACK_ATTRS} (got {tuple(attrs)}): {csv_path}"
            )
        try:
            team_id, player_id = int(float(teams[0])), int(float(players[0]))
        except ValueError as exc:
            raise ValueError(
                f"SoccerTrack CSV header group {g} has non-numeric TeamID/PlayerID: {csv_path}"
            ) from exc
        if team_id not in (0, 1, _SOCCERTRACK_BALL_TEAM_ID):
            raise ValueError(
                f"SoccerTrack CSV header group {g} has unexpected TeamID {team_id} "
                f"(expected 0, 1, or {_SOCCERTRACK_BALL_TEAM_ID} for ball): {csv_path}"
            )
        groups.append((team_id, player_id))

    frames_by_track: dict[int, list[GroundTruthFrame]] = {}
    track_meta: dict[int, tuple[str, str | None]] = {}
    n_data_rows = 0
    for row in rows[3:]:
        if not row or not row[0].strip():
            continue
        n_data_rows += 1
        frame_idx = int(float(row[0]))
        for g, (team_id, player_id) in enumerate(groups):
            base = 1 + g * 4
            cell = row[base : base + 4]
            if len(cell) < 4 or any(not v.strip() for v in cell):
                continue  # player absent that frame
            try:
                x, y, w, h = (float(v) for v in cell)
            except ValueError as exc:
                raise ValueError(
                    f"SoccerTrack CSV row has non-numeric bbox value in group {g}: "
                    f"{csv_path}: {row!r}"
                ) from exc
            if any(math.isnan(v) for v in (x, y, w, h)):
                continue  # NaN-safe: treat as absent, same as an empty cell

            if team_id == _SOCCERTRACK_BALL_TEAM_ID:
                tid, role, team = _SOCCERTRACK_BALL_TRACK_ID, "ball", None
            else:
                tid = team_id * 1000 + player_id
                role, team = "player", ("left" if team_id == 0 else "right")
            track_meta[tid] = (role, team)
            frames_by_track.setdefault(tid, []).append(
                GroundTruthFrame(frame_idx=frame_idx, box=Box(x1=x, y1=y, x2=x + w, y2=y + h))
            )

    tracks = []
    for tid in sorted(frames_by_track):
        role, team = track_meta[tid]
        frames = sorted(frames_by_track[tid], key=lambda f: f.frame_idx)
        tracks.append(
            GroundTruthTrack(track_id=tid, role=role, team=team, jersey=None, frames=frames)
        )

    return GroundTruth(
        source="soccertrack",
        sequence=csv_path.stem,
        fps=float(fps),
        width=int(width),
        height=int(height),
        seq_length=n_data_rows,
        tracks=tracks,
    )


def _read_ini(path: Path) -> dict[str, str]:
    """Flat key=value parse (configparser chokes on the duplicate [Sequence]
    headers some gameinfo files have). Keys lowercased except trackletID_N,
    which we need verbatim for the id suffix."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        out[key if key.lower().startswith("trackletid_") else key.lower()] = value.strip()
    return out


def _parse_role(value: str) -> tuple[str, str | None, str | None]:
    """'player team left;10' -> ('player', 'left', '10');
    'referee;side top' -> ('referee', None, None); 'ball;1' -> ('ball', None, None)."""
    desc, _, tag = value.strip().partition(";")
    words = desc.strip().split()
    role = words[0].rstrip("s") if words else "other"  # "goalkeepers" -> "goalkeeper"
    if role not in ROLES:
        role = "other"
    team = None
    if "team" in words:
        side = words[words.index("team") + 1] if words.index("team") + 1 < len(words) else None
        team = side if side in ("left", "right") else None
    jersey = tag.strip() or None
    if role in ("referee", "ball", "other"):
        jersey = None  # tag is a position/count there, not a jersey
    return role, team, jersey
