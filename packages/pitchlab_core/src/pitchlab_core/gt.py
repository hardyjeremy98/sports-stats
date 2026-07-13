"""Ground-truth tracking labels.

Ground truth belongs to a *video*, not a run: every run over a labelled clip is
scored against the same reference. The canonical on-disk form is a single JSON
file (`GroundTruth`) stored next to the video and pointed at by the server's
`videos.gt_path` column; `load_soccernet_sequence` converts a SoccerNet
tracking sequence directory (MOTChallenge layout) into that form.
"""

from __future__ import annotations

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
