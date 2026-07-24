"""Oracle TEAM stage: GT roles/teams as a controlled benchmark input (same
philosophy as oracle detections — GT consumed as instrumentation, never as
perception). Isolates associate-layer benchmarks from kit-color noise."""

from __future__ import annotations

import json
from dataclasses import dataclass

from matchlab_core.registry import build
from matchlab_core.schemas import Box, Team, Tracklet
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.run import StageKind
from matchlab_core.schemas.tracks import TrackletFrame


@dataclass
class _FakeVideo:
    path: str = ""
    fps: float = 25.0
    width: int = 1920
    height: int = 1080


@dataclass
class _FakeCtx:
    video: _FakeVideo


def _tracklet(tid: int, x: float, frames: range) -> Tracklet:
    return Tracklet(
        tracklet_id=tid,
        cls=DetectionClass.PLAYER,
        frames=[
            TrackletFrame(
                frame_idx=fi, box=Box(x1=x, y1=0, x2=x + 20, y2=40), confidence=1.0
            )
            for fi in frames
        ],
    )


def _gt_track(track_id: int, role: str, team: str | None, x: float) -> dict:
    return {
        "track_id": track_id,
        "role": role,
        "team": team,
        "jersey": None,
        "frames": [
            {"frame_idx": fi, "box": {"x1": x, "y1": 0, "x2": x + 20, "y2": 40}}
            for fi in range(0, 10)
        ],
    }


def test_oracle_team_maps_gt_sides_and_roles(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    (tmp_path / "clip.gt.json").write_text(
        json.dumps(
            {
                "source": "test",
                "tracks": [
                    _gt_track(1, "player", "left", 0.0),
                    _gt_track(2, "player", "right", 100.0),
                    _gt_track(3, "referee", None, 200.0),
                ],
            }
        )
    )
    tracklets = [
        _tracklet(10, 0.0, range(0, 10)),  # overlaps GT left player
        _tracklet(11, 100.0, range(0, 10)),  # overlaps GT right player
        _tracklet(12, 200.0, range(0, 10)),  # overlaps GT referee
        _tracklet(13, 900.0, range(0, 10)),  # matches nothing
    ]
    stage = build(StageKind.TEAM, "oracle", {})
    out = stage.classify(_FakeCtx(video=_FakeVideo(path=str(video))), tracklets)

    by_tid = {a.tracklet_id: a for a in out}
    assert by_tid[10].team == Team.HOME  # left -> HOME
    assert by_tid[11].team == Team.AWAY  # right -> AWAY
    assert by_tid[12].team == Team.REFEREE
    assert by_tid[13].team == Team.UNKNOWN  # missing evidence is neutral
    assert by_tid[10].confidence == 1.0
