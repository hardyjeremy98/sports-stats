"""Deterministic match simulator.

Backs the keyless/GPU-less dev loop: the `synthetic` detector emits detections
from this simulation, and `pitchlab_server.demo` renders a matching video from
the same seed — so a full pipeline run (track → team → associate → fuse →
events) produces coherent, watchable output on a laptop.

Everything is seeded from the video filename: any component that derives its
sim from the same video gets identical states.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from pitchlab_core.pitch import SOCCER_PITCH, PitchSpec


@dataclass
class SimPlayer:
    pid: int
    team: int  # 0 home, 1 away
    x: float   # pitch cm
    y: float


@dataclass
class SimState:
    frame_idx: int
    players: list[SimPlayer]
    ball_x: float
    ball_y: float
    possessor_pid: int | None


def seed_from_name(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# 4-4-2 formation anchors as (x_frac, y_frac) of pitch length/width.
_FORMATION = [
    (0.06, 0.50),
    (0.22, 0.18), (0.22, 0.40), (0.22, 0.60), (0.22, 0.82),
    (0.45, 0.15), (0.45, 0.40), (0.45, 0.60), (0.45, 0.85),
    (0.70, 0.35), (0.70, 0.65),
]


class MatchSim:
    """Players jitter around formation anchors; the ball is passed between
    teammates with occasional turnovers. Precomputes all frames (cheap)."""

    def __init__(
        self,
        seed: int,
        fps: float,
        n_frames: int,
        pitch: PitchSpec = SOCCER_PITCH,
        n_per_team: int = 11,
    ):
        self.pitch = pitch
        self.fps = fps
        self.n_frames = n_frames
        rng = np.random.default_rng(seed)

        anchors = []
        teams = []
        pids = []
        for team in (0, 1):
            for i in range(n_per_team):
                fx, fy = _FORMATION[i % len(_FORMATION)]
                x = fx * pitch.length if team == 0 else (1 - fx) * pitch.length
                y = fy * pitch.width
                anchors.append((x, y))
                teams.append(team)
                pids.append(team * 100 + i)
        self.pids = pids
        self.teams = teams
        n = len(pids)

        # Smooth random walk: velocity OU-process around anchor pull.
        pos = np.array(anchors, dtype=np.float64)
        vel = np.zeros((n, 2))
        anchors_np = np.array(anchors, dtype=np.float64)
        self._pos = np.zeros((n_frames, n, 2))
        max_speed = 600 / fps  # ~6 m/s in cm per frame
        for f in range(n_frames):
            pull = (anchors_np - pos) * 0.002
            noise = rng.normal(0, 60 / fps, size=(n, 2))
            vel = 0.95 * vel + pull + noise
            speed = np.linalg.norm(vel, axis=1, keepdims=True)
            vel = np.where(speed > max_speed, vel * max_speed / (speed + 1e-9), vel)
            pos = pos + vel
            pos[:, 0] = np.clip(pos[:, 0], 100, pitch.length - 100)
            pos[:, 1] = np.clip(pos[:, 1], 100, pitch.width - 100)
            self._pos[f] = pos

        # Ball: hold -> pass -> travel -> hold. Record possessor per frame.
        self._ball = np.zeros((n_frames, 2))
        self._possessor: list[int | None] = [None] * n_frames
        holder = int(rng.integers(0, n))
        f = 0
        while f < n_frames:
            hold = int(rng.integers(int(0.8 * fps), int(2.5 * fps)))
            for k in range(min(hold, n_frames - f)):
                self._ball[f + k] = self._pos[f + k, holder]
                self._possessor[f + k] = self.pids[holder]
            f += hold
            if f >= n_frames:
                break
            # Choose receiver: mostly a teammate (completed pass), sometimes
            # an opponent (turnover -> a "missed pass" for the event engine).
            same_team = [i for i in range(n) if self.teams[i] == self.teams[holder] and i != holder]
            other_team = [i for i in range(n) if self.teams[i] != self.teams[holder]]
            receiver = int(
                rng.choice(same_team) if rng.random() > 0.25 else rng.choice(other_team)
            )
            start = self._pos[f - 1, holder].copy()
            travel = int(rng.integers(int(0.3 * fps), int(0.9 * fps)))
            for k in range(min(travel, n_frames - f)):
                target = self._pos[min(f + k, n_frames - 1), receiver]
                a = (k + 1) / travel
                self._ball[f + k] = start + a * (target - start)
                self._possessor[f + k] = None
            f += travel
            holder = receiver

    def state_at(self, frame_idx: int) -> SimState:
        f = min(frame_idx, self.n_frames - 1)
        players = [
            SimPlayer(pid=self.pids[i], team=self.teams[i], x=self._pos[f, i, 0], y=self._pos[f, i, 1])
            for i in range(len(self.pids))
        ]
        return SimState(
            frame_idx=frame_idx,
            players=players,
            ball_x=float(self._ball[f, 0]),
            ball_y=float(self._ball[f, 1]),
            possessor_pid=self._possessor[f],
        )


def demo_camera(pitch: PitchSpec, img_w: int, img_h: int) -> np.ndarray:
    """Fixed pitch→image homography for the synthetic world: a mild trapezoid
    that looks like an elevated sideline camera seeing the whole pitch."""
    import cv2

    src = np.array(
        [[0, 0], [pitch.length, 0], [pitch.length, pitch.width], [0, pitch.width]],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [img_w * 0.15, img_h * 0.18],
            [img_w * 0.85, img_h * 0.18],
            [img_w * 0.98, img_h * 0.92],
            [img_w * 0.02, img_h * 0.92],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def project(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = h @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])
