"""Render a synthetic match video from MatchSim. The `synthetic` detector
seeded with the same filename produces detections that match these pixels, so
`render_demo_video()` + `pipeline.stub.yaml` = a coherent end-to-end run with
zero models. Kits are drawn in two distinct colors so the kit-color team
classifier genuinely works on this footage."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from matchlab_core.pitch import SOCCER_PITCH, PitchSpec
from matchlab_core.sim import MatchSim, demo_camera, project, seed_from_name
from matchlab_core.video import VideoWriter

_KITS = {0: (200, 90, 30), 1: (40, 40, 200)}  # BGR: home teal-blue, away red


def render_demo_video(
    path: str | Path,
    duration_s: float = 60.0,
    fps: float = 25.0,
    width: int = 1280,
    height: int = 720,
    pitch: PitchSpec = SOCCER_PITCH,
) -> Path:
    path = Path(path)
    n_frames = int(duration_s * fps)
    sim = MatchSim(
        seed=seed_from_name(path.stem), fps=fps, n_frames=n_frames, pitch=pitch
    )
    cam = demo_camera(pitch, width, height)

    background = _render_background(pitch, cam, width, height)
    writer = VideoWriter(path, fps, (width, height))
    for f in range(n_frames):
        img = background.copy()
        state = sim.state_at(f)
        # Draw far-to-near so nearer players occlude farther ones.
        for sp in sorted(state.players, key=lambda s: project(cam, s.x, s.y)[1]):
            ix, iy = project(cam, sp.x, sp.y)
            h = 30 + 50 * (iy / height)
            w = h * 0.45
            kit = _KITS[sp.team]
            # Torso (kit color), head, legs — enough structure for crops.
            cv2.ellipse(img, (int(ix), int(iy - h * 0.55)), (int(w * 0.5), int(h * 0.3)),
                        0, 0, 360, kit, -1, cv2.LINE_AA)
            cv2.circle(img, (int(ix), int(iy - h * 0.85)), max(2, int(h * 0.12)),
                       (150, 190, 220), -1, cv2.LINE_AA)
            cv2.line(img, (int(ix - w * 0.15), int(iy - h * 0.25)),
                     (int(ix - w * 0.2), int(iy)), (30, 30, 30), 2, cv2.LINE_AA)
            cv2.line(img, (int(ix + w * 0.15), int(iy - h * 0.25)),
                     (int(ix + w * 0.2), int(iy)), (30, 30, 30), 2, cv2.LINE_AA)
        bx, by = project(cam, state.ball_x, state.ball_y)
        cv2.circle(img, (int(bx), int(by)), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(bx), int(by)), 6, (60, 60, 60), 1, cv2.LINE_AA)
        writer.write(img)
    writer.close()
    return path


def _render_background(pitch: PitchSpec, cam: np.ndarray, w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), (70, 90, 60), dtype=np.uint8)  # muddy surroundings
    # Grass with mow stripes, warped through the camera.
    grass = np.zeros((pitch.width // 10, pitch.length // 10, 3), dtype=np.uint8)
    stripe_w = grass.shape[1] // 12
    for i in range(12):
        shade = (52, 118, 46) if i % 2 == 0 else (60, 132, 52)
        grass[:, i * stripe_w : (i + 1) * stripe_w] = shade
    scale = np.diag([1 / 10, 1 / 10, 1.0])  # pitch cm -> grass px
    warped = cv2.warpPerspective(grass, cam @ np.linalg.inv(scale), (w, h),
                                 flags=cv2.INTER_LINEAR, borderValue=(70, 90, 60))
    mask = warped.any(axis=2)
    img[mask] = warped[mask]
    for a, b in pitch.edges:
        pa = project(cam, *pitch.vertices[a - 1])
        pb = project(cam, *pitch.vertices[b - 1])
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 (245, 245, 245), 2, cv2.LINE_AA)
    centre = [
        project(cam, pitch.length / 2 + pitch.centre_circle_radius * np.cos(a),
                pitch.width / 2 + pitch.centre_circle_radius * np.sin(a))
        for a in np.linspace(0, 2 * np.pi, 60)
    ]
    cv2.polylines(img, [np.array(centre, dtype=np.int32)], True, (245, 245, 245), 2, cv2.LINE_AA)
    return img
