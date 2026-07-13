from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pitchlab_core.schemas.run import VideoMeta


@dataclass
class Frame:
    frame_idx: int  # index in the SOURCE video (stride-independent)
    t: float
    image: np.ndarray  # BGR


def probe(path: str | Path, sample_stride: int = 1) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return VideoMeta(
        path=str(path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_s=frame_count / fps if fps else 0.0,
        sample_stride=sample_stride,
    )


def iter_frames(
    meta: VideoMeta,
    stride: int = 1,
    max_frames: int | None = None,
    resize_width: int | None = None,
) -> Iterator[Frame]:
    """Decode frames sequentially, yielding every `stride`-th frame.

    Sequential decode + skip is used instead of seeking: cv2 seeks are unreliable
    on variable-framerate phone footage, and offline throughput is what matters.
    """
    cap = cv2.VideoCapture(meta.path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {meta.path}")
    try:
        idx = 0
        yielded = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                if resize_width and image.shape[1] > resize_width:
                    scale = resize_width / image.shape[1]
                    image = cv2.resize(image, (resize_width, int(image.shape[0] * scale)))
                yield Frame(frame_idx=idx, t=idx / meta.fps, image=image)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Thin mp4 writer. cv2 often lacks a working h264 encoder, so on close()
    the mp4v output is remuxed to browser-playable h264 via the ffmpeg binary
    when available (it ships in the backend docker image)."""

    def __init__(self, path: str | Path, fps: float, size_wh: tuple[int, int]):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size_wh
        )

    def write(self, image: np.ndarray) -> None:
        self._writer.write(image)

    def close(self) -> None:
        self._writer.release()
        transcode_h264(self._path)


def transcode_h264(path: Path) -> bool:
    """Re-encode in place to h264 + faststart for browser <video> playback.
    No-op (returns False) when ffmpeg is missing."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    tmp = path.with_suffix(".h264.mp4")
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
         "-c:v", "libx264", "-preset", "fast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        capture_output=True,
    )
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True
