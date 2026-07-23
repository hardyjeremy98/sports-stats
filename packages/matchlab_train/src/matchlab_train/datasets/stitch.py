"""Shared frame-directory -> mp4 stitching for tracking-dataset ingest
(SoccerNet, SportsMOT: both ship raw `img1/NNNNNN.jpg` frame dumps rather
than pre-encoded video). Sequential frames are stitched into a
browser-playable h264 mp4 via the ffmpeg CLI rather than cv2: browsers
can't play cv2's mp4v, and yuv420p + libx264 is the compatibility
baseline. Requires ffmpeg on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def stitch_frames_to_mp4(img_dir: Path, fps: float, dest: Path, pattern: str = "%06d.jpg") -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(img_dir / pattern),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
