"""Random-access clip reading for FOOTPASS match video.

Encodes four findings verified on VAL on 2026-07-27. Three of them contradict what a
reasonable reading of the reference would assume, and every one of them would corrupt
training silently rather than loudly:

1. **The zips are AES-encrypted.** `unzip` fails with "unsupported compression method
   99". Extract with `7z x -p"$SOCCERNET_NDA_PASSWORD"`.
2. **One mp4 per MATCH, not per half, and frame indices are continuous across the
   halves.** `game_18_H1` spans 32-75,307 and `game_18_H2` spans 75,525-149,181 with
   no reset. So the tactical `frame` column indexes the single per-match mp4
   directly. Code that offsets H2 frames reads the wrong pixels.
3. **Video is already 640x352 @ 25 fps**, matching the model's (B,3,T,352,640) input
   with no resize.
4. **ROI scaling is x/3.0, y/3.068181** from full-HD. Verified by decoding 12 random
   action rows and confirming each scaled crop contains a centred player.

Decoding uses OpenCV rather than the reference's `decord` (not packaged for this
environment). Measured on `game_18.mp4`: a seek plus 50 sequential frames takes
0.04 s, so clip I/O is not a training bottleneck and needs no frame cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 352
EXPECTED_FPS = 25.0


def match_video_path(video_root: str | Path, half_key: str) -> Path:
    """`("videos_352x640", "game_18_H2")` -> `videos_352x640/game_18.mp4`.

    Both halves of a match share one file -- see finding 2.
    """
    game_id, _, _ = half_key.rpartition("_H")
    return Path(video_root) / f"{game_id or half_key}.mp4"


class MatchVideo:
    """A seekable handle on one match's mp4.

    Not thread-safe and not fork-safe: a `cv2.VideoCapture` cannot be shared across
    DataLoader workers, so each worker must open its own (see `footpass_clips`).
    """

    def __init__(self, path: str | Path, *, check_properties: bool = True) -> None:
        import cv2

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"match video not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open {self.path}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if check_properties:
            self._check_properties()

    def _check_properties(self) -> None:
        if (self.width, self.height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
            raise ValueError(
                f"{self.path.name} is {self.width}x{self.height}, expected "
                f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT} -- ROI scaling assumes the "
                f"pre-scaled 352x640 release, not full HD"
            )
        if abs(self.fps - EXPECTED_FPS) > 0.01:
            raise ValueError(
                f"{self.path.name} is {self.fps} fps, expected {EXPECTED_FPS} -- "
                f"the tactical frame column indexes 25 fps frames"
            )

    def read_clip(self, start_frame: int, length: int) -> np.ndarray:
        """`length` consecutive frames from `start_frame`, as uint8 RGB (T, H, W, 3).

        Raises rather than padding a short read: a silently truncated clip would
        train the model on a mislabelled time axis.
        """
        import cv2

        if start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {start_frame}")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        frames = []
        for _ in range(length):
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(
                    f"{self.path.name}: read failed at frame "
                    f"{start_frame + len(frames)} of clip [{start_frame}, "
                    f"{start_frame + length}) -- video has {self.frame_count} frames"
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> MatchVideo:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
