"""The `(9, 26, T)` stage-1 -> stage-2 contract.

This array is the entire interface between the visual action head and the sequence
denoiser: 9 class logits for each of the 26 tactical role slots at each video frame,
fp16, indexed by ABSOLUTE video frame so a half's array can be sliced by the same
frame numbers the tactical HDF5 uses.

Keeping it a frozen, on-disk contract is what makes the two stages independently
trainable and independently replaceable -- Phase 3 swaps MatchDay tracking and role
assignment in behind exactly this shape, without the denoiser knowing.

Slots with no observed player at a frame are all-zero rather than argmax-background.
Zero is distinguishable from a confident background prediction, which matters because
the sequence stage's job includes inferring actions for players it cannot see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from matchlab_core.pcbas.schema import N_CLASSES, N_SLOTS

LOGITS_DTYPE = np.float16


def empty_logits(n_frames: int) -> np.ndarray:
    return np.zeros((N_CLASSES, N_SLOTS, n_frames), dtype=LOGITS_DTYPE)


def validate_logits(array: np.ndarray) -> None:
    """Raise unless `array` satisfies the contract."""
    if array.ndim != 3:
        raise ValueError(f"logits must be 3-D (9, 26, T), got {array.shape}")
    if array.shape[0] != N_CLASSES or array.shape[1] != N_SLOTS:
        raise ValueError(
            f"logits must be ({N_CLASSES}, {N_SLOTS}, T), got {array.shape}"
        )
    if array.dtype != LOGITS_DTYPE:
        raise ValueError(f"logits must be {LOGITS_DTYPE}, got {array.dtype}")


def save_logits(array: np.ndarray, path: str | Path) -> None:
    validate_logits(array)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.save(f, array)


def load_logits(path: str | Path) -> np.ndarray:
    array = np.load(Path(path))
    validate_logits(array)
    return array


def logits_filename(half_key: str) -> str:
    """The reference's naming, kept so its artifacts and ours are interchangeable."""
    return f"avg_logits_{half_key}.npy"


class WindowAccumulator:
    """Averages overlapping sliding-window predictions into one per-frame array.

    The reference runs two tilings 25 frames apart and averages them, which is the
    same thing as a stride-25 sliding window averaged over its covering windows --
    except that this divides each frame by how many windows ACTUALLY covered it.
    At the sequence boundaries that is one, not two, so an unconditional /2 would
    halve the logits for the first and last 25 frames of every half and hand the
    denoiser a systematically under-confident edge.
    """

    def __init__(self, n_frames: int, offset: int = 0) -> None:
        self.offset = offset
        self._sum = np.zeros((N_CLASSES, N_SLOTS, n_frames), dtype=np.float32)
        self._count = np.zeros(n_frames, dtype=np.int32)

    def add(self, window: np.ndarray, start_frame: int) -> None:
        """`window` is (9, 26, T) for absolute frames [start_frame, start_frame+T)."""
        lo = start_frame - self.offset
        hi = lo + window.shape[2]
        if lo < 0 or hi > self._sum.shape[2]:
            raise ValueError(
                f"window [{start_frame}, {start_frame + window.shape[2]}) falls "
                f"outside the accumulator's frame range"
            )
        self._sum[:, :, lo:hi] += window.astype(np.float32)
        self._count[lo:hi] += 1

    def result(self) -> np.ndarray:
        counts = np.maximum(self._count, 1)
        return (self._sum / counts).astype(LOGITS_DTYPE)

    @property
    def coverage(self) -> np.ndarray:
        """How many windows covered each frame. Zero means never predicted."""
        return self._count.copy()


def window_starts(first_frame: int, last_frame: int, length: int, stride: int) -> list[int]:
    """Window starts covering `[first_frame, last_frame]` inclusive.

    The final window is pulled back so it ends exactly on `last_frame` rather than
    running past the end of the video -- reading past the end raises in
    `MatchVideo.read_clip`, deliberately, so it must not be asked to.
    """
    if last_frame < first_frame:
        return []
    span = last_frame - first_frame + 1
    if span <= length:
        return [first_frame]
    starts = list(range(first_frame, last_frame - length + 2, stride))
    final = last_frame - length + 1
    if starts[-1] != final:
        starts.append(final)
    return starts
