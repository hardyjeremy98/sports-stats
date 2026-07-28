"""FOOTPASS tactical-data loader (SoccerNet SN-PCBAS-2026).

Schema and acquisition: `docs/reference/footpass-setup.md`. One HDF5 per split,
keyed `game_<id>_H<half>`; each value is an N x 14 float32 array of
per-player-per-frame rows (N x 13 for CHALLENGE, which drops CLS).

`ROI_X` is NaN when the player is not visible in frame -- 58% of rows on a
broadcast half. That flag is the whole point of this loader for B2: a player's
observable spans ARE their tracklets, so exit/re-entry pairs come straight out
of the data, over 90 minutes instead of SoccerNet's 30-second clips.

Position is present even when the player is off-camera. The deployed system
never has that, so anything measuring re-ID evidence must read positions only
inside observable spans -- see `matchlab_core.reid.occupancy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np


class COL:
    """Column indices, upstream `tactical_data_format.txt` order."""

    FRAME: Final = 0
    PLAYER_ID: Final = 1
    TEAM: Final = 2
    SHIRT: Final = 3
    ROLE: Final = 4
    X: Final = 5
    Y: Final = 6
    VX: Final = 7
    VY: Final = 8
    ROI_X: Final = 9
    ROI_Y: Final = 10
    ROI_W: Final = 11
    ROI_H: Final = 12
    CLS: Final = 13


ROLE_NAMES: Final[dict[int, str]] = {
    1: "GK",
    2: "LB",
    3: "LCB",
    4: "MCB",
    5: "RCB",
    6: "LM",
    7: "RM",
    8: "DM",
    9: "AM",
    10: "LW",
    11: "RW",
    12: "CF",
    13: "RB",
}


@dataclass
class FootpassHalf:
    game_id: str
    half: int
    rows: np.ndarray

    @property
    def player_ids(self) -> list[int]:
        return sorted({int(v) for v in self.rows[:, COL.PLAYER_ID]})

    def player_rows(self, player_id: int) -> np.ndarray:
        return self.rows[self.rows[:, COL.PLAYER_ID] == player_id]

    def role_of(self, player_id: int) -> int:
        r = self.player_rows(player_id)
        return int(r[0, COL.ROLE]) if len(r) else 0

    def team_of(self, player_id: int) -> int:
        r = self.player_rows(player_id)
        return int(r[0, COL.TEAM]) if len(r) else -1


def half_keys(path: str | Path) -> list[str]:
    import h5py

    with h5py.File(str(path), "r") as f:
        return sorted(f.keys())


def load_half(path: str | Path, key: str) -> FootpassHalf:
    import h5py

    game_id, _, half = key.rpartition("_H")
    with h5py.File(str(path), "r") as f:
        rows = np.asarray(f[key][:], dtype=np.float32)
    return FootpassHalf(game_id=game_id, half=int(half or 0), rows=rows)


def observable_spans(
    half: FootpassHalf, player_id: int, *, max_gap_frames: int = 2
) -> list[tuple[int, int]]:
    """Contiguous frame ranges where the player is visible in frame.

    Gaps of at most `max_gap_frames` are bridged, matching the GT-fragment
    harness's `gap_frames` so FOOTPASS fragments are constructed the same way
    SoccerNet ones are (`stages/track/oracle.py`). Returns inclusive
    (start_frame, end_frame) pairs, ascending.
    """
    rows = half.player_rows(player_id)
    if not len(rows):
        return []
    rows = rows[np.argsort(rows[:, COL.FRAME])]
    frames = rows[:, COL.FRAME][~np.isnan(rows[:, COL.ROI_X])].astype(int)
    if not len(frames):
        return []
    spans: list[tuple[int, int]] = []
    start = prev = int(frames[0])
    for raw in frames[1:]:
        f = int(raw)
        if f - prev > max_gap_frames + 1:
            spans.append((start, prev))
            start = f
        prev = f
    spans.append((start, prev))
    return spans
