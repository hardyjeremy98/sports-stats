"""FOOTPASS tactical HDF5 -> MatchLab schemas, for player-centric action spotting.

One HDF5 per split, keyed `game_<id>_H<half>`; each value is an (N, 14) array of
per-player-per-frame rows in the column order fixed by `matchlab_core.pcbas.schema`.

Deliberately separate from `datasets/footpass.py` (the re-ID observability loader):
that module answers "when was this player visible", this one answers "who did what,
when". They share only the column layout, which now lives in
`matchlab_core.pcbas.schema` -- collapse `footpass.py`'s duplicate `COL` into it when
the branches meet.

Two facts drive every function here:

* `roi_x` is NaN for 59% of rows -- the player is off-screen in the broadcast. Those
  rows still carry position, role and LABEL, so they must contribute events and
  roster entries while contributing no bounding box. Dropping them would silently
  delete 17.5% of the actions, which are exactly the ones only the sequence stage
  can recover.
* Frame indices are continuous ACROSS halves within a match (measured 2026-07-27:
  `game_18_H1` 32-75,307, `game_18_H2` 75,525-149,181), and there is one mp4 per
  match. Never offset H2 frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matchlab_core.pcbas.events import PCBASEvent, PCBASEvents
from matchlab_core.pcbas.schema import (
    BACKGROUND,
    CLS,
    FRAME,
    LEFT_TO_RIGHT,
    N_COLUMNS_LABELLED,
    PLAYER_ID,
    ROI_HEIGHT,
    ROI_SCALE_X,
    ROI_SCALE_Y,
    ROI_WIDTH,
    ROI_X,
    ROI_Y,
    ROLE_ID,
    SHIRT_NUMBER,
    slot_index,
)
from matchlab_core.schemas.detections import DetectionClass
from matchlab_core.schemas.geometry import Box
from matchlab_core.schemas.team import Team, TeamAssignment
from matchlab_core.schemas.tracks import Tracklet, TrackletFrame

DEFAULT_FPS = 25.0

# `left_to_right` is an ATTACKING DIRECTION, not a home/away fact, and it inverts at
# half time. Mapping it onto Team.HOME/AWAY is an arbitrary but consistent convention
# so downstream MatchLab code that expects a `Team` has something to read; it mirrors
# `stages/team/oracle.py`. Nothing may infer real home/away from it.
_TEAM_BY_SIDE = {0: Team.HOME, 1: Team.AWAY}


def list_halves(h5_path: str | Path) -> list[str]:
    """Half keys in the file, sorted. E.g. `["game_18_H1", "game_18_H2", ...]`."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        return sorted(f.keys())


def load_half(h5_path: str | Path, key: str) -> np.ndarray:
    """The (N, 14) row block for one half. (N, 13) on the unlabelled CHALLENGE split."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        return np.asarray(f[key][:], dtype=np.float64)


def parse_key(key: str) -> tuple[str, int]:
    """`"game_18_H2"` -> `("game_18", 2)`. Unrecognised keys get half 0."""
    game_id, sep, half = key.rpartition("_H")
    if not sep or not half.isdigit():
        return key, 0
    return game_id, int(half)


def _require_labels(arr: np.ndarray) -> None:
    if arr.shape[1] < N_COLUMNS_LABELLED:
        raise ValueError(
            f"array has {arr.shape[1]} columns -- this is the unlabelled CHALLENGE/TEST "
            f"split, which withholds the `class` column and cannot be scored locally"
        )


def _row_slot(row: np.ndarray) -> int | None:
    """Slot for a row, or None if side/role are missing or out of range."""
    ltr, role = row[LEFT_TO_RIGHT], row[ROLE_ID]
    if np.isnan(ltr) or np.isnan(role):
        return None
    try:
        return slot_index(int(ltr), int(role))
    except ValueError:
        return None


def half_to_tracklets(arr: np.ndarray) -> tuple[list[Tracklet], list[TeamAssignment]]:
    """One tracklet per player, containing only the frames where they are ON SCREEN.

    A player who is never on screen produces no tracklet at all -- an empty
    `Tracklet` would break `start_frame`/`end_frame`, and there is nothing for a
    visual model to pool. Their events survive via `half_to_events`.

    Boxes are scaled from full-HD into the 352x640 video the model consumes, exactly
    as the reference dataset does.
    """
    observed = arr[~np.isnan(arr[:, ROI_X])]
    tracklets: list[Tracklet] = []
    teams: list[TeamAssignment] = []
    for pid in sorted({int(v) for v in observed[:, PLAYER_ID]}):
        rows = observed[observed[:, PLAYER_ID] == pid]
        rows = rows[np.argsort(rows[:, FRAME])]
        frames = [
            TrackletFrame(
                frame_idx=int(r[FRAME]),
                box=Box(
                    x1=r[ROI_X] / ROI_SCALE_X,
                    y1=r[ROI_Y] / ROI_SCALE_Y,
                    x2=(r[ROI_X] + r[ROI_WIDTH]) / ROI_SCALE_X,
                    y2=(r[ROI_Y] + r[ROI_HEIGHT]) / ROI_SCALE_Y,
                ),
                confidence=1.0,
                source="observed",
            )
            for r in rows
        ]
        tracklets.append(
            Tracklet(tracklet_id=pid, cls=DetectionClass.PLAYER, frames=frames)
        )
        side = int(rows[0][LEFT_TO_RIGHT])
        teams.append(
            TeamAssignment(
                tracklet_id=pid,
                team=_TEAM_BY_SIDE.get(side, Team.UNKNOWN),
                confidence=1.0,
            )
        )
    return tracklets, teams


def half_to_events(
    arr: np.ndarray, key: str, *, fps: float = DEFAULT_FPS
) -> PCBASEvents:
    """Every non-background row becomes one `PCBASEvent`, on-screen or not."""
    _require_labels(arr)
    game_id, half = parse_key(key)
    events: list[PCBASEvent] = []
    labelled = arr[~np.isnan(arr[:, CLS]) & (arr[:, CLS] != BACKGROUND)]
    for row in labelled[np.argsort(labelled[:, FRAME])]:
        slot = _row_slot(row)
        if slot is None:
            continue
        frame_idx = int(row[FRAME])
        shirt = row[SHIRT_NUMBER]
        events.append(
            PCBASEvent(
                frame_idx=frame_idx,
                left_to_right=int(row[LEFT_TO_RIGHT]),
                role_id=int(row[ROLE_ID]),
                slot=slot,
                shirt_number=-1 if np.isnan(shirt) else int(shirt),
                class_id=int(row[CLS]),
                score=1.0,
                t=frame_idx / fps,
            )
        )
    return PCBASEvents(key=key, game_id=game_id, half=half, fps=fps, events=events)


def roster_lookup(arr: np.ndarray) -> dict[tuple[int, int], int]:
    """`(frame, slot) -> shirt_number`, the ADR 008 export-time remap table.

    Per frame rather than per match: substitutes reuse slots (~15% of VAL events sit
    after at least one substitution), so a match-level mapping is provably wrong.
    """
    lut: dict[tuple[int, int], int] = {}
    for row in arr:
        slot = _row_slot(row)
        shirt = row[SHIRT_NUMBER]
        if slot is None or np.isnan(shirt) or np.isnan(row[FRAME]):
            continue
        lut[(int(row[FRAME]), slot)] = int(shirt)
    return lut
