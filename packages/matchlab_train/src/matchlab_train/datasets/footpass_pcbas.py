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

import json
from pathlib import Path
from typing import Literal

import numpy as np
from matchlab_core.pcbas.events import PCBASEvent, PCBASEvents
from matchlab_core.pcbas.schema import (
    BACKGROUND,
    CLASS_NAMES,
    CLS,
    FRAME,
    LEFT_TO_RIGHT,
    N_COLUMNS_LABELLED,
    N_SLOTS,
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
from pydantic import BaseModel

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


class PCBASSplitStats(BaseModel):
    """What one split's HDF5 actually contains -- the Phase 0 ingest gate.

    `frac_rows_with_bbox` and `frac_events_with_bbox` are deliberately separate.
    They are very different numbers (roughly 41% of ROWS carry a box against 82.5%
    of EVENTS), and conflating them would either overstate or understate what a
    purely visual model can reach.
    """

    path: str
    labelled: bool
    n_columns: int
    n_halves: int
    keys: list[str]
    n_rows: int
    n_events: int
    events_per_class: dict[str, int]
    rows_with_bbox: int
    frac_rows_with_bbox: float
    events_with_bbox: int
    frac_events_with_bbox: float
    slots_seen: list[int]
    frame_range_per_half: dict[str, tuple[int, int]]


def split_stats(h5_path: str | Path) -> PCBASSplitStats:
    """Summarise a whole split without materialising every half at once."""
    keys = list_halves(h5_path)
    n_columns = 0
    n_rows = n_events = rows_with_bbox = events_with_bbox = 0
    per_class: dict[str, int] = {}
    slots: set[int] = set()
    frame_range: dict[str, tuple[int, int]] = {}

    for key in keys:
        arr = load_half(h5_path, key)
        n_columns = arr.shape[1]
        n_rows += arr.shape[0]
        has_box = ~np.isnan(arr[:, ROI_X])
        rows_with_bbox += int(has_box.sum())
        frames = arr[:, FRAME]
        if len(frames):
            frame_range[key] = (int(np.nanmin(frames)), int(np.nanmax(frames)))
        for row in arr:
            slot = _row_slot(row)
            if slot is not None:
                slots.add(slot)
        if n_columns >= N_COLUMNS_LABELLED:
            is_event = ~np.isnan(arr[:, CLS]) & (arr[:, CLS] != BACKGROUND)
            n_events += int(is_event.sum())
            events_with_bbox += int((is_event & has_box).sum())
            for cls_value in arr[is_event, CLS]:
                name = CLASS_NAMES[int(cls_value)]
                per_class[name] = per_class.get(name, 0) + 1

    return PCBASSplitStats(
        path=str(h5_path),
        labelled=n_columns >= N_COLUMNS_LABELLED,
        n_columns=n_columns,
        n_halves=len(keys),
        keys=keys,
        n_rows=n_rows,
        n_events=n_events,
        events_per_class=dict(sorted(per_class.items())),
        rows_with_bbox=rows_with_bbox,
        frac_rows_with_bbox=rows_with_bbox / n_rows if n_rows else 0.0,
        events_with_bbox=events_with_bbox,
        frac_events_with_bbox=events_with_bbox / n_events if n_events else 0.0,
        slots_seen=sorted(slots),
        frame_range_per_half=frame_range,
    )


def load_playbyplay(
    path: str | Path, kind: Literal["gt", "pred"]
) -> dict[str, list[PCBASEvent]]:
    """Read the reference's playbyplay JSON exchange format, keyed by half.

    The two row layouts differ in a way that a width sniff would get silently wrong
    -- GT column 4 is a bbox FLAG, prediction column 4 is a SCORE -- so `kind` is
    required rather than inferred:

        GT   [frame, team, shirt, class, bbox, replay]
        pred [frame, team, shirt, class, score]

    These rows are SHIRT-native and carry no role, so the returned events have no
    `slot`. Score them with `score_events(..., identity="shirt")`.
    """
    raw = json.loads(Path(path).read_text())
    out: dict[str, list[PCBASEvent]] = {}
    for key in (str(k) for k in raw.get("keys", [])):
        events: list[PCBASEvent] = []
        for row in raw["events"].get(key, []):
            if len(row) < 4:
                continue
            common = {
                "frame_idx": int(row[0]),
                "left_to_right": int(row[1]),
                "shirt_number": int(row[2]),
                "class_id": int(row[3]),
            }
            if kind == "gt":
                events.append(
                    PCBASEvent(
                        **common,
                        score=1.0,
                        has_bbox=bool(int(row[4])) if len(row) > 4 else None,
                        is_replay=bool(int(row[5])) if len(row) > 5 else None,
                    )
                )
            else:
                events.append(
                    PCBASEvent(**common, score=float(row[4]) if len(row) > 4 else 1.0)
                )
        out[key] = events
    return out


def slot_shirt_table(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """`(26, n_frames)` int table of which shirt occupies each slot at each frame.

    This is the ADR 008 **export-time remap**, and the reason ADR 007's per-match
    bijection is impossible: a slot is a tactical role, so a substitution rebinds it
    mid-match and `left_to_right` inverts at half time. The mapping is therefore
    per frame, never per match.

    Gaps are forward-filled from the last known occupant and then backward-filled for
    the frames before anyone was seen -- a slot's occupant does not change while the
    camera is looking elsewhere. Slots never occupied stay -1.

    Where several rows claim one slot at one frame (substitution overlap) the LAST
    wins, matching the reference. Note that inference's `slot_rois_masks` keeps the
    FIRST for the box; the two are inconsistent in the reference and the
    inconsistency is reproduced rather than quietly fixed, because changing it would
    move the numbers away from the comparand.
    """
    table = np.full((N_SLOTS, n_frames), -1, dtype=np.int64)
    have_shirt = ~np.isnan(arr[:, SHIRT_NUMBER])
    for row in arr[have_shirt]:
        frame = row[FRAME]
        slot = _row_slot(row)
        if slot is None or np.isnan(frame):
            continue
        frame = int(frame)
        if 0 <= frame < n_frames:
            table[slot, frame] = int(row[SHIRT_NUMBER])

    for slot in range(N_SLOTS):
        last = -1
        for t in range(n_frames):
            if table[slot, t] == -1:
                table[slot, t] = last
            else:
                last = table[slot, t]
        nxt = -1
        for t in range(n_frames - 1, -1, -1):
            if table[slot, t] == -1:
                table[slot, t] = nxt
            else:
                nxt = table[slot, t]
    return table


def assign_shirts(events: list[PCBASEvent], table: np.ndarray) -> list[PCBASEvent]:
    """Attach a shirt number to each slot-native event, via the per-frame roster.

    Events keep their slot: the shirt is added, never substituted for it. An event
    whose slot was never occupied gets shirt -1 rather than being dropped, so a
    roster gap is visible in the output instead of silently shrinking the prediction
    set (which would flatter precision).
    """
    out: list[PCBASEvent] = []
    n_frames = table.shape[1]
    for e in events:
        shirt = -1
        if e.slot is not None and 0 <= e.frame_idx < n_frames:
            shirt = int(table[e.slot, e.frame_idx])
        out.append(e.model_copy(update={"shirt_number": shirt}))
    return out


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
