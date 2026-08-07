"""FOOTPASS / PCBAS tactical-data schema -- the single source of truth.

Column layout is quoted verbatim from the dataset's own `tactical_data_format.txt`:

    frame, player_id, left_to_right, shirt_number, role_id,
    x, y, speed_x, speed_y, roi_x, roi_y, roi_width, roi_height, class

TRAIN/VAL carry 14 columns; the CHALLENGE/TEST split carries 13 -- it has no `class`,
so its labels are withheld and it cannot be scored locally.

`roi_*` are FULL-HD pixel coordinates and are NaN when the player is off-screen in
the broadcast (59% of rows). The reference model consumes 352x640 video and divides
roi_x by 3 and roi_y by 3.068181 to match.

Role slots are TACTICAL, not roster slots -- see ADR 008. `left_to_right` is an
attacking direction that INVERTS at half time (measured 2026-07-27: 17 of 18 VAL
shirts swap sides between H1 and H2), and substitutes reuse a slot. A slot therefore
identifies "whoever is playing left-back for the left-attacking side right now",
never a person.

Measured on VAL 2026-07-27: only 11 of the 13 role ids appear per side --
role 4 (MCB) and role 8 (DM) are unused in these matches -- so 22 of 26 slots are
occupied. Code must never assume all 26 are filled.

`matchlab_train.datasets.footpass` (the re-ID observability loader, separate branch)
carries a duplicate `COL` class and `ROLE_NAMES`; collapse it into this module when
the branches meet.
"""

from __future__ import annotations

from typing import Final

# --- column indices ---
FRAME: Final = 0
PLAYER_ID: Final = 1
LEFT_TO_RIGHT: Final = 2
SHIRT_NUMBER: Final = 3
ROLE_ID: Final = 4
X_POS: Final = 5
Y_POS: Final = 6
X_SPEED: Final = 7
Y_SPEED: Final = 8
ROI_X: Final = 9
ROI_Y: Final = 10
ROI_WIDTH: Final = 11
ROI_HEIGHT: Final = 12
CLS: Final = 13

N_COLUMNS_LABELLED: Final = 14  # TRAIN / VAL
N_COLUMNS_UNLABELLED: Final = 13  # CHALLENGE / TEST -- no `class`

# --- classes ---
# FROZEN ORDER. Index is the on-disk label value; reordering invalidates every
# checkpoint and every reported number.
CLASS_NAMES: Final[list[str]] = [
    "background",
    "drive",
    "pass",
    "cross",
    "throw-in",
    "shot",
    "header",
    "tackle",
    "block",
]
N_CLASSES: Final = len(CLASS_NAMES)
BACKGROUND: Final = 0
ACTION_CLASSES: Final[tuple[int, ...]] = tuple(range(1, N_CLASSES))  # the 8 scored classes

# --- role slots ---
N_ROLES: Final = 13
N_TEAMS: Final = 2
N_SLOTS: Final = N_ROLES * N_TEAMS  # 26

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

# Full-HD -> 352x640 scaling, matching the reference dataset exactly.
ROI_SCALE_X: Final = 3.0
ROI_SCALE_Y: Final = 3.068181


def slot_index(left_to_right: int, role_id: int) -> int:
    """Pack (attacking side, tactical role) into a 0..25 slot. `role_id` is 1-based."""
    if left_to_right not in (0, 1):
        raise ValueError(f"left_to_right must be 0 or 1, got {left_to_right}")
    if not 1 <= role_id <= N_ROLES:
        raise ValueError(f"role_id must be 1..{N_ROLES}, got {role_id}")
    return left_to_right * N_ROLES + (role_id - 1)


def slot_to_role(slot: int) -> tuple[int, int]:
    """Inverse of `slot_index`: slot -> (left_to_right, role_id)."""
    if not 0 <= slot < N_SLOTS:
        raise ValueError(f"slot must be 0..{N_SLOTS - 1}, got {slot}")
    return slot // N_ROLES, (slot % N_ROLES) + 1


def slot_name(slot: int) -> str:
    """Human-readable slot label, e.g. `L-LB` or `R-CF`."""
    ltr, role_id = slot_to_role(slot)
    return f"{'R' if ltr else 'L'}-{ROLE_NAMES[role_id]}"
