"""30-second windows for the DST sequence stage.

Each sample pairs a `framespan`-frame slice of the stage-1 `(9,26,T)` logits and
per-slot kinematics (the encoder source) with the ordered list of actions that
actually happened in that window (the decoder target).

Pitch-symmetry augmentation is the interesting part and the easy thing to get wrong.
Mirroring the pitch is only label-preserving if the ROLE SLOTS are remapped too: a
left-back reflected across the X axis is a right-back on the other side, so a flip
that moves the coordinates but not the slots teaches the model that left-backs stand
on the right. Three symmetries are available, each with its own remap:

* **X** -- attacking direction flips, so both the side bit AND left/right roles swap.
* **Y** -- side is unchanged, only left/right roles swap.
* **XY** -- side flips, roles are unchanged (two mirrors cancel).

This is the same ADR 008 fact from the other direction: slots are tactical, so any
geometric transform of the pitch is also a permutation of the slot index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matchlab_core.pcbas.denoiser import (
    ABSENT_FILL,
    ACTION_VOCAB,
    ENCODER_FEATURE_DIM,
    EOS_ACTION,
    FEATURES_PER_SLOT,
    FRAMESPAN,
    KINEMATIC_FEATURES,
    NEUTRAL_ROLE,
    ROLE_VOCAB,
    SOS_ACTION,
)
from matchlab_core.pcbas.logits import load_logits, logits_filename
from matchlab_core.pcbas.schema import (
    BACKGROUND,
    CLS,
    FRAME,
    LEFT_TO_RIGHT,
    N_CLASSES,
    N_ROLES,
    N_SLOTS,
    ROI_X,
    ROLE_ID,
    X_POS,
    X_SPEED,
    Y_POS,
    Y_SPEED,
    slot_index,
    slot_to_role,
)

from matchlab_train.datasets.footpass_pcbas import list_halves, load_half

# Left/right mirrored role pairs, from the reference's ROLE_NAMES:
# LB<->RB, LCB<->RCB, LM<->RM, LW<->RW. GK, MCB, DM, AM, CF are self-mirrored.
_MIRRORED_ROLES: dict[int, int] = {2: 13, 13: 2, 3: 5, 5: 3, 6: 7, 7: 6, 10: 11, 11: 10}


def _mirror_role(role_id: int) -> int:
    return _MIRRORED_ROLES.get(role_id, role_id)


def slot_permutation(axis: str) -> np.ndarray:
    """`perm[old_slot] = new_slot` for a pitch symmetry.

    A flip that moves coordinates but not slots teaches the model that left-backs
    stand on the right, so this must accompany every geometric augmentation.
    """
    if axis not in ("x", "y", "xy"):
        raise ValueError(f"axis must be x, y or xy, got {axis!r}")
    perm = np.zeros(N_SLOTS, dtype=np.int64)
    for slot in range(N_SLOTS):
        side, role = slot_to_role(slot)
        if axis == "x":
            side, role = 1 - side, _mirror_role(role)
        elif axis == "y":
            role = _mirror_role(role)
        else:  # xy
            side = 1 - side
        perm[slot] = slot_index(side, role)
    return perm


@dataclass(frozen=True)
class WindowSample:
    """One encoder source and decoder target, before tensor conversion."""

    key: str
    start_frame: int
    features: np.ndarray  # (T, 364)
    # (T,) WINDOW-LOCAL 1-based positions for the positional encoding -- NOT absolute
    # video frames. The reference calls this `Encoder_abs_frame_nb` and comments it
    # "Absolute frame number", but line 237 of its DST_Dataset subtracts
    # `min_Frame - 1`, making it 1..framespan. Feeding true absolute frames (up to
    # ~157,000) aliases the sinusoidal encoding AND puts the encoder in a different
    # coordinate system from the decoder, whose timestamps are window-local -- so the
    # model cannot align a decoder query with an encoder position at all.
    frames: np.ndarray
    action_tokens: np.ndarray  # (N+2, 10) with SOS and EOS
    role_tokens: np.ndarray  # (N+2, 27)
    event_frames: np.ndarray  # (N+2,) window-local, 0 for SOS


def build_kinematics(rows: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """`(5, 26, T)` of x, y, vx, vy, observable, filled with ABSENT_FILL.

    Position is present in the tactical data even when the player is off-camera; the
    fifth channel is the OBSERVABILITY FLAG, not a coordinate, and it is the only
    thing telling the model whether the visual logits for that slot mean anything.
    """
    t = len(frames)
    out = np.full((KINEMATIC_FEATURES, N_SLOTS, t), ABSENT_FILL, dtype=np.float32)
    if not len(rows):
        return out
    position = {int(f): i for i, f in enumerate(frames)}
    for row in rows:
        pos = position.get(int(row[FRAME]))
        if pos is None:
            continue
        ltr, role = row[LEFT_TO_RIGHT], row[ROLE_ID]
        if np.isnan(ltr) or np.isnan(role) or not 1 <= int(role) <= N_ROLES:
            continue
        slot = slot_index(int(ltr), int(role))
        out[0, slot, pos] = row[X_POS]
        out[1, slot, pos] = row[Y_POS]
        out[2, slot, pos] = row[X_SPEED]
        out[3, slot, pos] = row[Y_SPEED]
        out[4, slot, pos] = 0.0 if np.isnan(row[ROI_X]) else 1.0
    return out


def encoder_features(kinematics: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """`(5,26,T)` + `(9,26,T)` -> `(T, 364)`, slot-major within each frame.

    Layout matters: `SlotAttentionEncoderEmbedding` reshapes this back to
    `(T, 26, 14)`, so all 14 of a slot's features must be contiguous.
    """
    if kinematics.shape[1:] != logits.shape[1:]:
        raise ValueError(
            f"kinematics {kinematics.shape} and logits {logits.shape} disagree on "
            f"(slots, frames)"
        )
    stacked = np.concatenate([kinematics, logits.astype(np.float32)], axis=0)  # (14,26,T)
    return stacked.transpose(2, 1, 0).reshape(len(stacked[0][0]), ENCODER_FEATURE_DIM)


def apply_symmetry(
    kinematics: np.ndarray, logits: np.ndarray, events: np.ndarray, axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the pitch, remapping slots so the labels stay true.

    `events` is `(N, 3)` of `(window_frame, slot, class_id)`.
    """
    perm = slot_permutation(axis)
    kin, log = kinematics.copy(), logits.copy()
    kin[:, perm, :] = kinematics
    log[:, perm, :] = logits

    observed = kin[4] != ABSENT_FILL
    if axis in ("x", "xy"):
        np.copyto(kin[0], 1.0 - kin[0], where=observed)
        np.copyto(kin[2], -kin[2], where=observed)
    if axis in ("y", "xy"):
        np.copyto(kin[1], 1.0 - kin[1], where=observed)
        np.copyto(kin[3], -kin[3], where=observed)

    moved = events.copy()
    if len(moved):
        moved[:, 1] = perm[events[:, 1]]
    return kin, log, moved


def build_tokens(
    events: np.ndarray, window_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(N,3)` events -> SOS-wrapped `(N+2, 10)`, `(N+2, 27)`, `(N+2,)` streams.

    Events are emitted in time order. `class_id - 1` maps the 8 action classes onto
    action indices 0-7, leaving 8 for SOS and 9 for EOS -- the action stream has no
    background slot at all, because background is expressed by emitting nothing.
    """
    ordered = events[np.argsort(events[:, 0], kind="stable")] if len(events) else events
    n = len(ordered)
    actions = np.zeros((n + 2, ACTION_VOCAB), dtype=np.float32)
    roles = np.zeros((n + 2, ROLE_VOCAB), dtype=np.float32)
    frames = np.zeros(n + 2, dtype=np.int64)

    actions[0, SOS_ACTION] = 1.0
    roles[0, NEUTRAL_ROLE] = 1.0
    actions[-1, EOS_ACTION] = 1.0
    roles[-1, NEUTRAL_ROLE] = 1.0

    for i, (frame, slot, class_id) in enumerate(ordered, start=1):
        if not 1 <= class_id < N_CLASSES:
            continue
        actions[i, int(class_id) - 1] = 1.0
        roles[i, int(slot)] = 1.0
        frames[i] = int(frame) + 1  # 0 is reserved for SOS
    frames[-1] = (int(ordered[-1][0]) + 2) if n else min(window_length + 1, window_length + 1)
    return actions, roles, frames


def window_events(rows: np.ndarray, start: int, length: int) -> np.ndarray:
    """`(N, 3)` of `(window_frame, slot, class_id)` for the actions in the window."""
    labelled = rows[~np.isnan(rows[:, CLS]) & (rows[:, CLS] != BACKGROUND)]
    out: list[tuple[int, int, int]] = []
    for row in labelled:
        frame = int(row[FRAME])
        if not start <= frame < start + length:
            continue
        ltr, role = row[LEFT_TO_RIGHT], row[ROLE_ID]
        if np.isnan(ltr) or np.isnan(role) or not 1 <= int(role) <= N_ROLES:
            continue
        out.append((frame - start, slot_index(int(ltr), int(role)), int(row[CLS])))
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 3)


class FootpassWindowDataset:
    """`torch.utils.data.Dataset` of DST training windows.

    Requires stage-1 logits on disk: the sequence stage is trained on what the visual
    stage ACTUALLY predicts, warts and all, not on ground truth. Training it on clean
    labels would produce a model that has never seen the noise it exists to remove.
    """

    def __init__(
        self,
        h5_path: str | Path,
        logits_dir: str | Path,
        *,
        framespan: int = FRAMESPAN,
        stride: int | None = None,
        train: bool = True,
        seed: int = 345,
        halves: list[str] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.logits_dir = Path(logits_dir)
        self.framespan = framespan
        self.stride = stride if stride is not None else framespan // 2
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.keys = [
            k
            for k in (halves or list_halves(h5_path))
            if (self.logits_dir / logits_filename(k)).is_file()
        ]
        if not self.keys:
            raise RuntimeError(f"no stage-1 logits found under {self.logits_dir}")
        self._rows: dict[str, np.ndarray] = {}
        self._logits: dict[str, np.ndarray] = {}
        self.index = self._build_index()

    def _build_index(self) -> list[tuple[str, int]]:
        index: list[tuple[str, int]] = []
        for key in self.keys:
            rows = self._half_rows(key)
            frames = rows[:, FRAME]
            first, last = int(np.nanmin(frames)), int(np.nanmax(frames))
            last = min(last, self._half_logits(key).shape[2] - 1)
            for start in range(first, max(first + 1, last - self.framespan + 2), self.stride):
                index.append((key, start))
        return index

    def _half_rows(self, key: str) -> np.ndarray:
        if key not in self._rows:
            if len(self._rows) > 2:
                self._rows.clear()
            self._rows[key] = load_half(self.h5_path, key)
        return self._rows[key]

    def _half_logits(self, key: str) -> np.ndarray:
        if key not in self._logits:
            if len(self._logits) > 2:
                self._logits.clear()
            self._logits[key] = load_logits(self.logits_dir / logits_filename(key))
        return self._logits[key]

    def __len__(self) -> int:
        return len(self.index)

    def sample(self, i: int) -> WindowSample:
        key, start = self.index[i]
        rows = self._half_rows(key)
        logits = self._half_logits(key)
        # Absolute frames index the tactical rows; window-local 1-based positions go
        # to the model. Conflating the two is what made DST's first end-to-end run
        # score 0.035 -- see `WindowSample.frames`.
        absolute = np.arange(start, start + self.framespan)
        positions = np.arange(1, self.framespan + 1)

        window_rows = rows[
            (rows[:, FRAME] >= start) & (rows[:, FRAME] < start + self.framespan)
        ]
        kinematics = build_kinematics(window_rows, absolute)
        window_logits = np.zeros((N_CLASSES, N_SLOTS, self.framespan), dtype=np.float32)
        available = min(self.framespan, max(0, logits.shape[2] - start))
        if available > 0:
            window_logits[:, :, :available] = logits[:, :, start : start + available]
        events = window_events(rows, start, self.framespan)

        if self.train:
            axis = self.rng.choice(["none", "x", "y", "xy"])
            if axis != "none":
                kinematics, window_logits, events = apply_symmetry(
                    kinematics, window_logits, events, axis
                )

        actions, roles, event_frames = build_tokens(events, self.framespan)
        return WindowSample(
            key=key,
            start_frame=start,
            features=encoder_features(kinematics, window_logits),
            frames=positions,
            action_tokens=actions,
            role_tokens=roles,
            event_frames=event_frames,
        )

    def __getitem__(self, i: int):
        import torch

        s = self.sample(i)
        return (
            torch.from_numpy(s.features),
            torch.from_numpy(s.frames),
            torch.from_numpy(s.action_tokens),
            torch.from_numpy(s.role_tokens),
            torch.from_numpy(s.event_frames),
        )


def one_hot_frames(frames: np.ndarray, framespan: int) -> np.ndarray:
    """`(T,)` window-local frame indices -> `(T, framespan+2)` one-hot.

    Concatenated onto the features to make the encoder embedding 1116 wide.
    """
    out = np.zeros((len(frames), framespan + 2), dtype=np.float32)
    clipped = np.clip(frames, 0, framespan + 1)
    out[np.arange(len(frames)), clipped] = 1.0
    return out


assert FEATURES_PER_SLOT == KINEMATIC_FEATURES + N_CLASSES
