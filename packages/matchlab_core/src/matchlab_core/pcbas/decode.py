"""Turn a `(9, 26, T)` logit array into a list of slot-attributed events.

Softmax, argmax per (slot, frame), then per-(slot, class) non-maximum suppression
over a +/-`nms_window` frame neighbourhood, matching
`utils/metric_utils.py::post_processing_TAAD_preds_and_export`.

NMS is per SLOT and per CLASS, not global. A player can legitimately pass twice in a
second, and two different players certainly can, so suppressing across slots or
across classes would delete true events rather than duplicates.

The events this produces are **slot-native and carry no shirt number**. Attaching one
is a separate, export-time step (ADR 008) that needs the roster, which the model
never sees -- see `matchlab_train.datasets.footpass_pcbas.slot_shirt_table`.
"""

from __future__ import annotations

import numpy as np

from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.schema import BACKGROUND, N_CLASSES, slot_to_role

DEFAULT_NMS_WINDOW = 25


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable softmax. The reference's `np_softmax` omits the max
    subtraction, which overflows fp32 for logits above ~88; ours does not, and the
    two agree everywhere the reference does not overflow."""
    shifted = logits.astype(np.float64) - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def suppress(
    frames: np.ndarray, scores: np.ndarray, window: int
) -> list[int]:
    """Greedy NMS over 1-D frame positions. Returns the kept frames, ascending.

    Highest score first; each kept peak suppresses everything within +/-`window`.
    """
    order = np.argsort(-scores, kind="stable")
    ordered = frames[order]
    kept: list[int] = []
    suppressed = np.zeros(len(ordered), dtype=bool)
    for i, t0 in enumerate(ordered):
        if suppressed[i]:
            continue
        kept.append(int(t0))
        for j in range(i + 1, len(ordered)):
            if not suppressed[j] and abs(int(ordered[j]) - int(t0)) <= window:
                suppressed[j] = True
    return sorted(kept)


def decode_logits(
    logits: np.ndarray,
    *,
    nms_window: int = DEFAULT_NMS_WINDOW,
    min_score: float = 0.0,
) -> list[PCBASEvent]:
    """`(9, 26, T)` logits -> one `PCBASEvent` per surviving peak.

    `min_score` is applied AFTER softmax and before NMS. It defaults to 0 because the
    scoring metric applies its own `conf_thresh`; raising it here as well would make
    two thresholds interact and the reported operating point ambiguous.
    """
    if logits.ndim != 3 or logits.shape[0] != N_CLASSES:
        raise ValueError(f"expected (9, n_slots, T) logits, got {logits.shape}")

    probs = softmax(logits, axis=0)  # (9, M, T)
    predicted = probs.argmax(axis=0)  # (M, T)
    confidence = probs.max(axis=0)  # (M, T)

    events: list[PCBASEvent] = []
    n_slots = logits.shape[1]
    for slot in range(n_slots):
        left_to_right, role_id = slot_to_role(slot)
        for class_id in range(1, N_CLASSES):
            frames = np.flatnonzero(
                (predicted[slot] == class_id) & (confidence[slot] >= min_score)
            )
            if frames.size == 0:
                continue
            for frame in suppress(frames, confidence[slot, frames], nms_window):
                events.append(
                    PCBASEvent(
                        frame_idx=frame,
                        left_to_right=left_to_right,
                        role_id=role_id,
                        slot=slot,
                        class_id=class_id,
                        score=float(confidence[slot, frame]),
                    )
                )
    events.sort(key=lambda e: (e.frame_idx, e.slot))
    return events


def frames_with_no_prediction(logits: np.ndarray) -> int:
    """Frames where every slot is all-zero -- i.e. never covered by any window.

    Reported rather than silently decoded as background: an uncovered region and a
    confidently-empty region are different failures.
    """
    return int((np.abs(logits).sum(axis=(0, 1)) == 0).sum())


def argmax_is_background(logits: np.ndarray) -> np.ndarray:
    """(M, T) boolean -- where the model's top class is background."""
    return logits.argmax(axis=0) == BACKGROUND
