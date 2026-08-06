"""Turn a scored half into the Lab's `pcbas_events.json`.

Lives in core beside the metric it wraps: the whole point is that the per-event
rows and the headline report come from ONE matcher call. Anything that rebuilds
the pairing to draw it is a bug waiting to disagree with the number it sits under.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from matchlab_core.pcbas.eval import (
    DEFAULT_CONF_THRESH,
    DEFAULT_DELTA,
    Identity,
    Verdict,
    score_events_with_verdicts,
)
from matchlab_core.pcbas.events import PCBASEvent
from matchlab_core.pcbas.schema import CLASS_NAMES
from matchlab_core.schemas.pcbas_lab import PCBASLabEvent, PCBASLabEvents

# (frame_idx, left_to_right, shirt_number) -> box in 640x352 video pixels.
BoxIndex = Mapping[tuple[int, int, int], tuple[float, float, float, float]]


def _row(
    verdict: Verdict, fps: float, boxes: BoxIndex | None
) -> PCBASLabEvent:
    """One matcher decision as a Lab row.

    The anchor differs by kind: a tp/fp is anchored at the PREDICTION (that is where
    the system claims something happened, and where a reviewer should look to judge
    it), an fn at the ground truth (there is no prediction to anchor to).
    """
    anchor = verdict.pred if verdict.pred is not None else verdict.gt
    assert anchor is not None  # the matcher never emits a verdict with neither side
    truth = verdict.gt

    frame_error = None
    if verdict.pred is not None and truth is not None:
        frame_error = verdict.pred.frame_idx - truth.frame_idx

    # Role/slot come from ground truth where we have it: shirt-native predictions
    # (the reference's playbyplay exchange) carry no role at all.
    role_source = truth if truth is not None else anchor

    box = None
    if boxes is not None and anchor.shirt_number is not None:
        box = boxes.get(
            (anchor.frame_idx, anchor.left_to_right, anchor.shirt_number)
        )

    return PCBASLabEvent(
        verdict=verdict.kind,
        frame_idx=anchor.frame_idx,
        t=anchor.t if anchor.t is not None else anchor.frame_idx / fps,
        class_id=anchor.class_id,
        class_name=CLASS_NAMES[anchor.class_id],
        score=verdict.pred.score if verdict.pred is not None else None,
        gt_frame_idx=truth.frame_idx if truth is not None else None,
        frame_error=frame_error,
        shirt_number=anchor.shirt_number,
        left_to_right=anchor.left_to_right,
        role_id=role_source.role_id,
        slot=role_source.slot,
        has_bbox=truth.has_bbox if truth is not None else None,
        is_replay=truth.is_replay if truth is not None else None,
        box=box,
    )


def build_lab_events(
    *,
    key: str,
    game_id: str,
    half: int,
    fps: float,
    gt: Iterable[PCBASEvent],
    pred: Iterable[PCBASEvent],
    boxes: BoxIndex | None = None,
    identity: Identity = "shirt",
    delta: int = DEFAULT_DELTA,
    conf_thresh: float = DEFAULT_CONF_THRESH,
) -> PCBASLabEvents:
    """Score one half and lay its decisions out for the Lab, ordered by frame."""
    report, verdicts = score_events_with_verdicts(
        gt, pred, delta=delta, conf_thresh=conf_thresh, identity=identity
    )
    rows = [_row(v, fps, boxes) for v in verdicts]
    rows.sort(key=lambda r: (r.frame_idx, r.verdict))
    return PCBASLabEvents(
        key=key,
        game_id=game_id,
        half=half,
        fps=fps,
        identity=identity,
        delta=delta,
        conf_thresh=conf_thresh,
        report=report,
        events=rows,
    )
