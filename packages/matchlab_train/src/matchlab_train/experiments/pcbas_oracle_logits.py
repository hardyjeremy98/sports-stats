"""Synthesise a PERFECT stage 1, to attribute DST's failure (SPO-96).

Measured on FOOTPASS VAL: stage 1 alone scores micro-F1 0.3274, and TAAD→DST scores
**0.1188** — the sequence stage makes the pipeline nearly 3x worse. Two explanations
survive, and they imply opposite next actions:

* **DST cannot learn the task here** (model, objective or training budget). Then more
  stage-1 work is wasted.
* **DST is starved by weak input.** Our stage 1 finds only 30% of events, so 70% of
  what DST must emit has no visual evidence at all. Then stage-1 quality is the
  binding constraint and the programme should move there.

Three tuning runs would not separate those. Replacing stage 1 with an oracle does: if
DST still fails on perfect input, the fault is DST's; if it succeeds, the fault is the
input's. This is the same oracle-stage pattern the repo already uses for tracking.

The oracle is deliberately PURE — it encodes every ground-truth event, including the
17.5% whose player has no bounding box and which no visual model can reach. So it is
an upper bound on what any stage 1 could hand the sequence stage, not a realistic one.

Logit magnitudes mimic real stage-1 output (measured: background mean +3.9 std 0.86,
action mean −3.6 std 2.05) so the input distribution DST sees is comparable and the
comparison is not confounded by scale.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from matchlab_core.pcbas.logits import LOGITS_DTYPE, logits_filename, save_logits
from matchlab_core.pcbas.schema import (
    BACKGROUND,
    CLS,
    FRAME,
    LEFT_TO_RIGHT,
    N_CLASSES,
    N_ROLES,
    N_SLOTS,
    ROLE_ID,
    slot_index,
)
from pydantic import BaseModel

from matchlab_train.datasets.footpass_pcbas import list_halves, load_half
from matchlab_train.experiments.base import Experiment
from matchlab_train.registry import register

# Chosen to sit in the same range as real stage-1 output, so DST sees a comparable
# input distribution rather than an out-of-scale one.
BACKGROUND_LOGIT = 4.0
ACTION_LOGIT = 6.0
SUPPRESSED_LOGIT = -3.0

# Matches the label dilation stage 1 trains with, so an oracle peak has the same
# temporal width a well-trained action head would produce.
DEFAULT_DILATION = 1
# Dilated neighbours sit slightly BELOW the centre frame. A flat plateau would leave
# the decoder's NMS tie-break to pick the earliest frame of the plateau, putting every
# oracle event `dilation` frames early -- harmless at delta=12 but wrong, and a real
# trained action head produces a peak rather than a plateau anyway.
NEIGHBOUR_PENALTY = 0.5


class Params(BaseModel):
    h5_path: str = "data/footpass/tactical/train_tactical_data.h5"
    out_dir: str = "data/footpass/logits/oracle_train"
    dilation: int = DEFAULT_DILATION
    halves: list[str] | None = None


def oracle_logits(
    rows: np.ndarray, n_frames: int, dilation: int = DEFAULT_DILATION
) -> np.ndarray:
    """`(9, 26, n_frames)` fp16 encoding every ground-truth event perfectly.

    Background wins everywhere except within `dilation` frames of an event, where that
    event's class wins for the ACTING SLOT ONLY, peaking on the exact event frame — the identity problem is preserved,
    because the point is to test whether DST can read a correct signal, not to delete
    the part of the task DST exists to do.
    """
    out = np.full((N_CLASSES, N_SLOTS, n_frames), SUPPRESSED_LOGIT, dtype=np.float32)
    out[BACKGROUND] = BACKGROUND_LOGIT

    labelled = rows[~np.isnan(rows[:, CLS]) & (rows[:, CLS] != BACKGROUND)]
    for row in labelled:
        ltr, role, frame = row[LEFT_TO_RIGHT], row[ROLE_ID], row[FRAME]
        if np.isnan(ltr) or np.isnan(role) or np.isnan(frame):
            continue
        if not 1 <= int(role) <= N_ROLES:
            continue
        class_id = int(row[CLS])
        if not 1 <= class_id < N_CLASSES:
            continue
        slot = slot_index(int(ltr), int(role))
        lo = max(0, int(frame) - dilation)
        hi = min(n_frames, int(frame) + dilation + 1)
        if lo >= hi:
            continue
        out[class_id, slot, lo:hi] = ACTION_LOGIT - NEIGHBOUR_PENALTY
        out[class_id, slot, int(frame)] = ACTION_LOGIT
        out[BACKGROUND, slot, lo:hi] = SUPPRESSED_LOGIT
    return out.astype(LOGITS_DTYPE)


@register("pcbas-oracle-logits")
class PCBASOracleLogitsExperiment(Experiment):
    """Write oracle `(9,26,T)` logits for every half of a split. CPU only."""

    def run(self) -> dict:
        p = Params(**self.config.params)
        workdir = self.workdir()
        out_dir = Path(p.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = []
        for key in p.halves or list_halves(p.h5_path):
            t0 = time.time()
            rows = load_half(p.h5_path, key)
            n_frames = int(np.nanmax(rows[:, FRAME])) + 1
            logits = oracle_logits(rows, n_frames, p.dilation)
            save_logits(logits, out_dir / logits_filename(key))
            n_events = int(
                (~np.isnan(rows[:, CLS]) & (rows[:, CLS] != BACKGROUND)).sum()
            )
            summary.append(
                {
                    "half": key,
                    "frames": n_frames,
                    "events": n_events,
                    "seconds": round(time.time() - t0, 1),
                }
            )
            print(f"  {key}: {n_events} events over {n_frames} frames", flush=True)

        result = {
            "halves": summary,
            "out_dir": str(out_dir),
            "total_events": sum(s["events"] for s in summary),
            "dilation": p.dilation,
        }
        self.write_result(workdir, result)
        print(f"wrote {len(summary)} halves, {result['total_events']:,} events")
        return result
