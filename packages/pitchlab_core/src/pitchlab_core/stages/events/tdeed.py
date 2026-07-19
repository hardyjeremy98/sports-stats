"""`tdeed` EventSpotter stage (SPO-46): runs any contract-conforming external
action-spotting model — real GPL-3.0 T-DEED (its own sibling venv, see
docs/reference/external-spotters-setup.md), or the permissive in-repo
reference spotter — as a subprocess via `pitchlab_core.spotting.bridge`, and
writes its output straight through to the dedicated `spotting.json` artifact.

`spot()` always returns `[]`: a spotter's native taxonomy (e.g. T-DEED's
"PASS"/"DRIVE"/"SHOT" ball-action classes) is written verbatim to
spotting.json and never folded into events.json's EventType taxonomy here —
mapping one to the other, if and when consumed, is a documented table
applied downstream (see the exchange contract's "class" field note).

This module is the *pipeline* side of the GPL isolation boundary: it never
imports T-DEED or any non-commercial weights, only the bridge (which itself
knows nothing model-specific).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
from pydantic import BaseModel

from pitchlab_core.interfaces import EventSpotter, StageContext
from pitchlab_core.provenance import ModelProvenance
from pitchlab_core.registry import register
from pitchlab_core.schemas import ArtifactName, Event
from pitchlab_core.schemas.run import StageKind
from pitchlab_core.spotting.bridge import SpottingParams, run_spotter

# Runnable out of the box, with no real model or GPU: the permissive
# reference spotter CLI built in SPO-45. An operator points `command` at a
# real T-DEED entrypoint (its own venv) to swap in the genuine model.
_DEFAULT_COMMAND = [sys.executable, "-m", "pitchlab_core.spotting.reference_cli"]


class Params(BaseModel):
    command: list[str] = _DEFAULT_COMMAND
    weights: str = ""
    confidence: float = 0.3
    merge_window_s: float = 1.0
    device: str = "cpu"
    frames_ext: str = "jpg"


@register(StageKind.SPOTTING, "tdeed")
class TDeedSpotter(EventSpotter):
    def __init__(self, **params):
        self.params = Params(**params)

    def provenance(self) -> list[ModelProvenance]:
        return [
            ModelProvenance(
                architecture="external-spotter-cli",
                revision=" ".join(self.params.command),
                weights_path=self.params.weights or None,
            )
        ]

    def spot(self, ctx: StageContext) -> list[Event]:
        work_dir = Path(tempfile.mkdtemp(prefix="tdeed-bridge-"))
        frames_dir = work_dir / "frames"
        frames_dir.mkdir()
        try:
            for frame in ctx.frames():
                filename = f"{frame.frame_idx:08d}.{self.params.frames_ext}"
                cv2.imwrite(str(frames_dir / filename), frame.image)

            events = run_spotter(
                self.params.command,
                manifest_path=work_dir / "job.json",
                out_path=work_dir / "spotting_out.json",
                fps=ctx.video.fps,
                params=SpottingParams(
                    weights=self.params.weights,
                    confidence=self.params.confidence,
                    merge_window_s=self.params.merge_window_s,
                    device=self.params.device,
                ),
                frames_dir=frames_dir,
            )
            ctx.store.write_json(ArtifactName.SPOTTING, events)
        finally:
            # Frame dumps are large and purely an implementation detail of the
            # subprocess handoff — never a tracked run-dir artifact.
            shutil.rmtree(work_dir, ignore_errors=True)

        return []
