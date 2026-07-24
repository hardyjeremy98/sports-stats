"""Frozen tracklet-replay tracker (SPO-59): injects a source run's
tracklets.json as this run's TRACK output, carrying its frame_features.npz
along, so every associate-layer benchmark arm (do-no-harm comparators, the
anchor-economics sweep) scores byte-identical tracklet substrate without
re-running the GPU tracker. Same protocol role as the SPO-30 frozen
det-replay detector, one stage later in the pipeline.

Source resolution: an explicit `run_dir`, or `runs_dir` resolved per sequence
— `<runs_dir>/<video-stem>` first, else a unique `*-<video-stem>` match (the
benchmark experiment names its run dirs `<candidate>-<sequence>`). Ambiguity
or a missing tracklets.json refuses loudly; replay never guesses.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import BaseModel

from matchlab_core.interfaces import StageContext, Tracker
from matchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from matchlab_core.registry import register
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.run import ArtifactName, StageKind


class Params(BaseModel):
    # Exactly one source: an explicit run dir, or a parent dir resolved
    # per-sequence against the video stem.
    run_dir: str | None = None
    runs_dir: str | None = None

    def model_post_init(self, __context) -> None:
        if self.run_dir is None and self.runs_dir is None:
            raise ValueError(
                "Frozen-tracklets tracker: set either params.run_dir (one source "
                "run) or params.runs_dir (per-sequence <dir>/<video-stem> or "
                "unique <dir>/*-<video-stem>)."
            )


@register(StageKind.TRACK, "frozen-tracklets")
class FrozenTrackletsTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)
        self._resolved: Path | None = None

    def _resolve(self, ctx: StageContext) -> Path:
        if self.params.run_dir is not None:
            return Path(self.params.run_dir)
        runs_dir = Path(self.params.runs_dir)
        stem = Path(ctx.video.path).stem
        direct = runs_dir / stem
        if direct.is_dir():
            return direct
        matches = sorted(d for d in runs_dir.glob(f"*-{stem}") if d.is_dir())
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RuntimeError(
                f"Frozen-tracklets tracker: no source run for video stem {stem!r} "
                f"under {runs_dir} (tried '{stem}' and '*-{stem}')."
            )
        raise RuntimeError(
            f"Frozen-tracklets tracker: ambiguous source runs for video stem "
            f"{stem!r} under {runs_dir}: {[d.name for d in matches]} — use "
            "params.run_dir to disambiguate."
        )

    def prepare(self, ctx: StageContext) -> None:
        source = self._resolve(ctx)
        if not (source / "tracklets.json").exists():
            raise RuntimeError(
                f"Frozen-tracklets tracker: no tracklets.json in source run {source}."
            )
        self._resolved = source

    def provenance(self) -> list[ModelProvenance]:
        source = self._resolved
        if source is None and self.params.run_dir is not None:
            source = Path(self.params.run_dir)
        tracklets = source / "tracklets.json" if source is not None else None
        has_file = tracklets is not None and tracklets.exists()
        return [
            ModelProvenance(
                architecture="frozen-tracklets",
                revision="frozen-tracklets/v1",
                weights_path=str(tracklets)
                if tracklets is not None
                else f"{self.params.runs_dir}/<video-stem>/tracklets.json",
                weights_sha256=sha256_file(str(tracklets)) if has_file else None,
                lineage=f"replayed source run: {source if source is not None else 'unresolved'}",
                license=LicenseAxes(
                    code="n/a (file replay)",
                    weights="inherits source tracker run",
                    training_data="inherits source tracker run",
                ),
            )
        ]

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        if self._resolved is None:
            self.prepare(ctx)
        source = self._resolved
        tracklets = [
            Tracklet.model_validate(t)
            for t in json.loads((source / "tracklets.json").read_text())
        ]
        features = source / "frame_features.npz"
        if features.exists():
            shutil.copyfile(features, ctx.store.path(ArtifactName.FRAME_FEATURES))
        return tracklets
