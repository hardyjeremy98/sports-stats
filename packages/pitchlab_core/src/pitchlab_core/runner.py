from __future__ import annotations

import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from pitchlab_core import registry
from pitchlab_core.artifacts import ArtifactStore
from pitchlab_core.config import PipelineConfig
from pitchlab_core.interfaces import ProgressFn, Stage, StageContext
from pitchlab_core.provenance import (
    DEFAULT_PACKAGE_NAMES,
    RunProvenance,
    StageProvenance,
    collect_package_versions,
    git_revision,
)
from pitchlab_core.schemas import ArtifactName
from pitchlab_core.schemas.run import (
    RunManifest,
    StageKind,
    StageResult,
    StageStatus,
)
from pitchlab_core.timeline import compute_timeline
from pitchlab_core.video import probe

# Execution order. SPOTTING sits between EVENTS and ANNOTATE so its (future)
# events can be rendered; it is stub/disabled in every v1 config.
STAGE_ORDER: list[StageKind] = [
    StageKind.DETECT,
    StageKind.TRACK,
    StageKind.TEAM,
    StageKind.CALIBRATE,
    StageKind.ASSOCIATE,
    StageKind.IDENTITY,
    StageKind.FUSE,
    StageKind.EVENTS,
    StageKind.SPOTTING,
    StageKind.ANNOTATE,
]


class PipelineRunner:
    """Executes a PipelineConfig against one video, writing every intermediate
    to the run directory. The manifest is rewritten after each stage so the Lab
    UI can poll live progress."""

    def __init__(
        self,
        run_id: str,
        video_path: str | Path,
        config: PipelineConfig,
        run_dir: str | Path,
        device: str = "cpu",
        progress: ProgressFn | None = None,
    ):
        self.run_id = run_id
        self.config = config
        self.store = ArtifactStore(run_dir)
        self.device = device
        self._progress = progress or (lambda kind, frac, msg: None)
        meta = probe(video_path, sample_stride=config.video.sample_stride)
        self.ctx = StageContext(
            video=meta,
            config=config,
            store=self.store,
            device=device,
            progress=self._progress,
        )
        self.manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now(UTC),
            video=meta,
            config=config.model_dump(mode="json"),
            config_name=config.name,
            status=StageStatus.RUNNING,
            provenance=RunProvenance(
                git_revision=git_revision(Path(__file__).resolve().parent),
                package_versions=collect_package_versions(DEFAULT_PACKAGE_NAMES),
            ),
        )

    def run(self) -> RunManifest:
        try:
            self._run_stages()
            self.manifest.status = StageStatus.COMPLETED
        except Exception as exc:  # noqa: BLE001 — the manifest is the error report
            self.manifest.status = StageStatus.FAILED
            self.manifest.error = f"{exc}\n{traceback.format_exc()}"
        self._save_manifest()
        return self.manifest

    def _build_stage(self, kind: StageKind) -> Stage | None:
        cfg = self.config.stages.get(kind)
        if cfg is None or not cfg.enabled:
            return None
        return registry.build(kind, cfg.impl, cfg.params)

    def _run_stages(self) -> None:
        store = self.store
        ctx = self.ctx

        detect_out = self._exec(StageKind.DETECT, lambda s: s.detect(ctx), required=True)
        store.write_jsonl(ArtifactName.DETECTIONS, detect_out.frames)
        store.write_jsonl(ArtifactName.BALL, detect_out.ball)
        self._index(ArtifactName.DETECTIONS, ArtifactName.BALL)

        tracklets = self._exec(
            StageKind.TRACK, lambda s: s.track(ctx, detect_out.frames), required=True
        )
        store.write_json(ArtifactName.TRACKLETS, tracklets)
        self._index(ArtifactName.TRACKLETS)
        self._metric("n_tracklets", len(tracklets))

        teams = self._exec(StageKind.TEAM, lambda s: s.classify(ctx, tracklets)) or []
        store.write_json(ArtifactName.TEAMS, teams)
        self._index(ArtifactName.TEAMS)

        calibration = self._exec(StageKind.CALIBRATE, lambda s: s.calibrate(ctx)) or []
        store.write_jsonl(ArtifactName.CALIBRATION, calibration)
        self._index(ArtifactName.CALIBRATION)

        players = self._exec(
            StageKind.ASSOCIATE, lambda s: s.associate(ctx, tracklets, teams)
        )
        self._index(ArtifactName.ASSOCIATION, ArtifactName.REID_EMBEDDINGS)
        if players is None:
            # No associator configured: degrade to one entity per tracklet.
            from pitchlab_core.stages.associate.identity_fallback import one_entity_per_tracklet

            players = one_entity_per_tracklet(tracklets, teams)

        players = (
            self._exec(StageKind.IDENTITY, lambda s: s.resolve(ctx, players, tracklets))
            or players
        )
        store.write_json(ArtifactName.PLAYERS, players)
        self._index(ArtifactName.PLAYERS)
        self._metric("n_players", len(players))

        minimap = (
            self._exec(
                StageKind.FUSE,
                lambda s: s.fuse(ctx, players, tracklets, calibration, detect_out.ball),
            )
            or []
        )
        store.write_jsonl(ArtifactName.MINIMAP, minimap)
        self._index(ArtifactName.MINIMAP)

        events_out = self._exec(
            StageKind.EVENTS, lambda s: s.detect_events(ctx, minimap, players)
        )
        spotted = self._exec(StageKind.SPOTTING, lambda s: s.spot(ctx)) or []
        self._index(ArtifactName.SPOTTING)  # written by tdeed-style stages, guarded by exists()
        all_events = (events_out.events if events_out else []) + spotted
        if events_out:
            store.write_json(ArtifactName.POSSESSION, events_out.possession)
            store.write_json(ArtifactName.STATS, events_out.stats)
            store.write_json(ArtifactName.QA_ITEMS, events_out.qa_items)
            self._index(ArtifactName.POSSESSION, ArtifactName.STATS, ArtifactName.QA_ITEMS)
            self._metric("n_events", len(all_events))
            self._metric("n_qa_items", len(events_out.qa_items))
        store.write_json(ArtifactName.EVENTS, all_events)
        self._index(ArtifactName.EVENTS)

        timeline = compute_timeline(
            duration_s=ctx.video.duration_s,
            fps=ctx.video.fps,
            detections=detect_out.frames,
            tracklets=tracklets,
            calibration=calibration,
            minimap=minimap,
            players=players,
            events=all_events,
        )
        store.write_json(ArtifactName.TIMELINE, timeline)
        self._index(ArtifactName.TIMELINE)

        annotated = self._exec(
            StageKind.ANNOTATE,
            lambda s: s.render(ctx, detect_out, tracklets, players, minimap, all_events),
        )
        if annotated is not None:
            self._index(ArtifactName.ANNOTATED_VIDEO)

    def _exec(self, kind: StageKind, fn, required: bool = False):
        """Run one stage slot with timing/status bookkeeping. Optional stages
        that are absent return None; failures always fail the run (partial
        artifacts up to that stage remain readable in the Lab)."""
        stage = self._build_stage(kind)
        cfg = self.config.stages.get(kind)
        if stage is None:
            if required:
                raise RuntimeError(f"Pipeline config '{self.config.name}' must define stage {kind}")
            if cfg is not None:
                self.manifest.stages.append(
                    StageResult(kind=kind, impl=cfg.impl, status=StageStatus.SKIPPED)
                )
                self._save_manifest()
            return None

        result = StageResult(
            kind=kind,
            impl=stage.impl_name,
            status=StageStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        self.manifest.stages.append(result)
        self.manifest.provenance.stages[kind.value] = StageProvenance(
            impl=stage.impl_name,
            params=_resolved_params(stage),
            models=stage.provenance(),
        )
        self._save_manifest()
        self._progress(kind, 0.0, f"{kind}: starting ({stage.impl_name})")
        start = time.monotonic()
        try:
            stage.prepare(self.ctx)
            # Some stages only resolve model info (e.g. a downloaded weights
            # path) inside prepare(); refresh now that it has run. The
            # manifest gets rewritten below regardless, so this is free.
            self.manifest.provenance.stages[kind.value].models = stage.provenance()
            out = fn(stage)
            # Some stages only finish populating derived provenance (e.g. a
            # hosted-detection cache's content hash, written to as the stage
            # runs) once they've actually executed; refresh again now that
            # the stage has completed successfully. Mirrors the post-prepare
            # refresh above.
            self.manifest.provenance.stages[kind.value].models = stage.provenance()
        except Exception:
            result.status = StageStatus.FAILED
            result.finished_at = datetime.now(UTC)
            result.duration_s = round(time.monotonic() - start, 3)
            self._save_manifest()
            raise
        result.status = StageStatus.COMPLETED
        result.finished_at = datetime.now(UTC)
        result.duration_s = round(time.monotonic() - start, 3)
        self._save_manifest()
        self._progress(kind, 1.0, f"{kind}: done in {result.duration_s}s")
        return out

    def _index(self, *names: ArtifactName) -> None:
        for name in names:
            if self.store.exists(name):
                self.manifest.artifacts[name.value] = self.store.relpath(name)

    def _metric(self, key: str, value) -> None:
        self.manifest.metrics[key] = value

    def _save_manifest(self) -> None:
        (self.store.run_dir / "manifest.json").write_text(
            self.manifest.model_dump_json(indent=2)
        )


def _resolved_params(stage: Stage) -> dict:
    """The stage's fully-resolved params (defaults filled in), for the
    provenance snapshot. Every stage impl follows the `self.params = Params(
    **params)` pydantic convention; stages with no params attribute (or a
    non-pydantic one) simply show an empty dict rather than raising."""
    params = getattr(stage, "params", None)
    return params.model_dump(mode="json") if isinstance(params, BaseModel) else {}
