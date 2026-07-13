from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pitchlab_core.artifacts import ArtifactStore
from pitchlab_core.config import PipelineConfig
from pitchlab_core.pitch import SOCCER_PITCH, PitchSpec
from pitchlab_core.schemas import (
    BallObservation,
    Event,
    FrameCalibration,
    FrameDetections,
    MinimapFrame,
    PlayerEntity,
    PossessionSegment,
    QAItem,
    StatSheet,
    TeamAssignment,
    Tracklet,
)
from pitchlab_core.schemas.run import StageKind, VideoMeta
from pitchlab_core.video import Frame, iter_frames

ProgressFn = Callable[[StageKind, float, str], None]


@dataclass
class StageContext:
    """Everything a stage may need. Stages must not touch the filesystem outside
    `store` — that invariant is what keeps runs portable and diffable."""

    video: VideoMeta
    config: PipelineConfig
    store: ArtifactStore
    device: str = "cpu"
    pitch: PitchSpec = field(default_factory=lambda: SOCCER_PITCH)
    progress: ProgressFn = lambda kind, frac, msg: None

    def frames(self) -> Iterator[Frame]:
        """Decode the clip honoring the run's sampling config. Stages that need
        pixels (detect, team, calibrate, identity, annotate) each iterate this
        independently — offline product, sequential decode is cheap enough."""
        return iter_frames(
            self.video,
            stride=self.config.video.sample_stride,
            max_frames=self.config.video.max_frames,
            resize_width=self.config.video.resize_width,
        )


class Stage(ABC):
    """Base for all pipeline stages. Subclasses are constructed from
    StageConfig.params (validate with pydantic in __init__) and self-register
    via @register(kind, name)."""

    stage_kind: ClassVar[StageKind]
    impl_name: ClassVar[str]

    def prepare(self, ctx: StageContext) -> None:
        """Load models/weights. Called once before the stage runs; raising here
        fails the run early with a clear error (e.g. missing ROBOFLOW_API_KEY)."""


@dataclass
class DetectOutput:
    frames: list[FrameDetections]
    ball: list[BallObservation]


class Detector(Stage):
    @abstractmethod
    def detect(self, ctx: StageContext) -> DetectOutput: ...


class Tracker(Stage):
    """Short-term tracking. v1 default is BoT-SORT (heuristic). The second slot
    is reserved for a learned/query-propagation tracker (MOTRv2-style): such an
    implementation may ignore `detections` and consume ctx.frames() directly —
    the runner passes both, so swapping it in never touches calling code."""

    @abstractmethod
    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]: ...


class TeamClassifier(Stage):
    @abstractmethod
    def classify(
        self, ctx: StageContext, tracklets: list[Tracklet]
    ) -> list[TeamAssignment]: ...


class Calibrator(Stage):
    @abstractmethod
    def calibrate(self, ctx: StageContext) -> list[FrameCalibration]: ...


class Associator(Stage):
    """Offline global cross-tracklet association: groups tracklets that are the
    same physical player across the whole clip. Runs after team classification
    (team is a hard constraint) and before identity resolution."""

    @abstractmethod
    def associate(
        self,
        ctx: StageContext,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
    ) -> list[PlayerEntity]: ...


class IdentityResolver(Stage):
    """Assigns a human-meaningful identity to each PlayerEntity, from
    tracklet-level evidence (face crops at high-res candidate frames, jersey
    OCR, ...). Returns the entities with `identity` filled in."""

    @abstractmethod
    def resolve(
        self,
        ctx: StageContext,
        players: list[PlayerEntity],
        tracklets: list[Tracklet],
    ) -> list[PlayerEntity]: ...


class MinimapFuser(Stage):
    @abstractmethod
    def fuse(
        self,
        ctx: StageContext,
        players: list[PlayerEntity],
        tracklets: list[Tracklet],
        calibration: list[FrameCalibration],
        ball: list[BallObservation],
    ) -> list[MinimapFrame]: ...


@dataclass
class EventsOutput:
    possession: list[PossessionSegment]
    events: list[Event]
    stats: StatSheet
    qa_items: list[QAItem]


class EventEngine(Stage):
    """v1: heuristic possession attribution + pass state machine over the fused
    game state. Emits QA items for contested/low-confidence decisions."""

    @abstractmethod
    def detect_events(
        self,
        ctx: StageContext,
        minimap: list[MinimapFrame],
        players: list[PlayerEntity],
    ) -> EventsOutput: ...


class EventSpotter(Stage):
    """v2 seam: learned action spotting ("what happened, when") from raw frames.
    Output events are merged with the EventEngine's and attributed via the same
    QA path. v1 ships only a disabled stub implementation."""

    @abstractmethod
    def spot(self, ctx: StageContext) -> list[Event]: ...


class Annotator(Stage):
    @abstractmethod
    def render(
        self,
        ctx: StageContext,
        detections: DetectOutput,
        tracklets: list[Tracklet],
        players: list[PlayerEntity],
        minimap: list[MinimapFrame],
        events: list[Event],
    ) -> Path: ...
