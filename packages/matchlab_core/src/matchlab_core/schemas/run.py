from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from matchlab_core.provenance import RunProvenance


class StageKind(StrEnum):
    """The fixed stage slots of the pipeline, in execution order.

    Slots are the swap points: a pipeline config picks one implementation per
    slot. SPOTTING and ATTRIBUTION are v2 seams — registered, stub-only today.
    """

    DETECT = "detect"
    TRACK = "track"
    TEAM = "team"
    CALIBRATE = "calibrate"
    ASSOCIATE = "associate"
    IDENTITY = "identity"
    FUSE = "fuse"
    POSSESSION = "possession"  # per-frame ball possessor (image-space, SPO-77)
    EVENTS = "events"
    SPOTTING = "spotting"  # v2: learned action spotting
    ANNOTATE = "annotate"  # renders the annotated output video


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageResult(BaseModel):
    kind: StageKind
    impl: str
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    error: str | None = None
    # Small stage-reported numbers for the Lab (e.g. n_tracklets, mean_conf).
    metrics: dict[str, float | int | str] = {}


class VideoMeta(BaseModel):
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    # Pipeline may subsample; artifacts index by *source* frame_idx throughout.
    sample_stride: int = 1


class ArtifactName(StrEnum):
    DETECTIONS = "detections"        # detections.jsonl — FrameDetections rows
    BALL = "ball"                    # ball.jsonl — BallObservation rows
    TRACKLETS = "tracklets"          # tracklets.json — list[Tracklet]
    TEAMS = "teams"                  # teams.json — list[TeamAssignment]
    CALIBRATION = "calibration"      # calibration.jsonl — FrameCalibration rows
    PLAYERS = "players"              # players.json — list[PlayerEntity]
    MINIMAP = "minimap"              # minimap.jsonl — MinimapFrame rows
    POSSESSION = "possession"        # possession.json — list[PossessionSegment]
    POSSESSION_TIMELINE = "possession_timeline"  # possession_timeline.json — list[PossessorFrame]
    EVENTS = "events"                # events.json — list[Event]
    SPOTTING = "spotting"            # spotting.json — list[SpottedEvent] (native taxonomy)
    STATS = "stats"                  # stats.json — StatSheet
    QA_ITEMS = "qa_items"            # qa_items.json — list[QAItem]
    TIMELINE = "timeline"            # timeline.json — list[TimelineBucket]
    ANNOTATED_VIDEO = "annotated_video"  # annotated.mp4
    CROPS = "crops"                  # crops/ dir — identity evidence images
    EVAL = "eval"                    # eval.json — MOT metrics vs ground truth (post-hoc)
    ASSOCIATION = "association"      # association.json — AssociationReport (pair decisions)
    REID_EMBEDDINGS = "reid_embeddings"  # reid_embeddings.npz — per-tracklet re-ID features
    FRAME_FEATURES = "frame_features"    # frame_features.npz — per-(tracklet, frame) features
    NAMING = "naming"                    # naming.json — NamingReport (roster naming decisions)
    REID_DETAIL = "reid_detail"          # reid_detail.json — ReidDetailReport (merge inspector)


class TimelineBucket(BaseModel):
    """Per-second confidence aggregates powering the Lab timeline heatmap.
    Low values are exactly where an ML engineer should scrub to."""

    t: float  # bucket start, seconds
    detection_confidence: float = 0.0
    tracking_stability: float = 0.0   # 1 - (tracklet births+deaths / active tracks)
    calibration_confidence: float = 0.0
    identity_coverage: float = 0.0    # fraction of visible players with identity
    event_count: int = 0
    contested_event_count: int = 0
    flags: list[str] = []


class RunManifest(BaseModel):
    """manifest.json at the root of every run directory: the single file the
    Lab UI reads to know what a run is, how it went, and what artifacts exist."""

    run_id: str
    created_at: datetime
    video: VideoMeta
    # Full snapshot of the resolved pipeline config (reproducibility + diffing).
    config: dict
    config_name: str
    stages: list[StageResult] = []
    artifacts: dict[str, str] = {}  # ArtifactName -> path relative to run dir
    metrics: dict[str, float | int | str] = {}
    status: StageStatus = StageStatus.PENDING
    error: str | None = None
    # Append-only-after-run-completion provenance block (SPO-10): weights hashes, package
    # versions, license/commercial-use status, git revision, eval-set identity — what a
    # benchmark runner needs months later to refuse comparing mismatched runs. Only
    # evaluation-set identity (evaluation_set_hash/evaluation_set_source) is filled in later,
    # in place, by the server's GT auto-scoring hook -- everything else is fixed at write time.
    # Required-with-a-default so manifests written before this field existed
    # still parse (they simply get the all-"unknown" default).
    provenance: RunProvenance = Field(default_factory=RunProvenance)
