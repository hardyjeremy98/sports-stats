from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from matchlab_server.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    # Tracking ground truth (matchlab_core.gt.GroundTruth JSON) for this clip,
    # when it comes from a labelled benchmark. Runs on this video get scored
    # against it automatically.
    gt_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.gt_path)


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(Base):
    """One pipeline execution. Heavy results live in the run directory on the
    shared data volume; this row is queue state + progress + a metrics summary."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # uuid hex
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    config_name: Mapped[str] = mapped_column(String(120))
    # Full YAML snapshot at creation — runs stay reproducible/diffable even if
    # the file in configs/ changes later. The Lab can also submit ad-hoc YAML.
    config_yaml: Mapped[str] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, values_callable=lambda e: [m.value for m in e]),
        default=RunStatus.QUEUED,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_dir: Mapped[str] = mapped_column(Text)
    progress_stage: Mapped[str | None] = mapped_column(String(40), nullable=True)
    progress_frac: Mapped[float] = mapped_column(Float, default=0.0)
    progress_msg: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(Base):
    """The work queue (locked decision #6: job table + async worker, no cloud
    vendor SDK). Workers poll and claim; the run row carries user-facing state."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, values_callable=lambda e: [m.value for m in e]),
        default=JobStatus.QUEUED,
    )
    claimed_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QAStatus(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class QARecord(Base):
    """QA queue row, imported from a run's qa_items.json when it completes.
    Human decisions here become LabeledExample rows — the v2 training set."""

    __tablename__ = "qa_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    qa_id: Mapped[int] = mapped_column(Integer)  # id within the run artifact
    payload: Mapped[dict] = mapped_column(JSON)  # the QAItem (event, reason, candidates)
    status: Mapped[QAStatus] = mapped_column(
        Enum(QAStatus, values_callable=lambda e: [m.value for m in e]),
        default=QAStatus.PENDING,
    )
    corrected_player_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corrected_event_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LabeledExample(Base):
    __tablename__ = "labeled_examples"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    qa_record_id: Mapped[int] = mapped_column(ForeignKey("qa_records.id"))
    payload: Mapped[dict] = mapped_column(JSON)  # schemas.qa.LabeledExample
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IdentityLabelKind(str, enum.Enum):
    PAIR = "pair"
    MERGE = "merge"
    SPLIT = "split"
    ROSTER = "roster"


class IdentityLabel(Base):
    """Human identity-QA decision: same/different tracklet pairs, entity merge/split
    flags, or roster labels. An annotation only — never mutates run artifacts or
    entities. A later task exports pair labels as re-ID training data."""

    __tablename__ = "identity_labels"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    # Denormalized from the run for cross-run/export queries without a join.
    video_id: Mapped[int] = mapped_column(Integer)
    kind: Mapped[IdentityLabelKind] = mapped_column(
        Enum(IdentityLabelKind, values_callable=lambda e: [m.value for m in e])
    )
    payload: Mapped[dict] = mapped_column(JSON)  # kind-specific, see api/schemas.py
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
