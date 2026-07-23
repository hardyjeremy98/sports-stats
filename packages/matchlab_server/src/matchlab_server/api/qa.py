from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from matchlab_core.schemas.qa import LabeledExample as LabeledExampleSchema
from sqlalchemy import select
from sqlalchemy.orm import Session

from matchlab_server.api.schemas import QACorrect, QAOut
from matchlab_server.db import get_db
from matchlab_server.models import LabeledExample, QARecord, QAStatus

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.get("", response_model=list[QAOut])
def list_qa(
    run_id: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
):
    q = select(QARecord).order_by(QARecord.run_id, QARecord.qa_id)
    if run_id:
        q = q.where(QARecord.run_id == run_id)
    if status:
        q = q.where(QARecord.status == QAStatus(status))
    return db.scalars(q).all()


@router.post("/{record_id}/accept", response_model=QAOut)
def accept(record_id: int, db: Session = Depends(get_db)):
    """Human confirms the pipeline's call → positive label of the prediction."""
    rec = _pending(record_id, db)
    rec.status = QAStatus.ACCEPTED
    rec.decided_at = datetime.now(UTC)
    _write_label(rec, db, accepted=True, labeled_player_id=_predicted(rec))
    db.commit()
    return rec


@router.post("/{record_id}/correct", response_model=QAOut)
def correct(record_id: int, body: QACorrect, db: Session = Depends(get_db)):
    """Human fixes the attribution (and/or event type) → corrective label."""
    rec = _pending(record_id, db)
    rec.status = QAStatus.CORRECTED
    rec.corrected_player_id = body.player_id
    rec.corrected_event_type = body.event_type
    rec.note = body.note
    rec.decided_at = datetime.now(UTC)
    _write_label(rec, db, accepted=False, labeled_player_id=body.player_id)
    db.commit()
    return rec


@router.post("/{record_id}/reject", response_model=QAOut)
def reject(record_id: int, db: Session = Depends(get_db)):
    """The event didn't happen at all → negative label."""
    rec = _pending(record_id, db)
    rec.status = QAStatus.REJECTED
    rec.decided_at = datetime.now(UTC)
    _write_label(rec, db, accepted=False, labeled_player_id=None)
    db.commit()
    return rec


def _pending(record_id: int, db: Session) -> QARecord:
    rec = db.get(QARecord, record_id)
    if rec is None:
        raise HTTPException(404, "QA record not found")
    if rec.status != QAStatus.PENDING:
        raise HTTPException(409, f"QA record already {rec.status.value}")
    return rec


def _predicted(rec: QARecord) -> int | None:
    return rec.payload.get("event", {}).get("player_id")


def _write_label(rec: QARecord, db: Session, accepted: bool, labeled_player_id: int | None):
    event = rec.payload.get("event", {})
    label = LabeledExampleSchema(
        source_run_id=rec.run_id,
        qa_id=rec.qa_id,
        event_type=event.get("type", "unknown"),
        frame_idx=event.get("frame_idx", 0),
        t=event.get("t", 0.0),
        predicted_player_id=_predicted(rec),
        labeled_player_id=labeled_player_id,
        accepted=accepted,
    )
    db.add(
        LabeledExample(
            run_id=rec.run_id, qa_record_id=rec.id, payload=label.model_dump(mode="json")
        )
    )
