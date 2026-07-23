from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from matchlab_server.api.schemas import (
    IdentityLabelCreate,
    IdentityLabelOut,
    MergePayload,
    PairPayload,
    RosterPayload,
    SplitPayload,
)
from matchlab_server.db import get_db
from matchlab_server.models import IdentityLabel, IdentityLabelKind, Run

router = APIRouter(prefix="/api/identity_qa", tags=["identity_qa"])

# kind -> the pydantic model that validates its payload shape.
_PAYLOAD_MODELS: dict[IdentityLabelKind, type] = {
    IdentityLabelKind.PAIR: PairPayload,
    IdentityLabelKind.MERGE: MergePayload,
    IdentityLabelKind.SPLIT: SplitPayload,
    IdentityLabelKind.ROSTER: RosterPayload,
}


@router.get("", response_model=list[IdentityLabelOut])
def list_identity_labels(
    run_id: str | None = None,
    kind: str | None = None,
    db: Session = Depends(get_db),
):
    q = select(IdentityLabel).order_by(IdentityLabel.created_at.desc())
    if run_id:
        q = q.where(IdentityLabel.run_id == run_id)
    if kind:
        try:
            q = q.where(IdentityLabel.kind == IdentityLabelKind(kind))
        except ValueError as exc:
            raise HTTPException(422, f"Unknown identity label kind '{kind}'") from exc
    return db.scalars(q).all()


@router.post("", response_model=IdentityLabelOut)
def create_identity_label(body: IdentityLabelCreate, db: Session = Depends(get_db)):
    run = db.get(Run, body.run_id)
    if run is None:
        raise HTTPException(404, "Run not found")

    try:
        kind = IdentityLabelKind(body.kind)
    except ValueError as exc:
        raise HTTPException(422, f"Unknown identity label kind '{body.kind}'") from exc

    payload_model = _PAYLOAD_MODELS[kind]
    try:
        payload = payload_model.model_validate(body.payload)
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid payload for kind '{kind.value}': {exc}") from exc

    label = IdentityLabel(
        run_id=run.id,
        video_id=run.video_id,
        kind=kind,
        payload=payload.model_dump(),
        note=body.note,
    )
    db.add(label)
    db.commit()
    return label


@router.delete("/{label_id}")
def delete_identity_label(label_id: int, db: Session = Depends(get_db)):
    label = db.get(IdentityLabel, label_id)
    if label is None:
        raise HTTPException(404, "Identity label not found")
    db.delete(label)
    db.commit()
    return {"ok": True}
