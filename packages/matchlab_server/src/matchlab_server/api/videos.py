from __future__ import annotations

import re
import shutil

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from matchlab_core.video import probe
from sqlalchemy import select
from sqlalchemy.orm import Session

from matchlab_server.api.schemas import VideoOut
from matchlab_server.db import get_db
from matchlab_server.models import Video
from matchlab_server.settings import get_settings

router = APIRouter(prefix="/api/videos", tags=["videos"])


def register_video_file(db: Session, filename: str, path, meta) -> Video:
    """Insert-or-update a Video row for a file already on the videos volume.
    Used by upload and by dataset materialization; keyed by filename so
    re-materializing a split refreshes rather than duplicates."""
    video = db.query(Video).filter(Video.filename == filename).first()
    if video is None:
        video = Video(filename=filename, path=str(path))
        db.add(video)
    video.path = str(path)
    video.size_bytes = path.stat().st_size
    video.duration_s = meta.duration_s
    video.fps = meta.fps
    video.width = meta.width
    video.height = meta.height
    db.commit()
    return video


@router.post("", response_model=VideoOut)
def upload_video(file: UploadFile, db: Session = Depends(get_db)) -> Video:
    settings = get_settings()
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "upload.mp4")
    video = Video(filename=safe_name, path="")
    db.add(video)
    db.commit()

    dest = settings.videos_dir / f"{video.id}_{safe_name}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        meta = probe(dest)
    except FileNotFoundError as exc:
        db.delete(video)
        db.commit()
        raise HTTPException(400, "Uploaded file is not a readable video") from exc

    video.path = str(dest)
    video.size_bytes = dest.stat().st_size
    video.duration_s = meta.duration_s
    video.fps = meta.fps
    video.width = meta.width
    video.height = meta.height
    db.commit()
    return video


@router.get("", response_model=list[VideoOut])
def list_videos(db: Session = Depends(get_db)):
    return db.scalars(select(Video).order_by(Video.created_at.desc())).all()


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    return video


@router.get("/{video_id}/file")
def get_video_file(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    # FileResponse supports Range requests — required for <video> seeking.
    return FileResponse(video.path, media_type="video/mp4")


@router.get("/{video_id}/ground_truth")
def get_ground_truth(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    if not video.gt_path:
        raise HTTPException(404, "Video has no ground truth")
    return FileResponse(video.gt_path, media_type="application/json")
