"""Browse labelled training datasets (downloaded by pitchlab-train into
data/experiments/datasets/). Read-only: list datasets, sample labelled images.

Two layouts are understood:
  COCO detection  — <split>/_annotations.coco.json + images in the split dir
  YOLOv8 pose     — data.yaml (kpt_shape) + <split>/{images,labels}/ pairs
All annotation coordinates are returned normalized (0..1) so the UI can draw
without knowing pixel sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from pitchlab_server.db import get_db
from pitchlab_server.settings import get_settings

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

_SPLITS = ("train", "valid", "test")
_IMG_EXTS = {".jpg", ".jpeg", ".png"}


class DatasetOut(BaseModel):
    name: str
    kind: str  # "detection-coco" | "keypoints-yolo" | "unknown"
    images: int
    annotations: int
    classes: list[str]
    splits: dict[str, int]  # split -> image count


class SampleBox(BaseModel):
    x: float  # all normalized 0..1, xywh
    y: float
    w: float
    h: float
    label: str


class SampleKeypoint(BaseModel):
    x: float
    y: float
    v: float  # visibility


class Sample(BaseModel):
    url: str
    split: str
    boxes: list[SampleBox] = []
    keypoints: list[SampleKeypoint] = []


class SamplesOut(BaseModel):
    name: str
    kind: str
    samples: list[Sample]


def _root() -> Path:
    return get_settings().data_dir / "experiments" / "datasets"


def _kind(d: Path) -> str:
    if list(d.rglob("_annotations.coco.json")):
        return "detection-coco"
    if (d / "data.yaml").exists():
        return "keypoints-yolo" if "kpt_shape" in (d / "data.yaml").read_text() else "detection-yolo"
    return "unknown"


@router.get("", response_model=list[DatasetOut])
def list_datasets():
    root = _root()
    if not root.exists():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        kind = _kind(d)
        splits: dict[str, int] = {}
        images = 0
        annotations = 0
        classes: set[str] = set()
        if kind == "detection-coco":
            for split in _SPLITS:
                coco_path = d / split / "_annotations.coco.json"
                if not coco_path.exists():
                    continue
                coco = json.loads(coco_path.read_text())
                splits[split] = len(coco["images"])
                images += len(coco["images"])
                annotations += len(coco["annotations"])
                classes |= {c["name"] for c in coco["categories"] if c.get("supercategory") != "none" or True}
        else:
            names = []
            data_yaml = d / "data.yaml"
            if data_yaml.exists():
                names = list(yaml.safe_load(data_yaml.read_text()).get("names", []))
            classes |= set(map(str, names))
            for split in _SPLITS:
                img_dir = d / split / "images"
                if not img_dir.exists():
                    continue
                n = sum(1 for f in img_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
                splits[split] = n
                images += n
                lbl_dir = d / split / "labels"
                if lbl_dir.exists():
                    annotations += sum(1 for f in lbl_dir.glob("*.txt"))
        out.append(
            DatasetOut(
                name=d.name, kind=kind, images=images, annotations=annotations,
                classes=sorted(classes), splits=splits,
            )
        )
    return out


@router.get("/{name}/samples", response_model=SamplesOut)
def dataset_samples(name: str, n: int = 12, split: str | None = None):
    d = _dataset_dir(name)
    kind = _kind(d)
    if kind == "detection-coco":
        samples = _coco_samples(name, d, n, split)
    elif kind in ("keypoints-yolo", "detection-yolo"):
        samples = _yolo_samples(name, d, n, split, keypoints=kind == "keypoints-yolo")
    else:
        raise HTTPException(422, f"Don't know how to sample dataset '{name}'")
    return SamplesOut(name=name, kind=kind, samples=samples)


class MaterializeIn(BaseModel):
    split: str = "test"
    fps: float = 2.0  # slideshow pace: slow enough to eyeball per-frame quality


@router.post("/{name}/materialize")
def materialize_split(name: str, body: MaterializeIn, db: Session = Depends(get_db)):
    """Stitch a dataset split's images into an mp4 and register it as an
    uploaded Video, so labelled frames can be run through any pipeline config.

    Note: dataset splits mix frames from different matches — runs on this
    footage are for judging detection/calibration frame-by-frame, not
    tracking or stats (tracklets across unrelated frames are meaningless).
    """
    import cv2
    from pitchlab_core.video import VideoWriter, probe

    from pitchlab_server.api.videos import register_video_file

    if body.split not in _SPLITS:
        raise HTTPException(422, f"split must be one of {_SPLITS}")
    d = _dataset_dir(name)
    kind = _kind(d)
    if kind == "detection-coco":
        img_dir = d / body.split
        images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in _IMG_EXTS) if img_dir.exists() else []
    else:
        img_dir = d / body.split / "images"
        images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in _IMG_EXTS) if img_dir.exists() else []
    if not images:
        raise HTTPException(404, f"Dataset '{name}' has no images in split '{body.split}'")

    settings = get_settings()
    filename = f"dataset-{name}-{body.split}.mp4"
    dest = settings.videos_dir / filename

    # All frames must share one size: letterbox everything onto the first
    # frame's canvas (roboflow exports are uniform per version, so this is
    # usually a no-op).
    first = cv2.imread(str(images[0]))
    if first is None:
        raise HTTPException(500, f"Unreadable image {images[0].name}")
    h, w = first.shape[:2]
    writer = VideoWriter(dest, body.fps, (w, h))
    try:
        for f in images:
            img = cv2.imread(str(f))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = _letterbox(img, w, h)
            writer.write(img)
    finally:
        writer.close()

    meta = probe(dest)
    return register_video_file(db, filename, dest, meta)


def _letterbox(img, w: int, h: int):
    import cv2
    import numpy as np

    scale = min(w / img.shape[1], h / img.shape[0])
    nw, nh = int(img.shape[1] * scale), int(img.shape[0] * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=resized.dtype)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


@router.get("/{name}/images/{rel_path:path}")
def dataset_image(name: str, rel_path: str):
    d = _dataset_dir(name)
    target = (d / rel_path).resolve()
    if not target.is_relative_to(d.resolve()) or not target.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(target)


def _dataset_dir(name: str) -> Path:
    if "/" in name or ".." in name:
        raise HTTPException(404, "Dataset not found")
    d = _root() / name
    if not d.is_dir():
        raise HTTPException(404, "Dataset not found")
    return d


def _spread(items: list, n: int) -> list:
    """Evenly-spaced sample so the grid shows dataset variety, deterministic."""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _coco_samples(name: str, d: Path, n: int, split: str | None) -> list[Sample]:
    samples: list[Sample] = []
    splits = [split] if split else [s for s in _SPLITS if (d / s / "_annotations.coco.json").exists()]
    per_split = max(1, n // max(1, len(splits)))
    for s in splits:
        coco_path = d / s / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        coco = json.loads(coco_path.read_text())
        cat_by_id = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_img: dict[int, list] = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        # Prefer images that actually have labels.
        labelled = [im for im in coco["images"] if im["id"] in anns_by_img]
        for im in _spread(labelled, per_split):
            boxes = [
                SampleBox(
                    x=a["bbox"][0] / im["width"],
                    y=a["bbox"][1] / im["height"],
                    w=a["bbox"][2] / im["width"],
                    h=a["bbox"][3] / im["height"],
                    label=cat_by_id.get(a["category_id"], "?"),
                )
                for a in anns_by_img[im["id"]]
            ]
            samples.append(
                Sample(url=f"/api/datasets/{name}/images/{s}/{im['file_name']}", split=s, boxes=boxes)
            )
    return samples[:n]


def _yolo_samples(name: str, d: Path, n: int, split: str | None, keypoints: bool) -> list[Sample]:
    data_yaml = yaml.safe_load((d / "data.yaml").read_text()) if (d / "data.yaml").exists() else {}
    names = data_yaml.get("names", [])
    samples: list[Sample] = []
    splits = [split] if split else [s for s in _SPLITS if (d / s / "images").is_dir()]
    per_split = max(1, n // max(1, len(splits)))
    for s in splits:
        img_dir = d / s / "images"
        lbl_dir = d / s / "labels"
        imgs = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
        labelled = [f for f in imgs if (lbl_dir / f"{f.stem}.txt").exists()]
        for f in _spread(labelled, per_split):
            boxes: list[SampleBox] = []
            kps: list[SampleKeypoint] = []
            for line in (lbl_dir / f"{f.stem}.txt").read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append(
                    SampleBox(
                        x=cx - w / 2, y=cy - h / 2, w=w, h=h,
                        label=str(names[cls]) if cls < len(names) else str(cls),
                    )
                )
                if keypoints:
                    rest = list(map(float, parts[5:]))
                    for i in range(0, len(rest) - 2, 3):
                        kps.append(SampleKeypoint(x=rest[i], y=rest[i + 1], v=rest[i + 2]))
            samples.append(
                Sample(
                    url=f"/api/datasets/{name}/images/{s}/images/{f.name}",
                    split=s, boxes=boxes, keypoints=kps,
                )
            )
    return samples[:n]
