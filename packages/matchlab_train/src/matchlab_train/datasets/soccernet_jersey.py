"""SoccerNet jersey-number dataset adapter, for gate 1 only.

Gate 1 exists to reproduce a PUBLISHED number (87.45% tracklet accuracy) on
the data it was published on, before the reader is pointed at SNMOT. A reader
that silently mis-wires its tokenizer still produces plausible-looking
per-tracklet outputs, so the only way to catch that is a reference metric.
"""

from __future__ import annotations

import json
from pathlib import Path

_ILLEGIBLE = -1


def load_jersey_tracklets(root: Path, split: str) -> dict[str, tuple[list[Path], int | None]]:
    """Tracklet id -> (sorted image paths, GT number or None if illegible)."""
    root = Path(root)
    gt_path = root / split / f"{split}_gt.json"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"{gt_path} not found. The SoccerNet jersey split is not in this "
            "repo's data tree by default; see configs/datasets/README.md."
        )
    labels = json.loads(gt_path.read_text())
    images_root = root / split / "images"
    out: dict[str, tuple[list[Path], int | None]] = {}
    for tid, number in labels.items():
        paths = sorted((images_root / str(tid)).glob("*.jpg"))
        n = None if int(number) == _ILLEGIBLE else int(number)
        out[str(tid)] = (paths, n)
    return out
