"""Locate clips with ground truth for GT-scored training experiments.

Convention: a clip's ground truth lives in a sibling JSON file next to it,
`<stem>.gt.json` — the exact layout `matchlab_train.datasets.soccernet_tracking
.ingest_soccernet` writes under `data/videos/soccernet/` (e.g. `SNMOT-116.mp4`
+ `SNMOT-116.gt.json`). Shared by `eval_pipelines` (optional per-clip scoring)
and `reid_ablation` (the whole point of the experiment).
"""

from __future__ import annotations

from pathlib import Path

from matchlab_core.gt import GroundTruth


def sibling_gt_path(clip: str | Path) -> Path:
    clip = Path(clip)
    return clip.parent / f"{clip.stem}.gt.json"


def load_sibling_gt(clip: str | Path) -> GroundTruth | None:
    """The clip's sibling GT, or None if it has none (not every clip is
    labelled — callers must treat that as "skip", not an error)."""
    gt_path = sibling_gt_path(clip)
    if not gt_path.exists():
        return None
    return GroundTruth.model_validate_json(gt_path.read_text())


def discover_clips_with_gt(clips_dir: str | Path, max_clips: int) -> list[tuple[Path, GroundTruth]]:
    """Sorted `*.mp4` in `clips_dir` that have a sibling `.gt.json`, capped at
    `max_clips`. Clips without GT are silently excluded, not counted against
    the cap."""
    pairs: list[tuple[Path, GroundTruth]] = []
    for clip in sorted(Path(clips_dir).glob("*.mp4")):
        gt = load_sibling_gt(clip)
        if gt is not None:
            pairs.append((clip, gt))
        if len(pairs) >= max_clips:
            break
    return pairs
