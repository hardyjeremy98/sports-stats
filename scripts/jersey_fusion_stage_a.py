"""Stage A driver (task 7, jersey-OCR fusion ablation): run the oracle TRACK
stage's in-repo osnet embedder on every SNMOT clip that has GT, and write each
clip's frame_features.npz under data/experiments/jersey-fusion/features/<stem>/.

Deliberately NOT a matchlab_train experiment -- this is a one-shot substrate
build, not a repeated benchmark. Reuses PipelineConfig / StageContext exactly
as jersey_channel.py does, and the oracle stage itself (SPO-85), so the
fragment ids this writes are the SAME fragment_tracks() ids the jersey
evidence cache used (default gap_frames=2, min_fragment_frames=1).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from matchlab_core.artifacts import ArtifactStore
from matchlab_core.config import PipelineConfig
from matchlab_core.interfaces import StageContext
from matchlab_core.stages.track.oracle import OracleTracker
from matchlab_core.video import probe
from matchlab_train.gt_clips import discover_clips_with_gt

CLIPS_DIR = "data/videos/soccernet"
OUT_DIR = Path("data/experiments/jersey-fusion/features")
BASE_CFG = "configs/pipeline.baseline-observe.yaml"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = PipelineConfig.from_yaml(BASE_CFG)
    pairs = discover_clips_with_gt(CLIPS_DIR, 32)
    print(f"[stage-a] {len(pairs)} clips with GT", flush=True)

    tracker = OracleTracker(
        gap_frames=2,
        min_fragment_frames=1,
        features_backend="in-repo",
        features_model="osnet",
    )

    for i, (clip, _gt) in enumerate(pairs):
        dest_dir = OUT_DIR / clip.stem
        dest = dest_dir / "frame_features.npz"
        if dest.exists():
            print(f"[stage-a] {i + 1}/{len(pairs)} {clip.name}: already done, skipping",
                  flush=True)
            continue
        t0 = time.time()
        ctx = StageContext(
            video=probe(clip, sample_stride=base_cfg.video.sample_stride),
            config=base_cfg,
            store=ArtifactStore(dest_dir),
            device="cuda",
        )
        tracker.params.gt_path = str(clip.with_suffix(".gt.json"))
        tracklets = tracker.track(ctx, [])
        dt = time.time() - t0
        print(
            f"[stage-a] {i + 1}/{len(pairs)} {clip.name}: {len(tracklets)} fragments, "
            f"{dt:.1f}s -> {dest}",
            flush=True,
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
