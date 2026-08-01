"""Run the WASB-SBDT ball-detector family over our SoccerNet frames and write
detections into the same cache `scripts/detector_bench.py` scores.

WASB (BMVC 2023) and the TrackNet-family baselines it ships are heatmap
models, not box detectors: they take a short clip of consecutive frames,
predict a per-frame ball heatmap at 512x288, and post-process it to a ball
CENTER POINT plus a blob score. They are the published state of the art for
sports ball detection, so any honest ball comparison has to include them --
but they cannot be run through the box-detector path, hence this separate
entry point.

Two consequences the report must carry:

  * Output is a point. We write a synthetic fixed-size box centred on the
    predicted point purely so the cache format is shared; `score-ball` scores
    on centre distance and never reads the box extent. No box-IoU number is
    ever computed for these models.
  * The blob score is an unbounded sum of heatmap weights, not a probability.
    `score_ball` detects this and switches to a quantile-derived threshold
    grid so these models are compared at their own best operating point.

The clip is centred on the scored frame -- frames (f-1, f, f+1), output taken
from the middle position -- so each model sees the temporal context it was
trained with.

Needs the upstream repo and its own deps; run it against a clone:

  uv run --with hydra-core --with torch --with torchvision --with opencv-python-headless \
    python scripts/wasb_bench.py --repo ../external-ball/WASB-SBDT \
    --models wasb tracknetv2 --roles tuning --stride 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# Each entry maps our candidate name -> (WASB `model=` config name, checkpoint).
MODELS = {
    "wasb": ("wasb", "wasb_soccer_best.pth.tar"),
    "tracknetv2": ("tracknetv2", "tracknetv2_soccer_best.pth.tar"),
    "restracknetv2": ("restracknetv2", "restracknetv2_soccer_best.pth.tar"),
    "monotrack": ("monotrack", "monotrack_soccer_best.pth.tar"),
    "ballseg": ("ballseg", "ballseg_soccer_best.pth.tar"),
    "deepball": ("deepball", "deepball_soccer_best.pth.tar"),
    "deepball-large": ("deepball_large", "deepball-large_soccer_best.pth.tar"),
}
# deepball/deepball-large use a different detector head in the upstream factory.
DEEPBALL_DETECTOR = {"deepball", "deepball-large"}

BOX_SIZE = 12.0  # synthetic box side; never used by centre-distance scoring


def build_detector(repo: Path, model_cfg: str, ckpt: Path):
    from hydra import compose, initialize_config_dir

    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    detector_name = "deepball" if model_cfg.startswith("deepball") else "tracknetv2"
    with initialize_config_dir(config_dir=str(src / "configs"), version_base=None):
        cfg = compose(
            config_name="eval",
            overrides=[
                f"model={model_cfg}",
                "dataset=soccer",
                f"detector={detector_name}",
                f"detector.model_path={ckpt}",
                "runner.device=cuda",
                # Upstream defaults to a 4-GPU DataParallel; this box has one.
                "runner.gpus=[0]",
            ],
        )
    from detectors import build_detector as _bd

    return _bd(cfg), cfg


class WasbRunner:
    def __init__(self, repo: Path, model_cfg: str, ckpt: Path,
                 score_threshold: float | None = None):
        self.detector, self.cfg = build_detector(repo, model_cfg, ckpt)
        if score_threshold is not None:
            # Upstream defaults to 0.5, tuned on its own soccer dataset. On
            # SoccerNet broadcast framing that floor discards the true ball on
            # most frames (measured: it emits nothing on the majority of frames
            # where the ball is clearly visible and GT-labelled). Capture at a
            # low floor instead and let the scorer's quantile grid choose the
            # operating point -- otherwise we would be reporting upstream's
            # threshold choice, not the model's ability.
            self.detector._postprocessor._score_threshold = score_threshold
        self.input_wh = self.detector.input_wh
        self.frames_in = self.detector.frames_in
        from dataloaders import build_img_transforms

        _, self.transform = build_img_transforms(self.cfg)

    def infer_clip(self, images, target_pos):
        """images: list of `frames_in` BGR arrays, chronological. Returns the
        detections for position `target_pos` within the clip, in source-image
        coordinates. The caller passes the position explicitly because a clip
        clamped to a sequence boundary is not centred on the scored frame."""
        import cv2
        import numpy as np
        import torch
        from dataloaders import get_transform
        from PIL import Image

        first = images[0]
        trans_input = get_transform(np.asarray(first), self.input_wh)
        trans_inv = get_transform(np.asarray(first), self.input_wh, inv=1)

        tensors = []
        for img in images:
            rgb = img[:, :, ::-1]
            warped = cv2.warpAffine(np.ascontiguousarray(rgb), trans_input,
                                    self.input_wh, flags=cv2.INTER_LINEAR)
            tensors.append(self.transform(Image.fromarray(warped)))
        batch = torch.cat(tensors, dim=0).unsqueeze(0)

        affine = {0: torch.tensor(trans_inv, dtype=torch.float32).unsqueeze(0)}
        with torch.no_grad():
            results, _ = self.detector.run_tensor(batch, affine)

        preds = results[0].get(target_pos, [])
        out = []
        for p in preds:
            x, y = float(p["xy"][0]), float(p["xy"][1])
            h = BOX_SIZE / 2
            out.append((float(p["score"]), [x - h, y - h, x + h, y + h], "ball"))
        return out


def main() -> None:
    import cv2
    from detector_bench import REPO as BENCH_REPO
    from detector_bench import cache_path, image_path, select_sequences, seq_frames

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="../external-ball/WASB-SBDT")
    ap.add_argument("--weights-dir", default="data/weights/wasb")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--roles", nargs="*", default=["tuning"])
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-dir", default="data/detector-bench/cache")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score-threshold", type=float, default=0.05,
                    help="heatmap capture floor; keep low, the scorer sweeps it")
    args = ap.parse_args()

    repo = (BENCH_REPO / args.repo).resolve()
    wdir = BENCH_REPO / args.weights_dir
    cache_dir = BENCH_REPO / args.cache_dir
    seqs = select_sequences(tuple(args.roles))

    for name in args.models:
        model_cfg, ckpt_name = MODELS[name]
        ckpt = wdir / ckpt_name
        if not ckpt.exists():
            print(f"[{name}] missing checkpoint {ckpt}, skipping")
            continue
        cand = f"wasbfam-{name}"
        print(f"== {cand}")
        try:
            runner = WasbRunner(repo, model_cfg, ckpt, args.score_threshold)
        except Exception as exc:
            print(f"  [{cand}] BUILD FAILED: {type(exc).__name__}: {exc}")
            continue

        span = runner.frames_in // 2
        for s in seqs:
            out_path = cache_path(cache_dir, cand, s["name"])
            if out_path.exists() and not args.force:
                print(f"  [{cand}] {s['name']}: cached, skipping")
                continue
            frames = seq_frames(BENCH_REPO / s["gt"], args.stride, args.limit)
            seq_len = json.loads((BENCH_REPO / s["gt"]).read_text())["seq_length"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            t0, n = time.time(), 0
            with open(out_path.with_suffix(".tmp"), "w") as fh:
                for f in frames:
                    # Clip centred on f, clamped to the sequence bounds.
                    lo = max(0, min(f - span, seq_len - runner.frames_in))
                    idxs = [lo + i for i in range(runner.frames_in)]
                    imgs = [cv2.imread(str(image_path(s["name"], i))) for i in idxs]
                    if any(im is None for im in imgs):
                        raise SystemExit(f"missing frame image near {f} in {s['name']}")
                    dets = runner.infer_clip(imgs, idxs.index(f))
                    n += len(dets)
                    fh.write(json.dumps({"frame_idx": f, "dets": dets}) + "\n")
            out_path.with_suffix(".tmp").rename(out_path)
            dt = time.time() - t0
            print(f"  [{cand}] {s['name']}: {len(frames)} frames, {n} dets, "
                  f"{1000 * dt / len(frames):.0f} ms/frame")


if __name__ == "__main__":
    main()
