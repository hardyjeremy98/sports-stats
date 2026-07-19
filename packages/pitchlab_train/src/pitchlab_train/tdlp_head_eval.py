"""Fast cached-feature evaluation + parameter sweep for the TDLP head.

The full benchmark re-runs DINOv2 over every frozen-detection crop on every
eval (~15 min/tier), which makes tuning the association loop impractical. This
module splits that: `extract-holdout` runs the expensive DINOv2 pass ONCE over
the held-out frozen detections and caches per-detection features; `sweep` then
runs only the cheap loop + `evaluate_run` scoring at many parameter settings so
we can minimize **ID switches** (Jeremy: IDsw matters more than purity) without
re-embedding.

Cache npz per sequence: frame_ids, boxes_xyxy (pixel), confs, apps, width, height.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch
from pitchlab_core.evaluation import evaluate_run, headline_metrics
from pitchlab_core.gt import GroundTruth
from pitchlab_core.schemas.geometry import Box
from pitchlab_core.stages.associate.embedders.base import get_embedder
from pitchlab_core.stages.track.tdlp.feature_assembly import build_object_data
from pitchlab_core.stages.track.tdlp.loop import TDLPTracker
from pitchlab_core.stages.track.tdlp.model import ModalityConfig, build_head, feature_specs
from pitchlab_core.video import iter_frames, probe


def _parse_det_txt(path: Path) -> dict[int, list[tuple[list[float], float]]]:
    """MOT det.txt -> {0-based frame: [(xyxy, conf)]}."""
    by_frame: dict[int, list[tuple[list[float], float]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split(",")
        f = int(float(p[0])) - 1
        x, y, w, h = (float(p[i]) for i in (2, 3, 4, 5))
        conf = float(p[6])
        by_frame.setdefault(f, []).append(([x, y, x + w, y + h], conf))
    return by_frame


# --------------------------------------------------------------------------- #
def extract_holdout(gt_path, video_path, det_txt, embedder, *, min_box_height_px=1):
    gt = GroundTruth.model_validate(json.loads(Path(gt_path).read_text()))
    by_frame = _parse_det_txt(Path(det_txt))
    meta = probe(video_path, 1)
    W = gt.width or meta.width
    H = gt.height or meta.height
    frame_ids, boxes, confs, apps = [], [], [], []
    for fr in iter_frames(meta, stride=1):
        dets = by_frame.get(fr.frame_idx, [])
        if not dets:
            continue
        h_img, w_img = fr.image.shape[:2]
        crops, keep = [], []
        for k, (xyxy, _c) in enumerate(dets):
            x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
            x2, y2 = min(w_img, int(xyxy[2])), min(h_img, int(xyxy[3]))
            if x2 > x1 and y2 > y1 and (y2 - y1) >= min_box_height_px:
                crops.append(fr.image[y1:y2, x1:x2])
                keep.append(k)
        if not crops:
            continue
        emb, _q = embedder.embed(crops)
        for j, k in enumerate(keep):
            xyxy, conf = dets[k]
            frame_ids.append(fr.frame_idx)
            boxes.append(xyxy)
            confs.append(conf)
            apps.append(emb[j].astype(np.float32))
    return {
        "frame_ids": np.asarray(frame_ids, np.int64),
        "boxes": np.asarray(boxes, np.float32).reshape(-1, 4),
        "confs": np.asarray(confs, np.float32),
        "apps": np.stack(apps) if apps else np.zeros((0, embedder.dim), np.float32),
        "width": W, "height": H, "appearance_dim": embedder.dim,
        "frame_count": meta.frame_count, "fps": meta.fps or gt.fps,
    }


def _frames_from_cache(z, appearance_dim: int) -> list[tuple[int, list[dict]]]:
    W, H = int(z["width"]), int(z["height"])
    by_frame: dict[int, list[dict]] = {}
    for fi, box, conf, app in zip(z["frame_ids"], z["boxes"], z["confs"], z["apps"]):
        b = Box(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3]))
        data = build_object_data(
            b, float(conf), W, H, keypoints=None, appearance=np.asarray(app, np.float32),
            use_keypoints=False, use_appearance=True, appearance_dim=appearance_dim,
        )
        by_frame.setdefault(int(fi), []).append(data)
    return [(f, by_frame[f]) for f in sorted(by_frame)]


def _write_run_dir(run_dir: Path, tracklets, frame_count: int, fps: float):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "video": {"sample_stride": 1, "frame_count": int(frame_count), "fps": float(fps)},
    }))
    (run_dir / "tracklets.json").write_text(
        json.dumps([t.model_dump() for t in tracklets])
    )


def sweep(cache_dir, manifest_path, checkpoint, out_dir, *, device="cuda",
          iou_threshold=0.5, max_seqs=None, grid=None):
    payload = torch.load(checkpoint, map_location=device)
    cfg_d = payload["config"]
    appearance_dim = cfg_d["appearance_dim"]
    modality = ModalityConfig(use_keypoints=False, use_appearance=True,
                              appearance_dim=appearance_dim)
    model = build_head(modality, hidden_dim=cfg_d["hidden_dim"], mm_dim=cfg_d["hidden_dim"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    specs = feature_specs(modality)

    manifest = json.loads(Path(manifest_path).read_text())
    seqs = [s for s in manifest["sequences"] if s["role"] == "held_out"]
    if max_seqs:
        seqs = seqs[:max_seqs]

    # load caches + GT once
    loaded = []
    for s in seqs:
        cache = Path(cache_dir) / f"{s['name']}.npz"
        if not cache.exists():
            print(f"[sweep] missing cache {cache}, skipping {s['name']}")
            continue
        z = np.load(cache)
        gt = GroundTruth.model_validate(json.loads(Path(s["gt"]).read_text()))
        loaded.append((s["name"], z, gt))
    print(f"[sweep] {len(loaded)} sequences loaded")

    grid = grid or {
        "sim_threshold": [0.5, 0.7, 0.85, 0.95],
        "remember_threshold": [cfg_d.get("remember", 20)],
        "new_tracklet_detection_threshold": [0.5, 0.9],
        "min_length": [5],
    }
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        per_seq = []
        for name, z, gt in loaded:
            frames = _frames_from_cache(z, appearance_dim)
            tracker = TDLPTracker(
                model=model, feature_specs=specs, device=device,
                remember_threshold=params["remember_threshold"],
                detection_threshold=0.4, sim_threshold=params["sim_threshold"],
                initialization_threshold=1,
                new_tracklet_detection_threshold=params["new_tracklet_detection_threshold"],
            )
            tracklets = tracker.track_clip(frames, min_length=params["min_length"])
            rd = Path(out_dir) / f"{'_'.join(f'{k}{v}' for k, v in params.items())}" / name
            _write_run_dir(rd, tracklets, int(z["frame_count"]), float(z["fps"]))
            res = evaluate_run(rd, gt, iou_threshold, min_track_length=params["min_length"])
            per_seq.append(headline_metrics(res))
        agg = {m: round(st.mean([h[m] for h in per_seq if h.get(m) is not None]), 4)
               for m in ["idsw_tracklet", "hota_tracklet", "idf1_tracklet",
                         "tracklet_purity", "mixed_track_seconds"]
               if any(h.get(m) is not None for h in per_seq)}
        results.append((params, agg))
        print(f"[sweep] {params} -> {agg}", flush=True)

    results.sort(key=lambda r: (r[1].get("idsw_tracklet", 1e9),
                                -r[1].get("hota_tracklet", 0)))
    print("\n=== SWEEP RESULTS (sorted by IDsw asc, then HOTA desc) ===")
    for params, agg in results:
        print(f"  {agg}   <-  {params}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "sweep_results.json").write_text(
        json.dumps([{"params": p, "metrics": a} for p, a in results], indent=1)
    )
    return results


# --------------------------------------------------------------------------- #
def _cmd_extract(args):
    embedder = get_embedder(args.embedder)
    embedder.prepare(args.device)
    manifest = json.loads(Path(args.tier_manifest).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import time
    for s in manifest["sequences"]:
        if s["role"] != "held_out":
            continue
        det_txt = Path(args.exchange_dir) / s["name"] / "det.txt"
        if not det_txt.exists():
            print(f"[extract-holdout] missing {det_txt}, skip {s['name']}")
            continue
        t0 = time.time()
        cache = extract_holdout(s["gt"], s["video"], det_txt, embedder)
        np.savez_compressed(out_dir / f"{s['name']}.npz", **cache)
        print(f"[extract-holdout] {s['name']}: {len(cache['frame_ids'])} dets "
              f"-> {out_dir / (s['name'] + '.npz')} in {time.time() - t0:.1f}s", flush=True)


def _cmd_sweep(args):
    sweep(args.cache_dir, args.tier_manifest, args.checkpoint, args.out_dir,
          device=args.device, iou_threshold=args.iou_threshold, max_seqs=args.max_seqs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract-holdout")
    ex.add_argument("--tier-manifest", required=True)
    ex.add_argument("--exchange-dir", required=True)
    ex.add_argument("--out-dir", required=True)
    ex.add_argument("--embedder", default="dinov2")
    ex.add_argument("--device", default="cuda")
    ex.set_defaults(func=_cmd_extract)

    sw = sub.add_parser("sweep")
    sw.add_argument("--cache-dir", required=True)
    sw.add_argument("--tier-manifest", required=True)
    sw.add_argument("--checkpoint", required=True)
    sw.add_argument("--out-dir", required=True)
    sw.add_argument("--device", default="cuda")
    sw.add_argument("--iou-threshold", type=float, default=0.5)
    sw.add_argument("--max-seqs", type=int, default=None)
    sw.set_defaults(func=_cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
