"""Preliminary TDLP head training (SPO-40 harness) — corrupt-and-recover
link-prediction on multi-cue tracker-states.

SCOPE / LICENSING: this trains the vendored TDLP head (bbox + DINOv2 global
appearance; pose omitted for the first pass to keep extraction GPU-fast) on
GT-boxed tracker-states from the **tuning** split of SoccerNet/SportsMOT. Those
tiers are NON-COMMERCIAL / research-only, so any checkpoint produced here is a
**NON-SHIPPABLE PRELIMINARY** used only to (a) de-risk the training harness and
(b) produce a first Bar A data point. The SHIPPABLE retrain on permissive data
(MEVA / PeopleSansPeople) is SPO-40 proper and is blocked on the SPO-39
data-source decision (HITL). Every checkpoint records its data + license so the
SPO-41 gate refuses it on the shipping path.

Two entry points:
  extract  — decode a sequence, embed GT crops (DINOv2), dump tracker-states.
  train    — corrupt-and-recover link-prediction over dumped states -> checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pitchlab_core.gt import GroundTruth
from pitchlab_core.stages.associate.embedders.base import get_embedder
from pitchlab_core.stages.track.tdlp.model import BBOX_DIM, ModalityConfig, build_head
from pitchlab_core.video import iter_frames, probe
from torch import nn

APPEARANCE_DIM_DEFAULT = 384


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_sequence(
    gt_path: str,
    video_path: str,
    embedder,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    min_box_height_px: int = 40,
) -> dict:
    """Decode `video_path`, embed each GT crop, return a serializable dict:
    {"width","height","appearance_dim","frames": {frame_idx: [{"id","bbox","app"}]}}.
    bbox = [x, y, w, h, conf] normalized by image dims (conf=1 for GT)."""
    gt = GroundTruth.model_validate(json.loads(Path(gt_path).read_text()))
    per_frame: dict[int, list[tuple[int, object]]] = defaultdict(list)
    for tr in gt.tracks:
        if tr.role == "ball":
            continue
        for f in tr.frames:
            per_frame[f.frame_idx].append((tr.track_id, f.box))

    meta = probe(video_path, stride)
    W = gt.width or meta.width
    H = gt.height or meta.height
    frames_out: dict[int, list[dict]] = {}
    for fr in iter_frames(meta, stride=stride, max_frames=max_frames):
        objs = per_frame.get(fr.frame_idx, [])
        if not objs:
            continue
        h_img, w_img = fr.image.shape[:2]
        crops: list[np.ndarray] = []
        keep: list[int] = []
        for k, (_gid, box) in enumerate(objs):
            if (box.y2 - box.y1) < min_box_height_px:
                continue
            x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
            x2, y2 = min(w_img, int(box.x2)), min(h_img, int(box.y2))
            if x2 > x1 and y2 > y1:
                crops.append(fr.image[y1:y2, x1:x2])
                keep.append(k)
        if not crops:
            continue
        emb, _q = embedder.embed(crops)
        recs = []
        for j, k in enumerate(keep):
            gid, box = objs[k]
            bbox = [box.x1 / W, box.y1 / H, (box.x2 - box.x1) / W, (box.y2 - box.y1) / H, 1.0]
            recs.append({"id": int(gid), "bbox": bbox, "app": emb[j].astype(np.float32)})
        frames_out[fr.frame_idx] = recs
    return {
        "width": W,
        "height": H,
        "appearance_dim": embedder.dim,
        "frames": frames_out,
    }


def save_states(states: dict, path: str) -> None:
    """Flatten to npz (frame_idx, obj_id, bbox[5], app[D]) rows."""
    frame_ids, obj_ids, bboxes, apps = [], [], [], []
    for f_idx, recs in states["frames"].items():
        for r in recs:
            frame_ids.append(int(f_idx))
            obj_ids.append(int(r["id"]))
            bboxes.append(np.asarray(r["bbox"], dtype=np.float32))
            apps.append(np.asarray(r["app"], dtype=np.float32))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        obj_ids=np.asarray(obj_ids, dtype=np.int64),
        bboxes=np.stack(bboxes) if bboxes else np.zeros((0, BBOX_DIM), np.float32),
        apps=np.stack(apps) if apps else np.zeros((0, states["appearance_dim"]), np.float32),
        width=states["width"],
        height=states["height"],
        appearance_dim=states["appearance_dim"],
    )


def load_states(path: str) -> dict:
    z = np.load(path)
    frames: dict[int, list[dict]] = defaultdict(list)
    for fi, oi, bb, ap in zip(z["frame_ids"], z["obj_ids"], z["bboxes"], z["apps"]):
        frames[int(fi)].append({"id": int(oi), "bbox": bb, "app": ap})
    return {
        "appearance_dim": int(z["appearance_dim"]),
        "frames": dict(frames),
    }


# --------------------------------------------------------------------------- #
# Corrupt-and-recover clip sampling
# --------------------------------------------------------------------------- #
def _appearance_vec(app: np.ndarray) -> np.ndarray:
    return np.concatenate([app, np.array([1.0], dtype=np.float32)])


def sample_clip(
    seq_frames: dict[int, list[dict]],
    sorted_fidx: list[int],
    pos: int,
    *,
    remember: int,
    appearance_dim: int,
    rng: random.Random,
    max_gap: int = 0,
    history_dropout: float = 0.0,
):
    """Build one training clip anchored at sorted_fidx[pos] (the 'current'
    frame). Tracks = ids seen in the preceding `remember` sampled frames (their
    per-frame history); detections = objects at the current frame. Returns torch
    tensors + a (n_tracks, n_dets) 0/1 target of same-id links, or None if the
    clip is degenerate.

    Augmentations that directly target ID switches (re-linking after loss):
    - `max_gap`: end the observed window ``gap`` sampled-frames *before* the
      current frame (gap ~U[0, max_gap]), so the head must re-associate a track
      last seen several frames ago — the exact case a lost tracklet respawns on.
    - `history_dropout`: randomly drop observed frames per track (simulated
      missed detections), so history need not be contiguous."""
    gap = rng.randint(0, max_gap) if max_gap > 0 else 0
    cur_fidx = sorted_fidx[pos]
    win_end = max(0, pos - gap)
    window = sorted_fidx[max(0, win_end - remember):win_end]
    if not window or cur_fidx not in seq_frames:
        return None

    # history per id (chronological); most recent goes to the tail slot.
    hist: dict[int, list[dict]] = defaultdict(list)
    for fi in window:
        for r in seq_frames[fi]:
            if history_dropout > 0.0 and rng.random() < history_dropout:
                continue
            hist[r["id"]].append(r)
    track_ids = [tid for tid, h in hist.items() if h]
    dets = seq_frames[cur_fidx]
    if not track_ids or not dets:
        return None

    nt, nd, T = len(track_ids), len(dets), remember
    app_full = appearance_dim + 1
    track_bbox = torch.zeros(nt, T, BBOX_DIM)
    track_app = torch.zeros(nt, T, app_full)
    track_mask = torch.ones(nt, T, dtype=torch.bool)  # True = missing
    for i, tid in enumerate(track_ids):
        h = hist[tid][-T:]
        for j, r in enumerate(h):
            slot = T - len(h) + j  # tail-aligned (matches inference)
            track_bbox[i, slot] = torch.from_numpy(np.asarray(r["bbox"], np.float32))
            track_app[i, slot] = torch.from_numpy(_appearance_vec(r["app"]))
            track_mask[i, slot] = False

    det_bbox = torch.zeros(nd, BBOX_DIM)
    det_app = torch.zeros(nd, app_full)
    det_mask = torch.zeros(nd, dtype=torch.bool)
    for j, r in enumerate(dets):
        det_bbox[j] = torch.from_numpy(np.asarray(r["bbox"], np.float32))
        det_app[j] = torch.from_numpy(_appearance_vec(r["app"]))

    target = torch.zeros(nt, nd)
    det_ids = [r["id"] for r in dets]
    for i, tid in enumerate(track_ids):
        for j, did in enumerate(det_ids):
            if tid == did:
                target[i, j] = 1.0
    return {
        "track": {"bbox": track_bbox, "appearance": track_app},
        "track_mask": track_mask,
        "det": {"bbox": det_bbox, "appearance": det_app},
        "det_mask": det_mask,
        "target": target,
    }


def train(
    state_paths: list[str],
    out_ckpt: str,
    *,
    device: str = "cuda",
    epochs: int = 8,
    clips_per_epoch: int = 2000,
    remember: int = 20,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    ffn_dim: int = 256,
    seed: int = 0,
    max_gap: int = 0,
    history_dropout: float = 0.0,
    meta: dict | None = None,
) -> dict:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    seqs = []
    appearance_dim = APPEARANCE_DIM_DEFAULT
    for p in state_paths:
        s = load_states(p)
        appearance_dim = s["appearance_dim"]
        fidx = sorted(s["frames"].keys())
        if len(fidx) > 2:
            seqs.append((s["frames"], fidx))
    if not seqs:
        raise RuntimeError("no usable sequences in state_paths")

    cfg = ModalityConfig(use_keypoints=False, use_appearance=True, appearance_dim=appearance_dim)
    model = build_head(
        cfg, hidden_dim=hidden_dim, mm_dim=hidden_dim,
        track_encoder_n_heads=n_heads, track_encoder_n_layers=n_layers,
        track_encoder_ffn_dim=ffn_dim, sph_hidden_dim=hidden_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    history = []
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for _ in range(clips_per_epoch):
            frames, fidx = rng.choice(seqs)
            pos = rng.randint(1, len(fidx) - 1)
            clip = sample_clip(
                frames, fidx, pos, remember=remember, appearance_dim=appearance_dim, rng=rng,
                max_gap=max_gap, history_dropout=history_dropout,
            )
            if clip is None:
                continue
            tf = {k: v.unsqueeze(0).to(device) for k, v in clip["track"].items()}
            df = {k: v.unsqueeze(0).to(device) for k, v in clip["det"].items()}
            tm = clip["track_mask"].unsqueeze(0).to(device)
            dm = clip["det_mask"].unsqueeze(0).to(device)
            target = clip["target"].unsqueeze(0).to(device)

            agg_logits, _ = model(tf, tm, df, dm)  # (1, nt, nd)
            # class imbalance: ~1 positive per row -> weight positives up
            n_pos = target.sum().clamp(min=1.0)
            n_neg = (target.numel() - target.sum()).clamp(min=1.0)
            pos_w = (n_neg / n_pos).clamp(max=50.0)
            loss_el = bce(agg_logits, target)
            weight = torch.where(target > 0.5, pos_w, torch.ones_like(target))
            loss = (loss_el * weight).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        avg = total / max(1, n)
        history.append(avg)
        print(f"[tdlp-train] epoch {epoch + 1}/{epochs} loss={avg:.4f} clips={n}", flush=True)

    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": {
            "use_keypoints": False,
            "use_appearance": True,
            "appearance_dim": appearance_dim,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ffn_dim": ffn_dim,
            "remember": remember,
            "max_gap": max_gap,
            "history_dropout": history_dropout,
        },
        "train_meta": meta or {},
        "loss_history": history,
    }
    torch.save(payload, out_ckpt)
    print(f"[tdlp-train] saved {out_ckpt}", flush=True)
    return payload["config"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_extract(args):
    embedder = get_embedder(args.embedder)
    embedder.prepare(args.device)
    manifest = json.loads(Path(args.tier_manifest).read_text())
    roles = set(args.roles.split(","))
    out_dir = Path(args.out_dir)
    for seq in manifest["sequences"]:
        if seq["role"] not in roles:
            continue
        t0 = time.time()
        states = extract_sequence(
            seq["gt"], seq["video"], embedder,
            stride=args.stride, max_frames=args.max_frames,
        )
        out = out_dir / f"{seq['name']}.npz"
        save_states(states, str(out))
        n = sum(len(v) for v in states["frames"].values())
        print(f"[extract] {seq['name']} ({seq['role']}): {len(states['frames'])} frames, "
              f"{n} objs -> {out} in {time.time() - t0:.1f}s", flush=True)


def _cmd_train(args):
    paths = sorted(str(p) for p in Path(args.states_dir).glob("*.npz"))
    if args.include:
        keep = set(args.include.split(","))
        paths = [p for p in paths if Path(p).stem in keep]
    print(f"[train] {len(paths)} sequences: {[Path(p).stem for p in paths]}", flush=True)
    cfg = train(
        paths, args.out_ckpt, device=args.device, epochs=args.epochs,
        clips_per_epoch=args.clips_per_epoch, remember=args.remember,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads,
        ffn_dim=args.ffn_dim, seed=args.seed,
        max_gap=args.max_gap, history_dropout=args.history_dropout,
        meta={"states_dir": args.states_dir, "sequences": [Path(p).stem for p in paths],
              "note": "PRELIMINARY non-shippable: trained on NC eval-tier tuning data"},
    )
    print(f"[train] done: {cfg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract")
    ex.add_argument("--tier-manifest", required=True)
    ex.add_argument("--roles", default="tuning")
    ex.add_argument("--out-dir", required=True)
    ex.add_argument("--embedder", default="dinov2")
    ex.add_argument("--device", default="cuda")
    ex.add_argument("--stride", type=int, default=1)
    ex.add_argument("--max-frames", type=int, default=None)
    ex.set_defaults(func=_cmd_extract)

    tr = sub.add_parser("train")
    tr.add_argument("--states-dir", required=True)
    tr.add_argument("--out-ckpt", required=True)
    tr.add_argument("--include", default="")
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--clips-per-epoch", type=int, default=2000)
    tr.add_argument("--remember", type=int, default=20)
    tr.add_argument("--hidden-dim", type=int, default=128)
    tr.add_argument("--n-layers", type=int, default=2)
    tr.add_argument("--n-heads", type=int, default=4)
    tr.add_argument("--ffn-dim", type=int, default=256)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--max-gap", type=int, default=0,
                    help="train re-linking across gaps of up to N sampled frames (cuts IDsw)")
    tr.add_argument("--history-dropout", type=float, default=0.0,
                    help="randomly drop observed history frames (simulate missed detections)")
    tr.set_defaults(func=_cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
