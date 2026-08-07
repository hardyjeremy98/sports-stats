"""Jersey-OCR evidence for FOOTPASS fragments (with/without re-ID ablation).

The jersey channel's only prior validation was SNMOT oracle-fragment pairs
(50 fused zero-wrong merges vs 0 body-alone, 2026-08-01); this module puts it
on the FOOTPASS GT-fragment substrate so full re-ID (bootstrap_threads, both
passes) can be measured with and without it.

Two cached layers, so nothing is ever OCR'd twice:

  1. `<key>-crops.pkl` -- per-CROP reads, keyed (frame, player_id). Frames are
     the saved fullHD grid under data/experiments/footpass-appearance/<key>/img1
     (every 25th frame, the same grid the appearance embeddings used); boxes
     are the FOOTPASS GT ROI_* columns. Fragmentation-independent: any
     max_gap_frames setting can be aggregated from this layer without touching
     the reader again.
  2. `<key>-g{max_gap_frames}-post.pkl` -- per-FRAGMENT number likelihoods,
     aggregated exactly as the engine serves them (`tracklet_likelihood`,
     weights = legibility * confidence, margin_tau = 2.0), plus the confident
     argmax list a `number_prior` is fitted from.

Serving conventions are the engine's (`reid_engine._jersey_likelihoods`), not
a new rule: mean-in-log-domain pooling, tau-2 tracklet abstention, per-run
number prior. The reader stack is parseq-jersey.ckpt + the legibility
ResNet34 (both research-licensed local artifacts).

Usage:
    uv run python -m matchlab_train.experiments.footpass_jersey \
        --keys game_18_H1 game_18_H2 ... --max-gap-frames 30
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from matchlab_core.reid.jersey import (
    crop_number_logprobs,
    tracklet_likelihood,
    uniform_prior,
)

from matchlab_train.datasets.footpass import COL, load_half
from matchlab_train.experiments import bootstrap_threads as bt
from matchlab_train.experiments.position_evidence import VAL_H5

APPEARANCE_DIR = Path("data/experiments/footpass-appearance")
JERSEY_DIR = Path("data/experiments/footpass-jersey")
MARGIN_TAU = 2.0  # engine default (reid_engine.Params.jersey_margin_tau)


def _saved_frames(key: str) -> list[int]:
    d = APPEARANCE_DIR / key / "img1"
    return sorted(int(p.stem) for p in d.glob("*.jpg"))


def read_crops(key: str, *, device: str = "cuda:0", batch_frames: int = 200) -> list[dict]:
    """Per-crop jersey reads for every saved frame of `key`, cached.

    A row is {frame, player_id, logprobs (3,11), legibility, confidence}.
    Crops with no locatable band (the reader's abstention path) simply have no
    row -- absence of evidence, not a zero vote.
    """
    JERSEY_DIR.mkdir(parents=True, exist_ok=True)
    cache = JERSEY_DIR / f"{key}-crops.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())

    import cv2
    from matchlab_core.crops import ScoredCrop
    from matchlab_core.ocr.parseq import JerseyReader, number_band

    half = load_half(VAL_H5, key)
    rows = half.rows
    frames = _saved_frames(key)
    frame_set = set(frames)
    sel = rows[np.isin(rows[:, COL.FRAME].astype(int), list(frame_set))]
    sel = sel[~np.isnan(sel[:, COL.ROI_X])]

    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for r in sel:
        by_frame.setdefault(int(r[COL.FRAME]), []).append(
            (int(r[COL.PLAYER_ID]), r[[COL.ROI_X, COL.ROI_Y, COL.ROI_W, COL.ROI_H]].astype(float))
        )

    reader = JerseyReader()
    reader.prepare(device=device)

    out: list[dict] = []
    img_dir = APPEARANCE_DIR / key / "img1"
    for start in range(0, len(frames), batch_frames):
        chunk = frames[start:start + batch_frames]
        crops: list[ScoredCrop] = []
        who: list[tuple[int, int]] = []
        for fr in chunk:
            boxes = by_frame.get(fr)
            if not boxes:
                continue
            img = cv2.imread(str(img_dir / f"{fr:06d}.jpg"))
            if img is None:
                continue
            h, w = img.shape[:2]
            for pid, (x, y, bw, bh) in boxes:
                x0, y0 = max(int(x), 0), max(int(y), 0)
                x1, y1 = min(int(x + bw), w), min(int(y + bh), h)
                if x1 - x0 < 4 or y1 - y0 < 8:
                    continue
                crops.append(ScoredCrop(
                    image=img[y0:y1, x0:x1].copy(), quality=1.0,
                    frame_idx=fr, box_height=float(y1 - y0), isolation_iou=0.0,
                ))
                who.append((fr, pid))
        if not crops:
            continue
        # JerseyReader.read() silently drops sub-floor / degenerate-band crops,
        # which destroys positional alignment with `who` (and at this crop
        # scale most crops drop). Reproduce its exact two stages with the
        # alignment kept: one batched legibility pass over every band, then
        # the decoder only on survivors -- same floor, same outputs.
        bands = [number_band(c.image) for c in crops]
        scores = reader._legibility.score(bands)
        for (fr, pid), band, score in zip(who, bands, scores):
            if score < reader.min_legibility or band.size == 0:
                continue
            probs, confidence = reader._char_probs(band)
            out.append({
                "frame": fr, "player_id": pid,
                "logprobs": crop_number_logprobs(probs).astype(np.float32),
                "legibility": float(score),
                "confidence": float(confidence),
            })
        print(f"  {key}: frames {start + len(chunk)}/{len(frames)}, reads {len(out)}",
              flush=True)

    cache.write_bytes(pickle.dumps(out))
    return out


def fragment_posteriors(key: str, *, max_gap_frames: int = 30) -> dict:
    """{frag_idx: likelihood} at bootstrap_threads' fragmentation, cached.

    Also returns the confident argmax numbers (`number_prior` inputs) so the
    consumer can fit the prior on exactly the engine's convention: this run's
    own non-abstaining reads.
    """
    cache = JERSEY_DIR / f"{key}-g{max_gap_frames}-post.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())

    crops = read_crops(key)
    bt.MAX_GAP_FRAMES = max_gap_frames
    frags, _first, _last, _app = bt.half_frames(key)

    owner: dict[tuple[int, int], int] = {}
    for fi, f in enumerate(frags):
        for fr in range(int(f.start), int(f.end) + 1):
            owner[(int(f.player_id), fr)] = fi

    by_frag: dict[int, list[dict]] = {}
    for c in crops:
        fi = owner.get((c["player_id"], c["frame"]))
        if fi is not None:
            by_frag.setdefault(fi, []).append(c)

    flat = uniform_prior()
    likelihood: dict[int, np.ndarray] = {}
    confident: list[int] = []
    for fi, cs in by_frag.items():
        logprobs = np.stack([c["logprobs"].astype(np.float64) for c in cs])
        weights = np.array([c["legibility"] * c["confidence"] for c in cs])
        lk = tracklet_likelihood(logprobs, weights, margin_tau=MARGIN_TAU)
        likelihood[fi] = lk
        if not np.allclose(lk, flat):
            confident.append(int(np.argmax(lk)))

    out = {"likelihood": likelihood, "confident_numbers": confident,
           "n_fragments": len(frags), "n_reads": len(crops)}
    cache.write_bytes(pickle.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", nargs="+", default=[
        f"{m}_{h}" for m in bt.MATCHES for h in ("H1", "H2")
    ])
    ap.add_argument("--max-gap-frames", type=int, default=30)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for key in args.keys:
        read_crops(key, device=args.device)
        post = fragment_posteriors(key, max_gap_frames=args.max_gap_frames)
        n_conf = len(post["confident_numbers"])
        n_lk = len(post["likelihood"])
        print(f"{key}: {post['n_reads']} reads -> {n_lk}/{post['n_fragments']} "
              f"fragments with evidence, {n_conf} confident", flush=True)


if __name__ == "__main__":
    main()
