"""Per-frame PRTreID embeddings, keyed by (player_id, frame).

Appearance has entered every prior experiment on this stack as ONE MEAN VECTOR
per fragment: `footpass_appearance.aggregate` mean-pools the per-frame
embeddings, `bootstrap_threads.remap_appearance` mean-pools those again across
fragmentations, and `ThreadState.prototype` means a third time. A mean is the
worst possible summary of a multimodal set, and a fragment spanning a 30-frame
observability gap on a panning broadcast is multimodal by construction -- pose,
scale, occlusion and illumination all change inside it.

So "appearance is exhausted" has only ever been measured as "the mean of this
frozen extractor's features is exhausted". The per-frame embeddings the bridge
wrote were never deleted (`<key>/feat/*.pkl`, ~363 MB per half), so testing the
stronger claim needs no re-extraction -- only a join back to identity.

## Keyed by (player_id, frame), deliberately

`fragment_embeddings.pkl` is keyed by POSITION in a fragment list built at one
particular `max_gap_frames`, which is why reading it at any other setting
silently hands each fragment some other span's embedding. That is not
hypothetical: it withdrew every figure published on 2026-07-30, on the
highest-weighted channel, and nothing detected it because nothing checked.
`check_appearance_alignment`'s own docstring names the durable fix as keying by
identity rather than by position. This module is that fix.

Identity is recovered by joining the bridge's per-frame outputs back to the
tactical rows ON THE BOX, not by re-deriving the writer's ordering. Re-deriving
an ordering is a parallel implementation of the thing that went wrong last time.

Pure except for reading the extraction outputs. No config.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matchlab_train.datasets.footpass import COL, load_half
from matchlab_train.experiments.multi_input import APPEARANCE_DIR, VAL_H5

CACHE = Path("data/experiments/frame-embeddings")

#: Boxes are written to det.txt with two decimals, so the join tolerance only
#: has to survive that rounding.
BOX_TOL = 0.01


@dataclass
class FrameEmbeddings:
    """Per-(player, frame) unit embeddings and their extractor visibility.

    `visibility` is PRTreID's own part-visibility score, which the bridge has
    always written and nothing has ever read. It is the quality signal ADR 003
    asks for and the only one here that is not a frame-count proxy.
    """

    player: np.ndarray        # (n,) int
    frame: np.ndarray         # (n,) int
    emb: np.ndarray           # (n, d) float32, unit-normalised
    visibility: np.ndarray    # (n,) float

    def __len__(self) -> int:
        return len(self.frame)

    def index(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """player_id -> (frames sorted, row indices in that order).

        Sorted so a fragment's rows are one searchsorted slice rather than a
        full-array mask per fragment -- the naive form is 2,500 scans of a
        1.6M-row array per half, which is what made the first attempt at this
        die under memory pressure.
        """
        out = {}
        for p in np.unique(self.player):
            sel = np.flatnonzero(self.player == p)
            order = sel[np.argsort(self.frame[sel], kind="mergesort")]
            out[int(p)] = (self.frame[order], order)
        return out

    def for_span(self, idx, player_id: int, start: int, end: int) -> np.ndarray:
        """Row indices of this player's samples inside [start, end]."""
        got = idx.get(int(player_id))
        if got is None:
            return np.zeros(0, dtype=np.int64)
        frames, rows = got
        a = int(np.searchsorted(frames, start, "left"))
        b = int(np.searchsorted(frames, end, "right"))
        return rows[a:b]


def _det_rows(key: str) -> np.ndarray:
    """det.txt as (frame0, x, y, w, h), in file order.

    File order is what the bridge's per-frame outputs are ordered by, so row i
    of frame f's pickle corresponds to the i-th det.txt line for that frame.
    """
    p = APPEARANCE_DIR / key / "det" / "det.txt"
    raw = np.loadtxt(p, delimiter=",", usecols=(0, 2, 3, 4, 5), dtype=np.float64)
    raw[:, 0] -= 1.0  # det.txt is 1-based; tactical rows are 0-based
    return raw


def build(key: str) -> FrameEmbeddings:
    """Join the bridge's per-frame embeddings back to (player_id, frame).

    The join is on the BOX. `write_det_file` derived its line order from a dict
    of visible rows per frame; reproducing that ordering here would be a second
    implementation of the step that has already failed once, and it would fail
    silently. Matching the box that was actually written cannot.
    """
    dets = _det_rows(key)
    half = load_half(VAL_H5, key)
    rows = half.rows
    vis = rows[~np.isnan(rows[:, COL.ROI_X])]

    # (frame, x, y) -> player, with w/h checked, so the lookup is O(1) per det.
    lookup: dict[tuple[int, int, int], list[tuple[np.ndarray, int]]] = {}
    for r in vis:
        f = int(r[COL.FRAME])
        box = r[[COL.ROI_X, COL.ROI_Y, COL.ROI_W, COL.ROI_H]].astype(float)
        lookup.setdefault((f, int(round(box[0])), int(round(box[1]))), []).append(
            (box, int(r[COL.PLAYER_ID]))
        )

    by_frame: dict[int, list[int]] = {}
    for i, d in enumerate(dets):
        by_frame.setdefault(int(d[0]), []).append(i)

    players, frames, embs, viss = [], [], [], []
    unmatched = 0
    feat_dir = APPEARANCE_DIR / key / "feat"
    for pkl in sorted(feat_dir.glob("*.pkl")):
        # The bridge names its outputs by the 0-based frame index it was handed,
        # while det.txt numbers frames from 1 -- so the pickle stem is ALREADY
        # the tactical frame and must not be shifted again. Getting this wrong
        # joins nothing at all, which is why the guard below refuses an empty
        # result rather than returning one.
        frame0 = int(pkl.stem)
        order = by_frame.get(frame0)
        if not order:
            continue
        payload = pickle.loads(pkl.read_bytes())
        if not payload:
            continue
        for o, det in enumerate(payload):
            if o >= len(order):
                break
            d = dets[order[o]]
            cands = lookup.get((frame0, int(round(d[1])), int(round(d[2]))), ())
            pid = None
            for box, p in cands:
                if abs(box[2] - d[3]) <= BOX_TOL and abs(box[3] - d[4]) <= BOX_TOL:
                    pid = p
                    break
            if pid is None:
                unmatched += 1
                continue
            e = np.asarray(det["appearance_embeddings"], dtype=np.float32).ravel()
            n = float(np.linalg.norm(e))
            if n <= 1e-9:
                continue
            players.append(pid)
            frames.append(frame0)
            embs.append(e / n)
            v = det.get("appearance_visibility")
            viss.append(float(np.mean(np.asarray(v, dtype=float))) if v is not None
                        else 1.0)

    if not players:
        raise AssertionError(
            f"{key}: the det.txt/feat join produced NO rows. An empty join is "
            "the signature of a frame-numbering offset, and returning it would "
            "silently disable the appearance channel rather than fail."
        )
    if unmatched > 0.02 * max(len(players), 1):
        raise AssertionError(
            f"{key}: {unmatched} detections could not be matched back to a "
            f"tactical row against {len(players)} matched -- the det.txt/feat "
            "join is not the identity map it is assumed to be."
        )
    return FrameEmbeddings(
        player=np.array(players, dtype=np.int64),
        frame=np.array(frames, dtype=np.int64),
        emb=np.stack(embs) if embs else np.zeros((0, 256), dtype=np.float32),
        visibility=np.array(viss, dtype=np.float64),
    )


#: Most recent per-frame samples retained per THREAD side. A thread grows to
#: hundreds of samples over a half while the query side has a median of 7, and
#: an unbounded set would make the statistic a function of thread age rather
#: than of appearance. Reported alongside every result that uses it.
THREAD_SET_CAP = 64

#: Statistics computed over the cross-frame cosine matrix. `proto` is the
#: incumbent -- cosine of the two mean vectors -- included here so the arms and
#: the control are computed by the SAME code over the SAME sets, which is what
#: makes the comparison about the statistic rather than about the plumbing.
SET_STATS = ("proto", "top1", "top3", "top5", "median", "q90", "mutual_nn")


def set_statistics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Summaries of the cross-frame cosine matrix between two appearance sets.

    The incumbent squeezes this whole matrix to the cosine of its two mean
    vectors. That is the reduction under test: a mean discards the mode
    structure, and it is exactly the wrong summary when one side contains two
    visually distinct regimes -- which a fragment spanning an occlusion does.

    Every statistic is a similarity (higher = more alike), so each one can be
    calibrated by the ordinary `LLRCalibrator` and dropped into the fused sum in
    place of `body` with nothing else changed.
    """
    if not len(a) or not len(b):
        return dict.fromkeys(SET_STATS, float("nan"))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = a @ b.T
    flat = c.ravel()

    ma = a.mean(axis=0)
    mb = b.mean(axis=0)
    ma = ma / max(float(np.linalg.norm(ma)), 1e-9)
    mb = mb / max(float(np.linalg.norm(mb)), 1e-9)

    def topk(k: int) -> float:
        k = min(k, flat.size)
        return float(np.mean(np.partition(flat, -k)[-k:]))

    # Mutual nearest neighbours: the fraction of a's frames whose best match in
    # b picks them back. A set-level agreement statistic with no scalar analogue
    # -- it asks whether the two sets AGREE about each other, not merely how
    # close their averages are.
    best_ab = np.argmax(c, axis=1)
    best_ba = np.argmax(c, axis=0)
    mutual = float(np.mean(best_ba[best_ab] == np.arange(len(a))))

    return {
        "proto": float(ma @ mb),
        "top1": topk(1),
        "top3": topk(3),
        "top5": topk(5),
        "median": float(np.median(flat)),
        "q90": float(np.quantile(flat, 0.9)),
        "mutual_nn": mutual,
    }


def load(key: str) -> FrameEmbeddings:
    """Cached `build`. Keyed by half only -- the content is fragmentation-free,
    which is the entire point of keying on identity instead of position."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{key}.npz"
    if p.exists():
        z = np.load(p)
        return FrameEmbeddings(
            player=z["player"], frame=z["frame"],
            emb=z["emb"], visibility=z["visibility"],
        )
    fe = build(key)
    np.savez_compressed(
        p, player=fe.player, frame=fe.frame, emb=fe.emb, visibility=fe.visibility
    )
    return fe
