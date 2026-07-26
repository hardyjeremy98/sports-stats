"""Clip sampler for the TAAD action head.

One sample is a `clip_length`-frame window around one labelled action, plus up to
`nb_tracklets` other players sharing the window, so the model must learn to attribute
an action to the right player rather than to the moment. M is therefore
`nb_tracklets + 1` (5 with the reference defaults), NOT `nb_tracklets`.

Class balance is the whole reason this class exists. Pass and drive are 89% of the
events; sampling uniformly would train a two-class model. Following the reference,
each epoch draws at most `max_samples_per_class` anchors per class and **resamples
every epoch**, so the rare classes are seen repeatedly with different negatives while
the common ones rotate through fresh examples. Measured on TRAIN: only pass (45,621),
drive (35,527), header (3,642), cross (2,156) and throw-in (1,730) can fill a 500
quota; block (1,269) and shot (1,097) can, but tackle has just 285 examples total and
can never fill one.

Deviations from the reference, both recorded rather than hidden:

* `albumentations` is not packaged here, so its affine/rotate/random-scale/crop
  augmentations are replaced with horizontal flip plus brightness/contrast jitter,
  applied consistently to the frames and the boxes. The ROI-expansion jitter
  (coeff 1.125 +/- 0.125), which is the augmentation that actually regularises the
  pooling geometry, is kept exactly.
* Clips are read with OpenCV rather than `decord` (see `footpass_video`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matchlab_core.pcbas.schema import (
    BACKGROUND,
    CLASS_NAMES,
    CLS,
    FRAME,
    N_CLASSES,
    N_COLUMNS_LABELLED,
    PLAYER_ID,
    ROI_HEIGHT,
    ROI_SCALE_X,
    ROI_SCALE_Y,
    ROI_WIDTH,
    ROI_X,
    ROI_Y,
)

from matchlab_train.datasets.footpass_pcbas import list_halves, load_half
from matchlab_train.datasets.footpass_video import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    MatchVideo,
    match_video_path,
)

CLIP_LENGTH = 50
NB_TRACKLETS = 4
MAX_SAMPLES_PER_CLASS = 500
LABEL_DILATION = 1
NORM_MEAN, NORM_STD = 0.45, 0.225

# Box expansion around the annotated player before pooling. The reference jitters
# this by +/-0.125 during training; it is the augmentation that regularises pooling
# geometry, so it is kept exactly.
ROI_COEFF = 1.125
ROI_COEFF_JITTER = 0.125
MIN_BOX_W, MIN_BOX_H = 2, 2
# Where a player has no usable box. The mask zeroes these features, so the
# coordinates only need to be in-bounds and non-degenerate.
DUMMY_BOX = (100.0, 100.0, 200.0, 200.0)


@dataclass(frozen=True)
class ClipAnchor:
    """One labelled action, and the frame bounds of the half it lives in."""

    key: str
    frame_idx: int
    player_id: int
    class_id: int
    min_frame: int
    max_frame: int


def build_anchors(h5_path: str | Path, *, require_bbox: bool = True) -> list[ClipAnchor]:
    """Non-background rows become clip anchors. ~3 min on TRAIN.

    `require_bbox` defaults True, matching the reference and for a substantive
    reason: an anchor whose player has no bounding box is fully masked, so it
    contributes exactly zero loss AND consumes one of that class's places in the
    epoch's balanced quota. On the rare classes that is ruinous -- VAL has 26 tackles
    but only 16 with a box, so 38% of tackle anchors would teach nothing.

    Verified against the reference's own shipped `TAAD_sample_list.json`: with this
    filter our per-class VAL anchor counts match it exactly on all 8 classes
    (5,008 total). Without it we get 6,070, the full GT count. That agreement is a
    free control on the training anchor set -- see
    `test_anchors_match_the_reference_sample_list`.

    The no-bbox events are NOT lost from the problem: they remain in the ground truth
    and are exactly the events the sequence stage exists to recover.
    """
    anchors: list[ClipAnchor] = []
    for key in list_halves(h5_path):
        arr = load_half(h5_path, key)
        if arr.shape[1] < N_COLUMNS_LABELLED:
            continue
        frames = arr[:, FRAME]
        lo, hi = int(np.nanmin(frames)), int(np.nanmax(frames))
        is_event = ~np.isnan(arr[:, CLS]) & (arr[:, CLS] != BACKGROUND)
        if require_bbox:
            is_event &= ~np.isnan(arr[:, ROI_X])
        for row in arr[is_event]:
            anchors.append(
                ClipAnchor(
                    key=key,
                    frame_idx=int(row[FRAME]),
                    player_id=int(row[PLAYER_ID]),
                    class_id=int(row[CLS]),
                    min_frame=lo,
                    max_frame=hi,
                )
            )
    return anchors


def save_anchors(anchors: list[ClipAnchor], path: str | Path) -> None:
    Path(path).write_text(json.dumps([a.__dict__ for a in anchors]))


def load_anchors(path: str | Path) -> list[ClipAnchor]:
    return [ClipAnchor(**d) for d in json.loads(Path(path).read_text())]


def scale_box(
    roi_x: float, roi_y: float, roi_w: float, roi_h: float, coeff: float
) -> tuple[float, float, float, float]:
    """Full-HD box -> expanded, clamped 352x640 box, exactly as the reference does.

    Note the asymmetry, reproduced deliberately: the left/top edges move out by
    `(coeff-1)/2` of the box extent but the right/bottom edges are placed at
    `origin + coeff * extent`, so the box grows by `coeff - 1 + (coeff-1)/2` rather
    than symmetrically. Matching the reference matters more than tidiness here --
    the pretrained pooling geometry was learned with this framing.
    """
    tlx = max(min(1920.0, roi_x - (coeff - 1.0) * roi_w / 2), 0.0) / ROI_SCALE_X
    tly = max(min(1080.0, roi_y - (coeff - 1.0) * roi_h / 2), 0.0) / ROI_SCALE_Y
    brx = max(min(1920.0, roi_x + coeff * roi_w), 0.0) / ROI_SCALE_X
    bry = max(min(1080.0, roi_y + coeff * roi_h), 0.0) / ROI_SCALE_Y
    tlx = max(0.0, min(EXPECTED_WIDTH - 1.0, tlx))
    tly = max(0.0, min(EXPECTED_HEIGHT - 1.0, tly))
    brx = max(0.0, min(float(EXPECTED_WIDTH), brx))
    bry = max(0.0, min(float(EXPECTED_HEIGHT), bry))
    return tlx, tly, brx, bry


def dilate_labels(labels: np.ndarray, dilation: int) -> np.ndarray:
    """Widen each event label by +/-`dilation` frames along the time axis.

    The sharp labels stay available for metric computation; the dilated ones are what
    the loss uses, because a single-frame target against a 50-frame clip is too
    sparse to learn from.
    """
    if dilation <= 0:
        return labels.copy()
    out = labels.copy()
    m, t = labels.shape
    for i in range(m):
        for idx in np.flatnonzero(labels[i]):
            out[i, max(0, idx - dilation) : min(t, idx + dilation + 1)] = labels[i, idx]
    return out


def balanced_sample(
    anchors: list[ClipAnchor], max_per_class: int, rng: np.random.Generator
) -> list[ClipAnchor]:
    """At most `max_per_class` anchors per action class, shuffled.

    Classes with fewer than `max_per_class` anchors contribute all of them -- they are
    never oversampled, so the epoch is balanced only up to what the data allows.
    """
    by_class: dict[int, list[ClipAnchor]] = {}
    for a in anchors:
        by_class.setdefault(a.class_id, []).append(a)
    chosen: list[ClipAnchor] = []
    for class_id in range(1, N_CLASSES):
        pool = by_class.get(class_id, [])
        if not pool:
            continue
        take = min(len(pool), max_per_class)
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[i] for i in idx)
    rng.shuffle(chosen)  # type: ignore[arg-type]
    return chosen


class FootpassClipDataset:
    """`torch.utils.data.Dataset` over class-balanced action clips.

    Returns `(video, rois, masks, sharp_labels, dilated_labels)`:
    `(3,T,352,640) float32`, `(M,T,5)`, `(M,T)`, `(M,T) long`, `(M,T) long`.

    HDF5 and video handles are opened lazily per worker -- neither `h5py.File` nor
    `cv2.VideoCapture` survives a fork, so opening them in `__init__` corrupts reads
    the moment `num_workers > 0`.
    """

    def __init__(
        self,
        h5_path: str | Path,
        video_root: str | Path,
        anchors: list[ClipAnchor],
        *,
        clip_length: int = CLIP_LENGTH,
        nb_tracklets: int = NB_TRACKLETS,
        max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
        label_dilation: int = LABEL_DILATION,
        train: bool = True,
        seed: int = 345,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.video_root = Path(video_root)
        self.anchors = anchors
        self.clip_length = clip_length
        self.nb_tracklets = nb_tracklets
        self.max_samples_per_class = max_samples_per_class
        self.label_dilation = label_dilation
        self.train = train
        self.seed = seed
        self._epoch = 0
        self._rng = np.random.default_rng(seed)
        self._h5 = None
        self._videos: dict[str, MatchVideo] = {}
        self._half_cache: dict[str, np.ndarray] = {}
        self.resample()

    @property
    def n_players(self) -> int:
        """M -- the annotated player plus `nb_tracklets` distractors."""
        return self.nb_tracklets + 1

    def resample(self) -> None:
        """Redraw the epoch's anchors. Must be called between epochs."""
        rng = (
            np.random.default_rng(self.seed + self._epoch)
            if self.train
            else np.random.default_rng(self.seed)
        )
        self.sampled = balanced_sample(self.anchors, self.max_samples_per_class, rng)
        self._epoch += 1

    def __len__(self) -> int:
        return len(self.sampled)

    # --- lazy per-worker handles ---------------------------------------------------

    def _half(self, key: str) -> np.ndarray:
        if key not in self._half_cache:
            # One half is ~100 MB as float64; keep at most a few per worker.
            if len(self._half_cache) > 3:
                self._half_cache.clear()
            self._half_cache[key] = load_half(self.h5_path, key)
        return self._half_cache[key]

    def _video(self, key: str) -> MatchVideo:
        path = match_video_path(self.video_root, key)
        if path.name not in self._videos:
            if len(self._videos) > 2:
                for v in self._videos.values():
                    v.close()
                self._videos.clear()
            self._videos[path.name] = MatchVideo(path)
        return self._videos[path.name]

    # --- sampling ------------------------------------------------------------------

    def _clip_start(self, anchor: ClipAnchor, rng: np.random.Generator) -> int:
        """A start frame that keeps the anchor event inside the clip.

        The reference's window places the event between 10 frames after the start and
        10 frames before the midpoint-derived bound, so the action never sits exactly
        at the clip centre -- otherwise the model could learn "the action is always at
        t=25" instead of looking at the player.
        """
        half = self.clip_length // 2
        lo = max(anchor.min_frame, anchor.frame_idx - half - 10)
        hi = min(anchor.max_frame - self.clip_length, anchor.frame_idx - (half - 10))
        if hi <= lo:
            return max(anchor.min_frame, min(lo, anchor.max_frame - self.clip_length))
        return int(rng.integers(lo, hi))

    def _select_players(
        self, rows: np.ndarray, anchor: ClipAnchor, rng: np.random.Generator
    ) -> list[int]:
        """The anchor player plus up to `nb_tracklets` others visible in the window.

        Padded with a player id that cannot exist, so M is constant across the batch;
        those slots are fully masked.
        """
        visible = rows[(rows[:, PLAYER_ID] != anchor.player_id) & ~np.isnan(rows[:, ROI_X])]
        others = np.unique(visible[:, PLAYER_ID]) if len(visible) else np.array([])
        take = min(len(others), self.nb_tracklets)
        chosen = list(rng.choice(others, take, replace=False)) if take else []
        chosen.append(anchor.player_id)
        while len(chosen) < self.n_players:
            chosen.append(-1)  # never a real player id; fully masked
        return [int(p) for p in chosen]

    def _build_tracks(
        self, rows: np.ndarray, players: list[int], start: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = self.clip_length
        rois = np.zeros((len(players), t, 5), dtype=np.float32)
        masks = np.zeros((len(players), t), dtype=np.float32)
        labels = np.zeros((len(players), t), dtype=np.int64)

        for m, player in enumerate(players):
            coeff = (
                ROI_COEFF + ROI_COEFF_JITTER * (2.0 * rng.random() - 1.0)
                if self.train
                else ROI_COEFF
            )
            player_rows = rows[rows[:, PLAYER_ID] == player]
            by_frame = {int(r[FRAME]): r for r in player_rows}
            for ti in range(t):
                rois[m, ti, 0] = ti
                row = by_frame.get(start + ti)
                if row is None or np.isnan(row[ROI_X]):
                    rois[m, ti, 1:] = DUMMY_BOX
                    continue
                x1, y1, x2, y2 = scale_box(
                    row[ROI_X], row[ROI_Y], row[ROI_WIDTH], row[ROI_HEIGHT], coeff
                )
                if (x2 - x1) < MIN_BOX_W or (y2 - y1) < MIN_BOX_H:
                    rois[m, ti, 1:] = DUMMY_BOX
                    continue
                rois[m, ti, 1:] = (x1, y1, x2, y2)
                masks[m, ti] = 1.0
                cls = row[CLS]
                labels[m, ti] = 0 if np.isnan(cls) else int(cls)
        return rois, masks, labels

    def _augment(
        self, clip: np.ndarray, rois: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal flip and brightness/contrast jitter, boxes kept consistent.

        Flipping is safe for this stage: the action head predicts a class per player,
        never a side or a role slot, so mirroring the pitch does not relabel anything.
        It would NOT be safe for the sequence stage, whose slots encode
        `left_to_right`.
        """
        if rng.random() < 0.5:
            clip = clip[:, :, ::-1]
            x1 = rois[:, :, 1].copy()
            x2 = rois[:, :, 3].copy()
            rois[:, :, 1] = EXPECTED_WIDTH - x2
            rois[:, :, 3] = EXPECTED_WIDTH - x1
        brightness = 0.8 + 0.4 * rng.random()
        contrast = 0.8 + 0.4 * rng.random()
        clip = np.clip((clip.astype(np.float32) - 128.0) * contrast + 128.0 * brightness, 0, 255)
        return clip, rois

    def __getitem__(self, index: int):
        import torch

        anchor = self.sampled[index]
        rng = (
            np.random.default_rng()
            if self.train
            else np.random.default_rng(self.seed + index)
        )

        start = self._clip_start(anchor, rng)
        clip = self._video(anchor.key).read_clip(start, self.clip_length)

        arr = self._half(anchor.key)
        window = arr[
            (arr[:, FRAME] >= start) & (arr[:, FRAME] < start + self.clip_length)
        ]
        players = self._select_players(window, anchor, rng)
        rois, masks, labels = self._build_tracks(window, players, start, rng)

        if self.train:
            clip, rois = self._augment(clip, rois, rng)

        video = (clip.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
        dilated = dilate_labels(labels, self.label_dilation)

        return (
            torch.from_numpy(np.ascontiguousarray(video)).permute(3, 0, 1, 2),
            torch.from_numpy(rois),
            torch.from_numpy(masks),
            torch.from_numpy(labels),
            torch.from_numpy(dilated),
        )

    def close(self) -> None:
        for v in self._videos.values():
            v.close()
        self._videos.clear()


def class_histogram(anchors: list[ClipAnchor]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for a in anchors:
        name = CLASS_NAMES[a.class_id]
        hist[name] = hist.get(name, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))
