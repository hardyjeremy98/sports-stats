"""Camera motion across calibration blackouts: chained shape, direct correction.

When pitch registration fails for a stretch, camera motion is still recoverable
from the imagery. There are two ways to get it and they fail in OPPOSITE ways —
measured on SNMOT-122's 73-frame (3 s) blackout:

======================  ==========================  =========================
                        accuracy (L/R disagreement)  smoothness (median accel)
======================  ==========================  =========================
chained frame-to-frame  ~5.4 m  (6.1 m of drift)     8.8 cm
direct to the anchors   0.5-1.6 m                    83.0 cm  (9.4x worse)
======================  ==========================  =========================

Chaining is smooth because consecutive frames share almost their whole
composition, so errors are correlated — but that same sharing accumulates drift.
Direct registration is unbiased because there is only ever one homography between
a frame and its anchor — but each frame is registered INDEPENDENTLY, so its error
is uncorrelated with its neighbours' and the camera appears to shake. Projecting
players through a shaking camera manufactures exactly the implausible speeds this
whole exercise exists to remove, which is why direct registration alone scored
WORSE end-to-end (2.57% vs 1.27%) despite being the better estimator.

So use both. The chain supplies the SHAPE of the motion (real high-frequency
camera movement); direct registration supplies an unbiased measurement of where
the chain has drifted to. Their difference is drift plus noise — and drift is
smooth while noise is not, so a low-order fit separates them. Correcting the
chain by that fitted drift keeps the chain's smoothness and removes its bias.

That is a pose-graph optimisation in cheap form: relative constraints from the
chain, absolute constraints from direct registration, a smoothness prior — solved
by curve-fitting the residual instead of a full nonlinear solve.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "DriftCorrectedMotion",
    "gap_anchor_pairs",
    "register_pairs",
]

# Enough features to survive a multi-second baseline; the matcher keeps the best.
_ORB_FEATURES = 4000
_MAX_MATCHES = 800
_MIN_MATCHES = 20
_RANSAC_PX = 3.0
# Drift is slow, so it is sampled rather than measured at every frame — direct
# registration is the expensive part and the fit only needs enough support.
_REGISTER_STRIDE = 5
# Low order on purpose: high enough for accumulating drift, too low to chase the
# per-frame registration noise the correction exists to reject.
_DRIFT_POLY_ORDER = 2
_MIN_FIT_POINTS = 4
# A correction larger than this is not drift being removed, it is a bad direct
# registration being trusted. Long gaps over repetitive pitch texture do produce
# false ORB matches that pass RANSAC; unguarded they took SNMOT-121's left/right
# disagreement from 1.51 m to 13.75 m.
_MAX_CORRECTION_CM = 800.0
# Residuals this far from the robust centre are dropped before fitting.
_FIT_OUTLIER_CM = 500.0


def _correction_grid(frame_size: tuple[int, int]) -> np.ndarray:
    w, h = float(frame_size[0]), float(frame_size[1])
    xs = [0.1 * w, 0.5 * w, 0.9 * w]
    ys = [0.1 * h, 0.5 * h, 0.9 * h]
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def _apply(h: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    v = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(h, dtype=np.float64).T
    w = v[:, 2]
    if not np.all(np.isfinite(v)) or np.any(np.abs(w) < 1e-9):
        return None
    return v[:, :2] / w[:, None]


class DriftCorrectedMotion:
    """Image->image motion for blackout frames: chained, de-drifted by direct
    registration. Satisfies the smoother's ``CameraMotionLike`` protocol.

    Reports ``None`` when it has no basis for a span, so the smoother falls back
    to a straight line. It never returns identity as a stand-in: identity asserts
    the camera did not move, which freezes the fill on the anchor's calibration —
    measured to take SNMOT-121 from 0.33% to 3.63%.
    """

    def __init__(
        self,
        chained: object | None,
        registrations: dict[tuple[int, int], np.ndarray] | None = None,
        frame_size: tuple[int, int] = (1920, 1080),
    ) -> None:
        self._chained = chained
        self._reg = {k: np.asarray(v, dtype=np.float64) for k, v in (registrations or {}).items()}
        self._grid = _correction_grid(frame_size)
        self._fits: dict[int, np.ndarray | None] = {}
        self._spans: dict[int, tuple[float, float] | None] = {}

    def _chain(self, frame: int, anchor: int) -> np.ndarray | None:
        if self._chained is None:
            return None
        h = self._chained.homography(frame, anchor)  # type: ignore[attr-defined]
        return None if h is None else np.asarray(h, dtype=np.float64)

    def _drift_fit(self, anchor: int) -> np.ndarray | None:
        """Per-grid-point polynomial coefficients describing how far the chain has
        drifted from direct registration, as a function of frame index."""
        if anchor in self._fits:
            return self._fits[anchor]
        samples: list[tuple[int, np.ndarray]] = []
        for (frame, a), direct in self._reg.items():
            if a != anchor:
                continue
            chain = self._chain(frame, anchor)
            if chain is None:
                continue
            moved_direct = _apply(direct, self._grid)
            moved_chain = _apply(chain, self._grid)
            if moved_direct is None or moved_chain is None:
                continue
            samples.append((frame, moved_direct - moved_chain))

        fit = None
        span: tuple[float, float] | None = None
        if len(samples) >= _MIN_FIT_POINTS:
            samples.sort(key=lambda s: s[0])
            t = np.array([s[0] for s in samples], dtype=np.float64)
            resid = np.stack([s[1] for s in samples], axis=0)  # (S, 9, 2)
            # Drop samples whose residual is far from the robust centre: a false
            # match produces a wild residual that would drag a least-squares fit.
            magnitude = np.median(np.linalg.norm(resid, axis=2), axis=1)
            keep = np.abs(magnitude - np.median(magnitude)) <= _FIT_OUTLIER_CM
            if keep.sum() >= _MIN_FIT_POINTS:
                t, resid = t[keep], resid[keep]
            if len(t) >= _MIN_FIT_POINTS:
                order = min(_DRIFT_POLY_ORDER, len(t) - 1)
                fit = np.stack(
                    [
                        [np.polyfit(t, resid[:, k, c], order) for c in range(2)]
                        for k in range(resid.shape[1])
                    ],
                    axis=0,
                )  # (9, 2, order+1)
                span = (float(t.min()), float(t.max()))
        self._fits[anchor] = fit
        self._spans[anchor] = span
        return fit

    def variants(self) -> list[object]:
        """Both bridging models, for the smoother to choose between per gap.

        Drift correction is a large win where blackouts are long but registrable
        (SNMOT-117 8.28 m -> 1.16 m, SNMOT-122 5.26 m -> 0.85 m) and a loss where
        they are longest and the footage is repetitive (SNMOT-121 1.51 m ->
        3.44 m even with guards). Neither dominates, so neither is hardcoded.
        """
        return [self, _ChainOnly(self._chained)] if self._chained is not None else [self]

    def homography(self, from_frame: int, to_frame: int) -> np.ndarray | None:
        if from_frame == to_frame:
            return np.eye(3)
        chain = self._chain(from_frame, to_frame)
        if chain is None:
            return None
        fit = self._drift_fit(to_frame)
        if fit is None:
            return chain  # no correction available: smooth but drifting
        span = self._spans.get(to_frame)
        if span is None or not (span[0] <= from_frame <= span[1]):
            # Never EXTRAPOLATE the drift curve. A degree-2 fit leaves the
            # plausible range fast, and outside its support the "correction" is
            # invention rather than measurement.
            return chain
        moved = _apply(chain, self._grid)
        if moved is None:
            return chain
        correction = np.stack(
            [[np.polyval(fit[k, c], float(from_frame)) for c in range(2)] for k in range(len(moved))],
            axis=0,
        )
        if float(np.max(np.linalg.norm(correction, axis=1))) > _MAX_CORRECTION_CM:
            return chain  # implausible correction: trust the smooth chain instead
        corrected, _ = cv2.findHomography(self._grid, moved + correction, 0)
        if corrected is None or not np.all(np.isfinite(corrected)):
            return chain
        return corrected


class _ChainOnly:
    """The uncorrected chain, as a selectable candidate."""

    def __init__(self, chained: object) -> None:
        self._chained = chained

    def homography(self, from_frame: int, to_frame: int) -> np.ndarray | None:
        h = self._chained.homography(from_frame, to_frame)  # type: ignore[attr-defined]
        return None if h is None else np.asarray(h, dtype=np.float64)


def gap_anchor_pairs(
    frame_idxs: Sequence[int], has_estimate: Sequence[bool], stride: int = _REGISTER_STRIDE
) -> list[tuple[int, int]]:
    """(blackout_frame, anchor_frame) pairs to register directly, both anchors.

    Sampled every `stride` frames: the drift being measured is slow, and direct
    registration is the expensive step.
    """
    n = len(frame_idxs)
    prev_ok: list[int | None] = [None] * n
    next_ok: list[int | None] = [None] * n
    last = None
    for i in range(n):
        prev_ok[i] = last
        if has_estimate[i]:
            last = i
    last = None
    for i in range(n - 1, -1, -1):
        next_ok[i] = last
        if has_estimate[i]:
            last = i

    pairs: list[tuple[int, int]] = []
    run_pos = 0
    for i in range(n):
        if has_estimate[i]:
            run_pos = 0
            continue
        take = run_pos % max(stride, 1) == 0
        run_pos += 1
        if not take:
            continue
        for j in (prev_ok[i], next_ok[i]):
            if j is not None:
                pairs.append((frame_idxs[i], frame_idxs[j]))
    return pairs


def register_pairs(
    gray_by_frame: dict[int, np.ndarray], pairs: Iterable[tuple[int, int]]
) -> dict[tuple[int, int], np.ndarray]:
    """Directly register each (from, to) pair with ORB + RANSAC, which tolerates
    a multi-second baseline far better than the sparse optical flow used for
    frame-to-frame chaining. Pairs that fail to match are omitted, never filled
    with identity."""
    orb = cv2.ORB_create(_ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cache: dict[int, tuple] = {}

    def features(frame: int):
        if frame not in cache:
            cache[frame] = orb.detectAndCompute(gray_by_frame[frame], None)
        return cache[frame]

    out: dict[tuple[int, int], np.ndarray] = {}
    for src, dst in pairs:
        if src not in gray_by_frame or dst not in gray_by_frame:
            continue
        k1, d1 = features(src)
        k2, d2 = features(dst)
        if d1 is None or d2 is None or len(k1) < _MIN_MATCHES or len(k2) < _MIN_MATCHES:
            continue
        matches = sorted(matcher.match(d1, d2), key=lambda m: m.distance)[:_MAX_MATCHES]
        if len(matches) < _MIN_MATCHES:
            continue
        p1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        p2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        h, mask = cv2.findHomography(p1, p2, cv2.RANSAC, _RANSAC_PX)
        if h is None or not np.all(np.isfinite(h)):
            continue
        if mask is not None and int(mask.sum()) < _MIN_MATCHES:
            continue
        out[(src, dst)] = h
    return out
