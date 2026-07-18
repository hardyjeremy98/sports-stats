"""Offline TDLP association loop, re-implemented in the PitchLab idiom.

This is the ``vendor-and-extend`` counterpart to the vendored TDLP *head*
(`pitchlab_core._vendor.tdlp`): the learned nn.Module is vendored verbatim
(MIT), but the tracking *loop* — which upstream rides on the third-party
``motrack`` library — is re-implemented here so no heavy tracker dependency
enters the tree, and so our purity policies (SPO-43, terminate-over-force) can
hook the assignment margin directly.

Faithful port of TDLP ``tracker/online.py`` (`_convert_data` / `_association` /
`_track`): a causal frame-by-frame loop with a fixed memory window
(``remember_threshold``). Each frame, every live tracklet's recent history is
laid into a ``(N, T, D)`` observed tensor (most-recent frame at the tail — the
head uses a reversed positional encoding that pins the last slot), the current
detections form the ``(N, D)`` unobserved tensor, the head scores every
track×detection pair, and a Hungarian assignment (SciPy) links them under a
similarity gate. Unmatched tracks age out to LOST/DELETED; unmatched
high-confidence detections spawn new tracklets.

The head is offline-capable (whole-clip memory) but the association itself is
causal-with-memory, which matches the offline upload-and-process product
(ADR 002); true whole-clip re-linking is the downstream offline associator's
job and is left untouched.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pitchlab_core.schemas import DetectionClass, Tracklet, TrackletFrame
from pitchlab_core.schemas.geometry import Box

# A detection's per-frame feature payload. Keys present depend on the enabled
# modalities; values are the raw (pre-tensor) feature arrays plus the source
# pixel box (kept so we can emit TrackletFrame boxes without re-deriving them).
ObjectData = dict


class TrackletState(enum.Enum):
    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class _FrameObs:
    frame_index: int
    data: ObjectData


@dataclass
class _Tracklet:
    """Mutable per-track state during the loop (mirrors motrack.Tracklet, but
    only the fields this loop needs)."""

    id: int
    max_history: int
    state: TrackletState
    frame_index: int  # last-updated source frame index
    history: deque = field(default_factory=deque)  # bounded model window (last T)
    full_history: list = field(default_factory=list)  # unbounded, for output
    total_matches: int = 0
    age: int = 0

    def update(self, frame_index: int, data: ObjectData, state: TrackletState) -> None:
        obs = _FrameObs(frame_index, data)
        self.history.append(obs)
        while len(self.history) > self.max_history:
            self.history.popleft()
        self.full_history.append(obs)
        self.frame_index = frame_index
        self.state = state
        self.total_matches += 1
        self.age += 1


@dataclass
class FeatureSpec:
    """How one modality maps into the head's per-track/detection tensors.

    ``dim`` is the feature length; ``extract(data)`` returns a length-``dim``
    float sequence from a detection's ObjectData (missing modality -> zeros).
    """

    name: str
    dim: int
    key: str  # ObjectData key holding the ready-to-tensor sequence


def _stack_features(
    specs: list[FeatureSpec],
    n_tracks: int,
    temporal_length: int,
) -> dict[str, torch.Tensor]:
    return {
        spec.name: torch.zeros(n_tracks, temporal_length, spec.dim, dtype=torch.float32)
        for spec in specs
    }


def _set_features(
    specs: list[FeatureSpec],
    features: dict[str, torch.Tensor],
    object_index: int,
    clip_index: int,
    data: ObjectData,
) -> None:
    for spec in specs:
        vals = data.get(spec.key)
        if vals is None:
            continue
        t = torch.as_tensor(vals, dtype=torch.float32)
        if t.numel() != spec.dim:
            raise ValueError(
                f"feature '{spec.name}' expected dim {spec.dim}, got {t.numel()} "
                f"(key '{spec.key}')"
            )
        features[spec.name][object_index, clip_index, :] = t


def _hungarian_match(
    cost: np.ndarray, sim_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with a similarity gate. ``cost`` is (n_tracks,
    n_dets) in [0, 1]; a pair with cost > ``sim_threshold`` is forbidden.
    Returns (matches, unmatched_tracks, unmatched_dets)."""
    n_tracks, n_dets = cost.shape
    if n_tracks == 0:
        return [], [], list(range(n_dets))
    if n_dets == 0:
        return [], list(range(n_tracks)), []

    # SciPy can't consume inf; forbid gated pairs with a cost strictly worse
    # than any admissible one, then drop over-threshold matches post-hoc.
    big = 1e6
    clamped = np.where(cost > sim_threshold, big, cost)
    rows, cols = linear_sum_assignment(clamped)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= sim_threshold:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_rows]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_cols]
    return matches, unmatched_tracks, unmatched_dets


class TDLPTracker:
    """Causal offline TDLP tracker over pre-assembled per-frame ObjectData.

    ``model`` is a MultiModalTDSP-style head: ``model(obs_features, obs_mask,
    unobs_features, unobs_mask) -> (agg_logits, _)`` with ``agg_logits`` shaped
    (1, N, M); cost = 1 - sigmoid(agg_logits). ``feature_specs`` must match the
    modalities the head was built/trained with.
    """

    def __init__(
        self,
        model,
        feature_specs: list[FeatureSpec],
        device: str = "cpu",
        remember_threshold: int = 30,
        detection_threshold: float = 0.4,
        sim_threshold: float = 0.5,
        initialization_threshold: int = 1,
        new_tracklet_detection_threshold: float = 0.9,
    ) -> None:
        self._model = model
        self._specs = feature_specs
        self._device = device
        self._remember = remember_threshold
        self._detection_threshold = detection_threshold
        self._sim_threshold = sim_threshold
        self._initialization_threshold = initialization_threshold
        self._new_tracklet_detection_threshold = new_tracklet_detection_threshold
        self._next_id = 0

    # -- tensor assembly -------------------------------------------------
    def _convert(
        self, tracklets: list[_Tracklet], objects_data: list[ObjectData], frame_index: int
    ):
        n_tracks = len(tracklets)
        n_dets = len(objects_data)
        n = max(n_tracks, n_dets)

        obs_ts = torch.zeros(n, self._remember, dtype=torch.long)
        obs_mask = torch.ones(n, self._remember, dtype=torch.bool)  # True = missing
        obs_feat = _stack_features(self._specs, n, self._remember)
        time_offset = frame_index - self._remember

        for t_i, tk in enumerate(tracklets):
            for obs in tk.history:
                rel = obs.frame_index - time_offset
                if rel < 0 or rel >= self._remember:
                    continue
                obs_ts[t_i, rel] = obs.frame_index
                obs_mask[t_i, rel] = False
                _set_features(self._specs, obs_feat, t_i, rel, obs.data)

        unobs_ts = torch.zeros(n, dtype=torch.long)
        unobs_mask = torch.ones(n, dtype=torch.bool)
        unobs_feat_full = _stack_features(self._specs, n, 1)
        unobs_ts[:n_dets] = frame_index
        unobs_mask[:n_dets] = False
        for d_i, data in enumerate(objects_data):
            _set_features(self._specs, unobs_feat_full, d_i, 0, data)
        unobs_feat = {k: v[:, 0] for k, v in unobs_feat_full.items()}

        return obs_feat, obs_mask, unobs_feat, unobs_mask

    @torch.no_grad()
    def _cost_matrix(
        self, tracklets: list[_Tracklet], objects_data: list[ObjectData], frame_index: int
    ) -> np.ndarray:
        n_tracks = len(tracklets)
        n_dets = len(objects_data)
        obs_feat, obs_mask, unobs_feat, unobs_mask = self._convert(
            tracklets, objects_data, frame_index
        )
        to_dev = lambda x: x.unsqueeze(0).to(self._device)  # noqa: E731
        obs_feat = {k: to_dev(v) for k, v in obs_feat.items()}
        unobs_feat = {k: to_dev(v) for k, v in unobs_feat.items()}
        obs_mask = to_dev(obs_mask)
        unobs_mask = to_dev(unobs_mask)

        agg_logits, _ = self._model(obs_feat, obs_mask, unobs_feat, unobs_mask)
        probas = torch.sigmoid(agg_logits).cpu().numpy()
        return 1.0 - probas[0, :n_tracks, :n_dets]

    # -- per-frame step --------------------------------------------------
    def _step(
        self, tracklets: list[_Tracklet], objects_data: list[ObjectData], frame_index: int
    ) -> list[_Tracklet]:
        tracklets = [t for t in tracklets if t.state != TrackletState.DELETED]
        if not tracklets:
            matches, unmatched_tracks, unmatched_dets = [], [], list(range(len(objects_data)))
        else:
            cost = self._cost_matrix(tracklets, objects_data, frame_index)
            matches, unmatched_tracks, unmatched_dets = _hungarian_match(
                cost, self._sim_threshold
            )

        for t_i, d_i in matches:
            tk = tracklets[t_i]
            new_state = TrackletState.ACTIVE
            if tk.state == TrackletState.NEW and tk.total_matches + 1 < self._initialization_threshold:
                new_state = TrackletState.NEW
            tk.update(frame_index, objects_data[d_i], new_state)

        for t_i in unmatched_tracks:
            tk = tracklets[t_i]
            lost_time = frame_index - tk.frame_index if tk.frame_index is not None else 0
            if lost_time > self._remember or tk.age < self._initialization_threshold:
                tk.state = TrackletState.DELETED
            else:
                tk.state = TrackletState.LOST

        new_tracklets: list[_Tracklet] = []
        for d_i in unmatched_dets:
            data = objects_data[d_i]
            if data["bbox_conf"] < self._new_tracklet_detection_threshold:
                continue
            tk = _Tracklet(
                id=self._next_id,
                max_history=self._remember,
                state=(
                    TrackletState.NEW
                    if frame_index > self._initialization_threshold
                    else TrackletState.ACTIVE
                ),
                frame_index=frame_index,
            )
            tk.update(frame_index, data, tk.state)
            self._next_id += 1
            new_tracklets.append(tk)

        tracklets.extend(new_tracklets)
        return tracklets

    def track_clip(
        self, frames: list[tuple[int, list[ObjectData]]], min_length: int = 1
    ) -> list[Tracklet]:
        """Run the full clip. ``frames`` is an ordered list of (frame_index,
        objects_data) with confidences already present. Returns PitchLab
        ``Tracklet`` artifacts (id-remapped to 0..K, short tracks dropped)."""
        all_tracklets: dict[int, _Tracklet] = {}
        live: list[_Tracklet] = []
        for frame_index, objects_data in frames:
            kept = [d for d in objects_data if d["bbox_conf"] > self._detection_threshold]
            live = self._step(live, kept, frame_index)
            for tk in live:
                all_tracklets[tk.id] = tk

        out: list[Tracklet] = []
        next_out_id = 0
        for tk in sorted(all_tracklets.values(), key=lambda t: t.id):
            frames_hist = _dedup_history(tk.full_history)
            if len(frames_hist) < min_length:
                continue
            tl_frames = [
                TrackletFrame(
                    frame_idx=obs.frame_index,
                    box=obs.data["box"],
                    confidence=float(obs.data["bbox_conf"]),
                    source="observed",
                )
                for obs in frames_hist
            ]
            out.append(
                Tracklet(tracklet_id=next_out_id, cls=DetectionClass.PLAYER, frames=tl_frames)
            )
            next_out_id += 1
        return out


def _dedup_history(history: list[_FrameObs]) -> list[_FrameObs]:
    seen: set[int] = set()
    ordered: list[_FrameObs] = []
    for obs in sorted(history, key=lambda o: o.frame_index):
        if obs.frame_index in seen:
            continue
        seen.add(obs.frame_index)
        ordered.append(obs)
    return ordered
