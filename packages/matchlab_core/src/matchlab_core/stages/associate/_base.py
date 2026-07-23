"""Shared machinery for global cross-tracklet associators (locked decision #5).

Groups the tracker's short tracklets into whole-clip player entities using the
full video — this is an upload-and-process product, so we exploit offline
access to everything:

  hard constraints: same team; no temporal overlap between merged tracklets
                    (one body can't be in two places); plausible travel speed
                    across the gap between one tracklet's end and the next's
                    start.
  affinity:         an appearance distance (cheap color embedding, or a
                     learned re-ID embedding) plus a small penalty for long
                     gaps.

Greedy union-find over candidate pairs sorted by affinity — a deliberately
simple GTA-Link-shaped pass. Concrete associators subclass `GlobalAssociatorBase`
and only need to supply the feature extraction + distance/gate/affinity
functions; everything else (constraints, the AssociationReport decision
trail, union-find, entity building) lives here."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from pydantic import BaseModel

from matchlab_core.interfaces import Associator, StageContext
from matchlab_core.schemas import (
    ArtifactName,
    AssociationEntitySummary,
    AssociationPair,
    AssociationRejectReason,
    AssociationReport,
    DetectionClass,
    PlayerEntity,
    Team,
    TeamAssignment,
    Tracklet,
)


class BaseParams(BaseModel):
    max_gap_s: float = 20.0             # don't bridge longer disappearances
    max_speed_px_s: float = 800.0       # plausible travel across the gap
    overlap_tolerance_frames: int = 2   # tracker handoff jitter


class GlobalAssociatorBase(Associator):
    """Base class for global associators. Subclasses supply the appearance
    feature extraction and distance/gate/affinity functions via the hooks
    below; this class owns the constraint pipeline, the AssociationReport
    decision trail, and the greedy union-find merge."""

    impl_name: str
    gate_reject_reason: AssociationRejectReason
    params: BaseParams

    @abstractmethod
    def _features(self, ctx: StageContext, tracklets: list[Tracklet]) -> dict[int, np.ndarray]:
        ...

    @abstractmethod
    def _distance(self, fa: np.ndarray, fb: np.ndarray) -> float:
        ...

    @abstractmethod
    def _gate(self, dist: float) -> bool:
        """True = reject the pair on appearance distance alone."""
        ...

    @abstractmethod
    def _affinity(self, dist: float, gap_s: float) -> float:
        ...

    @abstractmethod
    def _record_distance(self, pair: AssociationPair, dist: float) -> None:
        """Set the report field that carries this associator's appearance
        distance (e.g. color_distance vs embed_distance)."""
        ...

    def associate(
        self, ctx: StageContext, tracklets: list[Tracklet], teams: list[TeamAssignment]
    ) -> list[PlayerEntity]:
        feats = self._features(ctx, tracklets)
        return self._associate_with_features(ctx, tracklets, teams, feats, fps=ctx.video.fps)

    def _associate_with_features(
        self,
        ctx: StageContext | None,
        tracklets: list[Tracklet],
        teams: list[TeamAssignment],
        feats: dict[int, np.ndarray],
        fps: float,
        write_report: bool = True,
    ) -> list[PlayerEntity]:
        """Core associate logic, factored out so `write_report=False` callers
        (e.g. the reid-ablation sweep) can pass `ctx=None` and skip the report."""
        if write_report and ctx is None:
            raise ValueError("write_report=True requires a StageContext")
        p = self.params
        team_by_tid = {t.tracklet_id: t.team for t in teams}

        idx = {tr.tracklet_id: tr for tr in tracklets}
        ids = [tr.tracklet_id for tr in tracklets]
        parent = {tid: tid for tid in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        candidates = []
        # Every pair that clears the structural filters (referee exclusion,
        # team mismatch) gets a report entry — those two filters stay silent
        # `continue`s and are never recorded, to bound the O(n^2) payload.
        report_pairs: list[AssociationPair] = []
        pending_pairs: dict[tuple[int, int], AssociationPair] = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                ta, tb = idx[a], idx[b]
                if ta.cls == DetectionClass.REFEREE or tb.cls == DetectionClass.REFEREE:
                    continue
                if team_by_tid.get(a, Team.UNKNOWN) != team_by_tid.get(b, Team.UNKNOWN):
                    continue
                # Earlier-starting tracklet first, purely for report readability
                # — does not affect the union-find below, which keeps using
                # the raw (a, b) loop order.
                first, second = (ta, tb) if ta.start_frame <= tb.start_frame else (tb, ta)
                pa, pb = first.tracklet_id, second.tracklet_id

                if a not in feats or b not in feats:
                    report_pairs.append(
                        AssociationPair(
                            a=pa, b=pb, decision="rejected",
                            reason=AssociationRejectReason.NO_FEATURES,
                        )
                    )
                    continue
                gap_frames = second.start_frame - first.end_frame
                if gap_frames < -p.overlap_tolerance_frames:
                    report_pairs.append(
                        AssociationPair(
                            a=pa, b=pb, decision="rejected",
                            reason=AssociationRejectReason.TEMPORAL_OVERLAP,
                        )
                    )
                    continue
                gap_s = max(gap_frames, 1) / fps
                if gap_s > p.max_gap_s:
                    report_pairs.append(
                        AssociationPair(
                            a=pa, b=pb, gap_s=gap_s, decision="rejected",
                            reason=AssociationRejectReason.GAP_TOO_LONG,
                        )
                    )
                    continue
                end_box = first.frames[-1].box
                start_box = second.frames[0].box
                dist_px = float(
                    np.hypot(
                        start_box.bottom_center.x - end_box.bottom_center.x,
                        start_box.bottom_center.y - end_box.bottom_center.y,
                    )
                )
                if dist_px / gap_s > p.max_speed_px_s:
                    report_pairs.append(
                        AssociationPair(
                            a=pa, b=pb, gap_s=gap_s, dist_px=dist_px, decision="rejected",
                            reason=AssociationRejectReason.SPEED_IMPLAUSIBLE,
                        )
                    )
                    continue
                dist = self._distance(feats[a], feats[b])
                if self._gate(dist):
                    pair = AssociationPair(
                        a=pa, b=pb, gap_s=gap_s, dist_px=dist_px,
                        decision="rejected", reason=self.gate_reject_reason,
                    )
                    self._record_distance(pair, dist)
                    report_pairs.append(pair)
                    continue
                affinity = self._affinity(dist, gap_s)
                candidates.append((affinity, a, b))
                # Decision is resolved in the union-find phase below; "rejected"
                # here is a placeholder that always gets overwritten, since
                # every entry in `candidates` is processed exactly once there.
                pair = AssociationPair(
                    a=pa, b=pb, gap_s=gap_s, dist_px=dist_px,
                    affinity=affinity, decision="rejected",
                )
                self._record_distance(pair, dist)
                pending_pairs[(a, b)] = pair
                report_pairs.append(pair)

        spans: dict[int, list[tuple[int, int]]] = {
            tid: [(idx[tid].start_frame, idx[tid].end_frame)] for tid in ids
        }
        merge_edges: list[tuple[int, int]] = []
        for _, a, b in sorted(candidates):
            pair = pending_pairs[(a, b)]
            ra, rb = find(a), find(b)
            if ra == rb:
                # Already in the same component (transitively merged via some
                # other pair) — a redundant but consistent candidate. Recorded
                # as "merged" since that's what it represents, not as a fresh
                # union-find edge.
                pair.decision = "merged"
                continue
            if _spans_overlap(spans[ra], spans[rb], p.overlap_tolerance_frames):
                pair.decision = "rejected"
                pair.reason = AssociationRejectReason.SPAN_CONFLICT
                continue
            parent[rb] = ra
            spans[ra] = sorted(spans[ra] + spans[rb])
            pair.decision = "merged"
            # Appended in the order union-find resolves candidates (ascending
            # affinity), not tracklet start order — don't assume otherwise.
            merge_edges.append((a, b))

        groups: dict[int, list[int]] = {}
        for tid in ids:
            groups.setdefault(find(tid), []).append(tid)

        merge_edges_by_root: dict[int, list[tuple[int, int]]] = {}
        for a, b in merge_edges:
            merge_edges_by_root.setdefault(find(a), []).append((a, b))

        entities = []
        report_entities = []
        for n, (root, members) in enumerate(sorted(groups.items()), start=1):
            tr0 = idx[root]
            team = (
                Team.REFEREE
                if tr0.cls == DetectionClass.REFEREE
                else team_by_tid.get(root, Team.UNKNOWN)
            )
            entities.append(
                PlayerEntity(
                    player_id=n,
                    tracklet_ids=sorted(members),
                    team=team,
                    association_confidence=1.0 if len(members) == 1 else 0.8,
                )
            )
            report_entities.append(
                AssociationEntitySummary(
                    player_id=n,
                    tracklet_ids=sorted(members),
                    merge_edges=merge_edges_by_root.get(root, []),
                )
            )

        if write_report:
            report = AssociationReport(
                impl=self.impl_name,
                params=self.params.model_dump(),
                pairs=report_pairs,
                entities=report_entities,
            )
            ctx.store.write_json(ArtifactName.ASSOCIATION, report)
        return entities


def _spans_overlap(
    a: list[tuple[int, int]], b: list[tuple[int, int]], tol: int
) -> bool:
    for s1, e1 in a:
        for s2, e2 in b:
            if min(e1, e2) - max(s1, s2) > tol:
                return True
    return False
