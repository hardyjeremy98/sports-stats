from __future__ import annotations

from collections import defaultdict

from matchlab_core.schemas import (
    Event,
    FrameCalibration,
    FrameDetections,
    MinimapFrame,
    PlayerEntity,
    TimelineBucket,
    Tracklet,
)


def compute_timeline(
    duration_s: float,
    fps: float,
    detections: list[FrameDetections],
    tracklets: list[Tracklet],
    calibration: list[FrameCalibration],
    minimap: list[MinimapFrame],
    players: list[PlayerEntity],
    events: list[Event],
    bucket_s: float = 1.0,
) -> list[TimelineBucket]:
    """Aggregate per-frame signals into per-second buckets for the Lab timeline.

    The point of this artifact is scrub-guidance: an ML engineer should be able
    to see at a glance *where* in the clip each stage got shaky.
    """
    n = max(1, int(duration_s / bucket_s) + 1)

    det_conf: dict[int, list[float]] = defaultdict(list)
    for fd in detections:
        b = int(fd.t / bucket_s)
        if fd.detections:
            det_conf[b].append(sum(d.confidence for d in fd.detections) / len(fd.detections))

    # Tracking stability: births + deaths per bucket, normalized by active tracks.
    births: dict[int, int] = defaultdict(int)
    deaths: dict[int, int] = defaultdict(int)
    active: dict[int, set[int]] = defaultdict(set)
    for tr in tracklets:
        start_t = tr.start_frame / fps
        end_t = tr.end_frame / fps
        births[int(start_t / bucket_s)] += 1
        deaths[int(end_t / bucket_s)] += 1
        for b in range(int(start_t / bucket_s), int(end_t / bucket_s) + 1):
            active[b].add(tr.tracklet_id)

    calib_conf: dict[int, list[float]] = defaultdict(list)
    for fc in calibration:
        calib_conf[int(fc.t / bucket_s)].append(fc.confidence)

    identified = {p.player_id for p in players if p.identity.kind != "none"}
    id_cov: dict[int, list[float]] = defaultdict(list)
    for mf in minimap:
        b = int(mf.t / bucket_s)
        if mf.players:
            id_cov[b].append(
                sum(1 for p in mf.players if p.player_id in identified) / len(mf.players)
            )

    ev_count: dict[int, int] = defaultdict(int)
    contested: dict[int, int] = defaultdict(int)
    for ev in events:
        b = int(ev.t / bucket_s)
        ev_count[b] += 1
        if ev.contested:
            contested[b] += 1

    buckets = []
    for b in range(n):
        n_active = len(active.get(b, set()))
        churn = births.get(b, 0) + deaths.get(b, 0)
        stability = max(0.0, 1.0 - churn / n_active) if n_active else 0.0
        flags = []
        dc = _mean(det_conf.get(b))
        cc = _mean(calib_conf.get(b))
        if dc < 0.4 and det_conf.get(b):
            flags.append("weak_detection")
        if cc < 0.3 and calib_conf.get(b):
            flags.append("calibration_gap")
        if n_active and stability < 0.5:
            flags.append("track_churn")
        if contested.get(b, 0):
            flags.append("contested_events")
        buckets.append(
            TimelineBucket(
                t=b * bucket_s,
                detection_confidence=round(dc, 4),
                tracking_stability=round(stability, 4),
                calibration_confidence=round(cc, 4),
                identity_coverage=round(_mean(id_cov.get(b)), 4),
                event_count=ev_count.get(b, 0),
                contested_event_count=contested.get(b, 0),
                flags=flags,
            )
        )
    return buckets


def _mean(vals: list[float] | None) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
