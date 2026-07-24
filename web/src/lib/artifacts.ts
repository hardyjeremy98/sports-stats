// Loading + preprocessing of run artifacts for the viewers. Everything is
// fetched once per run (react-query, staleTime Infinity) and preprocessed into
// frame-indexed structures for O(log n) lookup in the rAF draw loop.

import { useMemo } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { api, fetchJson, fetchJsonl } from "./api";
import type {
  AssociationReport,
  BallObservation,
  EvalResult,
  FrameCalibration,
  FrameDetections,
  GroundTruth,
  GroundTruthTrack,
  MatchEvent,
  MinimapFrame,
  NamingReport,
  PlayerEntity,
  ReidDetailReport,
  SpottedEvent,
  StatSheet,
  TeamAssignment,
  TimelineBucket,
  Tracklet,
  TrackletFrame,
} from "./types";

export interface TrackletBox {
  tracklet_id: number;
  cls: string;
  box: TrackletFrame["box"];
  confidence: number;
}

export interface RunArtifacts {
  detections: FrameDetections[] | null;
  ball: BallObservation[] | null;
  tracklets: Tracklet[] | null;
  teams: TeamAssignment[] | null;
  calibration: FrameCalibration[] | null;
  players: PlayerEntity[] | null;
  minimap: MinimapFrame[] | null;
  events: MatchEvent[] | null;
  spotting: SpottedEvent[] | null;
  stats: StatSheet | null;
  timeline: TimelineBucket[] | null;
  eval: EvalResult | null;
  association: AssociationReport | null;
  naming: NamingReport | null;
  reidDetail: ReidDetailReport | null;
  // Derived indexes.
  trackletBoxesByFrame: Map<number, TrackletBox[]>;
  teamByTracklet: Map<number, TeamAssignment>;
  entityByTracklet: Map<number, PlayerEntity>;
  sortedFrameIdxs: number[]; // frames that have tracklet boxes, ascending
  loading: boolean;
}

/** Largest element of `sorted` that is <= target, by key. */
export function floorRow<T>(sorted: T[], target: number, key: (row: T) => number): T | null {
  let lo = 0;
  let hi = sorted.length - 1;
  let best: T | null = null;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (key(sorted[mid]) <= target) {
      best = sorted[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

const artifactQueries = (runId: string, enabled: boolean) => [
  { key: "detections", fn: () => fetchJsonl<FrameDetections>(api.artifactUrl(runId, "detections")) },
  { key: "ball", fn: () => fetchJsonl<BallObservation>(api.artifactUrl(runId, "ball")) },
  { key: "tracklets", fn: () => fetchJson<Tracklet[]>(api.artifactUrl(runId, "tracklets")) },
  { key: "teams", fn: () => fetchJson<TeamAssignment[]>(api.artifactUrl(runId, "teams")) },
  { key: "calibration", fn: () => fetchJsonl<FrameCalibration>(api.artifactUrl(runId, "calibration")) },
  { key: "players", fn: () => fetchJson<PlayerEntity[]>(api.artifactUrl(runId, "players")) },
  { key: "minimap", fn: () => fetchJsonl<MinimapFrame>(api.artifactUrl(runId, "minimap")) },
  { key: "events", fn: () => fetchJson<MatchEvent[]>(api.artifactUrl(runId, "events")) },
  { key: "spotting", fn: () => fetchJson<SpottedEvent[]>(api.artifactUrl(runId, "spotting")) },
  { key: "stats", fn: () => fetchJson<StatSheet>(api.artifactUrl(runId, "stats")) },
  { key: "timeline", fn: () => fetchJson<TimelineBucket[]>(api.artifactUrl(runId, "timeline")) },
  { key: "eval", fn: () => fetchJson<EvalResult>(api.artifactUrl(runId, "eval")) },
  { key: "association", fn: () => fetchJson<AssociationReport>(api.artifactUrl(runId, "association")) },
  { key: "naming", fn: () => fetchJson<NamingReport>(api.artifactUrl(runId, "naming")) },
  { key: "reid_detail", fn: () => fetchJson<ReidDetailReport>(api.artifactUrl(runId, "reid_detail")) },
].map((q) => ({
  queryKey: ["artifact", runId, q.key],
  queryFn: q.fn,
  enabled,
  staleTime: Infinity,
  retry: false, // a missing artifact is a state, not a transient failure
}));

export function useRunArtifacts(runId: string, enabled: boolean): RunArtifacts {
  const results = useQueries({ queries: artifactQueries(runId, enabled) });

  const [
    detections,
    ball,
    tracklets,
    teams,
    calibration,
    players,
    minimap,
    events,
    spotting,
    stats,
    timeline,
    evalResult,
    association,
    naming,
    reidDetail,
  ] = results.map((r) => (r.isSuccess ? r.data : null));

  const trackletBoxesByFrame = new Map<number, TrackletBox[]>();
  if (tracklets) {
    for (const tr of tracklets as Tracklet[]) {
      for (const f of tr.frames) {
        let list = trackletBoxesByFrame.get(f.frame_idx);
        if (!list) {
          list = [];
          trackletBoxesByFrame.set(f.frame_idx, list);
        }
        list.push({
          tracklet_id: tr.tracklet_id,
          cls: tr.cls,
          box: f.box,
          confidence: f.confidence,
        });
      }
    }
  }

  const teamByTracklet = new Map<number, TeamAssignment>();
  for (const t of (teams as TeamAssignment[] | null) ?? []) teamByTracklet.set(t.tracklet_id, t);

  const entityByTracklet = new Map<number, PlayerEntity>();
  for (const p of (players as PlayerEntity[] | null) ?? [])
    for (const tid of p.tracklet_ids) entityByTracklet.set(tid, p);

  return {
    detections: detections as FrameDetections[] | null,
    ball: ball as BallObservation[] | null,
    tracklets: tracklets as Tracklet[] | null,
    teams: teams as TeamAssignment[] | null,
    calibration: calibration as FrameCalibration[] | null,
    players: players as PlayerEntity[] | null,
    minimap: minimap as MinimapFrame[] | null,
    events: events as MatchEvent[] | null,
    spotting: spotting as SpottedEvent[] | null,
    stats: stats as StatSheet | null,
    timeline: timeline as TimelineBucket[] | null,
    eval: evalResult as EvalResult | null,
    association: association as AssociationReport | null,
    naming: naming as NamingReport | null,
    reidDetail: reidDetail as ReidDetailReport | null,
    trackletBoxesByFrame,
    teamByTracklet,
    entityByTracklet,
    sortedFrameIdxs: [...trackletBoxesByFrame.keys()].sort((a, b) => a - b),
    loading: results.some((r) => r.isLoading),
  };
}

// ---- Ground truth (per video, not per run) ----

export interface GtBox {
  track_id: number;
  role: GroundTruthTrack["role"];
  team: GroundTruthTrack["team"];
  jersey: string | null;
  box: TrackletFrame["box"];
}

export interface GtIndex {
  gt: GroundTruth;
  byFrame: Map<number, GtBox[]>;
}

export function useGroundTruth(videoId: number | null | undefined, enabled: boolean): GtIndex | null {
  const query = useQuery({
    queryKey: ["ground_truth", videoId],
    queryFn: () => fetchJson<GroundTruth>(api.videoGtUrl(videoId!)),
    enabled: enabled && videoId != null,
    staleTime: Infinity,
    retry: false,
  });

  const gt = query.isSuccess ? query.data : null;
  return useMemo(() => {
    if (!gt) return null;
    const byFrame = new Map<number, GtBox[]>();
    for (const tr of gt.tracks) {
      if (tr.role === "ball") continue; // ball is a separate pipeline stream
      for (const f of tr.frames) {
        let list = byFrame.get(f.frame_idx);
        if (!list) {
          list = [];
          byFrame.set(f.frame_idx, list);
        }
        list.push({ track_id: tr.track_id, role: tr.role, team: tr.team, jersey: tr.jersey, box: f.box });
      }
    }
    return { gt, byFrame };
  }, [gt]);
}
