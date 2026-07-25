// The Lab's forensic video player: original footage + a canvas drawing
// toggleable pipeline layers, driven by a rAF loop reading video.currentTime.

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import { floorRow, type GtIndex, type RunArtifacts } from "../lib/artifacts";
import { teamColor, trackletColor } from "../lib/colors";
import { eventLabel } from "../lib/format";
import type { RunDetail } from "../lib/types";

export interface LayerState {
  detections: boolean;
  ball: boolean;
  tracklets: boolean;
  team: boolean;
  identity: boolean;
  keypoints: boolean;
  events: boolean;
  spotting: boolean;
  gt: boolean;
}

export const DEFAULT_LAYERS: LayerState = {
  detections: false,
  ball: true,
  tracklets: false,
  team: true,
  identity: true,
  keypoints: false,
  events: true,
  spotting: true,
  gt: true,
};

const LAYER_DEFS: {
  id: keyof LayerState;
  label: string;
  needs: (a: RunArtifacts, gt?: GtIndex | null) => boolean;
}[] = [
  { id: "detections", label: "Detections", needs: (a) => !!a.detections },
  { id: "ball", label: "Ball", needs: (a) => !!a.ball },
  { id: "tracklets", label: "Tracklets", needs: (a) => !!a.tracklets },
  { id: "team", label: "Team", needs: (a) => !!a.tracklets && !!a.teams },
  { id: "identity", label: "Identity", needs: (a) => !!a.tracklets && !!a.players },
  { id: "keypoints", label: "Pitch keypoints", needs: (a) => !!a.calibration },
  { id: "events", label: "Events", needs: (a) => !!a.events },
  { id: "spotting", label: "Spotting", needs: (a) => !!a.spotting?.length },
  { id: "gt", label: "Ground truth", needs: (_a, gt) => !!gt },
];

export interface VideoOverlayHandle {
  seek: (t: number) => void;
  getTime: () => number;
  video: HTMLVideoElement | null;
}

export const VideoOverlay = forwardRef<
  VideoOverlayHandle,
  {
    run: RunDetail;
    artifacts: RunArtifacts;
    layers: LayerState;
    gt?: GtIndex | null;
    highlightTrackletId?: number | null;
    highlightTrackletIds?: number[] | null;
    highlightPlayerId?: number | null;
    highlightGtTrackId?: number | null;
    controls?: boolean;
    scrubber?: boolean;
  }
>(function VideoOverlay(
  {
    run,
    artifacts,
    layers,
    gt,
    highlightTrackletId,
    highlightTrackletIds,
    highlightPlayerId,
    highlightGtTrackId,
    controls = true,
    scrubber = true,
  },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const stepFrames = (n: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    const frame = Math.round(v.currentTime * fps);
    const t = (frame + n) / fps;
    const max = v.duration || run.video?.duration_s || Infinity;
    v.currentTime = Math.min(Math.max(0, t), max);
  };

  useImperativeHandle(ref, () => ({
    seek: (t: number) => {
      if (videoRef.current) videoRef.current.currentTime = t;
    },
    getTime: () => videoRef.current?.currentTime ?? 0,
    video: videoRef.current,
  }));

  const fps = run.video?.fps || run.manifest?.video.fps || 25;
  const eventsSorted = useMemo(
    () => (artifacts.events ? [...artifacts.events].sort((a, b) => a.t - b.t) : null),
    [artifacts.events],
  );
  const spottingSorted = useMemo(
    () => (artifacts.spotting ? [...artifacts.spotting].sort((a, b) => a.t - b.t) : null),
    [artifacts.spotting],
  );

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (!video.videoWidth) return;
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const t = video.currentTime;
      const frameIdx = Math.floor(t * fps);
      // Native-resolution drawing; CSS scales the canvas with the video.
      const px = Math.max(1.5, canvas.width / 900);
      ctx.font = `${Math.round(11 * px)}px ui-monospace, monospace`;
      ctx.lineWidth = 2 * px;

      if (layers.detections && artifacts.detections) {
        const row = floorRow(artifacts.detections, frameIdx, (r) => r.frame_idx);
        if (row) {
          for (const d of row.detections) {
            ctx.strokeStyle =
              d.cls === "ball" ? "rgba(255,255,255,0.9)" : "rgba(154,165,157,0.8)";
            ctx.strokeRect(d.box.x1, d.box.y1, d.box.x2 - d.box.x1, d.box.y2 - d.box.y1);
            ctx.fillStyle = "rgba(154,165,157,0.9)";
            ctx.fillText(d.confidence.toFixed(2), d.box.x1, d.box.y1 - 3 * px);
          }
        }
      }

      // Ground truth first, so predicted boxes draw on top of the reference.
      if (layers.gt && gt) {
        const boxes = gt.byFrame.get(Math.min(frameIdx, gt.gt.seq_length - 1)) ?? [];
        for (const b of boxes) {
          const isHlGt = highlightGtTrackId != null && b.track_id === highlightGtTrackId;
          const color = isHlGt
            ? "#9BE532"
            : b.role === "referee"
              ? "rgba(245,197,24,0.85)"
              : "rgba(232,237,233,0.75)";
          ctx.setLineDash(isHlGt ? [] : [6 * px, 4 * px]);
          ctx.strokeStyle = color;
          ctx.lineWidth = (isHlGt ? 3.5 : 1.5) * px;
          ctx.strokeRect(b.box.x1, b.box.y1, b.box.x2 - b.box.x1, b.box.y2 - b.box.y1);
          const text =
            b.role === "referee"
              ? `GT${b.track_id} ref`
              : `GT${b.track_id}${b.jersey ? ` #${b.jersey}` : ""}`;
          ctx.fillStyle = color;
          // Below the box: predicted labels sit above, so both stay readable.
          ctx.fillText(text, b.box.x1, b.box.y2 + 11 * px);
        }
        ctx.setLineDash([]);
      }

      const showBoxes = layers.tracklets || layers.team || layers.identity;
      if (showBoxes && artifacts.tracklets) {
        const boxes = nearestFrameBoxes(artifacts, frameIdx);
        for (const b of boxes) {
          const ent = artifacts.entityByTracklet.get(b.tracklet_id);
          const team = artifacts.teamByTracklet.get(b.tracklet_id)?.team ?? ent?.team;
          let color = trackletColor(b.tracklet_id);
          if (layers.team && team) color = teamColor(team);
          const isHl =
            (highlightTrackletId != null && b.tracklet_id === highlightTrackletId) ||
            (highlightTrackletIds != null && highlightTrackletIds.includes(b.tracklet_id)) ||
            (highlightPlayerId != null && ent?.player_id === highlightPlayerId);
          ctx.strokeStyle = isHl ? "#9BE532" : color;
          ctx.lineWidth = (isHl ? 3.5 : 2) * px;
          ctx.strokeRect(b.box.x1, b.box.y1, b.box.x2 - b.box.x1, b.box.y2 - b.box.y1);

          const labels: string[] = [];
          if (layers.tracklets) labels.push(`T${b.tracklet_id}`);
          if (layers.identity && ent)
            labels.push(ent.identity.label ?? `#${ent.player_id}`);
          if (labels.length) {
            const text = labels.join(" · ");
            const tw = ctx.measureText(text).width;
            ctx.fillStyle = "rgba(11,15,13,0.85)";
            ctx.fillRect(b.box.x1, b.box.y1 - 15 * px, tw + 8 * px, 14 * px);
            ctx.fillStyle = isHl ? "#9BE532" : color;
            ctx.fillText(text, b.box.x1 + 4 * px, b.box.y1 - 4 * px);
          }
        }
      }

      if (layers.ball && artifacts.ball) {
        const idx = floorIndex(artifacts.ball, frameIdx);
        if (idx >= 0) {
          // Trail: last 10 observations.
          for (let k = Math.max(0, idx - 10); k <= idx; k++) {
            const o = artifacts.ball[k];
            const age = (idx - k) / 10;
            ctx.beginPath();
            ctx.arc(o.xy.x, o.xy.y, (2 + 4 * (1 - age)) * px, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(255,255,255,${0.15 + 0.75 * (1 - age)})`;
            ctx.lineWidth = 1.5 * px;
            ctx.stroke();
          }
          const cur = artifacts.ball[idx];
          if (cur.interpolated) {
            ctx.fillStyle = "rgba(245,197,24,0.9)";
            ctx.fillText("interp", cur.xy.x + 8 * px, cur.xy.y - 8 * px);
          }
        }
      }

      if (layers.keypoints && artifacts.calibration) {
        const row = floorRow(artifacts.calibration, frameIdx, (r) => r.frame_idx);
        if (row) {
          // Colour by the offline smoother's provenance status (fresh=green,
          // smoothed=amber, interpolated=blue; absent rows carry no keypoints so
          // nothing draws). Legacy rows without `status` fall back to the
          // fresh-vs-carried `smoothed` bool.
          const green = "rgba(155,229,50,0.9)";
          const amber = "rgba(245,197,24,0.9)";
          const blue = "rgba(90,170,245,0.9)";
          const statusColors: Record<string, string> = {
            fresh: green,
            smoothed: amber,
            interpolated: blue,
          };
          const kpColor = row.status
            ? (statusColors[row.status] ?? green)
            : row.smoothed
              ? amber
              : green;
          row.keypoints_image.forEach((kp, i) => {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 4 * px, 0, Math.PI * 2);
            ctx.fillStyle = kpColor;
            ctx.fill();
            const conf = row.keypoint_confidences[i];
            if (conf != null) {
              ctx.fillStyle = "rgba(232,237,233,0.75)";
              ctx.fillText(conf.toFixed(2), kp.x + 6 * px, kp.y - 4 * px);
            }
          });
          const provenance = row.status
            ? ` (${row.status})`
            : row.smoothed
              ? " (carried)"
              : "";
          ctx.fillStyle = "rgba(232,237,233,0.6)";
          ctx.fillText(
            `H conf ${row.confidence.toFixed(2)}${provenance} · ${row.n_keypoints} kp`,
            10 * px,
            canvas.height - 10 * px,
          );
        }
      }

      if (layers.events && eventsSorted) {
        // Show events within the last 1.2s as a caption stack.
        const recent = eventsSorted.filter((e) => e.t <= t && t - e.t < 1.2);
        recent.slice(-3).forEach((e, i) => {
          const text = `${eventLabel(e.type).toUpperCase()} — ${
            e.player_id != null ? `player ${e.player_id}` : "?"
          }${e.contested ? "  ⚑" : ""}`;
          ctx.font = `bold ${Math.round(13 * px)}px ui-monospace, monospace`;
          const tw = ctx.measureText(text).width;
          const y = (14 + i * 24) * px;
          ctx.fillStyle = "rgba(11,15,13,0.8)";
          ctx.fillRect(10 * px, y, tw + 12 * px, 20 * px);
          ctx.fillStyle = e.contested ? "#F5C518" : "#9BE532";
          ctx.fillText(text, 16 * px, y + 14 * px);
          ctx.font = `${Math.round(11 * px)}px ui-monospace, monospace`;
        });
      }

      if (layers.spotting && spottingSorted) {
        // Mirrors the events caption stack, but anchored to the right edge so
        // both layers can be on at once without overlapping.
        const recent = spottingSorted.filter((e) => e.t <= t && t - e.t < 1.2);
        recent.slice(-3).forEach((e, i) => {
          const text = `${e.class.toUpperCase()} — ${(e.confidence * 100).toFixed(0)}%`;
          ctx.font = `bold ${Math.round(13 * px)}px ui-monospace, monospace`;
          const tw = ctx.measureText(text).width;
          const y = (14 + i * 24) * px;
          const x = canvas.width - tw - 22 * px;
          ctx.fillStyle = "rgba(11,15,13,0.8)";
          ctx.fillRect(x, y, tw + 12 * px, 20 * px);
          ctx.fillStyle = "#2C91FF";
          ctx.fillText(text, x + 6 * px, y + 14 * px);
          ctx.font = `${Math.round(11 * px)}px ui-monospace, monospace`;
        });
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [
    artifacts,
    layers,
    gt,
    fps,
    eventsSorted,
    spottingSorted,
    highlightTrackletId,
    highlightTrackletIds,
    highlightPlayerId,
    highlightGtTrackId,
  ]);

  return (
    <div className="overflow-hidden rounded-xl border border-white/8 bg-black">
      <div className="relative">
        <video
          ref={videoRef}
          src={`/api/videos/${run.video_id}/file`}
          controls={controls}
          playsInline
          className="block w-full"
        />
        <canvas ref={canvasRef} className="pointer-events-none absolute inset-x-0 top-0 w-full" />
      </div>
      {scrubber && (
        <div className="flex items-center gap-2 border-t border-white/8 px-3 py-1.5 font-mono text-xs text-ink-400">
          <span className="text-ink-600">Scrub</span>
          {(
            [
              ["<<", -5, "back 5 frames"],
              ["<", -1, "back 1 frame"],
              [">", 1, "forward 1 frame"],
              [">>", 5, "forward 5 frames"],
            ] as const
          ).map(([label, delta, title]) => (
            <button
              key={label}
              className="rounded border border-white/8 px-2 py-0.5 hover:text-ink-100"
              title={title}
              onClick={() => stepFrames(delta)}
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
});

function nearestFrameBoxes(artifacts: RunArtifacts, frameIdx: number) {
  // Tracklet boxes exist on sampled frames only; snap to the nearest sampled
  // frame at or before frameIdx (tolerating sample_stride gaps).
  const idxs = artifacts.sortedFrameIdxs;
  if (idxs.length === 0) return [];
  let lo = 0;
  let hi = idxs.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (idxs[mid] <= frameIdx) {
      best = idxs[mid];
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (best < 0 || frameIdx - best > 12) return [];
  return artifacts.trackletBoxesByFrame.get(best) ?? [];
}

function floorIndex<T extends { frame_idx: number }>(rows: T[], frameIdx: number): number {
  let lo = 0;
  let hi = rows.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].frame_idx <= frameIdx) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

export function LayerChips({
  layers,
  onChange,
  artifacts,
  artifactsB,
  gt,
}: {
  layers: LayerState;
  onChange: (l: LayerState) => void;
  artifacts: RunArtifacts;
  artifactsB?: RunArtifacts;
  gt?: GtIndex | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {LAYER_DEFS.map((def) => {
        // Availability is the union: a chip is available if the layer's needs()
        // predicate passes for EITHER run's artifacts.
        const available = def.needs(artifacts, gt) || (artifactsB ? def.needs(artifactsB, gt) : false);
        const on = layers[def.id] && available;
        return (
          <button
            key={def.id}
            disabled={!available}
            title={
              available
                ? undefined
                : artifactsB
                  ? "artifact not available in either run"
                  : "artifact not available in this run"
            }
            onClick={() => onChange({ ...layers, [def.id]: !layers[def.id] })}
            className={`rounded-full border px-3 py-1 text-[12px] transition-colors ${
              on
                ? "border-volt-400/60 bg-volt-400/15 text-volt-300"
                : available
                  ? "border-white/10 text-ink-400 hover:border-white/25 hover:text-ink-100"
                  : "cursor-not-allowed border-white/5 text-ink-500/50"
            }`}
          >
            {def.label}
          </button>
        );
      })}
    </div>
  );
}
