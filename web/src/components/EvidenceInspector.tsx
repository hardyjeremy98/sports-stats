// Anchor-frame forensic inspector: seeks the main video to an identity
// evidence record's source frame, draws full-frame context with the crop box
// outlined, zooms into the crop region, and shows the raw/upscaled crop pair
// plus quality metadata. Purely presentational — LabRunViewer owns open-state
// and highlight state; this component only reports navigation intents.

import { useEffect, useMemo, useRef } from "react";
import type { RefObject } from "react";
import { api } from "../lib/api";
import { ACCENT } from "../lib/colors";
import { fmtClock, fmtConf } from "../lib/format";
import type { PlayerEntity } from "../lib/types";
import type { VideoOverlayHandle } from "./VideoOverlay";
import { Button, Mono } from "./ui";

const ZOOM_FACTOR = 4;
const ZOOM_PADDING = 0.25; // fraction of box size added on each side, before zooming

export interface EvidenceInspectorProps {
  runId: string;
  player: PlayerEntity;
  evidenceIdx: number;
  fps: number;
  playerRef: RefObject<VideoOverlayHandle>;
  onClose: () => void;
  onSelectTracklet: (trackletId: number) => void;
  onNavigate: (evidenceIdx: number, trackletId: number) => void;
}

export function EvidenceInspector({
  runId,
  player,
  evidenceIdx,
  fps,
  playerRef,
  onClose,
  onSelectTracklet,
  onNavigate,
}: EvidenceInspectorProps) {
  const evidence = useMemo(
    () => player.identity.evidence.filter((e) => e.crop_artifact),
    [player],
  );
  const ev = evidence[evidenceIdx];
  const frameCanvasRef = useRef<HTMLCanvasElement>(null);
  const zoomCanvasRef = useRef<HTMLCanvasElement>(null);

  // Escape closes; arrow keys flip through this player's evidence.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowRight" && evidenceIdx < evidence.length - 1) {
        onNavigate(evidenceIdx + 1, evidence[evidenceIdx + 1].tracklet_id);
      } else if (e.key === "ArrowLeft" && evidenceIdx > 0) {
        onNavigate(evidenceIdx - 1, evidence[evidenceIdx - 1].tracklet_id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onNavigate, evidenceIdx, evidence]);

  // Seek the main video to this evidence's frame, then draw the frame (+
  // crop-box outline) and the zoom panel once the seek lands.
  useEffect(() => {
    if (!ev) return;
    const video = playerRef.current?.video ?? null;
    if (!video) return;

    const draw = () => {
      const frameCanvas = frameCanvasRef.current;
      if (!frameCanvas || !video.videoWidth) return;
      frameCanvas.width = video.videoWidth;
      frameCanvas.height = video.videoHeight;
      const ctx = frameCanvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0);

      if (ev.box) {
        const { x1, y1, x2, y2 } = ev.box;
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = Math.max(2, frameCanvas.width / 400);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      const zoomCanvas = zoomCanvasRef.current;
      if (zoomCanvas && ev.box) {
        const { x1, y1, x2, y2 } = ev.box;
        const bw = x2 - x1;
        const bh = y2 - y1;
        const sx = Math.max(0, x1 - bw * ZOOM_PADDING);
        const sy = Math.max(0, y1 - bh * ZOOM_PADDING);
        const sw = Math.min(video.videoWidth - sx, bw * (1 + 2 * ZOOM_PADDING));
        const sh = Math.min(video.videoHeight - sy, bh * (1 + 2 * ZOOM_PADDING));
        const zctx = zoomCanvas.getContext("2d");
        if (zctx && sw > 0 && sh > 0) {
          const dw = Math.round(sw * ZOOM_FACTOR);
          const dh = Math.round(sh * ZOOM_FACTOR);
          zoomCanvas.width = dw;
          zoomCanvas.height = dh;
          zctx.drawImage(video, sx, sy, sw, sh, 0, 0, dw, dh);
          zctx.strokeStyle = ACCENT;
          zctx.lineWidth = 2;
          zctx.strokeRect((x1 - sx) * ZOOM_FACTOR, (y1 - sy) * ZOOM_FACTOR, bw * ZOOM_FACTOR, bh * ZOOM_FACTOR);
        }
      }
    };

    playerRef.current?.seek(ev.frame_idx / fps);
    video.addEventListener("seeked", draw);
    // If the video is already sitting on the target frame (e.g. re-opening
    // the same evidence), `seeked` never fires — draw eagerly too.
    draw();
    return () => video.removeEventListener("seeked", draw);
  }, [ev, fps, playerRef]);

  if (!ev) return null;

  const video = playerRef.current?.video ?? null;
  const boxW = ev.box ? Math.round(ev.box.x2 - ev.box.x1) : null;
  const boxH = ev.box ? Math.round(ev.box.y2 - ev.box.y1) : null;

  const goto = (idx: number) => {
    if (idx < 0 || idx >= evidence.length) return;
    onNavigate(idx, evidence[idx].tracklet_id);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-white/10 bg-turf-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-white/8 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <span className="text-[13px] font-medium">
              {player.identity.label ?? `Player ${player.player_id}`}
            </span>
            <span className="font-mono text-[11px] text-ink-500">
              evidence {evidenceIdx + 1} / {evidence.length}
            </span>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-md px-2 py-1 text-lg leading-none text-ink-400 transition-colors hover:bg-turf-800 hover:text-ink-100"
          >
            ×
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
            {/* full-frame context + zoom */}
            <div className="flex flex-col gap-3">
              <div className="overflow-hidden rounded-lg border border-white/8 bg-black">
                {video ? (
                  <canvas ref={frameCanvasRef} className="block w-full" />
                ) : (
                  <div className="flex h-40 items-center justify-center text-[12px] text-ink-500">
                    Video not ready.
                  </div>
                )}
              </div>
              {ev.box ? (
                <div className="overflow-hidden rounded-lg border border-white/8 bg-black">
                  <canvas ref={zoomCanvasRef} className="block w-full" />
                </div>
              ) : (
                <div className="rounded-lg border border-dashed border-white/10 p-3 text-[12px] text-ink-500">
                  No crop geometry recorded on this evidence — re-run pipeline to capture crop
                  geometry.
                </div>
              )}
            </div>

            {/* crop pair + quality */}
            <div className="flex flex-col gap-4">
              <div>
                <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
                  Crop
                </div>
                <div className="flex gap-3">
                  <CropPreview label="upscaled" src={api.runFileUrl(runId, ev.crop_artifact!)} />
                  {ev.raw_crop_artifact && (
                    <CropPreview
                      label="raw"
                      src={api.runFileUrl(runId, ev.raw_crop_artifact)}
                    />
                  )}
                </div>
              </div>

              <div>
                <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
                  Quality
                </div>
                <div className="flex flex-col gap-1.5 rounded-lg border border-white/8 p-3 text-[12px]">
                  <Row label="Face score" value={fmtConf(ev.score)} />
                  <Row label="Upscaled" value={ev.upscaled ? "yes" : "no"} />
                  {boxW != null && boxH != null && (
                    <Row label="Crop box" value={`${boxW} × ${boxH}px`} />
                  )}
                  <Row
                    label="Frame"
                    value={`${ev.frame_idx} · ${fmtClock(ev.frame_idx / fps)}`}
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-ink-500">Tracklet</span>
                    <button
                      onClick={() => onSelectTracklet(ev.tracklet_id)}
                      className="font-mono text-[12px] text-volt-300 hover:underline"
                    >
                      T{ev.tracklet_id}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* evidence strip: flip through this player's anchors without closing */}
          {evidence.length > 1 && (
            <div className="mt-4 flex items-center gap-2 border-t border-white/8 pt-3">
              <Button
                variant="ghost"
                onClick={() => goto(evidenceIdx - 1)}
                disabled={evidenceIdx === 0}
                className="!px-2 !py-1"
              >
                ‹
              </Button>
              <div className="flex flex-1 gap-1.5 overflow-x-auto">
                {evidence.map((e, i) => (
                  <button
                    key={`${e.tracklet_id}-${e.frame_idx}`}
                    onClick={() => goto(i)}
                    title={`T${e.tracklet_id} · frame ${e.frame_idx} · score ${fmtConf(e.score)}`}
                    className={`h-14 w-14 shrink-0 overflow-hidden rounded-md border transition-colors ${
                      i === evidenceIdx
                        ? "border-volt-400/70"
                        : "border-white/10 hover:border-white/30"
                    }`}
                  >
                    <img
                      src={api.runFileUrl(runId, e.crop_artifact!)}
                      className="h-full w-full object-cover"
                    />
                  </button>
                ))}
              </div>
              <Button
                variant="ghost"
                onClick={() => goto(evidenceIdx + 1)}
                disabled={evidenceIdx === evidence.length - 1}
                className="!px-2 !py-1"
              >
                ›
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-ink-500">{label}</span>
      <Mono className="text-ink-100">{value}</Mono>
    </div>
  );
}

function CropPreview({ label, src }: { label: string; src: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="h-24 w-24 overflow-hidden rounded-md border border-white/10 bg-turf-950">
        <img src={src} className="h-full w-full object-cover" />
      </div>
      <span className="font-mono text-[10px] text-ink-500">{label}</span>
    </div>
  );
}
