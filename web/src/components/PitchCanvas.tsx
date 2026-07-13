// 2D pitch replay: draws the pitch spec + player dots + ball, synced to an
// external time source (usually a <video> element) via rAF.

import { useEffect, useRef } from "react";
import { floorRow } from "../lib/artifacts";
import { teamColor } from "../lib/colors";
import type { MinimapFrame } from "../lib/types";

// Pitch geometry in cm — mirrors pitchlab_core.pitch.SOCCER_PITCH.
const L = 12000;
const W = 7000;
const PBL = 2015;
const PBW = 4100;
const GBL = 550;
const GBW = 1832;
const CCR = 915;

export function drawPitch(ctx: CanvasRenderingContext2D, w: number, h: number) {
  const sx = w / L;
  const sy = h / W;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#122015";
  ctx.fillRect(0, 0, w, h);
  // Mow stripes.
  ctx.fillStyle = "rgba(255,255,255,0.018)";
  for (let i = 0; i < 12; i += 2) {
    ctx.fillRect((i * w) / 12, 0, w / 12, h);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.lineWidth = 1;
  const rect = (x: number, y: number, rw: number, rh: number) =>
    ctx.strokeRect(x * sx, y * sy, rw * sx, rh * sy);
  rect(0, 0, L, W);
  ctx.beginPath();
  ctx.moveTo(w / 2, 0);
  ctx.lineTo(w / 2, h);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(w / 2, h / 2, CCR * sx, CCR * sy, 0, 0, Math.PI * 2);
  ctx.stroke();
  // Penalty + goal boxes.
  rect(0, (W - PBW) / 2, PBL, PBW);
  rect(L - PBL, (W - PBW) / 2, PBL, PBW);
  rect(0, (W - GBW) / 2, GBL, GBW);
  rect(L - GBL, (W - GBW) / 2, GBL, GBW);
}

export function PitchCanvas({
  minimap,
  getTime,
  highlightPlayerId,
  className = "",
}: {
  minimap: MinimapFrame[] | null;
  getTime: () => number;
  highlightPlayerId?: number | null;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const { width: cssW } = canvas.getBoundingClientRect();
      const cssH = cssW * (W / L);
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(cssW * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawPitch(ctx, cssW, cssH);
      if (!minimap || minimap.length === 0) return;

      const row = floorRow(minimap, getTime(), (r) => r.t);
      if (!row) return;
      const sx = cssW / L;
      const sy = cssH / W;
      for (const p of row.players) {
        const highlighted = highlightPlayerId != null && p.player_id === highlightPlayerId;
        ctx.beginPath();
        ctx.arc(p.x * sx, p.y * sy, highlighted ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle = teamColor(p.team);
        ctx.globalAlpha = 0.4 + 0.6 * Math.min(1, p.confidence + 0.3);
        ctx.fill();
        ctx.globalAlpha = 1;
        if (highlighted) {
          ctx.strokeStyle = "#9BE532";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      if (row.ball) {
        ctx.beginPath();
        ctx.arc(row.ball.x * sx, row.ball.y * sy, 3, 0, Math.PI * 2);
        ctx.fillStyle = row.ball.interpolated ? "rgba(255,255,255,0.5)" : "#ffffff";
        ctx.fill();
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [minimap, getTime, highlightPlayerId]);

  return (
    <canvas
      ref={canvasRef}
      className={`w-full rounded-lg border border-white/8 ${className}`}
      style={{ aspectRatio: `${L} / ${W}` }}
    />
  );
}
