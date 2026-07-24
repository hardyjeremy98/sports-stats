// Client-side player crops: one hidden <video> element seeks through the
// requested (frame_idx, box) list sequentially and draws each box region to a
// canvas, yielding data-URL crops. No server-side crop images exist for these
// — the run-directory contract stays JSON + the original video.

import { useEffect, useRef, useState } from "react";
import type { Box } from "./types";

export interface CropRequest {
  key: string; // stable identity, e.g. `t12-p0-f345`
  frame_idx: number;
  box: Box;
}

const PAD = 0.15; // context padding around the box, fraction of box size

export function useFrameCrops(
  videoUrl: string | null,
  fps: number,
  requests: CropRequest[],
): Record<string, string> {
  const [crops, setCrops] = useState<Record<string, string>>({});
  const cacheRef = useRef<Record<string, string>>({});

  // A stable signature so re-renders with identical requests don't reseek.
  const signature = requests.map((r) => r.key).join("|");

  useEffect(() => {
    if (!videoUrl || requests.length === 0) return;
    const pending = requests.filter((r) => !(r.key in cacheRef.current));
    if (pending.length === 0) {
      setCrops({ ...cacheRef.current });
      return;
    }

    let cancelled = false;
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.crossOrigin = "anonymous";
    video.src = videoUrl;

    const canvas = document.createElement("canvas");

    const cropOne = (req: CropRequest) =>
      new Promise<void>((resolve) => {
        const onSeeked = () => {
          video.removeEventListener("seeked", onSeeked);
          if (cancelled) return resolve();
          const ctx = canvas.getContext("2d");
          if (!ctx) return resolve();
          const bw = req.box.x2 - req.box.x1;
          const bh = req.box.y2 - req.box.y1;
          const px = bw * PAD;
          const py = bh * PAD;
          const sx = Math.max(0, req.box.x1 - px);
          const sy = Math.max(0, req.box.y1 - py);
          const sw = Math.min(video.videoWidth - sx, bw + 2 * px);
          const sh = Math.min(video.videoHeight - sy, bh + 2 * py);
          if (sw <= 0 || sh <= 0) return resolve();
          canvas.width = Math.round(sw);
          canvas.height = Math.round(sh);
          ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
          try {
            cacheRef.current[req.key] = canvas.toDataURL("image/jpeg", 0.85);
          } catch {
            // tainted canvas (cross-origin without CORS) — leave the crop out
          }
          resolve();
        };
        video.addEventListener("seeked", onSeeked);
        video.currentTime = (req.frame_idx + 0.5) / fps;
      });

    const run = async () => {
      await new Promise<void>((resolve) => {
        if (video.readyState >= 1) return resolve();
        video.addEventListener("loadedmetadata", () => resolve(), { once: true });
        video.addEventListener("error", () => resolve(), { once: true });
      });
      if (cancelled || video.videoWidth === 0) return;
      for (const req of pending) {
        if (cancelled) break;
        await cropOne(req);
        if (!cancelled) setCrops({ ...cacheRef.current });
      }
    };
    void run();

    return () => {
      cancelled = true;
      video.removeAttribute("src");
      video.load();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoUrl, fps, signature]);

  return crops;
}
