// The confidence ribbon: per-second heat cells for a selectable pipeline
// signal, an event marker row, and a click-to-seek cursor. The Lab's main
// scrub-guidance instrument.

import { useEffect, useRef, useState } from "react";
import { signalColor } from "../lib/colors";
import { fmtClock, fmtConf } from "../lib/format";
import type { MatchEvent, SpottedEvent, TimelineBucket } from "../lib/types";

export const SIGNALS = [
  { id: "detection_confidence", label: "Detection" },
  { id: "tracking_stability", label: "Tracking" },
  { id: "calibration_confidence", label: "Calibration" },
  { id: "identity_coverage", label: "Identity" },
] as const;

export type SignalId = (typeof SIGNALS)[number]["id"];

export function SignalPicker({
  value,
  onChange,
}: {
  value: SignalId;
  onChange: (s: SignalId) => void;
}) {
  return (
    <div className="flex items-center gap-1">
      {SIGNALS.map((s) => (
        <button
          key={s.id}
          onClick={() => onChange(s.id)}
          className={`rounded-md px-2.5 py-1 text-[12px] transition-colors ${
            value === s.id
              ? "bg-turf-800 text-ink-100"
              : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
          }`}
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}

export function TimelineStrip({
  timeline,
  events,
  spotting,
  duration,
  signal,
  onSeek,
  getTime,
  markers,
}: {
  timeline: TimelineBucket[] | null;
  events: MatchEvent[] | null;
  spotting?: SpottedEvent[] | null;
  duration: number;
  signal: SignalId;
  onSeek: (t: number) => void;
  getTime?: () => number;
  markers?: { t: number; color: string; title: string; onClick?: () => void }[];
}) {
  const cursorRef = useRef<HTMLDivElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<TimelineBucket | null>(null);

  // Cursor tracks the time source without React re-renders.
  useEffect(() => {
    if (!getTime) return;
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const el = cursorRef.current;
      if (!el || duration <= 0) return;
      el.style.left = `${(Math.min(getTime(), duration) / duration) * 100}%`;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [getTime, duration]);

  const seekFromMouse = (e: React.MouseEvent) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect || duration <= 0) return;
    onSeek(((e.clientX - rect.left) / rect.width) * duration);
  };

  return (
    <div>
      <div
        ref={wrapRef}
        className="relative cursor-pointer select-none"
        onClick={seekFromMouse}
        onMouseLeave={() => setHover(null)}
      >
        {/* Heat cells */}
        <div className="flex h-7 overflow-hidden rounded-md border border-white/8">
          {(timeline ?? []).map((b) => {
            const v = b[signal] as number;
            const flagged = b.flags.length > 0;
            return (
              <div
                key={b.t}
                className="h-full flex-1"
                style={{
                  background: signalColor(v),
                  opacity: flagged ? 1 : 0.75,
                  boxShadow: flagged ? "inset 0 -3px 0 rgba(231,80,60,0.9)" : undefined,
                }}
                onMouseEnter={() => setHover(b)}
              />
            );
          })}
          {(!timeline || timeline.length === 0) && (
            <div className="flex h-full w-full items-center justify-center bg-turf-900 text-[11px] text-ink-500">
              no timeline artifact
            </div>
          )}
        </div>

        {/* Event markers */}
        <div className="relative mt-1 h-3">
          {(events ?? []).map((ev) => (
            <div
              key={ev.event_id}
              title={`${ev.type} @ ${fmtClock(ev.t)}`}
              className="absolute top-0 h-3 w-[3px] rounded-sm"
              style={{
                left: `${(ev.t / Math.max(duration, 0.01)) * 100}%`,
                background: ev.contested ? "#F5C518" : "rgba(232,237,233,0.55)",
              }}
              onClick={(e) => {
                e.stopPropagation();
                onSeek(ev.t);
              }}
            />
          ))}
        </div>

        {/* Spotting markers (spotter-detected actions, e.g. T-DEED) */}
        {spotting && spotting.length > 0 && (
          <div className="relative mt-1 h-3">
            {spotting.map((ev, i) => (
              <div
                key={i}
                title={`${ev.class} · ${fmtConf(ev.confidence)} @ ${fmtClock(ev.t)}`}
                className="absolute top-0 h-3 w-[3px] rounded-sm bg-team-home"
                style={{
                  left: `${(ev.t / Math.max(duration, 0.01)) * 100}%`,
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek(ev.t);
                }}
              />
            ))}
          </div>
        )}

        {/* Eval failure markers */}
        {markers && markers.length > 0 && (
          <div className="relative mt-1 h-3">
            {markers.map((m, i) => (
              <div
                key={i}
                title={m.title}
                className="absolute top-0 h-3 w-[3px] rounded-sm"
                style={{
                  left: `${(m.t / Math.max(duration, 0.01)) * 100}%`,
                  background: m.color,
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  m.onClick?.();
                }}
              />
            ))}
          </div>
        )}

        {/* Cursor */}
        {getTime && (
          <div
            ref={cursorRef}
            className="pointer-events-none absolute -top-1 bottom-0 w-px bg-volt-400"
            style={{ left: 0 }}
          >
            <div className="absolute -left-[3px] -top-1 h-[7px] w-[7px] rounded-full bg-volt-400" />
          </div>
        )}
      </div>

      <div className="mt-1.5 flex items-center justify-between font-mono text-[10px] text-ink-500">
        <span>0:00</span>
        {hover && (
          <span className="text-ink-400">
            {fmtClock(hover.t)} · {SIGNALS.find((s) => s.id === signal)?.label}{" "}
            {(hover[signal] as number).toFixed(2)}
            {hover.flags.length > 0 && (
              <span className="text-team-ref"> · {hover.flags.join(", ")}</span>
            )}
          </span>
        )}
        <span>{fmtClock(duration)}</span>
      </div>
    </div>
  );
}
