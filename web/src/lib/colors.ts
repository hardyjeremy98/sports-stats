import type { Team } from "./types";

// Single source of truth for team colors — keep close to the backend
// annotator palette (matchlab_core/stages/annotate/overlay.py).
export const TEAM_COLORS: Record<Team, string> = {
  home: "#2C91FF",
  away: "#E7503C",
  referee: "#F5C518",
  unknown: "#8B949E",
};

export const ACCENT = "#9BE532"; // volt

export function teamColor(team: Team | undefined | null): string {
  return TEAM_COLORS[team ?? "unknown"];
}

// Stable, distinct-ish color per tracklet id (golden-angle hue walk).
export function trackletColor(id: number): string {
  const hue = (id * 137.508) % 360;
  return `hsl(${hue.toFixed(0)} 75% 60%)`;
}

// Confidence ramp for the timeline: red (bad) -> amber -> volt (good).
export function signalColor(v: number): string {
  const clamped = Math.max(0, Math.min(1, v));
  if (clamped < 0.5) {
    // red -> amber
    const t = clamped / 0.5;
    return mix("#E7503C", "#F5C518", t);
  }
  const t = (clamped - 0.5) / 0.5;
  return mix("#F5C518", "#9BE532", t);
}

function mix(a: string, b: string, t: number): string {
  const pa = hex(a);
  const pb = hex(b);
  const c = pa.map((x, i) => Math.round(x + (pb[i] - x) * t));
  return `rgb(${c[0]} ${c[1]} ${c[2]})`;
}

function hex(h: string): number[] {
  return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
}
