// Shared rendering for a single eval ID-switch instance: level pill, clock,
// GT label, and prev->new predicted ids. Used by LabRunViewer's eval tab and
// the LabDiff identity-changes card so both stay visually identical.

import { fmtClock } from "../lib/format";
import type { EvalInstance } from "../lib/types";
import { Mono } from "./ui";

export function fmtPredId(level: "tracklet" | "entity", id: number | null): string {
  if (id == null) return "?";
  // Entities ≥ 100000 are unassociated tracklets given a synthetic identity.
  if (level === "entity") return id >= 100000 ? `T${id - 100000} (unassoc)` : `#${id}`;
  return `T${id}`;
}

export function SwitchInstanceRow({
  inst,
  onClick,
}: {
  inst: EvalInstance;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2.5 rounded-lg border border-transparent px-3 py-2 text-left transition-colors hover:bg-turf-800"
    >
      <Mono className="w-12 shrink-0 text-volt-300">{fmtClock(inst.t)}</Mono>
      <span
        className={`rounded-full px-2 py-0.5 font-mono text-[10px] ${
          inst.level === "tracklet" ? "bg-turf-800 text-ink-400" : "bg-team-ref/15 text-team-ref"
        }`}
      >
        {inst.level}
      </span>
      <span className="text-[12px] text-ink-100">GT {inst.gt_label}</span>
      <span className="ml-auto font-mono text-[11px] text-ink-500">
        {fmtPredId(inst.level, inst.prev_id)} → {fmtPredId(inst.level, inst.new_id)}
      </span>
    </button>
  );
}
