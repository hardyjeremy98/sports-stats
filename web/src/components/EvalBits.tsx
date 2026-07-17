// Shared rendering for a single eval ID-switch instance: level pill, clock,
// GT label, and prev->new predicted ids. Used by LabRunViewer's eval tab and
// the LabDiff identity-changes card so both stay visually identical.

import { fmtClock } from "../lib/format";
import type { AttributionLayer, EvalInstance, EvalInstanceAttribution } from "../lib/types";
import { Mono } from "./ui";

// SPO-19 layer attribution: label + pill styling per layer. "ambiguous" is
// deliberately neutral — it is an honest outcome, not an error state.
export const LAYER_LABEL: Record<AttributionLayer, string> = {
  detection: "detection",
  online_association: "online assoc",
  refinement: "refinement",
  offline_association: "offline assoc",
  ambiguous: "ambiguous",
};

const LAYER_CLASS: Record<AttributionLayer, string> = {
  detection: "bg-team-away/15 text-team-away",
  online_association: "bg-volt-400/15 text-volt-300",
  refinement: "bg-white/10 text-ink-400",
  offline_association: "bg-team-ref/15 text-team-ref",
  ambiguous: "bg-turf-800 text-ink-400",
};

export function AttributionPill({ attribution }: { attribution?: EvalInstanceAttribution }) {
  if (!attribution) {
    return (
      <span
        className="rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-500"
        title="This eval predates layer attribution — re-evaluate the run to attribute its switches."
      >
        unattributed
      </span>
    );
  }
  const tooltip = attribution.evidence
    .map((e) => e.detail ?? `${e.kind}${e.outcome ? `: ${e.outcome}` : ""}`)
    .join("; ");
  return (
    <span
      className={`rounded-full px-2 py-0.5 font-mono text-[10px] ${LAYER_CLASS[attribution.layer]}`}
      title={tooltip}
    >
      {LAYER_LABEL[attribution.layer]}
    </span>
  );
}

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
      <AttributionPill attribution={inst.attribution} />
      <span className="text-[12px] text-ink-100">GT {inst.gt_label}</span>
      <span className="ml-auto font-mono text-[11px] text-ink-500">
        {fmtPredId(inst.level, inst.prev_id)} → {fmtPredId(inst.level, inst.new_id)}
      </span>
    </button>
  );
}
