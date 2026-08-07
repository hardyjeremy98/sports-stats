// Inspector for pcbas_events.json: the two-stage player-centric action spotter's
// output next to the ground truth it was scored against.
//
// The list is one row per MATCHER DECISION, not per prediction, so a miss is as
// visible as a hit -- a panel showing only predictions would make a model that
// finds nothing look flawless.

import { useMemo, useState } from "react";
import type { PCBASLabEvent, PCBASLabEvents, PCBASVerdict } from "../lib/types";

const VERDICT_STYLE: Record<PCBASVerdict, { label: string; cls: string; dot: string }> = {
  tp: { label: "Hit", cls: "text-emerald-300", dot: "bg-emerald-400" },
  fp: { label: "False alarm", cls: "text-red-300", dot: "bg-red-400" },
  fn: { label: "Missed", cls: "text-amber-300", dot: "bg-amber-400" },
};

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

export function PCBASPanel({
  data,
  onSeek,
}: {
  data: PCBASLabEvents;
  onSeek: (t: number) => void;
}) {
  const [verdicts, setVerdicts] = useState<Set<PCBASVerdict>>(
    () => new Set<PCBASVerdict>(["tp", "fp", "fn"]),
  );
  const [cls, setCls] = useState<string>("all");
  const [offscreenOnly, setOffscreenOnly] = useState(false);

  const classNames = useMemo(
    () => [...new Set(data.events.map((e) => e.class_name))].sort(),
    [data.events],
  );

  const rows = useMemo(
    () =>
      data.events.filter(
        (e) =>
          verdicts.has(e.verdict) &&
          (cls === "all" || e.class_name === cls) &&
          (!offscreenOnly || e.has_bbox === false),
      ),
    [data.events, verdicts, cls, offscreenOnly],
  );

  const toggle = (v: PCBASVerdict) => {
    const next = new Set(verdicts);
    if (next.has(v)) next.delete(v);
    else next.add(v);
    setVerdicts(next);
  };

  const r = data.report;
  const counts: Record<PCBASVerdict, number> = { tp: r.tp, fp: r.fp, fn: r.fn };

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-white/8 p-3">
        <div className="mb-2 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[11px]">
          <span className="font-mono uppercase tracking-[0.18em] text-ink-500">
            {data.key}
          </span>
          <span className="text-ink-300">
            micro-F1 <span className="font-mono text-ink-100">{r.micro_f1.toFixed(4)}</span>
          </span>
          <span className="text-ink-300">
            macro-F1 <span className="font-mono text-ink-100">{r.macro_f1.toFixed(4)}</span>
          </span>
          <span className="text-ink-300">
            P <span className="font-mono text-ink-100">{pct(r.precision)}</span>
          </span>
          <span className="text-ink-300">
            R <span className="font-mono text-ink-100">{pct(r.recall)}</span>
          </span>
        </div>
        <div className="mb-2 text-[10px] leading-relaxed text-ink-500">
          Matched on <span className="text-ink-300">{data.identity}</span> identity within{" "}
          <span className="text-ink-300">±{data.delta} frames</span>, predictions below{" "}
          <span className="text-ink-300">{data.conf_thresh}</span> confidence dropped before
          matching. Class, time <em>and</em> player must all agree for a hit.
        </div>
        {r.gt_without_bbox > 0 && (
          <div className="mb-2 rounded border border-white/8 bg-white/[0.02] px-2 py-1 text-[10px] text-ink-400">
            Recovered{" "}
            <span className="font-mono text-ink-100">
              {r.tp_without_bbox}/{r.gt_without_bbox}
            </span>{" "}
            actions whose player is off-screen — unreachable for any purely visual model.
          </div>
        )}
        <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
          {(["tp", "fp", "fn"] as PCBASVerdict[]).map((v) => (
            <button
              key={v}
              onClick={() => toggle(v)}
              className={`flex items-center gap-1.5 rounded border px-2 py-0.5 ${
                verdicts.has(v)
                  ? "border-white/20 text-ink-100"
                  : "border-white/8 text-ink-600"
              }`}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${VERDICT_STYLE[v].dot}`} />
              {VERDICT_STYLE[v].label}
              <span className="font-mono text-ink-500">{counts[v]}</span>
            </button>
          ))}
          <select
            value={cls}
            onChange={(e) => setCls(e.target.value)}
            className="rounded border border-white/8 bg-transparent px-1.5 py-0.5 text-ink-200"
          >
            <option value="all">all classes</option>
            {classNames.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <button
            onClick={() => setOffscreenOnly(!offscreenOnly)}
            className={`rounded border px-2 py-0.5 ${
              offscreenOnly ? "border-white/20 text-ink-100" : "border-white/8 text-ink-600"
            }`}
            title="Ground-truth actions whose player has no bounding box"
          >
            off-screen only
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <table className="w-full text-[11px]">
          <thead className="sticky top-0 bg-surface-900/95 text-ink-500">
            <tr className="text-left">
              <th className="px-2 py-1 font-normal">verdict</th>
              <th className="px-2 py-1 font-normal">time</th>
              <th className="px-2 py-1 font-normal">player</th>
              <th className="px-2 py-1 font-normal">action</th>
              <th className="px-2 py-1 text-right font-normal">conf</th>
              <th className="px-2 py-1 text-right font-normal">Δf</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e, i) => (
              <Row key={`${e.frame_idx}-${e.verdict}-${i}`} e={e} onSeek={onSeek} />
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <div className="p-4 text-center text-[11px] text-ink-600">
            No events match these filters.
          </div>
        )}
      </div>
      <div className="border-t border-white/8 px-3 py-1.5 text-[10px] text-ink-500">
        {rows.length} of {data.events.length} decisions · click a row to seek
      </div>
    </div>
  );
}

function Row({ e, onSeek }: { e: PCBASLabEvent; onSeek: (t: number) => void }) {
  const s = VERDICT_STYLE[e.verdict];
  const mins = Math.floor(e.t / 60);
  const secs = (e.t % 60).toFixed(1).padStart(4, "0");
  return (
    <tr
      onClick={() => onSeek(e.t)}
      className="cursor-pointer border-t border-white/5 hover:bg-white/[0.04]"
    >
      <td className={`px-2 py-1 ${s.cls}`}>
        <span className="flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
          {s.label}
        </span>
      </td>
      <td className="px-2 py-1 font-mono text-ink-300">
        {mins}:{secs}
      </td>
      <td className="px-2 py-1 font-mono text-ink-200">
        {e.shirt_number != null && e.shirt_number >= 0 ? `#${e.shirt_number}` : "—"}
        {e.has_bbox === false && (
          <span className="ml-1 text-ink-600" title="player off-screen in the ground truth">
            ⌀
          </span>
        )}
      </td>
      <td className="px-2 py-1 text-ink-200">{e.class_name}</td>
      <td className="px-2 py-1 text-right font-mono text-ink-400">
        {e.score != null ? e.score.toFixed(2) : "—"}
      </td>
      <td
        className="px-2 py-1 text-right font-mono text-ink-500"
        title={e.frame_error != null ? "frames between prediction and ground truth" : ""}
      >
        {e.frame_error != null ? (e.frame_error > 0 ? `+${e.frame_error}` : e.frame_error) : "—"}
      </td>
    </tr>
  );
}
