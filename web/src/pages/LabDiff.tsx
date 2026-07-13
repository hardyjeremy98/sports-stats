// Side-by-side run comparison: config deltas, metric deltas, stats, and
// aligned timeline strips — the tracker-A/B instrument.

import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useRunDiff } from "../lib/hooks";
import { fmtClock } from "../lib/format";
import type { RunDetail, StatSheet, TimelineBucket } from "../lib/types";
import { StatTable } from "./Report";
import { SignalPicker, TimelineStrip, type SignalId } from "../components/TimelineStrip";
import { Card, ErrorNote, Mono, PageTitle, Spinner, StatusChip } from "../components/ui";

export default function LabDiff() {
  const { a = "", b = "" } = useParams();
  const diff = useRunDiff(a, b);
  const [signal, setSignal] = useState<SignalId>("tracking_stability");

  if (diff.isLoading) return <Spinner label="Computing diff" />;
  if (diff.isError) return <ErrorNote>{(diff.error as Error).message}</ErrorNote>;
  const d = diff.data!;
  const duration = Math.max(
    d.run_a.video?.duration_s ?? 0,
    d.run_b.video?.duration_s ?? 0,
    0.01,
  );

  return (
    <div>
      <PageTitle
        title="Run diff"
        sub="Two configs, one clip — where do the pipelines disagree?"
      />

      <div className="mb-6 grid gap-4 md:grid-cols-2">
        <RunHeader tag="A" run={d.run_a} accent="text-volt-300" />
        <RunHeader tag="B" run={d.run_b} accent="text-team-home" />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-4">
          <SectionLabel>Config changes</SectionLabel>
          {d.config_changes.length === 0 ? (
            <div className="py-4 text-[13px] text-ink-500">Identical configs.</div>
          ) : (
            <table className="mt-2 w-full text-[13px]">
              <thead>
                <tr className="border-b border-white/8 text-left font-mono text-[10px] uppercase tracking-wider text-ink-500">
                  <th className="py-2 pr-2 font-normal">Path</th>
                  <th className="px-2 py-2 font-normal">A</th>
                  <th className="pl-2 py-2 font-normal">B</th>
                </tr>
              </thead>
              <tbody>
                {d.config_changes.map((c) => (
                  <tr key={c.path} className="border-b border-white/5 last:border-0">
                    <td className="py-1.5 pr-2 font-mono text-[11px] text-ink-400">{c.path}</td>
                    <td className="px-2 py-1.5 font-mono text-[12px] text-volt-300">
                      {fmtVal(c.a)}
                    </td>
                    <td className="py-1.5 pl-2 font-mono text-[12px] text-team-home">
                      {fmtVal(c.b)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <Card className="p-4">
          <SectionLabel>Metric deltas</SectionLabel>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
            {Object.entries(d.metric_deltas).map(([k, v]) => (
              <div key={k} className="rounded-lg border border-white/8 p-3">
                <div className="font-mono text-[10px] uppercase tracking-wider text-ink-500">
                  {k.replace("n_", "")}
                </div>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="font-mono text-lg">{String(v.a ?? "—")}</span>
                  <span className="text-ink-500">→</span>
                  <span className="font-mono text-lg">{String(v.b ?? "—")}</span>
                  {v.delta != null && v.delta !== 0 && (
                    <span
                      className={`ml-auto font-mono text-[12px] ${
                        v.delta > 0 ? "text-volt-400" : "text-team-away"
                      }`}
                    >
                      {v.delta > 0 ? "▲" : "▼"} {Math.abs(v.delta)}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card className="mt-6 p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionLabel>Timelines (aligned)</SectionLabel>
          <SignalPicker value={signal} onChange={setSignal} />
        </div>
        <TimelineRow tag="A" timeline={d.timeline_a} duration={duration} signal={signal} />
        <div className="h-3" />
        <TimelineRow tag="B" timeline={d.timeline_b} duration={duration} signal={signal} />
        <div className="mt-2 flex justify-between font-mono text-[10px] text-ink-500">
          <span>0:00</span>
          <span>{fmtClock(duration)}</span>
        </div>
      </Card>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <StatsCard tag="A" stats={d.stats_a} />
        <StatsCard tag="B" stats={d.stats_b} />
      </div>
    </div>
  );
}

function RunHeader({ tag, run, accent }: { tag: string; run: RunDetail; accent: string }) {
  return (
    <Card className="flex items-center gap-4 p-4">
      <span className={`font-mono text-2xl font-bold ${accent}`}>{tag}</span>
      <div className="min-w-0 flex-1">
        <Link to={`/lab/runs/${run.id}`}>
          <Mono className={`${accent} hover:underline`}>{run.id}</Mono>
        </Link>
        <div className="truncate text-[12px] text-ink-400">
          {run.config_name}
          {run.label ? ` · ${run.label}` : ""}
        </div>
      </div>
      <StatusChip status={run.status} />
    </Card>
  );
}

function TimelineRow({
  tag,
  timeline,
  duration,
  signal,
}: {
  tag: string;
  timeline: TimelineBucket[] | null;
  duration: number;
  signal: SignalId;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-4 shrink-0 font-mono text-[12px] text-ink-500">{tag}</span>
      <div className="min-w-0 flex-1">
        <TimelineStrip
          timeline={timeline}
          events={null}
          duration={duration}
          signal={signal}
          onSeek={() => {}}
        />
      </div>
    </div>
  );
}

function StatsCard({ tag, stats }: { tag: string; stats: StatSheet | null }) {
  return (
    <Card className="p-4">
      <SectionLabel>Stats — run {tag}</SectionLabel>
      <div className="mt-2">
        {stats ? (
          <StatTable players={stats.players} />
        ) : (
          <div className="py-4 text-[13px] text-ink-500">No stats artifact.</div>
        )}
      </div>
    </Card>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">{children}</div>
  );
}

function fmtVal(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
