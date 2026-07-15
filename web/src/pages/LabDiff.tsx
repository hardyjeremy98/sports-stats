// Side-by-side run comparison: config deltas, metric deltas, stats, two
// synchronized VideoOverlays, seekable timelines with eval/switch-diff
// markers, and an identity-changes browser — the tracker-A/B instrument.

import { useEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { useEvaluateRun, useRunDiff } from "../lib/hooks";
import { useGroundTruth, useRunArtifacts } from "../lib/artifacts";
import { fmtClock } from "../lib/format";
import type { EvalInstance, EvalResult, RunDetail, StatSheet, TimelineBucket } from "../lib/types";
import { StatTable } from "./Report";
import { SignalPicker, TimelineStrip, type SignalId } from "../components/TimelineStrip";
import { SwitchInstanceRow } from "../components/EvalBits";
import {
  DEFAULT_LAYERS,
  LayerChips,
  VideoOverlay,
  type LayerState,
  type VideoOverlayHandle,
} from "../components/VideoOverlay";
import { Button, Card, ErrorNote, Mono, PageTitle, Spinner, StatusChip } from "../components/ui";

type Marker = { t: number; color: string; title: string; onClick?: () => void };

export default function LabDiff() {
  const { a = "", b = "" } = useParams();
  const diff = useRunDiff(a, b);
  const qc = useQueryClient();
  const invalidateDiff = () => qc.invalidateQueries({ queryKey: ["diff", a, b] });

  const artifactsA = useRunArtifacts(a, true);
  const artifactsB = useRunArtifacts(b, true);
  const videoId = diff.data?.run_a.video_id;
  const hasGt = !!diff.data?.run_a.video?.has_ground_truth;
  const gt = useGroundTruth(videoId, hasGt);

  const reEvaluateA = useEvaluateRun(a);
  const reEvaluateB = useEvaluateRun(b);

  const [signal, setSignal] = useState<SignalId>("tracking_stability");
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);

  const aRef = useRef<VideoOverlayHandle>(null);
  const bRef = useRef<VideoOverlayHandle>(null);

  // A is the master: its play/pause/seeked events drive B, plus a rAF loop
  // nudges B back in sync if it drifts. B is never listened to, so there's
  // no feedback loop.
  useEffect(() => {
    const va = aRef.current?.video;
    const vb = bRef.current?.video;
    if (!va || !vb) return;

    const onPlay = () => {
      vb.play().catch(() => {});
    };
    const onPause = () => vb.pause();
    const onSeeked = () => {
      vb.currentTime = va.currentTime;
    };
    va.addEventListener("play", onPlay);
    va.addEventListener("pause", onPause);
    va.addEventListener("seeked", onSeeked);

    let raf = 0;
    const correct = () => {
      raf = requestAnimationFrame(correct);
      if (Math.abs(vb.currentTime - va.currentTime) > 0.15) {
        vb.currentTime = va.currentTime;
      }
    };
    raf = requestAnimationFrame(correct);

    return () => {
      va.removeEventListener("play", onPlay);
      va.removeEventListener("pause", onPause);
      va.removeEventListener("seeked", onSeeked);
      cancelAnimationFrame(raf);
    };
  }, [diff.data]);

  const seek = useMemo(() => (t: number) => aRef.current?.seek(t), []);
  const getTime = useMemo(() => () => aRef.current?.getTime() ?? 0, []);

  const markersA = useMemo(() => {
    const own = evalMarkers(artifactsA.eval, seek);
    const fixed = (diff.data?.switch_diff?.fixed ?? []).map((inst) =>
      switchMarker(inst, "#9BE532", "fixed", seek),
    );
    return [...own, ...fixed];
  }, [artifactsA.eval, diff.data?.switch_diff, seek]);

  const markersB = useMemo(() => {
    const own = evalMarkers(artifactsB.eval, seek);
    const introduced = (diff.data?.switch_diff?.introduced ?? []).map((inst) =>
      switchMarker(inst, "#E7503C", "introduced", seek),
    );
    return [...own, ...introduced];
  }, [artifactsB.eval, diff.data?.switch_diff, seek]);

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

      <Card className="mb-6 p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionLabel>Synchronized playback</SectionLabel>
          <span className="text-[11px] text-ink-500">A drives B · scrub A or the timelines</span>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="flex flex-col gap-2">
            <Mono className="text-volt-300">A · {d.run_a.id}</Mono>
            <VideoOverlay ref={aRef} run={d.run_a} artifacts={artifactsA} layers={layers} gt={gt} />
          </div>
          <div className="flex flex-col gap-2">
            <Mono className="text-team-home">B · {d.run_b.id}</Mono>
            <VideoOverlay
              ref={bRef}
              run={d.run_b}
              artifacts={artifactsB}
              layers={layers}
              gt={gt}
              controls={false}
            />
          </div>
        </div>
        <div className="mt-3">
          <LayerChips layers={layers} onChange={setLayers} artifacts={artifactsA} gt={gt} />
        </div>
      </Card>

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

      {d.eval_a && d.eval_b && (
        <Card className="mt-6 p-4">
          <SectionLabel>Identity metrics</SectionLabel>
          <IdentityMetricsTable evalA={d.eval_a} evalB={d.eval_b} />
        </Card>
      )}

      <Card className="mt-6 p-4">
        <SectionLabel>Identity changes</SectionLabel>
        {!d.switch_diff ? (
          <div className="flex flex-col items-center gap-3 py-6 text-center text-[13px] text-ink-500">
            <div>Evaluate both runs first.</div>
            <div className="flex items-center gap-2">
              <Button
                onClick={() => reEvaluateA.mutate(undefined, { onSuccess: invalidateDiff })}
                disabled={reEvaluateA.isPending}
              >
                {reEvaluateA.isPending ? "Evaluating A…" : "Re-evaluate A"}
              </Button>
              <Button
                onClick={() => reEvaluateB.mutate(undefined, { onSuccess: invalidateDiff })}
                disabled={reEvaluateB.isPending}
              >
                {reEvaluateB.isPending ? "Evaluating B…" : "Re-evaluate B"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="mt-3 grid gap-4 lg:grid-cols-3">
            <SwitchSection
              title="Fixed"
              count={d.switch_diff.counts.fixed}
              instances={d.switch_diff.fixed}
              onSeek={seek}
              emptyLabel="No switches fixed."
            />
            <SwitchSection
              title="Introduced"
              count={d.switch_diff.counts.introduced}
              instances={d.switch_diff.introduced}
              onSeek={seek}
              emptyLabel="No new switches."
            />
            <SwitchSection
              title="Persisted"
              count={d.switch_diff.counts.persisted}
              instances={d.switch_diff.persisted.map((p) => p.a)}
              onSeek={seek}
              emptyLabel="No persisted switches."
            />
          </div>
        )}
      </Card>

      <Card className="mt-6 p-4">
        <div className="mb-3 flex items-center justify-between">
          <SectionLabel>Timelines (synchronized)</SectionLabel>
          <SignalPicker value={signal} onChange={setSignal} />
        </div>
        <TimelineRow
          tag="A"
          timeline={d.timeline_a}
          duration={duration}
          signal={signal}
          onSeek={seek}
          getTime={getTime}
          markers={markersA}
        />
        <div className="h-3" />
        <TimelineRow
          tag="B"
          timeline={d.timeline_b}
          duration={duration}
          signal={signal}
          onSeek={seek}
          getTime={getTime}
          markers={markersB}
        />
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

/* ---------- eval helpers ---------- */

function evalMarkers(ev: EvalResult | null, seek: (t: number) => void): Marker[] {
  if (!ev || ev.instances.length === 0) return [];
  return ev.instances.map((inst) => ({
    t: Math.max(0, inst.t),
    color: inst.level === "entity" ? "#F5C518" : "#8B949E",
    title: `${inst.level} switch @ ${fmtClock(inst.t)} GT ${inst.gt_label}`,
    onClick: () => seek(Math.max(0, inst.t - 1)),
  }));
}

function switchMarker(
  inst: EvalInstance,
  color: string,
  kind: string,
  seek: (t: number) => void,
): Marker {
  return {
    t: Math.max(0, inst.t),
    color,
    title: `${kind} ${inst.level} switch @ ${fmtClock(inst.t)} GT ${inst.gt_label}`,
    onClick: () => seek(Math.max(0, inst.t - 1)),
  };
}

const IDENTITY_METRIC_ROWS: {
  key: string;
  label: string;
  get: (ev: EvalResult) => number;
  higherIsBetter: boolean;
  ratio?: boolean;
}[] = [
  {
    key: "idf1_tracklet",
    label: "IDF1 (tracklet)",
    get: (ev) => ev.levels.tracklet.idf1,
    higherIsBetter: true,
    ratio: true,
  },
  {
    key: "idf1_entity",
    label: "IDF1 (entity)",
    get: (ev) => ev.levels.entity.idf1,
    higherIsBetter: true,
    ratio: true,
  },
  {
    key: "idsw_tracklet",
    label: "IDSW (tracklet)",
    get: (ev) => ev.levels.tracklet.num_switches,
    higherIsBetter: false,
  },
  {
    key: "idsw_entity",
    label: "IDSW (entity)",
    get: (ev) => ev.levels.entity.num_switches,
    higherIsBetter: false,
  },
  {
    key: "mota_entity",
    label: "MOTA (entity)",
    get: (ev) => ev.levels.entity.mota,
    higherIsBetter: true,
    ratio: true,
  },
  {
    key: "assoc_gain",
    label: "Assoc IDF1 gain",
    get: (ev) => ev.association.idf1_gain,
    higherIsBetter: true,
    ratio: true,
  },
];

function IdentityMetricsTable({ evalA, evalB }: { evalA: EvalResult; evalB: EvalResult }) {
  const fmt = (v: number, ratio?: boolean) => (ratio ? v.toFixed(3) : String(Math.round(v)));
  return (
    <table className="mt-2 w-full text-[13px]">
      <thead>
        <tr className="border-b border-white/8 text-left font-mono text-[10px] uppercase tracking-wider text-ink-500">
          <th className="py-1.5 font-normal">Metric</th>
          <th className="py-1.5 text-right font-normal text-volt-300">A</th>
          <th className="py-1.5 text-right font-normal text-team-home">B</th>
          <th className="py-1.5 pl-3 text-right font-normal">Δ</th>
        </tr>
      </thead>
      <tbody>
        {IDENTITY_METRIC_ROWS.map((row) => {
          const a = row.get(evalA);
          const b = row.get(evalB);
          const delta = b - a;
          const good = row.higherIsBetter ? delta > 0 : delta < 0;
          const bad = row.higherIsBetter ? delta < 0 : delta > 0;
          return (
            <tr key={row.key} className="border-b border-white/5 last:border-0">
              <td className="py-1.5 text-ink-400">{row.label}</td>
              <td className="py-1.5 text-right font-mono text-ink-100">{fmt(a, row.ratio)}</td>
              <td className="py-1.5 text-right font-mono text-ink-100">{fmt(b, row.ratio)}</td>
              <td
                className={`py-1.5 pl-3 text-right font-mono text-[12px] ${
                  good ? "text-volt-400" : bad ? "text-team-away" : "text-ink-500"
                }`}
              >
                {Math.abs(delta) > (row.ratio ? 0.0005 : 0.5)
                  ? `${delta > 0 ? "+" : ""}${fmt(delta, row.ratio)}`
                  : "—"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* ---------- identity changes ---------- */

function SwitchSection({
  title,
  count,
  instances,
  onSeek,
  emptyLabel,
}: {
  title: string;
  count: number;
  instances: EvalInstance[];
  onSeek: (t: number) => void;
  emptyLabel: string;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-ink-500">{title}</span>
        <span className="font-mono text-[11px] text-ink-500">{count}</span>
      </div>
      {instances.length === 0 ? (
        <div className="py-2 text-center text-[12px] text-ink-500">{emptyLabel}</div>
      ) : (
        <div className="flex flex-col gap-1">
          {instances.map((inst, i) => (
            <SwitchInstanceRow
              key={i}
              inst={inst}
              onClick={() => onSeek(Math.max(0, inst.t - 1))}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------- headers, timelines, stats ---------- */

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
  onSeek,
  getTime,
  markers,
}: {
  tag: string;
  timeline: TimelineBucket[] | null;
  duration: number;
  signal: SignalId;
  onSeek: (t: number) => void;
  getTime?: () => number;
  markers?: Marker[];
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
          onSeek={onSeek}
          getTime={getTime}
          markers={markers}
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
