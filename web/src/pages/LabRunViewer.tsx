// The Lab's forensic run viewer: overlay video, confidence timeline, and an
// inspector (stages / tracklets / players / events / QA).

import { useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useGroundTruth, useRunArtifacts, type GtIndex, type RunArtifacts } from "../lib/artifacts";
import { trackletColor } from "../lib/colors";
import { eventLabel, fmtClock, fmtConf, fmtDuration } from "../lib/format";
import { useEvaluateRun, useQA, useQAActions, useRun, useRuns } from "../lib/hooks";
import type { EvalInstance, EvalLevelMetrics, QARecord, RunDetail } from "../lib/types";
import { EvidenceInspector } from "../components/EvidenceInspector";
import { PitchCanvas } from "../components/PitchCanvas";
import { SignalPicker, TimelineStrip, type SignalId } from "../components/TimelineStrip";
import {
  DEFAULT_LAYERS,
  LayerChips,
  VideoOverlay,
  type LayerState,
  type VideoOverlayHandle,
} from "../components/VideoOverlay";
import {
  Button,
  Card,
  ErrorNote,
  Mono,
  PageTitle,
  Spinner,
  StatusChip,
  Tabs,
  TeamDot,
} from "../components/ui";

type TabId = "stages" | "tracklets" | "players" | "events" | "eval" | "qa";

export default function LabRunViewer() {
  const { runId = "" } = useParams();
  const run = useRun(runId);
  const artifacts = useRunArtifacts(runId, run.data?.status === "completed");
  const gt = useGroundTruth(run.data?.video_id, !!run.data?.video?.has_ground_truth);
  const qa = useQA({ run_id: runId });
  const reEvaluate = useEvaluateRun(runId);

  const playerRef = useRef<VideoOverlayHandle>(null);
  const [layers, setLayers] = useState<LayerState>(DEFAULT_LAYERS);
  const [signal, setSignal] = useState<SignalId>("detection_confidence");
  const [tab, setTab] = useState<TabId>("stages");
  const [hlTracklet, setHlTracklet] = useState<number | null>(null);
  const [hlPlayer, setHlPlayer] = useState<number | null>(null);
  const [hlGtTrack, setHlGtTrack] = useState<number | null>(null);
  const [inspector, setInspector] = useState<{ playerId: number; evidenceIdx: number } | null>(
    null,
  );

  const getTime = useMemo(() => () => playerRef.current?.getTime() ?? 0, []);
  const seek = (t: number) => playerRef.current?.seek(t);

  const evalMarkers = useMemo(() => {
    const ev = artifacts.eval;
    if (!ev || ev.instances.length === 0) return undefined;
    return ev.instances.map((inst) => ({
      t: Math.max(0, inst.t),
      color: inst.level === "entity" ? "#F5C518" : "#8B949E",
      title: `${inst.level} switch @ ${fmtClock(inst.t)} GT ${inst.gt_label}`,
      onClick: () => seek(Math.max(0, inst.t - 1)),
    }));
  }, [artifacts.eval]);

  if (run.isLoading) return <Spinner label="Loading run" />;
  if (run.isError) return <ErrorNote>{(run.error as Error).message}</ErrorNote>;
  const r = run.data!;
  const duration = r.video?.duration_s ?? r.manifest?.video.duration_s ?? 0;
  const pendingQA = (qa.data ?? []).filter((q) => q.status === "pending").length;

  return (
    <div>
      <PageTitle
        title={
          <span className="flex items-center gap-3">
            <Mono className="text-base text-volt-300">{r.id}</Mono>
            <StatusChip status={r.status} />
          </span>
        }
        sub={
          <>
            {r.config_name}
            {r.label ? ` · ${r.label}` : ""} · {r.video?.filename ?? `video ${r.video_id}`}
          </>
        }
        right={<DiffPicker run={r} />}
      />

      {r.status === "failed" && <ErrorNote>{r.error}</ErrorNote>}
      {(r.status === "queued" || r.status === "running") && (
        <Card className="mb-6 p-5 text-sm text-ink-400">
          {r.progress_msg ?? "Waiting for a worker…"}
          <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-turf-800">
            <div
              className="h-full bg-volt-400 transition-all"
              style={{ width: `${Math.round(r.progress_frac * 100)}%` }}
            />
          </div>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
        {/* Left: player + timeline */}
        <div className="flex min-w-0 flex-col gap-4">
          <VideoOverlay
            ref={playerRef}
            run={r}
            artifacts={artifacts}
            layers={layers}
            gt={gt}
            highlightTrackletId={hlTracklet}
            highlightPlayerId={hlPlayer}
            highlightGtTrackId={hlGtTrack}
          />
          <LayerChips layers={layers} onChange={setLayers} artifacts={artifacts} gt={gt} />
          <Card className="p-4">
            <div className="mb-3 flex items-center justify-between">
              <SignalPicker value={signal} onChange={setSignal} />
              <span className="text-[11px] text-ink-500">
                red cells = low confidence · click to seek
              </span>
            </div>
            <TimelineStrip
              timeline={artifacts.timeline}
              events={artifacts.events}
              duration={duration}
              signal={signal}
              onSeek={seek}
              getTime={getTime}
              markers={evalMarkers}
            />
          </Card>
          <Card className="p-4">
            <div className="mb-2 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">
              Game state
            </div>
            <PitchCanvas
              minimap={artifacts.minimap}
              getTime={getTime}
              highlightPlayerId={hlPlayer}
            />
          </Card>
        </div>

        {/* Right: inspector */}
        <Card className="flex max-h-[calc(100vh-180px)] min-h-[500px] flex-col self-start overflow-hidden xl:sticky xl:top-20">
          <Tabs<TabId>
            tabs={[
              { id: "stages", label: "Stages" },
              { id: "tracklets", label: "Tracklets", count: artifacts.tracklets?.length },
              { id: "players", label: "Players", count: artifacts.players?.length },
              { id: "events", label: "Events", count: artifacts.events?.length },
              ...(artifacts.eval
                ? [{ id: "eval" as const, label: "Eval", count: artifacts.eval.instances.length }]
                : []),
              { id: "qa", label: "QA", count: pendingQA },
            ]}
            active={tab}
            onChange={setTab}
          />
          <div className="flex-1 overflow-y-auto p-3">
            {tab === "stages" && <StagesTab run={r} />}
            {tab === "tracklets" && (
              <TrackletsTab
                artifacts={artifacts}
                fps={r.video?.fps ?? 25}
                selected={hlTracklet}
                onSelect={(tid, t) => {
                  setHlTracklet(tid === hlTracklet ? null : tid);
                  setHlPlayer(null);
                  setHlGtTrack(null);
                  if (t != null) seek(t);
                }}
              />
            )}
            {tab === "players" && (
              <PlayersTab
                artifacts={artifacts}
                runId={r.id}
                fps={r.video?.fps || r.manifest?.video.fps || 25}
                selected={hlPlayer}
                onSelect={(pid, t) => {
                  setHlPlayer(pid === hlPlayer ? null : pid);
                  setHlTracklet(null);
                  setHlGtTrack(null);
                  if (t != null) seek(t);
                }}
                onEvidenceClick={(tid, pid, evidenceIdx) => {
                  setHlTracklet(tid);
                  setHlPlayer(pid);
                  setHlGtTrack(null);
                  setInspector({ playerId: pid, evidenceIdx });
                }}
              />
            )}
            {tab === "events" && <EventsTab artifacts={artifacts} onSeek={seek} />}
            {tab === "eval" && (
              <EvalTab
                artifacts={artifacts}
                gt={gt}
                hasGroundTruth={!!r.video?.has_ground_truth}
                onReEvaluate={() => reEvaluate.mutate()}
                reEvaluating={reEvaluate.isPending}
                onInstance={(inst) => {
                  seek(Math.max(0, inst.t - 1)); // land just before the switch
                  setLayers((l) => ({ ...l, gt: true, tracklets: true }));
                  setHlGtTrack(inst.gt_track_id);

                  const newId = inst.new_id;
                  if (newId == null) {
                    // no better identity known — leave existing highlights alone
                  } else if (newId >= 100000) {
                    // synthetic id for an unassociated tracklet — no real entity
                    setHlTracklet(newId - 100000);
                    setHlPlayer(null);
                  } else if (inst.level === "entity") {
                    setHlPlayer(newId);
                    // keep the tracklet highlight only if it still belongs to this entity
                    if (artifacts.entityByTracklet.get(hlTracklet ?? -1)?.player_id !== newId) {
                      setHlTracklet(null);
                    }
                  } else {
                    setHlTracklet(newId);
                    setHlPlayer(artifacts.entityByTracklet.get(newId)?.player_id ?? null);
                  }
                }}
              />
            )}
            {tab === "qa" && <QATab records={qa.data ?? []} onSeek={seek} />}
          </div>
        </Card>
      </div>

      {inspector &&
        (() => {
          const inspectorPlayer = artifacts.players?.find(
            (p) => p.player_id === inspector.playerId,
          );
          if (!inspectorPlayer) return null;
          return (
            <EvidenceInspector
              runId={r.id}
              player={inspectorPlayer}
              evidenceIdx={inspector.evidenceIdx}
              fps={r.video?.fps || r.manifest?.video.fps || 25}
              playerRef={playerRef}
              onClose={() => setInspector(null)}
              onSelectTracklet={(tid) => {
                setHlTracklet(tid);
                setHlPlayer(null);
                setHlGtTrack(null);
                setTab("tracklets");
                setInspector(null);
              }}
              onNavigate={(evidenceIdx, tid) => {
                setInspector({ playerId: inspector.playerId, evidenceIdx });
                setHlTracklet(tid);
                setHlPlayer(inspector.playerId);
              }}
            />
          );
        })()}
    </div>
  );
}

/* ---------- header: diff picker ---------- */

function DiffPicker({ run }: { run: RunDetail }) {
  const runs = useRuns(run.video_id);
  const navigate = useNavigate();
  const others = (runs.data ?? []).filter((x) => x.id !== run.id && x.status === "completed");
  if (run.status !== "completed" || others.length === 0) return null;
  return (
    <select
      defaultValue=""
      onChange={(e) => {
        if (e.target.value) navigate(`/lab/diff/${run.id}/${e.target.value}`);
      }}
      className="rounded-lg border border-white/10 bg-turf-850 px-3 py-1.5 text-[13px] text-ink-400 outline-none focus:border-volt-400/60"
    >
      <option value="" disabled>
        Diff against…
      </option>
      {others.map((o) => (
        <option key={o.id} value={o.id}>
          {o.id} — {o.label ?? o.config_name}
        </option>
      ))}
    </select>
  );
}

/* ---------- tabs ---------- */

function StagesTab({ run }: { run: RunDetail }) {
  const [showYaml, setShowYaml] = useState(false);
  const stages = run.manifest?.stages ?? [];
  return (
    <div className="flex flex-col gap-2">
      {stages.length === 0 && (
        <div className="py-6 text-center text-[13px] text-ink-500">No manifest yet.</div>
      )}
      {stages.map((s) => (
        <div key={s.kind} className="rounded-lg border border-white/8 p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  s.status === "completed"
                    ? "bg-volt-500"
                    : s.status === "failed"
                      ? "bg-team-away"
                      : s.status === "running"
                        ? "bg-volt-400 animate-pulse-dot"
                        : "bg-ink-500"
                }`}
              />
              <span className="font-mono text-[11px] uppercase tracking-wider text-ink-400">
                {s.kind}
              </span>
              <span className="text-[13px]">{s.impl}</span>
            </div>
            <Mono className="text-ink-500">{fmtDuration(s.duration_s)}</Mono>
          </div>
          {Object.keys(s.metrics).length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-[11px] text-ink-500">
              {Object.entries(s.metrics).map(([k, v]) => (
                <span key={k}>
                  {k}=<span className="text-ink-400">{String(v)}</span>
                </span>
              ))}
            </div>
          )}
          {s.error && <ErrorNote>{s.error}</ErrorNote>}
        </div>
      ))}
      <button
        onClick={() => setShowYaml(!showYaml)}
        className="mt-1 self-start text-[12px] text-ink-400 hover:text-ink-100"
      >
        {showYaml ? "▾ hide config" : "▸ show resolved config"}
      </button>
      {showYaml && (
        <pre className="overflow-x-auto rounded-lg border border-white/8 bg-turf-950 p-3 font-mono text-[11px] leading-relaxed text-ink-400">
          {run.config_yaml}
        </pre>
      )}
    </div>
  );
}

function TrackletsTab({
  artifacts,
  fps,
  selected,
  onSelect,
}: {
  artifacts: RunArtifacts;
  fps: number;
  selected: number | null;
  onSelect: (tid: number, seekTo: number | null) => void;
}) {
  if (!artifacts.tracklets)
    return <div className="py-6 text-center text-[13px] text-ink-500">No tracklets artifact.</div>;
  return (
    <div className="flex flex-col gap-1">
      {artifacts.tracklets.map((tr) => {
        const team = artifacts.teamByTracklet.get(tr.tracklet_id);
        const ent = artifacts.entityByTracklet.get(tr.tracklet_id);
        const start = tr.frames[0];
        const end = tr.frames[tr.frames.length - 1];
        const meanConf =
          tr.frames.reduce((acc, f) => acc + f.confidence, 0) / Math.max(1, tr.frames.length);
        const isSel = selected === tr.tracklet_id;
        return (
          <button
            key={tr.tracklet_id}
            onClick={() => onSelect(tr.tracklet_id, start ? start.frame_idx / fps : null)}
            className={`flex items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors ${
              isSel ? "border-volt-400/60 bg-volt-400/10" : "border-transparent hover:bg-turf-800"
            }`}
          >
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{ background: trackletColor(tr.tracklet_id) }}
            />
            <Mono className="w-10 shrink-0 text-ink-100">T{tr.tracklet_id}</Mono>
            <span className="w-16 shrink-0 text-[12px] text-ink-500">{tr.cls}</span>
            {team && <TeamDot team={team.team} />}
            <span className="flex-1 truncate font-mono text-[11px] text-ink-500">
              {fmtClock(start.frame_idx / fps)}–{fmtClock(end.frame_idx / fps)} ·{" "}
              {tr.frames.length}f · conf {fmtConf(meanConf)}
            </span>
            {ent && <Mono className="text-ink-400">#{ent.player_id}</Mono>}
          </button>
        );
      })}
    </div>
  );
}

function PlayersTab({
  artifacts,
  runId,
  fps,
  selected,
  onSelect,
  onEvidenceClick,
}: {
  artifacts: RunArtifacts;
  runId: string;
  fps: number;
  selected: number | null;
  onSelect: (pid: number, seekTo: number | null) => void;
  onEvidenceClick: (trackletId: number, playerId: number, evidenceIdx: number) => void;
}) {
  if (!artifacts.players)
    return <div className="py-6 text-center text-[13px] text-ink-500">No players artifact.</div>;
  const trackletById = new Map((artifacts.tracklets ?? []).map((t) => [t.tracklet_id, t]));
  return (
    <div className="flex flex-col gap-1.5">
      {artifacts.players.map((p) => {
        const first = p.tracklet_ids
          .map((tid) => trackletById.get(tid)?.frames[0]?.frame_idx)
          .filter((x): x is number => x != null)
          .sort((a, b) => a - b)[0];
        const isSel = selected === p.player_id;
        const evidence = p.identity.evidence.filter((ev) => ev.crop_artifact);
        return (
          <div
            key={p.player_id}
            className={`rounded-lg border transition-colors ${
              isSel ? "border-volt-400/60 bg-volt-400/10" : "border-transparent hover:bg-turf-800"
            }`}
          >
            <button
              onClick={() => onSelect(p.player_id, first != null ? first / fps : null)}
              className="w-full px-3 py-2.5 text-left"
            >
              <div className="flex items-center gap-2.5">
                <TeamDot team={p.team} />
                <span className="text-[13px] font-medium">
                  {p.identity.label ?? `Player ${p.player_id}`}
                </span>
                {p.identity.kind !== "none" && (
                  <span className="rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-400">
                    {p.identity.kind} {fmtConf(p.identity.confidence)}
                  </span>
                )}
                <span className="ml-auto font-mono text-[11px] text-ink-500">
                  {p.tracklet_ids.length} tracklet{p.tracklet_ids.length === 1 ? "" : "s"}
                  {p.association_confidence < 1 && ` · assoc ${fmtConf(p.association_confidence)}`}
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1 font-mono text-[10px] text-ink-500">
                {p.tracklet_ids.map((tid) => (
                  <span key={tid} className="rounded bg-turf-800 px-1.5 py-0.5">
                    T{tid}
                  </span>
                ))}
              </div>
            </button>
            {evidence.length > 0 && (
              <div className="flex gap-1.5 overflow-x-auto px-3 pb-2.5">
                {evidence.map((ev, i) => (
                  <button
                    key={`${ev.tracklet_id}-${ev.frame_idx}`}
                    onClick={() => onEvidenceClick(ev.tracklet_id, p.player_id, i)}
                    title={`frame ${ev.frame_idx} · score ${fmtConf(ev.score)}${ev.upscaled ? " · upscaled" : ""}`}
                    className="h-20 w-20 shrink-0 overflow-hidden rounded-md border border-white/10 transition-colors hover:border-volt-400/60"
                  >
                    <img
                      src={api.runFileUrl(runId, ev.crop_artifact!)}
                      className="h-full w-full object-cover"
                    />
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function EventsTab({
  artifacts,
  onSeek,
}: {
  artifacts: RunArtifacts;
  onSeek: (t: number) => void;
}) {
  if (!artifacts.events)
    return <div className="py-6 text-center text-[13px] text-ink-500">No events artifact.</div>;
  return (
    <div className="flex flex-col gap-1">
      {artifacts.events.map((ev) => (
        <button
          key={ev.event_id}
          onClick={() => onSeek(ev.t)}
          className="flex items-center gap-3 rounded-lg border border-transparent px-3 py-2 text-left transition-colors hover:bg-turf-800"
        >
          <Mono className="w-12 shrink-0 text-ink-500">{fmtClock(ev.t)}</Mono>
          <TeamDot team={ev.team} />
          <span className="text-[13px] capitalize">{eventLabel(ev.type)}</span>
          <span className="text-[12px] text-ink-500">
            {ev.player_id != null ? `player ${ev.player_id}` : "—"}
          </span>
          <span className="ml-auto flex items-center gap-2">
            {ev.contested && (
              <span className="rounded-full bg-team-ref/15 px-2 py-0.5 font-mono text-[10px] text-team-ref">
                contested
              </span>
            )}
            <Mono className="text-ink-500">{fmtConf(ev.confidence)}</Mono>
          </span>
        </button>
      ))}
      {artifacts.events.length === 0 && (
        <div className="py-6 text-center text-[13px] text-ink-500">No events detected.</div>
      )}
    </div>
  );
}

/* ---------- eval vs ground truth ---------- */

const EVAL_ROWS: { key: keyof EvalLevelMetrics; label: string; ratio?: boolean }[] = [
  { key: "idf1", label: "IDF1", ratio: true },
  { key: "idp", label: "ID precision", ratio: true },
  { key: "idr", label: "ID recall", ratio: true },
  { key: "mota", label: "MOTA", ratio: true },
  { key: "num_switches", label: "ID switches" },
  { key: "num_fragmentations", label: "Fragmentations" },
  { key: "num_false_positives", label: "False positives" },
  { key: "num_misses", label: "Misses" },
  { key: "mostly_tracked", label: "Mostly tracked" },
  { key: "mostly_lost", label: "Mostly lost" },
];

function fmtPredId(level: "tracklet" | "entity", id: number | null): string {
  if (id == null) return "?";
  // Entities ≥ 100000 are unassociated tracklets given a synthetic identity.
  if (level === "entity")
    return id >= 100000 ? `T${id - 100000} (unassoc)` : `#${id}`;
  return `T${id}`;
}

type EvalLevelFilter = "all" | "tracklet" | "entity";
type EvalSortBy = "time" | "gt";

function EvalTab({
  artifacts,
  gt,
  hasGroundTruth,
  onReEvaluate,
  reEvaluating,
  onInstance,
}: {
  artifacts: RunArtifacts;
  gt: GtIndex | null;
  hasGroundTruth: boolean;
  onReEvaluate: () => void;
  reEvaluating: boolean;
  onInstance: (inst: EvalInstance) => void;
}) {
  const ev = artifacts.eval;
  const [levelFilter, setLevelFilter] = useState<EvalLevelFilter>("all");
  const [gtFilter, setGtFilter] = useState<number | "all">("all");
  const [sortBy, setSortBy] = useState<EvalSortBy>("time");

  const gtOptions = useMemo(() => {
    if (!ev) return [];
    const byTrack = new Map<number, string>();
    for (const inst of ev.instances)
      if (!byTrack.has(inst.gt_track_id)) byTrack.set(inst.gt_track_id, inst.gt_label);
    return [...byTrack.entries()].sort((a, b) => a[0] - b[0]);
  }, [ev]);

  const visibleInstances = useMemo(() => {
    if (!ev) return [];
    let list = ev.instances;
    if (levelFilter !== "all") list = list.filter((inst) => inst.level === levelFilter);
    if (gtFilter !== "all") list = list.filter((inst) => inst.gt_track_id === gtFilter);
    return [...list].sort((a, b) =>
      sortBy === "time" ? a.t - b.t : a.gt_track_id - b.gt_track_id || a.t - b.t,
    );
  }, [ev, levelFilter, gtFilter, sortBy]);

  if (!ev)
    return (
      <div className="flex flex-col items-center gap-3 py-6 text-center text-[13px] text-ink-500">
        <div>No eval artifact — this video has no ground truth, or the run predates it.</div>
        {hasGroundTruth && (
          <>
            <div className="text-[12px]">
              This run predates ground truth — re-evaluate to score it.
            </div>
            <Button onClick={onReEvaluate} disabled={reEvaluating}>
              {reEvaluating ? "Evaluating…" : "Re-evaluate"}
            </Button>
          </>
        )}
      </div>
    );
  const gain = ev.association.idf1_gain;
  const fmt = (v: number, ratio?: boolean) => (ratio ? v.toFixed(3) : String(Math.round(v)));
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-[11px] text-ink-500">
          {ev.sequence ?? ev.source} · {ev.n_frames_evaluated} frames (stride {ev.sample_stride}) ·{" "}
          {ev.n_gt_tracks} GT tracks · IoU ≥ {ev.iou_threshold}
        </div>
        {hasGroundTruth && (
          <Button
            variant="ghost"
            onClick={onReEvaluate}
            disabled={reEvaluating}
            className="!py-1 shrink-0"
          >
            {reEvaluating ? "Evaluating…" : "Re-evaluate"}
          </Button>
        )}
      </div>

      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-white/8 text-left font-mono text-[10px] uppercase tracking-wider text-ink-500">
            <th className="py-1.5 font-normal">Metric</th>
            <th className="py-1.5 text-right font-normal" title="raw tracker output">
              Tracklet
            </th>
            <th className="py-1.5 text-right font-normal" title="after association">
              Entity
            </th>
          </tr>
        </thead>
        <tbody>
          {EVAL_ROWS.map((row) => (
            <tr key={row.key} className="border-b border-white/5 last:border-0">
              <td className="py-1.5 text-ink-400">{row.label}</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {fmt(ev.levels.tracklet[row.key], row.ratio)}
              </td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {fmt(ev.levels.entity[row.key], row.ratio)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div
        className={`rounded-lg border px-3 py-2 text-[12px] ${
          gain > 0.001
            ? "border-volt-400/40 bg-volt-400/10 text-volt-300"
            : gain < -0.001
              ? "border-team-away/40 bg-team-away/10 text-team-away"
              : "border-white/10 text-ink-400"
        }`}
      >
        Association gain (entity − tracklet IDF1): {gain > 0 ? "+" : ""}
        {gain.toFixed(3)}
        {Math.abs(gain) <= 0.001 && " — the associator is not changing identity quality"}
      </div>

      <div className="mt-1 flex flex-wrap items-center justify-between gap-2">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
          ID switches — click to inspect
        </div>
        <span className="text-[11px] text-ink-500">
          {visibleInstances.length} of {ev.instances.length}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1">
          {(["all", "tracklet", "entity"] as const).map((lvl) => (
            <button
              key={lvl}
              onClick={() => setLevelFilter(lvl)}
              className={`rounded-md px-2.5 py-1 text-[12px] transition-colors ${
                levelFilter === lvl
                  ? "bg-turf-800 text-ink-100"
                  : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
              }`}
            >
              {lvl}
            </button>
          ))}
        </div>
        <select
          value={gtFilter === "all" ? "" : gtFilter}
          onChange={(e) => setGtFilter(e.target.value === "" ? "all" : Number(e.target.value))}
          className="rounded-lg border border-white/10 bg-turf-850 px-2.5 py-1 text-[12px] text-ink-400 outline-none focus:border-volt-400/60"
        >
          <option value="">All GT tracks</option>
          {gtOptions.map(([id, label]) => (
            <option key={id} value={id}>
              {label} (GT{id})
            </option>
          ))}
        </select>
        <div className="ml-auto flex items-center gap-1 text-[12px] text-ink-500">
          sort:
          {(["time", "gt"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              className={`rounded-md px-2.5 py-1 transition-colors ${
                sortBy === s
                  ? "bg-turf-800 text-ink-100"
                  : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
              }`}
            >
              {s === "time" ? "time" : "GT track"}
            </button>
          ))}
        </div>
      </div>

      {visibleInstances.length === 0 && (
        <div className="py-3 text-center text-[13px] text-ink-500">
          {ev.instances.length === 0 ? "No ID switches. Clean run." : "No switches match the filters."}
        </div>
      )}
      {visibleInstances.map((inst, i) => (
        <button
          key={i}
          onClick={() => onInstance(inst)}
          className="flex items-center gap-2.5 rounded-lg border border-transparent px-3 py-2 text-left transition-colors hover:bg-turf-800"
        >
          <Mono className="w-12 shrink-0 text-volt-300">{fmtClock(inst.t)}</Mono>
          <span
            className={`rounded-full px-2 py-0.5 font-mono text-[10px] ${
              inst.level === "tracklet"
                ? "bg-turf-800 text-ink-400"
                : "bg-team-ref/15 text-team-ref"
            }`}
          >
            {inst.level}
          </span>
          <span className="text-[12px] text-ink-100">GT {inst.gt_label}</span>
          <span className="ml-auto font-mono text-[11px] text-ink-500">
            {fmtPredId(inst.level, inst.prev_id)} → {fmtPredId(inst.level, inst.new_id)}
          </span>
        </button>
      ))}
      {gt && (
        <div className="mt-1 text-[11px] leading-relaxed text-ink-500">
          Dashed boxes in the player are ground-truth tracks (amber = referee). An ID switch means
          the {""}predicted id following a GT player changed — the moments the tracker or associator
          got identity wrong.
        </div>
      )}
    </div>
  );
}

/* ---------- QA (shared with LabQA page) ---------- */

export function QATab({
  records,
  onSeek,
  showRunId = false,
}: {
  records: QARecord[];
  onSeek?: (t: number) => void;
  showRunId?: boolean;
}) {
  const pending = records.filter((r) => r.status === "pending");
  const decided = records.filter((r) => r.status !== "pending");
  return (
    <div className="flex flex-col gap-2">
      {records.length === 0 && (
        <div className="py-6 text-center text-[13px] text-ink-500">Nothing to review.</div>
      )}
      {pending.map((rec) => (
        <QACard key={rec.id} rec={rec} onSeek={onSeek} showRunId={showRunId} />
      ))}
      {decided.length > 0 && (
        <>
          <div className="mt-2 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
            Decided
          </div>
          {decided.map((rec) => (
            <QACard key={rec.id} rec={rec} onSeek={onSeek} showRunId={showRunId} />
          ))}
        </>
      )}
    </div>
  );
}

function QACard({
  rec,
  onSeek,
  showRunId,
}: {
  rec: QARecord;
  onSeek?: (t: number) => void;
  showRunId?: boolean;
}) {
  const { accept, reject, correct } = useQAActions();
  const [correcting, setCorrecting] = useState(false);
  const [pickedPlayer, setPickedPlayer] = useState<number | null>(null);
  const [note, setNote] = useState("");
  const ev = rec.payload.event;
  const pending = rec.status === "pending";
  const busy = accept.isPending || reject.isPending || correct.isPending;

  return (
    <div
      className={`rounded-lg border p-3 ${
        pending ? "border-team-ref/25 bg-team-ref/[0.03]" : "border-white/8 opacity-70"
      }`}
    >
      <div className="flex items-center gap-2.5">
        <button
          onClick={() => onSeek?.(ev.t)}
          className="font-mono text-[11px] text-volt-300 hover:underline"
          title="seek"
        >
          {fmtClock(ev.t)}
        </button>
        <TeamDot team={ev.team} />
        <span className="text-[13px] capitalize">{eventLabel(ev.type)}</span>
        <span className="text-[12px] text-ink-500">
          {ev.player_id != null ? `player ${ev.player_id}` : "unattributed"} · conf{" "}
          {fmtConf(ev.confidence)}
        </span>
        <span className="ml-auto rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-400">
          {rec.payload.reason.replace(/_/g, " ")}
        </span>
      </div>
      {showRunId && (
        <Link to={`/lab/runs/${rec.run_id}`} className="mt-1 block font-mono text-[10px] text-ink-500 hover:text-volt-300">
          run {rec.run_id}
        </Link>
      )}

      {pending ? (
        <div className="mt-2.5">
          {!correcting ? (
            <div className="flex items-center gap-2">
              <Button onClick={() => accept.mutate(rec.id)} disabled={busy} className="!py-1">
                Accept
              </Button>
              <Button variant="ghost" onClick={() => setCorrecting(true)} disabled={busy} className="!py-1">
                Correct…
              </Button>
              <Button variant="danger" onClick={() => reject.mutate(rec.id)} disabled={busy} className="!py-1">
                Reject
              </Button>
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[12px] text-ink-500">actual player:</span>
                {rec.payload.candidate_player_ids.map((pid) => (
                  <button
                    key={pid}
                    onClick={() => setPickedPlayer(pid)}
                    className={`rounded-full border px-2.5 py-0.5 font-mono text-[11px] transition-colors ${
                      pickedPlayer === pid
                        ? "border-volt-400 bg-volt-400/15 text-volt-300"
                        : "border-white/10 text-ink-400 hover:border-white/30"
                    }`}
                  >
                    #{pid}
                  </button>
                ))}
                <input
                  type="number"
                  placeholder="id"
                  className="w-16 rounded-md border border-white/10 bg-turf-850 px-2 py-0.5 font-mono text-[11px] outline-none focus:border-volt-400/60"
                  onChange={(e) =>
                    setPickedPlayer(e.target.value === "" ? null : Number(e.target.value))
                  }
                />
              </div>
              <div className="flex items-center gap-2">
                <input
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="note (optional)"
                  className="flex-1 rounded-md border border-white/10 bg-turf-850 px-2 py-1 text-[12px] outline-none placeholder:text-ink-500 focus:border-volt-400/60"
                />
                <Button
                  onClick={() =>
                    correct.mutate(
                      { id: rec.id, player_id: pickedPlayer, note: note || null },
                      { onSuccess: () => setCorrecting(false) },
                    )
                  }
                  disabled={busy || pickedPlayer == null}
                  className="!py-1"
                >
                  Save
                </Button>
                <Button variant="ghost" onClick={() => setCorrecting(false)} className="!py-1">
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="mt-1.5 font-mono text-[11px] text-ink-500">
          {rec.status}
          {rec.corrected_player_id != null && ` → player ${rec.corrected_player_id}`}
          {rec.note && ` · "${rec.note}"`}
        </div>
      )}
    </div>
  );
}
