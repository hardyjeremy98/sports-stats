// The Lab's forensic run viewer: overlay video, confidence timeline, and an
// inspector (stages / tracklets / players / events / QA).

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useGroundTruth, useRunArtifacts, type GtIndex, type RunArtifacts } from "../lib/artifacts";
import { trackletColor } from "../lib/colors";
import { eventLabel, fmtClock, fmtConf, fmtDuration, fmtPct } from "../lib/format";
import {
  useEvaluateRun,
  useIdentityQAActions,
  useQA,
  useQAActions,
  useRun,
  useRuns,
} from "../lib/hooks";
import type {
  AssociationEntitySummary,
  AssociationPair,
  AttributionLayer,
  EvalIdentity,
  EvalInstance,
  EvalLevelMetrics,
  EvalResult,
  PersistentSwitchTransition,
  QARecord,
  RunDetail,
} from "../lib/types";
import { LAYER_LABEL, SwitchInstanceRow } from "../components/EvalBits";
import { EvidenceInspector } from "../components/EvidenceInspector";
import { MergeInspector } from "../components/MergeInspector";
import { IdentityQATab } from "../components/IdentityQATab";
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

type TabId = "stages" | "tracklets" | "assoc" | "players" | "events" | "eval" | "qa";

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
  const [switchClasses, setSwitchClasses] = useState({
    genuine: true,
    frameExit: false,
    raw: false,
  });
  const [tab, setTab] = useState<TabId>("stages");
  const [qaSubTab, setQaSubTab] = useState<"events" | "identity">("events");
  const [hlTracklet, setHlTracklet] = useState<number | null>(null);
  const [hlPlayer, setHlPlayer] = useState<number | null>(null);
  const [hlGtTrack, setHlGtTrack] = useState<number | null>(null);
  const [hlPair, setHlPair] = useState<[number, number] | null>(null);
  const [inspector, setInspector] = useState<{ playerId: number; evidenceIdx: number } | null>(
    null,
  );
  const [mergeInspect, setMergeInspect] = useState<AssociationPair | null>(null);

  const getTime = useMemo(() => () => playerRef.current?.getTime() ?? 0, []);
  const seek = (t: number) => playerRef.current?.seek(t);

  const goToTransition = (rec: PersistentSwitchTransition) => {
    seek(Math.max(0, rec.t_to - 1));
    setLayers((l) => ({ ...l, gt: true, tracklets: true }));
    setHlGtTrack(rec.gt_track_id);
    if (rec.level === "tracklet") {
      setHlPair([rec.prev_id, rec.new_id]);
      setHlTracklet(null);
      setHlPlayer(null);
    } else {
      setHlPair(null);
      setHlTracklet(null);
      setHlPlayer(rec.new_id < 100000 ? rec.new_id : null);
    }
  };

  const switchTransitions = useMemo(() => {
    const ps = artifacts.eval?.persistent_switches;
    if (!ps) return null;
    const recs = [...(ps.tracklet.transitions ?? []), ...(ps.entity.transitions ?? [])];
    return recs.length > 0 || ps.tracklet.transitions ? recs : null;
  }, [artifacts.eval]);

  const visibleTransitions = useMemo(() => {
    if (!switchTransitions) return [];
    return switchTransitions
      .filter((r) => (r.verdict === "genuine" ? switchClasses.genuine : switchClasses.frameExit))
      .sort((a, b) => a.t_to - b.t_to);
  }, [switchTransitions, switchClasses]);

  const evalMarkers = useMemo(() => {
    const ev = artifacts.eval;
    if (!ev) return undefined;
    // Pre-transitions artifacts: exactly the old raw view.
    if (!switchTransitions) {
      if (ev.instances.length === 0) return undefined;
      return ev.instances.map((inst) => ({
        t: Math.max(0, inst.t),
        color: inst.level === "entity" ? "#F5C518" : "#8B949E",
        title: `${inst.level} switch @ ${fmtClock(inst.t)} GT ${inst.gt_label}`,
        onClick: () => seek(Math.max(0, inst.t - 1)),
      }));
    }
    const markers = visibleTransitions.map((rec) => ({
      t: Math.max(0, rec.t_to),
      color: rec.verdict === "genuine" ? "#F87171" : "#64748B",
      title: `${rec.verdict === "genuine" ? "switch" : "frame exit"} ${rec.level} @ ${fmtClock(
        rec.t_to,
      )} GT ${rec.gt_label} · ${rec.prev_id} → ${rec.new_id}`,
      onClick: () => goToTransition(rec),
    }));
    if (switchClasses.raw) {
      markers.push(
        ...ev.instances.map((inst) => ({
          t: Math.max(0, inst.t),
          color: "#8B949E",
          title: `raw ${inst.level} switch @ ${fmtClock(inst.t)} GT ${inst.gt_label}`,
          onClick: () => seek(Math.max(0, inst.t - 1)),
        })),
      );
    }
    return markers.length > 0 ? markers : undefined;
  }, [artifacts.eval, switchTransitions, visibleTransitions, switchClasses.raw]);

  if (run.isLoading) return <Spinner label="Loading run" />;
  if (run.isError) return <ErrorNote>{(run.error as Error).message}</ErrorNote>;
  const r = run.data!;
  const duration = r.video?.duration_s ?? r.manifest?.video.duration_s ?? 0;
  const fps = r.video?.fps || r.manifest?.video.fps || 25;
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
            highlightTrackletIds={hlPair}
            highlightPlayerId={hlPlayer}
            highlightGtTrackId={hlGtTrack}
          />
          <LayerChips layers={layers} onChange={setLayers} artifacts={artifacts} gt={gt} />
          <Card className="p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-3">
                <SignalPicker value={signal} onChange={setSignal} />
                {switchTransitions && (
                  <div className="flex items-center gap-2 font-mono text-[11px] text-ink-500">
                    {(
                      [
                        ["genuine", "switches", "#F87171"],
                        ["frameExit", "frame exits", "#64748B"],
                        ["raw", "raw", "#8B949E"],
                      ] as const
                    ).map(([key, label, color]) => (
                      <button
                        key={key}
                        onClick={() => setSwitchClasses((s) => ({ ...s, [key]: !s[key] }))}
                        className={`rounded border px-2 py-0.5 ${
                          switchClasses[key]
                            ? "border-white/20 text-ink-100"
                            : "border-white/8 text-ink-600"
                        }`}
                      >
                        <span style={{ color: switchClasses[key] ? color : undefined }}>●</span>{" "}
                        {label}
                      </button>
                    ))}
                    <span className="mx-1 text-ink-700">·</span>
                    <button
                      className="rounded border border-white/8 px-2 py-0.5 hover:text-ink-100"
                      title="previous switch"
                      onClick={() => {
                        const t = getTime();
                        const prev = [...visibleTransitions].reverse().find((r) => r.t_to < t + 0.95);
                        if (prev) goToTransition(prev);
                      }}
                    >
                      ‹
                    </button>
                    <button
                      className="rounded border border-white/8 px-2 py-0.5 hover:text-ink-100"
                      title="next switch"
                      onClick={() => {
                        const t = getTime();
                        const next = visibleTransitions.find((r) => r.t_to - 1 > t + 0.05);
                        if (next) goToTransition(next);
                      }}
                    >
                      ›
                    </button>
                  </div>
                )}
              </div>
              <span className="text-[11px] text-ink-500">
                red cells = low confidence · click to seek
              </span>
            </div>
            <TimelineStrip
              timeline={artifacts.timeline}
              events={artifacts.events}
              spotting={artifacts.spotting}
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
              ...(artifacts.association
                ? [{ id: "assoc" as const, label: "Assoc", count: artifacts.association.pairs.length }]
                : []),
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
                  setHlPair(null);
                  if (t != null) seek(t);
                }}
              />
            )}
            {tab === "assoc" && (
              <AssocTab
                artifacts={artifacts}
                fps={fps}
                frameCount={r.manifest?.video.frame_count ?? Math.round(duration * fps)}
                selectedEntity={hlPlayer}
                onSelectEntity={(pid) => {
                  setHlPlayer(pid);
                  setHlTracklet(null);
                  setHlGtTrack(null);
                  setHlPair(null);
                }}
                onSelectTracklet={(tid, t) => {
                  setHlTracklet(tid === hlTracklet ? null : tid);
                  setHlGtTrack(null);
                  setHlPair(null);
                  if (t != null) seek(t);
                }}
                onSelectPair={(pair) => {
                  setHlPair([pair.a, pair.b]);
                  setHlTracklet(null);
                  setHlGtTrack(null);
                  const aTracklet = artifacts.tracklets?.find((t) => t.tracklet_id === pair.a);
                  const endFrame = aTracklet?.frames[aTracklet.frames.length - 1]?.frame_idx;
                  if (endFrame != null) seek(endFrame / fps);
                }}
                onInspectPair={(pair) => setMergeInspect(pair)}
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
                  setHlPair(null);
                  if (t != null) seek(t);
                }}
                onEvidenceClick={(tid, pid, evidenceIdx) => {
                  setHlTracklet(tid);
                  setHlPlayer(pid);
                  setHlGtTrack(null);
                  setHlPair(null);
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
                onTransition={goToTransition}
                onInstance={(inst) => {
                  seek(Math.max(0, inst.t - 1)); // land just before the switch
                  setLayers((l) => ({ ...l, gt: true, tracklets: true }));
                  setHlGtTrack(inst.gt_track_id);
                  setHlPair(null);

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
            {tab === "qa" && (
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-1">
                  {(["events", "identity"] as const).map((st) => (
                    <button
                      key={st}
                      onClick={() => setQaSubTab(st)}
                      className={`rounded-md px-2.5 py-1 text-[12px] capitalize transition-colors ${
                        qaSubTab === st
                          ? "bg-turf-800 text-ink-100"
                          : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
                      }`}
                    >
                      {st}
                    </button>
                  ))}
                </div>
                {qaSubTab === "events" && <QATab records={qa.data ?? []} onSeek={seek} />}
                {qaSubTab === "identity" && (
                  <IdentityQATab
                    runId={r.id}
                    artifacts={artifacts}
                    fps={fps}
                    seek={seek}
                    onSelectPairHighlight={(a, b) => {
                      setHlPair([a, b]);
                      setHlTracklet(null);
                      setHlPlayer(null);
                      setHlGtTrack(null);
                    }}
                  />
                )}
              </div>
            )}
          </div>
        </Card>
      </div>

      {mergeInspect && (
        <MergeInspector
          videoUrl={r.video_id != null ? `/api/videos/${r.video_id}/file` : null}
          fps={r.video?.fps || r.manifest?.video.fps || 25}
          pair={mergeInspect}
          artifacts={artifacts}
          onClose={() => setMergeInspect(null)}
          onSelectTracklet={(tid, t) => {
            setMergeInspect(null);
            setHlTracklet(tid);
            setHlPlayer(null);
            setHlGtTrack(null);
            setHlPair(null);
            setTab("tracklets");
            if (t != null) seek(t);
          }}
        />
      )}

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
                setHlPair(null);
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
  const { create } = useIdentityQAActions();
  const [merging, setMerging] = useState<Set<number>>(new Set());
  const [splittingId, setSplittingId] = useState<number | null>(null);
  const [splitSelected, setSplitSelected] = useState<Set<number>>(new Set());
  const [rosterInput, setRosterInput] = useState<Record<number, string>>({});
  const [flash, setFlash] = useState<{ id: string; msg: string } | null>(null);

  function showFlash(id: string, msg: string) {
    setFlash({ id, msg });
    window.setTimeout(() => setFlash((f) => (f?.id === id ? null : f)), 2500);
  }

  function toggleMerge(pid: number) {
    setMerging((prev) => {
      const next = new Set(prev);
      if (next.has(pid)) next.delete(pid);
      else next.add(pid);
      return next;
    });
  }

  function flagMerge() {
    const ids = [...merging];
    if (ids.length < 2) return;
    create.mutate(
      { run_id: runId, kind: "merge", payload: { player_ids: ids } },
      {
        onSuccess: () => {
          showFlash("merge", `Flagged merge of ${ids.length} entities`);
          setMerging(new Set());
        },
      },
    );
  }

  function openSplit(pid: number) {
    setSplittingId(pid === splittingId ? null : pid);
    setSplitSelected(new Set());
  }

  function toggleSplitTracklet(tid: number) {
    setSplitSelected((prev) => {
      const next = new Set(prev);
      if (next.has(tid)) next.delete(tid);
      else next.add(tid);
      return next;
    });
  }

  function confirmSplit(pid: number) {
    const ids = [...splitSelected];
    if (ids.length === 0) return;
    create.mutate(
      { run_id: runId, kind: "split", payload: { player_id: pid, tracklet_ids_out: ids } },
      {
        onSuccess: () => {
          showFlash(`split-${pid}`, "Flagged split");
          setSplittingId(null);
          setSplitSelected(new Set());
        },
      },
    );
  }

  function saveRoster(pid: number) {
    const label = (rosterInput[pid] ?? "").trim();
    if (!label) return;
    create.mutate(
      { run_id: runId, kind: "roster", payload: { player_id: pid, roster_label: label } },
      { onSuccess: () => showFlash(`roster-${pid}`, "Saved roster label") },
    );
  }

  if (!artifacts.players)
    return <div className="py-6 text-center text-[13px] text-ink-500">No players artifact.</div>;
  const trackletById = new Map((artifacts.tracklets ?? []).map((t) => [t.tracklet_id, t]));
  const namingByThread = new Map(
    (artifacts.naming?.threads ?? []).map((t) => [t.thread_id, t]),
  );
  const namedCount = [...namingByThread.values()].filter((t) => t.decision === "named").length;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="text-[11px] leading-relaxed text-ink-500">
        Merge/split/roster flags are annotations — they don't change this run's entities; they
        take effect on a future re-run or training pass.
      </div>
      {artifacts.naming && (
        <div className="flex items-center gap-3 rounded-lg border border-white/8 px-3 py-2 font-mono text-[11px] text-ink-400">
          <span className="uppercase tracking-[0.14em] text-ink-500">Naming</span>
          <span className="text-ink-100">{namedCount} named</span>
          <span>{namingByThread.size - namedCount} abstained</span>
          {(["auto_accept", "adjudicate", "qa"] as const).map((tier) => {
            const n = [...namingByThread.values()].filter((t) => t.tier === tier).length;
            return n > 0 ? (
              <span key={tier}>
                {tier.replace(/_/g, " ")} {n}
              </span>
            ) : null;
          })}
          <span className="ml-auto text-ink-500">{artifacts.naming.roster.length} roster</span>
        </div>
      )}
      {merging.size >= 2 && (
        <div className="sticky top-0 z-10 flex items-center gap-2 rounded-lg border border-volt-400/40 bg-turf-900 px-3 py-2">
          <span className="text-[12px] text-ink-100">{merging.size} selected</span>
          <Button className="!py-1" onClick={flagMerge} disabled={create.isPending}>
            Flag merge
          </Button>
          <button
            onClick={() => setMerging(new Set())}
            className="text-[11px] text-ink-400 hover:text-ink-100"
          >
            clear
          </button>
          {flash?.id === "merge" && (
            <span className="ml-auto text-[11px] text-volt-300">{flash.msg}</span>
          )}
        </div>
      )}
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
            <div className="flex items-start gap-1.5 pl-2 pr-1 pt-2">
              <input
                type="checkbox"
                checked={merging.has(p.player_id)}
                onChange={() => toggleMerge(p.player_id)}
                title="select for merge flag"
                className="mt-3.5 h-3.5 w-3.5 shrink-0 accent-volt-400"
              />
              <button
                onClick={() => onSelect(p.player_id, first != null ? first / fps : null)}
                className="min-w-0 flex-1 px-1.5 py-1 text-left"
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
                  {(() => {
                    const thread = namingByThread.get(p.player_id);
                    if (!thread) return null;
                    return thread.decision === "named" ? (
                      <span
                        className="rounded-full bg-volt-400/10 px-2 py-0.5 font-mono text-[10px] text-volt-300"
                        title={`posterior ${thread.label ? fmtConf(thread.posterior[thread.label] ?? 0) : "—"} · margin ${thread.margin != null ? fmtConf(thread.margin) : "—"}`}
                      >
                        {thread.label} · p {thread.label ? fmtConf(thread.posterior[thread.label] ?? 0) : "—"}
                        {thread.margin != null && ` · m ${fmtConf(thread.margin)}`}
                        {thread.tier && ` · ${thread.tier.replace(/_/g, " ")}`}
                      </span>
                    ) : (
                      // Abstention is a first-class outcome — visibly distinct,
                      // never styled like a low-confidence name.
                      <span className="rounded-full border border-dashed border-amber-400/40 px-2 py-0.5 font-mono text-[10px] text-amber-300/80">
                        abstained{thread.tier ? ` · ${thread.tier.replace(/_/g, " ")}` : ""}
                      </span>
                    );
                  })()}
                  <span className="ml-auto font-mono text-[11px] text-ink-500">
                    {p.tracklet_ids.length} tracklet{p.tracklet_ids.length === 1 ? "" : "s"}
                    {p.association_confidence < 1 &&
                      ` · assoc ${fmtConf(p.association_confidence)}`}
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
            </div>
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
            <div className="flex flex-wrap items-center gap-2 px-3 pb-2 text-[11px]">
              <button
                onClick={() => openSplit(p.player_id)}
                className="text-ink-400 hover:text-ink-100"
              >
                {splittingId === p.player_id ? "cancel split" : "flag split"}
              </button>
              <span className="text-ink-600">·</span>
              <input
                value={rosterInput[p.player_id] ?? ""}
                onChange={(e) =>
                  setRosterInput((s) => ({ ...s, [p.player_id]: e.target.value }))
                }
                placeholder="roster label"
                className="w-28 rounded-md border border-white/10 bg-turf-850 px-1.5 py-0.5 text-[11px] outline-none focus:border-volt-400/60"
              />
              <button
                onClick={() => saveRoster(p.player_id)}
                disabled={!(rosterInput[p.player_id] ?? "").trim() || create.isPending}
                className="text-ink-400 hover:text-ink-100 disabled:opacity-40"
              >
                save
              </button>
              {flash?.id === `roster-${p.player_id}` && (
                <span className="text-volt-300">{flash.msg}</span>
              )}
            </div>
            {splittingId === p.player_id && (
              <div className="flex flex-col gap-1.5 px-3 pb-2.5">
                <div className="flex flex-wrap gap-1.5">
                  {p.tracklet_ids.map((tid) => (
                    <button
                      key={tid}
                      onClick={() => toggleSplitTracklet(tid)}
                      className={`rounded-full border px-2 py-0.5 font-mono text-[10px] transition-colors ${
                        splitSelected.has(tid)
                          ? "border-volt-400 bg-volt-400/15 text-volt-300"
                          : "border-white/10 text-ink-400 hover:border-white/30"
                      }`}
                    >
                      T{tid}
                    </button>
                  ))}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    className="!py-1"
                    onClick={() => confirmSplit(p.player_id)}
                    disabled={splitSelected.size === 0 || create.isPending}
                  >
                    Confirm split ({splitSelected.size})
                  </Button>
                  {flash?.id === `split-${p.player_id}` && (
                    <span className="text-[11px] text-volt-300">{flash.msg}</span>
                  )}
                </div>
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

/* ---------- association inspector ---------- */

function AssocTab({
  artifacts,
  fps,
  frameCount,
  selectedEntity,
  onSelectEntity,
  onSelectTracklet,
  onSelectPair,
  onInspectPair,
}: {
  artifacts: RunArtifacts;
  fps: number;
  frameCount: number;
  selectedEntity: number | null;
  onSelectEntity: (playerId: number | null) => void;
  onSelectTracklet: (trackletId: number, seekTo: number | null) => void;
  onSelectPair: (pair: AssociationPair) => void;
  onInspectPair: (pair: AssociationPair) => void;
}) {
  const report = artifacts.association;
  const [showAllPairs, setShowAllPairs] = useState(false);
  useEffect(() => setShowAllPairs(false), [selectedEntity]);

  if (!report)
    return (
      <div className="py-6 text-center text-[13px] text-ink-500">No association artifact.</div>
    );

  const merged = report.pairs.filter((p) => p.decision === "merged").length;
  const rejected = report.pairs.length - merged;

  const rejectionCounts = new Map<string, number>();
  for (const p of report.pairs) {
    if (p.decision === "rejected" && p.reason) {
      rejectionCounts.set(p.reason, (rejectionCounts.get(p.reason) ?? 0) + 1);
    }
  }

  const entity =
    selectedEntity != null
      ? (report.entities.find((e) => e.player_id === selectedEntity) ?? null)
      : null;

  const trackletById = new Map((artifacts.tracklets ?? []).map((t) => [t.tracklet_id, t]));

  // All pairs when no entity is selected; otherwise only pairs touching one
  // of the entity's tracklets.
  const filteredPairs = entity
    ? report.pairs.filter(
        (p) => entity.tracklet_ids.includes(p.a) || entity.tracklet_ids.includes(p.b),
      )
    : report.pairs;

  const sortedPairs = [...filteredPairs].sort((x, y) => {
    if (x.decision !== y.decision) return x.decision === "merged" ? -1 : 1;
    if (x.affinity == null && y.affinity == null) return 0;
    if (x.affinity == null) return 1; // nulls last
    if (y.affinity == null) return -1;
    return x.affinity - y.affinity; // ascending
  });

  const visiblePairs = showAllPairs ? sortedPairs : sortedPairs.slice(0, 100);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-2">
          <Mono className="text-ink-100">{report.impl}</Mono>
          <span className="font-mono text-[11px] text-ink-500">
            {merged} merged · {rejected} rejected
          </span>
        </div>
        {Object.keys(report.params).length > 0 && (
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-ink-500">
            {Object.entries(report.params).map(([k, v]) => (
              <span key={k}>
                {k}=<span className="text-ink-400">{String(v)}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      <div>
        <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
          Rejections by constraint
        </div>
        {rejectionCounts.size === 0 ? (
          <div className="text-[12px] text-ink-500">No rejected pairs.</div>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {[...rejectionCounts.entries()].map(([reason, count]) => (
              <span
                key={reason}
                title={
                  reason === "color_too_far"
                    ? "appearance distance rejected the pair"
                    : "a structural gate rejected the pair before appearance was compared"
                }
                className={`rounded-full px-2.5 py-1 font-mono text-[11px] ${
                  reason === "color_too_far"
                    ? "bg-team-ref/15 text-team-ref"
                    : "bg-turf-800 text-ink-400"
                }`}
              >
                {reason.replace(/_/g, " ")} · {count}
              </span>
            ))}
          </div>
        )}
      </div>

      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
            Entity tracklets
          </div>
          {entity && (
            <button
              onClick={() => onSelectEntity(null)}
              className="text-[11px] text-ink-400 hover:text-ink-100"
            >
              ‹ change entity
            </button>
          )}
        </div>
        {!entity ? (
          <EntityPicker entities={report.entities} onSelect={onSelectEntity} />
        ) : (
          <div className="flex flex-col gap-1.5">
            {entity.tracklet_ids.map((tid) => {
              const tr = trackletById.get(tid);
              if (!tr || tr.frames.length === 0) return null;
              const start = tr.frames[0].frame_idx;
              const end = tr.frames[tr.frames.length - 1].frame_idx;
              const denom = Math.max(1, frameCount);
              const left = (start / denom) * 100;
              const width = Math.max(0.6, ((end - start) / denom) * 100);
              return (
                <button
                  key={tid}
                  onClick={() => onSelectTracklet(tid, start / fps)}
                  className="flex items-center gap-2 text-left"
                  title={`T${tid} · frames ${start}–${end}`}
                >
                  <Mono className="w-10 shrink-0 text-ink-400">T{tid}</Mono>
                  <div className="relative h-3 flex-1 overflow-hidden rounded-sm bg-turf-800">
                    <div
                      className="absolute top-0 h-full rounded-sm"
                      style={{ left: `${left}%`, width: `${width}%`, background: trackletColor(tid) }}
                    />
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
            Candidate pairs
          </div>
          <span className="text-[11px] text-ink-500">
            {visiblePairs.length} of {sortedPairs.length}
          </span>
        </div>
        <div className="flex flex-col gap-1">
          {visiblePairs.map((p, i) => (
            <AssocPairRow
              key={`${p.a}-${p.b}-${i}`}
              pair={p}
              onClick={() => onSelectPair(p)}
              onInspect={artifacts.reidDetail ? () => onInspectPair(p) : undefined}
            />
          ))}
          {sortedPairs.length === 0 && (
            <div className="py-3 text-center text-[13px] text-ink-500">No candidate pairs.</div>
          )}
        </div>
        {!showAllPairs && sortedPairs.length > 100 && (
          <button
            onClick={() => setShowAllPairs(true)}
            className="mt-2 text-[12px] text-ink-400 hover:text-ink-100"
          >
            Show {sortedPairs.length - 100} more
          </button>
        )}
      </div>
    </div>
  );
}

function EntityPicker({
  entities,
  onSelect,
}: {
  entities: AssociationEntitySummary[];
  onSelect: (playerId: number) => void;
}) {
  // Entities with more than one tracklet are the interesting forensic case —
  // surface them first.
  const sorted = [...entities].sort((a, b) => {
    const am = a.tracklet_ids.length > 1 ? 1 : 0;
    const bm = b.tracklet_ids.length > 1 ? 1 : 0;
    return bm - am;
  });
  return (
    <div className="flex flex-col gap-1">
      {sorted.map((e) => (
        <button
          key={e.player_id}
          onClick={() => onSelect(e.player_id)}
          className="flex items-center gap-2 rounded-lg border border-transparent px-2.5 py-1.5 text-left transition-colors hover:bg-turf-800"
        >
          <Mono className="w-14 shrink-0 text-ink-100">#{e.player_id}</Mono>
          <div className="flex flex-wrap gap-1 font-mono text-[10px] text-ink-500">
            {e.tracklet_ids.map((tid) => (
              <span key={tid} className="rounded bg-turf-800 px-1.5 py-0.5">
                T{tid}
              </span>
            ))}
          </div>
        </button>
      ))}
      {sorted.length === 0 && (
        <div className="py-3 text-center text-[13px] text-ink-500">No entities.</div>
      )}
    </div>
  );
}

function AssocPairRow({
  pair,
  onClick,
  onInspect,
}: {
  pair: AssociationPair;
  onClick: () => void;
  onInspect?: () => void;
}) {
  // dist_px / gap_s: an implausible speed is one of the constraints that can
  // reject a pair before appearance is even compared.
  const speed =
    pair.dist_px != null && pair.gap_s != null && pair.gap_s > 0
      ? pair.dist_px / pair.gap_s
      : null;
  // color distance is Lab-space units (~0-100s), embed distance is cosine
  // (~0-2) — whichever the associator populated, format with matching precision.
  const appearanceDist = pair.color_distance ?? pair.embed_distance;
  const appearanceDigits = pair.color_distance != null ? 1 : 3;
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 rounded-lg border border-transparent px-2.5 py-1.5 text-left transition-colors hover:bg-turf-800"
    >
      <Mono className="w-[4.5rem] shrink-0 text-ink-100">
        T{pair.a} ↔ T{pair.b}
      </Mono>
      {pair.decision === "merged" ? (
        <span className="shrink-0 rounded-full bg-volt-400/15 px-2 py-0.5 font-mono text-[10px] text-volt-300">
          merged
        </span>
      ) : (
        <span className="shrink-0 rounded-full bg-team-away/15 px-2 py-0.5 font-mono text-[10px] text-team-away">
          {pair.reason ? pair.reason.replace(/_/g, " ") : "rejected"}
        </span>
      )}
      <span className="ml-auto flex items-center gap-3 font-mono text-[11px] text-ink-500">
        <span title="appearance distance (color or embedding)">
          d {appearanceDist != null ? appearanceDist.toFixed(appearanceDigits) : "—"}
        </span>
        <span title="gap seconds">g {pair.gap_s != null ? `${pair.gap_s.toFixed(2)}s` : "—"}</span>
        <span title="implied speed = dist_px / gap_s">
          v {speed != null ? speed.toFixed(0) : "—"}
        </span>
        <span title="affinity" className="text-ink-400">
          a {pair.affinity != null ? pair.affinity.toFixed(2) : "—"}
        </span>
        {onInspect && (
          <span
            role="button"
            tabIndex={0}
            title="inspect this pair: prototype crops + per-part evidence"
            onClick={(e) => {
              e.stopPropagation();
              onInspect();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.stopPropagation();
                onInspect();
              }
            }}
            className="rounded px-1.5 py-0.5 text-ink-400 transition-colors hover:bg-turf-700 hover:text-ink-100"
          >
            ⧉
          </span>
        )}
      </span>
    </button>
  );
}

/* ---------- eval vs ground truth ---------- */

type EvalLevelKey = "tracklet" | "entity";

// Rows read the whole EvalResult rather than a `keyof EvalLevelMetrics`, so a
// row can pull from ev.levels (motmetrics), ev.hota (SPO-7) or ev.purity
// (SPO-6) -- all three are keyed by the same tracklet/entity levels, which is
// what lets them share one Metric x level table.
//
// `undefined` = the run predates this metric (re-evaluate to backfill);
// `null` = computed, but the evaluator abstained (nothing matched GT). Both
// render "—", with different tooltips -- neither is ever a 0.
interface EvalRow {
  label: string;
  hint?: string;
  get: (ev: EvalResult, level: EvalLevelKey) => number | null | undefined;
  fmt?: (v: number) => string;
}

interface EvalRowGroup {
  title?: string;
  rows: EvalRow[];
  caption?: (ev: EvalResult) => string | null;
}

const fmtRatio = (v: number) => v.toFixed(3);
const fmtCount = (v: number) => String(Math.round(v));

const lvl = (key: keyof EvalLevelMetrics) => (ev: EvalResult, level: EvalLevelKey) =>
  ev.levels[level][key];
const purityAgg = (ev: EvalResult, level: EvalLevelKey) => ev.purity?.[level]?.post_filter;

const EVAL_ROW_GROUPS: EvalRowGroup[] = [
  {
    rows: [
      { label: "IDF1", get: lvl("idf1"), fmt: fmtRatio },
      { label: "ID precision", get: lvl("idp"), fmt: fmtRatio },
      { label: "ID recall", get: lvl("idr"), fmt: fmtRatio },
      { label: "MOTA", get: lvl("mota"), fmt: fmtRatio },
      { label: "ID switches", get: lvl("num_switches") },
      { label: "Fragmentations", get: lvl("num_fragmentations") },
      { label: "False positives", get: lvl("num_false_positives") },
      { label: "Misses", get: lvl("num_misses") },
      { label: "Mostly tracked", get: lvl("mostly_tracked") },
      { label: "Mostly lost", get: lvl("mostly_lost") },
    ],
  },
  {
    title: "HOTA family",
    // LocA is omitted: it measures box tightness, a detection concern the
    // SPO-9 detection block already covers, and it does not inform the
    // association decisions this view exists to support.
    rows: [
      {
        label: "HOTA",
        hint: "Geometric mean of detection and association accuracy, averaged over TrackEval's own IoU grid (so it ignores the IoU threshold above).",
        get: (ev, level) => ev.hota?.[level]?.hota,
        fmt: fmtRatio,
      },
      {
        label: "DetA",
        hint: "Detection accuracy: how much of the GT was found, independent of identity.",
        get: (ev, level) => ev.hota?.[level]?.deta,
        fmt: fmtRatio,
      },
      {
        label: "AssA",
        hint: "Association accuracy: how well found detections are linked into consistent identities. This is the number the tracklet-modernization program is steered by.",
        get: (ev, level) => ev.hota?.[level]?.assa,
        fmt: fmtRatio,
      },
    ],
  },
  {
    // Layer-neutral title: this group spans both columns, and the Entity column
    // reports purity.entity -- not the `tracklet_purity` headline metric.
    title: "Track composition",
    rows: [
      {
        label: "Mean purity",
        hint: "Frame-weighted share of each tracklet's matched frames that sit on its majority GT identity. 1.0 = no tracklet ever mixes two players.",
        get: (ev, level) => purityAgg(ev, level)?.mean_purity,
        fmt: fmtRatio,
      },
      {
        label: "Impure fraction",
        hint: "Share of GT-matched tracklets that touch more than one GT identity.",
        get: (ev, level) => purityAgg(ev, level)?.frac_impure,
        fmt: fmtRatio,
      },
      {
        label: "Mixed-identity duration",
        hint: "Total time spent on a GT identity other than the tracklet's majority one — the contamination IDF1/MOTA cannot see.",
        get: (ev, level) => purityAgg(ev, level)?.total_mixed_seconds,
        fmt: (v) => fmtDuration(v),
      },
      {
        label: "Tracklets per GT player (mean)",
        hint: "Per-player fragmentation: how many separate tracklets a single GT player is broken into on average. 1.0 = no fragmentation.",
        // `summary` is null when nothing matched GT — an abstention. Returning
        // `agg?...summary?.mean` would collapse that null to `undefined` and
        // mislabel a current run as one that predates the metric.
        get: (ev, level) => {
          const agg = purityAgg(ev, level);
          if (!agg) return undefined;
          return agg.tracklets_per_gt_player.summary?.mean ?? null;
        },
        fmt: (v) => v.toFixed(1),
      },
      {
        label: "Tracklets per GT player (max)",
        hint: "The worst-fragmented GT player in this sequence.",
        get: (ev, level) => {
          const agg = purityAgg(ev, level);
          if (!agg) return undefined;
          return agg.tracklets_per_gt_player.summary?.max ?? null;
        },
      },
    ],
    caption: (ev) => {
      const p = ev.purity?.tracklet;
      if (!p) return null;
      if (p.min_track_length === null)
        return (
          "Unfiltered — the track stage's min_track_length could not be discovered for " +
          "this run, so the evaluator abstained from filtering rather than assuming 0."
        );
      return (
        `Tracklets shorter than ${p.min_track_length} frame` +
        `${p.min_track_length === 1 ? "" : "s"} excluded. Tracklets the track stage ` +
        `already dropped upstream never reached the evaluator and cannot be recovered here.`
      );
    },
  },
];

// The two "—" cases are deliberately distinguished. A run that predates SPO-6/
// SPO-7 can be fixed by the Re-evaluate button a few pixels away, so say so; an
// abstention is a real, final answer about this data and must not send anyone
// hunting for a backfill that will never come.
function EvalMetricCell({ ev, level, row }: { ev: EvalResult; level: EvalLevelKey; row: EvalRow }) {
  const v = row.get(ev, level);
  if (v == null)
    return (
      <td
        className="py-1.5 text-right font-mono text-ink-500"
        title={
          v === undefined
            ? "This run predates the metric — re-evaluate to backfill it."
            : "No value: nothing matched ground truth at this level, so the evaluator abstained."
        }
      >
        —
      </td>
    );
  return (
    <td className="py-1.5 text-right font-mono text-ink-100">
      {(row.fmt ?? fmtCount)(v)}
    </td>
  );
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
  onTransition,
}: {
  artifacts: RunArtifacts;
  gt: GtIndex | null;
  hasGroundTruth: boolean;
  onReEvaluate: () => void;
  reEvaluating: boolean;
  onInstance: (inst: EvalInstance) => void;
  onTransition: (rec: PersistentSwitchTransition) => void;
}) {
  const ev = artifacts.eval;
  const [levelFilter, setLevelFilter] = useState<EvalLevelFilter>("all");
  const [gtFilter, setGtFilter] = useState<number | "all">("all");
  const [layerFilter, setLayerFilter] = useState<AttributionLayer | "all">("all");
  const [sortBy, setSortBy] = useState<EvalSortBy>("time");

  const transitions = useMemo(() => {
    const ps = ev?.persistent_switches;
    if (!ps) return null;
    // Same presence check as the page-level `switchTransitions` memo: old
    // artifacts written before per-transition records existed have
    // `persistent_switches` but no `.transitions` key on either level, and
    // must keep rendering exactly today's raw-instance-only view.
    const recs = [...(ps.tracklet.transitions ?? []), ...(ps.entity.transitions ?? [])];
    if (!(recs.length > 0 || ps.tracklet.transitions)) return null;
    let list = recs;
    if (levelFilter !== "all") list = list.filter((r) => r.level === levelFilter);
    if (gtFilter !== "all") list = list.filter((r) => r.gt_track_id === gtFilter);
    return list.sort((a, b) => a.t_to - b.t_to);
  }, [ev, levelFilter, gtFilter]);

  const layerOptions = useMemo(() => {
    if (!ev) return [];
    const layers = new Set<AttributionLayer>();
    for (const inst of ev.instances) if (inst.attribution) layers.add(inst.attribution.layer);
    return [...layers].sort();
  }, [ev]);

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
    if (layerFilter !== "all")
      list = list.filter((inst) => inst.attribution?.layer === layerFilter);
    return [...list].sort((a, b) =>
      sortBy === "time" ? a.t - b.t : a.gt_track_id - b.gt_track_id || a.t - b.t,
    );
  }, [ev, levelFilter, gtFilter, layerFilter, sortBy]);

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
  // Captions render below the whole table (they are prose, and a <tr> here
  // would take over :last-child and strip the last data row's border). Only
  // one group has a caption today; a second would need its group named in the
  // text, since position no longer ties a caption to its rows.
  const captions = EVAL_ROW_GROUPS.map((group) => group.caption?.(ev)).filter(
    (c): c is string => !!c,
  );
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
          {EVAL_ROW_GROUPS.map((group, gi) => (
            <Fragment key={group.title ?? gi}>
              {group.title && (
                <tr>
                  <th
                    colSpan={3}
                    scope="colgroup"
                    className="pt-3 pb-1 text-left font-mono text-[10px] font-normal uppercase tracking-[0.18em] text-ink-500"
                  >
                    {group.title}
                  </th>
                </tr>
              )}
              {group.rows.map((row) => (
                <tr key={row.label} className="border-b border-white/5 last:border-0">
                  <td className="py-1.5 text-ink-400" title={row.hint}>
                    {row.label}
                  </td>
                  <EvalMetricCell ev={ev} level="tracklet" row={row} />
                  <EvalMetricCell ev={ev} level="entity" row={row} />
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>

      {/* Captions are prose, not tabular data — inside <tbody> the caption row
          would be the table's real :last-child, so `last:border-0` would stop
          stripping the final metric row's border whenever a caption rendered. */}
      {captions.map((caption, i) => (
        <div key={i} className="text-[11px] leading-relaxed text-ink-500">
          {caption}
        </div>
      ))}

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

      {ev.identity !== undefined && <IdentityMetricsBlock identity={ev.identity} />}

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
        {layerOptions.length > 0 && (
          <div className="flex items-center gap-1">
            {(["all", ...layerOptions] as (AttributionLayer | "all")[]).map((layer) => (
              <button
                key={layer}
                onClick={() => setLayerFilter(layer)}
                className={`rounded-md px-2.5 py-1 text-[12px] transition-colors ${
                  layerFilter === layer
                    ? "bg-turf-800 text-ink-100"
                    : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
                }`}
              >
                {layer === "all" ? "all layers" : LAYER_LABEL[layer]}
              </button>
            ))}
          </div>
        )}
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

      {transitions !== null && (
        <div>
          <div className="mb-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">
            Switches (persistent, 1s)
          </div>
          {transitions.length === 0 ? (
            <div className="py-2 text-[12px] text-ink-500">
              No persistent switches at the current filters.
            </div>
          ) : (
            <table className="w-full text-[12px]">
              <tbody>
                {transitions.map((rec, i) => (
                  <tr
                    key={i}
                    className="cursor-pointer border-b border-white/5 last:border-0 hover:bg-turf-900"
                    onClick={() => onTransition(rec)}
                  >
                    <td className="py-1.5">
                      <span
                        className={`rounded px-1.5 py-0.5 font-mono text-[10px] uppercase ${
                          rec.verdict === "genuine"
                            ? "bg-team-away/20 text-team-away"
                            : "bg-white/5 text-ink-500"
                        }`}
                      >
                        {rec.verdict === "genuine" ? "switch" : "frame exit"}
                      </span>
                    </td>
                    <td className="py-1.5 font-mono text-ink-300">
                      {rec.prev_id} → {rec.new_id}
                    </td>
                    <td className="py-1.5 text-ink-400">{rec.gt_label}</td>
                    <td className="py-1.5 font-mono text-ink-400">
                      {fmtClock(rec.t_from)}–{fmtClock(rec.t_to)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-[11px] text-ink-500">
                      {rec.prev_run_s.toFixed(1)}s → {rec.new_run_s.toFixed(1)}s
                    </td>
                    <td className="py-1.5 pl-2 text-[11px] text-ink-600">{rec.level}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {visibleInstances.length === 0 && (
        <div className="py-3 text-center text-[13px] text-ink-500">
          {ev.instances.length === 0 ? "No ID switches. Clean run." : "No switches match the filters."}
        </div>
      )}
      {visibleInstances.map((inst, i) => (
        <SwitchInstanceRow key={i} inst={inst} onClick={() => onInstance(inst)} />
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

/* ---------- semantic identity layer (ADR 004) ---------- */

function IdentityMetricsBlock({ identity }: { identity: EvalIdentity | null }) {
  return (
    <div>
      <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
        Identity
      </div>
      {identity === null ? (
        <div className="text-[12px] text-ink-500">Identity stage not run.</div>
      ) : (
        <table className="w-full text-[13px]">
          <tbody>
            <tr className="border-b border-white/5">
              <td className="py-1.5 text-ink-400">Coverage</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {fmtPct(identity.coverage)}
              </td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-1.5 text-ink-400">Abstention</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {fmtPct(identity.abstention_rate)}
              </td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-1.5 text-ink-400">Labeled / matched entities</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {identity.n_labeled} / {identity.n_entities_matched}
              </td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-1.5 text-ink-400">Clusters</td>
              <td className="py-1.5 text-right font-mono text-ink-100">{identity.n_clusters}</td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-1.5 text-ink-400">Cluster purity</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {identity.cluster_purity != null ? fmtPct(identity.cluster_purity) : "—"}
              </td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-1.5 text-ink-400">Cluster completeness</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {identity.cluster_completeness != null ? fmtPct(identity.cluster_completeness) : "—"}
              </td>
            </tr>
            <tr className="border-b border-white/5">
              <td className="py-1.5 text-ink-400">Roster precision</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {identity.naming?.roster_precision != null
                  ? fmtPct(identity.naming.roster_precision)
                  : "—"}
              </td>
            </tr>
            <tr className="last:border-0">
              <td className="py-1.5 text-ink-400">Named / judged / correct</td>
              <td className="py-1.5 text-right font-mono text-ink-100">
                {identity.naming
                  ? `${identity.naming.n_named} / ${identity.naming.n_judged} / ${identity.naming.n_correct}`
                  : "—"}
              </td>
            </tr>
          </tbody>
        </table>
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
