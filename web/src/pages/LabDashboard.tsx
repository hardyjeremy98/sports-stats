// Lab home: every run, live progress, key metrics, quick diff.

import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { fmtDuration, fmtWhen } from "../lib/format";
import { useRuns, useVideos } from "../lib/hooks";
import type { Run, RunStatus } from "../lib/types";
import { Button, Card, EmptyState, Mono, PageTitle, Spinner, StatusChip } from "../components/ui";

// idf1/idsw only exist for runs on ground-truth-labelled videos.
const METRIC_KEYS = ["n_tracklets", "n_players", "n_events", "n_qa_items", "idf1_entity", "idsw_entity"] as const;
const METRIC_LABELS: Record<string, string> = { idf1_entity: "idf1", idsw_entity: "idsw" };

type SortKey = (typeof METRIC_KEYS)[number] | "created_at" | "took";

function runTook(run: Run): number | null {
  return run.started_at && run.finished_at
    ? (new Date(run.finished_at).getTime() - new Date(run.started_at).getTime()) / 1000
    : null;
}

function sortValue(run: Run, key: SortKey): number | null {
  if (key === "created_at") return new Date(run.created_at).getTime();
  if (key === "took") return runTook(run);
  const v = run.metrics?.[key];
  if (v == null) return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

export default function LabDashboard() {
  const runs = useRuns();
  const videos = useVideos();
  const navigate = useNavigate();
  const [selected, setSelected] = useState<string[]>([]);

  const [videoFilter, setVideoFilter] = useState<number | "all">("all");
  const [configFilter, setConfigFilter] = useState<string | "all">("all");
  const [statusFilter, setStatusFilter] = useState<RunStatus | "all">("all");
  const [gtOnly, setGtOnly] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const videoName = useMemo(() => {
    const map = new Map<number, string>();
    for (const v of videos.data ?? []) map.set(v.id, v.filename);
    return (id: number) => map.get(id) ?? `video ${id}`;
  }, [videos.data]);

  const gtVideoIds = useMemo(
    () => new Set((videos.data ?? []).filter((v) => v.has_ground_truth).map((v) => v.id)),
    [videos.data],
  );

  const videoOptions = useMemo(
    () => (videos.data ?? []).map((v) => ({ id: v.id, filename: v.filename })),
    [videos.data],
  );
  const configOptions = useMemo(() => {
    const set = new Set<string>();
    for (const r of runs.data ?? []) set.add(r.config_name);
    return [...set].sort();
  }, [runs.data]);
  const statusOptions = useMemo(() => {
    const set = new Set<RunStatus>();
    for (const r of runs.data ?? []) set.add(r.status);
    return [...set];
  }, [runs.data]);

  const filtered = useMemo(() => {
    let list = runs.data ?? [];
    if (videoFilter !== "all") list = list.filter((r) => r.video_id === videoFilter);
    if (configFilter !== "all") list = list.filter((r) => r.config_name === configFilter);
    if (statusFilter !== "all") list = list.filter((r) => r.status === statusFilter);
    if (gtOnly) list = list.filter((r) => gtVideoIds.has(r.video_id));
    return list;
  }, [runs.data, videoFilter, configFilter, statusFilter, gtOnly, gtVideoIds]);

  const sorted = useMemo(() => {
    if (!sortKey) return filtered;
    const dir = sortDir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const va = sortValue(a, sortKey);
      const vb = sortValue(b, sortKey);
      if (va == null && vb == null) return 0;
      if (va == null) return 1; // missing metric sorts last regardless of direction
      if (vb == null) return -1;
      return (va - vb) * dir;
    });
  }, [filtered, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const toggle = (id: string) =>
    setSelected((cur) =>
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur.slice(-1), id],
    );

  const canDiff =
    selected.length === 2 &&
    selected.every((id) => runs.data?.find((r) => r.id === id)?.status === "completed");
  const selectedVideoId =
    selected.length === 1 ? runs.data?.find((r) => r.id === selected[0])?.video_id ?? null : null;

  const colCount = 5 + METRIC_KEYS.length + 2;

  return (
    <div>
      <PageTitle
        title="Runs"
        sub="Every pipeline execution, its config, and how it went."
        right={
          <div className="flex items-center gap-2">
            {selected.length === 2 && (
              <Button
                variant="ghost"
                disabled={!canDiff}
                onClick={() => navigate(`/lab/diff/${selected[0]}/${selected[1]}`)}
              >
                Diff selected
              </Button>
            )}
            <Button onClick={() => navigate("/lab/new")}>New run</Button>
          </div>
        }
      />

      {runs.isLoading ? (
        <Spinner label="Loading runs" />
      ) : (runs.data ?? []).length === 0 ? (
        <EmptyState
          title="No runs yet"
          hint="Create a run against any uploaded video, or seed demo data with: make demo"
          action={<Button onClick={() => navigate("/lab/new")}>New run</Button>}
        />
      ) : (
        <>
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <select
              value={videoFilter === "all" ? "" : videoFilter}
              onChange={(e) =>
                setVideoFilter(e.target.value === "" ? "all" : Number(e.target.value))
              }
              className="rounded-lg border border-white/10 bg-turf-850 px-3 py-1.5 text-[13px] text-ink-400 outline-none focus:border-volt-400/60"
            >
              <option value="">All videos</option>
              {videoOptions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.filename}
                </option>
              ))}
            </select>
            <select
              value={configFilter === "all" ? "" : configFilter}
              onChange={(e) => setConfigFilter(e.target.value === "" ? "all" : e.target.value)}
              className="rounded-lg border border-white/10 bg-turf-850 px-3 py-1.5 text-[13px] text-ink-400 outline-none focus:border-volt-400/60"
            >
              <option value="">All configs</option>
              {configOptions.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <select
              value={statusFilter === "all" ? "" : statusFilter}
              onChange={(e) =>
                setStatusFilter(e.target.value === "" ? "all" : (e.target.value as RunStatus))
              }
              className="rounded-lg border border-white/10 bg-turf-850 px-3 py-1.5 text-[13px] text-ink-400 outline-none focus:border-volt-400/60"
            >
              <option value="">All statuses</option>
              {statusOptions.map((s) => (
                <option key={s} value={s}>
                  {s[0].toUpperCase() + s.slice(1)}
                </option>
              ))}
            </select>
            <label className="flex items-center gap-1.5 text-[12px] text-ink-400">
              <input
                type="checkbox"
                checked={gtOnly}
                onChange={(e) => setGtOnly(e.target.checked)}
                className="accent-volt-400"
              />
              GT only
            </label>
          </div>

          <Card className="overflow-x-auto">
            <table className="w-full min-w-[900px] text-[13px]">
              <thead>
                <tr className="border-b border-white/8 text-left font-mono text-[10px] uppercase tracking-wider text-ink-500">
                  <th className="w-8 px-3 py-2.5" title="Select two completed runs to diff" />
                  <th className="px-3 py-2.5 font-normal">Run</th>
                  <th className="px-3 py-2.5 font-normal">Status</th>
                  <th className="px-3 py-2.5 font-normal">Config</th>
                  <th className="px-3 py-2.5 font-normal">Video</th>
                  {METRIC_KEYS.map((k) => (
                    <SortableHeader
                      key={k}
                      label={METRIC_LABELS[k] ?? k.replace("n_", "")}
                      sortKey={k}
                      activeKey={sortKey}
                      dir={sortDir}
                      onClick={toggleSort}
                    />
                  ))}
                  <SortableHeader
                    label="Took"
                    sortKey="took"
                    activeKey={sortKey}
                    dir={sortDir}
                    onClick={toggleSort}
                  />
                  <SortableHeader
                    label="Created"
                    sortKey="created_at"
                    activeKey={sortKey}
                    dir={sortDir}
                    onClick={toggleSort}
                  />
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <RunRow
                    key={r.id}
                    run={r}
                    videoName={videoName(r.video_id)}
                    selected={selected.includes(r.id)}
                    onToggle={() => toggle(r.id)}
                    otherSelectedVideoId={selectedVideoId}
                  />
                ))}
                {sorted.length === 0 && (
                  <tr>
                    <td colSpan={colCount} className="py-8 text-center text-ink-500">
                      No runs match the filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  );
}

function SortableHeader({
  label,
  sortKey,
  activeKey,
  dir,
  onClick,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey | null;
  dir: "asc" | "desc";
  onClick: (key: SortKey) => void;
}) {
  const active = sortKey === activeKey;
  return (
    <th className="px-3 py-2.5 text-right font-normal">
      <button
        onClick={() => onClick(sortKey)}
        className={`inline-flex items-center gap-1 transition-colors hover:text-ink-100 ${
          active ? "text-ink-100" : ""
        }`}
      >
        {label}
        {active && <span className="text-[9px]">{dir === "asc" ? "▲" : "▼"}</span>}
      </button>
    </th>
  );
}

function RunRow({
  run,
  videoName,
  selected,
  onToggle,
  otherSelectedVideoId,
}: {
  run: Run;
  videoName: string;
  selected: boolean;
  onToggle: () => void;
  otherSelectedVideoId: number | null;
}) {
  const took = runTook(run);
  const diffBlocked =
    !selected && otherSelectedVideoId != null && run.video_id !== otherSelectedVideoId;

  return (
    <tr className="border-b border-white/5 transition-colors last:border-0 hover:bg-turf-900/60">
      <td className="px-3 py-2.5">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          disabled={(run.status !== "completed" && !selected) || diffBlocked}
          title={diffBlocked ? "diff requires the same video" : undefined}
          className="accent-volt-400"
        />
      </td>
      <td className="px-3 py-2.5">
        <Link to={`/lab/runs/${run.id}`} className="group">
          <Mono className="text-volt-300 group-hover:underline">{run.id}</Mono>
          {run.label && <div className="text-[12px] text-ink-400">{run.label}</div>}
        </Link>
      </td>
      <td className="px-3 py-2.5">
        <StatusChip status={run.status} />
        {run.status === "running" && (
          <div className="mt-1.5 flex items-center gap-2">
            <div className="h-1 w-24 overflow-hidden rounded-full bg-turf-800">
              <div
                className="h-full bg-volt-400 transition-all"
                style={{ width: `${Math.round(run.progress_frac * 100)}%` }}
              />
            </div>
            <span className="font-mono text-[10px] text-ink-500">{run.progress_stage}</span>
          </div>
        )}
      </td>
      <td className="px-3 py-2.5 text-ink-400">{run.config_name}</td>
      <td className="max-w-[180px] truncate px-3 py-2.5 text-ink-400">{videoName}</td>
      {METRIC_KEYS.map((k) => (
        <td key={k} className="px-3 py-2.5 text-right font-mono text-ink-400">
          {run.metrics?.[k] ?? "—"}
        </td>
      ))}
      <td className="px-3 py-2.5 text-right font-mono text-ink-400">{fmtDuration(took)}</td>
      <td className="px-3 py-2.5 text-right text-ink-500">{fmtWhen(run.created_at)}</td>
    </tr>
  );
}
