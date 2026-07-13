// Lab home: every run, live progress, key metrics, quick diff.

import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { fmtDuration, fmtWhen } from "../lib/format";
import { useRuns, useVideos } from "../lib/hooks";
import type { Run } from "../lib/types";
import { Button, Card, EmptyState, Mono, PageTitle, Spinner, StatusChip } from "../components/ui";

// idf1/idsw only exist for runs on ground-truth-labelled videos.
const METRIC_KEYS = ["n_tracklets", "n_players", "n_events", "n_qa_items", "idf1_entity", "idsw_entity"] as const;
const METRIC_LABELS: Record<string, string> = { idf1_entity: "idf1", idsw_entity: "idsw" };

export default function LabDashboard() {
  const runs = useRuns();
  const videos = useVideos();
  const navigate = useNavigate();
  const [selected, setSelected] = useState<string[]>([]);

  const videoName = useMemo(() => {
    const map = new Map<number, string>();
    for (const v of videos.data ?? []) map.set(v.id, v.filename);
    return (id: number) => map.get(id) ?? `video ${id}`;
  }, [videos.data]);

  const toggle = (id: string) =>
    setSelected((cur) =>
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur.slice(-1), id],
    );

  const canDiff =
    selected.length === 2 &&
    selected.every((id) => runs.data?.find((r) => r.id === id)?.status === "completed");

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
                  <th key={k} className="px-3 py-2.5 text-right font-normal">
                    {METRIC_LABELS[k] ?? k.replace("n_", "")}
                  </th>
                ))}
                <th className="px-3 py-2.5 text-right font-normal">Took</th>
                <th className="px-3 py-2.5 text-right font-normal">Created</th>
              </tr>
            </thead>
            <tbody>
              {(runs.data ?? []).map((r) => (
                <RunRow
                  key={r.id}
                  run={r}
                  videoName={videoName(r.video_id)}
                  selected={selected.includes(r.id)}
                  onToggle={() => toggle(r.id)}
                />
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}

function RunRow({
  run,
  videoName,
  selected,
  onToggle,
}: {
  run: Run;
  videoName: string;
  selected: boolean;
  onToggle: () => void;
}) {
  const took =
    run.started_at && run.finished_at
      ? (new Date(run.finished_at).getTime() - new Date(run.started_at).getTime()) / 1000
      : null;

  return (
    <tr className="border-b border-white/5 transition-colors last:border-0 hover:bg-turf-900/60">
      <td className="px-3 py-2.5">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          disabled={run.status !== "completed" && !selected}
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
