// The user-facing deliverable: annotated video, per-player stat sheet, and a
// 2D minimap replay synced to the video.

import { useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../lib/api";
import { useRunArtifacts } from "../lib/artifacts";
import { fmtClock } from "../lib/format";
import { useRun } from "../lib/hooks";
import type { StatLine, Team } from "../lib/types";
import { PitchCanvas } from "../components/PitchCanvas";
import { Card, ErrorNote, PageTitle, Spinner, StatusChip, TeamDot } from "../components/ui";

export default function Report() {
  const { runId = "" } = useParams();
  const run = useRun(runId);
  const artifacts = useRunArtifacts(runId, run.data?.status === "completed");
  const videoRef = useRef<HTMLVideoElement>(null);
  const getTime = useMemo(() => () => videoRef.current?.currentTime ?? 0, []);

  if (run.isLoading) return <Spinner label="Loading report" />;
  if (run.isError) return <ErrorNote>{(run.error as Error).message}</ErrorNote>;
  const r = run.data!;

  if (r.status !== "completed") {
    return (
      <div className="mx-auto max-w-3xl">
        <PageTitle title={r.video?.filename ?? "Match report"} right={<StatusChip status={r.status} />} />
        {r.status === "failed" ? (
          <ErrorNote>{r.error ?? "Processing failed."}</ErrorNote>
        ) : (
          <Card className="p-8 text-center text-sm text-ink-400">
            Processing {r.progress_stage ? `— ${r.progress_stage}` : ""}… this page will update
            automatically.
            <div className="mx-auto mt-4 h-1 w-64 overflow-hidden rounded-full bg-turf-800">
              <div
                className="h-full bg-volt-400 transition-all"
                style={{ width: `${Math.round(r.progress_frac * 100)}%` }}
              />
            </div>
          </Card>
        )}
      </div>
    );
  }

  const hasAnnotated = !!r.manifest?.artifacts?.annotated_video;

  return (
    <div className="mx-auto max-w-6xl">
      <PageTitle
        title={r.video?.filename ?? "Match report"}
        sub={`Processed ${fmtClock(r.video?.duration_s ?? 0)} of footage`}
        right={<StatusChip status={r.status} />}
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="flex flex-col gap-6">
          <div className="overflow-hidden rounded-xl border border-white/8 bg-black">
            <video
              ref={videoRef}
              src={
                hasAnnotated
                  ? api.artifactUrl(runId, "annotated_video")
                  : `/api/videos/${r.video_id}/file`
              }
              controls
              playsInline
              className="block w-full"
            />
          </div>
          <Card className="p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-medium">Minimap replay</h2>
              <span className="text-[12px] text-ink-500">synced to the video</span>
            </div>
            <PitchCanvas
              minimap={artifacts.minimap}
              getTime={getTime}
              pitchName={(r.manifest?.config?.pitch as string | undefined) ?? undefined}
            />
          </Card>
        </div>

        <Card className="self-start p-4">
          <h2 className="mb-3 text-sm font-medium">Player stats</h2>
          {artifacts.loading ? (
            <Spinner />
          ) : artifacts.stats ? (
            <StatTable players={artifacts.stats.players} />
          ) : (
            <div className="py-6 text-center text-[13px] text-ink-500">No stats produced.</div>
          )}
        </Card>
      </div>
    </div>
  );
}

export function StatTable({ players }: { players: StatLine[] }) {
  const [team, setTeam] = useState<Team | "all">("all");
  const rows = players.filter((p) => team === "all" || p.team === team);
  return (
    <div>
      <div className="mb-2 flex items-center gap-1 text-[12px]">
        {(["all", "home", "away"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTeam(t)}
            className={`rounded-md px-2 py-1 capitalize transition-colors ${
              team === t ? "bg-turf-800 text-ink-100" : "text-ink-400 hover:text-ink-100"
            }`}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-white/8 text-left font-mono text-[10px] uppercase tracking-wider text-ink-500">
              <th className="py-2 pr-2 font-normal">Player</th>
              <th className="px-2 py-2 text-right font-normal" title="Completed passes">P</th>
              <th className="px-2 py-2 text-right font-normal" title="Missed passes">MP</th>
              <th className="px-2 py-2 text-right font-normal" title="Touches">T</th>
              <th className="px-2 py-2 text-right font-normal" title="Restarts">R</th>
              <th className="pl-2 py-2 text-right font-normal" title="Possession">Poss</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.player_id} className="border-b border-white/5 last:border-0">
                <td className="py-1.5 pr-2">
                  <span className="flex items-center gap-2">
                    <TeamDot team={p.team} />
                    {p.label}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right font-mono">{p.passes}</td>
                <td className="px-2 py-1.5 text-right font-mono">{p.missed_passes}</td>
                <td className="px-2 py-1.5 text-right font-mono">{p.touches}</td>
                <td className="px-2 py-1.5 text-right font-mono">{p.restarts}</td>
                <td className="py-1.5 pl-2 text-right font-mono">{fmtClock(p.possession_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
