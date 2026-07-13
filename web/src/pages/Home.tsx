// User-facing home: upload a match, see your videos, jump to the report.

import { useCallback, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useRuns, useUploadVideo, useVideos } from "../lib/hooks";
import { fmtBytes, fmtClock, fmtWhen } from "../lib/format";
import type { Run, Video } from "../lib/types";
import { Card, EmptyState, ErrorNote, PageTitle, Spinner, StatusChip } from "../components/ui";
import { PitchMark } from "../components/Shell";

export default function Home() {
  const videos = useVideos();
  const runs = useRuns();
  const upload = useUploadVideo();
  const [dragOver, setDragOver] = useState(false);

  const onFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) upload.mutate(file);
    },
    [upload],
  );

  const runsByVideo = useMemo(() => {
    const map = new Map<number, Run[]>();
    for (const r of runs.data ?? []) {
      const list = map.get(r.video_id) ?? [];
      list.push(r);
      map.set(r.video_id, list);
    }
    return map;
  }, [runs.data]);

  return (
    <div className="mx-auto max-w-4xl">
      <PageTitle
        title="Your matches"
        sub="Upload a match video — we return per-player stats, a minimap replay, and an annotated video."
      />

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          onFiles(e.dataTransfer.files);
        }}
        className={`mb-8 flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed py-12 transition-colors ${
          dragOver
            ? "border-volt-400 bg-volt-400/5"
            : "border-white/10 hover:border-white/25"
        }`}
      >
        <input
          type="file"
          accept="video/mp4,video/quicktime,video/*"
          className="hidden"
          onChange={(e) => onFiles(e.target.files)}
        />
        <PitchMark className="h-7 w-11 text-volt-400" />
        <div className="text-sm">
          {upload.isPending ? "Uploading…" : "Drop a match video here, or click to browse"}
        </div>
        <div className="text-[12px] text-ink-500">One ground-level phone recording is all it takes.</div>
      </label>
      {upload.isError && <ErrorNote>{(upload.error as Error).message}</ErrorNote>}

      {videos.isLoading ? (
        <Spinner label="Loading videos" />
      ) : (videos.data ?? []).length === 0 ? (
        <EmptyState
          title="No matches yet"
          hint="Upload your first match above, or seed a demo from the terminal: make demo"
        />
      ) : (
        <div className="flex flex-col gap-3">
          {(videos.data ?? []).map((v) => (
            <VideoRow key={v.id} video={v} runs={runsByVideo.get(v.id) ?? []} />
          ))}
        </div>
      )}
    </div>
  );
}

function VideoRow({ video, runs }: { video: Video; runs: Run[] }) {
  const completed = runs.filter((r) => r.status === "completed");
  const latest = completed[0]; // useRuns is created_at desc
  const active = runs.find((r) => r.status === "running" || r.status === "queued");

  return (
    <Card className="flex items-center gap-4 p-4">
      <div className="flex h-11 w-16 items-center justify-center rounded-lg bg-turf-800 text-ink-500">
        <PitchMark className="h-4 w-7" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium">{video.filename}</div>
        <div className="mt-0.5 text-[12px] text-ink-400">
          {fmtClock(video.duration_s)} · {video.width}×{video.height} · {fmtBytes(video.size_bytes)} ·{" "}
          {fmtWhen(video.created_at)}
        </div>
      </div>
      <div className="flex items-center gap-4">
        {active ? (
          <div className="flex items-center gap-3">
            <StatusChip status={active.status} />
            {active.status === "running" && (
              <div className="h-1 w-28 overflow-hidden rounded-full bg-turf-800">
                <div
                  className="h-full bg-volt-400 transition-all"
                  style={{ width: `${Math.round(active.progress_frac * 100)}%` }}
                />
              </div>
            )}
          </div>
        ) : latest ? (
          <Link
            to={`/report/${latest.id}`}
            className="rounded-lg bg-volt-400 px-3.5 py-1.5 text-[13px] font-medium text-turf-950 transition-colors hover:bg-volt-300"
          >
            View report
          </Link>
        ) : runs.some((r) => r.status === "failed") ? (
          <StatusChip status="failed" />
        ) : (
          <span className="text-[12px] text-ink-500">not processed</span>
        )}
      </div>
    </Card>
  );
}
