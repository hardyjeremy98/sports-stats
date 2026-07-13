// Global QA queue: review contested/low-confidence events across all runs.
// Every decision writes a labeled example — the v2 training set.

import { useMemo, useState } from "react";
import { useQA, useRuns } from "../lib/hooks";
import type { QAStatus } from "../lib/types";
import { QATab } from "./LabRunViewer";
import { Card, PageTitle, Spinner } from "../components/ui";

const FILTERS: { id: QAStatus | "all"; label: string }[] = [
  { id: "pending", label: "Pending" },
  { id: "accepted", label: "Accepted" },
  { id: "corrected", label: "Corrected" },
  { id: "rejected", label: "Rejected" },
  { id: "all", label: "All" },
];

export default function LabQA() {
  const [status, setStatus] = useState<QAStatus | "all">("pending");
  const [runId, setRunId] = useState<string>("");
  const qa = useQA({
    run_id: runId || undefined,
    status: status === "all" ? undefined : status,
  });
  const allQa = useQA(); // for the labels-collected counter
  const runs = useRuns();

  const labelsCollected = useMemo(
    () =>
      (allQa.data ?? []).filter((r) => r.status === "accepted" || r.status === "corrected")
        .length,
    [allQa.data],
  );

  return (
    <div className="mx-auto max-w-4xl">
      <PageTitle
        title="QA queue"
        sub="Contested and low-confidence pipeline decisions. Each verdict becomes a training label."
        right={
          <div className="rounded-lg border border-white/8 bg-turf-900 px-4 py-2 text-right">
            <div className="font-mono text-xl text-volt-300">{labelsCollected}</div>
            <div className="font-mono text-[10px] uppercase tracking-wider text-ink-500">
              labels collected
            </div>
          </div>
        }
      />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setStatus(f.id)}
              className={`rounded-md px-2.5 py-1 text-[12px] transition-colors ${
                status === f.id
                  ? "bg-turf-800 text-ink-100"
                  : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <select
          value={runId}
          onChange={(e) => setRunId(e.target.value)}
          className="ml-auto rounded-lg border border-white/10 bg-turf-850 px-3 py-1.5 text-[12px] text-ink-400 outline-none focus:border-volt-400/60"
        >
          <option value="">All runs</option>
          {(runs.data ?? []).map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} — {r.label ?? r.config_name}
            </option>
          ))}
        </select>
      </div>

      <Card className="p-3">
        {qa.isLoading ? <Spinner label="Loading queue" /> : <QATab records={qa.data ?? []} showRunId />}
      </Card>
    </div>
  );
}
