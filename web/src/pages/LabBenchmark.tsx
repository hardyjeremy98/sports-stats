// Batch benchmark matrix: config x GT-video, aggregating repeat completed
// runs into mean/range cells (ADR 004 — batch views aggregate GT metrics,
// never raw artifact counts).

import { Fragment, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useBenchmark } from "../lib/hooks";
import { signalColor } from "../lib/colors";
import { fmtDuration } from "../lib/format";
import type { BenchmarkCell, BenchmarkGroup } from "../lib/types";
import { Button, Card, EmptyState, Mono, PageTitle, Spinner, ErrorNote } from "../components/ui";

interface MetricDef {
  key: string;
  label: string;
  higherIsBetter: boolean;
  idsw?: boolean;
  seconds?: boolean;
}

const METRICS: MetricDef[] = [
  { key: "idf1_entity", label: "IDF1 (entity)", higherIsBetter: true },
  { key: "idf1_tracklet", label: "IDF1 (tracklet)", higherIsBetter: true },
  { key: "hota_entity", label: "HOTA (entity)", higherIsBetter: true },
  { key: "hota_tracklet", label: "HOTA (tracklet)", higherIsBetter: true },
  { key: "idsw_entity", label: "IDSW (entity)", higherIsBetter: false, idsw: true },
  // Flicker-insensitive: only switches where the new identity persists >=1s
  // count (spec: docs/superpowers/specs/2026-07-23-persistent-idsw-metric-design.md).
  { key: "idsw_persistent_entity", label: "IDSW ≥1s (entity)", higherIsBetter: false, idsw: true },
  { key: "idsw_persistent_tracklet", label: "IDSW ≥1s (tracklet)", higherIsBetter: false, idsw: true },
  { key: "mota_entity", label: "MOTA (entity)", higherIsBetter: true },
  { key: "assoc_idf1_gain", label: "Assoc gain", higherIsBetter: true },
  // Not backfillable: a null merge_precision means the associator merged
  // nothing (no pairs to score), not that the run predates the layer.
  { key: "merge_precision", label: "Merge precision", higherIsBetter: true },
  { key: "identity_coverage", label: "Coverage", higherIsBetter: true },
  // Two different purity metrics at two different layers are selectable here:
  // "Tracklet purity" is SPO-6's (does a tracklet mix two players?), "Cluster
  // purity" is the semantic identity layer's (ADR 004). Neither may be
  // labelled a bare "Purity" — the matrix would not say which layer it means.
  { key: "tracklet_purity", label: "Tracklet purity", higherIsBetter: true },
  { key: "mixed_track_seconds", label: "Mixed-identity time", higherIsBetter: false, seconds: true },
  { key: "cluster_purity", label: "Cluster purity", higherIsBetter: true },
  // SPO-49: action-spotting avg-mAP@1 (pure spotting runs, no tracking keys).
  { key: "spotting_map_at_1", label: "Spotting mAP@1", higherIsBetter: true },
  // SPO-69: pitch-space game-state metrics -- score the calibration's
  // projected geometry (via GT tracks), not the tracker.
  { key: "gs_coverage", label: "Calib coverage", higherIsBetter: true },
  { key: "gs_implausible_speed_rate", label: "Calib impl. speed", higherIsBetter: false },
  { key: "gs_teleports", label: "Calib teleports", higherIsBetter: false, idsw: true },
  { key: "gs_in_bounds_rate", label: "Calib in-bounds", higherIsBetter: true },
];

// Metrics whose absence means the run predates the layer that computes them
// (semantic identity — ADR 004; purity/HOTA — SPO-6/SPO-7; game-state —
// SPO-69) rather than a failure. Re-evaluating on the run viewer backfills
// them.
const BACKFILLABLE_METRIC_KEYS = new Set([
  "identity_coverage",
  "cluster_purity",
  "hota_entity",
  "hota_tracklet",
  "tracklet_purity",
  "mixed_track_seconds",
  "idsw_persistent_entity",
  "idsw_persistent_tracklet",
  "gs_coverage",
  "gs_implausible_speed_rate",
  "gs_teleports",
  "gs_in_bounds_rate",
]);

function fmtMetric(def: MetricDef, v: number): string {
  if (def.seconds) return fmtDuration(v);
  if (def.idsw) {
    const r = Math.round(v * 10) / 10;
    return Number.isInteger(r) ? String(r) : r.toFixed(1);
  }
  return v.toFixed(3);
}

function cellKey(group: BenchmarkGroup): string {
  return `${group.config_name}::${group.config_hash}`;
}

export default function LabBenchmark() {
  const benchmark = useBenchmark();
  const navigate = useNavigate();
  const [metric, setMetric] = useState<MetricDef>(METRICS[0]);
  const [expanded, setExpanded] = useState<{ group: string; videoId: string } | null>(null);

  const domain = useMemo(() => {
    const groups = benchmark.data?.groups ?? [];
    let min = Infinity;
    let max = -Infinity;
    for (const g of groups) {
      for (const cell of Object.values(g.cells)) {
        const v = cell.metrics_mean[metric.key];
        if (v == null) continue;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return min <= max ? { min, max } : null;
  }, [benchmark.data, metric]);

  function colorFor(v: number): string {
    if (!domain) return "transparent";
    const norm = domain.max > domain.min ? (v - domain.min) / (domain.max - domain.min) : 0.5;
    const t = metric.higherIsBetter ? norm : 1 - norm;
    return signalColor(t);
  }

  if (benchmark.isLoading) return <Spinner label="Loading benchmark" />;
  if (benchmark.isError) return <ErrorNote>{(benchmark.error as Error).message}</ErrorNote>;

  const data = benchmark.data!;
  const groups = data.groups;

  return (
    <div>
      <PageTitle
        title="Benchmark"
        sub="Config x ground-truth-video matrix, aggregated over every completed GT-scored run."
      />

      {groups.length === 0 ? (
        <EmptyState
          title="No benchmark data yet"
          hint="Run a pipeline config against a video with ground truth — completed, GT-scored runs group into this matrix automatically."
          action={<Button onClick={() => navigate("/lab/new")}>New run</Button>}
        />
      ) : (
        <>
          <div className="mb-4 flex flex-wrap items-center gap-1.5">
            {METRICS.map((m) => (
              <button
                key={m.key}
                onClick={() => setMetric(m)}
                className={`rounded-md px-2.5 py-1 text-[12px] transition-colors ${
                  metric.key === m.key
                    ? "bg-turf-800 text-ink-100"
                    : "text-ink-400 hover:bg-turf-900 hover:text-ink-100"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          <Card className="overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-white/8 text-left font-mono text-[10px] uppercase tracking-wider text-ink-500">
                  <th className="px-3 py-2.5 font-normal">Config</th>
                  {data.videos.map((v) => (
                    <th key={v.video_id} className="max-w-[140px] px-3 py-2.5 text-right font-normal">
                      <span className="block truncate" title={v.filename}>
                        {v.filename}
                      </span>
                    </th>
                  ))}
                  <th className="px-3 py-2.5 text-right font-normal text-ink-400">mean</th>
                </tr>
              </thead>
              <tbody>
                {groups.map((g) => {
                  const key = cellKey(g);
                  const cellsForMetric = data.videos
                    .map((v) => g.cells[String(v.video_id)]?.metrics_mean[metric.key])
                    .filter((v): v is number => v != null);
                  const groupMean =
                    cellsForMetric.length > 0
                      ? cellsForMetric.reduce((a, b) => a + b, 0) / cellsForMetric.length
                      : null;
                  return (
                    <Fragment key={key}>
                      <tr className="border-b border-white/5 last:border-0">
                        <td className="px-3 py-2.5">
                          <div className="flex items-center gap-2">
                            <span className="text-ink-100">{g.config_name}</span>
                            <Mono className="text-ink-500">{g.config_hash.slice(0, 8)}</Mono>
                            <span className="rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-400">
                              {g.n_runs} run{g.n_runs === 1 ? "" : "s"}
                            </span>
                          </div>
                        </td>
                        {data.videos.map((v) => {
                          const cell = g.cells[String(v.video_id)];
                          return (
                            <MatrixCell
                              key={v.video_id}
                              metric={metric}
                              cell={cell}
                              colorFor={colorFor}
                              expanded={expanded?.group === key && expanded?.videoId === String(v.video_id)}
                              onToggle={() =>
                                setExpanded((cur) =>
                                  cur?.group === key && cur?.videoId === String(v.video_id)
                                    ? null
                                    : cell
                                      ? { group: key, videoId: String(v.video_id) }
                                      : null,
                                )
                              }
                            />
                          );
                        })}
                        <td className="px-3 py-2.5 text-right font-mono text-ink-100">
                          {groupMean != null ? fmtMetric(metric, groupMean) : "—"}
                        </td>
                      </tr>
                      {expanded?.group === key && (
                        <ExpandedRow
                          colSpan={data.videos.length + 2}
                          cell={g.cells[expanded.videoId]}
                        />
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  );
}

function MatrixCell({
  metric,
  cell,
  colorFor,
  expanded,
  onToggle,
}: {
  metric: MetricDef;
  cell: BenchmarkCell | undefined;
  colorFor: (v: number) => string;
  expanded: boolean;
  onToggle: () => void;
}) {
  if (!cell) {
    return (
      <td className="px-3 py-2.5 text-right font-mono text-ink-600" title="no runs for this config/video">
        —
      </td>
    );
  }
  const mean = cell.metrics_mean[metric.key];
  // The cell has run data, but not for this particular metric (e.g. older
  // runs that predate the layer computing it) — still clickable to inspect the
  // underlying runs, just no value/color to show.
  if (mean == null) {
    const title = BACKFILLABLE_METRIC_KEYS.has(metric.key)
      ? "re-evaluate run(s) to backfill this metric"
      : "no data for this metric";
    return (
      <td className="px-1.5 py-1.5 text-right">
        <button
          onClick={onToggle}
          className={`w-full rounded-md px-2.5 py-1.5 text-right font-mono text-ink-600 transition-colors hover:bg-turf-800 ${
            expanded ? "ring-1 ring-inset ring-volt-400/60" : ""
          }`}
          title={`${title} — ${cell.n_runs} run${cell.n_runs === 1 ? "" : "s"}, click for run list`}
        >
          —
        </button>
      </td>
    );
  }
  const range = cell.metrics_range[metric.key];
  const spread = range ? (range[1] - range[0]) / 2 : null;
  return (
    <td className="px-1.5 py-1.5 text-right">
      <button
        onClick={onToggle}
        className={`w-full rounded-md px-2.5 py-1.5 text-right font-mono transition-colors ${
          expanded ? "ring-1 ring-inset ring-volt-400/60" : ""
        }`}
        style={{ background: colorFor(mean) }}
        title={`${cell.n_runs} run${cell.n_runs === 1 ? "" : "s"} — click for run list`}
      >
        <span className="text-turf-950">{fmtMetric(metric, mean)}</span>
        {spread != null && (
          <div className="text-[10px] text-turf-950/70">±{fmtMetric(metric, spread)}</div>
        )}
      </button>
    </td>
  );
}

function ExpandedRow({ colSpan, cell }: { colSpan: number; cell: BenchmarkCell | undefined }) {
  if (!cell) return null;
  return (
    <tr className="border-b border-white/5 bg-turf-950/40 last:border-0">
      <td colSpan={colSpan} className="px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
            Runs
          </span>
          {cell.run_ids.map((id) => (
            <Link
              key={id}
              to={`/lab/runs/${id}`}
              className="rounded-md bg-turf-800 px-2 py-1 font-mono text-[11px] text-volt-300 hover:underline"
            >
              {id}
            </Link>
          ))}
        </div>
      </td>
    </tr>
  );
}
