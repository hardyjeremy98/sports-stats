// Labelled-data browser: what training data is on disk, and what its labels
// actually look like — sample images with boxes / pitch keypoints drawn on.

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchJson } from "../lib/api";
import type { Video } from "../lib/types";
import { Button, Card, EmptyState, ErrorNote, Mono, PageTitle, Spinner } from "../components/ui";

interface DatasetInfo {
  name: string;
  kind: string;
  images: number;
  annotations: number;
  classes: string[];
  splits: Record<string, number>;
}

interface Sample {
  url: string;
  split: string;
  boxes: { x: number; y: number; w: number; h: number; label: string }[];
  keypoints: { x: number; y: number; v: number }[];
}

const CLASS_COLORS: Record<string, string> = {
  player: "#2C91FF",
  goalkeeper: "#9BE532",
  referee: "#F5C518",
  ball: "#FFFFFF",
};

const KIND_LABELS: Record<string, string> = {
  "detection-coco": "detection boxes · COCO",
  "detection-yolo": "detection boxes · YOLO",
  "keypoints-yolo": "pitch keypoints · YOLO pose",
};

export default function LabDatasets() {
  const datasets = useQuery({
    queryKey: ["datasets"],
    queryFn: () => fetchJson<DatasetInfo[]>("/api/datasets"),
  });
  const [selected, setSelected] = useState<string | null>(null);
  const active = selected ?? datasets.data?.[0]?.name ?? null;

  if (datasets.isLoading) return <Spinner label="Scanning datasets" />;

  return (
    <div>
      <PageTitle
        title="Datasets"
        sub="Labelled training data on disk (data/experiments/datasets) — downloaded via pitchlab-train."
      />
      {(datasets.data ?? []).length === 0 ? (
        <EmptyState
          title="No labelled datasets downloaded"
          hint="Set ROBOFLOW_API_KEY and run a training config, or use the dataset adapter: the roboflow/sports player, ball, and pitch-keypoint sets are preconfigured in configs/train/."
        />
      ) : (
        <>
          <div className="mb-6 grid gap-3 md:grid-cols-3">
            {(datasets.data ?? []).map((d) => (
              <button key={d.name} onClick={() => setSelected(d.name)} className="text-left">
                <Card
                  className={`h-full p-4 transition-colors ${
                    active === d.name ? "border-volt-400/60 bg-volt-400/5" : "hover:border-white/20"
                  }`}
                >
                  <Mono className="block truncate text-volt-300">{d.name}</Mono>
                  <div className="mt-1 text-[12px] text-ink-400">{KIND_LABELS[d.kind] ?? d.kind}</div>
                  <div className="mt-3 flex items-baseline gap-4">
                    <Stat n={d.images} label="images" />
                    <Stat n={d.annotations} label="labels" />
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {d.classes.filter((c) => c in CLASS_COLORS || d.classes.length <= 5).map((c) => (
                      <span
                        key={c}
                        className="rounded-full border border-white/10 px-2 py-0.5 font-mono text-[10px] text-ink-400"
                        style={{ borderColor: CLASS_COLORS[c] ? `${CLASS_COLORS[c]}55` : undefined }}
                      >
                        {c}
                      </span>
                    ))}
                    <span className="ml-auto font-mono text-[10px] text-ink-500">
                      {Object.entries(d.splits)
                        .map(([s, n]) => `${s} ${n}`)
                        .join(" · ")}
                    </span>
                  </div>
                </Card>
              </button>
            ))}
          </div>
          {active && (
            <>
              <MaterializeBar
                dataset={(datasets.data ?? []).find((d) => d.name === active)!}
              />
              <SampleGrid name={active} />
            </>
          )}
        </>
      )}
    </div>
  );
}

function Stat({ n, label }: { n: number; label: string }) {
  return (
    <span>
      <span className="font-mono text-lg text-ink-100">{n.toLocaleString()}</span>
      <span className="ml-1.5 font-mono text-[10px] uppercase tracking-wider text-ink-500">{label}</span>
    </span>
  );
}

function MaterializeBar({ dataset }: { dataset: DatasetInfo }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const materialize = useMutation({
    mutationFn: (split: string) =>
      fetchJson<Video>(`/api/datasets/${dataset.name}/materialize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ split, fps: 2 }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["videos"] });
      navigate("/lab/new");
    },
  });

  return (
    <Card className="mb-4 flex flex-wrap items-center gap-3 p-4">
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium">Use as footage</div>
        <p className="mt-0.5 text-[12px] leading-relaxed text-ink-500">
          Stitches a split into a 2fps video and registers it as footage, so labelled frames can
          run through any pipeline config. Frames come from different matches — use these runs to
          judge <span className="text-ink-400">detection and calibration</span> frame-by-frame;
          tracklets and stats across unrelated frames are meaningless.
        </p>
      </div>
      <div className="flex items-center gap-2">
        {Object.entries(dataset.splits).map(([split, count]) => (
          <Button
            key={split}
            variant={split === "test" ? "primary" : "ghost"}
            disabled={materialize.isPending}
            onClick={() => materialize.mutate(split)}
          >
            {materialize.isPending && materialize.variables === split
              ? "Building…"
              : `${split} (${count})`}
          </Button>
        ))}
      </div>
      {materialize.isError && (
        <div className="w-full">
          <ErrorNote>{(materialize.error as Error).message}</ErrorNote>
        </div>
      )}
    </Card>
  );
}

function SampleGrid({ name }: { name: string }) {
  const samples = useQuery({
    queryKey: ["dataset-samples", name],
    queryFn: () => fetchJson<{ kind: string; samples: Sample[] }>(`/api/datasets/${name}/samples?n=12`),
    staleTime: Infinity,
  });

  if (samples.isLoading) return <Spinner label="Loading samples" />;
  const rows = samples.data?.samples ?? [];

  return (
    <Card className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">
          Labelled samples
        </div>
        <span className="text-[11px] text-ink-500">labels drawn from the dataset's own annotations</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {rows.map((s) => (
          <figure key={s.url} className="overflow-hidden rounded-lg border border-white/8">
            <div className="relative">
              <img src={s.url} className="block w-full" loading="lazy" />
              <svg
                className="pointer-events-none absolute inset-0 h-full w-full"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
              >
                {s.boxes.map((b, i) => (
                  <rect
                    key={i}
                    x={b.x * 100}
                    y={b.y * 100}
                    width={b.w * 100}
                    height={b.h * 100}
                    fill="none"
                    stroke={CLASS_COLORS[b.label] ?? "#9BE532"}
                    strokeWidth="0.4"
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                {s.keypoints
                  .filter((k) => k.v > 0 && (k.x > 0.001 || k.y > 0.001))
                  .map((k, i) => (
                    <circle key={i} cx={k.x * 100} cy={k.y * 100} r={0.8} fill="#9BE532" />
                  ))}
              </svg>
            </div>
            <figcaption className="flex items-center justify-between px-2 py-1 font-mono text-[10px] text-ink-500">
              <span>{s.split}</span>
              <span>
                {s.keypoints.length > 0
                  ? `${s.keypoints.filter((k) => k.v > 0).length} keypoints`
                  : `${s.boxes.length} boxes`}
              </span>
            </figcaption>
          </figure>
        ))}
      </div>
    </Card>
  );
}
