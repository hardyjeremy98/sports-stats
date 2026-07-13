// Run creation: pick a video + base config, tweak stage implementations (the
// dropdowns rewrite the YAML), submit as ad-hoc config_yaml.

import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import YAML from "yaml";
import { fmtClock } from "../lib/format";
import { useConfigs, useCreateRun, useRegistry, useUploadVideo, useVideos } from "../lib/hooks";
import { Button, Card, ErrorNote, PageTitle, Spinner } from "../components/ui";

export default function LabNewRun() {
  const videos = useVideos();
  const configs = useConfigs();
  const registry = useRegistry();
  const createRun = useCreateRun();
  const upload = useUploadVideo();
  const navigate = useNavigate();

  const [videoId, setVideoId] = useState<number | null>(null);
  const [baseName, setBaseName] = useState<string | null>(null);
  const [yamlText, setYamlText] = useState("");
  const [label, setLabel] = useState("");
  const [yamlError, setYamlError] = useState<string | null>(null);

  // Defaults once data arrives.
  useEffect(() => {
    if (videoId == null && videos.data?.length) setVideoId(videos.data[0].id);
  }, [videos.data, videoId]);
  useEffect(() => {
    if (baseName == null && configs.data?.length) {
      const base = configs.data[0];
      setBaseName(base.name);
      setYamlText(base.yaml);
    }
  }, [configs.data, baseName]);

  const parsed = useMemo(() => {
    try {
      const doc = YAML.parse(yamlText) as {
        stages?: Record<string, { impl?: string; enabled?: boolean }>;
      } | null;
      setYamlError(null);
      return doc;
    } catch (e) {
      setYamlError((e as Error).message);
      return null;
    }
  }, [yamlText]);

  const pickBase = (name: string) => {
    const cfg = configs.data?.find((c) => c.name === name);
    if (cfg) {
      setBaseName(name);
      setYamlText(cfg.yaml);
    }
  };

  const setStageImpl = (stage: string, impl: string) => {
    try {
      const doc = YAML.parseDocument(yamlText);
      doc.setIn(["stages", stage, "impl"], impl);
      // A different implementation almost never shares params; reset them.
      doc.setIn(["stages", stage, "params"], {});
      setYamlText(doc.toString());
    } catch {
      // YAML currently invalid; dropdown edits resume when it parses again.
    }
  };

  const submit = () => {
    if (videoId == null || !parsed) return;
    createRun.mutate(
      { video_id: videoId, config_yaml: yamlText, label: label || undefined },
      { onSuccess: (run) => navigate(`/lab/runs/${run.id}`) },
    );
  };

  if (videos.isLoading || configs.isLoading) return <Spinner label="Loading" />;

  return (
    <div className="mx-auto max-w-5xl">
      <PageTitle title="New run" sub="Pick footage, choose a recipe, tweak the stages, launch." />

      <div className="grid gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <Card className="p-4">
            <SectionLabel>1 · Footage</SectionLabel>
            <select
              value={videoId ?? ""}
              onChange={(e) => setVideoId(Number(e.target.value))}
              className="mt-2 w-full rounded-lg border border-white/10 bg-turf-850 px-3 py-2 text-[13px] outline-none focus:border-volt-400/60"
            >
              {(videos.data ?? []).map((v) => (
                <option key={v.id} value={v.id}>
                  {v.filename} ({fmtClock(v.duration_s)})
                </option>
              ))}
            </select>
            <label className="mt-3 block cursor-pointer text-[12px] text-ink-400 hover:text-ink-100">
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f)
                    upload.mutate(f, { onSuccess: (v) => setVideoId(v.id) });
                }}
              />
              {upload.isPending ? "Uploading…" : "+ upload new footage"}
            </label>
          </Card>

          <Card className="p-4">
            <SectionLabel>2 · Base config</SectionLabel>
            <div className="mt-2 flex flex-col gap-1.5">
              {(configs.data ?? []).map((c) => (
                <button
                  key={c.name}
                  onClick={() => pickBase(c.name)}
                  className={`rounded-lg border px-3 py-2 text-left transition-colors ${
                    baseName === c.name
                      ? "border-volt-400/60 bg-volt-400/10"
                      : "border-white/8 hover:border-white/20"
                  }`}
                >
                  <div className="text-[13px] font-medium">{c.name}</div>
                  <div className="mt-0.5 line-clamp-2 text-[12px] text-ink-400">{c.description}</div>
                </button>
              ))}
            </div>
          </Card>

          <Card className="p-4">
            <SectionLabel>3 · Stage implementations</SectionLabel>
            <div className="mt-2 flex flex-col gap-2">
              {Object.entries(registry.data?.stages ?? {}).map(([stage, impls]) => {
                const current = parsed?.stages?.[stage];
                if (!current) return null;
                return (
                  <div key={stage} className="flex items-center justify-between gap-3">
                    <span className="font-mono text-[11px] uppercase tracking-wider text-ink-500">
                      {stage}
                      {current.enabled === false && (
                        <span className="ml-1.5 text-ink-500/60">(off)</span>
                      )}
                    </span>
                    <select
                      value={current.impl ?? ""}
                      onChange={(e) => setStageImpl(stage, e.target.value)}
                      className="w-44 rounded-md border border-white/10 bg-turf-850 px-2 py-1 text-[12px] outline-none focus:border-volt-400/60"
                    >
                      {impls.map((impl) => (
                        <option key={impl} value={impl}>
                          {impl}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 text-[11px] leading-relaxed text-ink-500">
              Changing an implementation resets that stage's params — fine-tune them in the YAML.
            </p>
          </Card>
        </div>

        <div className="flex flex-col gap-4">
          <Card className="flex min-h-[480px] flex-col p-4">
            <SectionLabel>Pipeline config (YAML)</SectionLabel>
            <textarea
              value={yamlText}
              onChange={(e) => setYamlText(e.target.value)}
              spellCheck={false}
              className="mt-2 flex-1 resize-none rounded-lg border border-white/10 bg-turf-950 p-3 font-mono text-[12px] leading-relaxed text-ink-100 outline-none focus:border-volt-400/60"
            />
            {yamlError && <ErrorNote>YAML: {yamlError}</ErrorNote>}
          </Card>

          <Card className="flex items-center gap-3 p-4">
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="Label (optional) — e.g. 'botsort, stride 2'"
              className="flex-1 rounded-lg border border-white/10 bg-turf-850 px-3 py-2 text-[13px] outline-none placeholder:text-ink-500 focus:border-volt-400/60"
            />
            <Button
              onClick={submit}
              disabled={videoId == null || !!yamlError || createRun.isPending}
            >
              {createRun.isPending ? "Launching…" : "Launch run"}
            </Button>
          </Card>
          {createRun.isError && <ErrorNote>{(createRun.error as Error).message}</ErrorNote>}
        </div>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">{children}</div>
  );
}
