import type {
  IdentityLabel,
  IdentityLabelKind,
  IdentityLabelPayload,
  PipelineConfigOut,
  QARecord,
  Registry,
  Run,
  RunDetail,
  RunDiff,
  Video,
} from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // non-JSON error body
    }
    throw new Error(detail);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),

  videos: () => request<Video[]>("/api/videos"),
  video: (id: number) => request<Video>(`/api/videos/${id}`),
  uploadVideo: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Video>("/api/videos", { method: "POST", body: form });
  },
  videoFileUrl: (id: number) => `/api/videos/${id}/file`,
  videoGtUrl: (id: number) => `/api/videos/${id}/ground_truth`,
  evaluateRun: (id: string) => request<RunDetail>(`/api/runs/${id}/evaluate`, { method: "POST" }),

  configs: () => request<PipelineConfigOut[]>("/api/configs"),
  registry: () => request<Registry>("/api/registry"),

  runs: (videoId?: number) =>
    request<Run[]>(videoId != null ? `/api/runs?video_id=${videoId}` : "/api/runs"),
  run: (id: string) => request<RunDetail>(`/api/runs/${id}`),
  createRun: (body: {
    video_id: number;
    config_name?: string;
    config_yaml?: string;
    label?: string;
  }) =>
    request<Run>("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  diff: (a: string, b: string) => request<RunDiff>(`/api/runs/${a}/diff/${b}`),
  artifactUrl: (runId: string, name: string) => `/api/runs/${runId}/artifacts/${name}`,
  runFileUrl: (runId: string, relPath: string) => `/api/runs/${runId}/files/${relPath}`,

  qa: (params?: { run_id?: string; status?: string }) => {
    const q = new URLSearchParams();
    if (params?.run_id) q.set("run_id", params.run_id);
    if (params?.status) q.set("status", params.status);
    const qs = q.toString();
    return request<QARecord[]>(`/api/qa${qs ? `?${qs}` : ""}`);
  },
  qaAccept: (id: number) => request<QARecord>(`/api/qa/${id}/accept`, { method: "POST" }),
  qaReject: (id: number) => request<QARecord>(`/api/qa/${id}/reject`, { method: "POST" }),
  qaCorrect: (
    id: number,
    body: { player_id?: number | null; event_type?: string | null; note?: string | null },
  ) =>
    request<QARecord>(`/api/qa/${id}/correct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),

  identityQa: (params?: { run_id?: string; kind?: IdentityLabelKind }) => {
    const q = new URLSearchParams();
    if (params?.run_id) q.set("run_id", params.run_id);
    if (params?.kind) q.set("kind", params.kind);
    const qs = q.toString();
    return request<IdentityLabel[]>(`/api/identity_qa${qs ? `?${qs}` : ""}`);
  },
  createIdentityLabel: (body: {
    run_id: string;
    kind: IdentityLabelKind;
    payload: IdentityLabelPayload;
    note?: string | null;
  }) =>
    request<IdentityLabel>("/api/identity_qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteIdentityLabel: (id: number) =>
    request<{ ok: boolean }>(`/api/identity_qa/${id}`, { method: "DELETE" }),
};

export async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  return request<T>(url, init);
}

export async function fetchJsonl<T>(url: string): Promise<T[]> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  const text = await resp.text();
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as T);
}
