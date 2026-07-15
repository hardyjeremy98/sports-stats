import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import type { IdentityLabelKind, Run } from "./types";

const ACTIVE = (runs: Run[] | undefined) =>
  runs?.some((r) => r.status === "queued" || r.status === "running") ?? false;

export function useVideos() {
  return useQuery({ queryKey: ["videos"], queryFn: api.videos });
}

export function useConfigs() {
  return useQuery({ queryKey: ["configs"], queryFn: api.configs, staleTime: 60_000 });
}

export function useRegistry() {
  return useQuery({ queryKey: ["registry"], queryFn: api.registry, staleTime: Infinity });
}

export function useRuns(videoId?: number) {
  return useQuery({
    queryKey: ["runs", videoId ?? "all"],
    queryFn: () => api.runs(videoId),
    // Poll while anything is in flight so status chips/progress bars live-update.
    refetchInterval: (query) => (ACTIVE(query.state.data) ? 2000 : false),
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.run(runId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "queued" || s === "running" ? 2000 : false;
    },
  });
}

export function useEvaluateRun(runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.evaluateRun(runId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["run", runId] });
      qc.invalidateQueries({ queryKey: ["artifact", runId, "eval"] });
    },
  });
}

export function useRunDiff(a: string, b: string) {
  return useQuery({
    queryKey: ["diff", a, b],
    queryFn: () => api.diff(a, b),
    staleTime: Infinity,
  });
}

export function useQA(params?: { run_id?: string; status?: string }) {
  return useQuery({
    queryKey: ["qa", params?.run_id ?? "all", params?.status ?? "all"],
    queryFn: () => api.qa(params),
  });
}

export function useQAActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: ["qa"] });
  const accept = useMutation({ mutationFn: api.qaAccept, onSuccess: invalidate });
  const reject = useMutation({ mutationFn: api.qaReject, onSuccess: invalidate });
  const correct = useMutation({
    mutationFn: ({ id, ...body }: { id: number; player_id?: number | null; event_type?: string | null; note?: string | null }) =>
      api.qaCorrect(id, body),
    onSuccess: invalidate,
  });
  return { accept, reject, correct };
}

export function useIdentityQA(runId?: string, kind?: IdentityLabelKind) {
  return useQuery({
    queryKey: ["identity_qa", runId ?? "all", kind ?? "all"],
    queryFn: () => api.identityQa({ run_id: runId, kind }),
  });
}

export function useIdentityQAActions() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: ["identity_qa"] });
  const create = useMutation({ mutationFn: api.createIdentityLabel, onSuccess: invalidate });
  const remove = useMutation({ mutationFn: api.deleteIdentityLabel, onSuccess: invalidate });
  return { create, remove };
}

export function useCreateRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.createRun,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["runs"] }),
  });
}

export function useUploadVideo() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.uploadVideo,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["videos"] }),
  });
}
