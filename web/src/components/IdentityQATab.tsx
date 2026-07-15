// Identity QA: a pair queue seeded from association near-misses and eval
// switches, plus a run-scoped label ledger. Labels are annotations only —
// they never mutate this run's tracklets/entities; merge/split/roster flags
// take effect on a future re-run or training pass.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";
import type { RunArtifacts } from "../lib/artifacts";
import { useIdentityQA, useIdentityQAActions } from "../lib/hooks";
import type {
  IdentityLabel,
  MergePayload,
  PairPayload,
  PairSource,
  PairVerdict,
  RosterPayload,
  SplitPayload,
  Tracklet,
} from "../lib/types";
import { Button, Card, Eyebrow, Mono, Spinner } from "./ui";

const ASSOC_CAP = 20; // per rejection-reason bucket, keeps the queue skimmable

interface PairCandidate {
  key: string; // unordered dedup key
  tracklet_a: number;
  tracklet_b: number;
  source: PairSource;
  reason?: string;
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable;
}

export function summarizeIdentityLabel(label: IdentityLabel): string {
  switch (label.kind) {
    case "pair": {
      const p = label.payload as PairPayload;
      return `T${p.tracklet_a} ↔ T${p.tracklet_b} — ${p.verdict} (${p.source.replace(/_/g, " ")})`;
    }
    case "merge": {
      const p = label.payload as MergePayload;
      return `merge players ${p.player_ids.map((id) => `#${id}`).join(", ")}`;
    }
    case "split": {
      const p = label.payload as SplitPayload;
      return `split #${p.player_id} → ${p.tracklet_ids_out.map((t) => `T${t}`).join(", ")}`;
    }
    case "roster": {
      const p = label.payload as RosterPayload;
      return `#${p.player_id} → "${p.roster_label}"`;
    }
    default:
      return JSON.stringify(label.payload);
  }
}

export function IdentityQATab({
  runId,
  artifacts,
  fps,
  seek,
  onSelectPairHighlight,
}: {
  runId: string;
  artifacts: RunArtifacts;
  fps: number;
  seek: (t: number) => void;
  onSelectPairHighlight: (a: number, b: number) => void;
}) {
  const pairLabels = useIdentityQA(runId, "pair");
  const allLabels = useIdentityQA(runId);
  const { create, remove } = useIdentityQAActions();

  const [manualA, setManualA] = useState("");
  const [manualB, setManualB] = useState("");
  const [manualCandidates, setManualCandidates] = useState<PairCandidate[]>([]);
  const [showAll, setShowAll] = useState(false);

  const trackletById = useMemo(
    () => new Map((artifacts.tracklets ?? []).map((t) => [t.tracklet_id, t])),
    [artifacts.tracklets],
  );

  const evidenceByTracklet = useMemo(() => {
    const m = new Map<number, { frame_idx: number; crop_artifact: string }[]>();
    for (const p of artifacts.players ?? []) {
      for (const ev of p.identity.evidence) {
        if (!ev.crop_artifact) continue;
        let list = m.get(ev.tracklet_id);
        if (!list) {
          list = [];
          m.set(ev.tracklet_id, list);
        }
        list.push({ frame_idx: ev.frame_idx, crop_artifact: ev.crop_artifact });
      }
    }
    for (const list of m.values()) list.sort((x, y) => x.frame_idx - y.frame_idx);
    return m;
  }, [artifacts.players]);

  const labeledPairKeys = useMemo(() => {
    const s = new Set<string>();
    for (const l of pairLabels.data ?? []) {
      const p = l.payload as PairPayload;
      s.add(pairKey(p.tracklet_a, p.tracklet_b));
    }
    return s;
  }, [pairLabels.data]);

  // Deterministic, deduped candidate derivation: manual entries, then
  // association near-misses, then eval-switch-derived pairs. Already-labeled
  // pairs (unordered equality) and duplicates within this pass are excluded.
  const candidates = useMemo(() => {
    const out: PairCandidate[] = [];
    const seen = new Set<string>();
    const add = (a: number, b: number, source: PairSource, reason?: string) => {
      if (a === b) return;
      const key = pairKey(a, b);
      if (seen.has(key) || labeledPairKeys.has(key)) return;
      seen.add(key);
      out.push({ key, tracklet_a: a, tracklet_b: b, source, reason });
    };

    for (const m of manualCandidates) add(m.tracklet_a, m.tracklet_b, "manual");

    const assoc = artifacts.association;
    if (assoc) {
      const colorTooFar = assoc.pairs
        .filter((p) => p.decision === "rejected" && p.reason === "color_too_far")
        .sort((x, y) => (x.color_distance ?? Infinity) - (y.color_distance ?? Infinity))
        .slice(0, ASSOC_CAP);
      for (const p of colorTooFar) add(p.a, p.b, "assoc_candidate", p.reason ?? undefined);

      const suspicious = assoc.pairs
        .filter(
          (p) =>
            p.decision === "rejected" &&
            (p.reason === "span_conflict" || p.reason === "speed_implausible"),
        )
        .slice(0, ASSOC_CAP);
      for (const p of suspicious) add(p.a, p.b, "assoc_candidate", p.reason ?? undefined);
    }

    const ev = artifacts.eval;
    if (ev && artifacts.players) {
      const playerById = new Map(artifacts.players.map((p) => [p.player_id, p]));
      for (const inst of ev.instances) {
        if (inst.level !== "entity") continue;
        if (inst.prev_id == null || inst.new_id == null) continue;
        if (inst.prev_id >= 100000 || inst.new_id >= 100000) continue;
        const prevEntity = playerById.get(inst.prev_id);
        const newEntity = playerById.get(inst.new_id);
        if (!prevEntity || !newEntity) continue;

        // prev tracklet: the one ending nearest before the switch frame,
        // unique — ties or "none ends before" are ambiguous, so skip.
        let prevTid: number | null = null;
        let bestEnd = -Infinity;
        let prevTie = false;
        for (const tid of prevEntity.tracklet_ids) {
          const tr = trackletById.get(tid);
          if (!tr || tr.frames.length === 0) continue;
          const end = tr.frames[tr.frames.length - 1].frame_idx;
          if (end > inst.frame_idx) continue;
          if (end > bestEnd) {
            bestEnd = end;
            prevTid = tid;
            prevTie = false;
          } else if (end === bestEnd) {
            prevTie = true;
          }
        }
        if (prevTid == null || prevTie) continue;

        // new tracklet: active at the switch frame; if none is active, the
        // one starting soonest at/after it. Multiple matches are ambiguous.
        let newTid: number | null = null;
        const activeAt: number[] = [];
        for (const tid of newEntity.tracklet_ids) {
          const tr = trackletById.get(tid);
          if (!tr || tr.frames.length === 0) continue;
          const start = tr.frames[0].frame_idx;
          const end = tr.frames[tr.frames.length - 1].frame_idx;
          if (start <= inst.frame_idx && inst.frame_idx <= end) activeAt.push(tid);
        }
        if (activeAt.length === 1) {
          newTid = activeAt[0];
        } else if (activeAt.length === 0) {
          let bestStart = Infinity;
          let newTie = false;
          for (const tid of newEntity.tracklet_ids) {
            const tr = trackletById.get(tid);
            if (!tr || tr.frames.length === 0) continue;
            const start = tr.frames[0].frame_idx;
            if (start < inst.frame_idx) continue;
            if (start < bestStart) {
              bestStart = start;
              newTid = tid;
              newTie = false;
            } else if (start === bestStart) {
              newTie = true;
            }
          }
          if (newTie) newTid = null;
        }
        if (newTid == null) continue;

        add(prevTid, newTid, "eval_switch");
      }
    }

    return out;
  }, [manualCandidates, artifacts.association, artifacts.eval, artifacts.players, trackletById, labeledPairKeys]);

  const handleVerdict = useCallback(
    (candidate: PairCandidate, verdict: PairVerdict) => {
      const evA = evidenceByTracklet.get(candidate.tracklet_a)?.[0];
      const evB = evidenceByTracklet.get(candidate.tracklet_b)?.[0];
      const trA = trackletById.get(candidate.tracklet_a);
      const trB = trackletById.get(candidate.tracklet_b);
      create.mutate(
        {
          run_id: runId,
          kind: "pair",
          payload: {
            tracklet_a: candidate.tracklet_a,
            tracklet_b: candidate.tracklet_b,
            verdict,
            crop_a: evA?.crop_artifact ?? null,
            crop_b: evB?.crop_artifact ?? null,
            frame_a: evA?.frame_idx ?? trA?.frames[0]?.frame_idx ?? null,
            frame_b: evB?.frame_idx ?? trB?.frames[0]?.frame_idx ?? null,
            source: candidate.source,
          },
        },
        {
          onSuccess: () => {
            if (candidate.source === "manual") {
              setManualCandidates((prev) => prev.filter((c) => c.key !== candidate.key));
            }
          },
        },
      );
    },
    [evidenceByTracklet, trackletById, runId, create],
  );

  // Keyboard shortcuts act on the first visible candidate; guarded so typing
  // in an input/textarea (e.g. the manual pair entry) never fires a verdict.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (isTypingTarget(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const first = candidates[0];
      if (!first) return;
      if (e.key === "s") handleVerdict(first, "same");
      else if (e.key === "d") handleVerdict(first, "different");
      else if (e.key === "u") handleVerdict(first, "unsure");
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [candidates, handleVerdict]);

  function queueManual() {
    const a = Number(manualA);
    const b = Number(manualB);
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return;
    const key = pairKey(a, b);
    setManualCandidates((prev) => (prev.some((c) => c.key === key) ? prev : [...prev, { key, tracklet_a: a, tracklet_b: b, source: "manual" }]));
    setManualA("");
    setManualB("");
  }

  const visibleCandidates = showAll ? candidates : candidates.slice(0, 25);
  const pairs = pairLabels.data ?? [];
  const same = pairs.filter((l) => (l.payload as PairPayload).verdict === "same").length;
  const different = pairs.filter((l) => (l.payload as PairPayload).verdict === "different").length;
  const unsure = pairs.filter((l) => (l.payload as PairPayload).verdict === "unsure").length;

  return (
    <div className="flex flex-col gap-4">
      <Card className="p-3">
        <div className="mb-2 flex items-center justify-between">
          <Eyebrow>Pair queue</Eyebrow>
          <span className="font-mono text-[11px] text-ink-500">{candidates.length} candidates</span>
        </div>

        <div className="mb-3 flex items-center gap-2">
          <input
            type="number"
            value={manualA}
            onChange={(e) => setManualA(e.target.value)}
            placeholder="tracklet a"
            className="w-24 rounded-md border border-white/10 bg-turf-850 px-2 py-1 text-[12px] outline-none focus:border-volt-400/60"
          />
          <input
            type="number"
            value={manualB}
            onChange={(e) => setManualB(e.target.value)}
            placeholder="tracklet b"
            className="w-24 rounded-md border border-white/10 bg-turf-850 px-2 py-1 text-[12px] outline-none focus:border-volt-400/60"
          />
          <Button variant="ghost" className="!py-1" onClick={queueManual}>
            Queue pair
          </Button>
          <span className="ml-auto text-[11px] text-ink-500">
            no association/eval artifact? queue pairs manually
          </span>
        </div>

        {candidates.length === 0 ? (
          <div className="py-6 text-center text-[13px] text-ink-500">
            No candidate pairs — nothing suspicious from association or eval, and nothing queued
            manually.
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {visibleCandidates.map((c, i) => (
              <PairCandidateCard
                key={c.key}
                candidate={c}
                isFirst={i === 0}
                runId={runId}
                fps={fps}
                trackletById={trackletById}
                evidenceByTracklet={evidenceByTracklet}
                seek={seek}
                onSelectPairHighlight={onSelectPairHighlight}
                onVerdict={handleVerdict}
                busy={create.isPending}
              />
            ))}
          </div>
        )}
        {!showAll && candidates.length > 25 && (
          <button
            onClick={() => setShowAll(true)}
            className="mt-2 text-[12px] text-ink-400 hover:text-ink-100"
          >
            Show {candidates.length - 25} more
          </button>
        )}
      </Card>

      <Card className="p-3">
        <div className="flex items-center justify-between">
          <Eyebrow>Labels collected (this run)</Eyebrow>
          <span className="font-mono text-xl text-volt-300">{pairs.length}</span>
        </div>
        <div className="mt-1.5 flex gap-3 font-mono text-[11px] text-ink-500">
          <span>same {same}</span>
          <span>different {different}</span>
          <span>unsure {unsure}</span>
        </div>
      </Card>

      <Card className="p-3">
        <Eyebrow>Recent labels (this run, all kinds)</Eyebrow>
        {allLabels.isLoading ? (
          <Spinner label="Loading labels" />
        ) : (
          <div className="mt-2 flex flex-col gap-1.5">
            {(allLabels.data ?? []).length === 0 && (
              <div className="py-4 text-center text-[13px] text-ink-500">
                No identity labels yet.
              </div>
            )}
            {(allLabels.data ?? []).map((label) => (
              <div
                key={label.id}
                className="flex items-center gap-2 rounded-lg border border-white/8 px-2.5 py-1.5 text-[12px]"
              >
                <span className="shrink-0 rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-400">
                  {label.kind}
                </span>
                <span className="flex-1 truncate text-ink-300">{summarizeIdentityLabel(label)}</span>
                <span className="shrink-0 font-mono text-[10px] text-ink-500">
                  {new Date(label.created_at).toLocaleTimeString()}
                </span>
                <button
                  onClick={() => remove.mutate(label.id)}
                  disabled={remove.isPending}
                  className="shrink-0 text-[11px] text-ink-500 hover:text-team-away"
                  title="delete / undo this label"
                >
                  undo
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function PairCandidateCard({
  candidate,
  isFirst,
  runId,
  fps,
  trackletById,
  evidenceByTracklet,
  seek,
  onSelectPairHighlight,
  onVerdict,
  busy,
}: {
  candidate: PairCandidate;
  isFirst: boolean;
  runId: string;
  fps: number;
  trackletById: Map<number, Tracklet>;
  evidenceByTracklet: Map<number, { frame_idx: number; crop_artifact: string }[]>;
  seek: (t: number) => void;
  onSelectPairHighlight: (a: number, b: number) => void;
  onVerdict: (candidate: PairCandidate, verdict: PairVerdict) => void;
  busy: boolean;
}) {
  const { tracklet_a: a, tracklet_b: b, source, reason } = candidate;
  const trA = trackletById.get(a);
  const trB = trackletById.get(b);

  const showBoth = () => {
    const endA = trA?.frames[trA.frames.length - 1]?.frame_idx;
    const endB = trB?.frames[trB.frames.length - 1]?.frame_idx;
    const boundary = endA != null && endB != null ? Math.min(endA, endB) : (endA ?? endB ?? null);
    if (boundary != null) seek(boundary / fps);
    onSelectPairHighlight(a, b);
  };

  return (
    <div
      className={`rounded-lg border p-2.5 ${
        isFirst ? "border-volt-400/40 bg-volt-400/[0.04]" : "border-white/8"
      }`}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <Mono className="text-ink-100">
          T{a} ↔ T{b}
        </Mono>
        <span className="rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-400">
          {source.replace(/_/g, " ")}
        </span>
        {reason && (
          <span className="font-mono text-[10px] text-ink-500">{reason.replace(/_/g, " ")}</span>
        )}
        <button onClick={showBoth} className="ml-auto text-[11px] text-ink-400 hover:text-ink-100">
          show both
        </button>
      </div>
      <div className="mb-2 flex gap-3">
        <TrackletStrip runId={runId} tid={a} tr={trA} evidence={evidenceByTracklet.get(a)} fps={fps} seek={seek} />
        <TrackletStrip runId={runId} tid={b} tr={trB} evidence={evidenceByTracklet.get(b)} fps={fps} seek={seek} />
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          className="!py-1"
          disabled={busy}
          title={isFirst ? "Same (shortcut: s, acts on this card)" : "Same"}
          onClick={() => onVerdict(candidate, "same")}
        >
          Same
        </Button>
        <Button
          variant="ghost"
          className="!py-1"
          disabled={busy}
          title={isFirst ? "Different (shortcut: d, acts on this card)" : "Different"}
          onClick={() => onVerdict(candidate, "different")}
        >
          Different
        </Button>
        <Button
          variant="ghost"
          className="!py-1"
          disabled={busy}
          title={isFirst ? "Unsure (shortcut: u, acts on this card)" : "Unsure"}
          onClick={() => onVerdict(candidate, "unsure")}
        >
          Unsure
        </Button>
      </div>
    </div>
  );
}

function TrackletStrip({
  runId,
  tid,
  tr,
  evidence,
  fps,
  seek,
}: {
  runId: string;
  tid: number;
  tr: Tracklet | undefined;
  evidence: { frame_idx: number; crop_artifact: string }[] | undefined;
  fps: number;
  seek: (t: number) => void;
}) {
  return (
    <div className="min-w-0 flex-1">
      <Mono className="mb-1 block text-ink-500">T{tid}</Mono>
      {evidence && evidence.length > 0 ? (
        <div className="flex gap-1 overflow-x-auto">
          {evidence.slice(0, 4).map((ev, i) => (
            <img
              key={i}
              src={api.runFileUrl(runId, ev.crop_artifact)}
              title={`frame ${ev.frame_idx}`}
              className="h-14 w-14 shrink-0 rounded-md border border-white/10 object-cover"
            />
          ))}
        </div>
      ) : (
        <button
          onClick={() => tr?.frames[0] && seek(tr.frames[0].frame_idx / fps)}
          disabled={!tr?.frames.length}
          className="rounded-md border border-white/10 px-2 py-1 text-[11px] text-ink-400 hover:bg-turf-800 disabled:opacity-40"
        >
          seek to start
        </button>
      )}
    </div>
  );
}
