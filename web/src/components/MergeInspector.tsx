// Merge inspector: the re-ID engine's working for one candidate pair, shown.
// Both tracklets' view-prototype crop strips (client-side crops from the
// original video), the winning prototype pair's per-part cosine evidence,
// each side's ranked candidates with the margin the decision rule saw, and
// the final decision. Rejected pairs render identically to merged ones —
// "why not" is the same view as "why".

import { useEffect, useMemo } from "react";
import { trackletColor } from "../lib/colors";
import { useFrameCrops } from "../lib/useFrameCrops";
import type { CropRequest } from "../lib/useFrameCrops";
import type { RunArtifacts } from "../lib/artifacts";
import type {
  AssociationPair,
  CandidateRank,
  PairDetail,
  TrackletDetail,
} from "../lib/types";
import { Mono } from "./ui";

export interface MergeInspectorProps {
  videoUrl: string | null;
  fps: number;
  pair: AssociationPair;
  artifacts: RunArtifacts;
  onClose: () => void;
  onSelectTracklet: (trackletId: number, seekTo: number | null) => void;
}

export function MergeInspector({
  videoUrl,
  fps,
  pair,
  artifacts,
  onClose,
  onSelectTracklet,
}: MergeInspectorProps) {
  const detail = artifacts.reidDetail;
  const key = pair.a < pair.b ? [pair.a, pair.b] : [pair.b, pair.a];
  const pairDetail: PairDetail | null =
    detail?.pairs.find((p) => p.a === key[0] && p.b === key[1]) ?? null;
  const sideA: TrackletDetail | null =
    detail?.tracklets.find((t) => t.tracklet_id === pair.a) ?? null;
  const sideB: TrackletDetail | null =
    detail?.tracklets.find((t) => t.tracklet_id === pair.b) ?? null;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // One crop per prototype exemplar, both sides.
  const trackletById = useMemo(
    () => new Map((artifacts.tracklets ?? []).map((t) => [t.tracklet_id, t])),
    [artifacts.tracklets],
  );
  const requests = useMemo(() => {
    const reqs: CropRequest[] = [];
    for (const side of [sideA, sideB]) {
      if (!side) continue;
      const tr = trackletById.get(side.tracklet_id);
      if (!tr) continue;
      side.prototypes.forEach((proto, k) => {
        const frame = tr.frames.find((f) => f.frame_idx === proto.exemplar_frame_idx);
        if (frame)
          reqs.push({
            key: `t${side.tracklet_id}-p${k}-f${proto.exemplar_frame_idx}`,
            frame_idx: proto.exemplar_frame_idx,
            box: frame.box,
          });
      });
    }
    return reqs;
  }, [sideA, sideB, trackletById]);
  const crops = useFrameCrops(videoUrl, fps, requests);

  const params = detail?.params ?? {};
  const rule = String(params["merge_decision_rule"] ?? "threshold");
  const floor = Number(params["min_similarity"] ?? NaN);
  const minMargin = Number(params["merge_min_margin"] ?? 0);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="flex max-h-[92vh] w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-white/10 bg-turf-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-white/8 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <Mono className="text-[13px] text-ink-100">
              T{pair.a} ↔ T{pair.b}
            </Mono>
            {pair.decision === "merged" ? (
              <span className="rounded-full bg-volt-400/15 px-2 py-0.5 font-mono text-[10px] text-volt-300">
                merged
              </span>
            ) : (
              <span className="rounded-full bg-team-away/15 px-2 py-0.5 font-mono text-[10px] text-team-away">
                {pair.reason ? pair.reason.replace(/_/g, " ") : "rejected"}
              </span>
            )}
            {(() => {
              const gtIds = artifacts.eval?.association?.gt_id_of_tracklet;
              const ga = gtIds?.[String(pair.a)];
              const gb = gtIds?.[String(pair.b)];
              if (ga == null || gb == null) return null;
              const same = ga === gb;
              return (
                <span
                  title={`GT identities: T${pair.a} → track ${ga}, T${pair.b} → track ${gb}`}
                  className={`rounded-full px-2 py-0.5 font-mono text-[10px] ${
                    same ? "bg-volt-400/15 text-volt-300" : "bg-team-away/15 text-team-away"
                  }`}
                >
                  {same ? "GT: same player" : "GT: different players"}
                </span>
              );
            })()}
            {pair.affinity != null && (
              <span className="font-mono text-[11px] text-ink-400">
                affinity {pair.affinity.toFixed(3)}
              </span>
            )}
            {pair.gap_s != null && (
              <span className="font-mono text-[11px] text-ink-500">
                gap {pair.gap_s.toFixed(2)}s
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-md px-2 py-1 text-lg leading-none text-ink-400 transition-colors hover:bg-turf-800 hover:text-ink-100"
          >
            ×
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {!detail ? (
            <div className="rounded-lg border border-dashed border-white/10 p-4 text-[13px] text-ink-500">
              No reid_detail artifact on this run — re-run the pipeline (reid-engine
              associate stage) to capture prototype and scoring detail.
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <div className="grid gap-4 md:grid-cols-2">
                {[
                  { side: sideA, tid: pair.a, winning: pairDetail?.a_proto ?? null },
                  { side: sideB, tid: pair.b, winning: pairDetail?.b_proto ?? null },
                ].map(({ side, tid, winning }) => (
                  <PrototypeStrip
                    key={tid}
                    tid={tid}
                    side={side}
                    winning={winning}
                    crops={crops}
                    fps={fps}
                    partnerTid={tid === pair.a ? pair.b : pair.a}
                    onSelectTracklet={onSelectTracklet}
                    minMargin={rule === "mutual-best" ? minMargin : null}
                  />
                ))}
              </div>

              <PartEvidence detail={pairDetail} nParts={detail.n_parts} />

              <div className="flex flex-wrap gap-x-4 gap-y-1 rounded-lg bg-turf-800/60 px-3 py-2 font-mono text-[11px] text-ink-500">
                <span>
                  rule <span className="text-ink-300">{rule}</span>
                </span>
                {Number.isFinite(floor) && (
                  <span>
                    floor <span className="text-ink-300">{floor}</span>
                    {pair.affinity != null && Number.isFinite(floor) && (
                      <span className={pair.affinity >= floor ? " text-volt-300" : " text-team-away"}>
                        {" "}
                        {pair.affinity >= floor ? "cleared" : "below"}
                      </span>
                    )}
                  </span>
                )}
                {rule === "mutual-best" && (
                  <span>
                    min margin <span className="text-ink-300">{minMargin}</span>
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------- one side's prototype crops + candidate ranking ---------- */

function PrototypeStrip({
  tid,
  side,
  winning,
  crops,
  fps,
  partnerTid,
  onSelectTracklet,
  minMargin,
}: {
  tid: number;
  side: TrackletDetail | null;
  winning: number | null;
  crops: Record<string, string>;
  fps: number;
  partnerTid: number;
  onSelectTracklet: (trackletId: number, seekTo: number | null) => void;
  minMargin: number | null; // null = margin not part of the active rule
}) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-white/8 p-3">
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ background: trackletColor(tid) }}
        />
        <Mono className="text-[12px] text-ink-100">T{tid}</Mono>
        {side?.starved && (
          <span
            className="rounded-full bg-turf-800 px-2 py-0.5 font-mono text-[10px] text-ink-500"
            title="too few frames for a confident representation"
          >
            starved
          </span>
        )}
        <button
          onClick={() =>
            onSelectTracklet(
              tid,
              side?.prototypes[0] ? side.prototypes[0].exemplar_frame_idx / fps : null,
            )
          }
          className="ml-auto text-[11px] text-ink-400 hover:text-ink-100"
        >
          view in video ›
        </button>
      </div>

      {!side ? (
        <div className="py-3 text-[12px] text-ink-500">
          No representation (no usable features on this tracklet).
        </div>
      ) : (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {side.prototypes.map((proto, k) => {
            const crop = crops[`t${tid}-p${k}-f${proto.exemplar_frame_idx}`];
            const isWin = winning === k;
            return (
              <div
                key={k}
                className={`flex w-24 shrink-0 flex-col gap-1 rounded-md border p-1 ${
                  isWin ? "border-volt-400/70" : "border-white/8"
                }`}
                title={`prototype ${k + 1}: ${proto.member_frame_idxs.length} frame(s), exemplar frame ${proto.exemplar_frame_idx}`}
              >
                {crop ? (
                  <img src={crop} alt="" className="h-28 w-full rounded-sm object-cover" />
                ) : (
                  <div className="flex h-28 w-full items-center justify-center rounded-sm bg-turf-800 text-[10px] text-ink-500">
                    cropping…
                  </div>
                )}
                <div className="flex items-center justify-between font-mono text-[9px] text-ink-500">
                  <span>{isWin ? "▶ view " : "view "}{k + 1}</span>
                  <span>{proto.member_frame_idxs.length}f</span>
                </div>
                <VisibilityBar vis={proto.part_visibility} />
              </div>
            );
          })}
        </div>
      )}

      {side && <CandidateRanking side={side} partnerTid={partnerTid} minMargin={minMargin} />}
    </div>
  );
}

function VisibilityBar({ vis }: { vis: number[] }) {
  return (
    <div className="flex h-1.5 gap-px" title="per-part visibility">
      {vis.map((v, i) => (
        <div
          key={i}
          className="flex-1 rounded-[1px] bg-volt-400"
          style={{ opacity: Math.max(0.12, v) }}
        />
      ))}
    </div>
  );
}

function CandidateRanking({
  side,
  partnerTid,
  minMargin,
}: {
  side: TrackletDetail;
  partnerTid: number;
  minMargin: number | null;
}) {
  const top = side.candidates.slice(0, 3);
  const margin =
    side.candidates.length >= 2 ? side.candidates[0].affinity - side.candidates[1].affinity : null;
  const partnerIsBest = top[0]?.tracklet_id === partnerTid;
  return (
    <div className="flex flex-col gap-0.5">
      <div className="font-mono text-[9px] uppercase tracking-[0.18em] text-ink-500">
        ranked candidates
      </div>
      {top.length === 0 && <div className="text-[11px] text-ink-500">none gate-passing</div>}
      {top.map((c: CandidateRank, i: number) => (
        <div
          key={c.tracklet_id}
          className={`flex items-center gap-2 font-mono text-[11px] ${
            c.tracklet_id === partnerTid ? "text-ink-100" : "text-ink-500"
          }`}
        >
          <span className="w-4">{i + 1}.</span>
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: trackletColor(c.tracklet_id) }}
          />
          <span>T{c.tracklet_id}</span>
          {c.tracklet_id === partnerTid && <span className="text-ink-400">(this pair)</span>}
          <span className="ml-auto">{c.affinity.toFixed(3)}</span>
        </div>
      ))}
      {minMargin != null && margin != null && (
        <div className="mt-0.5 font-mono text-[11px]">
          <span className="text-ink-500">margin </span>
          <span className={margin >= minMargin ? "text-volt-300" : "text-team-away"}>
            {margin.toFixed(3)} {margin >= minMargin ? "≥" : "<"} {minMargin}
          </span>
          {!partnerIsBest && <span className="text-team-away"> · partner not ranked first</span>}
        </div>
      )}
      {minMargin != null && margin == null && top.length > 0 && (
        <div className="mt-0.5 font-mono text-[11px] text-ink-500">
          single candidate — margin unbounded
        </div>
      )}
    </div>
  );
}

/* ---------- winning-pair per-part cosine evidence ---------- */

function PartEvidence({ detail, nParts }: { detail: PairDetail | null; nParts: number }) {
  if (!detail)
    return (
      <div className="rounded-lg border border-dashed border-white/10 p-3 text-[12px] text-ink-500">
        No similarity detail for this pair — it was gate-vetoed before appearance was
        scored, or one side has no usable features.
      </div>
    );
  const parts = detail.part_cosines.length || nParts;
  return (
    <div className="rounded-lg border border-white/8 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-500">
          winning pair evidence · view {detail.a_proto + 1} ↔ view {detail.b_proto + 1}
        </div>
        <Mono className="text-[12px] text-ink-100">{detail.affinity.toFixed(3)}</Mono>
      </div>
      <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${parts}, 1fr)` }}>
        {detail.part_cosines.map((cos, i) => {
          const w = detail.part_weights[i];
          return (
            <div key={i} className="flex flex-col items-center gap-1">
              <div className="h-16 w-full overflow-hidden rounded-sm bg-turf-800">
                {cos != null && (
                  <div
                    className="w-full rounded-sm bg-volt-400"
                    style={{
                      height: `${Math.max(0, Math.min(1, cos)) * 100}%`,
                      marginTop: `${(1 - Math.max(0, Math.min(1, cos))) * 100}%`,
                      opacity: w != null ? Math.max(0.3, w) : 1,
                    }}
                  />
                )}
              </div>
              <span className="font-mono text-[9px] text-ink-500">P{i + 1}</span>
              <span className="font-mono text-[10px] text-ink-400">
                {cos != null ? cos.toFixed(2) : "—"}
              </span>
            </div>
          );
        })}
      </div>
      <div className="mt-1.5 text-[11px] text-ink-500">
        Per-part cosine of the winning prototype pair; bar opacity is the part weight
        (min visibility across both sides). "—" parts were invisible on at least one
        side and excluded from the score.
      </div>
    </div>
  );
}
