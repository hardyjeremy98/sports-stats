// Mirrors packages/matchlab_core/src/matchlab_core/schemas/ + server API models.

export type Team = "home" | "away" | "referee" | "unknown";
export type DetectionClass = "ball" | "goalkeeper" | "player" | "referee";
export type RunStatus = "queued" | "running" | "completed" | "failed";

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  box: Box;
  confidence: number;
  cls: DetectionClass;
}

export interface FrameDetections {
  frame_idx: number;
  t: number;
  detections: Detection[];
}

export interface BallObservation {
  frame_idx: number;
  t: number;
  xy: { x: number; y: number };
  confidence: number;
  interpolated: boolean;
}

export interface TrackletFrame {
  frame_idx: number;
  box: Box;
  confidence: number;
  // Box provenance (SPO-15). Optional because artifacts written before this
  // field existed have no `source` key at all; the pydantic side defaults
  // missing/older data to "observed" but that default isn't visible here, so
  // treat `undefined` as "observed" if you need a concrete value.
  source?: "observed" | "predicted" | "interpolated";
}

export interface Tracklet {
  tracklet_id: number;
  cls: DetectionClass;
  frames: TrackletFrame[];
}

export interface TeamAssignment {
  tracklet_id: number;
  team: Team;
  confidence: number;
  kit_color: [number, number, number] | null;
}

export interface FrameCalibration {
  frame_idx: number;
  t: number;
  homography: number[][] | null;
  n_keypoints: number;
  keypoints_image: { x: number; y: number }[];
  keypoint_confidences: number[];
  confidence: number;
  smoothed: boolean;
}

export interface IdentityEvidence {
  tracklet_id: number;
  frame_idx: number;
  score: number;
  crop_artifact: string | null;
  upscaled: boolean;
  box?: Box | null;
  raw_crop_artifact?: string | null;
}

export interface PlayerIdentity {
  kind: "face" | "jersey" | "manual" | "none";
  label: string | null;
  confidence: number;
  evidence: IdentityEvidence[];
}

export interface PlayerEntity {
  player_id: number;
  tracklet_ids: number[];
  team: Team;
  identity: PlayerIdentity;
  association_confidence: number;
}

export interface MinimapPlayer {
  player_id: number;
  // Pitch coordinates in centimeters; extent depends on the run's pitch spec
  // (manifest.config.pitch) -- 12000x7000 for roboflow, 10500x6800 for fifa.
  // See web/src/components/PitchCanvas.tsx pitchDims() / matchlab_core.pitch.
  x: number;
  y: number;
  team: Team;
  confidence: number;
}

export interface MinimapFrame {
  frame_idx: number;
  t: number;
  players: MinimapPlayer[];
  ball: { x: number; y: number; confidence: number; interpolated: boolean } | null;
  calibration_confidence: number;
}

export type EventType =
  | "touch"
  | "pass"
  | "missed_pass"
  | "restart"
  | "possession_gain"
  | "shot"
  | "tackle"
  | "interception";

export interface MatchEvent {
  event_id: number;
  type: EventType;
  frame_idx: number;
  t: number;
  player_id: number | null;
  team: Team;
  confidence: number;
  contested: boolean;
  attrs: Record<string, unknown>;
}

// One event from the spotting exchange contract's output array
// (docs/reference/spotting-exchange-contract.md), written by the `tdeed`
// spotting stage to spotting.json. `class` is the spotter's native taxonomy
// (e.g. T-DEED's "PASS"/"DRIVE"/"SHOT"), verbatim and unmapped to
// EventType — the mapping, if consumed, lives downstream, not here.
export interface SpottedEvent {
  class: string;
  frame_idx: number;
  t: number;
  confidence: number;
  half: number | null;
}

export interface PossessionSegment {
  player_id: number;
  team: Team;
  start_frame: number;
  end_frame: number;
  start_t: number;
  end_t: number;
  confidence: number;
}

export interface StatLine {
  player_id: number;
  label: string;
  team: Team;
  passes: number;
  missed_passes: number;
  touches: number;
  restarts: number;
  possession_seconds: number;
}

export interface StatSheet {
  players: StatLine[];
}

export interface TimelineBucket {
  t: number;
  detection_confidence: number;
  tracking_stability: number;
  calibration_confidence: number;
  identity_coverage: number;
  event_count: number;
  contested_event_count: number;
  flags: string[];
}

export interface StageResult {
  kind: string;
  impl: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  started_at: string | null;
  finished_at: string | null;
  duration_s: number | null;
  error: string | null;
  metrics: Record<string, number | string>;
}

// ---- Provenance (SPO-10): what produced a run, for months-later comparison ----
// Every declared field is always present -- unknown values are the literal
// string "unknown" (or null where the schema allows it), never an absent key.

export interface LicenseAxes {
  code: string;
  weights: string;
  training_data: string;
}

export interface ModelProvenance {
  architecture: string;
  revision: string;
  weights_path: string | null;
  weights_sha256: string | null;
  lineage: string;
  training_commit: string | null;
  training_config: string | null;
  training_seed: number | null;
  dataset_split_manifest: string | null;
  dataset_split_manifest_sha256: string | null;
  detections_cache_hash: string | null;
  license: LicenseAxes;
}

export interface StageProvenance {
  impl: string;
  params: Record<string, unknown>;
  models: ModelProvenance[];
}

export interface RunProvenance {
  git_revision: string;
  package_versions: Record<string, string>;
  stages: Record<string, StageProvenance>; // keyed by StageKind value
  evaluation_set_hash: string;
  evaluation_set_source: string | null;
}

export interface RunManifest {
  run_id: string;
  created_at: string;
  video: {
    path: string;
    fps: number;
    frame_count: number;
    width: number;
    height: number;
    duration_s: number;
    sample_stride: number;
  };
  config: Record<string, unknown>;
  config_name: string;
  stages: StageResult[];
  artifacts: Record<string, string>;
  metrics: Record<string, number | string>;
  status: string;
  error: string | null;
  // Optional because the server serves old manifests as raw JSON without
  // pydantic revalidation (matchlab_server/api/runs.py's `_detail`, a plain
  // `json.loads`), so pre-provenance-feature manifests have no `provenance`
  // key at all -- see the SPO-15 optional-`source` precedent on
  // `TrackletFrame` above.
  provenance?: RunProvenance;
}

// ---- Ground truth + evaluation (labelled benchmark clips) ----

export interface GroundTruthTrack {
  track_id: number;
  role: "player" | "goalkeeper" | "referee" | "ball" | "other";
  team: "left" | "right" | null;
  jersey: string | null;
  frames: { frame_idx: number; box: Box }[];
}

export interface GroundTruth {
  source: string;
  sequence: string | null;
  fps: number;
  width: number;
  height: number;
  seq_length: number;
  tracks: GroundTruthTrack[];
}

export interface EvalLevelMetrics {
  idf1: number;
  idp: number;
  idr: number;
  mota: number;
  num_switches: number;
  num_fragmentations: number;
  num_false_positives: number;
  num_misses: number;
  num_objects: number;
  num_unique_objects: number;
  mostly_tracked: number;
  mostly_lost: number;
  precision: number;
  recall: number;
}

// HOTA family (SPO-7), computed per evaluation level via vendored TrackEval.
// Alpha-averaged over TrackEval's own IoU grid, so unlike the motmetrics-backed
// `EvalLevelMetrics` it does NOT honour EvalResult.iou_threshold.
export interface EvalHotaMetrics {
  hota: number;
  deta: number;
  assa: number;
  loca: number;
}

// min/p25/median/p75/max/mean, numpy linear-interpolation percentiles.
export interface EvalDistributionSummary {
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
  mean: number;
}

// SPO-6 per-tracklet GT composition. `purity`/`majority_gt_track_id` are null
// when the tracklet never matched a GT box above threshold -- no verdict, not a
// bad one.
export interface EvalPurityTracklet {
  tracklet_id: number;
  length: number;
  matched_frames: number;
  unmatched_frames: number;
  gt_composition: Record<string, number>;
  majority_gt_track_id: number | null;
  purity: number | null;
  mixed_frames: number;
  mixed_seconds: number;
}

// Frame-weighted, so one long tracklet outweighs many short ones. `mean_purity`
// and `frac_impure` are null when no tracklet matched GT; `track_length` and
// `tracklets_per_gt_player.summary` are null when there is nothing to
// summarise. Render these as "—", never 0 — an abstention is not a zero score.
export interface EvalPurityAggregate {
  n_tracklets: number;
  n_tracklets_matched: number;
  mean_purity: number | null;
  frac_impure: number | null;
  total_mixed_seconds: number;
  tracklets_per_gt_player: {
    counts: Record<string, number>;
    summary: { mean: number; median: number; max: number } | null;
  };
  track_length: EvalDistributionSummary | null;
}

// `min_track_length` null means the threshold was not discoverable from the
// manifest, in which case `post_filter` is identical to `pre_filter` rather
// than assuming an unfiltered 0 -- see `note`, which explains that both views
// only re-apply the filter to what reached the evaluator and cannot recover
// tracklets the track stage already dropped upstream.
export interface EvalPurityLevel {
  min_track_length: number | null;
  note: string;
  tracklets: EvalPurityTracklet[];
  pre_filter: EvalPurityAggregate;
  post_filter: EvalPurityAggregate;
}

// SPO-19: evidence-based layer attribution per ID switch. "refinement" is
// reserved (Phase 4 refined-tracklet layer) and never emitted today;
// "ambiguous" is a first-class honest outcome, not a fallback guess.
export type AttributionLayer =
  | "detection"
  | "online_association"
  | "refinement"
  | "offline_association"
  | "ambiguous";

export interface AttributionEvidence {
  kind: string; // oracle_input | oracle_comparison | tracklet_counterpart | entity_only | insufficient_evidence
  detail?: string;
  outcome?: "persists" | "disappears";
  oracle_run?: string | null;
  oracle_frame_idx?: number;
  oracle_t?: number;
  frame_idx?: number;
  t?: number;
  tol_s?: number;
}

export interface EvalInstanceAttribution {
  layer: AttributionLayer;
  evidence: AttributionEvidence[];
}

export interface EvalInstance {
  level: "tracklet" | "entity";
  kind: "id_switch";
  frame_idx: number;
  t: number;
  gt_track_id: number;
  gt_label: string;
  prev_id: number | null;
  new_id: number | null;
  // Absent on eval.json files written before SPO-19 — re-evaluate to attribute.
  attribution?: EvalInstanceAttribution;
}

// Third eval layer (ADR 004): does `identity.label` correspond to the right
// person, judged against GT tracks. null = identity stage not run; non-null
// with coverage 0 + cluster_purity null = it ran but abstained everywhere.
// Naming-vs-GT sub-block (SPO-52): identity labels judged against the
// argmax-overlap GT track's jersey identity. null (with naming_note) when the
// GT carries no identified jerseys; roster_precision null when nothing named.
export interface EvalNaming {
  n_entities_matched: number;
  n_named: number;
  n_judged: number;
  n_correct: number;
  coverage: number;
  abstention_rate: number;
  roster_precision: number | null;
  precision_at_abstention: { precision: number | null; abstention: number };
}

export interface EvalIdentity {
  n_entities_matched: number;
  n_labeled: number;
  coverage: number;
  abstention_rate: number;
  n_clusters: number;
  cluster_purity: number | null;
  cluster_completeness: number | null;
  // Absent on eval.json files written before SPO-52 — re-evaluate to backfill.
  naming?: EvalNaming | null;
  naming_note?: string | null;
}

// Detection-quality layer (SPO-9): level-independent (computed once, never
// per tracklet/entity/identity level). null when detections.jsonl is absent
// (imported external runs -- see exchange.py::import_mot_tracklets).
export interface DetectionHeightBin {
  bin: string;
  edges: [number | null, number | null];
  n_gt: number;
  n_det: number;
  precision: number | null;
  recall: number | null;
  ap: number | null;
}

export interface DetectionBurstSummary {
  min: number | null;
  median: number | null;
  p95: number | null;
  max: number | null;
  mean: number | null;
}

export interface DetectionMissBursts {
  stride: number;
  fps: number;
  per_track: Record<
    string,
    {
      n_bursts: number;
      max_burst: number | null;
      burst_lengths_summary: DetectionBurstSummary | null;
    }
  >;
  overall: DetectionBurstSummary & { burst_seconds_p95: number | null };
  n_tracks_with_bursts: number;
}

export interface DetectionEval {
  iou_threshold: number;
  height_bin_edges: number[];
  stride: number;
  fps: number;
  n_frames_evaluated: number;
  n_detections: number;
  n_gt_boxes: number;
  precision: number | null;
  recall: number | null;
  ap: number | null;
  by_height_bin: DetectionHeightBin[];
  miss_bursts: DetectionMissBursts;
  duplicates: { n_duplicates: number; duplicate_rate: number | null };
  jitter: {
    center_jitter_mean: number | null;
    center_jitter_p95: number | null;
    height_jitter_mean: number | null;
    height_jitter_p95: number | null;
    n_pairs: number;
  };
}

export interface PersistentSwitchTransition {
  level: "tracklet" | "entity";
  gt_track_id: number;
  gt_label: string;
  prev_id: number;
  new_id: number;
  t_from: number;
  t_to: number;
  prev_run_s: number;
  new_run_s: number;
  verdict: "genuine" | "frame_exit";
  absence: { t_from: number; t_to: number } | null;
}

export interface PersistentSwitchLevel {
  "t_0.5s": number;
  t_1s: number;
  "t_2s": number;
  frame_exit?: { "t_0.5s": number; t_1s: number; "t_2s": number };
  transitions?: PersistentSwitchTransition[];
}

export interface EvalResult {
  source: string;
  sequence: string | null;
  iou_threshold: number;
  sample_stride: number;
  n_frames_evaluated: number;
  n_gt_tracks: number;
  n_gt_tracks_excluded: number;
  levels: { tracklet: EvalLevelMetrics; entity: EvalLevelMetrics };
  // Optional for the same reason as `attribution` below: eval.json files
  // written before SPO-7 / SPO-6 have no such key, and the server serves old
  // artifacts as raw JSON without pydantic revalidation. Re-evaluate to
  // backfill.
  hota?: { tracklet: EvalHotaMetrics; entity: EvalHotaMetrics };
  purity?: { tracklet: EvalPurityLevel; entity: EvalPurityLevel };
  // Flicker-insensitive ID switches (keys "t_0.5s" | "t_1s" | "t_2s");
  // threshold_headline_s names the threshold runs.metrics headlines use.
  // frame_exit tallies switches exempted because the player verifiably left
  // the frame (border-touching GT boxes both sides of a GT-empty gap) --
  // excluded from the t_* counts but never silently dropped.
  persistent_switches?: {
    threshold_headline_s: number;
    tracklet: PersistentSwitchLevel;
    entity: PersistentSwitchLevel;
  };
  association: {
    idf1_gain: number;
    idsw_delta: number;
    n_entities_merged: number;
    merge_precision: number | null;
    n_pairs: number;
    n_pairs_correct: number;
    n_pairs_unmatched: number;
    merged_pairs: {
      a: number;
      b: number;
      player_id: number;
      gt_a: number | null;
      gt_b: number | null;
      correct: boolean;
    }[];
  };
  instances: EvalInstance[];
  // SPO-19 context block: how attribution was derived for this payload.
  attribution?: {
    detect_impl: string | null;
    oracle_input: boolean;
    oracle_comparison: { oracle_run: string | null } | null;
    tol_s: number;
    counts: Record<"tracklet" | "entity", Partial<Record<AttributionLayer, number>>>;
  };
  identity?: EvalIdentity | null;
  detection?: DetectionEval | null;
}

// ---- Benchmark matrix (config x GT-video, aggregating repeat runs) ----

export interface BenchmarkVideo {
  video_id: number;
  filename: string;
  sequence: string | null;
}

export interface BenchmarkCell {
  n_runs: number;
  run_ids: string[]; // newest first
  metrics_mean: Record<string, number>;
  metrics_range: Record<string, [number, number]>;
}

export interface BenchmarkGroup {
  config_name: string;
  config_hash: string;
  n_runs: number;
  cells: Record<string, BenchmarkCell>; // keyed by video_id (string)
}

export interface Benchmark {
  videos: BenchmarkVideo[];
  groups: BenchmarkGroup[]; // sorted by config_name then config_hash
}

// ---- Cross-tracklet association decision trail (association.json) ----

export type AssociationRejectReason =
  | "no_features"
  | "temporal_overlap"
  | "gap_too_long"
  | "speed_implausible"
  | "color_too_far"
  | "embed_too_far"
  | "span_conflict"
  | "team_mismatch"
  | "motion_infeasible"
  | "anchor_conflict";

export interface AssociationPair {
  a: number; // tracklet ids, a = earlier-starting
  b: number;
  gap_s: number | null;
  dist_px: number | null;
  color_distance: number | null;
  embed_distance: number | null;
  affinity: number | null;
  decision: "merged" | "rejected";
  reason: AssociationRejectReason | null;
}

export interface AssociationEntitySummary {
  player_id: number;
  tracklet_ids: number[];
  merge_edges: [number, number][]; // accepted union-find edges, in union order
}

export interface AssociationReport {
  impl: string;
  params: Record<string, unknown>;
  pairs: AssociationPair[];
  entities: AssociationEntitySummary[];
}

// ---- Closed-roster naming decisions (naming.json, reid-engine) ----

export type NamingDecision = "named" | "abstain";
export type ConfidenceTier = "auto_accept" | "adjudicate" | "qa";

export interface AnchorRecord {
  tracklet_id: number;
  candidate: string;
  log_lr: number;
  source: string; // anchor-source name, e.g. "oracle-jersey"
}

export interface ThreadNaming {
  thread_id: number; // == PlayerEntity.player_id
  tracklet_ids: number[];
  label: string | null; // roster candidate when decision == "named"
  posterior: Record<string, number>; // roster candidate -> probability
  margin: number | null; // top1 - top2 posterior
  decision: NamingDecision;
  tier: ConfidenceTier | null; // null until the tier slice routes it
  anchors_consumed: AnchorRecord[];
}

export interface NamingReport {
  impl: string;
  params: Record<string, unknown>;
  roster: string[];
  threads: ThreadNaming[];
  calibration: Record<string, unknown>;
}

// ---- Identity QA labels (human annotations; never mutate run artifacts/entities) ----

export type IdentityLabelKind = "pair" | "merge" | "split" | "roster";
export type PairVerdict = "same" | "different" | "unsure";
export type PairSource = "manual" | "assoc_candidate" | "eval_switch";

export interface PairPayload {
  tracklet_a: number;
  tracklet_b: number;
  verdict: PairVerdict;
  crop_a: string | null;
  crop_b: string | null;
  frame_a: number | null;
  frame_b: number | null;
  source: PairSource;
}

export interface MergePayload {
  player_ids: number[];
}

export interface SplitPayload {
  player_id: number;
  tracklet_ids_out: number[];
}

export interface RosterPayload {
  player_id: number;
  roster_label: string;
}

export type IdentityLabelPayload = PairPayload | MergePayload | SplitPayload | RosterPayload;

export interface IdentityLabel {
  id: number;
  run_id: string;
  video_id: number;
  kind: IdentityLabelKind;
  payload: IdentityLabelPayload;
  note: string | null;
  created_at: string;
}

// ---- API models ----

export interface Video {
  id: number;
  filename: string;
  size_bytes: number;
  duration_s: number;
  fps: number;
  width: number;
  height: number;
  has_ground_truth: boolean;
  created_at: string;
}

export interface PipelineConfigOut {
  name: string;
  description: string;
  sport: string;
  yaml: string;
  stages: Record<string, { impl: string; params: Record<string, unknown>; enabled: boolean }>;
}

export interface Registry {
  stages: Record<string, string[]>;
}

export interface Run {
  id: string;
  video_id: number;
  config_name: string;
  label: string | null;
  status: RunStatus;
  error: string | null;
  progress_stage: string | null;
  progress_frac: number;
  progress_msg: string | null;
  metrics: Record<string, number | string> | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunDetail extends Run {
  config_yaml: string;
  manifest: RunManifest | null;
  video: Video | null;
}

export interface ConfigChange {
  path: string;
  a: unknown;
  b: unknown;
}

export interface SwitchDiff {
  fixed: EvalInstance[];
  introduced: EvalInstance[];
  persisted: { a: EvalInstance; b: EvalInstance }[];
  counts: { fixed: number; introduced: number; persisted: number };
}

export interface RunDiff {
  run_a: RunDetail;
  run_b: RunDetail;
  config_changes: ConfigChange[];
  metric_deltas: Record<string, { a: unknown; b: unknown; delta: number | null }>;
  stats_a: StatSheet | null;
  stats_b: StatSheet | null;
  timeline_a: TimelineBucket[] | null;
  timeline_b: TimelineBucket[] | null;
  eval_a: EvalResult | null;
  eval_b: EvalResult | null;
  switch_diff: SwitchDiff | null;
}

export type QAStatus = "pending" | "accepted" | "corrected" | "rejected";

export interface QARecord {
  id: number;
  run_id: string;
  qa_id: number;
  payload: {
    qa_id: number;
    event: MatchEvent;
    reason: string;
    status: string;
    candidate_player_ids: number[];
    note: string | null;
  };
  status: QAStatus;
  corrected_player_id: number | null;
  corrected_event_type: string | null;
  note: string | null;
  decided_at: string | null;
}
