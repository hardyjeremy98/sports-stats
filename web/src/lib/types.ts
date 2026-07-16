// Mirrors packages/pitchlab_core/src/pitchlab_core/schemas/ + server API models.

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
  x: number; // pitch cm, 0..12000
  y: number; // pitch cm, 0..7000
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
  provenance: RunProvenance;
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

export interface EvalInstance {
  level: "tracklet" | "entity";
  kind: "id_switch";
  frame_idx: number;
  t: number;
  gt_track_id: number;
  gt_label: string;
  prev_id: number | null;
  new_id: number | null;
}

// Third eval layer (ADR 004): does `identity.label` correspond to the right
// person, judged against GT tracks. null = identity stage not run; non-null
// with coverage 0 + cluster_purity null = it ran but abstained everywhere.
export interface EvalIdentity {
  n_entities_matched: number;
  n_labeled: number;
  coverage: number;
  abstention_rate: number;
  n_clusters: number;
  cluster_purity: number | null;
  cluster_completeness: number | null;
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
  association: {
    idf1_gain: number;
    idsw_delta: number;
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
  identity?: EvalIdentity | null;
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
  | "span_conflict";

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
