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

export interface EvalResult {
  source: string;
  sequence: string | null;
  iou_threshold: number;
  sample_stride: number;
  n_frames_evaluated: number;
  n_gt_tracks: number;
  n_gt_tracks_excluded: number;
  levels: { tracklet: EvalLevelMetrics; entity: EvalLevelMetrics };
  association: { idf1_gain: number; idsw_delta: number };
  instances: EvalInstance[];
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

export interface RunDiff {
  run_a: RunDetail;
  run_b: RunDetail;
  config_changes: ConfigChange[];
  metric_deltas: Record<string, { a: unknown; b: unknown; delta: number | null }>;
  stats_a: StatSheet | null;
  stats_b: StatSheet | null;
  timeline_a: TimelineBucket[] | null;
  timeline_b: TimelineBucket[] | null;
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
