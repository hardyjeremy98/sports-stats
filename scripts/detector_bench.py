"""Standalone detector comparison harness (detector-selection research).

Isolates the DETECTION layer: runs a candidate detector over sampled frames of
the SoccerNet-Tracking sequences declared in `configs/datasets/soccernet.json`
and scores its boxes against that video's GroundTruth with the repo's own
`matchlab_core.detection_eval.evaluate_detections` -- no tracker, no
association, no pipeline. The detector is the only variable.

Two-stage by design, because inference is the expensive part and thresholding
is not:

  1. `infer`  -- decode frames, run the model at a very low confidence floor,
                 cache every raw box to `cache/<candidate>/<seq>.jsonl`.
  2. `score`  -- replay the cache through `evaluate_detections` at many
                 confidence floors, so a candidate is compared at its own best
                 operating point rather than at a threshold that happens to
                 suit the incumbent.

Scoring convention is copied from `evaluation.py::_load_detections`: only
person-like classes (player / goalkeeper / referee) are kept on the detection
side, and only `_SCORED_ROLES` GT tracks on the truth side. Ball is excluded
symmetrically -- it is a separate pipeline stream and a separate problem.

Frame indexing: SoccerNet `img1/%06d.jpg` is 1-based, GroundTruth.frame_idx is
0-based, so image N holds frame_idx N-1. `--verify-alignment` asserts this
rather than trusting it (an off-by-one here would silently penalise every
candidate equally and look like "detection is hard").
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Two scoring targets. `roles` selects GT tracks; `names` selects detector
# classes. Ball is excluded from the player target and vice versa, symmetrically
# on both sides -- the same convention `evaluation.py::_load_detections` uses.
TARGETS = {
    "player": {
        "roles": ("player", "goalkeeper", "referee"),
        "names": {"player", "goalkeeper", "goalkeepers", "referee",
                  "person", "players"},
    },
    "ball": {
        "roles": ("ball",),
        "names": {"ball", "sports ball", "soccer ball", "football"},
    },
}
SCORED_ROLES = TARGETS["player"]["roles"]
PERSON_NAMES = TARGETS["player"]["names"]


# --------------------------------------------------------------------------
# candidate registry
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    backend: str  # "ultralytics" | "rfdetr"
    weights: str | None = None
    imgsz: int = 1280
    # Provenance for the report. `contaminated` = the model's training or
    # validation data plausibly includes SoccerNet-Tracking sequences, which
    # is the split we evaluate on -> not comparable against clean candidates.
    trained_on: str = "unknown"
    contaminated: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)


CANDIDATES: dict[str, Candidate] = {}


def _add(c: Candidate) -> None:
    CANDIDATES[c.name] = c


# --- incumbent -------------------------------------------------------------
_add(Candidate(
    "incumbent-yolov8x-roboflow", "ultralytics",
    weights="data/weights/football-player-detection.pt", imgsz=1280,
    trained_on="roboflow/sports football-players-detection (small, ~600 imgs)",
    notes="Currently wired into every production config at imgsz=1280 conf=0.4.",
))

# --- clean tier: no SoccerNet-Tracking exposure ----------------------------
_add(Candidate(
    "yolo11m-martinjolif", "ultralytics",
    weights="data/weights/bench/martinjolif-yolo11m.pt", imgsz=1280,
    trained_on="martinjolif/football-player-detection (roboflow-style)",
))
_add(Candidate(
    "yolov26m-sportsmot", "ultralytics",
    weights="data/weights/bench/hlouy-yolov26m-sportsmot.pt", imgsz=1280,
    trained_on="SportsMOT (football subset)",
))
_add(Candidate(
    "yolo-mobadam", "ultralytics",
    weights="data/weights/bench/mobadam-player_detector.pt", imgsz=1280,
    trained_on="unspecified football dataset",
))
_add(Candidate(
    "yolo-gianpaj", "ultralytics",
    weights="data/weights/bench/gianpaj-best.pt", imgsz=1280,
    trained_on="football-players-detection (roboflow)",
))
_add(Candidate(
    "yolo11x-coco", "ultralytics", weights="yolo11x.pt", imgsz=1280,
    trained_on="COCO (generic person class)",
    notes="Generic-person control: how much does football-specific fine-tuning buy?",
))
_add(Candidate(
    "yolov8x-coco", "ultralytics", weights="yolov8x.pt", imgsz=1280,
    trained_on="COCO (generic person class)",
))
_add(Candidate(
    "rfdetr-base-coco", "rfdetr", weights=None, imgsz=728,
    trained_on="Objects365 + COCO (generic person class)",
    extra={"model_size": "base"},
))
_add(Candidate(
    # RFDETRLarge requires a resolution divisible by 32 (patch 16 x 2 windows);
    # the base model's 728 is not, hence the different size here.
    "rfdetr-large-coco", "rfdetr", weights=None, imgsz=768,
    trained_on="Objects365 + COCO (generic person class)",
    extra={"model_size": "large"},
))

# --- contaminated tier: trained/validated on SoccerNet-Tracking ------------
_add(Candidate(
    "rfdetr-large-soccernet", "rfdetr",
    weights="data/weights/bench/julianzu-rfdetr-soccernet.pth", imgsz=1280,
    trained_on="SoccerNet-Tracking 2023",
    contaminated=True,
    notes="Model card reports evaluation on SoccerNet-Tracking-2023-test, "
          "the same split as our eval sequences.",
    # The published checkpoint predates the current rfdetr release: it was
    # trained with patch_size=14 and reports 3 classes, so both must be passed
    # explicitly or loading raises rather than silently mis-shaping the head.
    extra={"model_size": "large", "num_classes": 3, "patch_size": 14,
           "class_map": {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}},
))
_add(Candidate(
    "yolov8x6-soccernet-gsr", "ultralytics",
    weights="data/weights/bench/soccermaster-yolo_v8x6_gsr.pt", imgsz=1280,
    trained_on="SoccerNet-GSR train split",
    contaminated=True,
    notes="SoccerNet GSR baseline detector; GSR clips are drawn from the same "
          "broadcast corpus as SoccerNet-Tracking.",
))

# --- dedicated ball detectors (scored only by the `ball` target) -----------
# Ball-only YOLO models. Run at high imgsz: the ball is ~10-20 px in a 1080p
# frame, so downscaling to 640 destroys it outright.
for _n, _w in [
    ("ball-martinjolif", "data/weights/bench/martinjolif-ball.pt"),
    ("ball-rajatdave", "data/weights/bench/rajatdave-ball.pt"),
    ("ball-raghav", "data/weights/bench/raghav-ball.pt"),
]:
    for _sz in (1280, 1920):
        _add(Candidate(
            f"{_n}@{_sz}", "ultralytics", weights=_w, imgsz=_sz,
            trained_on="football ball datasets (not SoccerNet-Tracking)",
            extra={"single_class": "ball", "max_det": 30},
        ))

# --- resolution variants ---------------------------------------------------
# Players in the far half of a 1920x1080 broadcast frame are ~25-40 px tall, so
# input size is a first-order variable, not a tuning detail. Each variant is
# the same weights as its base candidate at a different imgsz.
for _base, _sizes in [
    ("incumbent-yolov8x-roboflow", (960, 1920)),
    ("yolo11x-coco", (960, 1600, 1920)),
    ("yolo-mobadam", (960, 1600, 1920)),
    ("yolov26m-sportsmot", (1920,)),
    ("yolov8x6-soccernet-gsr", (1920,)),
]:
    for _sz in _sizes:
        _c = CANDIDATES[_base]
        _add(Candidate(
            f"{_base}@{_sz}", _c.backend, weights=_c.weights, imgsz=_sz,
            trained_on=_c.trained_on, contaminated=_c.contaminated,
            notes=f"resolution variant of {_base}", extra=dict(_c.extra),
        ))


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------


def load_manifest(tier: str = "soccernet") -> list[dict]:
    path = REPO / "configs" / "datasets" / f"{tier}.json"
    return json.loads(path.read_text())["sequences"]


def select_sequences(roles: tuple[str, ...], tier: str = "soccernet") -> list[dict]:
    seqs = [s for s in load_manifest(tier) if s["role"] in roles]
    if not seqs:
        raise SystemExit(f"no sequences with role(s) {roles} in tier {tier}")
    return seqs


def gt_by_frame(gt_path: Path, frames: list[int],
                target: str = "player") -> dict[int, list[tuple[int, list[float]]]]:
    """frame_idx -> [(gt_track_id, xywh)] for the target's GT roles only."""
    roles = TARGETS[target]["roles"]
    gt = json.loads(gt_path.read_text())
    wanted = set(frames)
    out: dict[int, list[tuple[int, list[float]]]] = {f: [] for f in frames}
    for track in gt["tracks"]:
        if track.get("role", "player") not in roles:
            continue
        tid = track["track_id"]
        for fr in track["frames"]:
            f = fr["frame_idx"]
            if f not in wanted:
                continue
            b = fr["box"]
            out[f].append((tid, [b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"]]))
    return out


def image_path(seq_name: str, frame_idx: int) -> Path:
    """img1 is 1-based; GroundTruth.frame_idx is 0-based."""
    return REPO / "data/soccernet/tracking/test" / seq_name / "img1" / f"{frame_idx + 1:06d}.jpg"


def seq_frames(gt_path: Path, stride: int, limit: int | None) -> list[int]:
    gt = json.loads(gt_path.read_text())
    n = gt["seq_length"]
    frames = list(range(0, n, stride))
    return frames[:limit] if limit else frames


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


class UltralyticsBackend:
    def __init__(self, cand: Candidate):
        from ultralytics import YOLO

        w = cand.weights
        p = REPO / w
        self.model = YOLO(str(p) if p.exists() else w)
        self.imgsz = cand.imgsz
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.names = {int(k): str(v).lower() for k, v in names.items()}
        known = set().union(*(t["names"] for t in TARGETS.values()))
        forced = cand.extra.get("single_class")
        self.assumed_person = False
        if not (set(self.names.values()) & known) and len(self.names) == 1:
            # Single-class detectors sometimes name their class something
            # uninformative ("0", "object"). A ball-only model mislabelled as
            # the person class would score 0 and look broken, so the candidate
            # may declare `single_class`; otherwise assume person and say so.
            self.names = {next(iter(self.names)): forced or "player"}
            self.assumed_person = forced is None
        self.max_det = cand.extra.get("max_det", 100)

    def describe(self) -> dict:
        return {"classes": self.names, "assumed_single_class_is_person":
                self.assumed_person}

    def infer(self, img, conf: float):
        r = self.model.predict(img, imgsz=self.imgsz, conf=conf, device=0,
                               verbose=False, max_det=self.max_det)[0]
        b = r.boxes
        return [
            (float(c), [float(v) for v in box], self.names.get(int(k), str(int(k))))
            for box, c, k in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())
        ]


class RfdetrBackend:
    def __init__(self, cand: Candidate):
        import rfdetr

        size = cand.extra.get("model_size", "base")
        cls = rfdetr.RFDETRLarge if size == "large" else rfdetr.RFDETRBase
        kwargs = {"resolution": cand.imgsz}
        if cand.weights:
            kwargs["pretrain_weights"] = str(REPO / cand.weights)
            for k in ("num_classes", "patch_size"):
                if k in cand.extra:
                    kwargs[k] = cand.extra[k]
        self.model = cls(**kwargs)
        cmap = cand.extra.get("class_map")
        if cmap:
            self.names = {int(k): str(v).lower() for k, v in cmap.items()}
        else:
            # COCO-pretrained RF-DETR keeps the 91-entry COCO id space
            # (matching stages/detect/rfdetr.py): 1 = person, 37 = sports ball.
            self.names = {1: "person", 37: "sports ball"}

    def describe(self) -> dict:
        return {"classes": self.names}

    def infer(self, img, conf: float):
        from PIL import Image

        pil = Image.fromarray(img[:, :, ::-1])
        d = self.model.predict(pil, threshold=conf)
        return [
            (float(c), [float(v) for v in box], self.names.get(int(k), str(int(k))))
            for box, c, k in zip(d.xyxy, d.confidence, d.class_id)
        ]


def make_backend(cand: Candidate):
    if cand.backend == "ultralytics":
        return UltralyticsBackend(cand)
    if cand.backend == "rfdetr":
        return RfdetrBackend(cand)
    raise SystemExit(f"unknown backend {cand.backend}")


# --------------------------------------------------------------------------
# stage 1: inference -> cache
# --------------------------------------------------------------------------


def cache_path(cache_dir: Path, cand: str, seq: str) -> Path:
    return cache_dir / cand / f"{seq}.jsonl"


def run_infer(cand: Candidate, seqs: list[dict], stride: int, limit: int | None,
              conf_floor: float, cache_dir: Path, force: bool) -> dict:
    import cv2

    backend = None
    stats = {"candidate": cand.name, "sequences": {}}
    for s in seqs:
        out_path = cache_path(cache_dir, cand.name, s["name"])
        if out_path.exists() and not force:
            print(f"  [{cand.name}] {s['name']}: cached, skipping")
            continue
        if backend is None:
            backend = make_backend(cand)
            stats["backend"] = backend.describe()
            print(f"  [{cand.name}] classes={backend.describe()}")
        frames = seq_frames(REPO / s["gt"], stride, limit)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        n_boxes = 0
        with open(out_path.with_suffix(".tmp"), "w") as fh:
            for f in frames:
                ip = image_path(s["name"], f)
                img = cv2.imread(str(ip))
                if img is None:
                    raise SystemExit(f"cannot read frame image {ip}")
                dets = backend.infer(img, conf_floor)
                n_boxes += len(dets)
                fh.write(json.dumps({"frame_idx": f, "dets": dets}) + "\n")
        out_path.with_suffix(".tmp").rename(out_path)
        dt = time.time() - t0
        stats["sequences"][s["name"]] = {
            "n_frames": len(frames), "n_boxes": n_boxes,
            "seconds": round(dt, 1), "ms_per_frame": round(1000 * dt / len(frames), 1),
        }
        print(f"  [{cand.name}] {s['name']}: {len(frames)} frames, {n_boxes} boxes, "
              f"{1000 * dt / len(frames):.0f} ms/frame")
    return stats


# --------------------------------------------------------------------------
# stage 2: scoring
# --------------------------------------------------------------------------


def load_cache(path: Path, offset: int = 0,
               target: str = "player") -> dict[int, list[tuple[float, list[float]]]]:
    """Cached xyxy -> xywh, keyed by frame_idx (+ optional offset, used only by
    the alignment check), filtered to the target's detector class names.

    The cache stores every class the model emitted, so the same inference pass
    scores both the player and the ball target.
    """
    names = TARGETS[target]["names"]
    out: dict[int, list[tuple[float, list[float]]]] = {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            boxes = [
                (c, [b[0], b[1], b[2] - b[0], b[3] - b[1]])
                for c, b, n in row["dets"] if n in names
            ]
            out[row["frame_idx"] + offset] = boxes
    return out


def score_candidate(cand_name: str, seqs: list[dict], stride: int, limit: int | None,
                    cache_dir: Path, conf_grid: list[float],
                    target: str = "player", iou: float = 0.5) -> dict:
    from matchlab_core.detection_eval import evaluate_detections

    per_seq = {}
    pooled_det: dict[int, list] = {}
    pooled_gt: dict[int, list] = {}
    base = 0
    for s in seqs:
        cp = cache_path(cache_dir, cand_name, s["name"])
        if not cp.exists():
            continue
        frames = seq_frames(REPO / s["gt"], stride, limit)
        det = load_cache(cp, target=target)
        gt = gt_by_frame(REPO / s["gt"], frames, target)
        per_seq[s["name"]] = evaluate_detections(
            {f: det.get(f, []) for f in frames}, gt, stride=stride, iou_threshold=iou
        )
        # Pool across sequences with a disjoint frame-index offset so a single
        # AP is computed over the whole evaluation set, not averaged per clip.
        for f in frames:
            pooled_det[base + f] = det.get(f, [])
            pooled_gt[base + f] = gt[f]
        base += 10_000_000

    if not per_seq:
        return {"candidate": cand_name, "status": "no-cache"}

    pooled = evaluate_detections(pooled_det, pooled_gt, stride=stride,
                                 iou_threshold=iou)

    # Operating curve: re-threshold the same cached boxes.
    curve = []
    for c in conf_grid:
        d = {f: [x for x in v if x[0] >= c] for f, v in pooled_det.items()}
        r = evaluate_detections(d, pooled_gt, stride=stride, iou_threshold=iou)
        p, rec = r["precision"], r["recall"]
        f1 = (2 * p * rec / (p + rec)) if (p and rec) else 0.0
        curve.append({"conf": c, "precision": p, "recall": rec, "f1": round(f1, 4),
                      "n_detections": r["n_detections"]})
    best = max(curve, key=lambda x: x["f1"])

    # `pooled` is measured at the capture floor (every box the model emitted),
    # which is right for AP but not for "what would the pipeline actually see".
    # Re-score at the candidate's own best-F1 threshold for the per-height-bin
    # and miss-burst numbers a downstream tracker really experiences.
    at_best = evaluate_detections(
        {f: [x for x in v if x[0] >= best["conf"]] for f, v in pooled_det.items()},
        pooled_gt, stride=stride, iou_threshold=iou,
    )

    return {
        "candidate": cand_name,
        "status": "ok",
        "pooled": pooled,
        "pooled_at_best_f1": at_best,
        "best_f1_operating_point": best,
        "curve": curve,
        "per_sequence": {k: {"ap": v["ap"], "precision": v["precision"],
                             "recall": v["recall"], "n_gt_boxes": v["n_gt_boxes"]}
                         for k, v in per_seq.items()},
    }


# --------------------------------------------------------------------------
# ball scoring (center distance, not IoU)
# --------------------------------------------------------------------------
#
# The ball is ~10-20 px across in a 1080p broadcast frame, so box IoU is both
# unstable and unfair across architectures: the strongest ball detectors in the
# literature (WASB and its TrackNet-family peers) are heatmap models that emit a
# CENTER POINT and no box at all. Scoring on center distance is what the sports-
# ball-tracking literature uses and is the only metric these two model families
# can share.
#
# Single-instance assumption: at most one ball is in play, so each frame's
# prediction is the single highest-confidence candidate, exactly how a pipeline
# would consume it. This is deliberately stricter than the player metric --
# a model that sprays ball candidates gains nothing here.


def score_ball(cand_name: str, seqs: list[dict], stride: int, limit: int | None,
               cache_dir: Path, conf_grid: list[float], tol: float = 10.0) -> dict:
    """Per-frame top-1 ball prediction scored by center distance <= tol px.

    Frames are split by whether GT labels a ball at all:
      * `n_gt_frames`   -- ball present. A top-1 within tol is a hit.
      * `n_empty_frames`-- no GT ball (out of play / unlabelled). Any confident
        prediction here is counted as a false alarm, reported separately rather
        than folded into precision, because SoccerNet ball GT has genuine gaps
        and we should not punish a detector for an unlabelled ball.
    """
    rows: list[tuple[float, float, bool]] = []  # (conf, dist, gt_present)
    n_gt_frames = 0
    n_empty_frames = 0
    # Diagnostic: was the ball among ANY candidate the model emitted, even if
    # it did not rank first? This separates "the detector cannot see the ball"
    # from "the detector sees it but scores clutter higher" -- the second is
    # recoverable by the temporal ball-track resolution the pipeline already
    # runs (`detect/yolo_local.py::resolve_ball_track`), the first is not.
    # It matters most for the heatmap models, whose published results pair the
    # detector with an online tracker we deliberately do not run here.
    n_any_hit = 0
    for s in seqs:
        cp = cache_path(cache_dir, cand_name, s["name"])
        if not cp.exists():
            continue
        frames = seq_frames(REPO / s["gt"], stride, limit)
        det = load_cache(cp, target="ball")
        gt = gt_by_frame(REPO / s["gt"], frames, "ball")
        for f in frames:
            g = gt[f]
            cands = det.get(f, [])
            best = max(cands, key=lambda x: x[0]) if cands else None
            if g:
                n_gt_frames += 1
                gx, gy, gw, gh = g[0][1]
                gcx, gcy = gx + gw / 2, gy + gh / 2
                if any(((c[1][0] + c[1][2] / 2 - gcx) ** 2
                        + (c[1][1] + c[1][3] / 2 - gcy) ** 2) ** 0.5 <= tol
                       for c in cands):
                    n_any_hit += 1
                if best is not None:
                    bx, by, bw, bh = best[1]
                    d = ((bx + bw / 2 - gcx) ** 2 + (by + bh / 2 - gcy) ** 2) ** 0.5
                    rows.append((best[0], d, True))
            else:
                n_empty_frames += 1
                if best is not None:
                    rows.append((best[0], float("inf"), False))

    if not n_gt_frames:
        return {"candidate": cand_name, "status": "no-ball-gt"}

    # Adaptive threshold grid. YOLO/DETR confidences are probabilities in
    # [0,1], but the heatmap models (WASB and its TrackNet-family peers) emit
    # an unbounded blob score -- a fixed 0.05..0.9 grid would compare them at
    # meaningless operating points. Deriving thresholds from each candidate's
    # own score quantiles finds every model its best point on the same footing.
    scores = sorted(r[0] for r in rows)
    if scores and scores[-1] > 1.0:
        qs = [scores[int(q * (len(scores) - 1))] for q in
              [i / 24 for i in range(24)]]
        conf_grid = sorted(set(round(v, 4) for v in qs))

    curve = []
    for c in conf_grid:
        kept = [r for r in rows if r[0] >= c]
        tp = sum(1 for r in kept if r[2] and r[1] <= tol)
        fp = sum(1 for r in kept if r[2] and r[1] > tol)
        false_alarm = sum(1 for r in kept if not r[2])
        recall = tp / n_gt_frames
        precision = tp / (tp + fp) if (tp + fp) else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else 0.0
        curve.append({
            "conf": c, "precision": precision, "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "localisation_errors": fp,
            "false_alarms_on_ball_absent_frames": false_alarm,
            "false_alarm_rate": round(false_alarm / n_empty_frames, 4)
            if n_empty_frames else None,
        })
    best = max(curve, key=lambda x: x["f1"])
    return {
        "candidate": cand_name, "status": "ok", "tolerance_px": tol,
        "n_gt_frames": n_gt_frames, "n_empty_frames": n_empty_frames,
        "recall_any_candidate": round(n_any_hit / n_gt_frames, 4),
        "best_f1_operating_point": best, "curve": curve,
    }


# --------------------------------------------------------------------------
# alignment sanity check
# --------------------------------------------------------------------------


def verify_alignment(cand_name: str, seqs: list[dict], stride: int, limit: int | None,
                     cache_dir: Path) -> None:
    """Disconfirming check: recall at the assumed frame alignment must be
    clearly better than at +/-1 frame. If it is not, the img1<->frame_idx
    mapping is wrong and every candidate's score is meaningless."""
    from matchlab_core.detection_eval import evaluate_detections

    s = seqs[0]
    cp = cache_path(cache_dir, cand_name, s["name"])
    frames = seq_frames(REPO / s["gt"], stride, limit)
    gt = gt_by_frame(REPO / s["gt"], frames)
    print(f"alignment check on {s['name']} using {cand_name}:")
    results = {}
    for off in (-stride, 0, stride):
        det = load_cache(cp, offset=off)
        r = evaluate_detections({f: det.get(f, []) for f in frames}, gt, stride=stride)
        results[off] = r["recall"]
        print(f"  offset {off:+d}: recall={r['recall']}  precision={r['precision']}")
    if not (results[0] > results[-stride] and results[0] > results[stride]):
        raise SystemExit(
            "FRAME ALIGNMENT IS WRONG: offset 0 is not the best. "
            f"{results}"
        )
    print("  -> alignment OK (offset 0 wins)")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["infer", "score", "score-ball", "verify", "list"])
    ap.add_argument("--tol", type=float, default=10.0,
                    help="score-ball: center-distance tolerance in px")
    ap.add_argument("--candidates", nargs="*", default=None)
    ap.add_argument("--roles", nargs="*", default=["tuning"])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None,
                    help="max frames per sequence after striding")
    ap.add_argument("--conf-floor", type=float, default=0.05)
    ap.add_argument("--cache-dir", default="data/detector-bench/cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--target", choices=list(TARGETS), default="player")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="match IoU; a ~12px ball warrants a looser threshold "
                         "than a player")
    args = ap.parse_args()

    if args.mode == "list":
        for n, c in CANDIDATES.items():
            tag = "CONTAMINATED" if c.contaminated else "clean"
            print(f"{n:32s} {c.backend:12s} {tag:12s} {c.trained_on}")
        return

    cache_dir = REPO / args.cache_dir
    seqs = select_sequences(tuple(args.roles))
    names = args.candidates or list(CANDIDATES)

    if args.mode == "verify":
        verify_alignment(names[0], seqs, args.stride, args.limit, cache_dir)
        return

    if args.mode == "infer":
        for n in names:
            print(f"== infer {n}")
            try:
                run_infer(CANDIDATES[n], seqs, args.stride, args.limit,
                          args.conf_floor, cache_dir, args.force)
            except SystemExit:
                raise
            except Exception as exc:  # a broken candidate must not kill the sweep
                print(f"  [{n}] FAILED: {type(exc).__name__}: {exc}")
        return

    grid = [round(0.05 * i, 2) for i in range(1, 19)]

    if args.mode == "score-ball":
        rep = {"roles": args.roles, "stride": args.stride, "tolerance_px": args.tol,
               "sequences": [s["name"] for s in seqs], "candidates": {}}
        for n in names:
            r = score_ball(n, seqs, args.stride, args.limit, cache_dir, grid, args.tol)
            c = CANDIDATES.get(n)
            if c:
                r["trained_on"] = c.trained_on
                r["contaminated"] = c.contaminated
            rep["candidates"][n] = r
            if r["status"] == "ok":
                b = r["best_f1_operating_point"]
                print(f"{n:34s} bestF1={b['f1']:.4f} @conf{b['conf']}  "
                      f"P={b['precision']!s:>7} R={b['recall']:.4f}  "
                      f"falseAlarm={b['false_alarm_rate']}")
            else:
                print(f"{n:34s} {r['status']}")
        if args.out:
            out = REPO / args.out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rep, indent=2))
            print(f"wrote {out}")
        return

    report = {"roles": args.roles, "stride": args.stride, "target": args.target,
              "iou_threshold": args.iou,
              "sequences": [s["name"] for s in seqs], "candidates": {}}
    for n in names:
        r = score_candidate(n, seqs, args.stride, args.limit, cache_dir, grid,
                            target=args.target, iou=args.iou)
        c = CANDIDATES[n]
        r["trained_on"] = c.trained_on
        r["contaminated"] = c.contaminated
        r["imgsz"] = c.imgsz
        r["notes"] = c.notes
        report["candidates"][n] = r
        if r["status"] == "ok":
            b = r["best_f1_operating_point"]
            print(f"{n:32s} AP={r['pooled']['ap']!s:>8}  "
                  f"bestF1={b['f1']:.4f} @conf{b['conf']}  "
                  f"P={b['precision']!s:>7} R={b['recall']!s:>7}"
                  f"{'  [CONTAMINATED]' if c.contaminated else ''}")
    if args.out:
        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
