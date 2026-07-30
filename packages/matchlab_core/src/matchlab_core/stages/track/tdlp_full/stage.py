"""`tdlp-full` TRACK stage: run the SOTA TDLP-full tracker over the pipeline's
detections by bridging to the external-trackers venvs.

Data flow per run (see bridge.py for the frame-index bookkeeping):

    ctx.frames() + detections
      → bridge.stage_sequence  →  <run>/_tdlp_work/test/<seq>/{img1, det, seqinfo}
      → [CAMEL venv]  gen_features.py   →  per-frame KPR+pose feature pkls
      → [TDLP venv]   run_tdlp_frozen.py →  <seq>.txt (MOT)
      → bridge.parse_tracker_output  →  list[Tracklet]  (frame_idx in source space)

Nothing heavy is imported in-process: both models run in their own venv as a
subprocess. `prepare()` validates the whole external toolchain up front so a
missing checkout/venv/weight fails the run early with a pointed message.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from pydantic import BaseModel

from matchlab_core.interfaces import StageContext, Tracker
from matchlab_core.provenance import LicenseAxes, ModelProvenance
from matchlab_core.registry import register
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.run import ArtifactName, StageKind
from matchlab_core.stages.track._assembly import assign_classes_by_overlap
from matchlab_core.stages.track.tdlp_full import bridge


class Params(BaseModel):
    # Root of the isolated external-trackers checkout. All other paths are
    # resolved relative to it, so a non-default location only needs this set.
    # None → auto-resolve as a sibling of the lab repo (move/rename-resilient).
    external_root: str | None = None
    # MOT has no class column, so the external tracker cannot return a
    # detection's class. Recover it by matching tracklet boxes back to the input
    # detections. False restores the pre-2026-07-30 behaviour, where every
    # tracklet arrived as PLAYER and the referee exclusion / goalkeeper handling
    # downstream were silently inert.
    recover_detection_class: bool = True
    # Per-venv python interpreters (relative to external_root).
    camel_python: str = "CAMELTrack/.venv/bin/python"
    tdlp_python: str = "TDLP/.venv/bin/python"
    # Scripts (relative to external_root). gen_features runs in the CAMEL venv;
    # the tracker runs in the TDLP venv.
    gen_features_script: str = "bridge/gen_features.py"
    tracker_script: str = "run_tdlp_frozen.py"
    # Released full-model head + the published config that carries its transform
    # and tracker thresholds (relative to external_root).
    head_weights: str = "weights/tdlp_sportsmot"
    history_yaml: str = "TDLP/history/SportsMOT/tdlp.yaml"
    # KPR appearance weights consumed by gen_features (relative to external_root).
    kpr_weights: str = (
        "CAMELTrack/pretrained_models/reid/"
        "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"
    )
    # Appearance model behind the frame_features artifact — the re-ID engine's
    # input layer. The TRACKER always uses KPR internally (the released TDLP
    # head consumes its 6-part parts_appearance shape); this selects what the
    # ASSOCIATE layer downstream gets. "prtreid" costs a second feature-gen
    # pass over the same staged frames, spending the PRD's "reuse the tracker's
    # own features at zero extra inference cost" property — deliberately, see
    # docs/reports/2026-07-25-spo85-gt-tracklet-embedder-comparison.md.
    # Default since 2026-07-26: PRTreID is the adopted re-ID backbone. Set
    # "kpr" to skip the second feature-gen pass when a run needs tracking only
    # and will never be associated.
    reid_model: str = "prtreid"
    prtreid_weights: str = (
        "CAMELTrack/pretrained_models/reid/prtreid-soccernet-baseline.pth.tar"
    )
    # Device passed to both external steps. KPR/RTMPose want CUDA; "cpu" works
    # but is very slow (onnxruntime here is CPU-only anyway for pose).
    device: str = "cuda:0"
    # Hard ceiling per external step (feature-gen dominates). None = no limit.
    timeout_s: float | None = 7200.0
    # Keep the _tdlp_work scratch (frames + feature pkls) after a successful run
    # for debugging; it is always kept on failure.
    keep_work: bool = False
    # Sequence name for the fabricated MOT layout; defaults to the run id.
    seq_name: str | None = None


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "seq"


def _default_external_root() -> Path:
    """external-trackers is a sibling of the lab repo checkout. Resolve it
    relative to this file so the default survives the checkout being moved or
    renamed (the repo root is the ancestor holding both `packages/` and
    `configs/`); fall back to the historical path if the layout is unexpected."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "packages").is_dir() and (parent / "configs").is_dir():
            return parent.parent / "external-trackers"
    return Path("~/code/sport-stats/external-trackers").expanduser()


@register(StageKind.TRACK, "tdlp-full")
class TdlpFullTracker(Tracker):
    """Bridge tracker: SOTA TDLP-full via the external venvs (research/local
    tool — the released head/appearance weights are separately licensed and
    the heavy deps deliberately stay out of the lab dependency tree)."""

    def __init__(self, **params):
        self.params = Params(**params)
        self._root: Path | None = None

    def _paths(self) -> dict[str, Path]:
        root = (
            Path(self.params.external_root).expanduser()
            if self.params.external_root
            else _default_external_root()
        )
        return {
            "root": root,
            "camel_python": root / self.params.camel_python,
            "tdlp_python": root / self.params.tdlp_python,
            "gen_features": root / self.params.gen_features_script,
            "tracker": root / self.params.tracker_script,
            "head_weights": root / self.params.head_weights,
            "history_yaml": root / self.params.history_yaml,
            "kpr_weights": root / self.params.kpr_weights,
            **(
                {"prtreid_weights": root / self.params.prtreid_weights}
                if self.params.reid_model == "prtreid"
                else {}
            ),
        }

    def prepare(self, ctx: StageContext) -> None:
        p = self._paths()
        missing = [
            f"{key} → {path}"
            for key, path in p.items()
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "The 'tdlp-full' tracker needs the external-trackers checkout and its "
                "two venvs (heavy/separately-licensed deps kept out of the lab tree). "
                "Missing:\n  " + "\n  ".join(missing) + "\n"
                "Point `external_root` at your checkout, or use the 'botsort' tracker."
            )
        self._root = p["root"]

    def provenance(self) -> list[ModelProvenance]:
        p = self._paths()
        return [
            ModelProvenance(
                architecture="TDLP (multimodal offline link-prediction head)",
                revision="tdlp_sportsmot",
                weights_path=str(p["head_weights"]),
                lineage="released Robotmurlock/tdlp_sportsmot; trained on SportsMOT",
                license=LicenseAxes(
                    code="MIT", weights="CC-BY-NC-4.0", training_data="SportsMOT (CC-BY-NC)"
                ),
            ),
            *(
                [
                    ModelProvenance(
                        architecture="PRTreID (SoccerNet-trained, hrnet32, 'globl' 1x256)",
                        revision="prtreid-soccernet-baseline",
                        weights_path=str(p.get("prtreid_weights", "")),
                        lineage=(
                            "VlSomers/prtreid; SoccerNet-trained; behind frame_features "
                            "(the re-ID engine's input) — the tracker still uses KPR"
                        ),
                        license=LicenseAxes(
                            code="Hippocratic HL3-LAW-MEDIA-MIL-SOC-SV",
                            weights="research",
                            training_data="SoccerNet",
                        ),
                    )
                ]
                if self.params.reid_model == "prtreid"
                else []
            ),
            ModelProvenance(
                architecture="KPR (keypoint-promptable ReID, 6-part appearance)",
                revision=Path(self.params.kpr_weights).stem,
                weights_path=str(p["kpr_weights"]),
                lineage="tracklab KPReId; DanceTrack+SportsMOT+PoseTrack21+OccludedDuke+Market",
                license=LicenseAxes(
                    code="MIT", weights="research-only", training_data="research-only"
                ),
            ),
            ModelProvenance(
                architecture="RTMPose-m (body7, COCO-17 keypoints)",
                revision="rtmpose-m_simcc-body7_420e-256x192",
                lineage="rtmlib pretrained (openmmlab)",
                license=LicenseAxes(code="Apache-2.0", weights="Apache-2.0"),
            ),
        ]

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        if self._root is None:
            self.prepare(ctx)
        p = self._paths()
        params = self.params

        seq = _sanitize(params.seq_name or ctx.store.run_dir.name)
        # Absolute: every path below is handed to subprocesses running with
        # cwd=external_root, where a relative run dir (as passed by the
        # matchlab-run CLI) would not resolve.
        work = (ctx.store.run_dir / "_tdlp_work").resolve()
        # gen_features writes pkls into features_root/<seq>/; the tracker's
        # ExtraFeaturesReader resolves <feat-dir>/<scene>/<frame>.pkl, so it
        # takes the PARENT (features_root), not the per-sequence dir.
        features_root = work / "features"
        feat_dir = features_root / seq
        mot_out = work / "mot"

        def prog(msg: str) -> None:
            ctx.progress(StageKind.TRACK, 0.0, msg)

        prog(f"TDLP-full: staging clip → MOT layout ({seq})")
        layout = bridge.stage_sequence(
            ctx.frames(), detections, work, seq_name=seq, fps=ctx.video.fps
        )
        prog(
            f"TDLP-full: {len(layout.local_to_source)} frames, "
            f"{layout.n_detections} detections staged"
        )

        # 1) Appearance + pose features (CAMEL venv).
        bridge.run_external(
            p["camel_python"],
            p["gen_features"],
            [
                "--img-dir", str(layout.img_dir),
                "--det-file", str(layout.det_file),
                "--out-dir", str(feat_dir),
                "--weights", str(p["kpr_weights"]),
                "--device", params.device,
            ],
            cwd=p["root"],
            timeout_s=params.timeout_s,
            label="feature generation (RTMPose + KPR)",
            progress=prog,
        )

        # 2) Offline association head (TDLP venv). Reuses the existing frozen
        #    driver against the single fabricated sequence.
        mot_out.mkdir(parents=True, exist_ok=True)
        bridge.run_external(
            p["tdlp_python"],
            p["tracker"],
            [
                "--history-yaml", str(p["history_yaml"]),
                "--hf-model-dir", str(p["head_weights"]),
                "--data-root", str(layout.data_root),
                "--split", layout.split,
                "--feat-dir", str(features_root),
                "--out-dir", str(mot_out),
                "--device", params.device,
                "--seqs", seq,
            ],
            cwd=p["root"],
            timeout_s=params.timeout_s,
            label="offline tracking (TDLP head)",
            # The `tdlp` package is the TDLP repo itself; make it importable
            # without depending on a (move-fragile) editable install.
            extra_pythonpath=p["root"] / "TDLP",
            progress=prog,
        )

        mot_file = mot_out / f"{seq}.txt"
        if not mot_file.exists():
            raise RuntimeError(
                f"TDLP-full produced no MOT output at {mot_file}. The tracker step "
                "reported success but wrote nothing — check the run's _tdlp_work."
            )
        tracklets = bridge.parse_tracker_output(mot_file, layout.local_to_source)
        # MOT has no class column, so every tracklet returns as PLAYER whatever
        # it was detected as. Recover it from the input detections, which do
        # carry the class -- without this the referee exclusion in
        # reid_engine.eligible() and the goalkeeper confidence discount in both
        # team classifiers are dead code (SNMOT-118: 641 goalkeeper + 749
        # referee detections in, 25 uniformly-`player` tracklets out, and a
        # referee merged into a player's thread as that run's only wrong merge).
        if params.recover_detection_class:
            tracklets = assign_classes_by_overlap(tracklets, detections)
        prog(f"TDLP-full: {len(tracklets)} tracklets")

        # Persist the bridge's per-frame KPR embeddings + pose keypoints as the
        # frame_features artifact (SPO-51) — the re-ID engine's input layer —
        # before the work dir (and its pkls) can be cleaned up.
        reid_feat_dir = feat_dir
        if params.reid_model == "prtreid":
            # 3) Second appearance pass for the ASSOCIATE layer only. Pose is
            #    skipped: nothing under matchlab_core/reid/ reads keypoints.
            reid_feat_dir = work / "feat_prtreid"
            reid_feat_dir.mkdir(parents=True, exist_ok=True)
            bridge.run_external(
                p["camel_python"],
                p["gen_features"],
                [
                    "--img-dir", str(layout.img_dir),
                    "--det-file", str(layout.det_file),
                    "--out-dir", str(reid_feat_dir),
                    "--weights", str(p["prtreid_weights"]),
                    "--reid-model", "prtreid",
                    "--device", params.device,
                    # Flags last: keeps the arg list parseable pairwise.
                    "--no-pose",
                ],
                cwd=p["root"],
                timeout_s=params.timeout_s,
                label="re-ID feature generation (PRTreID)",
                progress=prog,
            )

        features = bridge.join_features_to_tracklets(
            tracklets,
            reid_feat_dir,
            layout.local_to_source,
            width=layout.width,
            height=layout.height,
        )
        features.meta["reid_model"] = params.reid_model
        features.save(ctx.store.path(ArtifactName.FRAME_FEATURES))
        prog(f"TDLP-full: {len(features)} feature rows exported")

        if not params.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        return tracklets
