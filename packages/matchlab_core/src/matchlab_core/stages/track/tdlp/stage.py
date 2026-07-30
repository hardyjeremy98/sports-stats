"""Registered `tdlp-shippable` track stage: the assembled, licensing-clean
multi-cue TDLP tracker (SPO-42).

Pipeline: permissive detector output (in) -> RTMPose keypoints (SPO-37) +
DINOv2 global appearance (SPO-38) per detection -> feature assembly -> vendored
TDLP link-prediction head (SPO-40 arch) -> causal offline association loop ->
Tracklet artifacts. Runs on ARBITRARY video (not benchmark-state replay); its
raw tracklets flow unchanged into the existing offline associator.

Head weights: pass a `checkpoint` trained on permissive data (SPO-40). With no
checkpoint the head is randomly initialized and the stage runs end-to-end but
produces UNTRAINED associations — smoke/plumbing only, never a quality claim.
The stage logs loudly in that case.
"""

from __future__ import annotations

import logging

import torch
from pydantic import BaseModel

from matchlab_core.interfaces import StageContext, Tracker
from matchlab_core.provenance import LicenseAxes, ModelProvenance, sha256_file
from matchlab_core.registry import register
from matchlab_core.schemas import FrameDetections, Tracklet
from matchlab_core.schemas.run import StageKind
from matchlab_core.stages.associate.embedders.base import get_embedder
from matchlab_core.stages.track.botsort_reid import embed_detections
from matchlab_core.stages.track.tdlp.feature_assembly import build_object_data
from matchlab_core.stages.track.tdlp.loop import TDLPTracker
from matchlab_core.stages.track.tdlp.model import ModalityConfig, build_head, feature_specs

logger = logging.getLogger("tdlp-shippable")


class Params(BaseModel):
    checkpoint: str = ""  # trained head weights; "" -> random init (smoke only)
    embedder: str = "dinov2"
    use_keypoints: bool = True
    use_appearance: bool = True
    pose_onnx: str = ""  # "" -> rtmpose module default
    # crop quality gate (mirrors the frozen offline associator)
    min_box_height_px: int = 60
    min_crop_confidence: float = 0.3
    # loop / tracker params
    remember_threshold: int = 30
    detection_threshold: float = 0.4
    sim_threshold: float = 0.5
    initialization_threshold: int = 1
    new_tracklet_detection_threshold: float = 0.9
    min_length: int = 3
    # motion gate (normalized bbox-centre): forbid implausible matches
    gate_base_radius: float = 0.0
    gate_per_frame_radius: float = 0.0
    # head hyperparameters (must match a loaded checkpoint's training config)
    hidden_dim: int = 128
    mm_dim: int = 128
    track_encoder_n_heads: int = 4
    track_encoder_n_layers: int = 2
    track_encoder_ffn_dim: int = 256


@register(StageKind.TRACK, "tdlp-shippable")
class TdlpShippableTracker(Tracker):
    def __init__(self, **params):
        self.params = Params(**params)
        self._model = None
        self._embedder = None
        self._pose = None
        self._modality = ModalityConfig(
            use_keypoints=self.params.use_keypoints,
            use_appearance=self.params.use_appearance,
        )

    def prepare(self, ctx: StageContext) -> None:
        p = self.params
        appearance_dim = 384
        if p.use_appearance:
            self._embedder = get_embedder(p.embedder)
            self._embedder.prepare(ctx.device)
            appearance_dim = self._embedder.dim
        self._modality = ModalityConfig(
            use_keypoints=p.use_keypoints,
            use_appearance=p.use_appearance,
            appearance_dim=appearance_dim,
        )
        if p.use_keypoints:
            from matchlab_core.pose.rtmpose import RTMPoseEstimator

            kwargs = {"onnx_model": p.pose_onnx} if p.pose_onnx else {}
            self._pose = RTMPoseEstimator(**kwargs)
            self._pose.prepare(ctx.device)

        # Head hyperparameters: a checkpoint's own recorded config wins (so a
        # checkpoint trained with a different capacity loads without a config
        # mismatch); otherwise fall back to the stage Params.
        hp = dict(
            hidden_dim=p.hidden_dim, mm_dim=p.mm_dim,
            track_encoder_n_heads=p.track_encoder_n_heads,
            track_encoder_n_layers=p.track_encoder_n_layers,
            track_encoder_ffn_dim=p.track_encoder_ffn_dim,
        )
        ckpt_payload = None
        if p.checkpoint:
            ckpt_payload = torch.load(p.checkpoint, map_location=ctx.device)
            cfg_d = ckpt_payload.get("config", {}) if isinstance(ckpt_payload, dict) else {}
            if cfg_d:
                hp.update(
                    hidden_dim=cfg_d.get("hidden_dim", hp["hidden_dim"]),
                    mm_dim=cfg_d.get("hidden_dim", hp["mm_dim"]),
                    track_encoder_n_heads=cfg_d.get("n_heads", hp["track_encoder_n_heads"]),
                    track_encoder_n_layers=cfg_d.get("n_layers", hp["track_encoder_n_layers"]),
                    track_encoder_ffn_dim=cfg_d.get("ffn_dim", hp["track_encoder_ffn_dim"]),
                )
        self._model = build_head(self._modality, sph_hidden_dim=hp["hidden_dim"], **hp)
        if p.checkpoint:
            state = ckpt_payload.get("model", ckpt_payload) if isinstance(ckpt_payload, dict) else ckpt_payload
            self._model.load_state_dict(state)
            logger.info("tdlp-shippable: loaded head checkpoint %s", p.checkpoint)
        else:
            logger.warning(
                "tdlp-shippable: NO checkpoint — head is randomly initialized. "
                "Output tracklets are plumbing-only (untrained), not a quality "
                "result. Train a head (SPO-40) and pass `checkpoint`."
            )
        self._model.to(ctx.device)
        self._model.eval()

    def _frame_walker(self, ctx: StageContext):
        """Yield images aligned to requested frame_idx (single sequential decode
        over ctx.frames()), mirroring the botsort-reid pattern."""
        frame_iter = ctx.frames()
        cache = {"cur": None}

        def get(target_idx: int):
            while True:
                cur = cache["cur"]
                if cur is not None and cur[0] == target_idx:
                    return cur[1]
                if cur is not None and cur[0] > target_idx:
                    return None  # sampling skipped this frame_idx
                try:
                    fr = next(frame_iter)
                except StopIteration:
                    return None
                cache["cur"] = (fr.frame_idx, fr.image)

        return get

    def track(self, ctx: StageContext, detections: list[FrameDetections]) -> list[Tracklet]:
        p = self.params
        W, H = ctx.video.width, ctx.video.height
        get_image = self._frame_walker(ctx)
        appearance_dim = self._modality.appearance_dim

        frames: list[tuple[int, list[dict]]] = []
        n = len(detections) or 1
        for i, fd in enumerate(detections):
            image = get_image(fd.frame_idx)
            dets = fd.detections
            boxes = [d.box for d in dets]

            poses = None
            if p.use_keypoints and self._pose is not None and image is not None and boxes:
                poses = self._pose.estimate(image, boxes)

            app_emb = app_ok = None
            if p.use_appearance and self._embedder is not None:
                app_emb, app_ok = embed_detections(
                    image, dets, self._embedder, p.min_box_height_px, p.min_crop_confidence
                )

            objects_data: list[dict] = []
            for j, d in enumerate(dets):
                kpts = poses[j].keypoints if poses is not None else None
                appearance = (
                    app_emb[j] if (app_emb is not None and app_ok is not None and app_ok[j]) else None
                )
                objects_data.append(
                    build_object_data(
                        d.box,
                        d.confidence,
                        W,
                        H,
                        keypoints=kpts,
                        appearance=appearance,
                        use_keypoints=p.use_keypoints,
                        use_appearance=p.use_appearance,
                        appearance_dim=appearance_dim,
                        cls=d.cls,
                    )
                )
            frames.append((fd.frame_idx, objects_data))
            if i % 50 == 0:
                ctx.progress(StageKind.TRACK, min(i / n, 0.99), f"tdlp: frame {i}")

        tracker = TDLPTracker(
            model=self._model,
            feature_specs=feature_specs(self._modality),
            device=ctx.device,
            remember_threshold=p.remember_threshold,
            detection_threshold=p.detection_threshold,
            sim_threshold=p.sim_threshold,
            initialization_threshold=p.initialization_threshold,
            new_tracklet_detection_threshold=p.new_tracklet_detection_threshold,
            gate_base_radius=p.gate_base_radius,
            gate_per_frame_radius=p.gate_per_frame_radius,
        )
        return tracker.track_clip(frames, min_length=p.min_length)

    def provenance(self) -> list[ModelProvenance]:
        p = self.params
        provs: list[ModelProvenance] = []
        ckpt_sha = sha256_file(p.checkpoint) if p.checkpoint else None
        provs.append(
            ModelProvenance(
                architecture="tdlp-link-prediction-head (vendored MIT, global-appearance)",
                revision=p.checkpoint or "untrained-random-init",
                weights_path=p.checkpoint or None,
                weights_sha256=ckpt_sha,
                lineage=(
                    "TDLP head (Robotmurlock/TDLP @50344b9, MIT) retrained in-house "
                    "on permissive data (SPO-40); NOT the CC-BY-NC released weights"
                ),
                license=LicenseAxes(
                    code="MIT (vendored TDLP head)",
                    weights=(
                        "in-house checkpoint (permissive)" if p.checkpoint
                        else "none (random init — not a shippable artifact)"
                    ),
                    training_data=(
                        "recorded with the trained checkpoint (SPO-40/SPO-39)"
                        if p.checkpoint else "n/a (untrained)"
                    ),
                ),
            )
        )
        if self._pose is not None:
            provs.append(self._pose.provenance())
        if p.use_appearance:
            # Record the appearance embedder so the SPO-41 gate vets its axes
            # too (an embedder is a shipping-path component). Cheap to construct
            # (weights load only in prepare()); falls back to "unknown" axes if
            # the embedder declares none.
            emb = self._embedder
            if emb is None:
                try:
                    emb = get_embedder(p.embedder)
                except Exception:
                    emb = None
            emb_license = getattr(emb, "license", None) or LicenseAxes(
                code="unknown", weights="unknown", training_data="unknown"
            )
            provs.append(
                ModelProvenance(
                    architecture=f"appearance-embedder:{p.embedder}",
                    revision=getattr(emb, "model_name", p.embedder),
                    lineage=getattr(emb, "lineage", "appearance embedding cue"),
                    license=emb_license,
                )
            )
        return provs
